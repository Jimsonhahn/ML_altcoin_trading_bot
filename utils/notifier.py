# utils/notifier.py
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Dict, Any

from config.settings import Settings  # Assuming Settings class is accessible
from utils.secure_http import SecureHTTPClient

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manages sending notifications via various channels (e.g., Telegram, Email).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.telegram_config = self.settings.get('notifications.telegram', {})
        self.email_config = self.settings.get('notifications.email', {})

        self.telegram_enabled = self.telegram_config.get('enabled', False)
        self.email_enabled = self.email_config.get('enabled', False)

        # Get credentials from SecretManager
        self._load_secure_credentials()

        # Initialize secure HTTP client for Telegram
        self.http_client = SecureHTTPClient(
            timeout=(5, 30),
            max_retries=3,
            user_agent="TradingBot-Notifier/1.0"
        )

        if self.telegram_enabled:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                logger.warning(
                    "Telegram notifications enabled but bot_token or chat_id not found in SecretManager. Disabling Telegram.")
                self.telegram_enabled = False
            else:
                logger.info("Telegram notifications initialized with SecretManager credentials.")

        if self.email_enabled:
            if not self.email_config.get('sender_email') or not self.email_config.get('recipient_email') or \
                    not self.email_config.get('smtp_server') or not self.email_config.get('smtp_port') or \
                    not self.email_username or not self.email_password:
                logger.warning("Email notifications enabled but required SMTP settings are missing. Disabling Email.")
                self.email_enabled = False
            else:
                logger.info("Email notifications initialized with SecretManager credentials.")

        if not self.telegram_enabled and not self.email_enabled:
            logger.info("No external notifications enabled.")
    
    def _load_secure_credentials(self):
        """Load credentials from SecretManager"""
        try:
            from utils.secret_manager import SecretManager
            sm = SecretManager()
            
            # Load Telegram credentials
            self.telegram_bot_token = sm.get_secret('telegram_bot_token')
            self.telegram_chat_id = sm.get_secret('telegram_chat_id')
            
            # Load Email credentials  
            self.email_username = sm.get_secret('email_username')
            self.email_password = sm.get_secret('email_password')
            
        except Exception as e:
            logger.warning(f"Could not load credentials from SecretManager: {e}")
            # Fallback to empty credentials
            self.telegram_bot_token = None
            self.telegram_chat_id = None
            self.email_username = None
            self.email_password = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'http_client'):
            self.http_client.close()

    def send_alert(self, message: str, level: str = "INFO"):
        """
        Sends an alert message through configured channels.
        """
        full_message = f"[{level.upper()}] {message}"
        logger.info(f"ALERT: {full_message}")

        if self.telegram_enabled:
            self._send_telegram_message(full_message)

        if self.email_enabled:
            self._send_email(full_message)

    def _send_telegram_message(self, message: str):
        """Sends a message to Telegram via Bot API."""
        token = self.telegram_bot_token
        chat_id = self.telegram_chat_id
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'  # Optional, for bold/italic etc.
        }
        try:
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.debug(f"Telegram message sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def _send_email(self, message: str):
        """Sends an email notification via SMTP."""
        sender = self.email_config['sender_email']
        recipient = self.email_config['recipient_email']
        subject = f"Trading Bot Alert: {message[:70]}..."  # Take first 70 chars for subject

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient

        try:
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()  # Enable TLS encryption
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            logger.info("Email alert sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")