# config/notifications_example.py
"""
Example configuration for notifications
Copy this to your main settings and adjust values
"""

NOTIFICATIONS_CONFIG = {
    'notifications': {
        'telegram': {
            'enabled': True,  # Set to False to disable Telegram
            # Bot token and chat ID are loaded from:
            # 1. SecretManager (recommended)
            # 2. Environment variables: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            # 3. These config values (not recommended for security)
            'bot_token': None,  # Optional fallback
            'chat_id': None     # Optional fallback
        },
        
        'email': {
            'enabled': True,  # Set to False to disable Email
            'smtp_server': 'smtp.gmail.com',  # Gmail SMTP
            'smtp_port': 587,
            'use_tls': True,
            'sender_email': 'your-bot@example.com',
            'recipient_email': 'your-alerts@example.com',
            # Username and password are loaded from:
            # 1. SecretManager (recommended)
            # 2. Environment variables: EMAIL_USERNAME, EMAIL_PASSWORD
            # 3. These config values (not recommended for security)
            'username': None,  # Optional fallback
            'password': None   # Optional fallback
        },
        
        'alerts': {
            'min_level': 'INFO',  # Minimum alert level: INFO, WARNING, ERROR, CRITICAL
            'enabled_types': [    # Types of alerts to send
                'STRATEGY_CHANGE',
                'MARKET_PHASE_CHANGE', 
                'DRAWDOWN',
                'API_ERROR',
                'BOT_CRASH',
                'TRADE_EXECUTED',
                'SYSTEM_STATUS',
                'PORTFOLIO_UPDATE'
            ]
        }
    }
}

# Usage in your main settings:
# settings.update(NOTIFICATIONS_CONFIG)