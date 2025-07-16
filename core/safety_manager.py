import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from config.settings import Settings

logger = logging.getLogger(__name__)


class SafetyManager:
    """
    Manages safety features like killswitch based on drawdown and monitors critical metrics.
    """

    def __init__(self, settings: Settings, trading_bot_instance):
        self.settings = settings
        self.bot = trading_bot_instance
        self.killswitch_enabled = self.settings.get('risk_management.killswitch.enabled', True)
        self.max_drawdown_percent = self.settings.get('risk_management.killswitch.max_drawdown',
                                                      0.15) * 100  # In percentage
        self.auto_reactivate_after_hours = self.settings.get('risk_management.killswitch.auto_reactivate_after_hours',
                                                             24)
        self.last_killswitch_activation_time: Optional[datetime] = None
        self.is_killswitch_active = False

        logger.info("SafetyManager initialized.")
        if self.killswitch_enabled:
            logger.info(f"Killswitch enabled: Max Drawdown {self.max_drawdown_percent:.1f}%")

    def check_and_apply_killswitch(self, current_portfolio_value: float, peak_portfolio_value: float):
        """
        Checks if the drawdown threshold is reached and activates the killswitch if necessary.

        Args:
            current_portfolio_value: The current value of the portfolio.
            peak_portfolio_value: The highest recorded portfolio value.
        """
        if not self.killswitch_enabled:
            return

        if peak_portfolio_value <= 0:  # Avoid division by zero
            return

        drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value * 100

        if drawdown >= self.max_drawdown_percent and not self.is_killswitch_active:
            self._activate_killswitch(f"Drawdown reached {drawdown:.2f}% (Threshold: {self.max_drawdown_percent:.1f}%)")
        elif self.is_killswitch_active:
            # Check for auto-reactivation
            self._check_auto_reactivation(current_portfolio_value, peak_portfolio_value)

    def _activate_killswitch(self, reason: str):
        """Activates the killswitch, stops the bot, and logs the event."""
        if self.is_killswitch_active:
            return

        self.is_killswitch_active = True
        self.last_killswitch_activation_time = datetime.now()

        logger.critical(f"KILLSWITCH ACTIVATED! Reason: {reason}")
        self.bot.stop()  # Calls the bot's stop method
        self.bot._notify_error("killswitch_activated", reason)  # Notify callbacks

    def _check_auto_reactivation(self, current_portfolio_value: float, peak_portfolio_value: float):
        """
        Checks if the conditions for auto-reactivation are met.
        Conditions:
        1. Auto-reactivation is enabled and time since activation has passed.
        2. Portfolio is no longer in severe drawdown (e.g., drawdown < 50% of max drawdown).
        """
        if not self.auto_reactivate_after_hours or not self.is_killswitch_active:
            return

        time_since_activation = (datetime.now() - self.last_killswitch_activation_time).total_seconds() / 3600

        if time_since_activation >= self.auto_reactivate_after_hours:
            current_drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value * 100
            if current_drawdown < (self.max_drawdown_percent / 2):  # Recovered sufficiently
                self._deactivate_killswitch(
                    f"Automatic reactivation after {time_since_activation:.1f} hours. Current drawdown: {current_drawdown:.2f}%")
                self.bot.run()  # Restart the bot
            else:
                logger.info(
                    f"Killswitch still active: Not enough recovery for auto-reactivation. Current drawdown: {current_drawdown:.2f}%")

    def _deactivate_killswitch(self, reason: str):
        """Deactivates the killswitch and logs the event."""
        self.is_killswitch_active = False
        self.last_killswitch_activation_time = None
        logger.warning(f"KILLSWITCH DEACTIVATED. Reason: {reason}")
        self.bot._notify_status_update({"killswitch_status": "deactivated", "reason": reason})

    def is_active(self) -> bool:
        """Returns True if the killswitch is currently active."""
        return self.is_killswitch_active

    def trigger_manual_killswitch(self, reason: str = "Manual activation"):
        """Allows manual activation of the killswitch."""
        self._activate_killswitch(reason)

    def deactivate_manual_killswitch(self, reason: str = "Manual deactivation"):
        """Allows manual deactivation of the killswitch."""
        if self.is_killswitch_active:
            self._deactivate_killswitch(reason)
        else:
            logger.info("Killswitch is not active, no deactivation needed.")