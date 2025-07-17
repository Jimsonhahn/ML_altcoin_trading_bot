# core/safety_manager.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, TYPE_CHECKING, Dict

from config.settings import Settings
from utils.notifier import NotificationManager

# To avoid circular import, use TYPE_CHECKING for type hints
if TYPE_CHECKING:
    from core.trading_bot import TradingBot

logger = logging.getLogger(__name__)


class SafetyManager:
    """
    Manages safety mechanisms like killswitch based on drawdown.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.killswitch_enabled = self.settings.get('risk_management.killswitch.enabled', True)
        self.max_drawdown_percent = self.settings.get('risk_management.killswitch.max_drawdown', 0.15)
        self.auto_reactivate_after_hours = self.settings.get('risk_management.killswitch.auto_reactivate_after_hours',
                                                             24)
        self.notification_on_trigger = self.settings.get('risk_management.killswitch.notification_on_trigger', True)

        self.is_killswitch_active: bool = False
        self.last_killswitch_activation_time: Optional[datetime] = None
        self.peak_equity: float = self.settings.get('trading.initial_capital',
                                                    10000)  # Initial capital as starting peak
        self.current_drawdown_percent: float = 0.0

        self.notification_manager = NotificationManager(self.settings)

        self.trading_bot: Optional['TradingBot'] = None  # Reference to the trading bot instance

        logger.info(
            f"SafetyManager initialized. Killswitch enabled: {self.killswitch_enabled}, Max Drawdown: {self.max_drawdown_percent:.2%}")

    def set_trading_bot(self, bot: 'TradingBot'):
        """Sets the reference to the trading bot instance."""
        self.trading_bot = bot
        logger.info("SafetyManager received TradingBot instance reference.")

    def update_equity(self, current_equity: float):
        """
        Updates the current equity and checks for drawdown to trigger killswitch.
        """
        if not self.killswitch_enabled:
            return

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown_percent = 0.0
        else:
            if self.peak_equity > 0:  # Avoid division by zero
                self.current_drawdown_percent = (self.peak_equity - current_equity) / self.peak_equity
            else:
                self.current_drawdown_percent = 0.0  # Or some other handling for zero peak equity

        if self.current_drawdown_percent >= self.max_drawdown_percent and not self.is_killswitch_active:
            self._activate_killswitch(
                f"Drawdown {self.current_drawdown_percent:.2%} exceeded {self.max_drawdown_percent:.2%} threshold from peak equity of {self.peak_equity:.2f}.")

        self._check_auto_reactivation()

    def _activate_killswitch(self, reason: str):
        """Activates the killswitch and pauses trading operations."""
        self.is_killswitch_active = True
        self.last_killswitch_activation_time = datetime.now()
        logger.critical(f"KILLSWITCH ACTIVATED! Reason: {reason}")
        if self.notification_on_trigger:
            self.notification_manager.send_alert(f"KILLSWITCH ACTIVATED! Reason: {reason}. Bot paused.",
                                                 level="CRITICAL")

        if self.trading_bot:
            # The bot should have its own internal mechanism to pause trading when killswitch is active
            # This can be done by checking `is_killswitch_active()` in its main loop.
            logger.info("Trading operations effectively paused due to killswitch.")
            # Optionally, if the bot needs explicit stopping of active strategies:
            # self.trading_bot.strategy_router._deactivate_all_strategies() 
            # Or self.trading_bot.current_active_strategy.stop_trading() if single strategy

    def _check_auto_reactivation(self):
        """Checks if the killswitch should be automatically deactivated."""
        if self.is_killswitch_active and self.auto_reactivate_after_hours > 0:
            if self.last_killswitch_activation_time and \
                    (
                            datetime.now() - self.last_killswitch_activation_time).total_seconds() / 3600 >= self.auto_reactivate_after_hours:
                self._deactivate_killswitch(f"Automatic reactivation after {self.auto_reactivate_after_hours} hours.")

    def _deactivate_killswitch(self, reason: str):
        """Deactivates the killswitch and resumes trading operations."""
        self.is_killswitch_active = False
        self.last_killswitch_activation_time = None
        self.current_drawdown_percent = 0.0  # Reset drawdown after reactivation
        self.peak_equity = self.trading_bot.position_manager.get_total_capital(
            self.trading_bot.exchange.get_current_prices()) if self.trading_bot else self.settings.get(
            'trading.initial_capital', 10000)  # Reset peak equity

        logger.warning(f"KILLSWITCH DEACTIVATED. Reason: {reason}. Trading can resume.")
        if self.notification_on_trigger:
            self.notification_manager.send_alert(f"KILLSWITCH DEACTIVATED. Reason: {reason}. Bot can resume trading.",
                                                 level="WARNING")

        if self.trading_bot:
            # If the strategy router was used, it might need to re-evaluate regime and reactivate strategies
            if self.trading_bot.strategy_router:
                logger.info("Triggering StrategyRouter to re-evaluate market regime and reactivate strategies.")
                # Force a regime check and strategy update
                # This assumes a mock current_total_capital can be passed,
                # or that strategy_router gets it from bot's position manager
                self.trading_bot.strategy_router.update_market_regime(
                    self.trading_bot.strategy_router.get_current_regime(),
                    # Re-evaluate current regime or force a new check
                    self.trading_bot.position_manager.get_total_capital(self.trading_bot.exchange.get_current_prices())
                )
            logger.info("Trading bot can resume operations.")

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the safety manager."""
        return {
            "killswitch_enabled": self.killswitch_enabled,
            "is_killswitch_active": self.is_killswitch_active,
            "last_activation_time": self.last_killswitch_activation_time.isoformat() if self.last_killswitch_activation_time else None,
            "current_drawdown_percent": self.current_drawdown_percent,
            "peak_equity": self.peak_equity
        }