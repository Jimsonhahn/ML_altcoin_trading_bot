# core/strategy_router.py
import logging
from typing import Dict, Any, Type, Optional, List
from datetime import datetime

from config.settings import Settings
from strategies.strategy_base import Strategy
from strategies import STRATEGIES  # Assuming STRATEGIES is a dict mapping name to class
from utils.notifier import NotificationManager  # New import for notifications

logger = logging.getLogger(__name__)


class StrategyRouter:
    """
    Routes trading strategies based on market regime and manages capital allocation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Configuration for regime-to-strategy mapping
        self.regime_strategies_config: Dict[str, Dict[str, float]] = \
            self.settings.get('strategy_router.regime_strategies', {})

        self.notification_manager = NotificationManager(self.settings)  # Initialize notification manager

        self.active_strategies: Dict[str, Strategy] = {}
        self.current_market_regime: str = "unknown"
        self.last_strategy_change_time: Optional[datetime] = None

        logger.info("StrategyRouter initialized.")

    def update_market_regime(self, new_regime: str, current_total_capital: float) -> bool:
        """
        Updates the current market regime and adjusts active strategies accordingly.

        Args:
            new_regime: The newly detected market regime (e.g., "bull", "bear", "sideways").
            current_total_capital: The total capital available to the bot.

        Returns:
            True if strategies were updated, False otherwise.
        """
        if new_regime == self.current_market_regime:
            # logger.debug(f"Market regime remains '{new_regime}'. No strategy change needed.")
            return False

        log_message = f"Market regime changed from '{self.current_market_regime}' to '{new_regime}'."
        logger.info(log_message)
        self.notification_manager.send_alert(f"Market Regime Change: {self.current_market_regime} -> {new_regime}",
                                             level="INFO")

        self.current_market_regime = new_regime
        self.last_strategy_change_time = datetime.now()

        return self._adjust_strategies_for_regime(new_regime, current_total_capital)

    def _adjust_strategies_for_regime(self, regime: str, total_capital: float) -> bool:
        """
        Activates/deactivates strategies and allocates capital based on the given regime.
        """
        target_allocations = self.regime_strategies_config.get(regime, {})

        # Handle special case: "manual_intervention_required" or no strategies for regime
        if regime == "manual_intervention_required" or not target_allocations:
            logger.warning(
                f"Regime '{regime}' requires manual intervention or has no defined strategies. Pausing all active strategies.")
            self._deactivate_all_strategies()
            self.notification_manager.send_alert(
                f"Bot paused: Regime '{regime}' requires manual intervention or no strategies defined.",
                level="WARNING")
            return False

        strategies_to_activate = set(target_allocations.keys())
        currently_active_names = set(self.active_strategies.keys())

        # Deactivate strategies that are no longer needed
        for strategy_name in currently_active_names:
            if strategy_name not in strategies_to_activate:
                self._deactivate_strategy(strategy_name)

        # Activate or re-allocate capital for required strategies
        for strategy_name, allocation_ratio in target_allocations.items():
            allocated_capital = total_capital * allocation_ratio
            if strategy_name not in STRATEGIES:
                logger.error(
                    f"Strategy '{strategy_name}' not found in available strategies (STRATEGIES dict). Skipping.")
                self.notification_manager.send_alert(
                    f"Error: Strategy '{strategy_name}' not found in router config or STRATEGIES dict.", level="ERROR")
                continue

            if strategy_name not in self.active_strategies:
                # Initialize and activate new strategy
                strategy_class = STRATEGIES[strategy_name]
                strategy_params = self.settings.get(f'strategy_configs.{strategy_name}', {})
                new_strategy = strategy_class(strategy_params)
                self.active_strategies[strategy_name] = new_strategy
                logger.info(f"Activated strategy '{strategy_name}' with {allocated_capital:.2f} capital.")
                self.notification_manager.send_alert(
                    f"Strategy '{strategy_name}' activated for regime '{regime}'. Capital: {allocated_capital:.2f}",
                    level="INFO")

                # In a real bot, you'd also call a method like new_strategy.start_trading(allocated_capital)
                # or pass this allocated_capital to the strategy's position sizing logic.
                if hasattr(new_strategy, 'set_allocated_capital'):  # Example hook
                    new_strategy.set_allocated_capital(allocated_capital)
                # You might need to pass exchange/data_manager references to the strategy
                # This depends on how your Strategy base class expects its dependencies.

            else:
                # Update capital allocation for already active strategy
                logger.info(f"Re-allocated {allocated_capital:.2f} capital to active strategy '{strategy_name}'.")
                if hasattr(self.active_strategies[strategy_name], 'update_capital_allocation'):
                    self.active_strategies[strategy_name].update_capital_allocation(allocated_capital)

        logger.info(
            f"Strategies adjusted for regime '{regime}'. Active strategies: {list(self.active_strategies.keys())}")
        return True

    def _deactivate_strategy(self, strategy_name: str):
        """Deactivates a single strategy."""
        if strategy_name in self.active_strategies:
            # For a live bot, this would involve closing positions and stopping strategy threads/tasks
            # self.active_strategies[strategy_name].stop_trading() # Example API call
            logger.info(
                f"Deactivating strategy: {strategy_name}. (Note: Actual position closing not implemented here.)")
            del self.active_strategies[strategy_name]
            logger.info(f"Deactivated strategy: {strategy_name}.")
            self.notification_manager.send_alert(f"Strategy '{strategy_name}' deactivated.", level="INFO")

    def _deactivate_all_strategies(self):
        """Deactivates all currently active strategies."""
        for strategy_name in list(self.active_strategies.keys()):
            self._deactivate_strategy(strategy_name)
        logger.info("All active strategies deactivated.")

    def get_active_strategies(self) -> Dict[str, Strategy]:
        """Returns currently active strategy instances."""
        return self.active_strategies

    def get_current_regime(self) -> str:
        """Returns the last known market regime."""
        return self.current_market_regime