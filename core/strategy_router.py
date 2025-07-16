import logging
from typing import Dict, Any, Type, Optional

from config.settings import Settings
from strategies.strategy_base import Strategy
from strategies import STRATEGIES

logger = logging.getLogger(__name__)


class StrategyRouter:
    """
    Routes trading strategies based on market regime and manages capital allocation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy_configs = self.settings.get('strategy_router.regime_strategies', {})
        self.capital_allocations = self.settings.get('strategy_router.capital_allocation_rules', {})
        self.active_strategy: Optional[Strategy] = None
        self.current_strategy_name: Optional[str] = None

        logger.info("StrategyRouter initialized.")

    def set_active_strategy(self, strategy_name: str, current_market_data: Dict[str, Any]) -> bool:
        """
        Sets the active trading strategy.

        Args:
            strategy_name: The name of the strategy to activate.
            current_market_data: Current market data (e.g., OHLCV for relevant symbols).

        Returns:
            True if the strategy was successfully set, False otherwise.
        """
        if strategy_name == self.current_strategy_name:
            # logger.debug(f"Strategy {strategy_name} is already active.")
            return True

        strategy_class = STRATEGIES.get(strategy_name.lower())
        if strategy_class:
            try:
                # Load strategy-specific parameters from settings
                strategy_params = self.settings.get(f'strategy_configs.{strategy_name}', {})
                self.active_strategy = strategy_class(strategy_params)
                self.current_strategy_name = strategy_name
                logger.info(f"Strategy changed to: {strategy_name.upper()}")

                # Initialize or re-initialize the strategy with relevant market data
                if hasattr(self.active_strategy, 'initialize_strategy'):
                    self.active_strategy.initialize_strategy(current_market_data)

                return True
            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_name}: {e}")
                self.active_strategy = None
                self.current_strategy_name = None
                return False
        else:
            logger.warning(f"Strategy '{strategy_name}' not found in available strategies.")
            return False

    def get_active_strategy(self) -> Optional[Strategy]:
        """Returns the currently active strategy instance."""
        return self.active_strategy

    def get_current_strategy_name(self) -> Optional[str]:
        """Returns the name of the currently active strategy."""
        return self.current_strategy_name

    def route_strategy(self, market_regime_info: Dict[str, Any], current_market_data: Dict[str, Any]) -> str:
        """
        Selects and sets the optimal strategy based on the current market regime.

        Args:
            market_regime_info: Information about the current market regime.
                                Expected keys: 'label', 'regime', 'trading_rules'.
            current_market_data: Latest market data required for strategy initialization or updates.

        Returns:
            The name of the selected strategy.
        """
        regime_label = market_regime_info.get('label', 'unknown').lower()
        selected_strategy = None
        reasons = []

        # Find strategy based on regime mapping
        for regime_pattern, strategy_name in self.strategy_configs.items():
            if regime_pattern.lower() in regime_label:
                selected_strategy = strategy_name
                reasons.append(f"Market regime '{regime_label}' matches rule for '{strategy_name}'")
                break

        # Fallback if no specific rule matches
        if selected_strategy is None:
            selected_strategy = self.settings.get('strategy_router.default_strategy', 'momentum')
            reasons.append(f"No specific rule for '{regime_label}', falling back to default '{selected_strategy}'")

        # Apply trading rules if available for the regime
        trading_rules = market_regime_info.get('trading_rules', {})
        if trading_rules:
            top_performers = trading_rules.get('top_performers', [])
            bottom_performers = trading_rules.get('bottom_performers', [])

            # Example: Adjust strategy or parameters based on top/bottom performers
            # This is a placeholder for more complex logic
            if selected_strategy == 'momentum' and top_performers:
                # Prioritize trading pairs that are top performers in this regime
                ml_signal_data = current_market_data.get(top_performers[0], {}).get('ml_signal_data', {})
                if ml_signal_data.get('confidence', 0) > 0.7:
                    reasons.append(f"Prioritizing {top_performers[0]} due to strong performance in current regime.")
                    # In a live bot, you'd adjust trading pairs or weights here.
            elif selected_strategy == 'mean_reversion' and bottom_performers:
                # Avoid trading pairs that are bottom performers
                reasons.append(f"Avoiding {bottom_performers[0]} due to poor performance in current regime.")
                # Adjust trading pairs or risk parameters.

        self.set_active_strategy(selected_strategy, current_market_data)
        logger.info(f"Strategy routed to: {selected_strategy.upper()} because: {'. '.join(reasons)}")

        return selected_strategy

    def adjust_capital_allocation(self, market_regime_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjusts capital allocation based on the market regime.
        This would directly influence the `trading_bot`'s behavior if it supports multi-strategy allocation.

        Args:
            market_regime_info: Information about the current market regime.

        Returns:
            A dictionary with recommended capital allocation per strategy.
        """
        regime_id = market_regime_info.get('regime', 0)
        regime_label = market_regime_info.get('label', 'unknown').lower()

        # Default allocation
        allocation = self.settings.get('autopilot.capital_allocation', {})

        # Example: Adjust allocation based on specific regimes
        if 'sideways' in regime_label or 'niedrige-volatilität' in regime_label:
            # Increase allocation for arbitrage and grid trading
            allocation['arbitrage'] = allocation.get('arbitrage', 0) * 1.2
            allocation['grid_trading'] = allocation.get('grid_trading', 0) * 1.2
            # Decrease for momentum/trend strategies
            allocation['momentum'] = allocation.get('momentum', 0) * 0.8
            allocation['mean_reversion'] = allocation.get('mean_reversion', 0) * 0.9
            logger.info(f"Adjusted allocation for {regime_label} market.")

        elif 'bullish' in regime_label or 'aufwärtstrend' in regime_label:
            # Increase allocation for momentum and ML strategies
            allocation['momentum'] = allocation.get('momentum', 0) * 1.2
            allocation['ml'] = allocation.get('ml', 0) * 1.3
            # Decrease for arbitrage/grid
            allocation['arbitrage'] = allocation.get('arbitrage', 0) * 0.8
            allocation['grid_trading'] = allocation.get('grid_trading', 0) * 0.9
            logger.info(f"Adjusted allocation for {regime_label} market.")

        elif 'bearish' in regime_label or 'abwärtstrend' in regime_label:
            # Increase allocation for mean reversion (if shorting) or stablecoins/DeFi
            # Assuming mean reversion can profit from pullbacks in downtrends
            allocation['mean_reversion'] = allocation.get('mean_reversion', 0) * 1.2
            allocation['defi_yield'] = allocation.get('defi_yield', 0) * 1.1  # More capital in stablecoin yields
            # Reduce riskier strategies
            allocation['momentum'] = allocation.get('momentum', 0) * 0.7
            logger.info(f"Adjusted allocation for {regime_label} market.")

        # Normalize allocations to sum to 1.0
        total_sum = sum(allocation.values())
        if total_sum > 0:
            normalized_allocation = {k: v / total_sum for k, v in allocation.items()}
        else:
            normalized_allocation = allocation  # Return as is if sum is zero, implies all strategies are disabled or invalid.

        self.settings.set('autopilot.capital_allocation', normalized_allocation)
        logger.info(f"New capital allocation: {normalized_allocation}")
        return normalized_allocation