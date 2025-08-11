"""Ultimate AutoPilot Strategy"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .strategy_base import Strategy

logger = logging.getLogger(__name__)

class UltimateAutoPilotStrategy(Strategy):
    """Ultimate AutoPilot - Orchestrates all strategies"""

    def __init__(self, config: Dict[str, Any]):
        # Handle Settings object
        if hasattr(config, 'config'):
            config_dict = dict(config.config)
        else:
            config_dict = config if isinstance(config, dict) else {}

        super().__init__(config_dict)

        self.config = config_dict
        self.params = config_dict
        self.name = "Ultimate AutoPilot"
        self.version = "3.0"

        self.mode = config_dict.get('mode', 'balanced')
        self.rebalance_interval = config_dict.get('rebalance_interval', 3600)

        # Default capital allocation
        default_allocation = {
            'grid_trading': 0.25,
            'arbitrage': 0.20,
            'defi_yield': 0.20,
            'momentum': 0.15,
            'mean_reversion': 0.10,
            'ml': 0.05,
            'liquidation': 0.05
        }
        self.capital_allocation = config_dict.get('capital_allocation', default_allocation)

        self.strategy_performance = {}
        self.last_rebalance = datetime.now()
        self.sub_strategies = {}
        self.active_strategies = []
        self.defi_config = config_dict.get('defi', {})

        self._initialize_strategies()

        logger.info(f"🚀 Ultimate AutoPilot v{self.version} initialized")
        logger.info(f"💰 Capital allocation: {self.capital_allocation}")
        logger.info(f"🎯 Active strategies: {self.active_strategies}")

    def _initialize_strategies(self):
        """Initialize all sub-strategies"""
        strategy_classes = {
            'momentum': ('momentum', 'MomentumStrategy'),
            'mean_reversion': ('mean_reversion', 'MeanReversionStrategy'),
            'ml': ('ml_strategy', 'MLStrategy'),
            'grid_trading': ('grid_trading', 'GridTradingStrategy'),
            'arbitrage': ('arbitrage', 'ArbitrageStrategy'),
            'liquidation': ('liquidation', 'LiquidationStrategy'),
            'defi_yield': ('defi_yield', 'DeFiYieldStrategy')
        }

        for strategy_name, (module_name, class_name) in strategy_classes.items():
            if self.capital_allocation.get(strategy_name, 0) > 0:
                try:
                    module = __import__(f'strategies.{module_name}', fromlist=[class_name])
                    strategy_class = getattr(module, class_name)
                    strategy_params = dict(self.params)

                    if strategy_name == 'defi_yield' and self.defi_config:
                        strategy_params.update(self.defi_config)

                    self.sub_strategies[strategy_name] = strategy_class(strategy_params)
                    self.active_strategies.append(strategy_name)
                    logger.info(f"✅ {strategy_name.replace('_', ' ').title()} strategy loaded")

                except Exception as e:
                    logger.warning(f"Could not load {strategy_name}: {e}")
                    self.capital_allocation[strategy_name] = 0

        # Normalize allocations
        total = sum(self.capital_allocation.values())
        if total > 0:
            for strategy in self.capital_allocation:
                self.capital_allocation[strategy] /= total

        logger.info(f"AutoPilot initialized with {len(self.active_strategies)} strategies")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate combined signal from all strategies"""
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return 'HOLD', {'confidence': 0.0, 'reason': 'no_data'}

        if not self.active_strategies:
            logger.warning("No active strategies!")
            return 'HOLD', {'confidence': 0.0, 'reason': 'no_active_strategies'}

        if self._should_rebalance():
            self._rebalance_portfolio()

        # Collect signals
        all_signals = {}
        max_workers = max(1, len(self.active_strategies))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for strategy_name in self.active_strategies:
                if strategy_name in self.sub_strategies:
                    future = executor.submit(
                        self._get_strategy_signal,
                        strategy_name, symbol, data, current_price
                    )
                    futures[future] = strategy_name

            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    signal, signal_data = future.result()
                    all_signals[strategy_name] = (signal, signal_data)
                except Exception as e:
                    logger.error(f"Error from {strategy_name}: {e}")
                    all_signals[strategy_name] = ('HOLD', {'confidence': 0.0, 'error': str(e)})

        # Log signals
        if all_signals:
            logger.info(f"AutoPilot signals for {symbol}:")
            for name, (sig, data) in all_signals.items():
                conf = data.get('confidence', 0)
                logger.info(f"  - {name}: {sig} ({conf:.2f})")

        # Combine signals
        final_signal, final_data = self._combine_signals(all_signals)

        final_data['strategy'] = 'ultimate_autopilot'
        final_data['sub_signals'] = all_signals
        final_data['active_strategies'] = self.active_strategies

        return final_signal, final_data

    def _get_strategy_signal(self, strategy_name: str, symbol: str,
                           data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Get signal from a specific strategy"""
        try:
            strategy = self.sub_strategies[strategy_name]
            result = strategy.calculate_signal(symbol, data, current_price)

            # Handle both old and new formats
            if isinstance(result, tuple) and len(result) == 2:
                signal, confidence_or_dict = result

                # Convert Signal enum to string
                if hasattr(signal, 'value'):
                    signal = signal.value
                else:
                    signal = str(signal)

                # Handle confidence
                if isinstance(confidence_or_dict, dict):
                    return signal, confidence_or_dict
                else:
                    return signal, {
                        'confidence': float(confidence_or_dict),
                        'signal': signal,
                        'strategy': strategy_name
                    }

            return 'HOLD', {'confidence': 0.0, 'error': 'unexpected_format'}

        except Exception as e:
            logger.error(f"Error in {strategy_name}: {e}")
            return 'HOLD', {'confidence': 0.0, 'error': str(e)}

    def _combine_signals(self, all_signals: Dict) -> Tuple[str, Dict[str, Any]]:
        """Combine signals with weighted voting"""
        if not all_signals:
            return 'HOLD', {'confidence': 0.0, 'reason': 'no_signals'}

        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0

        for strategy_name, (signal, signal_data) in all_signals.items():
            weight = self.capital_allocation.get(strategy_name, 0)
            if weight == 0:
                continue

            confidence = signal_data.get('confidence', 0.5)
            vote_weight = weight * confidence

            if signal == 'BUY':
                buy_score += vote_weight
            elif signal == 'SELL':
                sell_score += vote_weight
            else:
                hold_score += vote_weight

            total_weight += weight
            weighted_confidence += confidence * weight

        # Determine final signal
        max_score = max(buy_score, sell_score, hold_score)

        if max_score == buy_score and buy_score > 0.3:
            final_signal = 'BUY'
        elif max_score == sell_score and sell_score > 0.3:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        action = "🟢" if final_signal == "BUY" else "🔴" if final_signal == "SELL" else "⚪"
        logger.info(f"{action} AutoPilot decision: {final_signal} (confidence: {final_confidence:.2f})")

        return final_signal, {
            'signal': final_signal,
            'confidence': final_confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'hold_score': hold_score
        }

    def _should_rebalance(self) -> bool:
        """Check if rebalancing is needed"""
        time_since = (datetime.now() - self.last_rebalance).total_seconds()
        return time_since >= self.rebalance_interval

    def _rebalance_portfolio(self):
        """Rebalance portfolio"""
        logger.info("🔄 Rebalancing AutoPilot portfolio...")
        self.last_rebalance = datetime.now()
