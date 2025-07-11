"""Ultimate AutoPilot Trading Strategy - 6 Strategy Orchestrator"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from .strategy_base import Strategy, Signal

logger = logging.getLogger(__name__)

class UltimateAutoPilotStrategy(Strategy):
    """Ultimate 6-Strategy Orchestrator"""

    def __init__(self, params: Dict = None):
        super().__init__(params)

        # Import all strategies dynamically
        self.strategies = {}
        try:
            from .momentum import MomentumStrategy
            self.strategies['momentum'] = MomentumStrategy(params.get('momentum_params', {}))
        except: pass

        try:
            from .mean_reversion import MeanReversionStrategy
            self.strategies['mean_reversion'] = MeanReversionStrategy(params.get('mr_params', {}))
        except: pass

        try:
            from .ml_strategy import MLStrategy
            self.strategies['ml'] = MLStrategy(params.get('ml_params', {}))
        except: pass

        try:
            from .grid_trading import GridTradingStrategy
            self.strategies['grid'] = GridTradingStrategy(params.get('grid_params', {}))
        except: pass

        try:
            from .arbitrage import ArbitrageStrategy
            self.strategies['arbitrage'] = ArbitrageStrategy(params.get('arb_params', {}))
        except: pass

        try:
            from .defi_yield import DeFiYieldStrategy
            self.strategies['defi'] = DeFiYieldStrategy(params.get('defi_params', {}))
        except: pass

        logger.info(f"AutoPilot initialized with {len(self.strategies)} strategies")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        if not self.strategies:
            return Signal.HOLD, 0.0

        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0

        # Collect votes from all strategies
        for name, strategy in self.strategies.items():
            try:
                signal, confidence = strategy.calculate_signal(symbol, data, current_price)
                votes[signal.value] += confidence
                total_confidence += confidence
            except Exception as e:
                logger.error(f"Error in {name}: {e}")

        if total_confidence == 0:
            return Signal.HOLD, 0.0

        # Determine winning signal
        if votes['BUY'] > votes['SELL'] and votes['BUY'] > votes['HOLD']:
            return Signal.BUY, votes['BUY'] / total_confidence
        elif votes['SELL'] > votes['BUY'] and votes['SELL'] > votes['HOLD']:
            return Signal.SELL, votes['SELL'] / total_confidence

        return Signal.HOLD, votes['HOLD'] / total_confidence
