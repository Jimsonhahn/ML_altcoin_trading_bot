"""
Ultimate AutoPilot Strategy - Orchestriert alle 6 Strategien
Grid + Arbitrage + DeFi + Liquidation + Copy Trading + ML
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from .strategy_base import Strategy, Signal

logger = logging.getLogger(__name__)


class UltimateAutoPilotStrategy(Strategy):
    """
    Der ultimative Orchestrator - Alle 6 Strategien arbeiten zusammen!
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)

        # Initialisiere alle Sub-Strategien
        self.strategies = {}
        self._load_strategies()

        # Strategie-Gewichtungen
        self.weights = {
            'momentum': 0.20,
            'mean_reversion': 0.15,
            'ml': 0.15,
            'grid_trading': 0.20,
            'arbitrage': 0.15,
            'liquidation': 0.15
        }

        logger.info(f"AutoPilot initialized with {len(self.strategies)} strategies")
        logger.info(f"Active strategies: {list(self.strategies.keys())}")

    def _load_strategies(self):
        """Lade alle verfügbaren Strategien"""
        # Core Strategies
        try:
            from .momentum import MomentumStrategy
            self.strategies['momentum'] = MomentumStrategy(self.params)
            logger.info("✅ Momentum strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load Momentum: {e}")

        try:
            from .mean_reversion import MeanReversionStrategy
            self.strategies['mean_reversion'] = MeanReversionStrategy(self.params)
            logger.info("✅ Mean Reversion strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load Mean Reversion: {e}")

        try:
            from .ml_strategy import MLStrategy
            self.strategies['ml'] = MLStrategy(self.params)
            logger.info("✅ ML strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load ML: {e}")

        # Advanced Strategies
        try:
            from .grid_trading import GridTradingStrategy
            self.strategies['grid_trading'] = GridTradingStrategy(self.params)
            logger.info("✅ Grid Trading strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load Grid Trading: {e}")

        try:
            from .arbitrage import ArbitrageStrategy
            self.strategies['arbitrage'] = ArbitrageStrategy(self.params)
            logger.info("✅ Arbitrage strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load Arbitrage: {e}")

        try:
            from .liquidation import LiquidationStrategy
            self.strategies['liquidation'] = LiquidationStrategy(self.params)
            logger.info("✅ Liquidation strategy loaded")
        except Exception as e:
            logger.warning(f"Could not load Liquidation: {e}")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        """
        Kombiniert Signale von allen 6 Strategien!
        """
        if not self.strategies:
            logger.error("No strategies loaded!")
            return Signal.HOLD, 0.0

        # Sammle Votes von allen Strategien
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        strategy_signals = []

        for name, strategy in self.strategies.items():
            try:
                # Hole Signal von jeder Strategie
                signal, confidence = strategy.calculate_signal(symbol, data, current_price)
                weight = self.weights.get(name, 0.15)

                # Weighted voting
                votes[signal.value] += confidence * weight
                total_weight += weight

                strategy_signals.append({
                    'strategy': name,
                    'signal': signal.value,
                    'confidence': confidence,
                    'weight': weight
                })

                logger.debug(f"{name}: {signal.value} (conf: {confidence:.2f}, weight: {weight:.2f})")

            except Exception as e:
                logger.error(f"Error in {name} strategy: {e}")
                continue

        # Log alle Signale
        logger.info(f"AutoPilot signals for {symbol}:")
        for sig in strategy_signals:
            logger.info(f"  - {sig['strategy']}: {sig['signal']} ({sig['confidence']:.2f})")

        if total_weight == 0:
            return Signal.HOLD, 0.0

        # Normalisiere Votes
        for signal in votes:
            votes[signal] /= total_weight

        # Bestimme finales Signal
        max_vote = max(votes.values())

        if votes['BUY'] == max_vote and votes['BUY'] > 0.4:
            confidence = votes['BUY']
            final_signal = Signal.BUY
            logger.info(f"🟢 AutoPilot decision: BUY (confidence: {confidence:.2f})")
        elif votes['SELL'] == max_vote and votes['SELL'] > 0.4:
            confidence = votes['SELL']
            final_signal = Signal.SELL
            logger.info(f"🔴 AutoPilot decision: SELL (confidence: {confidence:.2f})")
        else:
            confidence = votes['HOLD']
            final_signal = Signal.HOLD
            logger.info(f"⚪ AutoPilot decision: HOLD (confidence: {confidence:.2f})")

        return final_signal, confidence
