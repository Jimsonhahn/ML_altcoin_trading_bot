"""
Ultimate AutoPilot Trading Strategy - Lazy Millionaire Stack
Orchestriert alle 6 Handelsstrategien für maximale Profite
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from .strategy_base import Strategy, Signal
from .grid_trading import GridTradingStrategy
from .arbitrage import ArbitrageStrategy
from .defi_yield import DeFiYieldStrategy
from .liquidation import LiquidationStrategy
from .copy_trading import CopyTradingStrategy
from .ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyVote:
    """Speichert das Voting-Ergebnis einer Strategie"""
    strategy_name: str
    signal: Signal
    confidence: float
    position_size: float
    metadata: Dict[str, Any]


class UltimateAutoPilotStrategy(Strategy):
    """
    Der ultimative Orchestrator - Kombiniert alle 6 Strategien
    für maximale Profite im Lazy Millionaire Stack
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)

        # Initialisiere alle 6 Sub-Strategien
        self.strategies = {
            'grid_trading': GridTradingStrategy(params.get('grid_params', {})),
            'arbitrage': ArbitrageStrategy(params.get('arbitrage_params', {})),
            'defi_yield': DeFiYieldStrategy(params.get('defi_params', {})),
            'liquidation': LiquidationStrategy(params.get('liquidation_params', {})),
            'copy_trading': CopyTradingStrategy(params.get('copy_trading_params', {})),
            'ml_strategy': MLStrategy(params.get('ml_params', {}))
        }

        # Strategie-Gewichtungen (können dynamisch angepasst werden)
        self.strategy_weights = {
            'grid_trading': params.get('grid_weight', 0.20),
            'arbitrage': params.get('arbitrage_weight', 0.20),
            'defi_yield': params.get('defi_weight', 0.15),
            'liquidation': params.get('liquidation_weight', 0.15),
            'copy_trading': params.get('copy_trading_weight', 0.15),
            'ml_strategy': params.get('ml_weight', 0.15)
        }

        # Risk Management
        self.max_total_exposure = params.get('max_total_exposure', 0.95)
        self.max_per_strategy = params.get('max_per_strategy', 0.3)
        self.min_confidence_threshold = params.get('min_confidence_threshold', 0.6)

        # Performance Tracking
        self.strategy_performance = {name: {'wins': 0, 'losses': 0, 'total_return': 0.0}
                                     for name in self.strategies.keys()}
        self.last_rebalance = datetime.now()
        self.rebalance_interval = timedelta(hours=params.get('rebalance_hours', 24))

        # Multi-Exchange Support für Arbitrage
        self.exchanges = params.get('exchanges', ['binance'])

        # Capital Allocation
        self.capital_allocation = {}
        self._rebalance_capital()

        logger.info(f"Ultimate AutoPilot initialized with {len(self.strategies)} strategies")
        logger.info(f"Strategy weights: {self.strategy_weights}")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                         current_price: float) -> Tuple[Signal, float]:
        """
        Sammelt Signale von allen 6 Strategien und kombiniert sie intelligent
        """
        votes = []
        total_weight = 0

        # Sammle Votes von allen Strategien
        for name, strategy in self.strategies.items():
            try:
                # Manche Strategien brauchen zusätzliche Daten
                if name == 'arbitrage':
                    # Arbitrage braucht Preise von mehreren Exchanges
                    signal, confidence = self._get_arbitrage_signal(symbol, current_price)
                elif name == 'defi_yield':
                    # DeFi braucht Yield-Daten
                    signal, confidence = self._get_defi_signal(symbol, data, current_price)
                elif name == 'liquidation':
                    # Liquidation braucht Liquidationsdaten
                    signal, confidence = self._get_liquidation_signal(symbol, data, current_price)
                elif name == 'copy_trading':
                    # Copy Trading braucht Whale-Daten
                    signal, confidence = self._get_copy_trading_signal(symbol, data, current_price)
                else:
                    # Standard-Strategien
                    signal, confidence = strategy.calculate_signal(symbol, data, current_price)

                # Nur Votes mit ausreichender Konfidenz berücksichtigen
                if confidence >= self.min_confidence_threshold:
                    weight = self.strategy_weights[name] * confidence
                    position_size = self._calculate_strategy_position_size(
                        name, signal, confidence, current_price
                    )

                    votes.append(StrategyVote(
                        strategy_name=name,
                        signal=signal,
                        confidence=confidence,
                        position_size=position_size,
                        metadata={'weight': weight}
                    ))
                    total_weight += weight

                    logger.debug(f"{symbol} - {name}: {signal.value} "
                                 f"(confidence: {confidence:.2f}, weight: {weight:.2f})")

            except Exception as e:
                logger.error(f"Error in strategy {name}: {e}")
                continue

        if not votes:
            return Signal.HOLD, 0.0

        # Kombiniere Votes zu finalem Signal
        final_signal, final_confidence = self._combine_votes(votes, total_weight)

        # Dynamische Strategie-Anpassung basierend auf Performance
        if (datetime.now() - self.last_rebalance) > self.rebalance_interval:
            self._adapt_strategy_weights()
            self._rebalance_capital()

        return final_signal, final_confidence

    def _combine_votes(self, votes: List[StrategyVote],
                       total_weight: float) -> Tuple[Signal, float]:
        """
        Intelligente Kombination der Strategie-Votes
        """
        if total_weight == 0:
            return Signal.HOLD, 0.0

        # Zähle gewichtete Votes
        buy_weight = sum(v.metadata['weight'] for v in votes if v.signal == Signal.BUY)
        sell_weight = sum(v.metadata['weight'] for v in votes if v.signal == Signal.SELL)
        hold_weight = sum(v.metadata['weight'] for v in votes if v.signal == Signal.HOLD)

        # Normalisiere
        buy_score = buy_weight / total_weight
        sell_score = sell_weight / total_weight
        hold_score = hold_weight / total_weight

        # Bestimme stärkstes Signal
        if buy_score > sell_score and buy_score > hold_score:
            # Zusätzliche Checks für BUY
            if buy_score > 0.4:  # Mindestens 40% der gewichteten Votes
                avg_confidence = np.mean([v.confidence for v in votes if v.signal == Signal.BUY])
                return Signal.BUY, avg_confidence

        elif sell_score > buy_score and sell_score > hold_score:
            # Zusätzliche Checks für SELL
            if sell_score > 0.4:
                avg_confidence = np.mean([v.confidence for v in votes if v.signal == Signal.SELL])
                return Signal.SELL, avg_confidence

        # Default: HOLD
        return Signal.HOLD, hold_score

    def _get_arbitrage_signal(self, symbol: str, current_price: float) -> Tuple[Signal, float]:
        """
        Spezialisierte Arbitrage-Signal-Berechnung
        """
        try:
            # Simuliere Multi-Exchange-Preise (in Produktion: echte API-Calls)
            prices = {
                'binance': current_price,
                'kraken': current_price * (1 + np.random.uniform(-0.005, 0.005)),
                'coinbase': current_price * (1 + np.random.uniform(-0.005, 0.005))
            }

            max_price = max(prices.values())
            min_price = min(prices.values())
            spread = (max_price - min_price) / min_price

            # Arbitrage-Opportunity
            if spread > 0.002:  # 0.2% Spread
                confidence = min(spread * 100, 1.0)
                # Kaufe auf der billigsten Exchange
                if prices['binance'] == min_price:
                    return Signal.BUY, confidence
                # Verkaufe auf der teuersten Exchange
                elif prices['binance'] == max_price:
                    return Signal.SELL, confidence

            return Signal.HOLD, 0.3

        except Exception as e:
            logger.error(f"Arbitrage signal error: {e}")
            return Signal.HOLD, 0.0

    def _get_defi_signal(self, symbol: str, data: pd.DataFrame,
                         current_price: float) -> Tuple[Signal, float]:
        """
        DeFi Yield Farming Signal
        """
        try:
            # Simuliere Yield-Daten
            current_apy = np.random.uniform(0.05, 0.25)  # 5-25% APY

            # Wenn hoher Yield und stabiler Preis -> BUY
            price_volatility = data['close'].pct_change().std()

            if current_apy > 0.15 and price_volatility < 0.02:
                return Signal.BUY, 0.8
            elif current_apy < 0.08 or price_volatility > 0.05:
                return Signal.SELL, 0.7

            return Signal.HOLD, 0.5

        except Exception as e:
            logger.error(f"DeFi signal error: {e}")
            return Signal.HOLD, 0.0

    def _get_liquidation_signal(self, symbol: str, data: pd.DataFrame,
                                current_price: float) -> Tuple[Signal, float]:
        """
        Liquidation Hunter Signal
        """
        try:
            # Identifiziere potenzielle Liquidationslevel
            # (In Produktion: echte Liquidationsdaten von der Chain)
            recent_high = data['high'].rolling(24).max().iloc[-1]
            recent_low = data['low'].rolling(24).min().iloc[-1]

            # Liquidation-Cluster oft bei runden Zahlen oder wichtigen Levels
            distance_to_high = (recent_high - current_price) / current_price
            distance_to_low = (current_price - recent_low) / current_price

            # Wenn Preis nahe an Liquidationslevel
            if distance_to_low < 0.02:  # 2% über potenziellem Liquidationslevel
                return Signal.BUY, 0.85  # Bounce-Trade
            elif distance_to_high < 0.02:  # 2% unter Liquidationslevel
                return Signal.SELL, 0.85

            return Signal.HOLD, 0.4

        except Exception as e:
            logger.error(f"Liquidation signal error: {e}")
            return Signal.HOLD, 0.0

    def _get_copy_trading_signal(self, symbol: str, data: pd.DataFrame,
                                 current_price: float) -> Tuple[Signal, float]:
        """
        Copy Trading - Folge den Walen
        """
        try:
            # Simuliere Whale-Aktivität (In Produktion: On-Chain-Daten)
            whale_buying = np.random.random() > 0.7
            whale_selling = np.random.random() > 0.7

            if whale_buying and not whale_selling:
                return Signal.BUY, 0.9
            elif whale_selling and not whale_buying:
                return Signal.SELL, 0.9

            return Signal.HOLD, 0.3

        except Exception as e:
            logger.error(f"Copy trading signal error: {e}")
            return Signal.HOLD, 0.0

    def _calculate_strategy_position_size(self, strategy_name: str,
                                          signal: Signal, confidence: float,
                                          current_price: float) -> float:
        """
        Berechnet Positionsgröße für eine spezifische Strategie
        """
        if signal == Signal.HOLD:
            return 0.0

        # Basis-Allokation für diese Strategie
        allocated_capital = self.capital_allocation.get(strategy_name, 0.1)

        # Anpassung basierend auf Confidence
        position_size = allocated_capital * confidence

        # Strategie-spezifische Anpassungen
        if strategy_name == 'arbitrage':
            # Arbitrage: Volle Position bei guten Opportunities
            position_size *= 1.5
        elif strategy_name == 'liquidation':
            # Liquidation: Kleinere Positionen, höheres Risiko
            position_size *= 0.7
        elif strategy_name == 'grid_trading':
            # Grid: Viele kleine Positionen
            position_size *= 0.5

        # Sicherstellen, dass Limits eingehalten werden
        return min(position_size, self.max_per_strategy)

    def _adapt_strategy_weights(self):
        """
        Passt Strategie-Gewichtungen basierend auf Performance an
        """
        logger.info("Adapting strategy weights based on performance...")

        # Berechne Performance-Scores
        performance_scores = {}
        for name, perf in self.strategy_performance.items():
            total_trades = perf['wins'] + perf['losses']
            if total_trades > 0:
                win_rate = perf['wins'] / total_trades
                avg_return = perf['total_return'] / total_trades
                score = win_rate * 0.6 + min(avg_return, 1.0) * 0.4
                performance_scores[name] = score
            else:
                performance_scores[name] = 0.5  # Neutral für neue Strategien

        # Passe Gewichtungen an (max ±20% Änderung)
        for name in self.strategies:
            current_weight = self.strategy_weights[name]
            performance = performance_scores[name]

            if performance > 0.6:
                # Erhöhe Gewichtung für gut performende Strategien
                new_weight = min(current_weight * 1.2, 0.35)
            elif performance < 0.4:
                # Reduziere Gewichtung für schlecht performende Strategien
                new_weight = max(current_weight * 0.8, 0.05)
            else:
                new_weight = current_weight

            self.strategy_weights[name] = new_weight

        # Normalisiere Gewichtungen
        total = sum(self.strategy_weights.values())
        for name in self.strategy_weights:
            self.strategy_weights[name] /= total

        logger.info(f"Updated strategy weights: {self.strategy_weights}")
        self.last_rebalance = datetime.now()

    def _rebalance_capital(self):
        """
        Verteilt Kapital auf Strategien basierend auf Gewichtungen
        """
        self.capital_allocation = {}
        for name, weight in self.strategy_weights.items():
            self.capital_allocation[name] = weight * self.max_total_exposure

        logger.info(f"Capital allocation updated: {self.capital_allocation}")

    def calculate_position_size(self, symbol: str, signal: Signal,
                                confidence: float, current_price: float,
                                balance: float) -> float:
        """
        Finale Positionsgröße über alle Strategien
        """
        if signal == Signal.HOLD:
            return 0.0

        # Aggregiere Positionsgrößen von allen Strategien
        total_position = 0

        for name, strategy in self.strategies.items():
            strategy_signal, strategy_confidence = strategy.calculate_signal(
                symbol, pd.DataFrame(), current_price
            )

            if strategy_signal == signal and strategy_confidence >= self.min_confidence_threshold:
                position = self._calculate_strategy_position_size(
                    name, signal, strategy_confidence, current_price
                )
                total_position += position * self.strategy_weights[name]

        # Finale Position basierend auf verfügbarem Balance
        final_position = min(total_position * balance, balance * self.max_total_exposure)

        return final_position

    def update_performance(self, strategy_name: str, profit: float):
        """
        Aktualisiert Performance-Tracking für adaptive Gewichtung
        """
        if strategy_name in self.strategy_performance:
            if profit > 0:
                self.strategy_performance[strategy_name]['wins'] += 1
            else:
                self.strategy_performance[strategy_name]['losses'] += 1
            self.strategy_performance[strategy_name]['total_return'] += profit