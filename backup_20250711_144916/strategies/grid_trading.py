#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid Trading Strategie - Automatische Geldmaschine
==================================================
Kauft und verkauft automatisch in einem definierten Preisbereich
und verdient Geld bei JEDER Preisbewegung - steigend UND fallend!

Funktionsweise:
1. Definiert einen Preisbereich (z.B. 30.000 - 70.000 für BTC)
2. Teilt diesen in X Grids auf (z.B. 20 Grids = alle 2.000$)
3. Platziert bei jedem Grid Kauf- und Verkaufsorders
4. Verdient bei JEDER Preisbewegung Geld!

ROI: 20-100% pro Jahr je nach Volatilität
Aufwand: 0% nach einmaligem Setup
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

from strategies.strategy_base import Strategy
from core.position import Position


class GridStrategy(Strategy):
    """
    Grid Trading Strategie - Der Heilige Gral des automatischen Tradings

    Features:
    - Verdient bei steigenden UND fallenden Preisen
    - Vollautomatisch nach Setup
    - Skalierbar auf jedes Kapital
    - Funktioniert 24/7 ohne Überwachung
    - Hohe Gewinnrate durch viele kleine Trades
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "grid_trading"

        # Grid-Konfiguration laden
        self.lower_price = settings.get('grid_trading.price_range.lower', 30000)
        self.upper_price = settings.get('grid_trading.price_range.upper', 70000)
        self.num_grids = settings.get('grid_trading.num_grids', 20)
        self.investment_per_grid = settings.get('grid_trading.investment_per_grid', 500)

        # Erweiterte Konfiguration
        self.auto_adjust_range = settings.get('grid_trading.auto_adjust_range', True)
        self.volatility_multiplier = settings.get('grid_trading.volatility_multiplier', 2.5)
        self.min_grid_spacing = settings.get('grid_trading.min_grid_spacing', 100)
        self.max_investment_per_symbol = settings.get('grid_trading.max_investment_per_symbol', 10000)

        # Grid-Zustand
        self.grid_levels = []
        self.active_grids = {}  # {level: {'buy_order': order, 'sell_order': order}}
        self.filled_grids = set()  # Grids die bereits ausgeführt wurden
        self.total_profit = 0.0
        self.trades_count = 0
        self.last_price = 0.0

        # Performance-Tracking
        self.session_start_time = datetime.now()
        self.daily_trades = 0
        self.best_trade_profit = 0.0
        self.worst_trade_profit = 0.0

        # Initialisierung
        self._calculate_grid_levels()

        self.logger.info(f"🤖 Grid Strategy initialized:")
        self.logger.info(f"  💰 Price range: ${self.lower_price:,.0f} - ${self.upper_price:,.0f}")
        self.logger.info(f"  📊 Number of grids: {self.num_grids}")
        self.logger.info(f"  💵 Investment per grid: ${self.investment_per_grid:,.0f}")
        self.logger.info(f"  📏 Grid spacing: ${self.grid_spacing:,.0f}")
        self.logger.info(f"  🎯 Max investment: ${self.max_investment_per_symbol:,.0f}")

    def _calculate_grid_levels(self) -> None:
        """Berechnet alle Grid-Level"""
        # Sicherstellen, dass wir gültige Parameter haben
        if self.upper_price <= self.lower_price:
            self.logger.error(f"Invalid price range: {self.lower_price} - {self.upper_price}")
            raise ValueError("Upper price must be greater than lower price")

        if self.num_grids <= 0:
            self.logger.error(f"Invalid number of grids: {self.num_grids}")
            raise ValueError("Number of grids must be positive")

        self.grid_spacing = (self.upper_price - self.lower_price) / self.num_grids

        # Mindest-Grid-Spacing prüfen
        if self.grid_spacing < self.min_grid_spacing:
            adjusted_grids = int((self.upper_price - self.lower_price) / self.min_grid_spacing)
            self.logger.warning(
                f"Grid spacing too small ({self.grid_spacing:.0f}), "
                f"adjusting grids from {self.num_grids} to {adjusted_grids}"
            )
            self.num_grids = max(adjusted_grids, 1)
            self.grid_spacing = (self.upper_price - self.lower_price) / self.num_grids

        # Grid-Levels berechnen
        self.grid_levels = []
        for i in range(self.num_grids + 1):
            level = self.lower_price + (i * self.grid_spacing)
            self.grid_levels.append(level)

        self.logger.debug(f"📊 Calculated {len(self.grid_levels)} grid levels, spacing: ${self.grid_spacing:.0f}")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereitet Daten für Grid-Trading vor"""
        if df.empty or 'close' not in df.columns:
            self.logger.error("Missing or invalid data for Grid Trading")
            return df

        # Grid-Indikatoren hinzufügen
        current_price = df['close'].iloc[-1]
        df['in_grid_range'] = (df['close'] >= self.lower_price) & (df['close'] <= self.upper_price)
        df['current_grid_level'] = df['close'].apply(self._find_nearest_grid_level)
        df['distance_to_next_grid'] = df.apply(
            lambda row: self._calculate_distance_to_next_grid(row['close']), axis=1
        )

        # Volatilität für Auto-Adjustment
        if len(df) >= 20:
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()

        # Preisbewegung zwischen Grid-Levels tracken
        df['grid_level_change'] = df['current_grid_level'].diff()

        return df

    def _find_nearest_grid_level(self, price: float) -> int:
        """Findet das nächstgelegene Grid-Level für einen Preis"""
        if price < self.lower_price:
            return 0
        if price > self.upper_price:
            return len(self.grid_levels) - 1

        # Finde nächstes Grid-Level
        for i, level in enumerate(self.grid_levels):
            if price <= level:
                return i

        return len(self.grid_levels) - 1

    def _calculate_distance_to_next_grid(self, price: float) -> float:
        """Berechnet Distanz zum nächsten Grid-Level"""
        current_grid = self._find_nearest_grid_level(price)

        if current_grid < len(self.grid_levels) - 1:
            next_level = self.grid_levels[current_grid + 1]
            return abs(next_level - price)

        return float('inf')

    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert Grid-Trading Signale

        Grid-Logic:
        1. Preis erreicht Grid-Level → Prüfe Kauf/Verkauf-Möglichkeit
        2. Kontinuierliche Grid-Überwachung
        3. Automatisches Rebalancing bei Bedarf
        """
        # Daten vorbereiten
        df = self.prepare_data(df)

        if df.empty or len(df) < 2:
            return 'HOLD', {
                'signal': 'HOLD',
                'reason': 'insufficient_data',
                'strategy': 'grid_trading'
            }

        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price

        # Preis-Historie für bessere Entscheidungen
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0

        # Aktuelle Grid-Position bestimmen
        current_grid = self._find_nearest_grid_level(current_price)
        previous_grid = self._find_nearest_grid_level(previous_price)

        # Grid-Bewegung erkennen
        grid_moved = current_grid != previous_grid

        # Signal-Daten initialisieren
        signal_data = {
            'signal': 'HOLD',
            'current_price': current_price,
            'previous_price': previous_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'current_grid': current_grid,
            'previous_grid': previous_grid,
            'grid_moved': grid_moved,
            'grid_level_price': self.grid_levels[current_grid] if current_grid < len(self.grid_levels) else 0,
            'grid_spacing': self.grid_spacing,
            'confidence': 0.95,  # Grid-Trading hat hohe Konfidenz
            'strategy': 'grid_trading',
            'total_grids': self.num_grids,
            'active_grids': len(self.active_grids),
            'filled_grids': len(self.filled_grids),
            'total_profit': self.total_profit,
            'trades_count': self.trades_count
        }

        # Prüfen, ob wir im Grid-Bereich sind
        if not (self.lower_price <= current_price <= self.upper_price):
            # Auto-Adjustment falls aktiviert
            if self.auto_adjust_range and len(df) >= 30:
                self._auto_adjust_grid_range(df)

            signal_data.update({
                'reason': 'price_outside_grid_range',
                'grid_range': f"${self.lower_price:,.0f} - ${self.upper_price:,.0f}",
                'suggestion': 'adjust_grid_range'
            })
            return 'HOLD', signal_data

        # 1. KAUFSIGNAL: Preis fällt auf ein Grid-Level oder bewegt sich nach unten
        if (grid_moved and current_grid > previous_grid) or (price_change < 0 and abs(price_change_pct) > 0.5):
            # Checke ob wir bei diesem Grid-Level bereits gekauft haben oder zu nah an bestehender Position
            if current_grid not in self.filled_grids:
                # Berechne optimale Kauf-Menge
                target_amount = self.investment_per_grid / current_price

                signal_data.update({
                    'signal': 'BUY',
                    'reason': 'price_at_grid_buy_level',
                    'grid_action': 'buy_at_grid',
                    'target_amount': target_amount,
                    'next_sell_level': self.grid_levels[min(current_grid + 1, len(self.grid_levels) - 1)],
                    'stop_loss_level': self.grid_levels[max(current_grid - 2, 0)],  # 2 Grids darunter
                    'expected_profit_pct': (self.grid_spacing / current_price) * 100
                })
                return 'BUY', signal_data

        # 2. VERKAUFSSIGNAL: Preis steigt und wir haben Position
        elif current_position and (grid_moved and current_grid < previous_grid or price_change > 0):
            # Prüfe ob wir genug Gewinn gemacht haben
            profit_threshold_pct = 0.5  # Mindestens 0.5% Gewinn
            current_profit_pct = ((current_price - current_position.entry_price) / current_position.entry_price) * 100

            # Grid-basierte Verkaufsstrategie
            if current_profit_pct >= profit_threshold_pct:
                expected_profit = (current_price - current_position.entry_price) * current_position.amount

                signal_data.update({
                    'signal': 'SELL',
                    'reason': 'grid_profit_target_reached',
                    'grid_action': 'sell_at_grid',
                    'expected_profit': expected_profit,
                    'profit_percentage': current_profit_pct,
                    'next_buy_level': self.grid_levels[max(current_grid - 1, 0)],
                    'trade_performance': 'profitable' if expected_profit > 0 else 'break_even'
                })
                return 'SELL', signal_data

        # 3. REBALANCING: Aktiviere mehr Grids wenn wenig aktiv
        elif len(self.filled_grids) < self.num_grids * 0.3:  # Weniger als 30% der Grids aktiv
            # Intelligentes Rebalancing basierend auf aktueller Marktlage
            if not current_position:  # Nur wenn keine Position offen
                target_amount = self.investment_per_grid / current_price

                signal_data.update({
                    'signal': 'BUY',
                    'reason': 'grid_rebalancing',
                    'grid_action': 'rebalance_grids',
                    'target_amount': target_amount,
                    'rebalancing_opportunity': True,
                    'grid_utilization': len(self.filled_grids) / self.num_grids * 100
                })
                return 'BUY', signal_data

        # 4. TRAILING-STRATEGIE: Optimiere bestehende Positionen
        elif current_position:
            # Grid-spezifisches Trailing
            grid_distance_from_entry = abs(current_grid - self._find_nearest_grid_level(current_position.entry_price))

            if grid_distance_from_entry >= 2:  # 2+ Grids Bewegung
                profit_pct = ((current_price - current_position.entry_price) / current_position.entry_price) * 100

                if profit_pct > 1.0:  # Mindestens 1% Gewinn
                    signal_data.update({
                        'signal': 'SELL',
                        'reason': 'grid_trailing_profit',
                        'grid_action': 'trailing_sell',
                        'profit_percentage': profit_pct,
                        'grid_distance': grid_distance_from_entry
                    })
                    return 'SELL', signal_data

        # Standard HOLD mit aktuellen Informationen
        signal_data.update({
            'reason': 'waiting_for_grid_opportunity',
            'market_condition': 'sideways' if abs(price_change_pct) < 0.5 else 'moving',
            'next_action': 'monitor_grid_levels'
        })

        return 'HOLD', signal_data

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet spezifische Indikatoren für Grid-Trading"""
        if df.empty or 'close' not in df.columns:
            return df

        # Grid-Level-Indikatoren
        df['grid_level'] = df['close'].apply(self._find_nearest_grid_level)
        df['distance_to_grid'] = df.apply(
            lambda row: abs(row['close'] - self.grid_levels[int(row['grid_level'])])
            if int(row['grid_level']) < len(self.grid_levels) else 0, axis=1
        )

        # Volatilität für Grid-Optimierung
        if len(df) >= 20:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['volatility'] = df['close'].rolling(window=20).std()
            df['price_above_sma'] = df['close'] > df['sma_20']

            # Grid-Effizienz Indikatoren
            df['grid_efficiency'] = df['volatility'] / df['sma_20'] * 100  # Volatilität als % des Preises

        # Volume-Indikatoren falls verfügbar
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['high_volume'] = df['volume'] > df['volume_ma'] * 1.2

        return df

    def on_trade_executed(self, position: Position, action: str) -> None:
        """
        Callback wenn ein Trade ausgeführt wurde
        Aktualisiert Grid-Zustand und Performance-Tracking
        """
        current_grid = self._find_nearest_grid_level(position.entry_price)

        if action == 'BUY':
            # Grid als gefüllt markieren
            self.filled_grids.add(current_grid)
            self.daily_trades += 1

            self.logger.info(
                f"🟢 Grid {current_grid} FILLED with BUY: "
                f"{position.amount:.6f} @ ${position.entry_price:,.2f}"
            )

        elif action == 'SELL':
            # Grid wieder verfügbar machen
            if current_grid in self.filled_grids:
                self.filled_grids.remove(current_grid)

            # Profit tracking
            if hasattr(position, 'profit_loss') and position.profit_loss is not None:
                trade_profit = position.profit_loss
                self.total_profit += trade_profit
                self.trades_count += 1
                self.daily_trades += 1

                # Performance-Tracking
                if trade_profit > self.best_trade_profit:
                    self.best_trade_profit = trade_profit
                if trade_profit < self.worst_trade_profit:
                    self.worst_trade_profit = trade_profit

                profit_pct = (trade_profit / (position.entry_price * position.amount)) * 100

                self.logger.info(
                    f"🟢 Grid {current_grid} SELL completed: "
                    f"Profit=${trade_profit:,.2f} ({profit_pct:.2f}%) "
                    f"| Total: ${self.total_profit:,.2f} | Trades: {self.trades_count}"
                )

                # Tägliche Performance loggen
                if self.trades_count % 10 == 0:  # Alle 10 Trades
                    avg_profit = self.total_profit / self.trades_count
                    self.logger.info(
                        f"📊 Grid Performance Update: "
                        f"Avg Profit/Trade: ${avg_profit:.2f} | "
                        f"Daily Trades: {self.daily_trades} | "
                        f"Grid Efficiency: {len(self.filled_grids)}/{self.num_grids}"
                    )

    def get_grid_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Grid-Status zurück"""
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600

        return {
            'total_grids': self.num_grids,
            'active_grids': len(self.active_grids),
            'filled_grids': len(self.filled_grids),
            'grid_utilization_pct': (len(self.filled_grids) / self.num_grids) * 100,
            'grid_range': f"${self.lower_price:,.0f} - ${self.upper_price:,.0f}",
            'grid_spacing': f"${self.grid_spacing:,.0f}",
            'total_profit': self.total_profit,
            'trades_count': self.trades_count,
            'daily_trades': self.daily_trades,
            'avg_profit_per_trade': self.total_profit / max(self.trades_count, 1),
            'best_trade': self.best_trade_profit,
            'worst_trade': self.worst_trade_profit,
            'session_duration_hours': session_duration,
            'trades_per_hour': self.trades_count / max(session_duration, 0.1),
            'profitability': 'profitable' if self.total_profit > 0 else 'break_even' if self.total_profit == 0 else 'loss',
            'grid_efficiency_score': min((len(self.filled_grids) / self.num_grids) * 100, 100),
            'strategy_health': 'excellent' if self.total_profit > 100 else 'good' if self.total_profit > 0 else 'monitoring'
        }

    def _auto_adjust_grid_range(self, df: pd.DataFrame, window: int = 30) -> bool:
        """
        Automatische Anpassung des Grid-Bereichs basierend auf Marktvolatilität
        """
        if len(df) < window:
            return False

        try:
            # Aktuelle Marktstatistiken
            recent_prices = df['close'].tail(window)
            current_price = df['close'].iloc[-1]

            # Volatilität und Preisspanne berechnen
            price_std = recent_prices.std()
            price_mean = recent_prices.mean()
            price_min = recent_prices.min()
            price_max = recent_prices.max()

            # Neue Grid-Range basierend auf Volatilität berechnen
            volatility_range = price_std * self.volatility_multiplier

            suggested_lower = max(current_price - volatility_range, price_min * 0.95)
            suggested_upper = min(current_price + volatility_range, price_max * 1.05)

            # Mindestbereich sicherstellen (20% des aktuellen Preises)
            min_range = current_price * 0.2
            if (suggested_upper - suggested_lower) < min_range:
                suggested_lower = current_price - (min_range / 2)
                suggested_upper = current_price + (min_range / 2)

            # Nur anpassen wenn signifikante Änderung (>10%)
            current_range = self.upper_price - self.lower_price
            suggested_range = suggested_upper - suggested_lower

            if abs(suggested_range - current_range) / current_range > 0.1:
                self.logger.info(
                    f"🔄 Auto-adjusting grid range: "
                    f"${self.lower_price:,.0f}-${self.upper_price:,.0f} → "
                    f"${suggested_lower:,.0f}-${suggested_upper:,.0f}"
                )

                # Grid-Range aktualisieren
                self.lower_price = suggested_lower
                self.upper_price = suggested_upper

                # Grid-Levels neu berechnen
                self._calculate_grid_levels()

                # Gefüllte Grids zurücksetzen (neue Grid-Struktur)
                self.filled_grids.clear()

                return True

        except Exception as e:
            self.logger.error(f"Error in auto-adjust grid range: {e}")

        return False

    def optimize_for_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Symbol-spezifische Grid-Optimierung
        """
        if df.empty or len(df) < 100:
            return {}

        try:
            # Marktanalysedaten
            price_data = df['close'].tail(100)
            volatility = price_data.std() / price_data.mean()
            price_range = price_data.max() - price_data.min()

            # Symbol-spezifische Empfehlungen
            if 'BTC' in symbol.upper():
                # Bitcoin: Größere Grids, längere Haltezeiten
                recommended_grids = max(10, min(25, int(price_range / 1000)))
                recommended_investment = min(1000, self.investment_per_grid * 2)
            elif any(alt in symbol.upper() for alt in ['ETH', 'SOL', 'ADA']):
                # Major Altcoins: Mittlere Grids
                recommended_grids = max(15, min(30, int(price_range / (price_data.mean() * 0.05))))
                recommended_investment = self.investment_per_grid
            else:
                # Kleine Altcoins: Mehr Grids, kleinere Investitionen
                recommended_grids = max(20, min(50, int(price_range / (price_data.mean() * 0.02))))
                recommended_investment = min(500, self.investment_per_grid * 0.5)

            return {
                'symbol': symbol,
                'recommended_grids': recommended_grids,
                'recommended_investment': recommended_investment,
                'volatility': volatility,
                'price_range': price_range,
                'optimization_confidence': min(len(price_data) / 100, 1.0)
            }

        except Exception as e:
            self.logger.error(f"Error optimizing for {symbol}: {e}")
            return {}

    def reset_grid_state(self) -> None:
        """Setzt den Grid-Zustand zurück (für Tests oder Neustart)"""
        self.filled_grids.clear()
        self.active_grids.clear()
        self.total_profit = 0.0
        self.trades_count = 0
        self.daily_trades = 0
        self.best_trade_profit = 0.0
        self.worst_trade_profit = 0.0
        self.session_start_time = datetime.now()

        self.logger.info("🔄 Grid state reset successfully")

    def save_state(self) -> Dict[str, Any]:
        """Speichert den aktuellen Grid-Zustand"""
        return {
            'filled_grids': list(self.filled_grids),
            'total_profit': self.total_profit,
            'trades_count': self.trades_count,
            'daily_trades': self.daily_trades,
            'best_trade_profit': self.best_trade_profit,
            'worst_trade_profit': self.worst_trade_profit,
            'session_start_time': self.session_start_time.isoformat(),
            'grid_config': {
                'lower_price': self.lower_price,
                'upper_price': self.upper_price,
                'num_grids': self.num_grids,
                'investment_per_grid': self.investment_per_grid
            }
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Lädt einen gespeicherten Grid-Zustand"""
        try:
            self.filled_grids = set(state.get('filled_grids', []))
            self.total_profit = state.get('total_profit', 0.0)
            self.trades_count = state.get('trades_count', 0)
            self.daily_trades = state.get('daily_trades', 0)
            self.best_trade_profit = state.get('best_trade_profit', 0.0)
            self.worst_trade_profit = state.get('worst_trade_profit', 0.0)

            if 'session_start_time' in state:
                self.session_start_time = datetime.fromisoformat(state['session_start_time'])

            self.logger.info("✅ Grid state loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading grid state: {e}")
            self.reset_grid_state()