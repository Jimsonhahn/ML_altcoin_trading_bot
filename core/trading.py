#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Haupttrading-Logik für den Trading Bot.
Koordiniert die Interaktion zwischen Exchange, Strategien, und Position Management.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

from config.settings import Settings
from core.exchange import ExchangeBase, ExchangeFactory
from core.position import Position, PositionManager
from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from utils.logger import setup_logger
import pandas as pd


class TradingBot:
    """Hauptklasse für den Trading Bot"""

    def __init__(self, mode: str = "paper", strategy_name: str = "default", settings: Optional[Settings] = None):
        """
        Initialisiert den Trading Bot.

        Args:
            mode: Trading-Modus ('live', 'paper', 'backtest')
            strategy_name: Name der zu verwendenden Strategie
            settings: Bot-Konfiguration (optional, sonst Standardkonfiguration)
        """
        # Einstellungen initialisieren
        self.settings = settings or Settings()

        # Logger einrichten
        log_level = getattr(logging, self.settings.get('logging.level', 'INFO'))
        self.logger = setup_logger(log_level)

        # Trading-Parameter
        self.mode = mode
        self.strategy_name = strategy_name
        self.running = False
        self.check_interval = self.settings.get('timeframes.check_interval', 300)  # Sekunden

        # Exchange initialisieren
        self.exchange = ExchangeFactory.create(self.settings, mode)

        # Position Manager initialisieren
        self.position_manager = PositionManager()

        # Strategie initialisieren
        self.strategy = self._initialize_strategy(strategy_name)

        # Event Callbacks
        self.on_trade_callbacks = []

        # Performance-Tracking
        self.start_time = None
        self.start_balance = 0.0

        # Trading-Paare
        self.trading_pairs = self.settings.get('trading_pairs', ["BTC/USDT"])

        # Daten-Cache
        self.data_cache = {}  # Speichert OHLCV-Daten für jedes Symbol

        self.logger.info(f"Trading bot initialized. Mode: {mode}, Strategy: {strategy_name}")

    def _initialize_strategy(self, strategy_name: str) -> Strategy:
        """
        Initialisiert die Trading-Strategie.

        Args:
            strategy_name: Name der Strategie

        Returns:
            Strategy-Objekt
        """
        # Standardstrategie ist Momentum
        if strategy_name == "default" or strategy_name == "momentum":
            return MomentumStrategy(self.settings)
        elif strategy_name == "mean_reversion":
            return MeanReversionStrategy(self.settings)
        elif strategy_name == "ml":
            return MLStrategy(self.settings)
        else:
            self.logger.warning(f"Unknown strategy '{strategy_name}', using default")
            return MomentumStrategy(self.settings)

    def connect(self) -> bool:
        """
        Stellt eine Verbindung zum Exchange her.

        Returns:
            True bei erfolgreicher Verbindung, False sonst
        """
        try:
            success = self.exchange.connect()
            if success:
                self.logger.info(f"Connected to exchange in {self.mode} mode")
            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            return False

    def add_trade_callback(self, callback: Callable[[Position], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei jedem Trade aufgerufen wird.

        Args:
            callback: Callback-Funktion, die eine Position als Parameter erhält
        """
        self.on_trade_callbacks.append(callback)

    def _notify_trade(self, position: Position) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über einen Trade.

        Args:
            position: Position-Objekt des Trades
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")

    def _update_data_cache(self, symbol: str) -> pd.DataFrame:
        """
        Aktualisiert den Daten-Cache für ein Symbol.

        Args:
            symbol: Handelssymbol

        Returns:
            DataFrame mit OHLCV-Daten
        """
        timeframe = self.settings.get('timeframes.analysis', '1h')

        try:
            # Daten abrufen
            df = self.exchange.get_ohlcv(symbol, timeframe)

            # Im Cache speichern
            self.data_cache[symbol] = df

            return df
        except Exception as e:
            self.logger.error(f"Error updating data cache for {symbol}: {e}")

            # Vorhandene Daten zurückgeben, falls verfügbar
            if symbol in self.data_cache:
                return self.data_cache[symbol]

            # Leerer DataFrame, falls keine Daten verfügbar
            return pd.DataFrame()

    def _check_pair(self, symbol: str) -> None:
        """
        Überprüft ein Trading-Paar auf Handelssignale.

        Args:
            symbol: Handelssymbol
        """
        try:
            # Daten aktualisieren
            df = self._update_data_cache(symbol)

            if df.empty:
                self.logger.warning(f"No data available for {symbol}, skipping")
                return

            # Aktuelle Position für dieses Symbol abrufen
            current_position = self.position_manager.get_position_by_symbol(symbol)

            # Strategie anwenden, um Signal zu generieren
            signal, signal_data = self.strategy.generate_signal(df, symbol, current_position)

            # Aktuellen Preis abrufen
            current_price = df.iloc[-1]['close']

            # Signal loggen
            signal_str = signal_data.get('signal', 'UNKNOWN')
            confidence = signal_data.get('confidence', 0.0)

            self.logger.info(
                f"{symbol} @ {current_price}: Signal={signal_str}, "
                f"Confidence={confidence:.2f}, Position={'OPEN' if current_position else 'NONE'}"
            )

            # Signal verarbeiten
            self._process_signal(symbol, current_price, signal_data, current_position)

        except Exception as e:
            self.logger.error(f"Error checking {symbol}: {e}")

    def _process_signal(self, symbol: str, current_price: float, signal_data: Dict[str, Any],
                        current_position: Optional[Position]) -> None:
        """
        Verarbeitet ein Handelssignal.

        Args:
            symbol: Handelssymbol
            current_price: Aktueller Preis
            signal_data: Signaldaten
            current_position: Aktuelle Position (oder None)
        """
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0.0)

        # Risikomanagement-Parameter
        position_size = self.settings.get('risk.position_size', 0.05)
        stop_loss_pct = self.settings.get('risk.stop_loss', 0.03)
        take_profit_pct = self.settings.get('risk.take_profit', 0.06)
        max_positions = self.settings.get('risk.max_open_positions', 5)

        # 1. Kaufsignal
        if signal == 'BUY' and not current_position:
            # Minimale Konfidenz prüfen
            if confidence < 0.6:
                self.logger.info(f"Buy signal for {symbol} ignored due to low confidence: {confidence:.2f}")
                return

            # Maximale Anzahl offener Positionen prüfen
            if len(self.position_manager.get_all_positions()) >= max_positions:
                self.logger.info(f"Buy signal for {symbol} ignored due to max positions limit")
                return

            # Kapital für den Trade berechnen
            balance = self.exchange.get_balance()
            if balance <= 0:
                self.logger.warning(f"Insufficient balance for {symbol}")
                return

            trade_value = balance * position_size
            trade_amount = trade_value / current_price

            try:
                # Order platzieren
                order = self.exchange.place_order(
                    symbol=symbol,
                    order_type='market',
                    side='buy',
                    amount=trade_amount
                )

                # Position erstellen
                position = Position(
                    symbol=symbol,
                    entry_price=current_price,
                    amount=trade_amount,
                    side='buy',
                    order_id=order.get('id'),
                    entry_time=datetime.now()
                )

                # Stop-Loss und Take-Profit setzen
                position.set_stop_loss(percentage=stop_loss_pct)
                position.set_take_profit(percentage=take_profit_pct)

                # Trailing-Stop setzen, falls konfiguriert
                if signal_data.get('use_trailing_stop', False):
                    trailing_stop_pct = signal_data.get('trailing_stop_pct', stop_loss_pct)
                    activation_pct = signal_data.get('trailing_activation_pct', 0.02)
                    position.set_trailing_stop(trailing_stop_pct, activation_pct)

                # Position zum Manager hinzufügen
                self.position_manager.add_position(position)

                # Callback benachrichtigen
                self._notify_trade(position)

                self.logger.info(
                    f"Opened position for {symbol}: {trade_amount} @ {current_price}. "
                    f"Stop-loss: {position.stop_loss}, Take-profit: {position.take_profit}"
                )

            except Exception as e:
                self.logger.error(f"Failed to place buy order for {symbol}: {e}")

        # 2. Verkaufssignal
        elif signal == 'SELL' and current_position and current_position.side == 'buy':
            try:
                # Order platzieren
                order = self.exchange.place_order(
                    symbol=symbol,
                    order_type='market',
                    side='sell',
                    amount=current_position.amount
                )

                # Position schließen
                closed_position = self.position_manager.close_position(
                    current_position.id,
                    current_price,
                    "sell_signal"
                )

                if closed_position:
                    # Callback benachrichtigen
                    self._notify_trade(closed_position)

                    self.logger.info(
                        f"Closed position for {symbol} at {current_price}. "
                        f"P/L: {closed_position.profit_loss_percent:.2f}%"
                    )

            except Exception as e:
                self.logger.error(f"Failed to place sell order for {symbol}: {e}")

        # 3. Offene Positionen aktualisieren
        elif current_position:
            # Aktuelle Preise für alle offenen Positionen
            current_prices = {symbol: current_price}

            # Positionen aktualisieren
            closed_positions = self.position_manager.update_positions(current_prices)

            # Geschlossene Positionen verarbeiten
            for position in closed_positions:
                try:
                    # Verkaufsorder platzieren
                    order = self.exchange.place_order(
                        symbol=position.symbol,
                        order_type='market',
                        side='sell',
                        amount=position.amount
                    )

                    # Callback benachrichtigen
                    self._notify_trade(position)

                    self.logger.info(
                        f"Position closed automatically for {position.symbol} at {position.exit_price} "
                        f"({position.exit_reason}). P/L: {position.profit_loss_percent:.2f}%"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to place sell order for closed position {position.symbol}: {e}")

    def run(self) -> None:
        """
        Startet den Trading Bot im Live- oder Paper-Trading-Modus.
        """
        if not self.connect():
            self.logger.error("Failed to connect to exchange. Bot stopped.")
            return

        self.running = True
        self.start_time = datetime.now()
        self.start_balance = self.exchange.get_balance()

        self.logger.info(f"Starting trading bot with {len(self.trading_pairs)} pairs")
        self.logger.info(f"Initial balance: {self.start_balance}")

        try:
            while self.running:
                # Alle Trading-Paare überprüfen
                for symbol in self.trading_pairs:
                    if not self.running:
                        break

                    self._check_pair(symbol)

                    # Kurze Pause zwischen den Paaren
                    time.sleep(1)

                # Auf nächstes Intervall warten
                self.logger.debug(f"Sleeping for {self.check_interval} seconds")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()

    def run_in_thread(self) -> threading.Thread:
        """
        Startet den Trading Bot in einem separaten Thread.

        Returns:
            Thread-Objekt
        """
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()

        return thread

    def stop(self) -> None:
        """
        Stoppt den Trading Bot.
        """
        self.running = False

        # Endgütigen Kontostand abrufen
        final_balance = self.exchange.get_balance()

        # Gesamtergebnis anzeigen
        if self.start_time:
            duration = datetime.now() - self.start_time
            hours = duration.total_seconds() / 3600

            profit_loss = final_balance - self.start_balance
            profit_loss_pct = profit_loss / self.start_balance * 100 if self.start_balance > 0 else 0

            self.logger.info("-" * 50)
            self.logger.info(f"Bot stopped after {hours:.2f} hours")
            self.logger.info(f"Final balance: {final_balance:.2f} USDT")
            self.logger.info(f"P/L: {profit_loss:.2f} USDT ({profit_loss_pct:.2f}%)")

            # Positions-Statistiken
            stats = self.position_manager.get_position_stats()

            self.logger.info(f"Total trades: {stats['closed_positions']}")
            self.logger.info(f"Win rate: {stats['win_rate']:.2f}%")
            self.logger.info(f"Avg profit: {stats['avg_profit_pct']:.2f}%")
            self.logger.info(f"Avg loss: {stats['avg_loss_pct']:.2f}%")
            self.logger.info("-" * 50)

    def get_status(self) -> Dict[str, Any]:
        """
        Ruft den aktuellen Status des Bots ab.

        Returns:
            Dictionary mit Statusinformationen
        """
        # Aktueller Kontostand
        current_balance = self.exchange.get_balance()

        # Performance berechnen
        profit_loss = 0
        profit_loss_pct = 0
        duration = 0

        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds() / 3600
            profit_loss = current_balance - self.start_balance
            profit_loss_pct = profit_loss / self.start_balance * 100 if self.start_balance > 0 else 0

        # Positions-Statistiken
        stats = self.position_manager.get_position_stats()

        # Aktuell offene Positionen
        open_positions = []
        for position in self.position_manager.get_all_positions():
            # Aktuelle unrealisierte P/L berechnen
            current_price = 0
            unrealized_pnl = 0
            unrealized_pnl_pct = 0

            if position.symbol in self.data_cache and not self.data_cache[position.symbol].empty:
                current_price = self.data_cache[position.symbol].iloc[-1]['close']

                if position.side == 'buy':
                    unrealized_pnl = (current_price - position.entry_price) * position.amount
                    unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.amount
                    unrealized_pnl_pct = (position.entry_price - current_price) / position.entry_price * 100

            open_positions.append({
                'id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'amount': position.amount,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time.isoformat(),
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            })

        return {
            'mode': self.mode,
            'strategy': self.strategy_name,
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_hours': duration,
            'start_balance': self.start_balance,
            'current_balance': current_balance,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'pairs': self.trading_pairs,
            'open_positions': open_positions,
            'position_stats': stats
        }

    def run_backtest(self) -> Dict[str, Any]:
        """
        Führt einen Backtest mit historischen Daten durch.

        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        from core.backtesting import Backtester

        backtester = Backtester(self.settings, self.strategy)
        results = backtester.run(self.trading_pairs)

        return results