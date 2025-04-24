#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hauptmodul für die Trading-Logik des Altcoin Trading Bots.
Dieses Modul enthält die erweiterte TradingBot-Klasse und die
zugehörigen Hilfsfunktionen für das Trading-System.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from config.settings import Settings
from core.exchange import ExchangeFactory
from core.position import Position, PositionManager
from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from utils.logger import setup_logger
from data_sources import DataManager  # Korrigierter Import


class TradingBot:
    """
    Hauptklasse für den Trading Bot.

    Diese Klasse koordiniert alle Trading-Aktivitäten und ist die
    zentrale Schnittstelle für die Ausführung des Bots.
    """

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
        self.on_error_callbacks = []
        self.on_status_update_callbacks = []

        # Performance-Tracking
        self.start_time = None
        self.start_balance = 0.0

        # Trading-Paare
        self.trading_pairs = self.settings.get('trading_pairs', ["BTC/USDT"])

        # Daten-Cache
        self.data_cache = {}  # Speichert OHLCV-Daten für jedes Symbol
        self.data_cache_lock = threading.Lock()  # Thread-Sicherheit für Cache-Zugriffe

        # DataManager für erweiterte Datenquellen initialisieren
        self.data_manager = DataManager(self.settings)

        # Trading-Threads und Status
        self.trading_thread = None
        self.monitor_thread = None
        self.status_update_interval = self.settings.get('system.status_update_interval', 60)  # Sekunden

        # Thread-Pool für parallele Datenverarbeitung
        self.max_workers = self.settings.get('system.max_workers', min(32, (os.cpu_count() or 4) + 4))

        # Letzter Status für Benachrichtigungen
        self.last_status = {}

        # API Fehler-Tracking
        self.api_error_count = 0
        self.last_api_error_time = None
        self.max_api_errors = self.settings.get('system.max_api_errors', 5)
        self.api_error_window = self.settings.get('system.api_error_window', 300)  # Sekunden

        # Initialen Status melden
        self.logger.info(f"Trading bot initialized. Mode: {mode}, Strategy: {strategy_name}")
        self.logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")

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

                # Überprüfe API-Limits und Kontostatus
                if self.mode in ['live', 'paper']:
                    account_info = self.exchange.get_account_info()
                    if account_info:
                        self.logger.info(f"Account status: {account_info.get('status', 'Unknown')}")

                        # Limits abrufen
                        limits = account_info.get('limits', {})
                        if limits:
                            self.logger.info(f"API rate limits: {limits}")

            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            self._notify_error("connection_error", f"Failed to connect to exchange: {e}")
            return False

    def add_trade_callback(self, callback: Callable[[Position], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei jedem Trade aufgerufen wird.

        Args:
            callback: Callback-Funktion, die eine Position als Parameter erhält
        """
        self.on_trade_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei Fehlern aufgerufen wird.

        Args:
            callback: Callback-Funktion, die Fehlertyp und -nachricht erhält
        """
        self.on_error_callbacks.append(callback)

    def add_status_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei Statusaktualisierungen aufgerufen wird.

        Args:
            callback: Callback-Funktion, die den Status-Dictionary erhält
        """
        self.on_status_update_callbacks.append(callback)

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

    def _notify_error(self, error_type: str, error_message: str) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über einen Fehler.

        Args:
            error_type: Typ des Fehlers
            error_message: Fehlermeldung
        """
        # API-Fehler zählen, um wiederholte Probleme zu erkennen
        if error_type.startswith("api_") or error_type == "connection_error":
            current_time = time.time()

            # Reset-Fenster für API-Fehler
            if (self.last_api_error_time is None or
                    current_time - self.last_api_error_time > self.api_error_window):
                self.api_error_count = 1
            else:
                self.api_error_count += 1

            self.last_api_error_time = current_time

            # Sicherheitsabschaltung bei zu vielen API-Fehlern
            if self.api_error_count >= self.max_api_errors:
                self.logger.critical(
                    f"Too many API errors ({self.api_error_count}) within {self.api_error_window} seconds. "
                    f"Shutting down for safety."
                )
                self.stop()

                # Kritischer Fehler an alle Callback-Handler
                error_message = f"Bot stopped due to excessive API errors: {error_message}"
                error_type = "critical_api_failure"

        for callback in self.on_error_callbacks:
            try:
                callback(error_type, error_message)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")

    def _notify_status_update(self, status: Dict[str, Any]) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über eine Statusaktualisierung.

        Args:
            status: Status-Dictionary
        """
        # Nur signifikante Änderungen melden
        if self._is_significant_status_change(self.last_status, status):
            for callback in self.on_status_update_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"Error in status update callback: {e}")

            # Status speichern
            self.last_status = status.copy()

    def _is_significant_status_change(self, old_status: Dict[str, Any], new_status: Dict[str, Any]) -> bool:
        """
        Überprüft, ob sich der Status signifikant geändert hat.

        Args:
            old_status: Vorheriger Status
            new_status: Neuer Status

        Returns:
            True, wenn sich der Status signifikant geändert hat
        """
        if not old_status:
            return True

        # Änderungen, die immer als signifikant gelten
        if old_status.get('running') != new_status.get('running'):
            return True

        # Neue offene Positionen
        old_positions = {p['id']: p for p in old_status.get('open_positions', [])}
        new_positions = {p['id']: p for p in new_status.get('open_positions', [])}

        if set(old_positions.keys()) != set(new_positions.keys()):
            return True

        # Signifikante P/L-Änderung (mehr als 1%)
        old_pnl = old_status.get('profit_loss_pct', 0)
        new_pnl = new_status.get('profit_loss_pct', 0)

        if abs(new_pnl - old_pnl) > 1.0:
            return True

        # Positionen mit signifikanten P/L-Änderungen
        for pos_id, new_pos in new_positions.items():
            if pos_id in old_positions:
                old_pos = old_positions[pos_id]
                old_pnl = old_pos.get('unrealized_pnl_pct', 0)
                new_pnl = new_pos.get('unrealized_pnl_pct', 0)

                if abs(new_pnl - old_pnl) > 2.0:  # Mehr als 2% Änderung
                    return True

        # Standard-Update-Intervall (ca. alle 5 Minuten)
        old_timestamp = datetime.fromisoformat(old_status.get('timestamp', '2000-01-01T00:00:00'))
        new_timestamp = datetime.fromisoformat(new_status.get('timestamp', '2000-01-01T00:00:00'))

        if (new_timestamp - old_timestamp).total_seconds() > 300:
            return True

        return False

    def _update_data_cache(self, symbol: str) -> pd.DataFrame:
        """
        Aktualisiert den Daten-Cache für ein Symbol.

        Args:
            symbol: Handelssymbol

        Returns:
            DataFrame mit OHLCV-Daten
        """
        timeframe = self.settings.get('timeframes.analysis', '1h')
        # Optionale Einstellung für die Datenquelle
        data_source = self.settings.get('data.source', 'exchange')

        try:
            # Daten abrufen über den geeigneten Weg
            if data_source == 'exchange' or self.mode == 'live' or self.mode == 'paper':
                # Direkt vom Exchange für Live/Paper-Trading abrufen
                df = self.exchange.get_ohlcv(symbol, timeframe)
            else:
                # Alternativ den DataManager für historische oder alternative Daten verwenden
                source = self.settings.get('data.source_name', 'binance')
                df = self.data_manager.get_historical_data(
                    symbol=symbol,
                    source=source,
                    timeframe=timeframe,
                    start_date=datetime.now() - timedelta(days=30),  # Letzte 30 Tage
                    use_cache=True
                )

            # Sicherstellen, dass wir ausreichend Daten haben
            if len(df) < self.settings.get('data.min_candles', 50):
                self.logger.warning(
                    f"Insufficient data for {symbol}, got {len(df)} candles, "
                    f"need at least {self.settings.get('data.min_candles', 50)}"
                )

            # Daten synchron für Thread-Sicherheit im Cache speichern
            with self.data_cache_lock:
                self.data_cache[symbol] = df

            return df
        except Exception as e:
            error_msg = f"Error updating data cache for {symbol}: {e}"
            self.logger.error(error_msg)
            self._notify_error("data_error", error_msg)

            # Vorhandene Daten zurückgeben, falls verfügbar
            with self.data_cache_lock:
                if symbol in self.data_cache:
                    return self.data_cache[symbol]

            # Leerer DataFrame, falls keine Daten verfügbar
            return pd.DataFrame()

    def _update_all_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """
        Aktualisiert den Daten-Cache für alle Symbole parallel.

        Returns:
            Dictionary mit Symbol als Schlüssel und DataFrame als Wert
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Alle Symbole parallel verarbeiten
            futures = {executor.submit(self._update_data_cache, symbol): symbol
                       for symbol in self.trading_pairs}

            # Ergebnisse einsammeln
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error updating data for {symbol}: {e}")

        return results

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
            error_msg = f"Error checking {symbol}: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self._notify_error("signal_error", error_msg)

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
        min_confidence = self.settings.get('risk.min_confidence', 0.6)

        # Eigene Risikomanagement-Logik
        if self.settings.get('risk.dynamic_position_sizing', False):
            # Dynamisches Position Sizing basierend auf Volatilität
            if 'volatility' in signal_data:
                volatility = signal_data['volatility']
                # Bei höherer Volatilität kleinere Position
                position_size = position_size * (1.0 - min(volatility * 2.0, 0.8))

        # 1. Kaufsignal
        if signal == 'BUY' and not current_position:
            # Minimale Konfidenz prüfen
            if confidence < min_confidence:
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

            # Mindestorder-Größe prüfen
            if self.mode in ['live', 'paper']:
                min_order_size = self.exchange.get_min_order_size(symbol)
                if min_order_size and trade_amount < min_order_size:
                    self.logger.warning(
                        f"Trade amount {trade_amount} below minimum order size {min_order_size} for {symbol}"
                    )

                    # Entweder auf Mindestgröße erhöhen oder Signal ignorieren
                    if self.settings.get('risk.adjust_to_min_size', True):
                        trade_amount = min_order_size
                        self.logger.info(f"Adjusted trade amount to minimum order size: {min_order_size}")
                    else:
                        self.logger.info(f"Buy signal for {symbol} ignored due to insufficient trade amount")
                        return

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
                error_msg = f"Failed to place buy order for {symbol}: {e}"
                self.logger.error(error_msg)
                self._notify_error("order_error", error_msg)

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
                error_msg = f"Failed to place sell order for {symbol}: {e}"
                self.logger.error(error_msg)
                self._notify_error("order_error", error_msg)

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
                    error_msg = f"Failed to place sell order for closed position {position.symbol}: {e}"
                    self.logger.error(error_msg)
                    self._notify_error("order_error", error_msg)

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
                # Strategie-Thread-Sicherheit prüfen
                if hasattr(self.strategy, 'lock') and callable(self.strategy.lock):
                    # Strategie-Zustand sichern, falls erforderlich
                    if hasattr(self.strategy, 'save_state') and callable(self.strategy.save_state):
                        self.strategy.save_state()

                # Alle Marktdaten parallel aktualisieren (effizienter)
                if self.settings.get('system.parallel_data_updates', True):
                    self._update_all_data_parallel()

                # Alle Trading-Paare überprüfen - optional parallel
                if self.settings.get('system.parallel_signal_processing', True) and len(self.trading_pairs) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Alle Symbole parallel verarbeiten
                        list(executor.map(self._check_pair, self.trading_pairs))
                else:
                    # Sequentiell verarbeiten
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
            error_msg = f"Error in main loop: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self._notify_error("system_error", error_msg)
        finally:
            self.stop()

    def run_in_thread(self) -> threading.Thread:
        """
        Startet den Trading Bot in einem separaten Thread.

        Returns:
            Thread-Objekt
        """
        self.trading_thread = threading.Thread(target=self.run, name="TradingBot-Main")
        self.trading_thread.daemon = True
        self.trading_thread.start()

        # Statusmonitor-Thread starten
        self.monitor_thread = threading.Thread(target=self._status_monitor, name="TradingBot-Monitor")
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        return self.trading_thread

    def _status_monitor(self) -> None:
        """
        Ausführung eines separaten Threads zur Überwachung und Berichterstattung des Bot-Status.
        """
        while self.running:
            try:
                status = self.get_status()
                self._notify_status_update(status)

                # Zustand speichern, falls konfiguriert
                auto_save = self.settings.get('system.auto_save_interval', 0)
                if auto_save > 0:
                    # Speicherpfad generieren
                    current_time = datetime.now()
                    if current_time.minute % auto_save == 0 and current_time.second < 10:
                        save_path = os.path.join(
                            self.settings.get('system.save_directory', 'data/states'),
                            f"bot_state_{current_time.strftime('%Y%m%d_%H%M')}.json"
                        )
                        self.save_state(save_path)

            except Exception as e:
                self.logger.error(f"Error in status monitor: {e}")

            time.sleep(self.status_update_interval)

    def stop(self) -> None:
        """
        Stoppt den Trading Bot.
        """
        if not self.running:
            return  # Bot bereits gestoppt

        self.running = False
        self.logger.info("Stopping bot...")

        # Endgütigen Kontostand abrufen
        try:
            final_balance = self.exchange.get_balance()
        except Exception as e:
            self.logger.error(f"Error getting final balance: {e}")
            final_balance = 0

        # Strategie-Zustand sichern, falls erforderlich
        if hasattr(self.strategy, 'save_state') and callable(self.strategy.save_state):
            try:
                self.strategy.save_state()
                self.logger.info("Strategy state saved")
            except Exception as e:
                self.logger.error(f"Error saving strategy state: {e}")

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

            # Abschließenden Status senden
            final_status = self.get_status()
            final_status['final_result'] = {
                'duration_hours': hours,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'stats': stats
            }
            self._notify_status_update(final_status)

            # Automatischen Trading-Bericht generieren, falls aktiviert
            if self.settings.get('reports.auto_generate_on_stop', False):
                try:
                    report_days = self.settings.get('reports.default_days', 30)
                    report_format = self.settings.get('reports.default_format', 'html')

                    report_path = self.generate_trading_report(
                        days=min(report_days, int(hours / 24) + 1),
                        output_format=report_format
                    )

                    self.logger.info(f"Trading report generated: {report_path}")
                except Exception as e:
                    self.logger.error(f"Failed to generate auto trading report: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Ruft den aktuellen Status des Bots ab.

        Returns:
            Dictionary mit Statusinformationen
        """
        # Aktueller Kontostand
        try:
            current_balance = self.exchange.get_balance()
        except Exception as e:
            self.logger.error(f"Error getting current balance: {e}")
            current_balance = 0

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

            with self.data_cache_lock:
                if position.symbol in self.data_cache and not self.data_cache[position.symbol].empty:
                    current_price = self.data_cache[position.symbol].iloc[-1]['close']

            if current_price > 0:
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
                'take_profit': position.take_profit,
                'trailing_stop': position.trailing_stop,
                'trailing_activation': position.trailing_activation
            })

        # Cache-Statistiken
        cache_stats = {
            'symbols_cached': len(self.data_cache),
            'total_candles': sum(len(df) for df in self.data_cache.values() if isinstance(df, pd.DataFrame))
        }

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
            'position_stats': stats,
            'cache_stats': cache_stats,
            'timestamp': datetime.now().isoformat()
        }

    def save_state(self, filepath: str) -> bool:
        """
        Speichert den Zustand des Bots.

        Args:
            filepath: Pfad zur Ausgabedatei

        Returns:
            True bei erfolgreicher Speicherung, False sonst
        """
        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Status abrufen
            status = self.get_status()

            # Status speichern
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2, default=str)

            self.logger.info(f"Bot state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save bot state: {e}")
            return False

    def run_backtest(self) -> Dict[str, Any]:
        """
        Führt einen Backtest durch und gibt die Ergebnisse zurück.
        Diese Methode ist ein Wrapper für den EnhancedBacktester.

        Returns:
            Ein Dictionary mit den Backtest-Ergebnissen
        """
        from core.enhanced_backtesting import EnhancedBacktester

        self.logger.info("Starte Backtest-Prozess mit EnhancedBacktester...")

        # Überprüfe, ob die notwendigen Parameter in den Einstellungen vorhanden sind
        required_params = ['backtest.start_date', 'backtest.end_date', 'trading_pairs']
        for param in required_params:
            if not self.settings.get(param):
                self.logger.error(f"Parameter '{param}' fehlt in den Einstellungen")
                raise ValueError(f"Parameter '{param}' fehlt in den Einstellungen")

        # Trading-Paare extrahieren
        trading_pairs = self.settings.get('trading_pairs')
        if isinstance(trading_pairs, str):
            trading_pairs = [trading_pairs]

        # Überprüfen, ob trading_pairs eine Liste ist
        if not isinstance(trading_pairs, list):
            self.logger.error("trading_pairs muss eine Liste sein")
            raise ValueError("trading_pairs muss eine Liste sein")

        self.logger.info(f"Trading-Paare: {trading_pairs}")

        # Backtester initialisieren
        backtester = EnhancedBacktester(self.settings, self.strategy)

        # Zusätzliche Parameter extrahieren
        timeframe = self.settings.get('timeframes.analysis', '1h')
        source = self.settings.get('data.source', 'binance')
        use_cache = self.settings.get('data.use_cache', True)

        # Backtest ausführen
        try:
            self.logger.info(f"Führe Backtest aus: timeframe={timeframe}, source={source}, pairs={trading_pairs}")
            results = backtester.run(
                symbols=trading_pairs,
                source=source,
                timeframe=timeframe,
                use_cache=use_cache
            )

            # Ergebnisse visualisieren, falls konfiguriert
            if self.settings.get('backtest.create_plots', True):
                output_dir = os.path.join("../data/data/backtest_results", self.settings.get('backtest.output_dir', 'latest'))
                self.logger.info(f"Erstelle Visualisierungen in: {output_dir}")
                plot_files = backtester.plot_results(output_dir=output_dir)
                results['plot_files'] = plot_files

            # Ergebnisse exportieren, falls konfiguriert
            if self.settings.get('backtest.export_results', True):
                output_dir = os.path.join("../data/data/backtest_results", self.settings.get('backtest.output_dir', 'latest'))
                export_format = self.settings.get('backtest.export_format', 'excel')
                self.logger.info(f"Exportiere Ergebnisse nach: {output_dir} (Format: {export_format})")
                export_files = backtester.export_results(output_dir=output_dir, format=export_format)
                results['export_files'] = export_files

            self.logger.info(f"Backtest abgeschlossen. Total return: {results.get('total_return', 0):.2f}%")
            return results

        except Exception as e:
            self.logger.error(f"Fehler beim Ausführen des Backtests: {e}")
            self.logger.error(traceback.format_exc())
            raise