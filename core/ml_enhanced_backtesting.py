#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verbesserte ML-Integration für den Enhanced Backtester.
Diese Datei erweitert den Backtester um ML-Funktionalitäten.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Pfade für Importe konfigurieren
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Backtest-Module importieren
from core.enhanced_backtesting import EnhancedBacktester
from config.settings import Settings
from strategies.strategy_base import Strategy
from core.position import Position

# ML-Module importieren (sicherstellen, dass diese Pfade korrekt sind)
try:
    from ml_components.market_regime import MarketRegimeDetector
    from ml_components.asset_clusters import AssetClusterAnalyzer
    from ml_components.model_monitor import ModelPerformanceMonitor

    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML-Komponenten konnten nicht importiert werden: {e}")
    ML_AVAILABLE = False


class MLEnhancedBacktester(EnhancedBacktester):
    """Erweiterte Backtesting-Engine mit ML-Komponenten-Integration"""

    def __init__(self, settings: Settings, strategy: Strategy):
        """
        Initialisiert den ML-erweiterten Backtester.

        Args:
            settings: Bot-Konfiguration
            strategy: Zu testende Strategie
        """
        super().__init__(settings, strategy)
        self.logger = logging.getLogger(__name__)

        # ML-Komponenten initialisieren
        self.ml_enabled = settings.get('ml.enabled', False) and ML_AVAILABLE
        self.regime_detector = None
        self.asset_analyzer = None
        self.model_monitor = None

        if self.ml_enabled:
            self._initialize_ml_components()

        # Tracking von ML-basiertem Performance-Vergleich
        self.ml_based_signals = {}
        self.baseline_performance = None
        self.ml_performance = None

    def _initialize_ml_components(self):
        """Initialisiert alle ML-Komponenten basierend auf den Einstellungen."""
        try:
            # Pfade konfigurieren
            data_dir = self.settings.get('ml.data_dir', 'data/market_data')
            models_dir = self.settings.get('ml.models_dir', 'data/ml_models')
            output_dir = self.settings.get('ml.output_dir', 'data/ml_analysis')

            # Verzeichnisse sicherstellen
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Market Regime Detector initialisieren
            n_regimes = self.settings.get('ml.n_regimes', 5)
            self.regime_detector = MarketRegimeDetector(data_dir=data_dir, n_regimes=n_regimes)

            # Versuchen, ein vorhandenes Modell zu laden
            regime_model_path = os.path.join(models_dir, "regime_model.pkl")
            if os.path.exists(regime_model_path):
                self.regime_detector.load_model(regime_model_path)
                self.logger.info(f"Regime-Modell aus {regime_model_path} geladen")

            # Asset Cluster Analyzer initialisieren
            self.asset_analyzer = AssetClusterAnalyzer(data_dir=data_dir)

            # Model Performance Monitor initialisieren
            monitor_dir = os.path.join(output_dir, "model_monitor")
            self.model_monitor = ModelPerformanceMonitor(output_dir=monitor_dir)

            self.logger.info("ML-Komponenten erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der ML-Komponenten: {e}")
            self.ml_enabled = False

    def train_ml_models(self, symbols: List[str]):
        """
        Trainiert die ML-Modelle mit den Backtesting-Daten.

        Args:
            symbols: Liste von Trading-Symbolen für das Training

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.ml_enabled:
            self.logger.warning("ML-Funktionen sind deaktiviert oder nicht verfügbar")
            return False

        try:
            # Marktdaten für das Training laden
            self.logger.info(f"Lade Marktdaten für das Training von {len(symbols)} Symbolen")
            if self.regime_detector.load_market_data(
                    symbols=symbols,
                    data_manager=self.data_manager,
                    timeframe=self.settings.get('timeframes.analysis', '1d'),
                    start_date=self.start_date_str,
                    end_date=self.end_date_str
            ):
                # Features extrahieren
                features_df = self.regime_detector.extract_market_features()

                if not features_df.empty:
                    # Regime-Modell trainieren
                    self.logger.info("Trainiere Marktregime-Modell")
                    if self.regime_detector.train_regime_model(features_df):
                        # Modell speichern
                        models_dir = self.settings.get('ml.models_dir', 'data/ml_models')
                        os.makedirs(models_dir, exist_ok=True)
                        model_path = os.path.join(models_dir, "regime_model.pkl")
                        self.regime_detector.save_model(model_path)
                        self.logger.info(f"Regime-Modell gespeichert unter: {model_path}")

                # Asset-Cluster-Analyse durchführen
                self.logger.info("Führe Asset-Cluster-Analyse durch")
                if self.asset_analyzer.load_market_data(
                        symbols=symbols,
                        data_manager=self.data_manager,
                        timeframe=self.settings.get('timeframes.analysis', '1d'),
                        start_date=self.start_date_str,
                        end_date=self.end_date_str
                ):
                    # Korrelationsmatrix berechnen
                    self.asset_analyzer.calculate_correlation_matrix()

                    # Features extrahieren
                    self.asset_analyzer.extract_asset_features()

                    # Clustering durchführen
                    clusters = self.asset_analyzer.run_clustering()

                    if not clusters.empty:
                        self.logger.info(
                            f"Asset-Clustering abgeschlossen: {len(clusters)} Assets in {len(clusters['cluster'].unique())} Clustern")

                return True

            return False
        except Exception as e:
            self.logger.error(f"Fehler beim Training der ML-Modelle: {e}")
            return False

    def run_with_ml(self, symbols: List[str], source: str = 'binance',
                    timeframe: str = '1h', use_cache: bool = True) -> Dict[str, Any]:
        """
        Führt den Backtest mit ML-Unterstützung durch.

        Args:
            symbols: Liste von Handelssymbolen
            source: Datenquelle ('binance', 'coingecko')
            timeframe: Zeitrahmen
            use_cache: Cache verwenden, falls verfügbar

        Returns:
            Dictionary mit Backtesting-Ergebnissen, einschließlich ML-Vergleich
        """
        if not self.ml_enabled:
            self.logger.warning("ML-Funktionen sind deaktiviert. Führe Standard-Backtest durch.")
            return self.run(symbols, source, timeframe, use_cache)

        try:
            # 1. ML-Modelle trainieren, falls noch nicht geschehen
            if not self.regime_detector.model_trained:
                self.train_ml_models(symbols)

            # 2. Standard-Backtest durchführen (für Baseline-Performance)
            self.logger.info("Führe Baseline-Backtest ohne ML-Integration durch")
            baseline_results = self.run(symbols, source, timeframe, use_cache)
            self.baseline_performance = baseline_results

            # 3. ML-gestützten Backtest durchführen
            self.logger.info("Führe Backtest mit ML-Integration durch")

            # Daten laden, falls noch nicht geschehen
            if not self.data:
                self.load_real_data(symbols, source, timeframe, use_cache)

            if not self.data:
                self.logger.error("Keine Daten verfügbar für ML-gestützten Backtest")
                return baseline_results

            # Backtesting-Zustand initialisieren
            balance = self.initial_balance
            positions = {}  # Symbol -> Position
            trades = []
            equity_curve = []  # Balance + offene Positionen
            trade_history = []  # Detaillierte Handelshistorie
            ml_signals = {}  # Symbol -> Liste von ML-Signalen

            # Einheitlicher Zeitindex für alle Symbole erstellen
            all_dates = pd.DatetimeIndex([])
            for df in self.data.values():
                all_dates = all_dates.union(df.index)
            all_dates = all_dates.sort_values()

            # Startdatum für die Strategie (um Lookback-Periode zu berücksichtigen)
            strategy_start = all_dates[100] if len(all_dates) > 100 else all_dates[0]

            # Risikomanagement-Parameter aus Einstellungen laden
            position_size = self.settings.get('risk.position_size', 0.1)
            stop_loss_pct = self.settings.get('risk.stop_loss', 0.03)
            take_profit_pct = self.settings.get('risk.take_profit', 0.06)
            max_positions = self.settings.get('risk.max_open_positions', 5)

            # Trading-Simulation
            for current_date in all_dates:
                # Erst starten, wenn genug Daten vorhanden sind
                if current_date < strategy_start:
                    continue

                # Aktueller Portfoliowert (Bargeld + offene Positionen)
                portfolio_value = balance
                positions_value = 0.0
                current_prices = {}

                # Aktuelle Preise für alle Symbole sammeln
                for symbol in symbols:
                    if symbol in self.data and current_date in self.data[symbol].index:
                        current_prices[symbol] = self.data[symbol].loc[current_date, 'close']

                        # Wert offener Positionen aktualisieren
                        if symbol in positions:
                            position = positions[symbol]
                            position_value = position.amount * current_prices[symbol]
                            positions_value += position_value

                portfolio_value += positions_value

                # Offene Positionen aktualisieren
                for symbol, position in list(positions.items()):
                    if symbol in current_prices:
                        current_price = current_prices[symbol]

                        # Stop-Loss / Take-Profit prüfen
                        if position.update(current_price):
                            # Position wurde automatisch geschlossen
                            exit_value = position.amount * position.exit_price
                            commission = exit_value * self.commission

                            # Balance aktualisieren
                            balance += exit_value - commission

                            # Trade protokollieren
                            trade = position.to_dict()
                            trade['exit_date'] = current_date
                            trade['commission'] = commission
                            trades.append(trade)

                            # Handelshistorie protokollieren
                            trade_history.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'sell',
                                'reason': position.exit_reason,
                                'price': position.exit_price,
                                'amount': position.amount,
                                'value': exit_value,
                                'commission': commission,
                                'profit_loss': position.profit_loss,
                                'profit_loss_pct': position.profit_loss_percent,
                                'ml_enhanced': getattr(position, 'ml_enhanced', False)
                            })

                            # Position entfernen
                            del positions[symbol]
                            self.logger.debug(
                                f"Position closed for {symbol} at {current_date}: "
                                f"P/L: {position.profit_loss_percent:.2f}%"
                            )

                # ML-basierte Marktregime-Analyse durchführen
                current_regime = None
                regime_features = None

                # Alle verfügbaren Symbole zu diesem Zeitpunkt sammeln
                available_data = {}
                for symbol in symbols:
                    if symbol in self.data and current_date in self.data[symbol].index:
                        symbol_data = self.data[symbol].loc[:current_date].copy()
                        available_data[symbol] = symbol_data

                if available_data and self.regime_detector.model_trained:
                    # Market Features erstellen (für BTC und Top-Altcoins)
                    btc_symbol = next((s for s in available_data.keys() if s.startswith('BTC')), None)

                    if btc_symbol and len(available_data) >= 3:
                        # Feature-DataFrame erstellen
                        features_df = pd.DataFrame(index=[current_date])

                        # BTC Features extrahieren
                        btc_df = available_data[btc_symbol]

                        # Einfache Features berechnen
                        try:
                            # BTC Returns
                            btc_df['return'] = btc_df['close'].pct_change()
                            features_df['btc_return'] = btc_df['return'].iloc[-20:].mean()
                            features_df['btc_volatility'] = btc_df['return'].iloc[-20:].std()

                            # BTC Trend
                            if len(btc_df) >= 50:
                                ema20 = btc_df['close'].ewm(span=20, adjust=False).mean()
                                ema50 = btc_df['close'].ewm(span=50, adjust=False).mean()
                                features_df['btc_ema_ratio'] = ema20.iloc[-1] / ema50.iloc[-1]
                            else:
                                features_df['btc_ema_ratio'] = 1.0

                            # Top-Altcoin Features
                            for symbol, df in available_data.items():
                                if symbol != btc_symbol and 'USDT' in symbol:
                                    # Symbol-Kurzname
                                    short_name = symbol.split('/')[0]

                                    # Returns
                                    df['return'] = df['close'].pct_change()

                                    # Relative Stärke zu BTC
                                    df['rel_to_btc'] = df['return'] - btc_df['return']

                                    features_df[f'rel_strength_{short_name}'] = df['rel_to_btc'].iloc[-20:].mean()

                            # Marktregime vorhersagen
                            if not features_df.empty:
                                regime_features = features_df
                                current_regime = self.regime_detector.predict_regime(features_df)

                                # Regime-Performance aufzeichnen
                                if self.model_monitor and current_regime is not None:
                                    # In der Realität würde das tatsächliche Regime später bestimmt
                                    self.model_monitor.record_prediction(
                                        model_id="backtest_regime",
                                        model_type="regime",
                                        prediction=current_regime,
                                        actual=None  # Wird später aktualisiert
                                    )
                        except Exception as e:
                            self.logger.warning(f"Fehler bei der Regime-Analyse: {e}")

                # Trading-Signale für jedes Symbol generieren
                for symbol in symbols:
                    if symbol not in self.data or current_date not in self.data[symbol].index:
                        continue

                    # Historische Daten bis zum aktuellen Datum
                    symbol_data = self.data[symbol].loc[:current_date].copy()

                    # Aktuelle Position für dieses Symbol
                    current_position = positions.get(symbol)

                    # Standard-Signal generieren
                    standard_signal, standard_signal_data = self.strategy.generate_signal(
                        symbol_data,
                        symbol,
                        current_position
                    )

                    # ML-erweitertes Signal erzeugen
                    ml_signal, ml_signal_data = self._generate_ml_enhanced_signal(
                        standard_signal,
                        standard_signal_data,
                        symbol,
                        symbol_data,
                        current_regime,
                        current_position
                    )

                    # Signal für den aktuellen Backtest verwenden
                    signal = ml_signal
                    signal_data = ml_signal_data

                    # Signal in Historie speichern
                    if symbol not in ml_signals:
                        ml_signals[symbol] = []

                    ml_signals[symbol].append({
                        'date': current_date,
                        'standard_signal': standard_signal,
                        'ml_signal': ml_signal,
                        'confidence': signal_data.get('confidence', 0.5),
                        'regime': current_regime
                    })

                    # Aktuelle Preisdaten
                    current_price = symbol_data.iloc[-1]['close']

                    # Signal verarbeiten
                    if signal == "BUY" and symbol not in positions:
                        # Prüfen, ob genug Bargeld verfügbar ist und ob wir das Maximum an Positionen erreicht haben
                        confidence = signal_data.get('confidence', 0.5)

                        # Bei höherer Konfidenz größere Position
                        adjusted_size = position_size * min(confidence * 1.5, 1.0)

                        # Maximale Anzahl offener Positionen berücksichtigen
                        if len(positions) >= max_positions:
                            self.logger.debug(f"Maximum positions reached, skipping {symbol}")
                            continue

                        # Positionsgröße berechnen
                        trade_value = balance * adjusted_size
                        commission = trade_value * self.commission

                        if trade_value + commission > balance:
                            # Nicht genug Bargeld
                            continue

                        # Anzahl der Coins berechnen
                        amount = trade_value / current_price

                        # Position eröffnen
                        position = Position(
                            symbol=symbol,
                            entry_price=current_price,
                            amount=amount,
                            side='buy',
                            entry_time=current_date
                        )

                        # ML-Attribut hinzufügen
                        position.ml_enhanced = True

                        # Stop-Loss und Take-Profit setzen
                        position.set_stop_loss(percentage=stop_loss_pct)
                        position.set_take_profit(percentage=take_profit_pct)

                        # Trailing-Stop, falls konfiguriert
                        if signal_data.get('use_trailing_stop', False):
                            trailing_stop_pct = signal_data.get('trailing_stop_pct', stop_loss_pct)
                            activation_pct = signal_data.get('trailing_activation_pct', 0.02)
                            position.set_trailing_stop(trailing_stop_pct, activation_pct)

                        # Position zum Portfolio hinzufügen
                        positions[symbol] = position

                        # Bargeld reduzieren
                        balance -= (trade_value + commission)

                        # Handelshistorie protokollieren
                        trade_history.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'buy',
                            'reason': 'ml_signal' if ml_signal != standard_signal else 'signal',
                            'price': current_price,
                            'amount': amount,
                            'value': trade_value,
                            'commission': commission,
                            'profit_loss': 0,
                            'profit_loss_pct': 0,
                            'ml_enhanced': True,
                            'regime': current_regime
                        })

                        self.logger.debug(
                            f"Position opened for {symbol} at {current_date}: "
                            f"{amount:.6f} @ {current_price:.2f} (${trade_value:.2f})"
                        )

                    elif signal == "SELL" and symbol in positions:
                        # Position schließen
                        position = positions[symbol]

                        # Verkaufswert berechnen
                        exit_value = position.amount * current_price
                        commission = exit_value * self.commission

                        # Position schließen
                        position.close_position(current_price,
                                                "ml_sell_signal" if ml_signal != standard_signal else "sell_signal")

                        # Bargeld erhöhen
                        balance += exit_value - commission

                        # Trade protokollieren
                        trade = position.to_dict()
                        trade['exit_date'] = current_date
                        trade['commission'] = commission
                        trade['ml_enhanced'] = True
                        trade['regime'] = current_regime
                        trades.append(trade)

                        # Handelshistorie protokollieren
                        trade_history.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'sell',
                            'reason': 'ml_signal' if ml_signal != standard_signal else 'signal',
                            'price': current_price,
                            'amount': position.amount,
                            'value': exit_value,
                            'commission': commission,
                            'profit_loss': position.profit_loss,
                            'profit_loss_pct': position.profit_loss_percent,
                            'ml_enhanced': True,
                            'regime': current_regime
                        })

                        # Position entfernen
                        del positions[symbol]

                        self.logger.debug(
                            f"Position closed for {symbol} at {current_date}: "
                            f"P/L: {position.profit_loss_percent:.2f}%"
                        )

                # Equity-Kurve aktualisieren
                equity_curve.append({
                    'date': current_date,
                    'balance': balance,
                    'positions_value': positions_value,
                    'portfolio_value': portfolio_value,
                    'open_positions': len(positions),
                    'regime': current_regime
                })

            # Am Ende alle offenen Positionen schließen
            for symbol, position in list(positions.items()):
                if symbol in self.data and current_date in self.data[symbol].index:
                    # Letzten verfügbaren Preis verwenden
                    last_price = self.data[symbol].loc[current_date, 'close']

                    # Position schließen
                    position.close_position(last_price, "end_of_test")

                    # Verkaufswert berechnen
                    exit_value = position.amount * last_price
                    commission = exit_value * self.commission

                    # Bargeld erhöhen
                    balance += exit_value - commission

                    # Trade protokollieren
                    trade = position.to_dict()
                    trade['exit_date'] = current_date
                    trade['commission'] = commission
                    trade['ml_enhanced'] = getattr(position, 'ml_enhanced', False)
                    trades.append(trade)

                    # Handelshistorie protokollieren
                    trade_history.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'sell',
                        'reason': 'end_of_test',
                        'price': last_price,
                        'amount': position.amount,
                        'value': exit_value,
                        'commission': commission,
                        'profit_loss': position.profit_loss,
                        'profit_loss_pct': position.profit_loss_percent,
                        'ml_enhanced': getattr(position, 'ml_enhanced', False)
                    })

                    self.logger.debug(
                        f"Position closed at end of test for {symbol}: "
                        f"P/L: {position.profit_loss_percent:.2f}%"
                    )

            # Ergebnisse zusammenstellen
            equity_df = pd.DataFrame(equity_curve)
            if not equity_df.empty:
                equity_df.set_index('date', inplace=True)

            # Handelshistorie in DataFrame umwandeln
            trades_df = pd.DataFrame(trade_history)
            if not trades_df.empty and 'date' in trades_df.columns:
                trades_df.set_index('date', inplace=True)

            # Handelsstatistiken berechnen
            stats = self._calculate_statistics(trades, equity_df, self.initial_balance)

            # Erweiterte Statistiken berechnen
            extended_stats = self._calculate_extended_statistics(trades_df, equity_df)
            stats.update(extended_stats)

            # ML-Signal-Historie in DataFrame umwandeln
            ml_signals_flat = []
            for symbol, signals in ml_signals.items():
                for signal in signals:
                    signal['symbol'] = symbol
                    ml_signals_flat.append(signal)

            ml_signals_df = pd.DataFrame(ml_signals_flat)
            if not ml_signals_df.empty and 'date' in ml_signals_df.columns:
                ml_signals_df.set_index('date', inplace=True)

            # Regime-basierte Analyse
            regime_stats = self._analyze_regime_performance(trades_df)

            # ML-Vergleich
            ml_comparison = self._compare_with_baseline(baseline_results, {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_return': (balance / self.initial_balance - 1) * 100,
                'total_trades': len(trades),
                'statistics': stats
            })

            # Ergebnisse speichern
            results = {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_return': (balance / self.initial_balance - 1) * 100,
                'total_trades': len(trades),
                'equity_curve': equity_df,
                'trades': trades,
                'trades_df': trades_df,
                'statistics': stats,
                'ml_signals': ml_signals_df,
                'regime_stats': regime_stats,
                'ml_comparison': ml_comparison
            }

            self.ml_performance = results

            # Zusammenfassung loggen
            self.logger.info(f"ML-gestützter Backtest abgeschlossen. Total return: {results['total_return']:.2f}%")
            self.logger.info(f"Total trades: {results['total_trades']}")
            self.logger.info(f"Win rate: {stats['win_rate']:.2f}%")

            # ML-Vergleich loggen
            if ml_comparison:
                improvement = ml_comparison.get('return_improvement', 0)
                self.logger.info(f"ML-Verbesserung: {improvement:.2f}% gegenüber Baseline")

            return results

        except Exception as e:
            self.logger.error(f"Fehler beim ML-gestützten Backtest: {e}")
            return self.baseline_performance

    def _generate_ml_enhanced_signal(self, standard_signal: str, standard_signal_data: Dict[str, Any],
                                     symbol: str, symbol_data: pd.DataFrame,
                                     current_regime: Optional[int] = None,
                                     current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert ein durch ML verbessertes Handelssignal.

        Args:
            standard_signal: Signal der Standardstrategie
            standard_signal_data: Signaldaten der Standardstrategie
            symbol: Handelssymbol
            symbol_data: DataFrame mit historischen Daten
            current_regime: Aktuelles Marktregime (oder None)
            current_position: Aktuelle Position (oder None)

        Returns:
            Tuple aus (Signal, Signal-Daten-Dictionary)
        """
        if not self.ml_enabled or current_regime is None:
            return standard_signal, standard_signal_data

        try:
            # Signaldaten kopieren, um sie zu erweitern
            ml_signal_data = standard_signal_data.copy()
            ml_signal = standard_signal

            # 1. Regime-basierte Anpassungen
            if current_regime is not None and self.regime_detector.model_trained:
                # Trading-Regeln für das aktuelle Regime abrufen
                regime_info = self.regime_detector.get_current_regime_info()

                if regime_info and 'regime_id' in regime_info:
                    regime_id = regime_info['regime_id']
                    regime_label = regime_info.get('label', f"Regime {regime_id}")

                    # Regime-Empfehlungen anwenden
                    if 'strategy' in regime_info:
                        strategy = regime_info['strategy'].lower()

                        # Signal basierend auf Regime-Strategie anpassen
                        if 'bullish' in regime_label.lower() or 'aufwärtstrend' in regime_label.lower():
                            # In bullishem Regime: Buy-Signale verstärken, Sell-Signale abschwächen
                            if standard_signal == "BUY":
                                ml_signal_data['confidence'] = min(1.0,
                                                                   standard_signal_data.get('confidence', 0.5) * 1.2)
                            elif standard_signal == "SELL" and current_position:
                                # Verkauf nur, wenn deutlich unter Stop Loss
                                current_price = symbol_data.iloc[-1]['close']
                                if current_position.stop_loss and current_price < current_position.stop_loss * 0.98:
                                    ml_signal = "SELL"
                                else:
                                    ml_signal = "HOLD"

                        elif 'bearish' in regime_label.lower() or 'abwärtstrend' in regime_label.lower():
                            # In bearishem Regime: Sell-Signale verstärken, Buy-Signale abschwächen
                            if standard_signal == "SELL":
                                ml_signal = "SELL"
                            elif standard_signal == "BUY":
                                # Buy nur mit höherer Konfidenz
                                confidence = standard_signal_data.get('confidence', 0.5)
                                if confidence > 0.7:
                                    ml_signal = "BUY"
                                    # Aber Stop-Loss enger setzen
                                    ml_signal_data['stop_loss_pct'] = standard_signal_data.get('stop_loss_pct',
                                                                                               0.03) * 0.8
                                else:
                                    ml_signal = "HOLD"

                        elif 'neutral' in regime_label.lower():
                            # In neutralem Regime: Standard-Signale verwenden
                            ml_signal = standard_signal

                        elif 'volatility' in regime_label.lower() or 'volatilität' in regime_label.lower():
                            # In volatilen Phasen: Engere Stops, vorsichtigere Entries
                            ml_signal = standard_signal
                            if standard_signal == "BUY":
                                # Enger Stop-Loss in volatilen Phasen
                                ml_signal_data['stop_loss_pct'] = standard_signal_data.get('stop_loss_pct', 0.03) * 0.7
                                ml_signal_data['trailing_stop_pct'] = standard_signal_data.get('trailing_stop_pct',
                                                                                               0.02) * 0.7
                                ml_signal_data['use_trailing_stop'] = True

                    # Coin-spezifische Anpassungen basierend auf Regime-Performance
                    if 'top_performers' in regime_info:
                        top_performers = regime_info['top_performers']

                        # Kurznamen extrahieren
                        short_name = symbol.split('/')[0]

                        # Prüfen, ob dieser Coin ein Top-Performer im aktuellen Regime ist
                        if short_name in top_performers:
                            # Für Top-Performer aggressivere Positionsgröße
                            ml_signal_data['confidence'] = min(1.0, standard_signal_data.get('confidence', 0.5) * 1.3)

                            # Für Top-Performer weiter Stop Loss
                            if ml_signal == "BUY":
                                ml_signal_data['stop_loss_pct'] = standard_signal_data.get('stop_loss_pct', 0.03) * 1.2

                # Regime-Information zu den Signaldaten hinzufügen
                ml_signal_data['regime'] = current_regime
                ml_signal_data['regime_label'] = regime_label

            # 2. Asset-Cluster-basierte Anpassungen
            if self.asset_analyzer and hasattr(self.asset_analyzer,
                                               'clusters') and self.asset_analyzer.clusters is not None:
                clusters = self.asset_analyzer.clusters

                # Kurznamen extrahieren
                short_name = symbol.split('/')[0]

                # Cluster für dieses Asset finden
                if short_name in clusters.index:
                    cluster = clusters.loc[short_name, 'cluster']

                    # Cluster-Performance abrufen
                    if hasattr(self.asset_analyzer, 'cluster_performances'):
                        performances = self.asset_analyzer.cluster_performances

                        if cluster in performances:
                            cluster_perf = performances[cluster]

                            # Für positiv performende Cluster: Buy-Signale verstärken
                            if cluster_perf.get('mean_return', 0) > 0 and cluster_perf.get('sharpe_ratio', 0) > 0.5:
                                if standard_signal == "BUY":
                                    ml_signal_data['confidence'] = min(1.0, standard_signal_data.get('confidence',
                                                                                                     0.5) * 1.2)
                                elif standard_signal == "SELL" and cluster_perf.get('sharpe_ratio', 0) > 1.0:
                                    # Für sehr gute Cluster: Verkauf hinauszögern
                                    ml_signal = "HOLD"

                            # Für schlecht performende Cluster: Vorsichtiger sein
                            elif cluster_perf.get('mean_return', 0) < 0:
                                if standard_signal == "BUY":
                                    # Nur kaufen bei sehr hoher Konfidenz
                                    confidence = standard_signal_data.get('confidence', 0.5)
                                    if confidence < 0.8:
                                        ml_signal = "HOLD"
                                elif standard_signal == "SELL":
                                    # Verkaufssignale verstärken
                                    ml_signal = "SELL"

                    # Cluster-Information zu den Signaldaten hinzufügen
                    ml_signal_data['cluster'] = cluster

            # ML-Flag setzen
            ml_signal_data['ml_enhanced'] = True

            return ml_signal, ml_signal_data

        except Exception as e:
            self.logger.warning(f"Fehler bei der ML-Signal-Generierung: {e}")
            return standard_signal, standard_signal_data

    def _analyze_regime_performance(self, trades_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Analysiert die Performance je nach Marktregime.

        Args:
            trades_df: DataFrame mit Handelshistorie

        Returns:
            Dictionary mit Performance-Statistiken pro Regime
        """
        if trades_df.empty or 'regime' not in trades_df.columns:
            return {}

        try:
            # Analyse pro Regime
            regime_stats = {}

            # Alle Regimes in den Trades
            regimes = trades_df['regime'].dropna().unique()

            for regime in regimes:
                # Trades für dieses Regime filtern
                regime_trades = trades_df[trades_df['regime'] == regime]

                if len(regime_trades) < 2:
                    continue

                # Gewinn- und Verlust-Trades trennen
                profit_trades = regime_trades[regime_trades['profit_loss_pct'] > 0]
                loss_trades = regime_trades[regime_trades['profit_loss_pct'] <= 0]

                # Statistiken berechnen
                total_trades = len(regime_trades)
                win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0

                avg_profit = profit_trades['profit_loss_pct'].mean() if not profit_trades.empty else 0
                avg_loss = loss_trades['profit_loss_pct'].mean() if not loss_trades.empty else 0

                # Profitable Symbole identifizieren
                symbols_performance = regime_trades.groupby('symbol')['profit_loss_pct'].mean().sort_values(
                    ascending=False)
                top_symbols = symbols_performance.head(3).to_dict()

                # Regime-Label abrufen
                regime_label = ""
                if self.regime_detector and self.regime_detector.model_trained:
                    regime_label = self.regime_detector.regime_labels.get(regime, f"Regime {regime}")

                # Statistiken speichern
                regime_stats[regime] = {
                    'regime_label': regime_label,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'top_symbols': top_symbols
                }

            return regime_stats

        except Exception as e:
            self.logger.error(f"Fehler bei der Regime-Performance-Analyse: {e}")
            return {}

    def _compare_with_baseline(self, baseline_results: Dict[str, Any], ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vergleicht ML-gestützte Ergebnisse mit Baseline-Ergebnissen.

        Args:
            baseline_results: Ergebnisse des Standard-Backtests
            ml_results: Ergebnisse des ML-gestützten Backtests

        Returns:
            Dictionary mit Vergleichsdaten
        """
        if not baseline_results or not ml_results:
            return {}

        try:
            comparison = {}

            # Renditen vergleichen
            baseline_return = baseline_results.get('total_return', 0)
            ml_return = ml_results.get('total_return', 0)
            return_improvement = ml_return - baseline_return

            # Win-Rates vergleichen
            baseline_win_rate = baseline_results.get('statistics', {}).get('win_rate', 0)
            ml_win_rate = ml_results.get('statistics', {}).get('win_rate', 0)
            win_rate_improvement = ml_win_rate - baseline_win_rate

            # Sharpe Ratio vergleichen
            baseline_sharpe = baseline_results.get('statistics', {}).get('sharpe_ratio', 0)
            ml_sharpe = ml_results.get('statistics', {}).get('sharpe_ratio', 0)
            sharpe_improvement = ml_sharpe - baseline_sharpe

            # Max Drawdown vergleichen
            baseline_drawdown = abs(baseline_results.get('statistics', {}).get('max_drawdown', 0))
            ml_drawdown = abs(ml_results.get('statistics', {}).get('max_drawdown', 0))
            drawdown_improvement = baseline_drawdown - ml_drawdown

            # Vergleich speichern
            comparison = {
                'baseline_return': baseline_return,
                'ml_return': ml_return,
                'return_improvement': return_improvement,
                'baseline_win_rate': baseline_win_rate,
                'ml_win_rate': ml_win_rate,
                'win_rate_improvement': win_rate_improvement,
                'baseline_sharpe': baseline_sharpe,
                'ml_sharpe': ml_sharpe,
                'sharpe_improvement': sharpe_improvement,
                'baseline_drawdown': baseline_drawdown,
                'ml_drawdown': ml_drawdown,
                'drawdown_improvement': drawdown_improvement,
                'improvement_percentage': (ml_return / baseline_return - 1) * 100 if baseline_return > 0 else 0
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Fehler beim Vergleich mit Baseline: {e}")
            return {}

    def plot_ml_comparison(self, output_dir: str = 'data/backtest_results') -> List[str]:
        """
        Erstellt Visualisierungen zum Vergleich von ML-gestützten und Standard-Backtests.

        Args:
            output_dir: Ausgabeverzeichnis für die Grafiken

        Returns:
            Liste der erstellten Grafikdateien
        """
        if not self.baseline_performance or not self.ml_performance:
            self.logger.error("Keine Backtest-Ergebnisse für den Vergleich verfügbar")
            return []

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(output_dir, exist_ok=True)

            plot_files = []

            # 1. Equity-Kurven vergleichen
            baseline_equity = self.baseline_performance.get('equity_curve')
            ml_equity = self.ml_performance.get('equity_curve')

            if baseline_equity is not None and ml_equity is not None:
                plt.figure(figsize=(12, 6))

                # Portfolio-Werte plotten
                baseline_equity['portfolio_value'].plot(label='Baseline Strategy', alpha=0.7)
                ml_equity['portfolio_value'].plot(label='ML-Enhanced Strategy', alpha=0.7)

                plt.title('Comparison of Baseline vs ML-Enhanced Equity Curves')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # X-Achsen-Format anpassen
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)

                # Grafik speichern
                equity_filepath = os.path.join(output_dir, 'ml_comparison_equity.png')
                plt.savefig(equity_filepath, bbox_inches='tight')
                plt.close()

                plot_files.append(equity_filepath)

            # 2. Regime-basierte Performance visualisieren
            regime_stats = self.ml_performance.get('regime_stats', {})

            if regime_stats:
                plt.figure(figsize=(12, 6))

                # Daten für den Plot vorbereiten
                regimes = []
                win_rates = []
                returns = []
                labels = []

                for regime, stats in regime_stats.items():
                    regimes.append(regime)
                    win_rates.append(stats.get('win_rate', 0))

                    # Durchschnittliche Returns berechnen
                    avg_profit = stats.get('avg_profit', 0)
                    avg_loss = abs(stats.get('avg_loss', 0))
                    win_rate = stats.get('win_rate', 0) / 100

                    avg_return = avg_profit * win_rate - avg_loss * (1 - win_rate)
                    returns.append(avg_return * 100)  # In Prozent

                    # Label hinzufügen
                    labels.append(stats.get('regime_label', f"Regime {regime}"))

                # Breite der Balken
                width = 0.35

                # X-Positionen
                x = np.arange(len(regimes))

                # Win Rate und Returns plotten
                ax1 = plt.subplot(111)
                bars1 = ax1.bar(x - width / 2, win_rates, width, label='Win Rate (%)', color='skyblue')
                ax1.set_ylabel('Win Rate (%)')
                ax1.set_ylim(0, 100)

                ax2 = ax1.twinx()
                bars2 = ax2.bar(x + width / 2, returns, width, label='Avg Return (%)', color='salmon')
                ax2.set_ylabel('Average Return (%)')

                # Labels und Titel
                ax1.set_xlabel('Market Regime')
                ax1.set_title('Performance by Market Regime')
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels, rotation=45, ha='right')

                # Legende
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')

                plt.tight_layout()

                # Grafik speichern
                regime_filepath = os.path.join(output_dir, 'ml_regime_performance.png')
                plt.savefig(regime_filepath, bbox_inches='tight')
                plt.close()

                plot_files.append(regime_filepath)

            # 3. ML-Verbesserungsmetrik visualisieren
            comparison = self.ml_performance.get('ml_comparison', {})

            if comparison:
                plt.figure(figsize=(10, 6))

                # Metriken für den Vergleich
                metrics = [
                    ('Return (%)', comparison.get('baseline_return', 0), comparison.get('ml_return', 0)),
                    ('Win Rate (%)', comparison.get('baseline_win_rate', 0), comparison.get('ml_win_rate', 0)),
                    ('Sharpe Ratio', comparison.get('baseline_sharpe', 0), comparison.get('ml_sharpe', 0)),
                    ('Max Drawdown (%)', comparison.get('baseline_drawdown', 0), comparison.get('ml_drawdown', 0))
                ]

                # Daten vorbereiten
                labels = [m[0] for m in metrics]
                baseline_values = [m[1] for m in metrics]
                ml_values = [m[2] for m in metrics]

                x = np.arange(len(labels))
                width = 0.35

                # Balken plotten
                fig, ax = plt.subplots(figsize=(12, 6))
                rects1 = ax.bar(x - width / 2, baseline_values, width, label='Baseline Strategy', color='skyblue')
                rects2 = ax.bar(x + width / 2, ml_values, width, label='ML-Enhanced Strategy', color='salmon')

                # Labels und Titel
                ax.set_ylabel('Value')
                ax.set_title('Performance Metrics Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()

                # Werte über den Balken anzeigen
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')

                autolabel(rects1)
                autolabel(rects2)

                plt.tight_layout()

                # Grafik speichern
                metrics_filepath = os.path.join(output_dir, 'ml_metrics_comparison.png')
                plt.savefig(metrics_filepath, bbox_inches='tight')
                plt.close()

                plot_files.append(metrics_filepath)

            return plot_files

        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der ML-Vergleichsvisualisierungen: {e}")
            return []


if __name__ == "__main__":
    # Beispiel-Ausführung
    from config.settings import Settings
    from strategies.ml_strategy import MLStrategy

    # Logger einrichten
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Einstellungen
    settings = Settings()
    settings.set('ml.enabled', True)
    settings.set('backtest.start_date', '2022-01-01')
    settings.set('backtest.end_date', '2023-12-31')
    settings.set('trading_pairs', ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    # Strategie initialisieren
    strategy = MLStrategy(settings)

    # Backtester initialisieren
    backtester = MLEnhancedBacktester(settings, strategy)

    # ML-Backtest ausführen
    results = backtester.run_with_ml(
        symbols=settings.get('trading_pairs'),
        source='binance',
        timeframe='1d',
        use_cache=True
    )

    # Ergebnisse visualisieren
    backtester.plot_results()
    backtester.plot_ml_comparison()

    # Ergebnisse anzeigen
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['statistics']['win_rate']:.2f}%")

    if 'ml_comparison' in results:
        improvement = results['ml_comparison'].get('return_improvement', 0)
        logger.info(f"ML Improvement: {improvement:.2f}%")