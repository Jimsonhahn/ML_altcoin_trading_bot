#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verbesserte Backtesting-Modul für den Trading Bot.
Ermöglicht das Testen von Strategien auf echten historischen Daten.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import traceback
import seaborn as sns

from config.settings import Settings
from strategies.strategy_base import Strategy
from core.position import Position
from data_sources import DataManager  # Korrigierter Import  # Korrigierter Import!


class EnhancedBacktester:
    """Verbesserte Backtesting-Engine für Trading-Strategien mit echten Daten"""

    def __init__(self, settings: Settings, strategy: Strategy):
        """
        Initialisiert den verbesserten Backtester.

        Args:
            settings: Bot-Konfiguration
            strategy: Zu testende Strategie
        """
        self.settings = settings
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        # Data Manager initialisieren
        self.data_manager = DataManager(settings)

        # Backtesting-Parameter
        self.initial_balance = settings.get('backtest.initial_balance', 10000)
        self.commission = settings.get('backtest.commission', 0.001)  # 0.1%

        # Test-Zeitraum
        self.start_date_str = settings.get('backtest.start_date', '2023-01-01')
        self.end_date_str = settings.get('backtest.end_date', '2023-12-31')

        try:
            self.start_date = datetime.strptime(self.start_date_str, '%Y-%m-%d')
            self.end_date = datetime.strptime(self.end_date_str, '%Y-%m-%d')
        except ValueError:
            # Fallback auf aktuelle Daten
            self.logger.warning("Invalid date format. Using default period.")
            self.start_date = datetime.now() - timedelta(days=365)
            self.end_date = datetime.now()

        # Daten und Ergebnisse
        self.data = {}
        self.results = {}

    def load_real_data(self, symbols: List[str], source: str = 'binance',
                       timeframe: str = '1h', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Lädt echte historische Daten für die angegebenen Symbole.

        Args:
            symbols: Liste von Handelssymbolen
            source: Datenquelle ('binance', 'coingecko')
            timeframe: Zeitrahmen
            use_cache: Cache verwenden, falls verfügbar

        Returns:
            Dictionary mit Symbolen als Schlüssel und DataFrames als Werten
        """
        self.logger.info(f"Loading historical data for {len(symbols)} symbols from {source}")

        data = {}

        for symbol in symbols:
            try:
                # Daten über den Data Manager abrufen
                df = self.data_manager.get_historical_data(
                    symbol=symbol,
                    source=source,
                    timeframe=timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    use_cache=use_cache
                )

                if not df.empty:
                    data[symbol] = df
                    self.logger.info(f"Loaded {len(df)} data points for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol} in specified time range")

            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                self.logger.debug(traceback.format_exc())

        self.data = data
        return data

    def run(self, symbols: List[str], source: str = 'binance',
            timeframe: str = '1h', use_cache: bool = True) -> Dict[str, Any]:
        """
        Führt den Backtest für die angegebenen Symbole durch.

        Args:
            symbols: Liste von Handelssymbolen
            source: Datenquelle ('binance', 'coingecko')
            timeframe: Zeitrahmen
            use_cache: Cache verwenden, falls verfügbar

        Returns:
            Dictionary mit Backtesting-Ergebnissen
        """
        self.logger.info(f"Starting backtest with {len(symbols)} symbols")
        self.logger.info(f"Time period: {self.start_date_str} to {self.end_date_str}")
        self.logger.info(f"Initial balance: {self.initial_balance:.2f}")
        self.logger.info(f"Commission: {self.commission * 100:.2f}%")
        self.logger.info(f"Data source: {source}, Timeframe: {timeframe}")

        # Daten laden, falls noch nicht geschehen
        if not self.data:
            self.load_real_data(symbols, source, timeframe, use_cache)

        if not self.data:
            self.logger.error("No data available for backtesting")
            return {'error': 'no_data'}

        # Backtesting-Zustand initialisieren
        balance = self.initial_balance
        positions = {}  # Symbol -> Position
        trades = []
        equity_curve = []  # Balance + offene Positionen
        trade_history = []  # Detaillierte Handelshistorie

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
                            'profit_loss_pct': position.profit_loss_percent
                        })

                        # Position entfernen
                        del positions[symbol]
                        self.logger.debug(
                            f"Position closed for {symbol} at {current_date}: "
                            f"P/L: {position.profit_loss_percent:.2f}%"
                        )

            # Trading-Signale für jedes Symbol generieren
            for symbol in symbols:
                if symbol not in self.data or current_date not in self.data[symbol].index:
                    continue

                # Historische Daten bis zum aktuellen Datum
                symbol_data = self.data[symbol].loc[:current_date].copy()

                # Aktuelle Position für dieses Symbol
                current_position = positions.get(symbol)

                # Signal generieren
                signal, signal_data = self.strategy.generate_signal(
                    symbol_data,
                    symbol,
                    current_position
                )

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
                        'reason': 'signal',
                        'price': current_price,
                        'amount': amount,
                        'value': trade_value,
                        'commission': commission,
                        'profit_loss': 0,
                        'profit_loss_pct': 0
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
                    position.close_position(current_price, "sell_signal")

                    # Bargeld erhöhen
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
                        'reason': 'signal',
                        'price': current_price,
                        'amount': position.amount,
                        'value': exit_value,
                        'commission': commission,
                        'profit_loss': position.profit_loss,
                        'profit_loss_pct': position.profit_loss_percent
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
                'open_positions': len(positions)
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
                    'profit_loss_pct': position.profit_loss_percent
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

        # Ergebnisse speichern
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': (balance / self.initial_balance - 1) * 100,
            'total_trades': len(trades),
            'equity_curve': equity_df,
            'trades': trades,
            'trades_df': trades_df,
            'statistics': stats
        }

        self.results = results

        # Zusammenfassung loggen
        self.logger.info(f"Backtest completed. Total return: {results['total_return']:.2f}%")
        self.logger.info(f"Total trades: {results['total_trades']}")
        self.logger.info(f"Win rate: {stats['win_rate']:.2f}%")
        self.logger.info(f"Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"Max drawdown: {abs(stats.get('max_drawdown', 0)):.2f}%")

        return results

    def _calculate_statistics(self, trades: List[Dict], equity_curve: pd.DataFrame,
                              initial_balance: float) -> Dict[str, Any]:
        """
        Berechnet Handelsstatistiken basierend auf abgeschlossenen Trades.

        Args:
            trades: Liste von Trade-Dictionaries
            equity_curve: DataFrame mit Equity-Kurve
            initial_balance: Anfangskapital

        Returns:
            Dictionary mit Statistiken
        """
        if not trades:
            return {
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration': 0
            }

        # Gewinn- und Verlust-Trades trennen
        profit_trades = [t for t in trades if t.get('profit_loss_percent', 0) > 0]
        loss_trades = [t for t in trades if t.get('profit_loss_percent', 0) <= 0]

        # Win-Rate berechnen
        win_rate = len(profit_trades) / len(trades) * 100 if trades else 0

        # Durchschnittlicher Gewinn/Verlust
        avg_profit = np.mean([t.get('profit_loss_percent', 0) for t in profit_trades]) if profit_trades else 0
        avg_loss = np.mean([t.get('profit_loss_percent', 0) for t in loss_trades]) if loss_trades else 0

        # Gesamtgewinn und -verlust
        total_profit = sum([t.get('profit_loss', 0) for t in profit_trades])
        total_loss = sum([t.get('profit_loss', 0) for t in loss_trades])

        # Profit-Faktor
        profit_factor = abs(total_profit / total_loss) if total_loss else float('inf')

        # Maximaler Drawdown berechnen
        max_drawdown = 0
        if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
            portfolio_value = equity_curve['portfolio_value']
            rolling_max = portfolio_value.cummax()
            drawdown = (portfolio_value - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min() if not drawdown.empty else 0

        # Sharpe Ratio berechnen
        sharpe_ratio = 0
        if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
            # Tägliche Returns berechnen
            daily_returns = equity_curve['portfolio_value'].pct_change().dropna()

            if len(daily_returns) > 0:
                # Annualisierter Return und Volatilität
                annual_return = daily_returns.mean() * 252  # Annahme: 252 Handelstage pro Jahr
                annual_volatility = daily_returns.std() * np.sqrt(252)

                # Sharpe Ratio (Risikofreier Zinssatz = 0 für Einfachheit)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Durchschnittliche Trade-Dauer berechnen
        trade_durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']

                # Wenn die Zeiten als Strings vorliegen, konvertieren
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)

                duration = (exit_time - entry_time).total_seconds() / 3600  # in Stunden
                trade_durations.append(duration)

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        return {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': avg_trade_duration,
            'total_profit': total_profit,
            'total_loss': total_loss
        }

    def _calculate_extended_statistics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Berechnet erweiterte Handelsstatistiken.

        Args:
            trades_df: DataFrame mit Handelshistorie
            equity_df: DataFrame mit Equity-Kurve

        Returns:
            Dictionary mit erweiterten Statistiken
        """
        stats = {}

        # Prüfen, ob Daten verfügbar sind
        if trades_df.empty:
            return stats

        # 1. Consecutive Wins/Losses
        if 'profit_loss' in trades_df.columns:
            # Gewinn/Verlust als Serie
            pnl = trades_df['profit_loss'].map(lambda x: 1 if x > 0 else -1).values

            # Consecutive Wins/Losses berechnen
            cons_wins = []
            cons_losses = []
            current_streak = 0
            current_type = 0  # 0 = neutral, 1 = win, -1 = loss

            for result in pnl:
                if result == current_type:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        if current_type == 1:
                            cons_wins.append(current_streak)
                        elif current_type == -1:
                            cons_losses.append(current_streak)

                    current_streak = 1
                    current_type = result

            # Letzten Streak hinzufügen
            if current_streak > 0:
                if current_type == 1:
                    cons_wins.append(current_streak)
                elif current_type == -1:
                    cons_losses.append(current_streak)

            # Statistiken berechnen
            stats['max_consecutive_wins'] = max(cons_wins) if cons_wins else 0
            stats['max_consecutive_losses'] = max(cons_losses) if cons_losses else 0
            stats['avg_consecutive_wins'] = np.mean(cons_wins) if cons_wins else 0
            stats['avg_consecutive_losses'] = np.mean(cons_losses) if cons_losses else 0

        # 2. Trades pro Symbol
        if 'symbol' in trades_df.columns:
            symbol_counts = trades_df['symbol'].value_counts()
            stats['trades_per_symbol'] = symbol_counts.to_dict()

            # Top 3 Symbole nach Gewinn
            if 'profit_loss' in trades_df.columns:
                symbol_pnl = trades_df.groupby('symbol')['profit_loss'].sum()
                top_symbols = symbol_pnl.sort_values(ascending=False).head(3)
                stats['top_symbols'] = top_symbols.to_dict()

        # 3. Kelly-Kriterium berechnen
        if 'profit_loss_pct' in trades_df.columns:
            win_rate = (trades_df['profit_loss_pct'] > 0).mean()
            if win_rate > 0:
                avg_win = trades_df[trades_df['profit_loss_pct'] > 0]['profit_loss_pct'].mean()
                avg_loss = abs(trades_df[trades_df['profit_loss_pct'] <= 0]['profit_loss_pct'].mean())

                if avg_loss > 0:
                    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                    stats['kelly_criterion'] = max(0, kelly)  # Nicht negativ

        # 4. Tägliche/Wöchentliche Returns
        if not equity_df.empty and 'portfolio_value' in equity_df.columns:
            # Tägliche Returns
            daily_returns = equity_df['portfolio_value'].resample('D').last().pct_change().dropna()

            if not daily_returns.empty:
                stats['daily_return_mean'] = daily_returns.mean()
                stats['daily_return_std'] = daily_returns.std()
                stats['daily_return_min'] = daily_returns.min()
                stats['daily_return_max'] = daily_returns.max()
                stats['daily_sharpe'] = daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

                # Wöchentliche Returns
                weekly_returns = equity_df['portfolio_value'].resample('W').last().pct_change().dropna()

                if not weekly_returns.empty:
                    stats['weekly_return_mean'] = weekly_returns.mean()
                    stats['weekly_return_std'] = weekly_returns.std()
                    stats[
                        'weekly_sharpe'] = weekly_returns.mean() / weekly_returns.std() if weekly_returns.std() > 0 else 0

                # Sortino Ratio (nur negative Returns berücksichtigen)
                neg_returns = daily_returns[daily_returns < 0]
                if not neg_returns.empty:
                    downside_deviation = neg_returns.std()
                    stats['sortino_ratio'] = daily_returns.mean() / downside_deviation if downside_deviation > 0 else 0

        # 5. Calmar Ratio (Jahresrendite / Max Drawdown)
        if not equity_df.empty and 'portfolio_value' in equity_df.columns and 'max_drawdown' in stats:
            # Gesamtrendite annualisieren
            days = (equity_df.index.max() - equity_df.index.min()).days
            if days > 0:
                total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1
                annual_return = (1 + total_return) ** (365 / days) - 1

                # Calmar Ratio berechnen
                max_dd = abs(stats['max_drawdown']) / 100  # In Dezimal umwandeln
                stats['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0

        return stats

    def plot_results(self, output_dir: str = 'data/backtest_results') -> List[str]:
        """
        Erstellt Visualisierungen der Backtesting-Ergebnisse.

        Args:
            output_dir: Ausgabeverzeichnis für die Grafiken

        Returns:
            Liste der erstellten Grafikdateien
        """
        if not self.results:
            self.logger.error("No results available for plotting")
            return []

        if 'equity_curve' not in self.results or self.results['equity_curve'].empty:
            self.logger.error("No equity curve in results")
            return []

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(output_dir, exist_ok=True)

            # Ergebnisdaten
            equity_curve = self.results['equity_curve']
            trades_df = self.results.get('trades_df', pd.DataFrame())
            stats = self.results['statistics']

            # Dateipfade für generierte Plots
            plot_files = []

            # 1. Equity-Kurve mit Drawdown
            equity_filepath = self._plot_equity_curve(equity_curve, trades_df, output_dir)
            if equity_filepath:
                plot_files.append(equity_filepath)

            # 2. Trade-Verteilung
            if not trades_df.empty:
                trades_filepath = self._plot_trade_distribution(trades_df, output_dir)
                if trades_filepath:
                    plot_files.append(trades_filepath)

                # 3. Symbol-Performance
                symbol_filepath = self._plot_symbol_performance(trades_df, output_dir)
                if symbol_filepath:
                    plot_files.append(symbol_filepath)

                # 4. Monthly Returns Heatmap
                monthly_filepath = self._plot_monthly_returns(equity_curve, output_dir)
                if monthly_filepath:
                    plot_files.append(monthly_filepath)

            # 5. Performance-Kennzahlen
            metrics_filepath = self._plot_performance_metrics(stats, output_dir)
            if metrics_filepath:
                plot_files.append(metrics_filepath)

            return plot_files

        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _plot_equity_curve(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, output_dir: str) -> Optional[str]:
        """
        Plottet die Equity-Kurve mit Drawdown und Trade-Markierungen.

        Args:
            equity_df: DataFrame mit Equity-Kurve
            trades_df: DataFrame mit Trades
            output_dir: Ausgabeverzeichnis

        Returns:
            Pfad zur erstellten Grafikdatei oder None
        """
        try:
            plt.figure(figsize=(12, 10))

            # Subplot für Equity-Kurve
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

            # Equity-Kurve plotten
            equity_df['portfolio_value'].plot(ax=ax1, color='blue', linewidth=2)
            ax1.set_title('Portfolio Value and Trades')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)

            # Trade-Markierungen hinzufügen
            if not trades_df.empty and 'action' in trades_df.columns:
                # Kauf-Markierungen
                buy_trades = trades_df[trades_df['action'] == 'buy']
                if not buy_trades.empty:
                    ax1.scatter(buy_trades.index, buy_trades['price'],
                                marker='^', color='green', s=100, label='Buy')

                # Verkauf-Markierungen
                sell_trades = trades_df[trades_df['action'] == 'sell']
                if not sell_trades.empty:
                    # Farbe basierend auf Gewinn/Verlust
                    colors = sell_trades['profit_loss'].apply(
                        lambda x: 'blue' if x > 0 else 'red')

                    # Verkaufspunkte plotten
                    for i, (idx, trade) in enumerate(sell_trades.iterrows()):
                        ax1.scatter(idx, trade['price'], marker='v',
                                    color=colors.iloc[i], s=100)

            ax1.legend()

            # Subplot für Drawdown
            ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)

            # Drawdown berechnen und plotten
            portfolio_value = equity_df['portfolio_value']
            rolling_max = portfolio_value.cummax()
            drawdown = (portfolio_value - rolling_max) / rolling_max * 100
            drawdown.plot(ax=ax2, color='red', linewidth=2, label='Drawdown %')

            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            ax2.legend()

            # X-Achsen-Format anpassen
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Grafik speichern
            filepath = os.path.join(output_dir, 'equity_curve_with_trades.png')
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Equity curve plot saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
            return None

    def _plot_trade_distribution(self, trades_df: pd.DataFrame, output_dir: str) -> Optional[str]:
        """
        Plottet die Verteilung der Trade-Ergebnisse.

        Args:
            trades_df: DataFrame mit Trades
            output_dir: Ausgabeverzeichnis

        Returns:
            Pfad zur erstellten Grafikdatei oder None
        """
        try:
            if 'profit_loss_pct' not in trades_df.columns or trades_df.empty:
                return None

            plt.figure(figsize=(12, 10))

            # Subplot 1: Histogramm der Trade-Ergebnisse
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            sns.histplot(trades_df['profit_loss_pct'], kde=True, ax=ax1)
            ax1.set_title('Distribution of Trade Results')
            ax1.set_xlabel('Profit/Loss (%)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(x=0, color='red', linestyle='--')

            # Subplot 2: Cumulative P/L
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            cumulative_pnl = trades_df['profit_loss'].cumsum()
            cumulative_pnl.plot(ax=ax2)
            ax2.set_title('Cumulative Profit/Loss')
            ax2.set_ylabel('Profit/Loss ($)')
            ax2.grid(True)

            # Subplot 3: Gewinn/Verlust je Trade
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            trades_df['profit_loss_pct'].plot(kind='bar', ax=ax3)
            ax3.set_title('Profit/Loss per Trade')
            ax3.set_xlabel('Trade')
            ax3.set_ylabel('Profit/Loss (%)')
            ax3.axhline(y=0, color='red', linestyle='--')

            # X-Achsen-Beschriftungen auf 5% der Trades beschränken
            if len(trades_df) > 20:
                every_nth = max(1, len(trades_df) // 20)
                for idx, label in enumerate(ax3.xaxis.get_ticklabels()):
                    if idx % every_nth != 0:
                        label.set_visible(False)

            plt.tight_layout()

            # Grafik speichern
            filepath = os.path.join(output_dir, 'trade_distribution.png')
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Trade distribution plot saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting trade distribution: {e}")
            return None

    def _plot_symbol_performance(self, trades_df: pd.DataFrame, output_dir: str) -> Optional[str]:
        """
        Plottet die Performance nach Symbol.

        Args:
            trades_df: DataFrame mit Trades
            output_dir: Ausgabeverzeichnis

        Returns:
            Pfad zur erstellten Grafikdatei oder None
        """
        try:
            if 'symbol' not in trades_df.columns or 'profit_loss' not in trades_df.columns or trades_df.empty:
                return None

            plt.figure(figsize=(12, 10))

            # Subplot 1: Gesamtgewinn/-verlust je Symbol
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            symbol_pnl = trades_df.groupby('symbol')['profit_loss'].sum().sort_values()
            symbol_pnl.plot(kind='bar', ax=ax1)
            ax1.set_title('Total Profit/Loss by Symbol')
            ax1.set_xlabel('Symbol')
            ax1.set_ylabel('Profit/Loss ($)')

            # Subplot 2: Anzahl Trades je Symbol
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            trades_df['symbol'].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title('Number of Trades by Symbol')
            ax2.set_xlabel('Symbol')
            ax2.set_ylabel('Number of Trades')

            # Subplot 3: Durchschnittlicher Gewinn/Verlust je Symbol
            ax3 = plt.subplot2grid((2, 2), (1, 0))
            avg_pnl = trades_df.groupby('symbol')['profit_loss_pct'].mean().sort_values()
            avg_pnl.plot(kind='bar', ax=ax3)
            ax3.set_title('Average Profit/Loss by Symbol')
            ax3.set_xlabel('Symbol')
            ax3.set_ylabel('Average Profit/Loss (%)')

            # Subplot 4: Win-Rate je Symbol
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            win_rate = trades_df.groupby('symbol').apply(
                lambda x: (x['profit_loss'] > 0).mean() * 100).sort_values()
            win_rate.plot(kind='bar', ax=ax4)
            ax4.set_title('Win Rate by Symbol')
            ax4.set_xlabel('Symbol')
            ax4.set_ylabel('Win Rate (%)')

            plt.tight_layout()

            # Grafik speichern
            filepath = os.path.join(output_dir, 'symbol_performance.png')
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Symbol performance plot saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting symbol performance: {e}")
            return None

    def _plot_monthly_returns(self, equity_df: pd.DataFrame, output_dir: str) -> Optional[str]:
        """
        Plottet eine Heatmap der monatlichen Returns.

        Args:
            equity_df: DataFrame mit Equity-Kurve
            output_dir: Ausgabeverzeichnis

        Returns:
            Pfad zur erstellten Grafikdatei oder None
        """
        try:
            if 'portfolio_value' not in equity_df.columns or equity_df.empty:
                return None

            # Monatliche Returns berechnen
            monthly_returns = equity_df['portfolio_value'].resample('M').last().pct_change().dropna()

            if len(monthly_returns) < 2:
                return None

            # Returns in eine Jahres/Monats-Matrix umwandeln
            returns_matrix = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })

            # Pivotieren für Heatmap
            pivot_table = returns_matrix.pivot_table(
                index='Year', columns='Month', values='Return')

            # Monatsnamen
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_table.columns = [month_names[i - 1] for i in pivot_table.columns]

            plt.figure(figsize=(12, 8))

            # Heatmap erstellen
            sns.heatmap(pivot_table * 100, annot=True, fmt='.2f', cmap='RdYlGn',
                        linewidths=0.5, center=0, vmin=-10, vmax=10)

            plt.title('Monthly Returns (%)')
            plt.tight_layout()

            # Grafik speichern
            filepath = os.path.join(output_dir, 'monthly_returns_heatmap.png')
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Monthly returns heatmap saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting monthly returns: {e}")
            return None

    def _plot_performance_metrics(self, stats: Dict[str, Any], output_dir: str) -> Optional[str]:
        """
        Plottet die wichtigsten Performance-Kennzahlen.

        Args:
            stats: Dictionary mit Statistiken
            output_dir: Ausgabeverzeichnis

        Returns:
            Pfad zur erstellten Grafikdatei oder None
        """
        try:
            plt.figure(figsize=(12, 8))

            # Wichtige Kennzahlen auswählen
            metrics = {
                'Sharpe Ratio': stats.get('sharpe_ratio', 0),
                'Sortino Ratio': stats.get('sortino_ratio', 0),
                'Calmar Ratio': stats.get('calmar_ratio', 0),
                'Win Rate (%)': stats.get('win_rate', 0),
                'Profit Factor': stats.get('profit_factor', 0),
                'Avg Profit (%)': stats.get('avg_profit', 0),
                'Avg Loss (%)': abs(stats.get('avg_loss', 0)),
                'Max Drawdown (%)': abs(stats.get('max_drawdown', 0))
            }

            # Balkendiagramm erstellen
            bars = plt.bar(range(len(metrics)), list(metrics.values()), color='skyblue')

            # Werte über den Balken anzeigen
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                         f'{list(metrics.values())[i]:.2f}', ha='center', va='bottom')

            # Achsenbeschriftungen
            plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
            plt.title('Performance Metrics')
            plt.tight_layout()

            # Grafik speichern
            filepath = os.path.join(output_dir, 'performance_metrics.png')
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Performance metrics plot saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {e}")
            return None

    def export_results(self, output_dir: str = 'data/backtest_results', format: str = 'excel') -> Dict[str, str]:
        """
        Exportiert die Backtest-Ergebnisse in verschiedenen Formaten.

        Args:
            output_dir: Ausgabeverzeichnis
            format: Exportformat ('csv', 'json', 'excel')

        Returns:
            Dictionary mit exportierten Dateipfaden
        """
        if not self.results:
            self.logger.error("No results available for export")
            return {}

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(output_dir, exist_ok=True)

            # Exportierte Dateien
            exported_files = {}

            # Zeitstempel für Dateinamen
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 1. Equity-Kurve exportieren
            if 'equity_curve' in self.results and not self.results['equity_curve'].empty:
                equity_df = self.results['equity_curve']

                if format == 'csv':
                    filepath = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
                    equity_df.to_csv(filepath)
                elif format == 'json':
                    filepath = os.path.join(output_dir, f'equity_curve_{timestamp}.json')
                    equity_df.to_json(filepath, date_format='iso')
                elif format == 'excel':
                    filepath = os.path.join(output_dir, f'backtest_results_{timestamp}.xlsx')
                    with pd.ExcelWriter(filepath) as writer:
                        equity_df.to_excel(writer, sheet_name='Equity Curve')

                        # Weitere Blätter hinzufügen, falls im Excel-Format
                        if 'trades_df' in self.results and not self.results['trades_df'].empty:
                            self.results['trades_df'].to_excel(writer, sheet_name='Trades')

                        # Statistiken als DataFrame konvertieren und hinzufügen
                        if 'statistics' in self.results:
                            stats_df = pd.DataFrame.from_dict(self.results['statistics'], orient='index',
                                                              columns=['Value'])
                            stats_df.to_excel(writer, sheet_name='Statistics')

                exported_files['equity_curve'] = filepath
                self.logger.info(f"Exported equity curve to {filepath}")

            # 2. Trades exportieren (wenn nicht bereits in Excel exportiert)
            if format != 'excel' and 'trades_df' in self.results and not self.results['trades_df'].empty:
                trades_df = self.results['trades_df']

                if format == 'csv':
                    filepath = os.path.join(output_dir, f'trades_{timestamp}.csv')
                    trades_df.to_csv(filepath)
                elif format == 'json':
                    filepath = os.path.join(output_dir, f'trades_{timestamp}.json')
                    trades_df.to_json(filepath, date_format='iso')

                exported_files['trades'] = filepath
                self.logger.info(f"Exported trades to {filepath}")

            # 3. Statistiken exportieren (wenn nicht bereits in Excel exportiert)
            if format != 'excel' and 'statistics' in self.results:
                stats = self.results['statistics']

                if format == 'csv':
                    filepath = os.path.join(output_dir, f'statistics_{timestamp}.csv')
                    pd.DataFrame.from_dict(stats, orient='index', columns=['Value']).to_csv(filepath)
                elif format == 'json':
                    filepath = os.path.join(output_dir, f'statistics_{timestamp}.json')
                    with open(filepath, 'w') as f:
                        json.dump(stats, f, indent=2)

                exported_files['statistics'] = filepath
                self.logger.info(f"Exported statistics to {filepath}")

            return exported_files

        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return {}


if __name__ == "__main__":
    # Beispiel-Ausführung
    from config.settings import Settings
    from strategies.momentum import MomentumStrategy

    # Logger einrichten
    logging.basicConfig(level=logging.INFO)

    # Einstellungen und Strategie initialisieren
    settings = Settings()
    strategy = MomentumStrategy(settings)

    # Backtest-Zeitraum definieren
    settings.set('backtest.start_date', '2023-01-01')
    settings.set('backtest.end_date', '2023-12-31')
    settings.set('backtest.initial_balance', 10000)

    # Backtester initialisieren
    backtester = EnhancedBacktester(settings, strategy)

    # Backtest ausführen
    results = backtester.run(
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT'],
        source='binance',
        timeframe='1d',
        use_cache=True
    )

    # Ergebnisse visualisieren
    backtester.plot_results(output_dir="../data/data/visualizations")

    # Ergebnisse exportieren
    backtester.export_results(output_dir="../data/data/backtest_results", format='excel')