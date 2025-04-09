#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting-Modul für den Trading Bot.
Ermöglicht das Testen von Strategien auf historischen Daten.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import os

from config.settings import Settings
from strategies.strategy_base import Strategy
from core.position import Position


class Backtester:
    """Backtesting-Engine für Trading-Strategien"""

    def __init__(self, settings: Settings, strategy: Strategy):
        """
        Initialisiert den Backtester.

        Args:
            settings: Bot-Konfiguration
            strategy: Zu testende Strategie
        """
        self.settings = settings
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

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

    def load_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Lädt historische Daten für die angegebenen Symbole.

        In einer realen Implementierung würden diese Daten von einer API
        oder einer Datenbank geladen. Für dieses Beispiel verwenden wir
        simulierte Daten.

        Args:
            symbols: Liste von Handelssymbolen

        Returns:
            Dictionary mit Symbolen als Schlüssel und DataFrames als Werte
        """
        self.logger.info(f"Loading historical data for {len(symbols)} symbols")

        data = {}

        for symbol in symbols:
            try:
                # Hier würde man normalerweise Daten laden
                # In diesem Beispiel generieren wir simulierte Daten
                df = self._generate_test_data(symbol)

                # Daten filtern nach Zeitraum
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

                if not df.empty:
                    data[symbol] = df
                    self.logger.info(f"Loaded {len(df)} data points for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol} in specified time range")

            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")

        self.data = data
        return data

    def _generate_test_data(self, symbol: str) -> pd.DataFrame:
        """
        Generiert simulierte Testdaten für ein Symbol.

        In einer realen Implementierung würde diese Methode nicht existieren
        und stattdessen echte Daten geladen werden.

        Args:
            symbol: Handelssymbol

        Returns:
            DataFrame mit OHLCV-Daten
        """
        # Zeitraum generieren
        start = self.start_date - timedelta(days=30)  # Extra-Daten für Indikatoren
        end = self.end_date

        dates = pd.date_range(start=start, end=end, freq='1H')

        # Basispreis und Volatilität je nach Symbol
        base_price = 100
        if 'BTC' in symbol:
            base_price = 30000
            volatility = 0.02
        elif 'ETH' in symbol:
            base_price = 2000
            volatility = 0.025
        elif 'SOL' in symbol:
            base_price = 100
            volatility = 0.03
        else:
            base_price = 50
            volatility = 0.02

        # Preisbewegung simulieren
        np.random.seed(42)  # Für Reproduzierbarkeit

        # Zufällige Preisbewegungen mit Trend
        returns = np.random.normal(0.0001, volatility, size=len(dates))

        # Kleine Trends hinzufügen
        for i in range(1, len(returns)):
            if i % 100 == 0:
                trend = np.random.choice([-0.001, 0.001]) * 20
                for j in range(i, min(i + 100, len(returns))):
                    returns[j] += trend

        price = base_price * (1 + np.cumsum(returns))

        # OHLCV-Daten generieren
        df = pd.DataFrame(index=dates)
        df['close'] = price
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = price[0] * 0.999

        # High und Low relativ zu Open/Close
        daily_volatility = df['close'].pct_change().std()
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, size=len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, size=len(df)))

        # Volumen
        vol_base = base_price * 1000
        df['volume'] = vol_base * (1 + np.random.uniform(-0.5, 1.0, size=len(df)))

        # Erhöhtes Volumen bei starken Preisbewegungen
        volatility = df['close'].pct_change().abs()
        df['volume'] *= (1 + volatility * 10)

        # NaN-Werte entfernen
        df = df.dropna()

        return df

    def run(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Führt den Backtest für die angegebenen Symbole durch.

        Args:
            symbols: Liste von Handelssymbolen

        Returns:
            Dictionary mit Backtesting-Ergebnissen
        """
        self.logger.info(f"Starting backtest with {len(symbols)} symbols")
        self.logger.info(f"Time period: {self.start_date_str} to {self.end_date_str}")
        self.logger.info(f"Initial balance: {self.initial_balance:.2f}")
        self.logger.info(f"Commission: {self.commission * 100:.2f}%")

        # Daten laden, falls noch nicht geschehen
        if not self.data:
            self.load_data(symbols)

        if not self.data:
            self.logger.error("No data available for backtesting")
            return {'error': 'no_data'}

        # Backtesting-Zustand initialisieren
        balance = self.initial_balance
        positions = {}  # Symbol -> Position
        trades = []
        equity_curve = []  # Balance + offene Positionen

        # Einheitlicher Zeitindex für alle Symbole
        all_dates = pd.DatetimeIndex([])
        for df in self.data.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()

        # Startdatum für die Strategie (um Lookback-Periode zu berücksichtigen)
        strategy_start = all_dates[100]  # Mindestens 100 Datenpunkte für Indikatoren

        # Trading-Simulation
        for current_date in all_dates:
            # Erst starten, wenn genug Daten vorhanden sind
            if current_date < strategy_start:
                continue

            # Aktueller Portfoliowert (Bargeld + offene Positionen)
            portfolio_value = balance

            # Offene Positionen aktualisieren
            for symbol, position in list(positions.items()):
                # Prüfen, ob aktuelle Daten für das Symbol vorhanden sind
                if symbol in self.data and current_date in self.data[symbol].index:
                    current_price = self.data[symbol].loc[current_date, 'close']

                    # Position-Value aktualisieren
                    position_value = position.amount * current_price
                    portfolio_value += position_value

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
                symbol_data = self.data[symbol].loc[:current_date]

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
                    # Prüfen, ob genug Bargeld verfügbar ist
                    confidence = signal_data.get('confidence', 0.5)

                    # Positionsgröße basierend auf Konfidenz und Risikomanagement
                    position_size = self.settings.get('risk.position_size', 0.1)

                    # Bei höherer Konfidenz größere Position
                    adjusted_size = position_size * min(confidence * 1.5, 1.0)

                    # Maximale Anzahl offener Positionen berücksichtigen
                    max_positions = self.settings.get('risk.max_open_positions', 5)
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
                    stop_loss_pct = self.settings.get('risk.stop_loss', 0.03)
                    take_profit_pct = self.settings.get('risk.take_profit', 0.06)

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
                'portfolio_value': portfolio_value,
                'open_positions': len(positions)
            })

        # Am Ende alle offenen Positionen schließen
        for symbol, position in list(positions.items()):
            if symbol in self.data:
                # Letzten verfügbaren Preis verwenden
                last_price = self.data[symbol]['close'].iloc[-1]

                # Position schließen
                position.close_position(last_price, "end_of_test")

                # Verkaufswert berechnen
                exit_value = position.amount * last_price
                commission = exit_value * self.commission

                # Bargeld erhöhen
                balance += exit_value - commission

                # Trade protokollieren
                trade = position.to_dict()
                trade['exit_date'] = all_dates[-1]
                trade['commission'] = commission
                trades.append(trade)

                self.logger.debug(
                    f"Position closed at end of test for {symbol}: "
                    f"P/L: {position.profit_loss_percent:.2f}%"
                )

        # Ergebnisse zusammenstellen
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)

        # Handelsstatistiken berechnen
        stats = self._calculate_statistics(trades, equity_df, self.initial_balance)

        # Ergebnisse speichern
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': (balance / self.initial_balance - 1) * 100,
            'total_trades': len(trades),
            'equity_curve': equity_df,
            'trades': trades,
            'statistics': stats
        }

        self.results = results

        # Zusammenfassung loggen
        self.logger.info(f"Backtest completed. Total return: {results['total_return']:.2f}%")
        self.logger.info(f"Total trades: {results['total_trades']}")
        self.logger.info(f"Win rate: {stats['win_rate']:.2f}%")

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

    def plot_results(self, output_dir: str = 'results') -> Optional[str]:
        """
        Erstellt Visualisierungen der Backtesting-Ergebnisse.

        Args:
            output_dir: Ausgabeverzeichnis für die Grafiken

        Returns:
            Pfad zur erstellten Grafik oder None im Fehlerfall
        """
        if not self.results or 'equity_curve' not in self.results or self.results['equity_curve'].empty:
            self.logger.error("No results available for plotting")
            return None

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(output_dir, exist_ok=True)

            # Ergebnisdaten
            equity_curve = self.results['equity_curve']
            trades = self.results['trades']
            stats = self.results['statistics']

            # Mehrere Subplots erstellen
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

            # Plot 1: Equity-Kurve
            equity_curve['portfolio_value'].plot(ax=axs[0], label='Portfolio Value')

            # Buy/Sell-Punkte markieren
            for trade in trades:
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')

                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)

                # Wenn der Zeitpunkt im Index der Equity-Kurve ist
                if entry_time in equity_curve.index:
                    entry_value = equity_curve.loc[entry_time, 'portfolio_value']
                    axs[0].scatter(entry_time, entry_value, color='green', marker='^', s=80)

                if exit_time in equity_curve.index:
                    exit_value = equity_curve.loc[exit_time, 'portfolio_value']
                    profit_pct = trade.get('profit_loss_percent', 0)
                    color = 'red' if profit_pct < 0 else 'blue'
                    axs[0].scatter(exit_time, exit_value, color=color, marker='v', s=80)

            axs[0].set_title('Portfolio Value Over Time')
            axs[0].set_ylabel('Value ($)')
            axs[0].grid(True)
            axs[0].legend()

            # Plot 2: Drawdown
            if 'portfolio_value' in equity_curve.columns:
                portfolio_value = equity_curve['portfolio_value']
                rolling_max = portfolio_value.cummax()
                drawdown = (portfolio_value - rolling_max) / rolling_max * 100
                drawdown.plot(ax=axs[1], color='red', label='Drawdown %')
                axs[1].set_title('Drawdown Over Time')
                axs[1].set_ylabel('Drawdown (%)')
                axs[1].grid(True)
                axs[1].legend()

            # Plot 3: Offene Positionen
            if 'open_positions' in equity_curve.columns:
                equity_curve['open_positions'].plot(ax=axs[2], label='Open Positions', color='purple')
                axs[2].set_title('Number of Open Positions')
                axs[2].set_ylabel('Positions')
                axs[2].grid(True)
                axs[2].legend()

            # Statistiken als Text hinzufügen
            textstr = '\n'.join((
                f"Total Return: {self.results['total_return']:.2f}%",
                f"Total Trades: {self.results['total_trades']}",
                f"Win Rate: {stats['win_rate']:.2f}%",
                f"Avg Profit: {stats['avg_profit']:.2f}%",
                f"Avg Loss: {stats['avg_loss']:.2f}%",
                f"Profit Factor: {stats['profit_factor']:.2f}",
                f"Max Drawdown: {abs(stats['max_drawdown']):.2f}%",
                f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}"
            ))

            fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Layout anpassen
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # Grafik speichern
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"backtest_results_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()

            self.logger.info(f"Results plot saved to {filepath}")

            return filepath

        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            return None