#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance-Datenquelle für den Trading Bot.
Ermöglicht den Zugriff auf historische Daten von Binance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

import ccxt

from data_sources.data_sources.base import DataSourceBase, DataSourceException


class BinanceDataSource(DataSourceBase):
    """Implementierung für Binance als Datenquelle"""

    def __init__(self, settings: Dict = None):
        """
        Initialisiert die Binance-Datenquelle.

        Args:
            settings: Konfigurationseinstellungen
        """
        super().__init__(settings)
        self.name = "binance"
        self.rate_limit_delay = 0.5  # Binance erlaubt 1200 Anfragen pro Minute (60/1200 = 0.05, mit Puffer)
        self.logger = logging.getLogger(__name__)

        # CCXT-Exchange-Instanz erstellen
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def _throttle(self):
        """
        Simple rate limiting mechanism to avoid hitting Binance API limits.
        """
        import time
        time.sleep(self.rate_limit_delay)

    def get_historical_data(self, symbol: str, timeframe: str = "1h",
                            start_date=None, end_date=None,
                            use_cache: bool = True):
        """
        Implementiert die Schnittstelle von DataSourceBase für historische Daten.
        Versucht zuerst echte Daten abzurufen, fällt bei Fehlern auf Demo-Daten zurück.

        Args:
            symbol: Symbol wie 'BTC/USDT'
            timeframe: Zeitintervall ('1h', '1d', etc.)
            start_date: Startdatum
            end_date: Enddatum
            use_cache: Cache verwenden

        Returns:
            DataFrame mit OHLCV-Daten
        """
        use_demo_data = self.settings.get('data.use_demo', False) if self.settings else False

        # Versuche echte Daten zu laden, es sei denn, Demo-Daten wurden explizit angefordert
        if not use_demo_data:
            try:
                self.logger.info(f"Versuche echte Daten für {symbol} abzurufen")
                df = self.get_historical_prices(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                self.logger.info(f"Erfolgreich echte Daten für {symbol} abgerufen: {len(df)} Datenpunkte")
                return df
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen echter Daten für {symbol}: {e}. Verwende Demo-Daten.")
                # Fallback auf Demo-Daten

        # Demo-Daten generieren
        self.logger.info(f"Generiere Demo-Daten für {symbol} ({timeframe})")
        return self._generate_demo_data(symbol, timeframe, start_date, end_date)

    def _generate_demo_data(self, symbol: str, timeframe: str = "1h",
                            start_date=None, end_date=None):
        """
        Generiert Beispieldaten für historische Kursdaten.

        Args:
            symbol: Symbol wie 'BTC/USDT'
            timeframe: Zeitintervall ('1h', '1d', etc.)
            start_date: Startdatum
            end_date: Enddatum

        Returns:
            DataFrame mit OHLCV-Daten
        """
        # Standard-Zeitraum, falls nicht angegeben
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # Frequenz basierend auf Timeframe
        if timeframe == "1h":
            freq = "1H"
        elif timeframe == "1d":
            freq = "1D"
        elif timeframe == "15m":
            freq = "15min"
        else:
            # Standardwert
            freq = "1H"

        # Zeitindex erstellen
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Basispreis je nach Symbol
        if "BTC" in symbol:
            base_price = 30000
        elif "ETH" in symbol:
            base_price = 2000
        elif "SOL" in symbol:
            base_price = 70
        elif "ADA" in symbol:
            base_price = 0.5
        elif "XRP" in symbol:
            base_price = 0.6
        else:
            base_price = 100

        # Preisreihe generieren (zufälliger Trend mit etwas Volatilität)
        np.random.seed(hash(symbol) % 10000)  # Seed für Konsistenz

        # Basistrend
        trend = np.cumsum(np.random.normal(0.0002, 0.02, len(dates)))

        # Preisreihe
        close_prices = base_price * np.exp(trend)

        # OHLCV-Daten generieren
        data = pd.DataFrame({
            'open': close_prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'high': close_prices * (1 + np.random.uniform(0.005, 0.02, len(dates))),
            'low': close_prices * (1 - np.random.uniform(0.005, 0.02, len(dates))),
            'close': close_prices,
            'volume': np.random.normal(base_price * 100, base_price * 20, len(dates)) * (
                        1 + 0.5 * np.random.random(len(dates)))
        }, index=dates)

        # Korrigieren: high muss höher als open, close sein; low niedriger
        for i in range(len(data)):
            max_val = max(data.iloc[i]['open'], data.iloc[i]['close'])
            min_val = min(data.iloc[i]['open'], data.iloc[i]['close'])

            if data.iloc[i]['high'] < max_val:
                data.iloc[i, data.columns.get_loc('high')] = max_val * 1.005

            if data.iloc[i]['low'] > min_val:
                data.iloc[i, data.columns.get_loc('low')] = min_val * 0.995

        self.logger.info(f"Generierte {len(data)} Demo-Datenpunkte für {symbol}")
        return data

    def get_historical_prices(self, symbol: str, timeframe: str = "1h",
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        Ruft historische Preisdaten von Binance ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '1h', '1d', etc.)
            start_date: Startdatum
            end_date: Enddatum
            limit: Maximale Anzahl der Datenpunkte pro Anfrage

        Returns:
            DataFrame mit OHLCV-Daten
        """
        try:
            all_data = []

            # End-Datum standardmäßig auf jetzt setzen
            if end_date is None:
                end_date = datetime.now()

            # Start-Datum auf 1000 Kerzen vor End-Datum setzen, falls nicht angegeben
            if start_date is None:
                # Ungefähre Zeitspanne basierend auf Timeframe berechnen
                if timeframe == '1m':
                    start_date = end_date - timedelta(minutes=limit)
                elif timeframe == '5m':
                    start_date = end_date - timedelta(minutes=5 * limit)
                elif timeframe == '15m':
                    start_date = end_date - timedelta(minutes=15 * limit)
                elif timeframe == '1h':
                    start_date = end_date - timedelta(hours=limit)
                elif timeframe == '4h':
                    start_date = end_date - timedelta(hours=4 * limit)
                elif timeframe == '1d':
                    start_date = end_date - timedelta(days=limit)
                else:
                    start_date = end_date - timedelta(days=30)  # Default

            # Unix-Zeitstempel in Millisekunden
            end_ts = int(end_date.timestamp() * 1000)
            current_end = end_ts

            # Wiederholte Anfragen, um alle Daten im Bereich zu erhalten
            while True:
                self._throttle()

                self.logger.debug(
                    f"Fetching {symbol} {timeframe} data until {datetime.fromtimestamp(current_end / 1000)}")

                # OHLCV-Daten abrufen
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=None,  # Wir geben das "since" nicht an, da wir rückwärts arbeiten
                    limit=limit,
                    params={'endTime': current_end}
                )

                if not ohlcv or len(ohlcv) == 0:
                    break

                # Daten hinzufügen
                all_data = ohlcv + all_data

                # Ältesten Zeitstempel als neues Ende setzen
                current_end = ohlcv[0][0] - 1

                # Abbrechen, wenn wir das Start-Datum erreicht haben
                start_ts = int(start_date.timestamp() * 1000)
                if current_end <= start_ts:
                    break

            # DataFrame erstellen
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Nach Timestamp filtern
            start_ts = int(start_date.timestamp() * 1000)
            df = df[df['timestamp'] >= start_ts]

            # Timestamp in Datetime konvertieren
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Sortieren
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data from Binance for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch historical data: {e}")

    def get_trading_pairs(self, quote_currency: str = "USDT") -> List[str]:
        """
        Ruft alle verfügbaren Handelspaare für eine Quote-Währung ab.

        Args:
            quote_currency: Quote-Währung (z.B. 'USDT')

        Returns:
            Liste von Handelspaaren
        """
        self._throttle()

        try:
            # Märkte abrufen
            markets = self.exchange.fetch_markets()

            # Nach Quote-Währung filtern
            pairs = [market['symbol'] for market in markets if market['quote'] == quote_currency]

            return pairs

        except Exception as e:
            self.logger.error(f"Error fetching trading pairs: {e}")
            raise DataSourceException(f"Could not fetch trading pairs: {e}")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft den aktuellen Ticker für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')

        Returns:
            Ticker-Informationen
        """
        self._throttle()

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch ticker: {e}")

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Ruft das Orderbuch für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            limit: Anzahl der Einträge im Orderbuch

        Returns:
            Orderbuch-Informationen
        """
        self._throttle()

        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch order book: {e}")

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Ruft die letzten Trades für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            limit: Anzahl der abzurufenden Trades

        Returns:
            Liste von Trade-Informationen
        """
        self._throttle()

        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"Error fetching recent trades for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch recent trades: {e}")