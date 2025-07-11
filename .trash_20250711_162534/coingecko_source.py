#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CoinGecko-Datenquelle für den Trading Bot.
Ermöglicht den Zugriff auf Marktdaten von CoinGecko.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import logging
import time
from requests.exceptions import RequestException

from data_sources.base import DataSourceBase, DataSourceException


class CoinGeckoDataSource(DataSourceBase):
    """Implementierung für CoinGecko als Datenquelle"""

    def __init__(self, settings: Dict = None, api_key: str = None):
        """
        Initialisiert die CoinGecko-Datenquelle.

        Args:
            settings: Konfigurationseinstellungen
            api_key: CoinGecko API-Schlüssel (optional für Pro-API)
        """
        super().__init__(settings)
        self.name = "coingecko"
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Mit API-Schlüssel andere URL und Rate-Limit verwenden
        if api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
            self.rate_limit_delay = 0.1  # Pro API hat höhere Limits
        else:
            self.base_url = "https://api.coingecko.com/api/v3"
            self.rate_limit_delay = 1.5  # Kostenlose API hat 50 Anfragen pro Minute

        # Coins-Cache für ID-Mapping
        self.coins_cache = {}

    def _throttle(self):
        """Rate limiting to avoid hitting API limits."""
        time.sleep(self.rate_limit_delay)

    def get_historical_data(self, symbol: str, timeframe: str = "1d",
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

        # Timeframe konvertieren zum CoinGecko-Format
        cg_timeframe = "daily"
        if timeframe in ["1h", "hourly"]:
            cg_timeframe = "hourly"

        # Versuche echte Daten zu laden, es sei denn, Demo-Daten wurden explizit angefordert
        if not use_demo_data:
            try:
                self.logger.info(f"Versuche echte Daten für {symbol} von CoinGecko abzurufen")
                df = self.get_historical_prices(
                    symbol=symbol,
                    timeframe=cg_timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                self.logger.info(f"Erfolgreich echte Daten für {symbol} abgerufen: {len(df)} Datenpunkte")
                return df
            except Exception as e:
                self.logger.warning(
                    f"Fehler beim Abrufen echter Daten für {symbol} von CoinGecko: {e}. Verwende Demo-Daten.")
                # Fallback auf Demo-Daten

        # Demo-Daten generieren
        self.logger.info(f"Generiere Demo-Daten für {symbol} ({timeframe})")
        return self._generate_demo_data(symbol, timeframe, start_date, end_date)

    def _generate_demo_data(self, symbol: str, timeframe: str = "1d",
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
        if timeframe == "1h" or timeframe == "hourly":
            freq = "1H"
        else:
            # Standard ist täglich
            freq = "1D"

        # Zeitindex erstellen
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Basispreis je nach Symbol (für CoinGecko typischerweise ohne /USDT oder /USD)
        symbol_clean = symbol.split('/')[0] if '/' in symbol else symbol

        if "BTC" in symbol_clean:
            base_price = 30000
        elif "ETH" in symbol_clean:
            base_price = 2000
        elif "SOL" in symbol_clean:
            base_price = 70
        elif "ADA" in symbol_clean:
            base_price = 0.5
        elif "XRP" in symbol_clean:
            base_price = 0.6
        else:
            base_price = 100

        # Preisreihe generieren (zufälliger Trend mit etwas Volatilität)
        np.random.seed(hash(symbol) % 10000)  # Seed für Konsistenz

        # Basistrend - CoinGecko-Daten sind oft volatiler
        trend = np.cumsum(np.random.normal(0.0003, 0.03, len(dates)))

        # Preisreihe
        close_prices = base_price * np.exp(trend)

        # OHLCV-Daten generieren
        data = pd.DataFrame({
            'open': close_prices * (1 - np.random.uniform(0, 0.015, len(dates))),
            'high': close_prices * (1 + np.random.uniform(0.005, 0.03, len(dates))),
            'low': close_prices * (1 - np.random.uniform(0.005, 0.03, len(dates))),
            'close': close_prices,
            'volume': np.random.normal(base_price * 200, base_price * 50, len(dates)) * (
                        1 + 0.5 * np.random.random(len(dates))),
            'market_cap': close_prices * np.random.normal(10000000, 1000000, len(dates))
        }, index=dates)

        # Korrigieren: high muss höher als open, close sein; low niedriger
        for i in range(len(data)):
            max_val = max(data.iloc[i]['open'], data.iloc[i]['close'])
            min_val = min(data.iloc[i]['open'], data.iloc[i]['close'])

            if data.iloc[i]['high'] < max_val:
                data.iloc[i, data.columns.get_loc('high')] = max_val * 1.01

            if data.iloc[i]['low'] > min_val:
                data.iloc[i, data.columns.get_loc('low')] = min_val * 0.99

        self.logger.info(f"Generierte {len(data)} Demo-Datenpunkte für {symbol}")
        return data

    def _fetch_coin_id(self, symbol: str) -> str:
        """
        Ermittelt die CoinGecko-ID für ein Symbol.

        Args:
            symbol: Kryptowährungssymbol (z.B. 'BTC')

        Returns:
            CoinGecko-ID
        """
        # Cache prüfen
        if symbol in self.coins_cache:
            return self.coins_cache[symbol]

        # Symbol bereinigen (falls es ein Handelspaar ist)
        if '/' in symbol:
            symbol = symbol.split('/')[0]

        self._throttle()

        # Parameter für API-Anfrage
        params = {}
        if self.api_key:
            params['x_cg_pro_api_key'] = self.api_key

        try:
            # Coins-Liste abrufen
            response = requests.get(f"{self.base_url}/coins/list", params=params)
            response.raise_for_status()

            # Nach Symbol suchen
            coins = response.json()
            for coin in coins:
                if coin['symbol'].upper() == symbol.upper():
                    # In Cache speichern und zurückgeben
                    self.coins_cache[symbol] = coin['id']
                    return coin['id']

            raise DataSourceException(f"Could not find CoinGecko ID for symbol {symbol}")

        except RequestException as e:
            self.logger.error(f"Error fetching coin ID for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch coin ID: {e}")

    def get_historical_prices(self, symbol: str, timeframe: str = "daily",
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        Ruft historische Preisdaten von CoinGecko ab.

        Args:
            symbol: Kryptowährungssymbol (z.B. 'BTC')
            timeframe: Zeitrahmen ('daily' oder 'hourly')
            start_date: Startdatum
            end_date: Enddatum
            limit: Maximale Anzahl der Datenpunkte (wird ignoriert)

        Returns:
            DataFrame mit Preisdaten
        """
        try:
            # Coin-ID ermitteln
            coin_id = self._fetch_coin_id(symbol)

            # Start- und Enddatum in Unix-Zeitstempel (Sekunden)
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)

            if end_date is None:
                end_date = datetime.now()

            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())

            self._throttle()

            # Parameter für API-Anfrage
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }

            if self.api_key:
                params['x_cg_pro_api_key'] = self.api_key

            # API-Endpunkt basierend auf Timeframe
            if timeframe.lower() == 'hourly':
                endpoint = f"coins/{coin_id}/market_chart/range"
            else:
                # Standard ist 'daily'
                endpoint = f"coins/{coin_id}/market_chart/range"

            # Daten abrufen
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()

            data = response.json()

            # Preisdaten extrahieren und in DataFrame umwandeln
            prices = data.get('prices', [])

            if not prices:
                self.logger.warning(f"No price data returned for {symbol}")
                return pd.DataFrame()

            # DataFrame erstellen
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])

            # Volumes hinzufügen, falls verfügbar
            volumes = data.get('total_volumes', [])
            if volumes and len(volumes) == len(prices):
                df['volume'] = [v[1] for v in volumes]

            # Market Caps hinzufügen, falls verfügbar
            market_caps = data.get('market_caps', [])
            if market_caps and len(market_caps) == len(prices):
                df['market_cap'] = [m[1] for m in market_caps]

            # Timestamp in Datetime konvertieren (CoinGecko liefert ms)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # OHLCV-Format emulieren (nur Close-Preis ist verfügbar)
            if timeframe.lower() == 'daily':
                # Tägliche Daten in OHLCV-ähnliches Format umwandeln
                df_resampled = df['price'].resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                })

                # Volume hinzufügen, falls verfügbar
                if 'volume' in df.columns:
                    df_resampled['volume'] = df['volume'].resample('D').last()
                else:
                    df_resampled['volume'] = np.nan

                return df_resampled

            # Für stündliche Daten einfach umbenennen
            df.rename(columns={'price': 'close'}, inplace=True)

            # Fehlende OHLCV-Spalten hinzufügen
            if 'open' not in df.columns:
                df['open'] = df['close'].shift(1)
                df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']

            if 'high' not in df.columns:
                df['high'] = df['close']

            if 'low' not in df.columns:
                df['low'] = df['close']

            if 'volume' not in df.columns:
                df['volume'] = np.nan

            # Spalten neu anordnen für OHLCV-Format
            df = df[['open', 'high', 'low', 'close', 'volume']]

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data from CoinGecko for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch historical data: {e}")

    def get_market_data(self, symbols: List[str] = None, vs_currency: str = "usd") -> pd.DataFrame:
        """
        Ruft aktuelle Marktdaten für mehrere Coins ab.

        Args:
            symbols: Liste von Kryptowährungssymbolen (optional, max. 250)
            vs_currency: Basiswährung für Preise

        Returns:
            DataFrame mit Marktdaten
        """
        self._throttle()

        try:
            # Parameter für API-Anfrage
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': 250,
                'page': 1,
                'sparkline': 'false'
            }

            if self.api_key:
                params['x_cg_pro_api_key'] = self.api_key

            # Wenn Symbole angegeben wurden, IDs ermitteln
            if symbols:
                coin_ids = []
                for symbol in symbols:
                    try:
                        coin_id = self._fetch_coin_id(symbol)
                        coin_ids.append(coin_id)
                    except DataSourceException:
                        self.logger.warning(f"Could not find ID for {symbol}, skipping")

                if not coin_ids:
                    raise DataSourceException("No valid coin IDs found")

                params['ids'] = ','.join(coin_ids)

            # Daten abrufen
            response = requests.get(f"{self.base_url}/coins/markets", params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                self.logger.warning("No market data returned")
                return pd.DataFrame()

            # In DataFrame umwandeln
            df = pd.DataFrame(data)

            # Timestamps in Datetime konvertieren
            for col in ['last_updated', 'ath_date', 'atl_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            return df

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise DataSourceException(f"Could not fetch market data: {e}")

    def get_coin_info(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft detaillierte Informationen zu einer Kryptowährung ab.

        Args:
            symbol: Kryptowährungssymbol (z.B. 'BTC')

        Returns:
            Dictionary mit Coin-Informationen
        """
        try:
            # Coin-ID ermitteln
            coin_id = self._fetch_coin_id(symbol)

            self._throttle()

            # Parameter für API-Anfrage
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'false'
            }

            if self.api_key:
                params['x_cg_pro_api_key'] = self.api_key

            # Daten abrufen
            response = requests.get(f"{self.base_url}/coins/{coin_id}", params=params)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            self.logger.error(f"Error fetching coin info for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch coin info: {e}")

    def get_coin_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft Sentiment-Indikatoren zu einer Kryptowährung ab.

        Args:
            symbol: Kryptowährungssymbol (z.B. 'BTC')

        Returns:
            Dictionary mit Sentiment-Daten
        """
        try:
            # Coin-Informationen abrufen
            coin_info = self.get_coin_info(symbol)

            # Sentiment-Daten extrahieren
            sentiment = {
                'sentiment_votes_up_percentage': coin_info.get('sentiment_votes_up_percentage', 0),
                'sentiment_votes_down_percentage': coin_info.get('sentiment_votes_down_percentage', 0),
                'market_cap_rank': coin_info.get('market_cap_rank'),
                'coingecko_rank': coin_info.get('coingecko_rank'),
                'coingecko_score': coin_info.get('coingecko_score'),
                'community_score': coin_info.get('community_score'),
                'developer_score': coin_info.get('developer_score'),
                'liquidity_score': coin_info.get('liquidity_score'),
                'public_interest_score': coin_info.get('public_interest_score')
            }

            # Community-Daten hinzufügen
            community_data = coin_info.get('community_data', {})
            for key, value in community_data.items():
                sentiment[f'community_{key}'] = value

            return sentiment

        except Exception as e:
            self.logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            raise DataSourceException(f"Could not fetch sentiment data: {e}")