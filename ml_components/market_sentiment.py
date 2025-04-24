#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysiert Marktstimmung und On-Chain-Metriken für Kryptowährungen.
Diese Komponente ist Teil der ML-Integration für den Trading Bot.
"""

import logging
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MarketSentimentAnalyzer:
    """
    Analysiert Marktstimmung und On-Chain-Metriken für Kryptowährungen.
    """

    def __init__(self, data_dir: str = "data/sentiment_data", api_keys: Dict[str, str] = None):
        """
        Initialisiert den Sentiment-Analyzer.

        Args:
            data_dir: Verzeichnis für Sentiment-Daten
            api_keys: API-Schlüssel für verschiedene Datenquellen
        """
        self.data_dir = data_dir
        self.api_keys = api_keys or {}

        # Verzeichnis sicherstellen
        os.makedirs(data_dir, exist_ok=True)

        # Cache für Sentiment-Daten
        self.sentiment_cache = {}

    def get_fear_greed_index(self, days: int = 30) -> pd.DataFrame:
        """
        Ruft den Crypto Fear & Greed Index für die angegebene Anzahl von Tagen ab.

        Args:
            days: Anzahl der Tage für die Abfrage

        Returns:
            DataFrame mit Fear & Greed Daten
        """
        cache_file = os.path.join(self.data_dir, "fear_greed_index.json")

        # Cache prüfen
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Prüfen, ob Daten aktuell sind
                last_update = datetime.fromisoformat(cache_data.get("last_update", "2000-01-01"))
                if (datetime.now() - last_update).total_seconds() < 86400:  # 24 Stunden
                    df = pd.DataFrame(cache_data.get("data", []))
                    if not df.empty and len(df) >= days:
                        return df.tail(days)
            except Exception as e:
                logger.warning(f"Fehler beim Laden des Fear & Greed Index Cache: {e}")

        try:
            # API-Abfrage (vereinfachtes Beispiel)
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()

                # Daten extrahieren
                values = data.get("data", [])

                # In DataFrame umwandeln
                df = pd.DataFrame(values)

                # Datumsformat korrigieren
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)

                # Im Cache speichern
                cache_data = {
                    "last_update": datetime.now().isoformat(),
                    "data": values
                }

                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)

                return df
            else:
                logger.error(f"Fehler beim Abrufen des Fear & Greed Index: HTTP {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Fear & Greed Index: {e}")
            return pd.DataFrame()

    def get_social_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Ruft Social Media Sentiment-Daten für ein Symbol ab.

        Args:
            symbol: Symbol der Kryptowährung (z.B. "BTC")
            days: Anzahl der Tage für die Abfrage

        Returns:
            DataFrame mit Sentiment-Daten
        """
        # In einer echten Implementierung würde dies eine API wie Santiment, LunarCrush usw. verwenden
        # Dies ist ein vereinfachtes Beispiel

        cache_key = f"{symbol.lower()}_social"
        cache_file = os.path.join(self.data_dir, f"{cache_key}.json")

        # Cache prüfen
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Prüfen, ob Daten aktuell sind
                last_update = datetime.fromisoformat(cache_data.get("last_update", "2000-01-01"))
                if (datetime.now() - last_update).total_seconds() < 43200:  # 12 Stunden
                    df = pd.DataFrame(cache_data.get("data", []))
                    if not df.empty:
                        return df
            except Exception as e:
                logger.warning(f"Fehler beim Laden des Social Sentiment Cache für {symbol}: {e}")

        # Hier würde normalerweise eine externe API abgefragt werden
        # Für dieses Beispiel generieren wir simulierte Daten

        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')

        # Zufällige Sentiment-Werte generieren
        np.random.seed(42)  # Für Reproduzierbarkeit
        sentiment_data = []

        for date in dates:
            # Sentiment-Werte zwischen -100 und 100
            sentiment = np.random.normal(10, 30)  # Leicht positiver Mittelwert
            volume = np.random.randint(1000, 10000)

            sentiment_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "sentiment": sentiment,
                "volume": volume,
                "mentions": int(volume * np.random.uniform(0.1, 0.3))
            })

        # In DataFrame umwandeln
        df = pd.DataFrame(sentiment_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Im Cache speichern
        cache_data = {
            "last_update": datetime.now().isoformat(),
            "data": sentiment_data
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        return df

    def get_on_chain_metrics(self, symbol: str, metrics: List[str] = None) -> pd.DataFrame:
        """
        Ruft On-Chain-Metriken für ein Symbol ab.

        Args:
            symbol: Symbol der Kryptowährung (z.B. "BTC")
            metrics: Liste der abzurufenden Metriken

        Returns:
            DataFrame mit On-Chain-Metriken
        """
        # Standardmetriken festlegen
        if metrics is None:
            metrics = ["active_addresses", "transaction_count", "avg_transaction_value"]

        cache_key = f"{symbol.lower()}_onchain"
        cache_file = os.path.join(self.data_dir, f"{cache_key}.json")

        # Cache prüfen
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Prüfen, ob Daten aktuell sind
                last_update = datetime.fromisoformat(cache_data.get("last_update", "2000-01-01"))
                if (datetime.now() - last_update).total_seconds() < 86400:  # 24 Stunden
                    df = pd.DataFrame(cache_data.get("data", []))
                    if not df.empty:
                        return df
            except Exception as e:
                logger.warning(f"Fehler beim Laden der On-Chain-Metrik-Cache für {symbol}: {e}")

        # Hier würde normalerweise eine externe API abgefragt werden
        # Für dieses Beispiel generieren wir simulierte Daten

        days = 30  # Letzten 30 Tage
        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')

        # Simulierte On-Chain-Daten generieren
        np.random.seed(43)  # Für Reproduzierbarkeit
        onchain_data = []

        prev_values = {
            "active_addresses": 10000,
            "transaction_count": 200000,
            "avg_transaction_value": 0.5,
            "nvt_ratio": 25,
            "difficulty": 20000000000000
        }

        for date in dates:
            data_point = {"date": date.strftime('%Y-%m-%d')}

            # Für jede Metrik
            for metric in metrics:
                if metric in prev_values:
                    # Zufällige Änderung mit Trendkomponente
                    change = np.random.normal(0, 0.05) + 0.01
                    new_value = prev_values[metric] * (1 + change)
                    data_point[metric] = new_value
                    prev_values[metric] = new_value

            onchain_data.append(data_point)

        # In DataFrame umwandeln
        df = pd.DataFrame(onchain_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Im Cache speichern
        cache_data = {
            "last_update": datetime.now().isoformat(),
            "data": onchain_data
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        return df

    def extract_sentiment_features(self, ohlcv_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extrahiert Sentiment-Features und fügt sie zum OHLCV-DataFrame hinzu.

        Args:
            ohlcv_df: DataFrame mit OHLCV-Daten
            symbol: Symbol der Kryptowährung

        Returns:
            DataFrame mit hinzugefügten Sentiment-Features
        """
        try:
            # Kopie des DataFrames erstellen
            df = ohlcv_df.copy()

            # Symbol bereinigen
            clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol

            # Fear & Greed Index abrufen
            fg_df = self.get_fear_greed_index(days=len(df))

            # Social Sentiment abrufen
            social_df = self.get_social_sentiment(clean_symbol, days=len(df) // 30 * 30)

            # On-Chain-Metriken abrufen
            onchain_df = self.get_on_chain_metrics(clean_symbol)

            # Fear & Greed Index hinzufügen (anpassen an DataFrame-Zeitraum)
            if not fg_df.empty:
                # Daten auf täglicher Basis resampling
                if df.index.freq != 'D':
                    fg_daily = fg_df.resample('D').last()

                    # Auf den Index des OHLCV-DataFrames übertragen
                    for date, row in df.iterrows():
                        date_str = date.strftime('%Y-%m-%d')
                        if date_str in fg_daily.index:
                            df.loc[date, 'fear_greed_value'] = fg_daily.loc[date_str, 'value']
                            df.loc[date, 'fear_greed_classification'] = fg_daily.loc[date_str, 'value_classification']

            # Social Sentiment hinzufügen
            if not social_df.empty:
                # Ähnliches Vorgehen wie bei Fear & Greed
                social_daily = social_df.resample('D').mean()

                for date, row in df.iterrows():
                    date_only = date.strftime('%Y-%m-%d')
                    if date_only in social_daily.index:
                        df.loc[date, 'social_sentiment'] = social_daily.loc[date_only, 'sentiment']
                        df.loc[date, 'social_volume'] = social_daily.loc[date_only, 'volume']

            # On-Chain-Metriken hinzufügen
            if not onchain_df.empty:
                onchain_daily = onchain_df.resample('D').mean()

                for date, row in df.iterrows():
                    date_only = date.strftime('%Y-%m-%d')
                    if date_only in onchain_daily.index:
                        for metric in onchain_df.columns:
                            if metric != 'date':
                                df.loc[date, f'onchain_{metric}'] = onchain_daily.loc[date_only, metric]

            # NaN-Werte durch Forward-Fill ersetzen
            sentiment_cols = [col for col in df.columns if col.startswith(('fear_', 'social_', 'onchain_'))]
            if sentiment_cols:
                df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')

            return df

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren von Sentiment-Features für {symbol}: {e}")
            return ohlcv_df