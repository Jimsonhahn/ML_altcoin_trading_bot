#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Manager für Trading Bot.
Verwaltet den Zugriff auf verschiedene Datenquellen.
"""

import logging
import pandas as pd
from datetime import datetime
import os
from typing import Dict, Any, Optional, List

# Importe für die Datenquellen
from data_sources.data_sources.base import DataSourceBase
from data_sources.data_sources.binance_source import BinanceDataSource
from data_sources.data_sources.coingecko_source import CoinGeckoDataSource


class DataManager:
    """Manager für den Zugriff auf verschiedene Datenquellen."""

    def __init__(self, settings):
        """
        Initialisiert den Data Manager.

        Args:
            settings: Bot-Einstellungen
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.sources = {}

        # Datenquellen initialisieren
        self.add_source('binance', BinanceDataSource(settings))
        self.add_source("coingecko", CoinGeckoDataSource(settings))

        # Cache-Verzeichnis
        self.cache_dir = settings.get('data.cache_dir', 'data/market_data')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.default_source = settings.get('data.default_source', 'binance')
        self.logger.info(f"Data Manager initialisiert mit Quellen: {', '.join(self.sources.keys())}")

    def add_source(self, name: str, source: DataSourceBase) -> None:
        """
        Fügt eine Datenquelle hinzu.

        Args:
            name: Name der Datenquelle
            source: Datenquelle-Objekt
        """
        self.sources[name] = source
        self.logger.debug(f"Datenquelle hinzugefügt: {name}")

    def get_source(self, name: Optional[str] = None) -> DataSourceBase:
        """
        Gibt eine Datenquelle zurück.

        Args:
            name: Name der Datenquelle oder None für die Standardquelle

        Returns:
            Datenquelle-Objekt

        Raises:
            ValueError: Wenn die Datenquelle nicht gefunden wird
        """
        source_name = name or self.default_source

        if source_name not in self.sources:
            raise ValueError(f"Datenquelle {source_name} nicht gefunden")

        return self.sources[source_name]

    def get_historical_data(self, symbol: str, source: str = None, timeframe: str = "1h",
                          start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Ruft historische Daten für ein Symbol ab.

        Args:
            symbol: Das zu abzurufende Symbol (z.B. "BTC/USDT")
            source: Name der Datenquelle oder None für die Standardquelle
            timeframe: Der Zeitrahmen für die Daten (z.B. "1h", "1d")
            start_date: Startdatum für die Daten
            end_date: Enddatum für die Daten
            use_cache: Gibt an, ob der Cache verwendet werden soll

        Returns:
            DataFrame mit OHLCV-Daten
        """
        data_source = self.get_source(source)

        # Cache-Dateiname konstruieren
        cache_file = None
        if use_cache:
            source_name = source or self.default_source
            symbol_safe = symbol.replace("/", "_")

            start_str = start_date.strftime("%Y%m%d") if start_date else ""
            end_str = end_date.strftime("%Y%m%d") if end_date else ""

            cache_file = os.path.join(
                self.cache_dir, 
                source_name,
                f"{symbol_safe}_{timeframe}_{start_str}_{end_str}.csv"
            )

            # Prüfen, ob Cache-Datei existiert
            if os.path.exists(cache_file):
                self.logger.debug(f"Lade Daten aus Cache: {cache_file}")
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    return df
                except Exception as e:
                    self.logger.warning(f"Fehler beim Laden aus Cache: {e}")

        # Daten von der Quelle abrufen
        self.logger.info(f"Rufe Daten ab für {symbol} von {data_source.name}")
        df = data_source.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Im Cache speichern, falls aktiviert
        if use_cache and cache_file and not df.empty:
            try:
                # Verzeichnis erstellen, falls nicht vorhanden
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)

                # Als CSV speichern
                df.to_csv(cache_file)
                self.logger.debug(f"Daten im Cache gespeichert: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Fehler beim Speichern im Cache: {e}")

        return df

    def get_symbols(self, source: str = None) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren Symbole zurück.

        Args:
            source: Name der Datenquelle oder None für die Standardquelle

        Returns:
            Liste der Symbole als Strings
        """
        data_source = self.get_source(source)
        return data_source.get_symbols()

    def get_current_price(self, symbol: str, source: str = None) -> float:
        """
        Ruft den aktuellen Preis für ein Symbol ab.

        Args:
            symbol: Das zu abzurufende Symbol (z.B. "BTC/USDT")
            source: Name der Datenquelle oder None für die Standardquelle

        Returns:
            Aktueller Preis als Float
        """
        data_source = self.get_source(source)
        return data_source.get_current_price(symbol)

    def clear_cache(self, source: str = None, symbol: str = None) -> None:
        """
        Leert den Cache für eine Datenquelle und/oder ein Symbol.

        Args:
            source: Name der Datenquelle oder None für alle Quellen
            symbol: Das Symbol oder None für alle Symbole
        """
        if source:
            # Nur eine bestimmte Quelle löschen
            source_dir = os.path.join(self.cache_dir, source)
            if os.path.exists(source_dir):
                if symbol:
                    # Nur ein bestimmtes Symbol löschen
                    symbol_safe = symbol.replace("/", "_")
                    pattern = os.path.join(source_dir, f"{symbol_safe}_*.csv")
                    for file in glob.glob(pattern):
                        os.remove(file)
                    self.logger.info(f"Cache gelöscht für {symbol} von {source}")
                else:
                    # Alle Symbole für diese Quelle löschen
                    for file in glob.glob(os.path.join(source_dir, "*.csv")):
                        os.remove(file)
                    self.logger.info(f"Cache gelöscht für alle Symbole von {source}")
        else:
            # Alle Quellen löschen
            if symbol:
                # Nur ein bestimmtes Symbol in allen Quellen löschen
                symbol_safe = symbol.replace("/", "_")
                for source_dir in os.listdir(self.cache_dir):
                    source_path = os.path.join(self.cache_dir, source_dir)
                    if os.path.isdir(source_path):
                        pattern = os.path.join(source_path, f"{symbol_safe}_*.csv")
                        for file in glob.glob(pattern):
                            os.remove(file)
                self.logger.info(f"Cache gelöscht für {symbol} von allen Quellen")
            else:
                # Kompletten Cache löschen
                for source_dir in os.listdir(self.cache_dir):
                    source_path = os.path.join(self.cache_dir, source_dir)
                    if os.path.isdir(source_path):
                        for file in glob.glob(os.path.join(source_path, "*.csv")):
                            os.remove(file)
                self.logger.info("Kompletter Cache gelöscht")
