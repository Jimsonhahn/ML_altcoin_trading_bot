#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basisklassen für Datenquellen.
Definiert die grundlegenden Schnittstellen für alle Datenquellen.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any


class DataSourceException(Exception):
    """Basisklasse für alle Exceptions im data_sources-Modul."""
    pass


class DataSourceBase:
    """Basisklasse für alle Datenquellen."""

    def __init__(self, settings):
        """
        Initialisiert die Datenquelle.

        Args:
            settings: Einstellungen für die Datenquelle.
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.name = "base"

    def get_historical_data(self, symbol: str, timeframe: str = "1h", 
                           start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Ruft historische Daten für ein Symbol ab.

        Args:
            symbol: Das zu abzurufende Symbol (z.B. "BTC/USDT")
            timeframe: Der Zeitrahmen für die Daten (z.B. "1h", "1d")
            start_date: Startdatum für die Daten
            end_date: Enddatum für die Daten
            use_cache: Gibt an, ob der Cache verwendet werden soll

        Returns:
            DataFrame mit OHLCV-Daten
        """
        raise NotImplementedError(f"Diese Methode muss von der abgeleiteten Klasse implementiert werden")

    def get_current_price(self, symbol: str) -> float:
        """
        Ruft den aktuellen Preis für ein Symbol ab.

        Args:
            symbol: Das zu abzurufende Symbol (z.B. "BTC/USDT")

        Returns:
            Aktueller Preis als Float
        """
        raise NotImplementedError(f"Diese Methode muss von der abgeleiteten Klasse implementiert werden")

    def get_symbols(self) -> list:
        """
        Gibt eine Liste aller verfügbaren Symbole zurück.

        Returns:
            Liste der Symbole als Strings
        """
        raise NotImplementedError(f"Diese Methode muss von der abgeleiteten Klasse implementiert werden")
