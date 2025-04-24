#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import requests

# Logger einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_binance_data(symbol, timeframe="1d", start_days_ago=365):
    """Lädt historische Daten von Binance herunter"""
    try:
        # Symbol-Format für Binance anpassen
        binance_symbol = symbol.replace('/', '')

        # Zeitrahmen in Millisekunden umrechnen
        interval_map = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000
        }

        ms_interval = interval_map.get(timeframe, 24 * 60 * 60 * 1000)

        # Start- und Endzeit
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (start_days_ago * 24 * 60 * 60 * 1000)

        # URL für Binance API
        url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={timeframe}&startTime={start_time}&endTime={end_time}&limit=1000"

        # Daten abrufen
        response = requests.get(url)
        data = response.json()

        # Daten in DataFrame umwandeln
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Datentypen konvertieren
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        # DataFrame zurückgeben
        return df.set_index('timestamp')

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Daten für {symbol}: {e}")
        return pd.DataFrame()


def save_data_to_csv(df, symbol, timeframe, data_dir="data/market_data/binance"):
    """Speichert ein DataFrame als CSV-Datei"""
    try:
        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(data_dir, exist_ok=True)

        # Dateinamen generieren
        base, quote = symbol.split('/')
        filename = f"{base}_{quote}_{timeframe}.csv"
        filepath = os.path.join(data_dir, filename)

        # Daten speichern
        df.to_csv(filepath)
        logger.info(f"Daten für {symbol} ({timeframe}) gespeichert: {filepath}")

        return filepath

    except Exception as e:
        logger.error(f"Fehler beim Speichern der Daten für {symbol}: {e}")
        return None


def fetch_and_save_data_for_symbols(symbols, timeframe="1d", start_days_ago=365):
    """Lädt und speichert Daten für mehrere Symbole"""
    results = {}

    for symbol in symbols:
        logger.info(f"Lade Daten für {symbol} ({timeframe})...")
        df = fetch_binance_data(symbol, timeframe, start_days_ago)

        if not df.empty:
            logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")
            filepath = save_data_to_csv(df, symbol, timeframe)

            if filepath:
                results[symbol] = {
                    'status': 'success',
                    'rows': len(df),
                    'filepath': filepath
                }
            else:
                results[symbol] = {
                    'status': 'error_saving',
                    'rows': len(df)
                }
        else:
            results[symbol] = {
                'status': 'error_fetching',
                'rows': 0
            }

    return results


if __name__ == "__main__":
    # Symbole, für die Daten abgerufen werden sollen
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    # Zeitrahmen
    timeframes = ["1d", "1h"]

    # Daten für alle Symbole und Zeitrahmen abrufen
    for timeframe in timeframes:
        logger.info(f"Starte Datenabruf für Zeitrahmen {timeframe}...")
        results = fetch_and_save_data_for_symbols(symbols, timeframe, 365)

        # Ergebnisse anzeigen
        for symbol, result in results.items():
            logger.info(f"{symbol}: {result['status']}, {result.get('rows', 0)} Einträge")