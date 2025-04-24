#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NewCoinMonitor für den Trading Bot und Backtest-Modul.
Überwacht und analysiert neue Coins am Markt.
"""

import logging
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union


class NewCoinMonitor:
    """
    Überwacht und analysiert neue Coins am Markt.
    """

    def __init__(self, data_dir: str = "data/market_data", output_dir: str = "data/ml_analysis/new_coins"):
        """
        Initialisiert den New Coin Monitor.

        Args:
            data_dir: Verzeichnis mit Marktdaten
            output_dir: Verzeichnis für Analyseergebnisse
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.known_coins = set()
        self.new_coins_watchlist = {}  # Symbol -> Analyse-Info
        self.logger = logging.getLogger(__name__)
        self.last_check_time = datetime.now() - timedelta(days=1)  # Force initial check

        # Verzeichnisse sicherstellen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Watchlist laden, falls vorhanden
        self._load_watchlist()

    def update_known_coins(self, data_manager=None) -> bool:
        """
        Aktualisiert die Liste bekannter Coins.

        Args:
            data_manager: Optionaler DataManager für zusätzliche Daten

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Daten-Verzeichnis durchsuchen
            binance_dir = os.path.join(self.data_dir, "binance")

            if not os.path.exists(binance_dir):
                os.makedirs(binance_dir, exist_ok=True)
                self.logger.warning(f"Verzeichnis {binance_dir} erstellt")

            # Bekannte Coins aus Dateinamen extrahieren
            known_coins = set()

            for filename in os.listdir(binance_dir):
                if filename.endswith('.csv'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        symbol = f"{parts[0]}/{parts[1]}"
                        known_coins.add(symbol)

            # Falls DataManager vorhanden, verfügbare Symbole abfragen
            if data_manager:
                available_symbols = data_manager.get_available_symbols()
                if available_symbols:
                    known_coins.update(available_symbols)

            # Bei Binance-API nach allen verfügbaren Symbolen fragen
            try:
                response = requests.get('https://api.binance.com/api/v3/exchangeInfo')
                if response.status_code == 200:
                    exchange_info = response.json()
                    symbols = exchange_info.get('symbols', [])

                    for symbol_info in symbols:
                        base_asset = symbol_info.get('baseAsset')
                        quote_asset = symbol_info.get('quoteAsset')

                        if quote_asset == 'USDT':
                            symbol = f"{base_asset}/{quote_asset}"
                            known_coins.add(symbol)
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen der Exchange-Info: {e}")

            # Bekannte Coins aktualisieren
            self.known_coins = known_coins

            self.logger.info(f"{len(self.known_coins)} bekannte Coins aktualisiert")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der bekannten Coins: {e}")
            return False

    def check_for_new_coins(self, force_check: bool = False) -> List[str]:
        """
        Überprüft, ob neue Coins am Markt verfügbar sind.

        Args:
            force_check: Erzwingt eine Überprüfung unabhängig von der Zeit seit dem letzten Check

        Returns:
            Liste neu erkannter Coins
        """
        try:
            # Zeit seit letztem Check prüfen
            now = datetime.now()
            hours_since_last_check = (now - self.last_check_time).total_seconds() / 3600

            # Nur alle 6 Stunden prüfen, außer wenn erzwungen
            if hours_since_last_check < 6 and not force_check:
                return []

            self.last_check_time = now

            # Aktualisiere die Liste der bekannten Coins
            self.update_known_coins()

            # Bei der Binance-API nach aktuell handelbaren Symbolen fragen
            new_coins = []

            try:
                response = requests.get('https://api.binance.com/api/v3/ticker/24hr')
                if response.status_code == 200:
                    tickers = response.json()

                    for ticker in tickers:
                        symbol = ticker.get('symbol', '')

                        # Format in Base/Quote umwandeln (z.B. BTCUSDT -> BTC/USDT)
                        if symbol.endswith('USDT'):
                            base = symbol[:-4]
                            formatted_symbol = f"{base}/USDT"

                            # Prüfen, ob dieser Coin neu ist
                            if formatted_symbol not in self.known_coins:
                                # Weitere Filter anwenden (z.B. Mindestvolumen)
                                volume = float(ticker.get('volume', 0))

                                if volume > 1000:  # Mindestvolumen von 1000 Einheiten
                                    new_coins.append(formatted_symbol)
                                    self.logger.info(f"Neuer Coin erkannt: {formatted_symbol} (Volumen: {volume})")

                                    # Coin zur Watchlist hinzufügen, falls noch nicht drin
                                    if formatted_symbol not in self.new_coins_watchlist:
                                        self.new_coins_watchlist[formatted_symbol] = {
                                            'discovery_time': datetime.now().isoformat(),
                                            'initial_volume': volume,
                                            'analysis_status': 'pending',
                                            'data_available': False,
                                            'days_tracked': 0
                                        }
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen der Ticker-Daten: {e}")

            if new_coins:
                self.logger.info(f"{len(new_coins)} neue Coins entdeckt: {', '.join(new_coins)}")

                # Führe eine erste Datensammlung für neue Coins durch
                for coin in new_coins:
                    self.collect_data_for_coin(coin)

                # Watchlist speichern
                self._save_watchlist()
            else:
                self.logger.info("Keine neuen Coins entdeckt")

            return new_coins

        except Exception as e:
            self.logger.error(f"Fehler bei der Überprüfung auf neue Coins: {e}")
            return []

    def collect_data_for_coin(self, coin: str) -> bool:
        """
        Sammelt Daten für einen Coin.

        Args:
            coin: Symbol des Coins (z.B. 'BTC/USDT')

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Daten über die Binance-API abrufen
            base, quote = coin.split('/')
            symbol = f"{base}{quote}"

            # Kline-Daten abrufen (max. 1000 Einträge)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"

            response = requests.get(url)
            if response.status_code == 200:
                klines = response.json()

                if klines:
                    # Daten in DataFrame umwandeln
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])

                    # Datentypen konvertieren
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                    for col in ['open', 'high', 'low', 'close', 'volume',
                                'quote_asset_volume', 'taker_buy_base_asset_volume',
                                'taker_buy_quote_asset_volume']:
                        df[col] = pd.to_numeric(df[col])

                    # Speicherpfad
                    binance_dir = os.path.join(self.data_dir, "binance")
                    os.makedirs(binance_dir, exist_ok=True)

                    # Dateiname generieren
                    current_date = datetime.now().strftime("%Y%m%d")
                    filename = f"{base}_{quote}_1d_{current_date}.csv"
                    filepath = os.path.join(binance_dir, filename)

                    # Daten speichern
                    df.to_csv(filepath, index=False)

                    self.logger.info(f"Daten für {coin} gesammelt und gespeichert: {len(df)} Einträge")

                    # Watchlist aktualisieren
                    if coin in self.new_coins_watchlist:
                        self.new_coins_watchlist[coin]['data_available'] = True
                        self.new_coins_watchlist[coin]['first_data_date'] = df['timestamp'].min().isoformat()
                        self.new_coins_watchlist[coin]['days_tracked'] = len(df)
                        self.new_coins_watchlist[coin]['last_updated'] = datetime.now().isoformat()

                    # Watchlist speichern
                    self._save_watchlist()
                    return True

            self.logger.warning(f"Fehler beim Abrufen der Kline-Daten für {coin}: {response.status_code}")
            return False

        except Exception as e:
            self.logger.error(f"Fehler bei der Datensammlung für {coin}: {e}")
            return False

    def get_coins_for_analysis(self, min_days: int = 3) -> List[str]:
        """
        Gibt Liste von Coins zurück, die bereit für die Analyse sind.

        Args:
            min_days: Minimale Anzahl an Tagen für die Analyse

        Returns:
            Liste mit Coin-Symbolen
        """
        coins_to_analyze = []

        # Für jeden Coin in der Watchlist
        for coin, info in list(self.new_coins_watchlist.items()):
            days_tracked = info.get('days_tracked', 0)
            analysis_status = info.get('analysis_status', 'pending')

            # Wenn genügend Daten vorliegen und noch nicht analysiert
            if days_tracked >= min_days and analysis_status == 'pending':
                coins_to_analyze.append(coin)

        return coins_to_analyze

    def update_coin_status(self, coin: str, analysis_result: Dict[str, Any]) -> None:
        """
        Aktualisiert den Status eines Coins nach der Analyse.

        Args:
            coin: Symbol des Coins
            analysis_result: Ergebnis der Analyse
        """
        if coin in self.new_coins_watchlist:
            self.new_coins_watchlist[coin]['analysis_status'] = 'analyzed'
            self.new_coins_watchlist[coin]['analysis_time'] = datetime.now().isoformat()
            self.new_coins_watchlist[coin]['analysis_result'] = analysis_result

            # Watchlist speichern
            self._save_watchlist()

            # Ergebnis auch separat speichern
            self._save_analysis_result(coin, analysis_result)
        else:
            self.logger.warning(f"Coin {coin} nicht in der Watchlist gefunden")

    def get_coin_data(self, coin: str) -> pd.DataFrame:
        """
        Lädt die Daten für einen Coin.

        Args:
            coin: Symbol des Coins (z.B. 'BTC/USDT')

        Returns:
            DataFrame mit Kursdaten oder leerer DataFrame bei Fehler
        """
        try:
            base, quote = coin.split('/')
            binance_dir = os.path.join(self.data_dir, "binance")

            # Datei suchen
            csv_path = None
            for f in os.listdir(binance_dir):
                if f.startswith(f"{base}_{quote}_1d"):
                    csv_path = os.path.join(binance_dir, f)
                    break

            if not csv_path:
                self.logger.warning(f"Keine Daten gefunden für {coin}")
                return pd.DataFrame()

            # Daten laden
            df = pd.read_csv(csv_path)

            # Datum als Index setzen, falls vorhanden
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten für {coin}: {e}")
            return pd.DataFrame()

    def update_all_coins(self, asset_analyzer=None) -> Dict[str, Any]:
        """
        Aktualisiert alle Coins in der Watchlist und analysiert sie bei Bedarf.

        Args:
            asset_analyzer: Optional AssetClusterAnalyzer für die Analyse

        Returns:
            Dictionary mit Aktualisierungsergebnissen
        """
        results = {
            'updated': [],
            'analyzed': [],
            'errors': []
        }

        try:
            # Alle Coins in der Watchlist durchgehen
            for coin, info in list(self.new_coins_watchlist.items()):
                # 1. Daten aktualisieren
                updated = self.collect_data_for_coin(coin)

                if updated:
                    results['updated'].append(coin)

                    # 2. Analysieren, falls genügend Daten und noch nicht analysiert
                    if (info.get('days_tracked', 0) >= 3 and
                            info.get('analysis_status', 'pending') == 'pending' and
                            asset_analyzer is not None):

                        # Daten laden
                        df = self.get_coin_data(coin)

                        if not df.empty:
                            # Coin analysieren
                            analysis_result = asset_analyzer.analyze_new_coin(coin, df)

                            if analysis_result and analysis_result.get('status') == 'analyzed':
                                # Status aktualisieren
                                self.update_coin_status(coin, analysis_result)
                                results['analyzed'].append(coin)
                            else:
                                results['errors'].append((coin, "Analyse fehlgeschlagen"))
                        else:
                            results['errors'].append((coin, "Keine Daten verfügbar"))
                else:
                    results['errors'].append((coin, "Aktualisierung fehlgeschlagen"))

            return results

        except Exception as e:
            self.logger.error(f"Fehler bei der Aktualisierung aller Coins: {e}")
            return results

    def get_interesting_coins(self) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste interessanter Coins zurück, die für Trading in Frage kommen.

        Returns:
            Liste mit interessanten Coins und ihren Eigenschaften
        """
        interesting_coins = []

        for coin, info in self.new_coins_watchlist.items():
            # Nur analysierte Coins betrachten
            if info.get('analysis_status') == 'analyzed':
                analysis_result = info.get('analysis_result', {})

                # Nur Coins mit positiver Empfehlung
                strategy = analysis_result.get('recommended_strategy', '')
                if 'Long' in strategy and 'Vermeiden' not in strategy:
                    interesting_coins.append({
                        'symbol': coin,
                        'discovery_time': info.get('discovery_time'),
                        'days_tracked': info.get('days_tracked', 0),
                        'cluster': analysis_result.get('predicted_cluster', -1),
                        'strategy': strategy,
                        'similar_to': analysis_result.get('similar_coins', [])[:3]
                    })

        # Nach Entdeckungszeit sortieren (neueste zuerst)
        interesting_coins.sort(key=lambda x: x['discovery_time'], reverse=True)

        return interesting_coins

    def _save_watchlist(self) -> bool:
        """
        Speichert die Watchlist in einer Datei.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            filepath = os.path.join(self.output_dir, "watchlist.json")

            with open(filepath, 'w') as f:
                json.dump(self.new_coins_watchlist, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Watchlist: {e}")
            return False

    def _load_watchlist(self) -> bool:
        """
        Lädt die Watchlist aus einer Datei.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            filepath = os.path.join(self.output_dir, "watchlist.json")

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.new_coins_watchlist = json.load(f)
                self.logger.info(f"Watchlist mit {len(self.new_coins_watchlist)} Coins geladen")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Watchlist: {e}")
            return False

    def _save_analysis_result(self, coin: str, analysis_result: Dict[str, Any]) -> bool:
        """
        Speichert das Analyseergebnis eines Coins.

        Args:
            coin: Symbol des Coins
            analysis_result: Analyseergebnis

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Verzeichnis sicherstellen
            os.makedirs(self.output_dir, exist_ok=True)

            # Dateinamen generieren
            base, quote = coin.split('/')
            filename = f"{base}_{quote}_analysis.json"
            filepath = os.path.join(self.output_dir, filename)

            # Analyseergebnis speichern
            with open(filepath, 'w') as f:
                json.dump(analysis_result, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Analyseergebnisses für {coin}: {e}")
            return False