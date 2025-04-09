#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Konfigurationseinstellungen für den Trading Bot.
Lädt Einstellungen aus config-Dateien und Umgebungsvariablen.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()


class Settings:
    """Zentrale Konfigurationsklasse für den Trading Bot"""

    def __init__(self, profile: str = "default"):
        """
        Initialisiert die Konfigurationseinstellungen.

        Args:
            profile: Name des Konfigurationsprofils
        """
        self.profile = profile
        self.logger = logging.getLogger(__name__)

        # Grundeinstellungen
        self.base_config = {
            # Exchange Einstellungen
            "exchange": {
                "name": "binance",
                "testnet": True,  # Testnet oder Live
                "api_throttle_rate": 1.0,  # Anfragen pro Sekunde
            },

            # Trading Paare
            "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
            "watchlist": ["DOT/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT"],

            # Risikomanagement
            "risk": {
                "position_size": 0.05,  # 5% des Kapitals pro Trade
                "max_open_positions": 5,  # Maximale Anzahl offener Positionen
                "stop_loss": 0.03,  # 3% Stop Loss
                "take_profit": 0.06,  # 6% Take Profit
                "max_daily_loss": 0.05,  # 5% maximaler Tagesverlust
            },

            # Zeitrahmen und Intervalle
            "timeframes": {
                "analysis": "1h",  # Hauptzeitrahmen für Analyse
                "check_interval": 300,  # Sekunden zwischen Überprüfungen
                "social_check_interval": 3600,  # Sekunden zwischen Social Media Checks
            },

            # Technische Indikatoren
            "technical": {
                "rsi": {
                    "period": 14,
                    "oversold": 30,
                    "overbought": 70
                },
                "macd": {
                    "fast": 12,
                    "slow": 26,
                    "signal": 9
                },
                "ma": {
                    "short": 20,
                    "long": 50
                },
                "bollinger": {
                    "period": 20,
                    "std_dev": 2
                }
            },

            # Social Media Analyse
            "social_media": {
                "enabled": True,
                "sources": {
                    "reddit": {
                        "enabled": True,
                        "subreddits": ["CryptoCurrency", "altcoin", "SatoshiStreetBets"],
                        "post_limit": 100
                    },
                    "twitter": {
                        "enabled": True,
                        "influencers": ["cryptobirb", "CryptoKaleo", "cz_binance"],
                        "search_terms": ["crypto", "altcoin", "cryptocurrency"]
                    },
                    "youtube": {
                        "enabled": False
                    }
                },
                "sentiment_weight": 0.3  # Gewichtung im Gesamtsignal
            },

            # Machine Learning
            "machine_learning": {
                "enabled": True,
                "confidence_threshold": 0.7,
                "features": [
                    "trend", "rsi", "ma_short", "ma_long", "macd", "macd_signal",
                    "stochastic_k", "stochastic_d", "upper_band", "lower_band",
                    "volume", "price_change_24h", "reddit_sentiment", "twitter_sentiment"
                ],
                "training_interval": 86400  # Trainieren alle 24 Stunden
            },

            # Backtesting
            "backtest": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_balance": 10000,  # USDT
                "commission": 0.001  # 0.1%
            },

            # Logging und Benachrichtigungen
            "logging": {
                "level": "INFO",
                "file_enabled": True,
                "file_path": "logs/trading_bot.log",
                "notify_trades": True,
                "notify_errors": True
            }
        }

        # Lade profilspezifische Einstellungen
        self.load_profile(profile)

        # Lade Umgebungsvariablen, die Vorrang haben
        self.load_from_env()

        self.logger.info(f"Configuration loaded for profile: {profile}")

    def load_profile(self, profile: str) -> None:
        """
        Lädt profilspezifische Einstellungen aus Konfigurationsdatei.

        Args:
            profile: Name des zu ladenden Profils
        """
        # Absolute Pfade verwenden
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(base_dir, "config", "profiles", f"{profile}.json")

        self.logger.info(f"Looking for profile at: {profile_path}")

        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)

                # Optional log config for debugging
                # self.logger.info(f"Profile config: {json.dumps(profile_data, indent=2)}")

                # Rekursiv die Konfiguration aktualisieren
                self._update_nested_dict(self.base_config, profile_data)

                self.logger.info(f"Loaded configuration profile: {profile}")
            except Exception as e:
                self.logger.error(f"Error loading profile '{profile}': {e}")
        else:
            self.logger.warning(f"Profile '{profile}' not found, using default settings")

    def load_from_env(self) -> None:
        """Lädt Überschreibungen aus Umgebungsvariablen"""
        # Exchange Einstellungen
        if os.getenv('EXCHANGE_NAME'):
            self.base_config['exchange']['name'] = os.getenv('EXCHANGE_NAME')

        if os.getenv('EXCHANGE_TESTNET'):
            self.base_config['exchange']['testnet'] = os.getenv('EXCHANGE_TESTNET').lower() == 'true'

        # Trading Paare (kommagetrennte Liste)
        if os.getenv('TRADING_PAIRS'):
            self.base_config['trading_pairs'] = os.getenv('TRADING_PAIRS').split(',')

        # Risikomanagement
        if os.getenv('POSITION_SIZE'):
            self.base_config['risk']['position_size'] = float(os.getenv('POSITION_SIZE'))

        if os.getenv('STOP_LOSS'):
            self.base_config['risk']['stop_loss'] = float(os.getenv('STOP_LOSS'))

        if os.getenv('TAKE_PROFIT'):
            self.base_config['risk']['take_profit'] = float(os.getenv('TAKE_PROFIT'))

        # Weitere Umgebungsvariablen können nach Bedarf hinzugefügt werden

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Aktualisiert rekursiv ein verschachteltes Dictionary.

        Args:
            d: Ziel-Dictionary
            u: Quell-Dictionary mit Aktualisierungen

        Returns:
            Aktualisiertes Dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """
        Holt einen Konfigurationswert über einen Punktnotations-Pfad.

        Args:
            key: Pfad zum Konfigurationswert (z.B. 'exchange.name')
            default: Standardwert, falls der Schlüssel nicht existiert

        Returns:
            Konfigurationswert oder Standardwert
        """
        keys = key.split('.')
        value = self.base_config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Setzt einen Konfigurationswert über einen Punktnotations-Pfad.

        Args:
            key: Pfad zum Konfigurationswert (z.B. 'exchange.name')
            value: Zu setzender Wert
        """
        keys = key.split('.')
        config = self.base_config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_profile(self, profile_name: str = None) -> None:
        """
        Speichert die aktuelle Konfiguration als Profil.

        Args:
            profile_name: Name des zu speichernden Profils,
                          falls None wird der aktuelle Profilname verwendet
        """
        if profile_name is None:
            profile_name = self.profile

        profile_dir = "config/profiles"
        os.makedirs(profile_dir, exist_ok=True)

        profile_path = f"{profile_dir}/{profile_name}.json"

        try:
            with open(profile_path, 'w') as f:
                json.dump(self.base_config, f, indent=4)

            self.logger.info(f"Saved configuration as profile: {profile_name}")
        except Exception as e:
            self.logger.error(f"Error saving profile '{profile_name}': {e}")