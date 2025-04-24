
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug-Skript für ML-Komponenten.
"""

import os
import sys
import logging
import pandas as pd
import json

# Logger einrichten
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_imports():
    """Überprüft kritische Importe"""
    try:
        logger.info("Überprüfe scikit-learn...")
        import sklearn
        logger.info(f"scikit-learn Version: {sklearn.__version__}")

        logger.info("Überprüfe pandas...")
        import pandas
        logger.info(f"pandas Version: {pandas.__version__}")

        logger.info("Überprüfe numpy...")
        import numpy
        logger.info(f"numpy Version: {numpy.__version__}")

        # Projekt-spezifische Module
        try:
            logger.info("Überprüfe Projekt-Module...")

            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            from config.settings import Settings
            logger.info("Settings-Modul OK")

            from core.enhanced_backtesting import EnhancedBacktester
            logger.info("EnhancedBacktester-Modul OK")

            try:
                from core.ml_enhanced_backtesting import MLEnhancedBacktester
                logger.info("MLEnhancedBacktester-Modul OK")
            except ImportError as e:
                logger.error(f"MLEnhancedBacktester nicht gefunden: {e}")

            try:
                from ml_components.market_regime import MarketRegimeDetector
                logger.info("MarketRegimeDetector-Modul OK")
            except ImportError as e:
                logger.error(f"MarketRegimeDetector nicht gefunden: {e}")

            try:
                from ml_components.asset_clusters import AssetClusterAnalyzer
                logger.info("AssetClusterAnalyzer-Modul OK")
            except ImportError as e:
                logger.error(f"AssetClusterAnalyzer nicht gefunden: {e}")

            try:
                from strategies.ml_strategy import MLStrategy
                logger.info("MLStrategy-Modul OK")
            except ImportError as e:
                logger.error(f"MLStrategy nicht gefunden: {e}")

        except ImportError as e:
            logger.error(f"Fehler beim Importieren der Projektmodule: {e}")

    except ImportError as e:
        logger.error(f"Kritischer Importfehler: {e}")


def check_data_availability():
    """Überprüft die Verfügbarkeit von Daten"""
    data_dir = "data/market_data/binance"
    logger.info(f"Überprüfe Datenverzeichnis: {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"Verzeichnis {data_dir} existiert nicht!")
        return False

    files = os.listdir(data_dir)
    csv_files = [f for f in files if f.endswith('.csv')]

    if not csv_files:
        logger.error(f"Keine CSV-Dateien in {data_dir} gefunden!")
        return False

    logger.info(f"Gefundene CSV-Dateien: {len(csv_files)}")
    for csv_file in csv_files:
        try:
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)
            logger.info(f"Datei {csv_file}: {len(df)} Zeilen, Spalten: {df.columns.tolist()}")

            # Versuche Datum zu konvertieren, falls vorhanden
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.info(f"Zeitraum: {df['date'].min()} bis {df['date'].max()}")
        except Exception as e:
            logger.error(f"Fehler beim Lesen von {csv_file}: {e}")

    return len(csv_files) > 0


def check_configuration():
    """Überprüft die Konfigurationsdatei"""
    config_file = "config/backtest_config.json"
    logger.info(f"Überprüfe Konfigurationsdatei: {config_file}")

    if not os.path.exists(config_file):
        logger.error(f"Konfigurationsdatei {config_file} nicht gefunden!")
        return False

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Wichtige Einstellungen prüfen
        required_settings = [
            "backtest.start_date", "backtest.end_date", "backtest.initial_balance",
            "trading_pairs", "data.source", "ml.enabled"
        ]

        for setting in required_settings:
            if setting not in config:
                logger.warning(f"Einstellung {setting} fehlt in der Konfiguration!")

        if "ml.enabled" in config and config["ml.enabled"]:
            logger.info("ML-Funktionen sind aktiviert")

            ml_settings = ["ml.data_dir", "ml.models_dir", "ml.output_dir"]
            for setting in ml_settings:
                if setting in config:
                    path = config[setting]
                    if not os.path.exists(path):
                        logger.warning(f"ML-Verzeichnis {path} existiert nicht!")
                        os.makedirs(path, exist_ok=True)
                        logger.info(f"Verzeichnis {path} erstellt")

        logger.info(f"Konfiguration OK: {len(config)} Einstellungen gefunden")
        return True

    except Exception as e:
        logger.error(f"Fehler beim Lesen der Konfigurationsdatei: {e}")
        return False


def test_market_regime_detector():
    """Testet die MarketRegimeDetector-Komponente"""
    logger.info("Teste MarketRegimeDetector...")

    try:
        from ml_components.market_regime import MarketRegimeDetector

        # Detector initialisieren
        detector = MarketRegimeDetector(data_dir="data/market_data", n_regimes=5)
        logger.info("MarketRegimeDetector erfolgreich initialisiert")

        # Testdaten laden (erste gefundene CSV-Datei)
        data_dir = "data/market_data/binance"
        if not os.path.exists(data_dir):
            logger.error(f"Datenverzeichnis {data_dir} nicht gefunden!")
            return False

        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not files:
            logger.error("Keine CSV-Dateien zum Testen gefunden!")
            return False

        test_file = os.path.join(data_dir, files[0])
        logger.info(f"Verwende Testdatei: {test_file}")

        # Daten laden
        df = pd.read_csv(test_file)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)

        # Symbol extrahieren
        symbol_parts = files[0].split('_')
        if len(symbol_parts) >= 2:
            symbol = f"{symbol_parts[0]}/{symbol_parts[1]}"
        else:
            symbol = "BTC/USDT"

        # Testfunktion des Detectors ausführen
        logger.info(f"Teste Marktregime-Erkennung mit {symbol}...")

        # Marktdaten simulieren
        detector.market_data = {symbol: df}

        # Features extrahieren
        try:
            features_df = detector.extract_market_features()
            if not features_df.empty:
                logger.info(
                    f"Feature-Extraktion erfolgreich: {len(features_df)} Datenpunkte, {len(features_df.columns)} Features")
                logger.info(f"Feature-Namen: {features_df.columns.tolist()}")

                # Modell trainieren
                if detector.train_regime_model(features_df):
                    logger.info("Modelltraining erfolgreich")

                    # Regime vorhersagen
                    current_features = features_df.iloc[-1:].copy()
                    regime = detector.predict_regime(current_features)

                    logger.info(f"Aktuelles Regime: {regime}")
                    logger.info(f"Regime-Label: {detector.regime_labels.get(regime, 'Unbekannt')}")

                    # Modell speichern und laden testen
                    model_path = "data/ml_models/test_regime_model.pkl"
                    if detector.save_model(model_path):
                        logger.info(f"Modell erfolgreich gespeichert: {model_path}")

                    return True
                else:
                    logger.error("Modelltraining fehlgeschlagen")
            else:
                logger.error("Feature-Extraktion lieferte leeres DataFrame")
        except Exception as e:
            logger.error(f"Fehler bei der Feature-Extraktion: {e}")

        return False

    except ImportError as e:
        logger.error(f"MarketRegimeDetector nicht importierbar: {e}")
        return False
    except Exception as e:
        logger.error(f"Fehler beim Testen des MarketRegimeDetector: {e}")
        return False


if __name__ == "__main__":
    logger.info("=== ML-Komponenten Diagnose ===")

    logger.info("1. Import-Check...")
    check_imports()

    logger.info("\n2. Daten-Check...")
    data_available = check_data_availability()

    logger.info("\n3. Konfigurations-Check...")
    config_ok = check_configuration()

    logger.info("\n4. MarketRegimeDetector-Test...")
    regime_detector_ok = test_market_regime_detector()

    # Zusammenfassung
    logger.info("\n=== Diagnose-Zusammenfassung ===")
    logger.info(f"Daten verfügbar: {'Ja' if data_available else 'Nein'}")
    logger.info(f"Konfiguration OK: {'Ja' if config_ok else 'Nein'}")
    logger.info(f"MarketRegimeDetector OK: {'Ja' if regime_detector_ok else 'Nein'}")

    if data_available and config_ok and regime_detector_ok:
        logger.info("ML-Komponenten sind BEREIT FÜR DEN EINSATZ!")
    else:
        logger.warning("Es gibt PROBLEME mit den ML-Komponenten. Siehe Details oben.")
EOF

# Debug-Skript ausführbar machen (nur für Linux/Mac)
chmod + x
debug_ml_components.py