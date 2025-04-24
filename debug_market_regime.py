#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug-Skript für den MarketRegimeDetector.
Testet verschiedene Funktionen und identifiziert mögliche Probleme.
"""

import os
import logging
import pandas as pd
from ml_components.market_regime import MarketRegimeDetector

# Logging einrichten
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("market_regime_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("regime_debug")


def main():
    """
    Hauptfunktion für das Debugging des MarketRegimeDetector.
    """
    logger.info("MarketRegimeDetector erfolgreich importiert")
    logger.info("=== Start des Debug-Prozesses für MarketRegimeDetector ===")

    # Überprüfe Datendateien
    logger.info("=== Überprüfe Datendateien ===")
    data_dir = "data/market_data/binance"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.warning(f"Verzeichnis {data_dir} wurde erstellt, da es nicht existierte")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    logger.info(f"Gefundene CSV-Dateien: {len(csv_files)}")

    # Eine Datei zur Überprüfung auswählen
    for csv_file in csv_files:
        if "BTC" in csv_file and "1d" in csv_file:
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)
            logger.info(f"Datei {csv_file}: {len(df)} Zeilen, Spalten: {list(df.columns)}")

            # Überprüfe Timestamp-Format
            if 'timestamp' in df.columns:
                logger.info(f"Timestamp-Beispiel: {df['timestamp'].iloc[0]}, Typ: {type(df['timestamp'].iloc[0])}")

                # Überprüfe, ob Timestamp im datetime-Format ist
                if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    logger.warning("Timestamp ist nicht im datetime-Format")

                    # Versuch der Konvertierung
                    try:
                        datetime_example = pd.to_datetime(df['timestamp'].iloc[0])
                        logger.info(f"Erfolgreich als datetime geparst: {datetime_example}")
                    except Exception as e:
                        logger.error(f"Fehler beim Parsen des Timestamps: {e}")

            # Finde mindestens 2 Dateien für den Test
            if len([f for f in csv_files if f.startswith('BTC')]) >= 2:
                break

    logger.info("=== Dateiprüfung abgeschlossen ===")

    # Initialisiere den MarketRegimeDetector
    detector = MarketRegimeDetector(data_dir=os.path.dirname(data_dir))
    logger.info("MarketRegimeDetector initialisiert")

    # Lade BTC-Testdaten
    test_file = next((f for f in csv_files if "BTC" in f and "1d" in f), None)
    if test_file:
        test_path = os.path.join(data_dir, test_file)
        logger.info(f"Lade Testdaten aus {test_path}")

        df = pd.read_csv(test_path)

        # Timestamp als Index setzen
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            logger.info("Timestamp als Index gesetzt")

        logger.info(f"Daten geladen: {len(df)} Zeilen, Spalten: {list(df.columns)}")

        # Setze die Marktdaten
        detector.market_data = {"BTC/USDT": df}

        # Teste Feature-Extraktion
        logger.info("=== Teste Feature-Extraktion ===")
        features_df = detector.extract_market_features()

        if not features_df.empty:
            logger.info(f"Features extrahiert: {len(features_df)} Zeilen, {len(features_df.columns)} Spalten")
            logger.info(f"Feature-Spalten: {list(features_df.columns)}")

            # Überprüfe die zuvor fehlenden Features
            if 'btc_ema_ratio' in features_df.columns:
                logger.info("Feature btc_ema_ratio ist vorhanden")
            if 'btc_macd_signal_ratio' in features_df.columns:
                logger.info("Feature btc_macd_signal_ratio ist vorhanden")
            if 'btc_mean_return' in features_df.columns:
                logger.info("Feature btc_mean_return ist vorhanden")
            if 'btc_rsi' in features_df.columns:
                logger.info("Feature btc_rsi ist vorhanden")
            if 'btc_volatility' in features_df.columns:
                logger.info("Feature btc_volatility ist vorhanden")

            # Überprüfe avg_altcoin_correlation
            if 'avg_altcoin_correlation' not in features_df.columns and 'btc_avg_altcoin_correlation' not in features_df.columns:
                logger.warning("Feature avg_altcoin_correlation fehlt (auch mit btc_-Präfix)")

            # Teste Modell-Training
            logger.info("=== Teste Modell-Training ===")
            success = detector.train_regime_model(features_df)

            if success:
                logger.info("Modell erfolgreich trainiert")
                logger.info(f"Anzahl der Regimes: {detector.n_regimes}")
                logger.info(f"Feature-Spalten: {detector.feature_columns[:10]}... (gekürzt)")
                logger.info(f"Regime-Labels: {detector.regime_labels}")
                logger.info(f"Anzahl Feature-Spalten: {len(detector.feature_columns)}")

                # Teste Regime-Vorhersage
                logger.info("=== Teste Regime-Vorhersage ===")

                # Nehme die letzten Daten für die Vorhersage
                last_features = features_df.iloc[-1:].copy()
                logger.info(f"Vorhersage mit Features vom {last_features.index[0]}")

                # Überprüfe Feature-Konsistenz
                logger.info("Feature-Spalten im Modell:")
                logger.info(f"{detector.feature_columns[:10]}... (gekürzt)")
                logger.info("Feature-Spalten in den Testdaten:")
                logger.info(f"{list(last_features.columns)[:10]}... (gekürzt)")

                # Identifiziere zusätzliche oder fehlende Features
                extra_cols = set(last_features.columns) - set(detector.feature_columns)
                if extra_cols:
                    logger.warning(f"Zusätzliche Features: {extra_cols}")

                # Vorhersage machen
                regime_id, regime_label = detector.predict_regime(last_features)
                logger.info(f"Regime-Vorhersage: ID={regime_id}, Label={regime_label}")

                # Teste Modell-Speicherung und -Ladung
                logger.info("=== Teste Modell-Speicherung und -Ladung ===")

                # Erstelle das models-Verzeichnis
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "market_regime_model.joblib")

                save_success = detector.save_model(model_path)

                if save_success:
                    logger.info(f"Modell erfolgreich gespeichert unter {model_path}")

                    # Neuen Detektor erstellen und Modell laden
                    new_detector = MarketRegimeDetector()
                    load_success = new_detector.load_model(model_path)

                    if load_success:
                        logger.info("Modell erfolgreich geladen")

                        # Vorhersage mit geladenem Modell
                        new_regime_id, new_regime_label = new_detector.predict_regime(last_features)
                        logger.info(f"Neue Vorhersage: ID={new_regime_id}, Label={new_regime_label}")

                        if new_regime_id == regime_id:
                            logger.info("Vorhersagen stimmen überein - Modell-Speicherung/-Ladung erfolgreich")
                        else:
                            logger.error("Vorhersagen unterschiedlich - Modell-Speicherung/-Ladung fehlerhaft")
                    else:
                        logger.error("Modell-Ladung fehlgeschlagen")
                else:
                    logger.error("Modell-Speicherung fehlgeschlagen")
                    logger.error("Modell-Speicherung/-Ladung fehlgeschlagen")

                # Teste Regime-Analyse über viele Zeitpunkte
                logger.info("=== Teste Regime-Analyse ===")
                logger.info("Führe manuelle Analyse durch...")

                # Extrahiere Features für historische Daten
                historical_features = detector.extract_market_features()
                logger.info(f"Features erfolgreich extrahiert: {len(historical_features)} Zeilen")

                # Mache Vorhersagen für die ersten 100 Zeitpunkte (oder weniger)
                num_predictions = min(100, len(historical_features))
                for i in range(num_predictions):
                    single_features = historical_features.iloc[i:i + 1]
                    detector.predict_regime(single_features)

                # Zähle die Regime-Verteilung
                results_df = pd.DataFrame(index=historical_features.index)
                results_df['regime'] = None  # Initialisieren der Spalte

                for idx, row in historical_features.iterrows():
                    # Features-DataFrame für einen einzelnen Zeitpunkt
                    single_features = pd.DataFrame([row])

                    # Regime vorhersagen
                    regime, _ = detector.predict_regime(single_features)
                    results_df.loc[idx, 'regime'] = regime

                # Regime-Verteilung
                regime_counts = results_df['regime'].value_counts().to_dict()
                logger.info(f"Erfolgreich {len(results_df)} Regime-Vorhersagen gemacht")
                logger.info(f"Regime-Verteilung: {regime_counts}")
            else:
                logger.error("Modell-Training fehlgeschlagen")
        else:
            logger.error("Keine Features extrahiert")
    else:
        logger.error("Keine geeignete Testdatei gefunden")

    logger.info("=== Debug-Prozess abgeschlossen ===")
    logger.info("Bitte überprüfe market_regime_debug.log für Details.")


if __name__ == "__main__":
    main()