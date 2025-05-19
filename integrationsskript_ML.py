#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrationstests für ML-Komponenten.
Testet den vollständigen Workflow von Datenladen, Feature-Extraktion, Training und Vorhersage.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Pfad-Konfiguration für Importe
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import der zu testenden Komponenten
from config.settings import Settings
from ml_components.market_regime import MarketRegimeDetector
from ml_components.asset_clusters import AssetClusterAnalyzer
from strategies.ml_strategy import MLStrategy
from data_sources.data_sources.data_manager import DataManager

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("integration_test")


def test_market_regime_detector(settings):
    """Test des MarketRegimeDetectors"""
    logger.info("=== Testing MarketRegimeDetector ===")

    # Instanziieren des DataManagers
    data_manager = DataManager(settings)

    # Instanziieren des MarketRegimeDetectors
    detector = MarketRegimeDetector(
        data_dir=settings.get('ml.data_dir', 'data/market_data'),
        n_regimes=settings.get('ml.n_regimes', 5)
    )

    # Testparameter
    symbols = settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT"])
    timeframe = settings.get('timeframes.analysis', '1d')

    # Daten laden
    logger.info(f"Lade Daten für {symbols}")
    if detector.load_market_data(
            symbols=symbols,
            data_manager=data_manager,
            timeframe=timeframe
    ):
        logger.info("Daten erfolgreich geladen")

        # Features extrahieren
        logger.info("Extrahiere Features")
        features_df = detector.extract_market_features()

        if not features_df.empty:
            logger.info(f"Features extrahiert: {len(features_df)} Datenpunkte, {len(features_df.columns)} Features")

            # Modell trainieren
            logger.info("Trainiere Modell")
            if detector.train_regime_model(features_df):
                logger.info("Modell erfolgreich trainiert")

                # Vorhersage testen
                latest_features = features_df.iloc[-1:].copy()
                logger.info("Teste Vorhersage mit den neuesten Features")
                regime = detector.predict_regime(latest_features)

                if isinstance(regime, tuple):
                    logger.info(f"Aktuelles Regime: {regime[0]}, Label: {regime[1]}")
                else:
                    logger.info(f"Aktuelles Regime: {regime}, Label: {detector.regime_labels.get(regime, 'Unbekannt')}")

                # Speichern und Laden testen
                model_path = os.path.join(settings.get('ml.models_dir', 'data/ml_models'), "test_regime_model.pkl")
                logger.info(f"Speichere Modell in {model_path}")

                if detector.save_model(model_path):
                    logger.info("Modell erfolgreich gespeichert")

                    # Neuen Detector erstellen und Modell laden
                    new_detector = MarketRegimeDetector(
                        data_dir=settings.get('ml.data_dir', 'data/market_data'),
                        n_regimes=settings.get('ml.n_regimes', 5)
                    )

                    if new_detector.load_model(model_path):
                        logger.info("Modell erfolgreich geladen")

                        # Neue Vorhersage mit geladenem Modell testen
                        new_regime = new_detector.predict_regime(latest_features)

                        if isinstance(new_regime, tuple):
                            logger.info(f"Neue Vorhersage: {new_regime[0]}, Label: {new_regime[1]}")
                        else:
                            logger.info(
                                f"Neue Vorhersage: {new_regime}, Label: {new_detector.regime_labels.get(new_regime, 'Unbekannt')}")

                        # Vergleichen der Ergebnisse
                        if isinstance(regime, tuple) and isinstance(new_regime, tuple):
                            if regime[0] == new_regime[0]:
                                logger.info("BESTANDEN: Beide Vorhersagen stimmen überein")
                                return True
                            else:
                                logger.error(
                                    f"FEHLGESCHLAGEN: Vorhersagen stimmen nicht überein: {regime[0]} vs {new_regime[0]}")
                        else:
                            if regime == new_regime:
                                logger.info("BESTANDEN: Beide Vorhersagen stimmen überein")
                                return True
                            else:
                                logger.error(
                                    f"FEHLGESCHLAGEN: Vorhersagen stimmen nicht überein: {regime} vs {new_regime}")
                    else:
                        logger.error("Konnte Modell nicht laden")
                else:
                    logger.error("Konnte Modell nicht speichern")
            else:
                logger.error("Modelltraining fehlgeschlagen")
        else:
            logger.error("Feature-Extraktion lieferte leeres DataFrame")
    else:
        logger.error("Konnte keine Daten laden")

    return False


def test_asset_cluster_analyzer(settings):
    """Test des AssetClusterAnalyzers"""
    logger.info("=== Testing AssetClusterAnalyzer ===")

    # Instanziieren des DataManagers
    data_manager = DataManager(settings)

    # Instanziieren des AssetClusterAnalyzers
    analyzer = AssetClusterAnalyzer(
        data_dir=settings.get('ml.data_dir', 'data/market_data')
    )

    # Testparameter
    symbols = settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe = settings.get('timeframes.analysis', '1d')

    # Daten laden
    logger.info(f"Lade Daten für {symbols}")
    if analyzer.load_market_data(
            symbols=symbols,
            data_manager=data_manager,
            timeframe=timeframe
    ):
        logger.info("Daten erfolgreich geladen")

        # Korrelationsmatrix berechnen
        logger.info("Berechne Korrelationsmatrix")
        corr_matrix = analyzer.calculate_correlation_matrix()

        if not corr_matrix.empty:
            logger.info(f"Korrelationsmatrix: {corr_matrix.shape}")

            # Features extrahieren
            logger.info("Extrahiere Asset-Features")
            analyzer.extract_asset_features()

            if analyzer.feature_data is not None and not analyzer.feature_data.empty:
                logger.info(
                    f"Features extrahiert: {len(analyzer.feature_data)} Assets, {len(analyzer.feature_data.columns)} Features")

                # Clustering durchführen
                logger.info("Führe Clustering durch")
                clusters = analyzer.run_clustering()

                if not clusters.empty:
                    logger.info(
                        f"Clustering erfolgreich: {len(clusters)} Assets in {len(clusters['cluster'].unique())} Clustern")

                    # Portfolioempfehlung testen
                    recommendations = analyzer.recommend_portfolio(n_assets=3)

                    if recommendations and 'assets' in recommendations:
                        logger.info(f"Portfolioempfehlung: {recommendations['assets']}")
                        logger.info("BESTANDEN: AssetClusterAnalyzer funktioniert")
                        return True
                    else:
                        logger.error("Konnte keine Portfolioempfehlung generieren")
                else:
                    logger.error("Clustering fehlgeschlagen")
            else:
                logger.error("Feature-Extraktion fehlgeschlagen")
        else:
            logger.error("Korrelationsmatrix-Berechnung fehlgeschlagen")
    else:
        logger.error("Konnte keine Daten laden")

    return False


def test_ml_strategy(settings):
    """Test der ML-Strategie"""
    logger.info("=== Testing MLStrategy ===")

    # Instanziieren des DataManagers
    data_manager = DataManager(settings)

    # Instanziieren der ML-Strategie
    ml_strategy = MLStrategy(settings)

    # Testparameter
    symbol = settings.get('trading_pairs', ["BTC/USDT"])[0]  # Erstes Symbol verwenden
    timeframe = settings.get('timeframes.analysis', '1d')

    # Daten über DataManager laden
    logger.info(f"Lade Daten für {symbol}")
    df = data_manager.get_historical_data(
        symbol=symbol,
        source=settings.get('data.source', 'binance'),
        timeframe=timeframe,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        use_cache=True
    )

    if not df.empty:
        logger.info(f"Daten geladen: {len(df)} Datenpunkte")

        # ML-Komponenten trainieren
        logger.info("Trainiere ML-Komponenten")
        ml_strategy.train_ml_components(
            symbols=settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT"]),
            data_manager=data_manager
        )

        # Signal generieren
        logger.info("Generiere Handelssignal")
        signal, signal_data = ml_strategy.generate_signal(df, symbol)

        logger.info(f"Signal: {signal}, Confidence: {signal_data.get('confidence', 0)}")
        logger.info(f"Grund: {signal_data.get('reason', 'unknown')}")

        if 'ml_enhanced' in signal_data and signal_data['ml_enhanced']:
            logger.info("Signal wurde mit ML verbessert")

        if signal in ["BUY", "SELL", "HOLD"]:
            logger.info("BESTANDEN: MLStrategy funktioniert")
            return True
        else:
            logger.error(f"Ungültiges Signal: {signal}")
    else:
        logger.error(f"Konnte keine Daten für {symbol} laden")

    return False


def create_test_dirs(settings):
    """Erstellt die notwendigen Verzeichnisse für Tests"""
    dirs = [
        settings.get('ml.data_dir', 'data/market_data'),
        settings.get('ml.models_dir', 'data/ml_models'),
        settings.get('ml.output_dir', 'data/ml_analysis')
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Verzeichnis erstellt/überprüft: {directory}")


def main():
    """Hauptfunktion für die Tests"""
    logger.info("=== ML-Komponenten Integrationstests ===")

    # Settings laden
    settings = Settings()

    # Verzeichnisse erstellen
    create_test_dirs(settings)

    # Wir aktivieren ML in den Einstellungen
    settings.set('ml.enabled', True)

    # Tests durchführen
    regime_test = test_market_regime_detector(settings)
    cluster_test = test_asset_cluster_analyzer(settings)
    strategy_test = test_ml_strategy(settings)

    # Zusammenfassung
    logger.info("\n=== Testergebnisse ===")
    logger.info(f"MarketRegimeDetector: {'BESTANDEN' if regime_test else 'FEHLGESCHLAGEN'}")
    logger.info(f"AssetClusterAnalyzer: {'BESTANDEN' if cluster_test else 'FEHLGESCHLAGEN'}")
    logger.info(f"MLStrategy: {'BESTANDEN' if strategy_test else 'FEHLGESCHLAGEN'}")

    all_passed = regime_test and cluster_test and strategy_test
    logger.info(f"\nGesamtergebnis: {'ALLE TESTS BESTANDEN' if all_passed else 'EINIGE TESTS FEHLGESCHLAGEN'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())