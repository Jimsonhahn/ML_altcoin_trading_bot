# fix_ml_clusters.py
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import Settings
from ml_components import initialize_ml

# Logging einrichten
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verzeichnisse erstellen
required_dirs = [
    "data",
    "data/market_data",
    "data/market_data/binance",
    "data/ml_models",
    "data/ml_analysis"
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)


# Funktion zum Erstellen von Testdaten
def create_test_data(symbols):
    """Erstellt synthetische Testdaten für die angegebenen Symbole"""
    logger.info(f"Erstelle Testdaten für {len(symbols)} Symbole...")

    # Zeitraum: 1 Jahr tägliche Daten
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

    for symbol in symbols:
        base, quote = symbol.split('/')

        # Zufällige Preisdaten generieren
        np.random.seed(42 + ord(base[0]))  # Verschiedene Seeds für verschiedene Symbole

        # Basispreis je nach Symbol
        if base == 'BTC':
            base_price = 30000
        elif base == 'ETH':
            base_price = 2000
        elif base == 'SOL':
            base_price = 100
        elif base == 'BNB':
            base_price = 300
        elif base == 'XRP':
            base_price = 0.5
        else:
            base_price = 50

        # Zufällige Preisbewegung generieren
        price_changes = np.random.normal(0, 0.02, len(dates))  # 2% tägliche Volatilität
        cumulative_returns = np.cumsum(price_changes)
        prices = base_price * np.exp(cumulative_returns)

        # Dataframe erstellen
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'low': prices * (1 - np.random.normal(0, 0.01, len(dates))),
            'close': prices,
            'volume': base_price * 1000 * (1 + np.random.normal(0, 0.3, len(dates)))
        })

        # Sicherstellen, dass high immer höher als open/close und low immer tiefer ist
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        # Zusätzliche Spalten für das Binance-Format
        df['close_time'] = df['timestamp'] + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        df['quote_asset_volume'] = df['volume'] * df['close']
        df['number_of_trades'] = np.random.randint(1000, 5000, len(dates))
        df['taker_buy_base_asset_volume'] = df['volume'] * 0.5
        df['taker_buy_quote_asset_volume'] = df['taker_buy_base_asset_volume'] * df['close']
        df['ignore'] = 0

        # Zeitstempel in Unix-Zeit (Millisekunden) umwandeln
        df['timestamp_ms'] = df['timestamp'].astype(int) // 10 ** 6

        # In CSV speichern
        output_file = f"data/market_data/binance/{base}_{quote}_1d.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Testdaten für {symbol} in {output_file} gespeichert")

    return True


def main():
    """Hauptfunktion, die das ML-Clustering-Problem behebt"""
    logger.info("==== ASSET CLUSTER FIXER ====")

    # Trading-Paare definieren - ausreichend für Clustering
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    # Testdaten erstellen
    create_test_data(symbols)

    # Einstellungen erstellen
    settings = Settings()

    # ML-Einstellungen anpassen
    settings.set('data.market_data_dir', 'data/market_data')
    settings.set('ml.models_dir', 'data/ml_models')
    settings.set('ml.output_dir', 'data/ml_analysis')
    settings.set('ml.regime_count', 2)
    settings.set('ml.correlation_based_clustering', False)
    settings.set('trading_pairs', symbols)

    # ML-Komponenten initialisieren
    logger.info("Initialisiere ML-Komponenten...")
    ml_components = initialize_ml(
        settings=settings,
        data_dir=settings.get('data.market_data_dir'),
        models_dir=settings.get('ml.models_dir'),
        output_dir=settings.get('ml.output_dir')
    )

    if not ml_components:
        logger.error("ML-Komponenten konnten nicht initialisiert werden!")
        return

    logger.info("ML-Komponenten erfolgreich initialisiert")

    # Testen der Datenladung für Asset-Cluster
    logger.info("Teste Datenladung für Asset-Cluster...")
    data_loaded = ml_components.asset_cluster_analyzer.load_market_data(symbols=symbols)

    if not data_loaded:
        logger.error("Fehler beim Laden der Marktdaten für Asset-Cluster")
        return

    logger.info("Marktdaten erfolgreich geladen")

    # Testen des Clustering-Prozesses
    logger.info("Teste Clustering-Prozess...")

    # Korrelationsmatrix berechnen
    corr_matrix = ml_components.asset_cluster_analyzer.calculate_correlation_matrix()

    if corr_matrix.empty:
        logger.error("Korrelationsmatrix ist leer")
        return

    logger.info(f"Korrelationsmatrix erstellt: {corr_matrix.shape}")

    # Feature-Extraktion testen
    features = ml_components.asset_cluster_analyzer.extract_asset_features()

    if features.empty:
        logger.error("Feature-Extraktion fehlgeschlagen")
        return

    logger.info(f"Features extrahiert: {features.shape}")

    # Clustering mit expliziter Anzahl von Clustern durchführen
    n_clusters = min(2, len(symbols) - 1)  # Nicht mehr Cluster als Symbole-1
    logger.info(f"Führe Clustering mit {n_clusters} Clustern durch...")

    clusters = ml_components.asset_cluster_analyzer.run_clustering(n_clusters=n_clusters)

    if clusters.empty:
        logger.error("Clustering fehlgeschlagen")
        return

    logger.info(f"Clustering erfolgreich durchgeführt:")
    logger.info(f"Cluster-Zuordnungen: {clusters}")

    # Marktregime-Detektor testen
    logger.info("Teste Marktregime-Detektor...")

    regime_data_loaded = ml_components.market_regime_detector.load_market_data(symbols=symbols)

    if not regime_data_loaded:
        logger.error("Fehler beim Laden der Marktdaten für Regime-Detektor")
        return

    features_df = ml_components.market_regime_detector.extract_market_features()

    if features_df.empty:
        logger.error("Keine Features für Regime-Erkennung extrahiert")
        return

    logger.info(f"Regime-Features extrahiert: {features_df.shape}")

    # Modell trainieren
    training_success = ml_components.market_regime_detector.train_regime_model(features_df)

    if not training_success:
        logger.error("Training des Regime-Modells fehlgeschlagen")
        return

    logger.info("Regime-Modell erfolgreich trainiert")

    # Modelle speichern
    save_success = ml_components.save_models()

    if save_success:
        logger.info("ML-Modelle erfolgreich gespeichert")
    else:
        logger.warning("Fehler beim Speichern der ML-Modelle")

    logger.info("==== ASSET CLUSTER FIXER ABGESCHLOSSEN ====")
    logger.info("Sie können nun den Backtest mit den vorbereiteten Daten und Modellen ausführen")


if __name__ == "__main__":
    main()