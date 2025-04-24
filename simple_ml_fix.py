import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from config.settings import Settings
from core.trading_bot import TradingBot

# Logging einrichten
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Verzeichnisse erstellen
def setup_directories():
    required_dirs = [
        "data",
        "data/market_data",
        "data/market_data/binance",
        "data/ml_models",
        "data/ml_analysis",
        "data/backtest_results"
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Verzeichnis erstellt: {directory}")


# Einfache Testdaten erstellen
def create_simple_data():
    # Zielverzeichnis
    output_dir = "data/market_data/binance"

    # Trading-Paare
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    # Zeitraum: 2022-01-01 bis 2022-12-31 (täglich)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    days = (end_date - start_date).days + 1

    # Daten für jedes Symbol erstellen
    for symbol in symbols:
        base, quote = symbol.split('/')

        # Liste für Datenzeilen
        data_rows = []

        # Basispreis basierend auf dem Symbol
        if base == "BTC":
            price = 30000
        elif base == "ETH":
            price = 2000
        elif base == "SOL":
            price = 100
        elif base == "BNB":
            price = 300
        elif base == "XRP":
            price = 0.5
        else:
            price = 50

        # Seed für reproduzierbare Ergebnisse
        np.random.seed(42 + ord(base[0]))

        # Daten für jeden Tag generieren
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            timestamp = int(current_date.timestamp() * 1000)  # Millisekunden seit der Epoche

            # Zufällige Preisänderung mit Trend
            price_factor = 1 + 0.1 * np.sin(i / 30) + i / days * 0.2
            current_price = price * price_factor

            # Tageskerze
            open_price = current_price * (1 + np.random.normal(0, 0.01))
            close_price = current_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))

            # Volumen
            volume = price * 100 * (1 + np.random.normal(0, 0.3))

            # Datenzeile im Binance-Format
            row = [
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                timestamp + 86400000 - 1,
                volume * close_price,
                int(np.random.uniform(1000, 5000)),
                volume * 0.5,
                volume * 0.5 * close_price,
                0
            ]

            data_rows.append(row)

        # DataFrame erstellen
        df = pd.DataFrame(data_rows, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Datentypen explizit festlegen
        for col in ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume']:
            df[col] = pd.to_numeric(df[col])

        # In CSV speichern
        output_file = f"{output_dir}/{base}_{quote}_1d.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Daten für {symbol} gespeichert in {output_file}")


# Einfachen ML-Backtest ausführen (ohne komplexe ML-Komponenten)
def run_simple_backtest():
    # Einstellungen erstellen
    settings = Settings()

    # Grundlegende Einstellungen
    settings.set('backtest.start_date', '2022-01-01')
    settings.set('backtest.end_date', '2022-12-31')
    settings.set('backtest.initial_balance', 10000)
    settings.set('backtest.commission', 0.001)

    # Trading-Paare
    settings.set('trading_pairs', ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])

    # Risikomanagement mit liberaleren Einstellungen
    settings.set('risk.position_size', 0.2)  # 20% des Kapitals pro Trade
    settings.set('risk.stop_loss', 0.05)  # 5% Stop Loss
    settings.set('risk.take_profit', 0.1)  # 10% Take Profit
    settings.set('risk.max_open_positions', 3)
    settings.set('risk.min_confidence', 0.2)  # Niedrige Schwelle für Trades

    # Grundlegende Analyse-Einstellungen
    settings.set('timeframes.analysis', '1d')
    settings.set('data.source', 'binance')
    settings.set('data.use_cache', True)

    # Ausgabe-Einstellungen
    settings.set('backtest.create_plots', True)
    settings.set('backtest.export_results', True)
    settings.set('backtest.export_format', 'excel')
    settings.set('backtest.output_dir', 'simple_backtest')

    # Momentum-Strategie statt ML verwenden
    logger.info("Initialisiere Trading Bot mit Momentum-Strategie...")
    bot = TradingBot(mode="backtest", strategy_name="momentum", settings=settings)

    # Backtest ausführen
    logger.info("Starte Backtest...")
    results = bot.run_backtest()

    logger.info(f"Backtest abgeschlossen: Rendite: {results.get('total_return', 0):.2f}%")

    # Ergebnisse anzeigen
    print("\n" + "=" * 50)
    print(" EINFACHER BACKTEST ERGEBNISSE ")
    print("=" * 50)
    print(f"Zeitraum: {settings.get('backtest.start_date')} bis {settings.get('backtest.end_date')}")
    print(f"Anfangskapital: ${settings.get('backtest.initial_balance')}")
    print(f"Trading-Paare: {', '.join(settings.get('trading_pairs'))}")
    print(f"Strategie: Momentum (keine ML)")
    print("-" * 50)
    print(f"Gesamtrendite: {results.get('total_return', 0):.2f}%")
    print(f"Endkapital: ${results.get('final_balance', 0):.2f}")
    print(f"Anzahl Trades: {results.get('total_trades', 0)}")

    # Detaillierte Statistiken
    if 'statistics' in results:
        stats = results['statistics']
        print("\nPerformance-Metriken:")
        print(f"Win Rate: {stats.get('win_rate', 0):.2f}%")
        print(f"Durchschnittl. Gewinn: {stats.get('avg_profit', 0):.2f}%")
        print(f"Durchschnittl. Verlust: {stats.get('avg_loss', 0):.2f}%")
        print(f"Profit Faktor: {stats.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {abs(stats.get('max_drawdown', 0)):.2f}%")

    print("=" * 50)
    return results


def main():
    logger.info("==== EINFACHER ML-BACKTEST-FIXER ====")

    # Verzeichnisse einrichten
    setup_directories()

    # Testdaten erstellen
    logger.info("Erstelle Testdaten...")
    create_simple_data()

    # Einfachen Backtest ausführen
    logger.info("Starte einfachen Backtest (ohne ML)...")
    run_simple_backtest()

    logger.info("==== EINFACHER ML-BACKTEST-FIXER ABGESCHLOSSEN ====")
    logger.info("Falls dies funktioniert hat, können Sie später zu ML-basierten Backtests zurückkehren")


if __name__ == "__main__":
    main()