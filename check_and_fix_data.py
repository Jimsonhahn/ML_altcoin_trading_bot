# check_and_fix_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def check_csv_file(file_path):
    """Überprüft eine CSV-Datei und zeigt ihre Struktur"""
    print(f"Überprüfe Datei: {file_path}")

    try:
        # Datei laden
        df = pd.read_csv(file_path)

        # Grundlegende Informationen
        print(f"Spalten: {df.columns.tolist()}")
        print(f"Anzahl Zeilen: {len(df)}")

        # Beispielzeile
        print("\nErste Zeile:")
        print(df.iloc[0])

        # Datentypen
        print("\nDatentypen:")
        print(df.dtypes)

        return df
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return None


def create_compatible_data():
    """Erstellt garantiert kompatible Daten im richtigen Format"""
    print("Erstelle kompatible Testdaten...")

    # Zielverzeichnis
    output_dir = "data/market_data/binance"
    os.makedirs(output_dir, exist_ok=True)

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

        # Daten für jeden Tag generieren
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            timestamp = int(current_date.timestamp() * 1000)  # Millisekunden seit der Epoche

            # Einfache Preisbewegung (Sinus-Welle mit Trend)
            price_factor = 1 + 0.1 * np.sin(i / 30) + i / days * 0.2  # +20% Trend über den Zeitraum
            current_price = price * price_factor

            # Tägliche Schwankung
            daily_factor = 1 + np.random.normal(0, 0.02)  # 2% Volatilität
            open_price = current_price * daily_factor
            close_price = current_price * daily_factor
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))

            # Volumen
            volume = price * 100 * (1 + np.random.normal(0, 0.3))

            # Zeile im Binance-Format erstellen
            row = {
                'timestamp': timestamp,  # Unix-Zeit in ms
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'close_time': timestamp + 86400000 - 1,  # Ende des Tages
                'quote_asset_volume': volume * close_price,
                'number_of_trades': int(np.random.uniform(1000, 5000)),
                'taker_buy_base_asset_volume': volume * 0.5,
                'taker_buy_quote_asset_volume': volume * 0.5 * close_price,
                'ignore': 0
            }

            data_rows.append(row)

        # DataFrame erstellen
        df = pd.DataFrame(data_rows)

        # In CSV speichern
        output_file = f"{output_dir}/{base}_{quote}_1d.csv"
        df.to_csv(output_file, index=False)
        print(f"Daten für {symbol} gespeichert in {output_file}")

        # Erste Zeile zur Überprüfung anzeigen
        print(f"Erste Zeile für {symbol}:")
        print(df.iloc[0])
        print(f"Datentypen für {symbol}:")
        print(df.dtypes)
        print("\n")


def main():
    """Hauptfunktion"""
    print("==== CSV-CHECKER UND FIXER ====")

    # Kompatible Daten erstellen
    create_compatible_data()

    # Überprüfen der ersten Datei
    first_file = "data/market_data/binance/BTC_USDT_1d.csv"
    if os.path.exists(first_file):
        df = check_csv_file(first_file)

        if df is not None:
            print("\nDaten erfolgreich erstellt und überprüft!")
            print("Sie können jetzt fix_ml_clusters.py ausführen")

    print("==== CSV-CHECKER UND FIXER BEENDET ====")


if __name__ == "__main__":
    main()