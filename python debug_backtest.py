#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug-Skript für Backtest-Funktionalität
"""

import os
import json
import sys
import traceback
from datetime import datetime


def check_directories():
    """Überprüft und erstellt die benötigten Verzeichnisse"""
    dirs_to_check = [
        "data",
        "data/backtest_results",
        "data/visualizations",
        "data/market_data",
        "data/states"
    ]

    for directory in dirs_to_check:
        exists = os.path.exists(directory)
        is_dir = os.path.isdir(directory) if exists else False

        print(
            f"Verzeichnis {directory}: {'Existiert' if exists else 'Fehlt'} {'(ist ein Verzeichnis)' if is_dir else ''}")

        if not exists:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  - Verzeichnis {directory} erstellt")
            except Exception as e:
                print(f"  - FEHLER beim Erstellen von {directory}: {e}")


def check_config_file(config_path):
    """Überprüft, ob die Konfigurationsdatei existiert und gültig ist"""
    print(f"\nKonfigurationsdatei: {config_path}")

    if not os.path.exists(config_path):
        print(f"  - FEHLER: Datei existiert nicht")
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  - JSON gültig, enthält {len(config)} Schlüssel: {', '.join(config.keys())}")
        return True
    except json.JSONDecodeError as e:
        print(f"  - FEHLER: Ungültiges JSON: {e}")
        return False
    except Exception as e:
        print(f"  - FEHLER beim Lesen: {e}")
        return False


def test_write_files():
    """Testet das Schreiben von Dateien in die Verzeichnisse"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dirs = [
        "data/backtest_results",
        "data/visualizations",
        "data/states"
    ]

    print("\nTeste Schreibzugriff:")

    for directory in test_dirs:
        test_file = os.path.join(directory, f"test_{timestamp}.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test file created at {datetime.now().isoformat()}")
            print(f"  - Erfolgreich in {test_file} geschrieben")

            # Datei wieder löschen
            os.remove(test_file)
            print(f"  - Datei {test_file} gelöscht")
        except Exception as e:
            print(f"  - FEHLER beim Schreiben in {test_file}: {e}")


def check_imports():
    """Überprüft die wichtigsten Importe für den Backtest"""
    print("\nTeste kritische Importe:")

    imports_to_check = [
        ("backtest_helper", "Backtest-Hilfsfunktionen"),
        ("config.settings", "Einstellungsmodul"),
        ("core.trading_bot", "Trading-Bot-Kern"),
        ("core.data_sources", "Datenquellen-Modul")
    ]

    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            print(f"  - {module_name} ({description}): OK")
        except ImportError as e:
            print(f"  - {module_name} ({description}): FEHLER - {e}")
        except Exception as e:
            print(f"  - {module_name} ({description}): UNERWARTETER FEHLER - {e}")
            traceback.print_exc()


def check_registry_file():
    """Überprüft die Registry-Datei"""
    registry_path = "data/backtest_registry.json"
    print(f"\nBacktest-Registry: {registry_path}")

    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            print(f"  - Datei existiert und enthält {len(registry.get('backtests', []))} Backtests")
        except json.JSONDecodeError:
            print(f"  - Datei existiert, enthält aber ungültiges JSON")
        except Exception as e:
            print(f"  - Fehler beim Lesen: {e}")
    else:
        print(f"  - Datei existiert nicht, wird wahrscheinlich beim ersten Backtest erstellt")
        # Registry-Datei mit leerer Struktur erstellen
        try:
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            with open(registry_path, 'w') as f:
                json.dump({"backtests": [], "last_update": datetime.now().isoformat()}, f, indent=2)
            print(f"  - Leere Registry-Datei erstellt")
        except Exception as e:
            print(f"  - FEHLER beim Erstellen der Registry: {e}")


def main():
    print("\n=== BACKTEST DEBUG-TOOL ===\n")
    print(f"Aktuelles Verzeichnis: {os.getcwd()}")
    print(f"Python-Version: {sys.version}")
    print(f"Datum und Zeit: {datetime.now().isoformat()}")

    # Verzeichnisse prüfen
    check_directories()

    # Konfigurationsdateien prüfen
    config_path = "config/examples/momentum_standard.json"
    check_config_file(config_path)

    # Schreibzugriff testen
    test_write_files()

    # Importe prüfen
    check_imports()

    # Registry prüfen
    check_registry_file()

    print("\n=== DEBUG ABGESCHLOSSEN ===\n")


if __name__ == "__main__":
    main()