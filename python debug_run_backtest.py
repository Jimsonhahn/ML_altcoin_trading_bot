#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug-Wrapper für einen einzelnen Backtest.
Führt jeden Schritt des Backtests mit detaillierter Ausgabe aus.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from config.settings import Settings
from backtest_helper import run_backtest


def debug_backtest():
    # Konfigurationsdatei lesen
    config_path = "config/examples/momentum_standard.json"
    print(f"1. Lade Konfiguration aus {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   ✓ Konfiguration geladen: {len(config)} Einstellungen gefunden")
    except Exception as e:
        print(f"   ✗ Fehler beim Laden der Konfiguration: {e}")
        return

    # Parameter extrahieren
    print("2. Extrahiere Parameter aus der Konfiguration")
    params = {}

    # Strategie
    strategy_name = config.get('strategy', 'momentum')
    print(f"   ✓ Strategie: {strategy_name}")

    # Backtest-Parameter
    if 'backtest' in config:
        for key, value in config['backtest'].items():
            params[f'backtest.{key}'] = value

    # Risikomanagement-Parameter
    if 'risk' in config:
        for key, value in config['risk'].items():
            params[f'risk.{key}'] = value

    # Strategie-spezifische Parameter
    for strategy_type in ['momentum', 'mean_reversion', 'ml']:
        if strategy_type in config:
            for key, value in config[strategy_type].items():
                params[f'{strategy_type}.{key}'] = value

    # Trading-Paare
    if 'trading_pairs' in config:
        params['trading_pairs'] = config['trading_pairs']

    print(f"   ✓ {len(params)} Parameter extrahiert")

    # Testnamen generieren
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"debug_test_{timestamp}"

    print(f"3. Erstelle Ausgabeverzeichnis")
    output_dir = os.path.join("data/backtest_results", test_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   ✓ Verzeichnis erstellt: {output_dir}")

        # Einfache Testdatei schreiben um Schreibzugriff zu prüfen
        with open(os.path.join(output_dir, "debug_info.json"), "w") as f:
            debug_info = {
                "timestamp": timestamp,
                "test_name": test_name,
                "strategy": strategy_name,
                "params_count": len(params)
            }
            json.dump(debug_info, f, indent=2)
        print(f"   ✓ Testdatei in {output_dir}/debug_info.json geschrieben")
    except Exception as e:
        print(f"   ✗ Fehler beim Erstellen des Verzeichnisses: {e}")
        return

    # Einstellungen initialisieren
    print("4. Initialisiere Settings-Objekt")
    try:
        settings = Settings()
        for key, value in params.items():
            settings.set(key, value)
        print(f"   ✓ Einstellungen initialisiert und Parameter gesetzt")
    except Exception as e:
        print(f"   ✗ Fehler beim Initialisieren der Einstellungen: {e}")
        traceback.print_exc()
        return

    # Backtest ausführen
    print("\n5. Starte den Backtest")
    print(f"   Strategy: {strategy_name}")
    print(f"   Test Name: {test_name}")
    print(f"   Output Dir: {output_dir}")

    try:
        print("\n=== BACKTEST START ===\n")
        results, actual_output_dir = run_backtest(
            strategy_name=strategy_name,
            params=params,
            test_name=test_name,
            tags=["debug"]
        )
        print("\n=== BACKTEST ENDE ===\n")

        print(f"6. Backtest abgeschlossen")
        print(f"   ✓ Ergebnisse erhalten: {type(results)}")
        print(f"   ✓ Ausgabeverzeichnis: {actual_output_dir}")

        # Überprüfe, ob das Verzeichnis existiert und Dateien enthält
        if os.path.exists(actual_output_dir):
            files = os.listdir(actual_output_dir)
            print(f"   ✓ Verzeichnis existiert und enthält {len(files)} Dateien:")
            for file in files:
                file_path = os.path.join(actual_output_dir, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "Verzeichnis"
                print(f"     - {file} ({file_size} bytes)")
        else:
            print(f"   ✗ Ausgabeverzeichnis existiert nicht: {actual_output_dir}")

        # Überprüfen, ob die Registry-Datei aktualisiert wurde
        registry_path = "data/backtest_registry.json"
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                backtests = registry.get("backtests", [])

                if any(b.get("test_name") == test_name for b in backtests):
                    print(f"   ✓ Backtest wurde zur Registry hinzugefügt")
                else:
                    print(f"   ✗ Backtest wurde NICHT zur Registry hinzugefügt")
            except Exception as e:
                print(f"   ✗ Fehler beim Lesen der Registry: {e}")

    except Exception as e:
        print(f"   ✗ Fehler beim Ausführen des Backtests: {e}")
        print("\nDetaillierter Traceback:")
        traceback.print_exc()
        return

    print("\nDEBUG BACKTEST ABGESCHLOSSEN")


if __name__ == "__main__":
    debug_backtest()