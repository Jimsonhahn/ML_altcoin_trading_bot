#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Befehlszeilenwerkzeug zum Ausführen und Verwalten von Backtests.
"""

import os
import json
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from config.settings import Settings
from backtest_helper import run_backtest, BacktestRegistry, compare_backtests


def load_config_from_file(filepath):
    """Lädt eine Konfiguration aus einer JSON-Datei."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_from_config_file(filepath, test_name=None, tags=None):
    """Führt einen Backtest basierend auf einer Konfigurationsdatei aus."""
    config = load_config_from_file(filepath)

    # Strategie extrahieren
    strategy_name = config.get('strategy', 'momentum')

    # Parameter extrahieren
    params = {}

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

    # Zeitrahmen
    if 'timeframes' in config:
        for key, value in config['timeframes'].items():
            params[f'timeframes.{key}'] = value

    # Datenquellen
    if 'data' in config:
        for key, value in config['data'].items():
            params[f'data.{key}'] = value

    # Trading-Paare
    if 'trading_pairs' in config:
        params['trading_pairs'] = config['trading_pairs']

    # Testnamen generieren, falls nicht angegeben
    if not test_name:
        config_name = os.path.basename(filepath).replace('.json', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"{config_name}_{timestamp}"

    # Backtest ausführen
    return run_backtest(strategy_name, params, test_name, tags)


def list_available_configs():
    """Listet alle verfügbaren Konfigurationsdateien auf."""
    config_dir = 'config/examples/'
    configs = []

    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.json'):
                configs.append(file)

    return configs


def print_backtest_summary(top_n=10, metric='total_return'):
    """Druckt eine Zusammenfassung der besten Backtests."""
    registry = BacktestRegistry()
    backtests = registry.get_all_backtests()

    if not backtests:
        print("Keine Backtests gefunden.")
        return

    # Nach Metrik sortieren
    sorted_tests = sorted(
        backtests,
        key=lambda x: x['results'].get(metric, 0),
        reverse=True
    )[:top_n]

    print(f"\nTop {top_n} Backtests nach {metric}:\n")
    print(
        f"{'Test Name':<30} {'Strategie':<15} {'Datum':<25} {metric:<15} {'Win Rate':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 115)

    for test in sorted_tests:
        name = test['test_name']
        strategy = test['strategy']
        date = test['date'].split('T')[0]
        value = test['results'].get(metric, 0)
        win_rate = test['results'].get('win_rate', 0)
        sharpe = test['results'].get('sharpe_ratio', 0)
        max_dd = test['results'].get('max_drawdown', 0)

        print(f"{name:<30} {strategy:<15} {date:<25} {value:<15.2f} {win_rate:<10.2f} {sharpe:<10.2f} {max_dd:<10.2f}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Backtest Runner')

    # Hauptbefehle
    subparsers = parser.add_subparsers(dest='command', help='Befehl')

    # Hilfe, wenn kein Befehl angegeben wurde
    parser.set_defaults(command=None)

    # Befehl: run
    run_parser = subparsers.add_parser('run', help='Führt einen Backtest aus')
    run_parser.add_argument('--config', type=str, required=True, help='Pfad zur Konfigurationsdatei')
    run_parser.add_argument('--name', type=str, help='Name für den Backtest')
    run_parser.add_argument('--tags', type=str, help='Kommagetrennte Liste von Tags')
    run_parser.add_argument('--debug', action='store_true', help='Debug-Modus aktivieren')

    # Befehl: list
    list_parser = subparsers.add_parser('list', help='Listet verfügbare Ressourcen auf')
    list_parser.add_argument('--type', choices=['configs', 'backtests'], default='backtests',
                             help='Typ der aufzulistenden Ressourcen')
    list_parser.add_argument('--metric', type=str, default='total_return',
                             help='Metrik für die Sortierung (bei Backtests)')
    list_parser.add_argument('--top', type=int, default=10,
                             help='Anzahl der anzuzeigenden Top-Ergebnisse')

    # Befehl: compare
    compare_parser = subparsers.add_parser('compare', help='Vergleicht Backtests')
    compare_parser.add_argument('--tests', type=str, required=True,
                                help='Kommagetrennte Liste von Testnamen')
    compare_parser.add_argument('--metrics', type=str, default='total_return,sharpe_ratio,win_rate',
                                help='Kommagetrennte Liste von Metriken')
    compare_parser.add_argument('--output', type=str, help='Ausgabedatei für den Vergleich')

    # Befehl: export
    export_parser = subparsers.add_parser('export', help='Exportiert Backtest-Ergebnisse')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                               help='Exportformat')
    export_parser.add_argument('--output', type=str, help='Ausgabedatei')

    # Befehl: visualize
    viz_parser = subparsers.add_parser('visualize', help='Visualisiert Backtest-Ergebnisse')
    viz_parser.add_argument('--metric', type=str, default='total_return',
                            help='Zu visualisierende Metrik')
    viz_parser.add_argument('--top', type=int, default=10,
                            help='Anzahl der anzuzeigenden Top-Ergebnisse')
    viz_parser.add_argument('--output', type=str, help='Ausgabedatei')

    args = parser.parse_args()

    # Registrierung initialisieren
    registry = BacktestRegistry()

    # Befehl verarbeiten
    if args.command == 'run':
        # Tags parsen
        tags = args.tags.split(',') if args.tags else None

        print(f"Starte Backtest mit Konfiguration aus: {args.config}")

        # Debug-Informationen
        if args.debug:
            print(f"Debug-Modus aktiviert")
            print(f"Ausgabeverzeichnis wird sein: data/backtest_results/[testname]")

        results, output_dir = run_from_config_file(args.config, args.name, tags)

        print(f"\nBacktest abgeschlossen. Total return: {results.get('total_return', 0):.2f}%")
        print(f"Win rate: {results.get('statistics', {}).get('win_rate', 0):.2f}%")
        print(f"Sharpe ratio: {results.get('statistics', {}).get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {abs(results.get('statistics', {}).get('max_drawdown', 0)):.2f}%")
        print(f"Ergebnisse gespeichert in: {output_dir}")

        # Im Debug-Modus weitere Informationen anzeigen
        if args.debug:
            print(f"\nVerzeichnisinhalte:")
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                for file in files:
                    print(f"  - {file}")
            else:
                print(f"  Verzeichnis {output_dir} existiert nicht!")

    elif args.command == 'list':
        if args.type == 'configs':
            configs = list_available_configs()
            print("\nVerfügbare Konfigurationen:")
            for config in configs:
                print(f"  - {config}")
        else:  # backtests
            print_backtest_summary(args.top, args.metric)

    elif args.command == 'compare':
        # Testnamen parsen
        test_names = args.tests.split(',')
        metrics = args.metrics.split(',')

        # Vergleich durchführen
        comparison = compare_backtests(test_names, metrics)

        if comparison is None or comparison.empty:
            print("Keine Backtests zum Vergleichen gefunden.")
            return

        # Ergebnisse anzeigen
        print("\nVergleich der Backtests:\n")
        print(comparison.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        # Optional in Datei exportieren
        if args.output:
            output_file = args.output
            # Standardpfad in data/backtest_results/ wenn kein absoluter Pfad
            if not os.path.isabs(output_file):
                output_file = os.path.join("data/backtest_results", output_file)

            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            comparison.to_csv(output_file, index=False)
            print(f"\nVergleich exportiert nach: {output_file}")

    elif args.command == 'export':
        # Standardausgabedatei
        if args.output:
            output_file = args.output
        else:
            output_file = f"data/backtest_results/backtest_summary_{datetime.now().strftime('%Y%m%d')}.{args.format}"

        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if args.format == 'csv':
            filepath = registry.export_to_csv(output_file)
            print(f"Backtest-Zusammenfassung exportiert nach: {filepath}")
        else:  # json
            with open(output_file, 'w') as f:
                json.dump(registry.registry, f, indent=4)
            print(f"Backtest-Registry exportiert nach: {output_file}")

    elif args.command == 'visualize':
        # Standardausgabedatei
        if args.output:
            output_file = args.output
        else:
            output_file = f"data/visualizations/backtest_comparison_{datetime.now().strftime('%Y%m%d')}.png"

        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        viz_path = registry.visualize_results(metric=args.metric, top_n=args.top, save_path=output_file)

        if viz_path:
            print(f"Visualisierung gespeichert nach: {viz_path}")
        else:
            print("Fehler bei der Erstellung der Visualisierung.")

    else:
        # Wenn kein Befehl angegeben wurde, Hilfe anzeigen
        parser.print_help()


if __name__ == "__main__":
    main()