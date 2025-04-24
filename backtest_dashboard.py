#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard für Backtests - Zeigt Zusammenfassung und Vergleiche von Backtests.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse


def load_registry(registry_path="data/backtest_registry.json"):
    """Lädt die Backtest-Registry."""
    if not os.path.exists(registry_path):
        print(f"Registry-Datei nicht gefunden: {registry_path}")
        return None

    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        return registry
    except Exception as e:
        print(f"Fehler beim Laden der Registry: {e}")
        return None


def create_top_backtests_table(registry, metric='total_return', n=10):
    """Erstellt eine HTML-Tabelle mit den Top-N Backtests nach einer Metrik."""
    if not registry or 'backtests' not in registry or not registry['backtests']:
        return "<p>Keine Backtests in der Registry gefunden.</p>"

    # Nach Metrik sortieren
    sorted_tests = sorted(
        registry['backtests'],
        key=lambda x: x['results'].get(metric, 0),
        reverse=True
    )[:n]

    # HTML-Tabelle erstellen
    html = """
    <table class="table table-striped">
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Strategie</th>
                <th>Datum</th>
                <th>{metric}</th>
                <th>Win Rate</th>
                <th>Sharpe</th>
                <th>Max DD</th>
                <th>Aktionen</th>
            </tr>
        </thead>
        <tbody>
    """.format(metric=metric.replace('_', ' ').title())

    for test in sorted_tests:
        date = test['date'].split('T')[0] if 'T' in test.get('date', '') else test.get('date', '')
        result_path = test.get('result_path', '')

        html += """
        <tr>
            <td>{name}</td>
            <td>{strategy}</td>
            <td>{date}</td>
            <td>{value:.2f}</td>
            <td>{win_rate:.2f}%</td>
            <td>{sharpe:.2f}</td>
            <td>{max_dd:.2f}%</td>
            <td>
                <a href="javascript:void(0)" onclick="viewBacktest('{path}')" class="btn btn-sm btn-primary">Details</a>
            </td>
        </tr>
        """.format(
            name=test.get('test_name', ''),
            strategy=test.get('strategy', ''),
            date=date,
            value=test['results'].get(metric, 0),
            win_rate=test['results'].get('win_rate', 0),
            sharpe=test['results'].get('sharpe_ratio', 0),
            max_dd=abs(test['results'].get('max_drawdown', 0)),
            path=result_path
        )

    html += """
        </tbody>
    </table>
    """

    return html


def create_strategy_comparison_chart(registry):
    """Erstellt einen Vergleichschart der Strategien."""
    if not registry or 'backtests' not in registry or not registry['backtests']:
        return None

    # Strategien gruppieren
    strategies = {}
    for test in registry['backtests']:
        strategy = test.get('strategy', 'Unknown')
        if strategy not in strategies:
            strategies[strategy] = []

        strategies[strategy].append({
            'total_return': test['results'].get('total_return', 0),
            'sharpe_ratio': test['results'].get('sharpe_ratio', 0),
            'win_rate': test['results'].get('win_rate', 0),
            'max_drawdown': abs(test['results'].get('max_drawdown', 0))
        })

    # Durchschnittswerte berechnen
    avg_metrics = {}
    for strategy, tests in strategies.items():
        if not tests:
            continue

        avg_metrics[strategy] = {
            'total_return': np.mean([t['total_return'] for t in tests]),
            'sharpe_ratio': np.mean([t['sharpe_ratio'] for t in tests]),
            'win_rate': np.mean([t['win_rate'] for t in tests]),
            'max_drawdown': np.mean([t['max_drawdown'] for t in tests]),
            'count': len(tests)
        }

    # Chart erstellen
    plt.figure(figsize=(12, 10))

    # 1. Total Return
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    strategies_list = list(avg_metrics.keys())
    returns = [avg_metrics[s]['total_return'] for s in strategies_list]

    bars = ax1.bar(strategies_list, returns, color='skyblue')
    ax1.set_title('Durchschnittliche Gesamtrendite')
    ax1.set_ylabel('Rendite (%)')

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 2. Sharpe Ratio
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sharpes = [avg_metrics[s]['sharpe_ratio'] for s in strategies_list]

    bars = ax2.bar(strategies_list, sharpes, color='lightgreen')
    ax2.set_title('Durchschnittliche Sharpe Ratio')

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 3. Win Rate
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    win_rates = [avg_metrics[s]['win_rate'] for s in strategies_list]

    bars = ax3.bar(strategies_list, win_rates, color='orange')
    ax3.set_title('Durchschnittliche Win Rate')
    ax3.set_ylabel('Win Rate (%)')

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 4. Max Drawdown
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    drawdowns = [avg_metrics[s]['max_drawdown'] for s in strategies_list]

    bars = ax4.bar(strategies_list, drawdowns, color='salmon')
    ax4.set_title('Durchschnittlicher Max Drawdown')
    ax4.set_ylabel('Drawdown (%)')

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()

    # Speichern
    os.makedirs('data/visualizations', exist_ok=True)
    output_file = 'data/visualizations/strategy_comparison.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def create_dashboard_html(registry):
    """Erstellt ein Dashboard-HTML für alle Backtests."""
    if not registry:
        return None

    # Strategievergleich erstellen
    strategy_chart = create_strategy_comparison_chart(registry)

    # Dashboardverzeichnis erstellen
    dashboard_dir = 'data/dashboard'
    os.makedirs(dashboard_dir, exist_ok=True)

    # Statistiken berechnen
    total_backtests = len(registry.get('backtests', []))
    unique_strategies = set(test.get('strategy', '') for test in registry.get('backtests', []))
    latest_update = registry.get('last_update', '').split('T')[0] if 'T' in registry.get('last_update',
                                                                                         '') else registry.get(
        'last_update', '')

    if not latest_update:
        latest_update = datetime.now().strftime('%Y-%m-%d')

    # Top-Backtests-Tabellen erstellen
    top_return_table = create_top_backtests_table(registry, 'total_return', 10)
    top_sharpe_table = create_top_backtests_table(registry, 'sharpe_ratio', 10)

    # HTML erstellen
    html_path = os.path.join(dashboard_dir, 'index.html')

    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding-top: 20px; padding-bottom: 50px; }}
        .metric-card {{ 
            text-align: center;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-value {{ font-size: 2rem; font-weight: bold; }}
        .metric-title {{ font-size: 1.1rem; color: #666; }}
        .chart-container {{ margin-top: 30px; margin-bottom: 30px; }}
        .strategy-chart {{ 
            max-width: 100%; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}
        .nav-tabs {{ margin-bottom: 20px; }}
        .dashboard-header {{ 
            background-color: #f8f9fa; 
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .last-update {{ color: #666; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="dashboard-header">
            <div class="row">
                <div class="col-md-6">
                    <h1>Backtest Dashboard</h1>
                    <p class="last-update">Letztes Update: {latest_update}</p>
                </div>
                <div class="col-md-6 text-end">
                    <button class="btn btn-primary" onclick="refreshDashboard()">Dashboard aktualisieren</button>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="metric-card bg-light">
                    <div class="metric-value">{total_backtests}</div>
                    <div class="metric-title">Gesamtanzahl Backtests</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card bg-light">
                    <div class="metric-value">{len(unique_strategies)}</div>
                    <div class="metric-title">Einzigartige Strategien</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card bg-light">
                    <div class="metric-value">
                        {max((test['results'].get('total_return', 0) for test in registry.get('backtests', [])), default=0):.2f}%
                    </div>
                    <div class="metric-title">Höchste Rendite</div>
                </div>
            </div>
        </div>

        <div class="chart-container text-center">
            <h3>Strategie-Vergleich</h3>
            <img src="../visualizations/strategy_comparison.png" alt="Strategie-Vergleich" class="strategy-chart">
        </div>

        <ul class="nav nav-tabs" id="backtest-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="top-return-tab" data-bs-toggle="tab" data-bs-target="#top-return" 
                    type="button" role="tab" aria-controls="top-return" aria-selected="true">
                    Top Rendite
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="top-sharpe-tab" data-bs-toggle="tab" data-bs-target="#top-sharpe" 
                    type="button" role="tab" aria-controls="top-sharpe" aria-selected="false">
                    Top Sharpe Ratio
                </button>
            </li>
        </ul>

        <div class="tab-content" id="backtestTabContent">
            <div class="tab-pane fade show active" id="top-return" role="tabpanel" aria-labelledby="top-return-tab">
                <h3>Top 10 Backtests nach Gesamtrendite</h3>
                {top_return_table}
            </div>
            <div class="tab-pane fade" id="top-sharpe" role="tabpanel" aria-labelledby="top-sharpe-tab">
                <h3>Top 10 Backtests nach Sharpe Ratio</h3>
                {top_sharpe_table}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function viewBacktest(path) {{
            // Hier könnte man zu einer detaillierten Backtest-Ansicht navigieren
            alert('Backtest-Details für: ' + path + '\\nDiese Funktion wird in einer zukünftigen Version implementiert.');
        }}

        function refreshDashboard() {{
            // Dashboard neu laden
            location.reload();
        }}
    </script>
</body>
</html>
""")

    return html_path


def main():
    parser = argparse.ArgumentParser(description='Backtest Dashboard Generator')
    parser.add_argument('--registry', type=str, default='data/backtest_registry.json',
                        help='Pfad zur Backtest-Registry (Standard: data/backtest_registry.json)')

    args = parser.parse_args()

    print(f"Lade Backtest-Registry aus: {args.registry}")
    registry = load_registry(args.registry)

    if not registry:
        print("Keine Registry-Daten gefunden.")
        return

    print(f"Gefundene Backtests: {len(registry.get('backtests', []))}")

    print("Erstelle Dashboard...")
    dashboard_path = create_dashboard_html(registry)

    if dashboard_path:
        print(f"Dashboard erstellt: {dashboard_path}")
        print("Sie können das Dashboard in einem Webbrowser öffnen.")
    else:
        print("Fehler beim Erstellen des Dashboards.")


if __name__ == "__main__":
    main()