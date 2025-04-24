#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Erweitertes Visualisierungs-Tool für Backtest-Ergebnisse.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from datetime import datetime
import argparse
import glob


def load_backtest(backtest_dir):
    """
    Lädt Backtest-Ergebnisse aus einem Verzeichnis.

    Args:
        backtest_dir: Pfad zum Backtest-Verzeichnis

    Returns:
        Dictionary mit geladenen Daten
    """
    data = {}

    # Konfiguration laden
    config_path = os.path.join(backtest_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)

    # Zusammenfassung laden
    summary_path = os.path.join(backtest_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            data['summary'] = json.load(f)

    # XLSX-Dateien suchen
    excel_files = glob.glob(os.path.join(backtest_dir, '*.xlsx'))
    if excel_files:
        data['excel_files'] = excel_files

        # Equity Curve und Trades aus der ersten Excel-Datei laden
        try:
            xlsx_data = pd.read_excel(excel_files[0], sheet_name=None)

            # Equity Curve
            if 'Equity Curve' in xlsx_data:
                equity_df = xlsx_data['Equity Curve']
                if 'date' in equity_df.columns:
                    equity_df.set_index('date', inplace=True)
                elif 'index' in equity_df.columns:
                    equity_df.set_index('index', inplace=True)
                data['equity_curve'] = equity_df

            # Trades
            if 'Trades' in xlsx_data:
                trades_df = xlsx_data['Trades']
                if 'date' in trades_df.columns:
                    trades_df.set_index('date', inplace=True)
                elif 'index' in trades_df.columns:
                    trades_df.set_index('index', inplace=True)
                data['trades'] = trades_df

            # Statistiken
            if 'Statistics' in xlsx_data:
                stats_df = xlsx_data['Statistics']
                data['statistics'] = stats_df
        except Exception as e:
            print(f"Fehler beim Laden der Excel-Datei: {e}")

    # CSV-Dateien suchen
    csv_files = glob.glob(os.path.join(backtest_dir, '*.csv'))
    if csv_files and not excel_files:
        data['csv_files'] = csv_files

        # Versuchen, Equity Curve und Trades zu laden
        equity_files = [f for f in csv_files if 'equity' in f.lower()]
        trade_files = [f for f in csv_files if 'trade' in f.lower()]

        if equity_files:
            try:
                equity_df = pd.read_csv(equity_files[0])
                if 'date' in equity_df.columns:
                    equity_df.set_index('date', inplace=True)
                data['equity_curve'] = equity_df
            except Exception as e:
                print(f"Fehler beim Laden der Equity-Curve-Datei: {e}")

        if trade_files:
            try:
                trades_df = pd.read_csv(trade_files[0])
                if 'date' in trades_df.columns:
                    trades_df.set_index('date', inplace=True)
                data['trades'] = trades_df
            except Exception as e:
                print(f"Fehler beim Laden der Trades-Datei: {e}")

    return data


def create_advanced_visualizations(backtest_data, output_dir):
    """
    Erstellt erweiterte Visualisierungen für Backtest-Ergebnisse.

    Args:
        backtest_data: Dictionary mit Backtest-Daten
        output_dir: Verzeichnis für die Ausgabe

    Returns:
        Liste der erstellten Dateien
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # 1. Equity Curve mit Drawdown
    if 'equity_curve' in backtest_data and not backtest_data['equity_curve'].empty:
        equity_file = create_equity_curve(backtest_data, output_dir)
        if equity_file:
            created_files.append(equity_file)

    # 2. Trade-Analyse
    if 'trades' in backtest_data and not backtest_data['trades'].empty:
        trades_file = create_trade_analysis(backtest_data, output_dir)
        if trades_file:
            created_files.append(trades_file)

    # 3. Performance-Übersicht
    summary_file = create_performance_summary(backtest_data, output_dir)
    if summary_file:
        created_files.append(summary_file)

    # 4. Parameter-Übersicht
    if 'config' in backtest_data:
        params_file = create_parameter_overview(backtest_data, output_dir)
        if params_file:
            created_files.append(params_file)

    # 5. Gewinn/Verlust-Verteilung
    if 'trades' in backtest_data and not backtest_data['trades'].empty:
        pnl_file = create_pnl_distribution(backtest_data, output_dir)
        if pnl_file:
            created_files.append(pnl_file)

    # 6. Dashboard (eine Zusammenfassung aller Grafiken)
    dashboard_file = create_dashboard(backtest_data, output_dir, created_files)
    if dashboard_file:
        created_files.append(dashboard_file)

    return created_files


def create_equity_curve(backtest_data, output_dir):
    """Erstellt eine Grafik der Equity-Kurve mit Drawdown."""
    try:
        equity_df = backtest_data['equity_curve']

        # Nur die relevanten Spalten auswählen
        important_cols = ['portfolio_value', 'balance', 'positions_value', 'open_positions']
        existing_cols = [col for col in important_cols if col in equity_df.columns]

        if not existing_cols:
            return None

        plt.figure(figsize=(12, 8))

        # Hauptplot für Equity Curve
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

        # Portfolio-Wert plotten
        if 'portfolio_value' in equity_df.columns:
            ax1.plot(equity_df.index, equity_df['portfolio_value'], label='Portfolio-Wert', color='blue', linewidth=2)

        # Balance und Positionswert plotten, wenn verfügbar
        if 'balance' in equity_df.columns and 'positions_value' in equity_df.columns:
            ax1.plot(equity_df.index, equity_df['balance'], label='Balance', color='green', linestyle='--')
            ax1.plot(equity_df.index, equity_df['positions_value'], label='Positionswert', color='orange',
                     linestyle=':')

        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Wert')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Anzahl offener Positionen als zusätzliche Linie, wenn verfügbar
        if 'open_positions' in equity_df.columns:
            ax3 = ax1.twinx()
            ax3.plot(equity_df.index, equity_df['open_positions'], label='Offene Positionen', color='red', alpha=0.5)
            ax3.set_ylabel('Anzahl Positionen', color='red')
            ax3.legend(loc='upper right')

        # Drawdown berechnen und plotten
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

        if 'portfolio_value' in equity_df.columns:
            portfolio_value = equity_df['portfolio_value']
            rolling_max = portfolio_value.cummax()
            drawdown = ((portfolio_value - rolling_max) / rolling_max) * 100
            ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
            ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)

        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Datum')
        ax2.grid(True, alpha=0.3)

        # X-Achse formatieren
        date_format = '%Y-%m-%d'
        ax2.xaxis.set_major_formatter(DateFormatter(date_format))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Speichern
        output_file = os.path.join(output_dir, 'equity_curve_advanced.png')
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"Equity Curve Visualisierung gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen der Equity Curve: {e}")
        return None


def create_trade_analysis(backtest_data, output_dir):
    """Erstellt eine detaillierte Analyse der Trades."""
    try:
        trades_df = backtest_data['trades']

        # Sicherstellen, dass wir die richtigen Spalten haben
        required_cols = ['profit_loss', 'profit_loss_pct', 'symbol', 'action']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]

        if missing_cols:
            print(f"Fehlende Spalten für die Trade-Analyse: {missing_cols}")
            # Versuch, alternative Spaltennamen zu finden
            if 'profit_loss_percent' in trades_df.columns and 'profit_loss_pct' in missing_cols:
                trades_df['profit_loss_pct'] = trades_df['profit_loss_percent']
                missing_cols.remove('profit_loss_pct')

        # Überprüfen, ob nach der Anpassung noch Spalten fehlen
        if missing_cols:
            return None

        plt.figure(figsize=(15, 12))

        # 1. Gewinn/Verlust pro Trade
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        trades_df['profit_loss_pct'].plot(kind='bar', ax=ax1, color=trades_df['profit_loss_pct'].map(
            lambda x: 'green' if x > 0 else 'red'))
        ax1.set_title('Gewinn/Verlust pro Trade')
        ax1.set_xlabel('Trade Nr.')
        ax1.set_ylabel('Gewinn/Verlust (%)')
        ax1.axhline(y=0, color='black', linestyle='-')

        # X-Achse limitieren, wenn zu viele Trades
        if len(trades_df) > 30:
            ax1.set_xticks([])

        # 2. Kumulative P/L
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        cumulative_pnl = trades_df['profit_loss'].cumsum()
        cumulative_pnl.plot(ax=ax2)
        ax2.set_title('Kumulative Gewinn/Verlust')
        ax2.set_xlabel('Zeit')
        ax2.set_ylabel('Kumulativer G/V')
        ax2.grid(True, alpha=0.3)

        # 3. P/L Histogramm
        ax3 = plt.subplot2grid((3, 2), (1, 0))
        sns.histplot(trades_df['profit_loss_pct'], bins=20, kde=True, ax=ax3)
        ax3.set_title('G/V Verteilung')
        ax3.set_xlabel('Gewinn/Verlust (%)')
        ax3.set_ylabel('Anzahl Trades')
        ax3.axvline(x=0, color='black', linestyle='--')

        # 4. Leistung nach Symbol
        if 'symbol' in trades_df.columns:
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            symbol_performance = trades_df.groupby('symbol')['profit_loss'].sum().sort_values()
            bars = symbol_performance.plot(kind='bar', ax=ax4, color=symbol_performance.map(
                lambda x: 'green' if x > 0 else 'red'))
            ax4.set_title('G/V nach Symbol')
            ax4.set_xlabel('Symbol')
            ax4.set_ylabel('Gesamt G/V')

            # Werte über den Balken anzeigen
            for bar in bars.containers:
                ax4.bar_label(bar, fmt='%.2f', padding=3)

        # 5. Win/Loss im Zeitverlauf
        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

        # Erfolgreiche und verlustreiche Trades markieren
        win_trades = trades_df[trades_df['profit_loss'] > 0]
        loss_trades = trades_df[trades_df['profit_loss'] <= 0]

        if not win_trades.empty:
            ax5.scatter(win_trades.index, win_trades['profit_loss_pct'],
                        color='green', marker='^', s=100, label='Gewinn')

        if not loss_trades.empty:
            ax5.scatter(loss_trades.index, loss_trades['profit_loss_pct'],
                        color='red', marker='v', s=100, label='Verlust')

        ax5.axhline(y=0, color='black', linestyle='-')
        ax5.set_title('Trades im Zeitverlauf')
        ax5.set_xlabel('Zeit')
        ax5.set_ylabel('Gewinn/Verlust (%)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        plt.tight_layout()

        # Speichern
        output_file = os.path.join(output_dir, 'trade_analysis.png')
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"Trade-Analyse gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen der Trade-Analyse: {e}")
        return None


def create_performance_summary(backtest_data, output_dir):
    """Erstellt eine Übersicht der Performance-Kennzahlen."""
    try:
        # Metrics aus verschiedenen Quellen sammeln
        metrics = {}

        # Aus Summary
        if 'summary' in backtest_data:
            summary = backtest_data['summary']
            for key, value in summary.items():
                if isinstance(value, (int, float)) and key not in ['timestamp']:
                    metrics[key] = value

        # Aus Statistics (falls vorhanden)
        if 'statistics' in backtest_data:
            stats_df = backtest_data['statistics']
            for _, row in stats_df.iterrows():
                if len(row) >= 2:  # Mindestens Key und Value
                    key = row.iloc[0] if isinstance(row.iloc[0], str) else str(row.index[0])
                    value = row.iloc[1] if isinstance(row.iloc[1], (int, float)) else None
                    if value is not None:
                        metrics[key] = value

        # Keine Metriken gefunden
        if not metrics:
            print("Keine Performance-Metrics gefunden")
            return None

        # Wichtige Metriken für die Visualisierung auswählen
        important_metrics = [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'win_rate', 'max_drawdown', 'profit_factor', 'total_trades'
        ]

        # Verfügbare wichtige Metriken filtern
        display_metrics = {}
        for key in important_metrics:
            if key in metrics:
                display_metrics[key] = metrics[key]
            # Ähnliche Namen suchen, falls nicht gefunden
            elif key + '_pct' in metrics:
                display_metrics[key] = metrics[key + '_pct']
            elif key.replace('_', '') in metrics:
                display_metrics[key] = metrics[key.replace('_', '')]

        # Wenn immer noch keine wichtigen Metriken gefunden wurden, alle anzeigen
        if not display_metrics:
            display_metrics = metrics

        # Metriknamen für die Anzeige formatieren
        formatted_metrics = {}
        for key, value in display_metrics.items():
            # Schlüssel formatieren
            formatted_key = key.replace('_', ' ').title()
            # Wert formatieren
            if isinstance(value, float):
                if key in ['total_return', 'win_rate', 'max_drawdown']:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            formatted_metrics[formatted_key] = formatted_value

        # Visualisierung erstellen
        plt.figure(figsize=(10, 6))

        # Erstellen eines Tabellen-artigen Displays
        table_data = [[k, v] for k, v in formatted_metrics.items()]
        table = plt.table(cellText=table_data, colLabels=['Metrik', 'Wert'],
                          loc='center', cellLoc='center')

        # Tabellenstil anpassen
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        # Spaltentiteln Farbe geben
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#E0E0E0' if row % 2 == 0 else 'white')

        # Achsen entfernen
        plt.axis('off')

        # Titel hinzufügen
        plt.title('Performance-Kennzahlen', fontsize=16, pad=20)

        plt.tight_layout()

        # Speichern
        output_file = os.path.join(output_dir, 'performance_summary.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Performance-Übersicht gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen der Performance-Übersicht: {e}")
        return None


def create_parameter_overview(backtest_data, output_dir):
    """Erstellt eine Übersicht der verwendeten Parameter."""
    try:
        if 'config' not in backtest_data:
            return None

        config = backtest_data['config']

        # Parameter nach Kategorien gruppieren
        param_categories = {}

        # Direkte Parameter
        for key, value in config.items():
            if isinstance(value, dict):
                # Kategorie
                category = key
                if category not in param_categories:
                    param_categories[category] = []

                # Parameter in dieser Kategorie
                for subkey, subvalue in value.items():
                    param_name = f"{category}.{subkey}"
                    param_value = str(subvalue)
                    param_categories[category].append((param_name, param_value))
            else:
                # Unkategorisierte Parameter
                if 'General' not in param_categories:
                    param_categories['General'] = []

                param_categories['General'].append((key, str(value)))

        # Visuelle Darstellung erstellen
        n_categories = len(param_categories)

        # Größe des Plots basierend auf der Anzahl der Parameter anpassen
        total_params = sum(len(params) for params in param_categories.values())
        height_per_param = 0.4
        min_height = 6
        fig_height = max(min_height, total_params * height_per_param)

        plt.figure(figsize=(12, fig_height))

        # Eine Tabelle pro Kategorie erstellen
        current_y = 0.95  # Start oben
        y_step = 0.9 / n_categories

        for i, (category, params) in enumerate(param_categories.items()):
            # Position für diese Kategorie
            y_pos = current_y - (i * y_step)

            # Kategorie-Überschrift
            plt.figtext(0.5, y_pos, category, ha='center', va='center', fontsize=14,
                        fontweight='bold', bbox=dict(facecolor='#4472C4', alpha=0.7, edgecolor='none',
                                                     boxstyle='round,pad=0.5', color='white'))

            # Parameter-Tabelle
            table_data = params
            table = plt.table(cellText=table_data, colLabels=['Parameter', 'Wert'],
                              loc='center', cellLoc='left', bbox=[0.1, y_pos - (len(params) * 0.03) - 0.1, 0.8, 0.1])

            # Tabellenstil anpassen
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # Spaltentiteln Farbe geben
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#E0E0E0' if row % 2 == 0 else 'white')

        # Achsen entfernen
        plt.axis('off')

        # Titel hinzufügen
        plt.title('Backtest-Parameter', fontsize=16, y=0.99)

        # Speichern
        output_file = os.path.join(output_dir, 'parameter_overview.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Parameter-Übersicht gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen der Parameter-Übersicht: {e}")
        return None


def create_pnl_distribution(backtest_data, output_dir):
    """Erstellt eine Visualisierung der Gewinn/Verlust-Verteilung."""
    try:
        if 'trades' not in backtest_data or backtest_data['trades'].empty:
            return None

        trades_df = backtest_data['trades']

        # Sicherstellen, dass wir die notwendigen Spalten haben
        if 'profit_loss_pct' not in trades_df.columns:
            # Versuchen, alternative Spalten zu finden
            if 'profit_loss_percent' in trades_df.columns:
                trades_df['profit_loss_pct'] = trades_df['profit_loss_percent']
            else:
                return None

        plt.figure(figsize=(12, 8))

        # 1. Histogramm mit KDE
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        sns.histplot(trades_df['profit_loss_pct'], kde=True, ax=ax1, color='skyblue',
                     edgecolor='darkblue')
        ax1.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax1.axvline(x=trades_df['profit_loss_pct'].mean(), color='green', linestyle='-',
                    label=f'Durchschnitt: {trades_df["profit_loss_pct"].mean():.2f}%')
        ax1.set_title('Verteilung der Trade-Ergebnisse')
        ax1.set_xlabel('Gewinn/Verlust (%)')
        ax1.set_ylabel('Anzahl Trades')
        ax1.legend()

        # 2. Gewinn vs. Verlust
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        win_trades = trades_df[trades_df['profit_loss_pct'] > 0]
        loss_trades = trades_df[trades_df['profit_loss_pct'] <= 0]

        win_pct = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        loss_pct = 100 - win_pct

        ax2.pie([win_pct, loss_pct], labels=['Gewinn', 'Verlust'], autopct='%1.1f%%',
                colors=['green', 'red'], startangle=90)
        ax2.set_title('Gewinn- vs. Verlust-Trades')

        # 3. Durchschnittlicher Gewinn/Verlust
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        avg_win = win_trades['profit_loss_pct'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['profit_loss_pct'].mean() if not loss_trades.empty else 0

        bars = ax3.bar(['Durchschnittlicher Gewinn', 'Durchschnittlicher Verlust'],
                       [avg_win, avg_loss],
                       color=['green', 'red'])

        # Werte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 Punkte vertikaler Offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        ax3.set_title('Durchschnittlicher Gewinn vs. Verlust')
        ax3.set_ylabel('Prozent (%)')
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Speichern
        output_file = os.path.join(output_dir, 'pnl_distribution.png')
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"G/V-Verteilung gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen der G/V-Verteilung: {e}")
        return None


def create_dashboard(backtest_data, output_dir, image_files):
    """Erstellt ein Dashboard mit allen Visualisierungen."""
    try:
        if not image_files:
            return None

        # Backtest-Name extrahieren
        backtest_name = "Backtest"
        if 'summary' in backtest_data and 'test_name' in backtest_data['summary']:
            backtest_name = backtest_data['summary']['test_name']
        elif 'config' in backtest_data and 'strategy' in backtest_data['config']:
            backtest_name = f"Strategie: {backtest_data['config']['strategy']}"

        # Start/Ende-Datum extrahieren
        date_range = ""
        if 'config' in backtest_data and 'backtest' in backtest_data['config']:
            start_date = backtest_data['config']['backtest'].get('start_date', '')
            end_date = backtest_data['config']['backtest'].get('end_date', '')
            if start_date and end_date:
                date_range = f"Zeitraum: {start_date} bis {end_date}"

        # Wichtigste Kennzahlen extrahieren
        key_metrics = []
        if 'summary' in backtest_data:
            summary = backtest_data['summary']
            if 'total_return' in summary:
                key_metrics.append(f"Gesamtrendite: {summary['total_return']:.2f}%")
            if 'win_rate' in summary:
                key_metrics.append(f"Win Rate: {summary['win_rate']:.2f}%")
            if 'sharpe_ratio' in summary:
                key_metrics.append(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            if 'max_drawdown' in summary:
                key_metrics.append(f"Max Drawdown: {abs(summary['max_drawdown']):.2f}%")

        # HTML-Datei erstellen
        output_file = os.path.join(output_dir, 'backtest_dashboard.html')

        # Relative Pfade zu den Bildern erstellen
        rel_image_paths = [os.path.basename(img_file) for img_file in image_files if img_file]

        with open(output_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Dashboard - {backtest_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .key-metrics {{ 
            display: flex; 
            justify-content: space-around; 
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .metric {{ 
            background-color: #f0f0f0; 
            padding: 10px; 
            border-radius: 5px;
            margin: 5px;
            min-width: 150px;
            text-align: center;
        }}
        .chart-container {{ 
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .chart {{ 
            margin: 10px; 
            border: 1px solid #ddd; 
            padding: 10px; 
            border-radius: 5px;
            max-width: 100%;
        }}
        .chart img {{ max-width: 100%; height: auto; }}
        h1, h2 {{ color: #333; }}
        .footer {{ text-align: center; margin-top: 20px; color: #777; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{backtest_name}</h1>
        <p>{date_range}</p>
    </div>

    <div class="key-metrics">
""")

            # Key Metrics einfügen
            for metric in key_metrics:
                f.write(f'        <div class="metric">{metric}</div>\n')

            f.write("""    </div>

    <div class="chart-container">
""")

            # Bilder einfügen
            for img_path in rel_image_paths:
                chart_name = img_path.replace('.png', '').replace('_', ' ').title()
                f.write(f"""        <div class="chart">
            <h3>{chart_name}</h3>
            <img src="{img_path}" alt="{chart_name}">
        </div>
""")

            f.write("""    </div>

    <div class="footer">
        <p>Generiert am {}</p>
    </div>
</body>
</html>""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        print(f"Dashboard gespeichert: {output_file}")
        return output_file

    except Exception as e:
        print(f"Fehler beim Erstellen des Dashboards: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Erweiterte Visualisierung für Backtest-Ergebnisse')
    parser.add_argument('--dir', type=str, required=True, help='Verzeichnis mit Backtest-Ergebnissen')
    parser.add_argument('--output', type=str,
                        help='Ausgabeverzeichnis für Visualisierungen (Standard: Backtest-Verzeichnis)')

    args = parser.parse_args()

    # Standardmäßig im selben Verzeichnis ausgeben
    output_dir = args.output if args.output else args.dir

    print(f"Lade Backtest-Daten aus: {args.dir}")
    backtest_data = load_backtest(args.dir)

    if not backtest_data:
        print("Keine Backtest-Daten gefunden.")
        return

    print(f"Gefundene Daten: {', '.join(backtest_data.keys())}")

    print(f"Erstelle Visualisierungen in: {output_dir}")
    created_files = create_advanced_visualizations(backtest_data, output_dir)

    if created_files:
        print(f"Erstellung abgeschlossen. {len(created_files)} Dateien erstellt.")
    else:
        print("Keine Visualisierungen erstellt.")


if __name__ == "__main__":
    main()