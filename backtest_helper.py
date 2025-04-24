#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest-Helper-Modul zum Durchführen und Tracken von Backtest-Runs.
"""

import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import Settings
from core.trading_bot import TradingBot


class BacktestRegistry:
    """
    Zentrale Registrierung und Verwaltung aller Backtest-Ergebnisse.
    """

    def __init__(self, registry_file="data/backtest_registry.json"):
        """
        Initialisiert die Backtest-Registrierung.

        Args:
            registry_file: Pfad zur JSON-Datei für die Registrierung
        """
        self.registry_file = registry_file
        self._load_registry()

    def _load_registry(self):
        """Lädt die Registry aus der Datei oder erstellt eine neue."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.registry = json.load(f)
            except json.JSONDecodeError:
                # Falls die Datei beschädigt ist, neue Registry erstellen
                self.registry = {
                    "backtests": [],
                    "last_update": datetime.now().isoformat()
                }
        else:
            self.registry = {
                "backtests": [],
                "last_update": datetime.now().isoformat()
            }

    def _save_registry(self):
        """Speichert die Registry in der Datei."""
        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)

        # Aktualisierungszeit setzen
        self.registry["last_update"] = datetime.now().isoformat()

        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=4)

    def add_backtest(self, test_name, strategy, params, results):
        """
        Fügt einen durchgeführten Backtest zur Registry hinzu.

        Args:
            test_name: Name des Backtest-Runs
            strategy: Verwendete Strategie
            params: Dictionary mit verwendeten Parametern
            results: Dictionary mit Backtest-Ergebnissen
        """
        # Wichtige Ergebnisse extrahieren
        stats = results.get("statistics", {})

        # Neuen Eintrag erstellen
        entry = {
            "test_name": test_name,
            "strategy": strategy,
            "date": datetime.now().isoformat(),
            "parameters": params,
            "results": {
                "total_return": results.get("total_return", 0),
                "total_trades": results.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "max_drawdown": stats.get("max_drawdown", 0),
                "profit_factor": stats.get("profit_factor", 0),
                "sortino_ratio": stats.get("sortino_ratio", 0),
                "calmar_ratio": stats.get("calmar_ratio", 0)
            },
            "result_path": f"data/backtest_results/{test_name}"  # Korrigierter Pfad!
        }

        # Zur Registry hinzufügen
        self.registry["backtests"].append(entry)

        # Registry speichern
        self._save_registry()

        return entry

    def get_backtest(self, test_name):
        """
        Ruft einen einzelnen Backtest nach Namen ab.

        Args:
            test_name: Name des Backtest-Runs

        Returns:
            Dictionary mit Backtest-Informationen oder None
        """
        for test in self.registry["backtests"]:
            if test["test_name"] == test_name:
                return test
        return None

    def get_all_backtests(self):
        """
        Ruft alle Backtests ab.

        Returns:
            Liste aller Backtest-Einträge
        """
        return self.registry["backtests"]

    def get_best_backtest(self, metric="total_return"):
        """
        Ruft den besten Backtest basierend auf einer Metrik ab.

        Args:
            metric: Metrik für den Vergleich ("total_return", "sharpe_ratio", etc.)

        Returns:
            Bester Backtest oder None
        """
        if not self.registry["backtests"]:
            return None

        return max(self.registry["backtests"],
                   key=lambda x: x["results"].get(metric, 0))

    def export_to_csv(self, filepath="data/backtest_results/backtest_summary.csv"):
        """
        Exportiert alle Backtest-Ergebnisse als CSV.

        Args:
            filepath: Zieldatei für den CSV-Export

        Returns:
            Pfad zur exportierten Datei
        """
        if not self.registry["backtests"]:
            return None

        # Daten vorbereiten
        rows = []
        for test in self.registry["backtests"]:
            row = {
                "test_name": test["test_name"],
                "strategy": test["strategy"],
                "date": test["date"],
                "path": test["result_path"]
            }

            # Parameter hinzufügen (flach)
            for param_key, param_value in test["parameters"].items():
                row[f"param_{param_key}"] = param_value

            # Ergebnisse hinzufügen
            for result_key, result_value in test["results"].items():
                row[result_key] = result_value

            rows.append(row)

        # DataFrame erstellen und speichern
        df = pd.DataFrame(rows)

        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False)

        return filepath

    def visualize_results(self, metric="total_return", top_n=10,
                          save_path="data/visualizations/backtest_comparison.png"):
        """
        Visualisiert die Backtest-Ergebnisse.

        Args:
            metric: Zu visualisierende Metrik
            top_n: Anzahl der besten Ergebnisse
            save_path: Pfad zum Speichern der Visualisierung

        Returns:
            Pfad zur Visualisierungsdatei
        """
        if not self.registry["backtests"]:
            return None

        # Nach Metrik sortieren
        sorted_tests = sorted(
            self.registry["backtests"],
            key=lambda x: x["results"].get(metric, 0),
            reverse=True
        )[:top_n]

        # Daten vorbereiten
        names = [t["test_name"] for t in sorted_tests]
        values = [t["results"].get(metric, 0) for t in sorted_tests]
        strategies = [t["strategy"] for t in sorted_tests]

        # Plot erstellen
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, values)

        # Strategien als Farben darstellen
        unique_strategies = list(set(strategies))
        colors = plt.cm.tab10(range(len(unique_strategies)))
        color_map = dict(zip(unique_strategies, colors))

        for i, bar in enumerate(bars):
            bar.set_color(color_map[strategies[i]])

        # Beschriftungen und Titel
        plt.title(f"Top {top_n} Backtests by {metric}")
        plt.xlabel("Backtest")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Legende für Strategien
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s])
                   for s in unique_strategies]
        plt.legend(handles, unique_strategies, title="Strategy")

        plt.tight_layout()

        # Verzeichnis erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)
        plt.close()

        return save_path


def run_backtest(strategy_name, params, test_name=None, tags=None):
    """
    Führt einen einzelnen Backtest durch und speichert die Ergebnisse.

    Args:
        strategy_name: Name der Strategie
        params: Dictionary mit anzupassenden Parametern
        test_name: Optionaler benutzerdefinierter Name für den Test
        tags: Liste von Tags für die Kategorisierung

    Returns:
        Tuple aus Testergebnissen und Ausgabeverzeichnis
    """
    # Einstellungen initialisieren
    settings = Settings()

    # Parameter setzen
    for key, value in params.items():
        settings.set(key, value)

    # Zeitstempel für eindeutige Identifikation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Testnamen generieren, falls nicht angegeben
    if not test_name:
        test_name = f"{strategy_name}_{timestamp}"

    # Ausgabeverzeichnis erstellen
    output_dir = os.path.join("data/backtest_results", test_name)
    os.makedirs(output_dir, exist_ok=True)

    # Parameter-Konfiguration speichern
    config_with_meta = {
        "strategy": strategy_name,
        "date": timestamp,
        "tags": tags or [],
        "parameters": params
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_with_meta, f, indent=4)

    # Bot initialisieren und Backtest ausführen
    bot = TradingBot(mode="backtest", strategy_name=strategy_name, settings=settings)
    results = bot.run_backtest()

    # Visualisierungen und Exporte im spezifischen Verzeichnis speichern
    if "plot_files" not in results or not results["plot_files"]:
        # Manuell Ergebnisse visualisieren, falls nicht automatisch erfolgt
        from core.enhanced_backtesting import EnhancedBacktester
        backtester = EnhancedBacktester(settings, bot.strategy)
        backtester.results = results
        plot_files = backtester.plot_results(output_dir=output_dir)
        results["plot_files"] = plot_files

    # Ergebnisse exportieren
    export_format = settings.get("backtest.export_format", "excel")
    if "export_files" not in results or not results["export_files"]:
        from core.enhanced_backtesting import EnhancedBacktester
        backtester = EnhancedBacktester(settings, bot.strategy)
        backtester.results = results
        export_files = backtester.export_results(output_dir=output_dir, format=export_format)
        results["export_files"] = export_files

    # Zusammenfassung speichern
    stats = results.get("statistics", {})
    summary = {
        "test_name": test_name,
        "strategy": strategy_name,
        "timestamp": timestamp,
        "total_return": results.get("total_return", 0),
        "total_trades": results.get("total_trades", 0),
        "win_rate": stats.get("win_rate", 0),
        "sharpe_ratio": stats.get("sharpe_ratio", 0),
        "max_drawdown": stats.get("max_drawdown", 0),
        "profit_factor": stats.get("profit_factor", 0)
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Zur Registry hinzufügen
    registry = BacktestRegistry()
    registry.add_backtest(test_name, strategy_name, params, results)

    return results, output_dir


def compare_backtests(test_names, metrics=None):
    """
    Vergleicht mehrere Backtests miteinander.

    Args:
        test_names: Liste von Backtest-Namen
        metrics: Liste von zu vergleichenden Metriken
                (default: total_return, sharpe_ratio, win_rate)

    Returns:
        DataFrame mit Vergleichsdaten
    """
    if metrics is None:
        metrics = ["total_return", "sharpe_ratio", "win_rate", "max_drawdown"]

    registry = BacktestRegistry()

    # Backtests abrufen
    tests = []
    for name in test_names:
        test = registry.get_backtest(name)
        if test:
            tests.append(test)

    if not tests:
        return None

    # Vergleichstabelle erstellen
    comparison = []
    for test in tests:
        row = {"test_name": test["test_name"], "strategy": test["strategy"]}

        # Metriken hinzufügen
        for metric in metrics:
            row[metric] = test["results"].get(metric, 0)

        comparison.append(row)

    return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Beispiel für die Verwendung

    # 1. Backtest-Registry initialisieren
    registry = BacktestRegistry()

    # 2. Einen Backtest durchführen
    momentum_params = {
        "backtest.start_date": "2023-01-01",
        "backtest.end_date": "2023-12-31",
        "backtest.initial_balance": 10000,
        "trading_pairs": ["BTC/USDT", "ETH/USDT"],
        "momentum.rsi_period": 14,
        "momentum.rsi_overbought": 70,
        "momentum.rsi_oversold": 30,
        "risk.stop_loss": 0.03,
        "risk.take_profit": 0.06
    }

    results, output_dir = run_backtest(
        strategy_name="momentum",
        params=momentum_params,
        test_name="momentum_standard",
        tags=["standard", "momentum"]
    )

    print(f"Backtest completed. Total return: {results.get('total_return', 0):.2f}%")
    print(f"Results saved to: {output_dir}")

    # 3. Alle Backtests in CSV exportieren
    csv_path = registry.export_to_csv()
    print(f"Backtest summary exported to: {csv_path}")

    # 4. Visualisierung erstellen
    viz_path = registry.visualize_results(metric="sharpe_ratio")
    print(f"Visualization saved to: {viz_path}")