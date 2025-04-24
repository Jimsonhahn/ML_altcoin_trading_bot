#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript zur Ausführung von ML-gestützten Backtests für den Trading Bot.
Dieses Skript verwendet die MLEnhancedBacktester-Klasse, um Backtests
mit Integration von ML-Komponenten durchzuführen.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Pfad-Konfiguration
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import der notwendigen Komponenten
from config.settings import Settings
from core.ml_enhanced_backtesting import MLEnhancedBacktester
from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.ml_strategy import MLStrategy

# Logger konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ml_backtest")


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Lädt die Konfiguration aus einer JSON-Datei.

    Args:
        config_file: Pfad zur Konfigurationsdatei

    Returns:
        Dictionary mit Konfigurationswerten
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        logger.info(f"Konfiguration aus {config_file} geladen")
        return config
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration aus {config_file}: {e}")
        return {}


def create_directories() -> None:
    """
    Erstellt die notwendigen Verzeichnisse für den ML-Backtest.
    """
    directories = [
        "data/market_data",
        "data/market_data/binance",
        "data/ml_models",
        "data/ml_analysis",
        "data/backtest_results"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Verzeichnis erstellt/überprüft: {directory}")


def create_strategy(strategy_name: str, settings: Settings) -> Strategy:
    """
    Erstellt eine Strategie-Instanz basierend auf dem Namen.

    Args:
        strategy_name: Name der Strategie
        settings: Einstellungen für die Strategie

    Returns:
        Instanz der Strategie-Klasse
    """
    if strategy_name == "momentum":
        return MomentumStrategy(settings)
    elif strategy_name == "ml":
        return MLStrategy(settings)
    else:
        logger.warning(f"Unbekannte Strategie: {strategy_name}, verwende Momentum")
        return MomentumStrategy(settings)


def run_backtest(config_file: str, strategy_name: str = "ml",
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Führt einen ML-gestützten Backtest mit der angegebenen Konfiguration durch.

    Args:
        config_file: Pfad zur Konfigurationsdatei
        strategy_name: Name der zu verwendenden Strategie
        output_dir: Verzeichnis für Ausgabedateien (optional)

    Returns:
        Dictionary mit Backtest-Ergebnissen
    """
    # Verzeichnisse erstellen
    create_directories()

    # Konfiguration laden
    config = load_config(config_file)

    if not config:
        logger.error("Keine Konfiguration geladen, Backtest abgebrochen")
        return {"error": "no_config"}

    # Settings-Objekt erstellen
    settings = Settings()

    # Konfiguration in Settings übertragen (einzeln mit set statt mit load_dict)
    for key, value in config.items():
        settings.set(key, value)

    # Strategie erstellen
    strategy = create_strategy(strategy_name, settings)

    # Ausgabeverzeichnis festlegen
    if output_dir:
        backtest_output_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_output_dir = f"data/backtest_results/{strategy_name}_{timestamp}"

    os.makedirs(backtest_output_dir, exist_ok=True)

    # Backtest-Parameter
    symbols = settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT"])
    source = settings.get('data.source', 'binance')
    timeframe = settings.get('timeframes.analysis', '1d')
    use_cache = settings.get('data.use_cache', True)

    try:
        # ML-Backtester initialisieren
        backtester = MLEnhancedBacktester(settings, strategy)

        # ML-Modelle trainieren, falls nicht vorhanden
        if settings.get('ml.enabled', False) and settings.get('ml.train_on_start', False):
            logger.info("Training der ML-Modelle gestartet...")
            backtester.train_ml_models(symbols)

        # ML-Backtest ausführen
        logger.info(f"Starte ML-Backtest mit Strategie {strategy_name}...")
        results = backtester.run_with_ml(
            symbols=symbols,
            source=source,
            timeframe=timeframe,
            use_cache=use_cache
        )

        # Ergebnisse visualisieren
        logger.info(f"Visualisiere Ergebnisse in {backtest_output_dir}...")
        plot_files = backtester.plot_results(output_dir=backtest_output_dir)

        # ML-Vergleich visualisieren
        ml_plot_files = backtester.plot_ml_comparison(output_dir=backtest_output_dir)
        if ml_plot_files:
            plot_files.extend(ml_plot_files)

        # Ergebnisse exportieren
        export_format = settings.get('backtest.export_format', 'excel')
        exported_files = backtester.export_results(
            output_dir=backtest_output_dir,
            format=export_format
        )

        # Ergebnisse anzeigen
        logger.info("Backtest abgeschlossen.")
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Win Rate: {results['statistics']['win_rate']:.2f}%")

        if 'ml_comparison' in results:
            improvement = results['ml_comparison'].get('return_improvement', 0)
            logger.info(f"ML-Verbesserung: {improvement:.2f}%")

        # Erweiterte Statistiken anzeigen
        if 'statistics' in results:
            stats = results['statistics']
            logger.info(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {abs(stats.get('max_drawdown', 0)):.2f}%")
            logger.info(f"Profit Faktor: {stats.get('profit_factor', 0):.2f}")

        # Dateipfade zu den Ergebnissen hinzufügen
        results['plot_files'] = plot_files
        results['exported_files'] = exported_files
        results['output_dir'] = backtest_output_dir

        # Konfiguration speichern
        config_output = os.path.join(backtest_output_dir, "backtest_config.json")
        with open(config_output, 'w') as f:
            json.dump(config, f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Fehler während des Backtests: {e}", exc_info=True)
        return {"error": str(e)}


def perform_multiple_backtests(config_file: str, strategies: List[str],
                               compare: bool = True) -> Dict[str, Any]:
    """
    Führt mehrere Backtests mit verschiedenen Strategien durch und vergleicht sie.

    Args:
        config_file: Pfad zur Konfigurationsdatei
        strategies: Liste von Strategien, die getestet werden sollen
        compare: Ob die Ergebnisse verglichen werden sollen

    Returns:
        Dictionary mit Vergleichsergebnissen
    """
    results = {}

    for strategy in strategies:
        logger.info(f"Starte Backtest mit Strategie: {strategy}")
        strategy_results = run_backtest(config_file, strategy)
        results[strategy] = strategy_results

    if compare and len(strategies) > 1:
        # Vergleichsverzeichnis erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = f"data/backtest_results/comparison_{timestamp}"
        os.makedirs(comparison_dir, exist_ok=True)

        # Vergleich erstellen
        compare_results = {}

        for strategy, result in results.items():
            if "error" not in result:
                compare_results[strategy] = {
                    "total_return": result.get('total_return', 0),
                    "win_rate": result.get('statistics', {}).get('win_rate', 0),
                    "sharpe_ratio": result.get('statistics', {}).get('sharpe_ratio', 0),
                    "max_drawdown": result.get('statistics', {}).get('max_drawdown', 0),
                    "profit_factor": result.get('statistics', {}).get('profit_factor', 0)
                }

        # Vergleichsdatei speichern
        comparison_file = os.path.join(comparison_dir, "strategy_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(compare_results, f, indent=2)

        # Ergebnisvergleich visualisieren (falls matplotlib installiert ist)
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Daten für Balkendiagramm vorbereiten
            metrics = ["total_return", "win_rate", "sharpe_ratio", "profit_factor"]
            strategies_list = list(compare_results.keys())

            for metric in metrics:
                plt.figure(figsize=(10, 6))

                values = [compare_results[s][metric] for s in strategies_list]
                bars = plt.bar(strategies_list, values)

                # Werte über den Balken anzeigen
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{values[i]:.2f}",
                        ha='center'
                    )

                plt.title(f"Vergleich: {metric}")
                plt.ylabel(metric)
                plt.grid(axis='y', alpha=0.3)

                # Speichern
                plt.savefig(os.path.join(comparison_dir, f"compare_{metric}.png"))
                plt.close()

            # Equity-Kurven vergleichen
            plt.figure(figsize=(12, 6))

            for strategy, result in results.items():
                if "error" not in result and "equity_curve" in result:
                    equity_df = result["equity_curve"]
                    if not equity_df.empty and "portfolio_value" in equity_df.columns:
                        # Normalisieren für besseren Vergleich
                        initial_value = equity_df["portfolio_value"].iloc[0]
                        normalized = equity_df["portfolio_value"] / initial_value * 100
                        normalized.plot(label=strategy)

            plt.title("Equity-Kurven Vergleich (normalisiert)")
            plt.ylabel("Portfolio-Wert (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Speichern
            plt.savefig(os.path.join(comparison_dir, "equity_comparison.png"))
            plt.close()

        except ImportError:
            logger.warning("Matplotlib nicht installiert, keine Visualisierung des Vergleichs möglich")

        results["comparison"] = {
            "data": compare_results,
            "output_dir": comparison_dir
        }

    return results


def create_default_config(output_file: str = "config/backtest_config.json") -> None:
    """
    Erstellt eine Standardkonfiguration für den ML-Backtest.

    Args:
        output_file: Pfad zur Ausgabedatei
    """
    config = {
        "backtest.start_date": "2022-01-01",
        "backtest.end_date": "2023-12-31",
        "backtest.initial_balance": 10000,
        "backtest.commission": 0.001,
        "backtest.create_plots": True,
        "backtest.export_results": True,
        "backtest.export_format": "excel",

        "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        "data.source": "binance",
        "data.use_cache": True,

        "timeframes.analysis": "1d",
        "timeframes.check_interval": 86400,

        "risk.position_size": 0.1,
        "risk.stop_loss": 0.03,
        "risk.take_profit": 0.06,
        "risk.max_open_positions": 5,
        "risk.use_trailing_stop": True,
        "risk.trailing_stop": 0.02,
        "risk.trailing_activation": 0.02,

        "ml.enabled": True,
        "ml.data_dir": "data/market_data",
        "ml.models_dir": "data/ml_models",
        "ml.output_dir": "data/ml_analysis",
        "ml.n_regimes": 5,
        "ml.train_on_start": True
    }

    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Konfiguration speichern
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Standardkonfiguration in {output_file} gespeichert")


def main():
    """
    Hauptfunktion für die Ausführung des ML-Backtests.
    """
    parser = argparse.ArgumentParser(description="ML-Backtest für Trading Bot")

    parser.add_argument(
        "-c", "--config",
        default="config/backtest_config.json",
        help="Pfad zur Konfigurationsdatei"
    )

    parser.add_argument(
        "-s", "--strategy",
        default="ml",
        choices=["momentum", "ml"],
        help="Zu verwendende Strategie"
    )

    parser.add_argument(
        "-o", "--output",
        help="Ausgabeverzeichnis für Ergebnisse"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Erstellt eine Standardkonfiguration und beendet"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Vergleicht ML-Strategie mit Momentum-Strategie"
    )

    args = parser.parse_args()

    # Standardkonfiguration erstellen
    if args.create_config:
        create_default_config(args.config)
        return

    # Prüfen, ob Konfigurationsdatei existiert
    if not os.path.exists(args.config):
        logger.error(f"Konfigurationsdatei {args.config} nicht gefunden.")
        logger.info("Sie können eine Standardkonfiguration erstellen mit: --create-config")
        return

    # Strategievergleich durchführen
    if args.compare:
        results = perform_multiple_backtests(
            args.config,
            strategies=["momentum", "ml"],
            compare=True
        )

        # Vergleichsergebnisse anzeigen
        if "comparison" in results:
            compare_data = results["comparison"]["data"]
            logger.info("\nVergleichsergebnisse:")

            for strategy, metrics in compare_data.items():
                logger.info(f"\n{strategy}:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.2f}")

            logger.info(f"\nVergleichsvisualisierungen in: {results['comparison']['output_dir']}")

    # Einzelnen Backtest durchführen
    else:
        results = run_backtest(args.config, args.strategy, args.output)

        if "error" in results:
            logger.error(f"Backtest fehlgeschlagen: {results['error']}")
        else:
            logger.info(f"\nBacktest-Ergebnisse in: {results['output_dir']}")


if __name__ == "__main__":
    main()