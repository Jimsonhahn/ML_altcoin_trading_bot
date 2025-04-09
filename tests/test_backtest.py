#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Einfaches Skript zum Testen des Backtesting-Moduls.
Führt einen Backtest mit der Momentum-Strategie durch.
"""

import os
import sys
import logging
from datetime import datetime

# Projektverzeichnis zum Python-Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from strategies.momentum import MomentumStrategy
from core.backtesting import Backtester
from utils.logger import setup_logger


def main():
    # Logger einrichten
    logger = setup_logger(logging.INFO)
    logger.info("Starting backtest...")

    # Startzeit messen
    start_time = datetime.now()

    # Konfiguration laden
    settings = Settings()
    logger.info(f"Trading pairs: {settings.get('trading_pairs')}")
    logger.info(f"Position size: {settings.get('risk.position_size')}")
    logger.info(f"Stop loss: {settings.get('risk.stop_loss')}")
    logger.info(f"Take profit: {settings.get('risk.take_profit')}")
    logger.info(f"Timeframe: {settings.get('timeframes.analysis')}")

    # Strategie initialisieren
    strategy = MomentumStrategy(settings)

    # Trading Paare
    trading_pairs = settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"])

    # Backtester initialisieren
    backtester = Backtester(settings, strategy)

    # Backtest durchführen
    results = backtester.run(trading_pairs)

    # Ergebnisse visualisieren
    plot_path = backtester.plot_results()
    if plot_path:
        logger.info(f"Results plot saved to: {plot_path}")

    # Einige Statistiken anzeigen
    stats = results.get('statistics', {})
    logger.info(f"Total Return: {results.get('total_return', 0):.2f}%")
    logger.info(f"Total Trades: {results.get('total_trades', 0)}")
    logger.info(f"Win Rate: {stats.get('win_rate', 0):.2f}%")
    logger.info(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
    logger.info(f"Max Drawdown: {abs(stats.get('max_drawdown', 0)):.2f}%")
    logger.info(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")

    # Endzeit messen
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Backtest completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()