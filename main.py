#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Altcoin Trading Bot - Haupteinstiegspunkt
-----------------------------------------
Ein fortschrittlicher Trading Bot für Kryptowährungen mit Fokus auf Altcoins,
der technische Analyse, Social Media Sentiment und Machine Learning kombiniert.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Füge das Projektverzeichnis zum Pythonpfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.trading import TradingBot
from utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Altcoin Trading Bot')

    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['live', 'paper', 'backtest'],
                        help='Trading mode: live, paper, or backtest')

    parser.add_argument('--config', type=str, default='default',
                        help='Configuration profile to use')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    parser.add_argument('--strategy', type=str, default='default',
                        help='Trading strategy to use')

    return parser.parse_args()


def main():
    """Main function to start the trading bot"""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level)

    logger.info("Starting Altcoin Trading Bot...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Config profile: {args.config}")

    try:
        # Load settings
        settings = Settings(args.config)

        # Initialize trading bot
        bot = TradingBot(
            mode=args.mode,
            strategy_name=args.strategy,
            settings=settings
        )

        # Start the bot
        if args.mode == 'backtest':
            bot.run_backtest()
        else:
            bot.run()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.exception(f"Error running bot: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())