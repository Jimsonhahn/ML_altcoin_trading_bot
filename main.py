# main.py
import argparse
import logging
import os
import sys
import time
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Type

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.trading_bot import TradingBot
from utils.logger import setup_logger
from strategies import STRATEGIES  # Import existing STRATEGIES dict

# New/Updated imports for ML, Strategy Router, and Safety
from ml_components import MLComponents
from core.strategy_router import StrategyRouter
from core.safety_manager import SafetyManager
from data_sources.data_manager import DataManager  # Ensure this is imported for bot initialization


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot CLI")
    parser.add_argument("--config", type=str, default="default",
                        help="Name of the configuration profile to use (e.g., 'default', 'aggressive').")
    parser.add_argument("--strategy", type=str, default="momentum",
                        help="The trading strategy to use (e.g., 'momentum', 'grid_trading', 'ml_strategy'). "
                             "If --auto-strategy is enabled, this becomes the fallback/initial strategy.")
    parser.add_argument("--mode", type=str, default="live",
                        choices=["live", "paper", "backtest", "optimize"],
                        help="Operation mode: 'live', 'paper', 'backtest', 'optimize'.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging.")
    parser.add_argument("--validate-config", action="store_true",
                        help="Validate the selected configuration and exit.")
    parser.add_argument("--status-only", action="store_true",
                        help="Print current bot status and exit (for live/paper mode).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform setup and initialization but do not start trading loop. For testing startup.")
    parser.add_argument("--auto-strategy", action="store_true",
                        help="Enable automatic strategy routing based on market regimes. Overrides --strategy.")

    # Backtesting/Optimization arguments (expand as needed)
    parser.add_argument("--backtest-start", type=str,
                        help="Start date for backtesting (YYYY-MM-DD).")
    parser.add_argument("--backtest-end", type=str,
                        help="End date for backtesting (YYYY-MM-DD).")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Trading symbol for backtesting.")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe for backtesting data (e.g., '1h', '4h', '1d').")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup logger immediately
    log_level = 'DEBUG' if args.debug else 'INFO'
    main_logger = setup_logger(name='main', level=log_level)
    main_logger.info("Bot started.")

    # Load settings based on config profile
    settings = Settings(config_name=args.config)

    if args.validate_config:
        main_logger.info("Configuration validated successfully (basic check).")
        return

    # Initialize DataManager
    data_manager = DataManager(settings)
    main_logger.info("DataManager initialized.")

    # Initialize ML components (if ML is enabled in settings or auto-strategy is requested)
    ml_components_instance: Optional[MLComponents] = None
    if settings.get('ml.enabled', False) or args.auto_strategy:
        try:
            # Ensure MarketRegimeDetector has core symbols and min data points
            core_symbols = settings.get('ml.regime_core_symbols', ["BTC/USDT"])
            min_data_points = settings.get('ml.min_data_points_for_ml', 200)

            ml_components_instance = MLComponents(
                settings=settings,
                data_cache_dir=settings.get('data.cache_dir', 'data/market_data'),
                models_dir=settings.get('ml.models_dir', 'data/ml_models'),
                output_dir=settings.get('ml.output_dir', 'data/ml_analysis'),
                core_symbols=core_symbols,
                min_data_points_required=min_data_points
            )
            main_logger.info("ML Components initialized.")

            # Optional: Train ML models if they are not trained or mode is 'optimize'
            if not ml_components_instance.market_regime_detector.model_trained and \
                    (args.mode == 'optimize' or settings.get('ml.auto_train', False)):
                main_logger.info("Training Market Regime Detector model...")
                # Pass DataManager instance for data fetching
                ml_components_instance.market_regime_detector.train_model(
                    data_manager=data_manager,
                    timeframe=settings.get('timeframes.analysis', '1h'),
                    start_date=args.backtest_start or '2022-01-01',
                    # Use backtest dates if available, otherwise default
                    end_date=args.backtest_end or datetime.now().strftime('%Y-%m-%d')
                )
        except Exception as e:
            main_logger.error(f"Error initializing or training ML components: {e}", exc_info=True)
            ml_components_instance = None  # Disable ML if there's an error

    # Initialize StrategyRouter
    strategy_router_instance: Optional[StrategyRouter] = None
    if settings.get('strategy_router.enabled', False) or args.auto_strategy:
        strategy_router_instance = StrategyRouter(settings)
        main_logger.info("Strategy Router initialized.")
        if not ml_components_instance:
            main_logger.warning(
                "Strategy Router is enabled but ML components are not available. Automatic routing will not function.")

    # Initialize SafetyManager
    safety_manager_instance = SafetyManager(settings)  # Bot instance is set later by TradingBot
    main_logger.info("Safety Manager initialized.")

    # Determine initial strategy name
    initial_strategy_name = args.strategy
    if settings.get('strategy_router.enabled', False) or args.auto_strategy:
        initial_strategy_name = "auto_routed"  # A special name to indicate router control

    # Initialize TradingBot
    bot = TradingBot(
        mode=args.mode,
        strategy_name=initial_strategy_name,
        settings=settings,
        data_manager=data_manager,
        ml_components=ml_components_instance,
        strategy_router=strategy_router_instance,
        safety_manager=safety_manager_instance  # Pass safety manager instance
    )

    # Ensure safety_manager has a reference back to the bot for killswitch actions
    safety_manager_instance.set_trading_bot(bot)

    if args.dry_run:
        main_logger.info("Dry run complete. Exiting.")
        return

    # Start bot based on mode
    try:
        if args.status_only:
            bot.print_status()
        elif args.mode == 'backtest' or args.mode == 'optimize':
            main_logger.info(f"Running in {args.mode} mode.")
            # Ensure backtesting supports dynamic strategy if router is enabled
            if strategy_router_instance:
                # This would typically involve a specialized backtester that integrates with the router
                # For now, let's assume TradingBot's backtest method can handle this indirectly.
                # You might need to extend core/backtest_engine.py or ml_enhanced_backtesting.py
                main_logger.warning("Backtesting with Strategy Router requires specialized backtester implementation.")
                # Example: from core.ml_enhanced_backtesting import MLEnhancedBacktester
                # backtester = MLEnhancedBacktester(settings, strategy_router_instance)
                # backtester.run_backtest(...)

            # Simple backtest (if not using advanced backtester with router)
            main_logger.info(
                f"Starting backtest for {args.symbol} on {args.timeframe} from {args.backtest_start} to {args.backtest_end}...")
            bot.run_backtest(args.symbol, args.timeframe, args.backtest_start, args.backtest_end)

        else:  # Live or Paper trading
            main_logger.info(f"Starting bot in {args.mode} mode...")
            bot.start()
            # Keep main thread alive. Bot runs in background threads.
            while True:
                time.sleep(3600)  # Sleep for 1 hour, adjust as needed

    except KeyboardInterrupt:
        main_logger.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        main_logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        if bot.running:
            bot.stop()
        main_logger.info("Bot execution finished.")


if __name__ == "__main__":
    main()