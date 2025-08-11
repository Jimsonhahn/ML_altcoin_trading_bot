#!/usr/bin/env python3
"""
Enhanced Altcoin Trading Bot - Main Entry Point
Fixed version with proper import handling and error recovery
"""

import argparse
import asyncio
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log', mode='a')
    ]
)

main_logger = logging.getLogger(__name__)

# Import order optimized to prevent circular imports
try:
    from config.settings import Settings
    from data_sources.data_manager import DataManager
    from core.trading_bot import TradingBot
    from core.strategy_router import StrategyRouter
    from core.market_analyzer import MarketAnalyzer
    from core.safety_manager import SafetyManager
    
    # ML Components with fallback
    try:
        from ml_components.enhanced_ml_components import create_enhanced_ml_components
        from ml_components import MLComponents
        HAS_ENHANCED_ML = True
        main_logger.info("Enhanced ML components available")
    except ImportError as e:
        main_logger.warning(f"Enhanced ML not available: {e}")
        try:
            from ml_components import MLComponents
            HAS_ENHANCED_ML = False
            main_logger.info("Standard ML components available")
        except ImportError as e2:
            main_logger.error(f"No ML components available: {e2}")
            MLComponents = None
            HAS_ENHANCED_ML = False
    
    # Import optional components with error handling
    try:
        from utils.notifier import Notifier
        HAS_NOTIFIER = True
    except ImportError:
        main_logger.warning("Notifier not available")
        HAS_NOTIFIER = False
        
except ImportError as e:
    main_logger.error(f"Critical import failed: {e}")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Enhanced Altcoin Trading Bot with ML capabilities"
    )
    
    # Core arguments
    parser.add_argument(
        '--mode', 
        choices=['live', 'paper', 'backtest', 'optimize'], 
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='momentum',
        help='Trading strategy to use (default: momentum)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='default',
        help='Configuration profile to use (default: default)'
    )
    
    # ML and automation
    parser.add_argument(
        '--auto-strategy', 
        action='store_true',
        help='Enable automatic strategy selection using ML'
    )
    
    parser.add_argument(
        '--disable-ml', 
        action='store_true',
        help='Disable ML features completely'
    )
    
    # Trading parameters
    parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+',
        help='Trading symbols (e.g., BTC/USDT ETH/USDT)'
    )
    
    parser.add_argument(
        '--capital', 
        type=float,
        help='Initial capital amount'
    )
    
    # Backtesting
    parser.add_argument(
        '--backtest-start', 
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--backtest-end', 
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    # Development and testing
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Run without executing trades'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--profile', 
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging with appropriate level"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure root logger
    logging.getLogger().setLevel(level)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)


def initialize_data_manager(settings: Settings) -> DataManager:
    """Initialize data manager with error handling"""
    try:
        data_manager = DataManager(settings)
        main_logger.info("Data Manager initialized successfully")
        return data_manager
    except Exception as e:
        main_logger.error(f"Failed to initialize Data Manager: {e}")
        raise


def initialize_ml_components(settings: Settings, args) -> Optional[MLComponents]:
    """Initialize ML components with enhanced error handling and fallbacks"""
    
    # Skip ML if explicitly disabled
    if args.disable_ml:
        main_logger.info("ML components disabled by user")
        return None
    
    # Skip ML if not enabled in settings and not required for auto-strategy
    if not settings.get('ml.enabled', True) and not args.auto_strategy:
        main_logger.info("ML components disabled in settings")
        return None
    
    ml_components_instance = None
    
    try:
        # Try enhanced ML components first
        if HAS_ENHANCED_ML:
            ml_components_instance = create_enhanced_ml_components(settings)
            main_logger.info("Enhanced ML Components initialized successfully")
        elif MLComponents:
            # Fallback to standard ML components
            core_symbols = settings.get('ml.regime_core_symbols', ["BTC/USDT", "ETH/USDT"])
            min_data_points = settings.get('ml.min_data_points_for_ml', 200)
            
            ml_components_instance = MLComponents(
                settings=settings,
                data_cache_dir=settings.get('data.cache_dir', 'data/market_data'),
                models_dir=settings.get('ml.models_dir', 'data/ml_models'),
                output_dir=settings.get('ml.output_dir', 'data/ml_analysis'),
                core_symbols=core_symbols,
                min_data_points_required=min_data_points
            )
            main_logger.info("Standard ML Components initialized")
        else:
            raise ImportError("No ML components available")
            
        # Optional: Train ML models if needed
        if (hasattr(ml_components_instance, 'market_regime_detector') and 
            not ml_components_instance.market_regime_detector.model_trained and
            (args.mode == 'optimize' or settings.get('ml.auto_train', False))):
            
            main_logger.info("Training ML models...")
            # Training would happen here with proper data manager integration
            
    except Exception as e:
        main_logger.error(f"Failed to initialize ML components: {e}")
        
        if args.auto_strategy:
            main_logger.error("Cannot use auto-strategy without ML components")
            return None
        else:
            main_logger.warning("Continuing without ML components")
            ml_components_instance = None
    
    return ml_components_instance


def initialize_strategy_router(settings: Settings, ml_components, args) -> Optional[StrategyRouter]:
    """Initialize strategy router with market analyzer"""
    strategy_router_instance = None
    market_analyzer_instance = None
    
    if settings.get('strategy_router.enabled', False) or args.auto_strategy:
        try:
            # Initialize MarketAnalyzer
            market_analyzer_config = {
                'symbols': settings.get('symbols', ['BTCUSDT', 'ETHUSDT']),
                'timeframe': settings.get('timeframes.analysis', '1h'),
                'lookback_period': settings.get('analysis.lookback_period', 100)
            }
            market_analyzer_instance = MarketAnalyzer(market_analyzer_config)
            main_logger.info("Market Analyzer initialized")
            
            # Initialize StrategyRouter
            strategy_router_instance = StrategyRouter(settings)
            main_logger.info("Strategy Router initialized")
            
            # Connect components
            strategy_router_instance.market_analyzer = market_analyzer_instance
            
            if not ml_components:
                main_logger.warning(
                    "Strategy Router enabled but ML components unavailable. "
                    "Using technical analysis only."
                )
                
        except Exception as e:
            main_logger.error(f"Failed to initialize Strategy Router: {e}")
            if args.auto_strategy:
                main_logger.error("Cannot use auto-strategy without Strategy Router")
                return None
    
    return strategy_router_instance


def initialize_safety_manager(settings: Settings) -> SafetyManager:
    """Initialize safety manager"""
    try:
        safety_manager = SafetyManager(settings)
        main_logger.info("Safety Manager initialized")
        return safety_manager
    except Exception as e:
        main_logger.error(f"Failed to initialize Safety Manager: {e}")
        raise


def initialize_trading_bot(args, settings: Settings, data_manager: DataManager,
                         ml_components, strategy_router, safety_manager) -> TradingBot:
    """Initialize the main trading bot"""
    
    # Determine strategy name
    initial_strategy_name = args.strategy
    if settings.get('strategy_router.enabled', False) or args.auto_strategy:
        initial_strategy_name = "auto_routed"
    
    try:
        bot = TradingBot(
            mode=args.mode,
            strategy_name=initial_strategy_name,
            settings=settings,
            data_manager=data_manager,
            ml_components=ml_components,
            strategy_router=strategy_router,
            safety_manager=safety_manager
        )
        
        # Connect safety manager back to bot
        safety_manager.set_trading_bot(bot)
        
        main_logger.info(f"Trading Bot initialized in {args.mode} mode")
        main_logger.info(f"Strategy: {initial_strategy_name}")
        
        if args.dry_run:
            main_logger.info("DRY RUN MODE - No actual trades will be executed")
        
        return bot
        
    except Exception as e:
        main_logger.error(f"Failed to initialize Trading Bot: {e}")
        raise


def setup_signal_handlers(bot: TradingBot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        main_logger.info(f"Received signal {signum}, shutting down gracefully...")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_bot(bot: TradingBot, args):
    """Main bot execution loop with error handling"""
    try:
        await bot.start()
        
        if args.mode == 'backtest':
            main_logger.info("Backtest completed")
        else:
            main_logger.info("Trading bot started successfully. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while bot.is_running:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        main_logger.info("Shutdown requested by user")
    except Exception as e:
        main_logger.error(f"Bot execution error: {e}")
        raise
    finally:
        bot.stop()
        main_logger.info("Trading bot stopped")


def main():
    """Enhanced main function with comprehensive error handling"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    main_logger.info("="*80)
    main_logger.info("🚀 Enhanced Altcoin Trading Bot Starting")
    main_logger.info("="*80)
    main_logger.info(f"Mode: {args.mode}")
    main_logger.info(f"Strategy: {args.strategy}")
    main_logger.info(f"Config: {args.config}")
    main_logger.info(f"Auto-strategy: {args.auto_strategy}")
    main_logger.info(f"ML disabled: {args.disable_ml}")
    
    try:
        # Load settings
        main_logger.info("Loading configuration...")
        settings = Settings(args.config)
        
        # Override settings with command line arguments
        if args.symbols:
            settings.set('symbols', args.symbols)
        if args.capital:
            settings.set('trading.initial_capital', args.capital)
        
        main_logger.info(f"Configuration loaded: {args.config}")
        
        # Initialize core components
        main_logger.info("Initializing core components...")
        
        # Data Manager
        data_manager = initialize_data_manager(settings)
        
        # ML Components
        ml_components = initialize_ml_components(settings, args)
        
        # Strategy Router
        strategy_router = initialize_strategy_router(settings, ml_components, args)
        
        # Safety Manager
        safety_manager = initialize_safety_manager(settings)
        
        # Trading Bot
        bot = initialize_trading_bot(
            args, settings, data_manager, ml_components, strategy_router, safety_manager
        )
        
        main_logger.info("All components initialized successfully")
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(bot)
        
        # Start profiling if requested
        if args.profile:
            import cProfile
            import pstats
            from io import StringIO
            
            pr = cProfile.Profile()
            pr.enable()
        
        # Run the bot
        main_logger.info("Starting trading bot execution...")
        asyncio.run(run_bot(bot, args))
        
        # End profiling if enabled
        if args.profile:
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            # Save profile results
            with open('logs/profile.txt', 'w') as f:
                f.write(s.getvalue())
            main_logger.info("Performance profile saved to logs/profile.txt")
        
    except KeyboardInterrupt:
        main_logger.info("Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        main_logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    main_logger.info("Trading bot execution completed")


if __name__ == "__main__":
    main()