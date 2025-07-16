import argparse
import logging
import os
import sys
import time
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Type  # Added Type import

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.trading_bot import TradingBot
from utils.logger import setup_logger
from strategies import STRATEGIES

# New imports
from ml_components import initialize_ml, get_ml_components
from core.strategy_router import StrategyRouter
from core.safety_manager import SafetyManager
from data_sources.data_manager import DataManager


def parse_arguments():
    """Parses command line arguments for the trading bot."""
    parser = argparse.ArgumentParser(
        description="Ultimate Crypto Trading Bot",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'mode',
        type=str,
        choices=['live', 'paper', 'backtest', 'optimize'],
        help="Bot mode: 'live' (real trading), 'paper' (simulated trading), "
             "'backtest' (historical data simulation), 'optimize' (parameter optimization)."
    )
    parser.add_argument(
        'strategy',
        type=str,
        default='default',
        choices=list(STRATEGIES.keys()) + ['default', 'autopilot'],
        help="Trading strategy to use. Choose from: " + ', '.join(STRATEGIES.keys()) +
             ", 'default' (Momentum), 'autopilot' (Grid + Arbitrage + Rebalancing)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help="Configuration profile name (e.g., 'aggressive', 'conservative', 'default'). "
             "Looks for <name>.json in config/profiles."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Only load and validate configuration, then exit. No bot execution."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging for more verbose output."
    )
    parser.add_argument(
        '--status-only',
        action='store_true',
        help="Only print current bot status (if running). No trading operations."
    )

    auto_group = parser.add_mutually_exclusive_group(required=False)
    auto_group.add_argument(
        '--auto-strategy',
        action='store_true',
        help="Automatically select the optimal strategy based on market regime detection."
    )
    auto_group.add_argument(
        '--validate-config',
        action='store_true',
        help="Validate the loaded configuration and exit."
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help="Run strategy parameter optimization before starting the bot (or as a standalone task)."
    )

    return parser.parse_args()


def auto_select_strategy(settings: Settings, data_manager: DataManager) -> str:
    """
    Intelligente Auto-Strategie-Auswahl basierend auf Marktbedingungen
    Uses MarketRegimeDetector to determine the optimal strategy.
    """
    logger = logging.getLogger('trading_bot')
    logger.info("🤖 Analyzing market conditions for optimal strategy selection...")

    try:
        ml_components = get_ml_components()
        if ml_components is None:
            ml_components = initialize_ml(settings=settings)

        update_status = ml_components.update_all_components(data_manager=data_manager)

        if update_status.get("regime_updated") and "current_regime" in update_status:
            market_regime_info = ml_components.get_current_regime_info()

            strategy_router = StrategyRouter(settings)
            selected_strategy = strategy_router.route_strategy(
                market_regime_info, {}
            )

            logger.info(f"\n🎯 AUTO-SELECTED STRATEGY: {selected_strategy.upper()}")
            logger.info(f"💡 Reason: Market is in {market_regime_info.get('label')} regime.")

            return selected_strategy
        else:
            logger.warning("Market regime detection failed or not available. Falling back to default strategy.")
            return settings.get('strategy_router.default_strategy', 'autopilot')

    except Exception as e:
        logger.error(f"⚠️  Auto-selection failed: {e}")
        logger.info("🔄 Falling back to AutoPilot strategy")
        return settings.get('strategy_router.default_strategy', 'autopilot')


def apply_intelligent_overrides(settings: Settings, args: argparse.Namespace) -> None:
    """Applies configuration overrides based on CLI arguments."""
    if args.dry_run:
        settings.set('mode', 'dry_run')

    if args.debug:
        settings.set('logging.level', 'DEBUG')

    if args.auto_strategy:
        settings.set('auto_strategy', True)
        settings.set('strategy_router.enabled', True)
        settings.set('ml.enabled', True)


def print_ultimate_banner():
    """Prints a welcoming banner."""
    print("""
      ██╗      ██╗    ██╗ ███████╗ ████████╗  ██████╗ ██╗  ██╗██╗     ██╗    ██╗███████╗ ██████╗
      ██║      ██║    ██║ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██║  ██║██║     ██║    ██║██╔════╝ ██╔════╝
      ██║      ████████║ ███████╗    ██║    ██║      ███████║██║     ████████║███████╗ ███████╗
      ██║      ██╔════██║ ╚════██║    ██║    ██║      ██╔════██║     ██╔════██║╚════██║ ╚════██║
      ███████╗ ██║    ██║ ███████║    ██║    ╚██████╗ ██║  ██║███████╗██║    ██║███████║ ███████╗
      ╚══════╝ ╚═╝    ╚═╝ ╚══════╝    ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝    ╚═╝╚══════╝ ╚══════╝

      Ultimate Crypto Trading Bot - AI Enhanced & Fully Automated
      🚀 Ready to make some magic happen? 🚀
    """)


def print_strategy_info(settings: Settings, strategy: str):
    """Print strategy-specific configuration info, enhanced for regime-adaptive strategies."""

    def print_autopilot_info(s: Settings):
        print("\n" + "=" * 70)
        print("🤖 AUTOPILOT STACK - INTELLIGENTE ALLOKATION")
        print("=" * 70)
        print("Autopilot combines multiple strategies for diversified income streams.")
        print(f"📊 Basisstrategie: {strategy.upper()}")
        print(f"💰 Aktuelle Kapitalallokation (Standard): {s.get('autopilot.capital_allocation', {})}")
        print("----------------------------------------------------------------------")
        print("Sub-Strategien: Grid Trading (Side/Low-Vol), Arbitrage (Vol/Disparity),")
        print("ML-Enhanced Momentum/Mean Reversion (Trends), DeFi Yield (Bear/Stability).")
        print("======================================================================")

    def print_basic_info(s: Settings, strat: str):
        print("\n" + "=" * 70)
        print(f"🧠 STRATEGY: {strat.upper()}")
        print("=" * 70)
        print(f"📊 Trading Pairs: {s.get('trading_pairs', [])}")
        print(f"⏱️  Timeframe: {s.get('timeframes.analysis', '1h')}")
        print(f"💰 Position Size: {s.get('risk.position_size', 0.05) * 100:.1f}% of balance per trade")
        print(f"🛑 Stop Loss: {s.get('risk.stop_loss', 0.03) * 100:.1f}%")
        print(f"🚀 Take Profit: {s.get('risk.take_profit', 0.06) * 100:.1f}%")
        print("=" * 70)

    if strategy == 'autopilot':
        print_autopilot_info(settings)
    elif settings.get('strategy_router.enabled', False):
        print("\n" + "=" * 70)
        print("🤖 INTELLIGENTE STRATEGIELAUSWAHL AKTIV")
        print("=" * 70)
        print(f"🎯 Basisstrategie: {strategy.upper()}")
        print(f"📊 Regime-basierte Strategien: {settings.get('strategy_router.regime_strategies', {})}")
        print(f"💰 Kapitalallokationsregeln: {settings.get('strategy_router.capital_allocation_rules', {})}")
        print(f"💡 ML-Modelle aktiviert: {settings.get('ml.enabled', False)}")
        print(f"🚨 Killswitch aktiviert: {settings.get('risk_management.killswitch.enabled', False)}")
        print("=" * 70)
    else:
        print_basic_info(settings, strategy)


def validate_configuration(settings: Settings, strategy: str) -> bool:
    """Validates the loaded configuration for the selected strategy."""
    logger = logging.getLogger('trading_bot')
    issues = []

    if not settings.get('trading_pairs'):
        issues.append("❌ No trading pairs defined.")
    if not settings.get('exchange.name'):
        issues.append("❌ Exchange name not defined.")

    if strategy not in STRATEGIES and strategy != 'autopilot':
        issues.append(f"❌ Unknown strategy '{strategy}'. Available: {list(STRATEGIES.keys()) + ['autopilot']}.")

    if settings.get('ml.enabled', False):
        if not settings.get('ml.data_dir'):
            issues.append("❌ ML is enabled but 'ml.data_dir' is not set.")
        if not settings.get('ml.models_dir'):
            issues.append("❌ ML is enabled but 'ml.models_dir' is not set.")

    if settings.get('strategy_router.enabled', False):
        if not settings.get('ml.enabled', False):
            issues.append("❌ Strategy Router is enabled but ML components are not enabled.")
        if not settings.get('strategy_router.regime_strategies'):
            issues.append("⚠️  Strategy Router is enabled but 'regime_strategies' mapping is empty.")
        # This check requires STRATEGIES imported from strategies/__init__.py
        # For simplicity in this isolated file, assuming STRATEGIES is globally accessible
        # or that this check can be re-evaluated within the main bot logic.
        for strat_name in settings.get('strategy_router.regime_strategies', {}).values():
            if strat_name.lower() not in STRATEGIES:
                issues.append(f"❌ Strategy '{strat_name}' in 'regime_strategies' is not a valid strategy.")

    if settings.get('risk_management.killswitch.enabled', False):
        if settings.get('risk_management.killswitch.max_drawdown', 0) <= 0:
            issues.append("❌ Killswitch enabled but 'killswitch.max_drawdown' is not set or invalid.")

    if issues:
        logger.error("\n💥 Konfigurationsvalidierungsfehler:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False

    logger.info("\n✅ Konfiguration erfolgreich validiert!")
    return True


def optimize_parameters(settings: Settings, strategy_class: Type, symbol: str, start_date: datetime,
                        end_date: datetime):
    """
    Runs parameter optimization for a given strategy.
    """
    logger = logging.getLogger('trading_bot')

    param_grid_base = {
        'lookback_period': [20, 50, 100],
        'rsi_period': [10, 14, 20]
    }

    param_grid_ml = {
        'lookback_period': [50, 100, 150],
        'prediction_threshold': [0.55, 0.6, 0.65]
    }

    if settings.get('ml.enabled', False) and 'ml_strategy' in strategy_class.__name__.lower():
        from core.ml_enhanced_backtesting import MLEnhancedBacktester
        backtester = MLEnhancedBacktester(settings, strategy_class)
        param_grid = param_grid_ml
        logger.info(f"Using MLEnhancedBacktester for optimization with {strategy_class.__name__}")
    else:
        from core.enhanced_backtesting import EnhancedBacktester
        backtester = EnhancedBacktester(settings, strategy_class)
        param_grid = param_grid_base
        logger.info(f"Using EnhancedBacktester for optimization with {strategy_class.__name__}")

    optimization_results = backtester.optimize_parameters(
        strategy_class,
        symbol,
        param_grid=param_grid,
        start_date=start_date,
        end_date=end_date
    )

    return optimization_results


def main():
    """Ultimate main function with intelligence"""
    print_ultimate_banner()
    args = parse_arguments()

    log_level = 'DEBUG' if args.debug else 'INFO'
    logger = setup_logger(name='trading_bot', level=log_level)

    print(f"\n🚀 Ultimate Trading Bot wird gestartet...")
    print(f"⚙️  Modus: {args.mode.upper()}")
    print(f"🧠 Strategie: {args.strategy.upper()}")
    print(f"📋 Konfiguration: {args.config}")

    if args.dry_run:
        print("🔍 DRY RUN MODUS - Nur Konfigurationsvorschau")

    try:
        print("📂 Konfiguration wird geladen...")
        settings = Settings()
        settings.load_profile(args.config)

        apply_intelligent_overrides(settings, args)

        data_manager = DataManager(settings)

        if args.auto_strategy:
            args.strategy = auto_select_strategy(settings, data_manager)
            settings.set('strategy_router.active_strategy', args.strategy)

        if args.status_only:
            bot_instance = TradingBot(
                mode=args.mode,
                strategy_name=args.strategy,
                settings=settings,
                data_manager=data_manager
            )
            status = bot_instance.get_status()
            print("\n📊 Aktueller Bot-Status:")
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)

        if args.validate_config:
            is_valid = validate_configuration(settings, args.strategy)
            sys.exit(0 if is_valid else 1)

        if not validate_configuration(settings, args.strategy):
            print("\n💥 Konfigurationsvalidierung fehlgeschlagen!")
            return 1

        print_strategy_info(settings, args.strategy)

        if args.dry_run:
            print("\n✅ Konfiguration erfolgreich validiert!")
            print("🔍 DRY RUN abgeschlossen - kein Bot gestartet")
            return 0

        print("\n🤖 Ultimate Trading Bot wird initialisiert...")
        bot = TradingBot(
            mode=args.mode,
            strategy_name=args.strategy,
            settings=settings,
            data_manager=data_manager
        )

        print("✅ Bot erfolgreich initialisiert!")

        if args.optimize:
            strategy_class = STRATEGIES.get(args.strategy.lower())
            if not strategy_class:
                logger.error(f"Strategie {args.strategy} nicht zur Optimierung gefunden.")
                return 1

            optimization_results = optimize_parameters(
                settings,
                strategy_class,
                settings.get('trading_pairs', ['BTC/USDT'])[0],
                datetime.strptime(settings.get('backtest.start_date'), '%Y-%m-%d'),
                datetime.strptime(settings.get('backtest.end_date'), '%Y-%m-%d')
            )
            logger.info("📈 Optimierungsergebnisse:")
            logger.info(f"  Beste Parameter: {optimization_results.get('best_params')}")
            logger.info(
                f"  Bester Metrikwert ({optimization_results.get('optimization_metric')}): {optimization_results.get('best_metric_value'):.2f}")

            if args.mode == 'backtest':
                return 0

        if args.mode == 'backtest':
            print("\n📊 Backtest wird gestartet...")
            start_time = time.time()

            if settings.get('ml.enabled', False) and args.strategy.lower() == 'ml':
                from core.ml_enhanced_backtesting import MLEnhancedBacktester
                backtester = MLEnhancedBacktester(settings, bot.strategy)
                results = backtester.run_with_ml(
                    symbols=settings.get('trading_pairs'),
                    source=settings.get('data.source', 'binance'),
                    timeframe=settings.get('timeframes.analysis', '1h'),
                    use_cache=settings.get('data.use_cache', True)
                )
                if 'ml_comparison' in results:
                    backtester.plot_ml_comparison(
                        os.path.join("data/backtest_results", settings.get('backtest.output_dir', 'latest')))
            else:
                from core.enhanced_backtesting import EnhancedBacktester
                backtester = EnhancedBacktester(settings, bot.strategy)
                results = backtester.run(
                    symbols=settings.get('trading_pairs'),
                    source=settings.get('data.source', 'binance'),
                    timeframe=settings.get('timeframes.analysis', '1h'),
                    use_cache=settings.get('data.use_cache', True)
                )

            duration = time.time() - start_time

            print("\n" + "=" * 80)
            print("📈 BACKTEST-ERGEBNISSE - ULTIMATE EDITION")
            print("=" * 80)
            print(f"⏱️  Dauer: {duration:.1f}s")
            print(f"📈 Gesamt-Return: {results.get('total_return', 0):.2f}%")
            print(f"🔢 Gesamtzahl Trades: {results.get('total_trades', 0)}")
            print(f"🎯 Gewinnrate: {results.get('statistics', {}).get('win_rate', 0):.2f}%")
            print(f"📊 Sharpe Ratio: {results.get('statistics', {}).get('sharpe_ratio', 0):.2f}")
            print(f"💰 Maximaler Drawdown: {abs(results.get('statistics', {}).get('max_drawdown', 0)):.2f}%")

            if results.get('total_trades', 0) > 0:
                print(f"💵 Durchschnittlicher Gewinn/Trade: {results.get('statistics', {}).get('avg_profit', 0):.2f}%")

            if args.strategy == 'grid_trading' and 'grid_status' in results:
                grid_stats = results['grid_status']
                print(f"📊 Grid-Effizienz: {grid_stats.get('grid_efficiency', 0):.1f}%")

            if 'ml_comparison' in results:
                comp = results['ml_comparison']
                print("\n=== ML-VERBESSERT vs. BASELINE ===")
                print(f"  Return-Verbesserung: {comp.get('return_improvement', 0):.2f}%")
                print(f"  Sharpe-Verbesserung: {comp.get('sharpe_improvement', 0):.2f}")
                print(f"  Drawdown-Reduzierung: {comp.get('drawdown_improvement', 0):.2f}%")
                print("===============================")

            print("=" * 80)

        else:
            print(f"\n🏃 {args.mode} Handel wird gestartet...")

            if args.strategy == 'grid_trading':
                print("💰 Grid Trading aktiv - Gewinne bei jeder Preisbewegung!")
            elif args.strategy == 'arbitrage':
                print("🔍 Arbitrage Bot aktiv - Risikofreie Gewinne jagen!")
            elif args.strategy == 'autopilot':
                print("🤖 AutoPilot Stack aktiv - Mehrere Einkommensströme laufen!")
                print("📊 Grid Trading + Arbitrage + Smart Rebalancing = Maximaler Gewinn!")
            else:
                print(f"🧠 {args.strategy.upper()} Strategie aktiv und überwacht Marktbedingungen.")

            print("\n⚠️  Drücken Sie Strg+C, um den Bot zu stoppen")
            print("📊 Überwachen Sie die Logs für Echtzeit-Handelsupdates")

            ml_components_instance = get_ml_components()
            if ml_components_instance is None:
                ml_components_instance = initialize_ml(settings=settings)

            safety_manager = SafetyManager(settings, bot)
            bot.set_safety_manager(safety_manager)

            bot.run()

    except KeyboardInterrupt:
        print("\n\n⏹️  Ultimate Trading Bot vom Benutzer gestoppt")
        print("💰 Überprüfen Sie Ihre Gewinne in den Logs!")

    except ImportError as e:
        print(f"\n❌ Importfehler: {e}")
        print("\n🔧 Fehlende Dateien:")
        if 'grid_trading' in str(e):
            print("  ❌ strategies/grid_trading.py")
        if 'arbitrage' in str(e):
            print("  ❌ strategies/arbitrage.py")
        if 'autopilot' in str(e):
            print("  ❌ strategies/autopilot.py")
        if 'market_regime' in str(e):
            print("  ❌ ml_components/market_regime.py")
        if 'strategy_router' in str(e):
            print("  ❌ core/strategy_router.py")
        if 'safety_manager' in str(e):
            print("  ❌ core/safety_manager.py")
        return 1

    except Exception as e:
        print(f"\n💥 Fehler: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print("💡 Verwenden Sie --debug für detaillierte Fehlerinformationen")
        return 1

    print("\n✅ Ultimate Trading Bot Sitzung abgeschlossen!")
    print("🚀 Bereit für die nächste Geld-verdienen-Sitzung!")
    return 0


if __name__ == "__main__":
    sys.exit(main())