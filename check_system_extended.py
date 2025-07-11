#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate System Check - Lazy Millionaire Stack Diagnostics
=========================================================
Überprüft ALLE Komponenten des Trading Bot Systems:

✅ Strategy Imports & Verfügbarkeit
✅ Konfigurationsdateien
✅ Trading Bot Initialisierung
✅ Grid Trading Funktionalität
✅ Arbitrage Bot Status
✅ AutoPilot Stack Integration
✅ Exchange Verbindungen
✅ Risk Management
✅ Performance Metriken
✅ Database & Data Storage
✅ API Keys & Credentials
✅ Logging System
✅ Backtest Engine
✅ Machine Learning Models
✅ Market Data Feeds
✅ Order Management System
✅ System Requirements

Gibt klare Diagnose und Lösungsvorschläge!
"""

import os
import sys
import json
import traceback
import subprocess
import platform
import pkg_resources
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import importlib.util

# Füge Projektpfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Druckt schöne Section-Header"""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


def print_success(message: str):
    """Druckt Success-Message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Druckt Error-Message"""
    print(f"❌ {message}")


def print_warning(message: str):
    """Druckt Warning-Message"""
    print(f"⚠️  {message}")


def print_info(message: str):
    """Druckt Info-Message"""
    print(f"📊 {message}")


def check_file_exists(filepath: str, description: str) -> bool:
    """Prüft ob Datei existiert"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print_success(f"{description}: {filepath} ({size} bytes)")
        return True
    else:
        print_error(f"{description} fehlt: {filepath}")
        return False


def check_strategy_imports() -> Dict[str, bool]:
    """Überprüft alle Strategy-Imports"""
    print_header("STRATEGY IMPORT CHECK")

    results = {}

    try:
        # Basis-Import
        print_info("Testing basic imports...")
        from strategies.strategy_base import Strategy
        print_success("Strategy base class imported")
        results['strategy_base'] = True
    except Exception as e:
        print_error(f"Strategy base import failed: {e}")
        results['strategy_base'] = False

    # Standard Strategien
    standard_strategies = {
        'momentum': 'strategies.momentum.MomentumStrategy',
        'mean_reversion': 'strategies.mean_reversion.MeanReversionStrategy',
        'ml_strategy': 'strategies.ml_strategy.MLStrategy',
        'grid_trading': 'strategies.grid_trading.GridStrategy',
        'arbitrage': 'strategies.arbitrage.ArbitrageStrategy',
        'autopilot': 'strategies.autopilot.AutoPilotStrategy'
    }

    for strategy_name, import_path in standard_strategies.items():
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            print_success(f"{strategy_name}: {class_name} imported")
            results[strategy_name] = True
        except Exception as e:
            print_error(f"{strategy_name}: Import failed - {e}")
            results[strategy_name] = False

    # Strategies Package Test
    try:
        import strategies
        strategy_map = getattr(strategies, 'STRATEGY_MAP', {})
        print_info(f"Available strategies in STRATEGY_MAP: {list(strategy_map.keys())}")

        if len(strategy_map) >= 6:
            print_success(f"STRATEGY_MAP complete: {len(strategy_map)} strategies")
        else:
            print_warning(f"STRATEGY_MAP incomplete: only {len(strategy_map)} strategies")

        results['strategy_map'] = len(strategy_map) >= 6
    except Exception as e:
        print_error(f"Strategies package import failed: {e}")
        results['strategy_map'] = False

    return results


def check_configuration_files() -> Dict[str, bool]:
    """Überprüft alle Konfigurationsdateien"""
    print_header("CONFIGURATION FILES CHECK")

    results = {}

    # Basis-Konfigurationsdateien
    config_files = {
        'settings.py': 'config/settings.py',
        'default.json': 'config/profiles/default.json',
        'grid.json': 'config/profiles/grid.json',
        'arbitrage.json': 'config/profiles/arbitrage.json',
        'autopilot.json': 'config/profiles/autopilot.json'
    }

    for name, filepath in config_files.items():
        results[name] = check_file_exists(filepath, f"Config {name}")

    # Konfiguration-Parsing Test
    try:
        from config.settings import Settings

        # Test Standard-Konfiguration
        settings = Settings('default')
        print_success("Default settings loading successful")

        # Test Grid-Konfiguration
        if os.path.exists('config/profiles/grid.json'):
            grid_settings = Settings('grid')
            grid_config = grid_settings.get('grid_trading')
            if grid_config:
                print_success("Grid configuration parsing successful")
                print_info(
                    f"  Grid range: ${grid_config.get('price_range', {}).get('lower', 0):,.0f} - ${grid_config.get('price_range', {}).get('upper', 0):,.0f}")
            else:
                print_warning("Grid configuration incomplete")

        # Test AutoPilot-Konfiguration
        if os.path.exists('config/profiles/autopilot.json'):
            autopilot_settings = Settings('autopilot')
            autopilot_config = autopilot_settings.get('autopilot')
            if autopilot_config:
                print_success("AutoPilot configuration parsing successful")
                allocation = autopilot_config.get('capital_allocation', {})
                print_info(f"  Capital allocation: {allocation}")
            else:
                print_warning("AutoPilot configuration incomplete")

        results['config_parsing'] = True
    except Exception as e:
        print_error(f"Configuration parsing failed: {e}")
        results['config_parsing'] = False

    return results


def check_trading_bot_initialization() -> Dict[str, bool]:
    """Überprüft Trading Bot Initialisierung"""
    print_header("TRADING BOT INITIALIZATION CHECK")

    results = {}

    try:
        from core.trading_bot import TradingBot
        from config.settings import Settings

        print_info("Testing bot initialization for each strategy...")

        # Test verschiedene Strategien
        test_strategies = ['momentum', 'grid_trading']

        # Füge weitere Strategien hinzu wenn verfügbar
        try:
            import strategies
            if hasattr(strategies, 'STRATEGY_MAP'):
                available_strategies = list(strategies.STRATEGY_MAP.keys())
                for strategy in ['arbitrage', 'autopilot']:
                    if strategy in available_strategies:
                        test_strategies.append(strategy)
        except:
            pass

        for strategy in test_strategies:
            try:
                settings = Settings('default')
                bot = TradingBot(
                    mode='paper',
                    strategy_name=strategy,
                    settings=settings
                )
                print_success(f"Bot initialization successful: {strategy}")
                results[f'bot_{strategy}'] = True
            except Exception as e:
                print_error(f"Bot initialization failed for {strategy}: {e}")
                results[f'bot_{strategy}'] = False

        results['bot_initialization'] = True

    except Exception as e:
        print_error(f"Trading bot import/initialization failed: {e}")
        results['bot_initialization'] = False

    return results


def check_exchange_connectivity() -> Dict[str, bool]:
    """Überprüft Exchange-Verbindungen"""
    print_header("EXCHANGE CONNECTIVITY CHECK")

    results = {}

    try:
        from core.exchange import ExchangeFactory
        from config.settings import Settings

        settings = Settings('default')

        # Paper Trading Exchange Test
        print_info("Testing paper trading exchange...")
        paper_exchange = ExchangeFactory.create(settings, 'paper')

        if paper_exchange.connect():
            print_success("Paper trading exchange connection successful")

            # Test basic functionality
            balance = paper_exchange.get_balance()
            print_info(f"Initial paper balance: ${balance:,.2f}")

            results['paper_exchange'] = True
        else:
            print_error("Paper trading exchange connection failed")
            results['paper_exchange'] = False

    except Exception as e:
        print_error(f"Exchange connectivity check failed: {e}")
        results['exchange_connectivity'] = False

    return results


def check_strategy_functionality() -> Dict[str, bool]:
    """Überprüft Strategy-spezifische Funktionalität"""
    print_header("STRATEGY FUNCTIONALITY CHECK")

    results = {}

    # Test Grid Trading
    try:
        from strategies.grid_trading import GridStrategy
        from config.settings import Settings
        import pandas as pd
        import numpy as np

        print_info("Testing Grid Trading functionality...")

        settings = Settings('default')
        # Setze Test-Parameter
        settings.set('grid_trading.price_range.lower', 100000)
        settings.set('grid_trading.price_range.upper', 120000)
        settings.set('grid_trading.num_grids', 10)
        settings.set('grid_trading.investment_per_grid', 500)

        grid_strategy = GridStrategy(settings)

        # Test Daten erstellen
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        prices = 110000 + np.cumsum(np.random.randn(100) * 1000)
        test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(100, 1000, 100)
        })

        # Test Signal-Generierung
        signal, signal_data = grid_strategy.generate_signal(test_df, 'BTC/USDT')

        print_success(f"Grid Trading signal generation: {signal}")
        print_info(f"  Signal data keys: {list(signal_data.keys())}")

        # Test Grid-Status
        status = grid_strategy.get_grid_status()
        print_info(f"  Grid status: {status['total_grids']} grids configured")

        results['grid_trading'] = True

    except Exception as e:
        print_error(f"Grid Trading functionality test failed: {e}")
        results['grid_trading'] = False

    # Test Arbitrage (falls vorhanden)
    try:
        from strategies.arbitrage import ArbitrageStrategy

        print_info("Testing Arbitrage functionality...")

        settings = Settings('default')
        settings.set('arbitrage.min_profit_threshold', 0.005)
        settings.set('arbitrage.max_position_size', 1000)

        arbitrage_strategy = ArbitrageStrategy(settings)

        # Test mit simulierten Daten
        signal, signal_data = arbitrage_strategy.generate_signal(test_df, 'BTC/USDT')
        print_success(f"Arbitrage signal generation: {signal}")

        status = arbitrage_strategy.get_arbitrage_status()
        print_info(f"  Arbitrage threshold: {status['min_profit_threshold']:.1f}%")

        results['arbitrage'] = True

    except Exception as e:
        print_warning(f"Arbitrage functionality test failed: {e}")
        results['arbitrage'] = False

    # Test AutoPilot (falls vorhanden)
    try:
        from strategies.autopilot import AutoPilotStrategy

        print_info("Testing AutoPilot functionality...")

        settings = Settings('default')
        autopilot_strategy = AutoPilotStrategy(settings)

        status = autopilot_strategy.get_autopilot_status()
        print_success(f"AutoPilot initialization: {status['active_strategies']} sub-strategies")

        results['autopilot'] = True

    except Exception as e:
        print_warning(f"AutoPilot functionality test failed: {e}")
        results['autopilot'] = False

    return results


def check_main_cli() -> Dict[str, bool]:
    """Überprüft main.py CLI Funktionalität"""
    print_header("MAIN.PY CLI CHECK")

    results = {}

    try:
        # Test main.py Import
        import main
        print_success("main.py import successful")

        # Test verfügbare Strategien in CLI
        import subprocess

        result = subprocess.run([
            sys.executable, 'main.py', '--help'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            help_output = result.stdout
            if 'autopilot' in help_output and 'arbitrage' in help_output:
                print_success("CLI help shows all strategies including autopilot and arbitrage")
            else:
                print_warning("CLI help missing some strategies")

            results['cli_help'] = True
        else:
            print_error("CLI help command failed")
            results['cli_help'] = False

        # Test dry-run
        result = subprocess.run([
            sys.executable, 'main.py', '--strategy=grid_trading', '--dry-run'
        ], capture_output=True, text=True, timeout=15)

        if result.returncode == 0 and 'Configuration validated successfully' in result.stdout:
            print_success("CLI dry-run test successful")
            results['cli_dryrun'] = True
        else:
            print_warning("CLI dry-run test had issues")
            results['cli_dryrun'] = False

    except Exception as e:
        print_error(f"Main.py CLI check failed: {e}")
        results['main_cli'] = False

    return results


def check_risk_management() -> Dict[str, bool]:
    """Überprüft Risk Management Komponenten"""
    print_header("RISK MANAGEMENT CHECK")

    results = {}

    try:
        # Risk Manager Import
        from core.risk_manager import RiskManager
        from config.settings import Settings

        print_info("Testing Risk Manager initialization...")
        settings = Settings('default')
        risk_manager = RiskManager(settings)

        print_success("Risk Manager imported successfully")

        # Test Risk Limits
        test_position_size = 1000
        max_allowed = risk_manager.calculate_max_position_size('BTC/USDT', 50000, 10000)
        print_info(f"Max position size calculation: ${max_allowed:,.2f}")

        # Test Stop Loss
        stop_loss = risk_manager.calculate_stop_loss(50000, 0.02)
        print_info(f"Stop loss at 2% risk: ${stop_loss:,.2f}")

        # Test Portfolio Risk
        portfolio_risk = risk_manager.get_portfolio_risk_metrics()
        print_info(f"Portfolio risk metrics: {list(portfolio_risk.keys())}")

        results['risk_manager'] = True

        # Test Risk Configurations
        risk_config = settings.get('risk_management', {})
        if risk_config:
            print_success(
                f"Risk management configured: max_position_size=${risk_config.get('max_position_size', 0):,.0f}")
            results['risk_config'] = True
        else:
            print_warning("Risk management configuration missing")
            results['risk_config'] = False

    except Exception as e:
        print_error(f"Risk management check failed: {e}")
        results['risk_management'] = False

    return results


def check_performance_metrics() -> Dict[str, bool]:
    """Überprüft Performance Tracking und Metriken"""
    print_header("PERFORMANCE METRICS CHECK")

    results = {}

    try:
        # Performance Tracker Import
        from analysis.performance_tracker import PerformanceTracker
        from config.settings import Settings

        print_info("Testing Performance Tracker...")
        settings = Settings('default')
        tracker = PerformanceTracker(settings)

        print_success("Performance Tracker imported successfully")

        # Test Metric Calculations
        metrics = tracker.get_performance_summary()
        print_info(f"Available metrics: {list(metrics.keys())}")

        # Test Trade History
        trade_history = tracker.get_trade_history(limit=10)
        print_info(f"Trade history records: {len(trade_history)}")

        results['performance_tracker'] = True

        # Test Reports Directory
        reports_dir = 'data/reports'
        if os.path.exists(reports_dir):
            report_files = os.listdir(reports_dir)
            print_success(f"Reports directory exists: {len(report_files)} files")
            results['reports_dir'] = True
        else:
            print_warning("Reports directory missing")
            results['reports_dir'] = False

    except Exception as e:
        print_error(f"Performance metrics check failed: {e}")
        results['performance_metrics'] = False

    return results


def check_database_storage() -> Dict[str, bool]:
    """Überprüft Database und Data Storage"""
    print_header("DATABASE & DATA STORAGE CHECK")

    results = {}

    # Check data directories
    data_dirs = {
        'market_data': 'data/market_data',
        'backtest_results': 'data/backtest_results',
        'ml_models': 'data/ml_models',
        'logs': 'logs',
        'reports': 'data/reports'
    }

    for name, path in data_dirs.items():
        if os.path.exists(path):
            file_count = len(os.listdir(path))
            print_success(f"{name} directory: {path} ({file_count} files)")
            results[f'dir_{name}'] = True
        else:
            print_warning(f"{name} directory missing: {path}")
            results[f'dir_{name}'] = False

    # Check database (if using)
    try:
        # SQLite check
        db_path = 'data/trading_bot.db'
        if os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            print_success(f"SQLite database found: {len(tables)} tables")
            results['sqlite_db'] = True
        else:
            print_info("No SQLite database found (optional)")
            results['sqlite_db'] = None

    except Exception as e:
        print_warning(f"Database check error: {e}")
        results['database'] = False

    return results


def check_api_credentials() -> Dict[str, bool]:
    """Überprüft API Keys und Credentials"""
    print_header("API KEYS & CREDENTIALS CHECK")

    results = {}

    # Check .env file
    env_path = '.env'
    if os.path.exists(env_path):
        print_success(".env file found")
        results['env_file'] = True

        # Check for required keys (without exposing values)
        try:
            from dotenv import load_dotenv
            load_dotenv()

            required_keys = [
                'BINANCE_API_KEY',
                'BINANCE_API_SECRET',
                'TWITTER_API_KEY',
                'TWITTER_API_SECRET',
                'TELEGRAM_BOT_TOKEN'
            ]

            missing_keys = []
            for key in required_keys:
                if os.getenv(key):
                    print_success(f"{key}: Configured")
                else:
                    print_warning(f"{key}: Not configured")
                    missing_keys.append(key)

            results['api_keys'] = len(missing_keys) == 0

        except ImportError:
            print_warning("python-dotenv not installed - cannot verify API keys")
            results['dotenv'] = False
    else:
        print_error(".env file missing - API credentials not configured")
        print_info("Create .env file with your API keys")
        results['env_file'] = False

    return results


def check_logging_system() -> Dict[str, bool]:
    """Überprüft Logging System"""
    print_header("LOGGING SYSTEM CHECK")

    results = {}

    try:
        # Logger Import
        from utils.logger import setup_logger

        test_logger = setup_logger('test_check')
        test_logger.info("Test log message")
        print_success("Logger setup successful")
        results['logger_setup'] = True

        # Check log directory
        log_dir = 'logs'
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            print_success(f"Logs directory exists: {len(log_files)} log files")

            # Check recent logs
            if log_files:
                latest_log = max([os.path.join(log_dir, f) for f in log_files],
                                 key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest_log))
                print_info(f"Latest log: {os.path.basename(latest_log)} (modified {mod_time})")

            results['log_directory'] = True
        else:
            print_warning("Logs directory missing")
            results['log_directory'] = False

    except Exception as e:
        print_error(f"Logging system check failed: {e}")
        results['logging_system'] = False

    return results


def check_backtest_engine() -> Dict[str, bool]:
    """Überprüft Backtest Engine"""
    print_header("BACKTEST ENGINE CHECK")

    results = {}

    try:
        # Backtest Engine Import
        from core.backtest_engine import BacktestEngine
        from config.settings import Settings

        print_info("Testing Backtest Engine...")
        settings = Settings('default')

        # Initialize engine
        engine = BacktestEngine(settings)
        print_success("Backtest Engine imported successfully")

        # Check backtest data
        backtest_data_dir = 'data/market_data'
        if os.path.exists(backtest_data_dir):
            data_files = [f for f in os.listdir(backtest_data_dir) if f.endswith('.csv')]
            print_info(f"Backtest data files available: {len(data_files)}")
            results['backtest_data'] = len(data_files) > 0
        else:
            print_warning("No backtest data directory found")
            results['backtest_data'] = False

        # Check backtest results
        results_dir = 'data/backtest_results'
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
            print_info(f"Previous backtest results: {len(result_files)} files")
            results['backtest_results'] = True
        else:
            print_warning("No backtest results directory")
            results['backtest_results'] = False

        results['backtest_engine'] = True

    except Exception as e:
        print_error(f"Backtest engine check failed: {e}")
        results['backtest_engine'] = False

    return results


def check_ml_models() -> Dict[str, bool]:
    """Überprüft Machine Learning Models"""
    print_header("MACHINE LEARNING MODELS CHECK")

    results = {}

    try:
        # ML Strategy Import
        from strategies.ml_strategy import MLStrategy
        print_success("ML Strategy imported successfully")
        results['ml_strategy'] = True

        # Check model files
        models_dir = 'data/ml_models'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.h5', '.pt'))]
            print_info(f"ML models found: {len(model_files)} files")

            for model_file in model_files:
                size = os.path.getsize(os.path.join(models_dir, model_file))
                print_info(f"  - {model_file} ({size / 1024 / 1024:.1f} MB)")

            results['ml_models'] = len(model_files) > 0
        else:
            print_warning("ML models directory missing")
            results['ml_models'] = False

        # Check ML dependencies
        ml_packages = ['scikit-learn', 'tensorflow', 'torch', 'xgboost']
        ml_available = []

        for package in ml_packages:
            try:
                __import__(package.replace('-', '_'))
                ml_available.append(package)
            except ImportError:
                pass

        if ml_available:
            print_success(f"ML packages available: {', '.join(ml_available)}")
            results['ml_packages'] = True
        else:
            print_warning("No ML packages installed")
            results['ml_packages'] = False

    except Exception as e:
        print_error(f"ML models check failed: {e}")
        results['ml_models'] = False

    return results


def check_market_data_feeds() -> Dict[str, bool]:
    """Überprüft Market Data Feeds"""
    print_header("MARKET DATA FEEDS CHECK")

    results = {}

    try:
        # Data Collector Import
        from core.data_collector import DataCollector
        from config.settings import Settings

        print_info("Testing Data Collector...")
        settings = Settings('default')
        collector = DataCollector(settings)

        print_success("Data Collector imported successfully")

        # Test data collection
        test_symbols = ['BTC/USDT', 'ETH/USDT']
        for symbol in test_symbols:
            try:
                # Test historical data
                data = collector.get_historical_data(symbol, '1h', limit=10)
                if data is not None and len(data) > 0:
                    print_success(f"{symbol}: Historical data available ({len(data)} candles)")
                    results[f'data_{symbol}'] = True
                else:
                    print_warning(f"{symbol}: No historical data")
                    results[f'data_{symbol}'] = False
            except Exception as e:
                print_error(f"{symbol}: Data collection failed - {e}")
                results[f'data_{symbol}'] = False

        results['data_collector'] = True

    except Exception as e:
        print_error(f"Market data feeds check failed: {e}")
        results['market_data'] = False

    return results


def check_order_management() -> Dict[str, bool]:
    """Überprüft Order Management System"""
    print_header("ORDER MANAGEMENT SYSTEM CHECK")

    results = {}

    try:
        # Order Manager Import
        from core.order_manager import OrderManager
        from config.settings import Settings

        print_info("Testing Order Manager...")
        settings = Settings('default')
        order_manager = OrderManager(settings)

        print_success("Order Manager imported successfully")

        # Test order types
        supported_orders = order_manager.get_supported_order_types()
        print_info(f"Supported order types: {supported_orders}")

        # Test order validation
        test_order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000
        }

        is_valid = order_manager.validate_order(test_order)
        print_info(f"Order validation test: {'Passed' if is_valid else 'Failed'}")

        results['order_manager'] = True

        # Check order history
        order_history = order_manager.get_order_history(limit=10)
        print_info(f"Order history records: {len(order_history)}")
        results['order_history'] = True

    except Exception as e:
        print_error(f"Order management check failed: {e}")
        results['order_management'] = False

    return results


def check_system_requirements() -> Dict[str, bool]:
    """Überprüft System Requirements"""
    print_header("SYSTEM REQUIREMENTS CHECK")

    results = {}

    # Python Version
    python_version = sys.version_info
    print_info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major == 3 and python_version.minor >= 8:
        print_success("Python version OK (3.8+)")
        results['python_version'] = True
    else:
        print_error("Python version too old (requires 3.8+)")
        results['python_version'] = False

    # Operating System
    os_info = platform.platform()
    print_info(f"Operating System: {os_info}")
    results['os_info'] = True

    # Required Packages
    required_packages = [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'ccxt>=2.0.0',
        'ta>=0.10.0',
        'scikit-learn>=1.0.0',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0'
    ]

    missing_packages = []
    outdated_packages = []

    for package_spec in required_packages:
        package_name = package_spec.split('>=')[0]
        min_version = package_spec.split('>=')[1] if '>=' in package_spec else None

        try:
            installed_version = pkg_resources.get_distribution(package_name).version
            if min_version:
                if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version):
                    print_success(f"{package_name}: {installed_version} ✓")
                else:
                    print_warning(f"{package_name}: {installed_version} (requires >={min_version})")
                    outdated_packages.append(package_name)
            else:
                print_success(f"{package_name}: {installed_version} ✓")
        except pkg_resources.DistributionNotFound:
            print_error(f"{package_name}: NOT INSTALLED")
            missing_packages.append(package_name)

    results['packages'] = len(missing_packages) == 0 and len(outdated_packages) == 0

    # Memory Check
    try:
        import psutil
        memory = psutil.virtual_memory()
        print_info(f"Available Memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
        results['memory'] = memory.available > 1024 * 1024 * 1024  # 1GB minimum
    except ImportError:
        print_warning("psutil not installed - cannot check memory")
        results['memory'] = None

    # Disk Space Check
    disk_usage = os.statvfs('.')
    free_space = disk_usage.f_bavail * disk_usage.f_frsize / 1024 / 1024 / 1024
    print_info(f"Free Disk Space: {free_space:.1f} GB")
    results['disk_space'] = free_space > 1  # 1GB minimum

    return results


def generate_system_report(all_results: Dict[str, Dict[str, bool]]) -> None:
    """Generiert System-Bericht mit Empfehlungen"""
    print_header("SYSTEM REPORT & RECOMMENDATIONS")

    # Berechne Gesamtstatistiken
    total_checks = 0
    passed_checks = 0
    critical_failures = []
    minor_issues = []

    for category, results in all_results.items():
        for check, result in results.items():
            if result is not None:  # Skip None values (optional checks)
                total_checks += 1
                if result:
                    passed_checks += 1
                else:
                    # Kategorisiere Fehler
                    critical_checks = [
                        'strategy_base', 'bot_initialization', 'config_parsing',
                        'python_version', 'packages', 'exchange_connectivity',
                        'risk_management', 'order_management'
                    ]

                    if check in critical_checks:
                        critical_failures.append(f"{category}.{check}")
                    else:
                        minor_issues.append(f"{category}.{check}")

    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

    print_info(f"OVERALL SYSTEM STATUS")
    print_info(f"  Total Checks: {total_checks}")
    print_info(f"  Passed: {passed_checks}")
    print_info(f"  Failed: {total_checks - passed_checks}")
    print_info(f"  Success Rate: {success_rate:.1f}%")

    # System-Status
    if success_rate >= 90:
        print_success("🎉 SYSTEM STATUS: EXCELLENT - Ready for production trading!")
    elif success_rate >= 75:
        print_success("✅ SYSTEM STATUS: GOOD - Minor improvements recommended")
    elif success_rate >= 50:
        print_warning("⚠️  SYSTEM STATUS: NEEDS WORK - Several important issues")
    else:
        print_error("❌ SYSTEM STATUS: CRITICAL - Major setup required")

    # Kritische Probleme
    if critical_failures:
        print_error("\nCRITICAL ISSUES (must fix before trading):")
        for issue in critical_failures:
            if 'grid_trading' in issue:
                print("  📁 Create: strategies/grid_trading.py")
            elif 'arbitrage' in issue:
                print("  📁 Create: strategies/arbitrage.py")
            elif 'autopilot' in issue:
                print("  📁 Create: strategies/autopilot.py")
            elif 'config_parsing' in issue:
                print("  📁 Create missing config files in config/profiles/")
            elif 'risk_management' in issue:
                print("  📁 Create: core/risk_manager.py")
            elif 'order_management' in issue:
                print("  📁 Create: core/order_manager.py")
            elif 'packages' in issue:
                print("  📦 Install missing packages: pip install -r requirements.txt")
            else:
                print(f"  ❌ {issue}")

    # Kleinere Probleme
    if minor_issues:
        print_warning("\nMINOR ISSUES (recommended fixes):")
        for issue in minor_issues:
            print(f"  ⚠️  {issue}")

    # Empfehlungen
    print_info("\nRECOMMENDATIONS:")

    if success_rate >= 90:
        print("  🚀 System ready for production trading!")
        print("  💰 Start with: python main.py --strategy=autopilot --config=autopilot")
        print("  📊 Monitor with: python main.py --monitor")
    elif success_rate >= 75:
        print("  🔧 Fix critical issues first, then start paper trading")
        print("  🧪 Test with: python main.py --mode=paper --strategy=grid_trading")
    else:
        print("  📁 Create missing core files (risk_manager.py, order_manager.py)")
        print("  📦 Install missing dependencies: pip install -r requirements.txt")
        print("  🔄 Re-run this check after fixes: python check_system.py")

    # Quick Fix Commands
    print_info("\nQUICK FIX COMMANDS:")

    missing_packages = []
    if missing_packages:
        print(f"  pip install {' '.join(missing_packages)}")

    if not os.path.exists('.env'):
        print("  cp .env.example .env  # Then add your API keys")

    missing_dirs = ['logs', 'data/market_data', 'data/backtest_results', 'data/ml_models']
    for dir_path in missing_dirs:
        if not os.path.exists(dir_path):
            print(f"  mkdir -p {dir_path}")


def save_diagnostic_report(all_results: Dict[str, Dict[str, bool]]) -> None:
    """Speichert detaillierten Diagnose-Bericht"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'system_diagnostic_{timestamp}.json'

    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print_success(f"\nDetailed diagnostic report saved: {report_path}")


def main():
    """Hauptfunktion für System-Check"""
    print("🤖 ULTIMATE SYSTEM CHECK - LAZY MILLIONAIRE STACK")
    print("=" * 80)
    print("🔍 Checking all components of your automated trading system...")
    print(f"📅 Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Führe alle Checks durch
    all_results['imports'] = check_strategy_imports()
    all_results['configs'] = check_configuration_files()
    all_results['trading_bot'] = check_trading_bot_initialization()
    all_results['exchange'] = check_exchange_connectivity()
    all_results['strategies'] = check_strategy_functionality()
    all_results['cli'] = check_main_cli()

    # Neue erweiterte Checks
    all_results['risk'] = check_risk_management()
    all_results['performance'] = check_performance_metrics()
    all_results['database'] = check_database_storage()
    all_results['credentials'] = check_api_credentials()
    all_results['logging'] = check_logging_system()
    all_results['backtest'] = check_backtest_engine()
    all_results['ml'] = check_ml_models()
    all_results['market_data'] = check_market_data_feeds()
    all_results['orders'] = check_order_management()
    all_results['system'] = check_system_requirements()

    # Generiere Bericht
    generate_system_report(all_results)

    # Speichere detaillierten Bericht
    save_diagnostic_report(all_results)

    print("\n" + "=" * 80)
    print("🏁 SYSTEM CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()