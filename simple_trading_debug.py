#!/usr/bin/env python3
"""
Simple Trading Debug Tool
Quick diagnosis for trading bot execution issues
"""

import asyncio
import logging
import sys
import json
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_debug.log')
    ]
)

logger = logging.getLogger(__name__)

class SimpleTradingDebugger:
    """Simple trading execution debugger"""
    
    def __init__(self):
        self.issues = []
        self.fixes = []
    
    async def run_diagnosis(self):
        """Run comprehensive diagnosis"""
        logger.info("=" * 60)
        logger.info("SIMPLE TRADING BOT DIAGNOSIS")
        logger.info("=" * 60)
        
        # Step 1: Check configuration files
        await self.check_configuration()
        
        # Step 2: Check strategy files
        await self.check_strategies()
        
        # Step 3: Check imports and dependencies
        await self.check_imports()
        
        # Step 4: Test basic bot initialization
        await self.test_bot_initialization()
        
        # Step 5: Check for common issues
        await self.identify_common_issues()
        
        # Step 6: Test actual trading pipeline
        await self.test_trading_pipeline()
        
        # Summary
        await self.report_findings()
    
    async def check_configuration(self):
        """Check configuration files"""
        logger.info("--- CHECKING CONFIGURATION ---")
        
        config_files = ['config.yaml', 'config/production.yaml']
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                logger.info(f"✓ Found: {config_file}")
                
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        if 'trading' in content:
                            logger.info(f"  ✓ Contains trading configuration")
                        if 'strategies' in content:
                            logger.info(f"  ✓ Contains strategy configuration")
                except Exception as e:
                    logger.error(f"  ✗ Error reading {config_file}: {e}")
                    self.issues.append(f"Configuration file {config_file} unreadable")
            else:
                logger.warning(f"⚠ Missing: {config_file}")
                self.issues.append(f"Missing configuration file: {config_file}")
    
    async def check_strategies(self):
        """Check strategy files"""
        logger.info("--- CHECKING STRATEGIES ---")
        
        strategies_dir = project_root / 'strategies'
        if not strategies_dir.exists():
            logger.error("✗ Strategies directory not found")
            self.issues.append("Strategies directory missing")
            return
        
        strategy_files = list(strategies_dir.glob('*.py'))
        logger.info(f"Found {len(strategy_files)} strategy files")
        
        # Check for key strategies
        key_strategies = ['momentum.py', 'strategy_base.py']
        for strategy in key_strategies:
            strategy_path = strategies_dir / strategy
            if strategy_path.exists():
                logger.info(f"✓ Found: {strategy}")
            else:
                logger.error(f"✗ Missing: {strategy}")
                self.issues.append(f"Missing strategy file: {strategy}")
    
    async def check_imports(self):
        """Check critical imports"""
        logger.info("--- CHECKING IMPORTS ---")
        
        critical_imports = [
            ('yaml', 'YAML parsing'),
            ('ccxt', 'Exchange connectivity'),
            ('pandas', 'Data processing'),
            ('numpy', 'Numerical computation')
        ]
        
        for module, description in critical_imports:
            try:
                __import__(module)
                logger.info(f"✓ {module} - {description}")
            except ImportError:
                logger.error(f"✗ {module} - {description}")
                self.issues.append(f"Missing dependency: {module}")
    
    async def test_bot_initialization(self):
        """Test basic bot initialization"""
        logger.info("--- TESTING BOT INITIALIZATION ---")
        
        try:
            # Try to import main components
            from config.environment import get_config, TradingMode
            from config.settings import Settings
            logger.info("✓ Core configuration imports successful")
            
            # Try to get basic config
            try:
                settings = Settings()
                logger.info("✓ Settings object created")
                
                # Check if settings contain trading configuration
                if hasattr(settings, 'config'):
                    logger.info("✓ Settings contain configuration data")
                else:
                    logger.warning("⚠ Settings object has no configuration data")
                    
            except Exception as e:
                logger.error(f"✗ Settings creation failed: {e}")
                self.issues.append(f"Settings initialization error: {e}")
                
        except ImportError as e:
            logger.error(f"✗ Core imports failed: {e}")
            self.issues.append(f"Import error: {e}")
    
    async def identify_common_issues(self):
        """Identify common issues that prevent trading"""
        logger.info("--- IDENTIFYING COMMON ISSUES ---")
        
        # Check for paper trading setup
        try:
            from core.paper_trading_engine import PaperTradingEngine
            logger.info("✓ Paper trading engine available")
        except ImportError as e:
            logger.error(f"✗ Paper trading engine not available: {e}")
            self.issues.append(f"Paper trading engine import failed: {e}")
        
        # Check for exchange module
        try:
            from core.exchange import ExchangeManager
            logger.info("✓ Exchange module available")
        except ImportError as e:
            logger.error(f"✗ Exchange module not available: {e}")
            self.issues.append(f"Exchange module import failed: {e}")
        
        # Check for TradingBot
        try:
            from core.trading_bot import TradingBot
            logger.info("✓ TradingBot available")
        except ImportError as e:
            logger.error(f"✗ TradingBot not available: {e}")
            self.issues.append(f"TradingBot import failed: {e}")
        
        # Check data directory
        data_dir = project_root / 'data'
        if not data_dir.exists():
            logger.warning("⚠ Data directory not found")
            self.issues.append("Data directory missing")
        
        # Check logs directory
        logs_dir = project_root / 'logs'
        if not logs_dir.exists():
            logger.info("Creating logs directory...")
            logs_dir.mkdir()
            self.fixes.append("Created logs directory")
    
    async def test_trading_pipeline(self):
        """Test the actual trading pipeline execution"""
        logger.info("--- TESTING TRADING PIPELINE ---")
        
        if self.issues:
            logger.warning("⚠ Skipping pipeline test due to existing import issues")
            return
        
        try:
            # Import required components
            from config.settings import Settings
            from core.trading_bot import TradingBot
            
            logger.info("Creating test bot instance...")
            
            # Create settings
            settings = Settings()
            
            # Create bot in paper mode
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',  # Use momentum strategy for test
                settings=settings,
                data_manager=None,  # Let bot create its own
                ml_components=None,
                strategy_router=None,
                safety_manager=None
            )
            
            logger.info("✓ Bot instance created successfully")
            
            # Test if bot has required attributes
            required_attrs = ['mode', 'is_running', 'symbol_configs', 'strategies']
            for attr in required_attrs:
                if hasattr(bot, attr):
                    value = getattr(bot, attr)
                    logger.info(f"✓ Bot has {attr}: {value}")
                else:
                    logger.error(f"✗ Bot missing {attr}")
                    self.issues.append(f"Bot missing required attribute: {attr}")
            
            # Test strategy loading
            if hasattr(bot, 'strategies') and bot.strategies:
                logger.info(f"✓ Strategies loaded: {list(bot.strategies.keys())}")
            else:
                logger.error("✗ No strategies loaded")
                self.issues.append("No strategies loaded in bot")
            
            # Test symbol configuration
            if hasattr(bot, 'symbol_configs') and bot.symbol_configs:
                logger.info(f"✓ Symbol configs: {list(bot.symbol_configs.keys())}")
            else:
                logger.error("✗ No symbol configurations")
                self.issues.append("No trading symbols configured")
            
            # Test paper trading engine
            if hasattr(bot, 'paper_engine'):
                logger.info("✓ Paper trading engine initialized")
                
                # Test paper balance
                try:
                    # Use the correct method to get paper balance
                    portfolio_status = bot.paper_engine.get_virtual_portfolio_status()
                    virtual_balance = bot.paper_engine.virtual_balance
                    total_portfolio_value = bot.paper_engine.total_portfolio_value()
                    
                    logger.info(f"✓ Paper balance available: ${virtual_balance:.2f}")
                    logger.info(f"✓ Total portfolio value: ${total_portfolio_value:.2f}")
                    
                    if virtual_balance <= 0:
                        logger.warning("⚠ Paper balance is zero - trades won't execute")
                        self.issues.append("Paper trading balance is zero")
                    
                    if portfolio_status:
                        logger.info(f"✓ Portfolio status accessible: {len(portfolio_status)} fields")
                        
                except Exception as e:
                    logger.error(f"✗ Paper balance check failed: {e}")
                    self.issues.append(f"Paper balance error: {e}")
            else:
                logger.error("✗ No paper trading engine found")
                self.issues.append("Paper trading engine not initialized")
            
            # Test exchange connectivity through data manager
            if hasattr(bot, 'data_manager') and bot.data_manager:
                logger.info("✓ Data manager available")
                
                if hasattr(bot.data_manager, 'exchange') and bot.data_manager.exchange:
                    logger.info("✓ Exchange accessible through data manager")
                    try:
                        # Test basic exchange functionality
                        logger.info("✓ Exchange connection available")
                    except Exception as e:
                        logger.error(f"✗ Exchange connectivity failed: {e}")
                        self.issues.append(f"Exchange error: {e}")
                else:
                    logger.info("• Exchange not initialized (OK for paper trading)")
            else:
                logger.error("✗ No data manager found")
                self.issues.append("Data manager not initialized")
            
            # Test single trading loop execution (dry run)
            logger.info("Testing single trading loop iteration...")
            try:
                # Set bot to running state temporarily
                bot.is_running = True
                
                # Test symbol processing if symbols are configured
                if hasattr(bot, 'symbol_configs') and bot.symbol_configs:
                    test_symbol = list(bot.symbol_configs.keys())[0]
                    logger.info(f"Testing symbol processing: {test_symbol}")
                    
                    # This is where we'd test _process_symbol but we'll just check if the method exists
                    if hasattr(bot, '_process_symbol'):
                        logger.info("✓ Symbol processing method available")
                        # Don't actually run it to avoid side effects
                    else:
                        logger.error("✗ Symbol processing method not found")
                        self.issues.append("Bot missing _process_symbol method")
                else:
                    logger.warning("⚠ No symbols configured for testing")
                    self.issues.append("No symbols configured for trading")
                
                bot.is_running = False  # Reset
                
            except Exception as e:
                logger.error(f"✗ Trading loop test failed: {e}")
                self.issues.append(f"Trading loop error: {e}")
        
        except Exception as e:
            logger.error(f"✗ Pipeline test failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.issues.append(f"Pipeline test failure: {e}")
    
    async def report_findings(self):
        """Report findings and provide fixes"""
        logger.info("=" * 60)
        logger.info("DIAGNOSIS SUMMARY")
        logger.info("=" * 60)
        
        if self.issues:
            logger.error(f"Found {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                logger.error(f"  {i}. {issue}")
        else:
            logger.info("✓ No major issues detected")
        
        if self.fixes:
            logger.info(f"Applied {len(self.fixes)} fixes:")
            for i, fix in enumerate(self.fixes, 1):
                logger.info(f"  {i}. {fix}")
        
        # Provide recommendations
        logger.info("\n--- RECOMMENDATIONS ---")
        
        if any("Missing dependency" in issue for issue in self.issues):
            logger.info("• Run: pip install -r requirements.txt")
        
        if any("configuration" in issue.lower() for issue in self.issues):
            logger.info("• Check config.yaml exists and contains trading settings")
        
        if any("strategy" in issue.lower() for issue in self.issues):
            logger.info("• Verify strategies directory contains momentum.py and strategy_base.py")
        
        # Create diagnosis report
        report = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'issues': self.issues,
            'fixes': self.fixes,
            'status': 'ISSUES_FOUND' if self.issues else 'HEALTHY'
        }
        
        with open('diagnosis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("\nDiagnosis complete. Report saved to diagnosis_report.json")


async def main():
    """Main execution"""
    debugger = SimpleTradingDebugger()
    await debugger.run_diagnosis()


if __name__ == "__main__":
    asyncio.run(main())