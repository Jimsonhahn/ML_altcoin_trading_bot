#!/usr/bin/env python3
"""
Trading Execution Debug Tool
Comprehensive debugging and diagnostics for trading bot execution pipeline
"""

import asyncio
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if main.py exists and use it to start the bot
from main import create_config_manager


class TradingExecutionDebugger:
    """Comprehensive trading execution debugger"""
    
    def __init__(self):
        self.debug_log = self.setup_debug_logging()
        self.bot = None
        self.config_manager = None
        self.container = None
        
    def setup_debug_logging(self):
        """Setup detailed debug logging"""
        debug_logger = logging.getLogger('trading_execution_debug')
        debug_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        debug_logger.handlers.clear()
        
        # File handler for debug logs
        log_file = 'trading_execution_debug.log'
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        debug_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        debug_logger.addHandler(console_handler)
        
        return debug_logger
        
    async def initialize_bot(self, mode='paper'):
        """Initialize trading bot for debugging"""
        try:
            self.debug_log.info(f"=== INITIALIZING BOT IN {mode.upper()} MODE ===")
            
            # Initialize config manager
            self.config_manager = ConfigManager()
            self.config_manager.load_config()
            
            # Initialize dependency container
            self.container = DependencyContainer()
            self.container.configure(
                config=self.config_manager,
                mode=mode,
                enable_ml=False,  # Disable ML for debugging
                enable_notifications=False  # Disable notifications for debugging
            )
            
            # Get trading bot from container
            self.bot = self.container.get_trading_bot()
            
            self.debug_log.info("✓ Bot initialized successfully")
            return True
            
        except Exception as e:
            self.debug_log.error(f"✗ Bot initialization failed: {e}")
            self.debug_log.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def diagnose_complete_pipeline(self):
        """Run complete trading pipeline diagnosis"""
        self.debug_log.info("=" * 80)
        self.debug_log.info("COMPLETE TRADING PIPELINE DIAGNOSIS")
        self.debug_log.info("=" * 80)
        
        # Step 1: Check bot status
        await self.check_bot_status()
        
        # Step 2: Test exchange connectivity
        await self.test_exchange_connectivity()
        
        # Step 3: Test market data pipeline
        await self.test_market_data_pipeline()
        
        # Step 4: Test strategy signal generation
        await self.test_strategy_signal_generation()
        
        # Step 5: Test risk management
        await self.test_risk_management()
        
        # Step 6: Test trade execution pipeline
        await self.test_trade_execution_pipeline()
        
        # Step 7: Identify common issues
        await self.identify_common_issues()
        
        # Step 8: Run single trading loop iteration
        await self.test_single_trading_loop()
        
        self.debug_log.info("=" * 80)
        self.debug_log.info("DIAGNOSIS COMPLETE")
        self.debug_log.info("=" * 80)
    
    async def check_bot_status(self):
        """Check bot initialization and status"""
        self.debug_log.info("--- CHECKING BOT STATUS ---")
        
        try:
            self.debug_log.info(f"Bot instance: {type(self.bot)}")
            self.debug_log.info(f"Bot running status: {self.bot.is_running if self.bot else 'No bot instance'}")
            self.debug_log.info(f"Bot mode: {self.bot.mode if self.bot else 'Unknown'}")
            self.debug_log.info(f"Symbol configs: {len(self.bot.symbol_configs) if self.bot and self.bot.symbol_configs else 0}")
            
            if self.bot and hasattr(self.bot, 'strategies'):
                self.debug_log.info(f"Active strategies: {list(self.bot.strategies.keys()) if self.bot.strategies else 'None'}")
            
            self.debug_log.info("✓ Bot status check complete")
            
        except Exception as e:
            self.debug_log.error(f"✗ Bot status check failed: {e}")
    
    async def test_exchange_connectivity(self):
        """Test exchange connectivity and credentials"""
        self.debug_log.info("--- TESTING EXCHANGE CONNECTIVITY ---")
        
        try:
            if not self.bot or not hasattr(self.bot, 'exchange'):
                self.debug_log.error("✗ No exchange instance available")
                return
            
            exchange = self.bot.exchange
            self.debug_log.info(f"Exchange type: {type(exchange)}")
            
            # Test basic connectivity
            if hasattr(exchange, 'test_connection'):
                connected = await exchange.test_connection()
                self.debug_log.info(f"Exchange connection test: {'✓ PASS' if connected else '✗ FAIL'}")
            
            # Test balance retrieval
            try:
                balance = await exchange.get_balance()
                self.debug_log.info(f"Balance retrieval: ✓ PASS")
                self.debug_log.info(f"Sample balance data: {dict(list(balance.items())[:3]) if balance else 'None'}")
            except Exception as e:
                self.debug_log.error(f"Balance retrieval: ✗ FAIL - {e}")
            
            # Test market data access
            try:
                markets = await exchange.get_markets()
                self.debug_log.info(f"Markets retrieval: ✓ PASS ({len(markets)} markets)")
            except Exception as e:
                self.debug_log.error(f"Markets retrieval: ✗ FAIL - {e}")
                
        except Exception as e:
            self.debug_log.error(f"✗ Exchange connectivity test failed: {e}")
    
    async def test_market_data_pipeline(self):
        """Test market data retrieval and processing"""
        self.debug_log.info("--- TESTING MARKET DATA PIPELINE ---")
        
        try:
            if not self.bot:
                self.debug_log.error("✗ No bot instance available")
                return
            
            # Test with first configured symbol
            if not self.bot.symbol_configs:
                self.debug_log.error("✗ No symbol configurations found")
                return
            
            test_symbol = list(self.bot.symbol_configs.keys())[0]
            self.debug_log.info(f"Testing with symbol: {test_symbol}")
            
            # Test market data retrieval
            market_data = await self.bot._fetch_market_data(test_symbol)
            
            if market_data:
                self.debug_log.info(f"✓ Market data retrieved successfully")
                self.debug_log.info(f"Data keys: {list(market_data.keys())}")
                
                # Check for required fields
                required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                for field in required_fields:
                    if field in market_data:
                        self.debug_log.info(f"  ✓ {field}: present")
                    else:
                        self.debug_log.warning(f"  ⚠ {field}: missing")
            else:
                self.debug_log.error("✗ No market data retrieved")
                
        except Exception as e:
            self.debug_log.error(f"✗ Market data pipeline test failed: {e}")
            self.debug_log.error(f"Traceback: {traceback.format_exc()}")
    
    async def test_strategy_signal_generation(self):
        """Test strategy signal generation"""
        self.debug_log.info("--- TESTING STRATEGY SIGNAL GENERATION ---")
        
        try:
            if not self.bot or not hasattr(self.bot, 'strategies'):
                self.debug_log.error("✗ No strategies available")
                return
            
            if not self.bot.strategies:
                self.debug_log.error("✗ No active strategies found")
                return
            
            # Test each strategy
            for strategy_name, strategy in self.bot.strategies.items():
                self.debug_log.info(f"Testing strategy: {strategy_name}")
                
                try:
                    # Get test symbol and market data
                    test_symbol = list(self.bot.symbol_configs.keys())[0]
                    market_data = await self.bot._fetch_market_data(test_symbol)
                    
                    if not market_data:
                        self.debug_log.warning(f"  ⚠ No market data for strategy test")
                        continue
                    
                    # Test signal generation
                    signals = await strategy.generate_signals(test_symbol, market_data)
                    
                    if signals:
                        self.debug_log.info(f"  ✓ Generated {len(signals)} signals")
                        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                            self.debug_log.info(f"    Signal {i+1}: {signal}")
                    else:
                        self.debug_log.info(f"  • No signals generated (market conditions may not be met)")
                        
                except Exception as e:
                    self.debug_log.error(f"  ✗ Strategy {strategy_name} failed: {e}")
                    
        except Exception as e:
            self.debug_log.error(f"✗ Strategy signal generation test failed: {e}")
    
    async def test_risk_management(self):
        """Test risk management system"""
        self.debug_log.info("--- TESTING RISK MANAGEMENT ---")
        
        try:
            if not self.bot or not hasattr(self.bot, 'risk_manager'):
                self.debug_log.error("✗ No risk manager available")
                return
            
            risk_manager = self.bot.risk_manager
            self.debug_log.info(f"Risk manager type: {type(risk_manager)}")
            
            # Test risk settings
            if hasattr(risk_manager, 'get_current_settings'):
                settings = risk_manager.get_current_settings()
                self.debug_log.info(f"Risk settings: {settings}")
            
            # Create test signal for risk validation
            test_signal = {
                'symbol': list(self.bot.symbol_configs.keys())[0],
                'action': 'BUY',
                'amount': 0.01,
                'price': None,
                'strategy': 'debug_test',
                'confidence': 0.8
            }
            
            # Test risk approval
            if hasattr(risk_manager, 'validate_signal'):
                approved = await risk_manager.validate_signal(test_signal)
                self.debug_log.info(f"Risk validation result: {'✓ APPROVED' if approved else '✗ REJECTED'}")
            else:
                self.debug_log.warning("⚠ Risk manager doesn't have validate_signal method")
                
        except Exception as e:
            self.debug_log.error(f"✗ Risk management test failed: {e}")
    
    async def test_trade_execution_pipeline(self):
        """Test trade execution pipeline"""
        self.debug_log.info("--- TESTING TRADE EXECUTION PIPELINE ---")
        
        try:
            if not self.bot:
                self.debug_log.error("✗ No bot instance available")
                return
            
            # Create test signal
            test_signal = {
                'symbol': list(self.bot.symbol_configs.keys())[0],
                'action': 'BUY',
                'amount': 0.01,
                'price': None,
                'strategy': 'debug_test',
                'confidence': 0.8
            }
            
            self.debug_log.info(f"Testing with signal: {test_signal}")
            
            # Test signal processing (without actual execution)
            if hasattr(self.bot, '_process_signals'):
                # This is a dry run to test the pipeline
                self.debug_log.info("Testing signal processing pipeline...")
                # Don't actually execute, just test the pathway
            
            # Check if order manager is available
            if hasattr(self.bot, 'order_manager'):
                self.debug_log.info("✓ Order manager available")
            else:
                self.debug_log.warning("⚠ No order manager found")
            
            # Check if position manager is available
            if hasattr(self.bot, 'position_manager'):
                self.debug_log.info("✓ Position manager available")
            else:
                self.debug_log.warning("⚠ No position manager found")
                
        except Exception as e:
            self.debug_log.error(f"✗ Trade execution pipeline test failed: {e}")
    
    async def identify_common_issues(self):
        """Identify common issues that prevent trading"""
        self.debug_log.info("--- IDENTIFYING COMMON ISSUES ---")
        
        issues_found = []
        
        try:
            # Check 1: Bot configuration
            if not self.bot:
                issues_found.append("Bot not initialized")
            elif not self.bot.is_running:
                issues_found.append("Bot not in running state")
            
            # Check 2: Symbol configuration
            if not self.bot or not self.bot.symbol_configs:
                issues_found.append("No trading symbols configured")
            
            # Check 3: Strategy configuration
            if not self.bot or not hasattr(self.bot, 'strategies') or not self.bot.strategies:
                issues_found.append("No active strategies loaded")
            
            # Check 4: Exchange connectivity
            if self.bot and hasattr(self.bot, 'exchange'):
                try:
                    balance = await self.bot.exchange.get_balance()
                    if not balance:
                        issues_found.append("Cannot retrieve account balance")
                except:
                    issues_found.append("Exchange connectivity issues")
            else:
                issues_found.append("No exchange connection")
            
            # Check 5: Paper trading balance (if in paper mode)
            if self.bot and self.bot.mode == 'paper':
                # Check paper trading balance
                if hasattr(self.bot, 'paper_engine'):
                    try:
                        paper_balance = self.bot.paper_engine.get_balance()
                        if paper_balance.get('USDT', 0) <= 0:
                            issues_found.append("Paper trading balance is zero or negative")
                    except:
                        issues_found.append("Paper trading engine issues")
            
            # Report findings
            if issues_found:
                self.debug_log.warning("Issues found:")
                for i, issue in enumerate(issues_found, 1):
                    self.debug_log.warning(f"  {i}. {issue}")
            else:
                self.debug_log.info("✓ No obvious issues detected")
            
            return issues_found
            
        except Exception as e:
            self.debug_log.error(f"✗ Issue identification failed: {e}")
            return ["Issue identification process failed"]
    
    async def test_single_trading_loop(self):
        """Test a single trading loop iteration"""
        self.debug_log.info("--- TESTING SINGLE TRADING LOOP ITERATION ---")
        
        try:
            if not self.bot:
                self.debug_log.error("✗ No bot instance available")
                return
            
            # Temporarily set bot to running state
            original_state = self.bot.is_running
            self.bot.is_running = True
            
            # Test single symbol processing
            if self.bot.symbol_configs:
                test_symbol = list(self.bot.symbol_configs.keys())[0]
                self.debug_log.info(f"Testing single iteration for symbol: {test_symbol}")
                
                try:
                    await self.bot._process_symbol(test_symbol)
                    self.debug_log.info("✓ Single trading loop iteration completed")
                except Exception as e:
                    self.debug_log.error(f"✗ Single trading loop iteration failed: {e}")
                    self.debug_log.error(f"Traceback: {traceback.format_exc()}")
            else:
                self.debug_log.error("✗ No symbols to test")
            
            # Restore original state
            self.bot.is_running = original_state
            
        except Exception as e:
            self.debug_log.error(f"✗ Single trading loop test failed: {e}")
    
    async def apply_common_fixes(self, issues):
        """Apply fixes for common issues"""
        self.debug_log.info("--- APPLYING COMMON FIXES ---")
        
        fixes_applied = []
        
        for issue in issues:
            try:
                if "Paper trading balance is zero" in issue:
                    if self.bot and hasattr(self.bot, 'paper_engine'):
                        # Initialize paper balance
                        self.bot.paper_engine.initialize_balance({'USDT': 10000})
                        fixes_applied.append("Initialized paper trading balance to $10,000")
                        self.debug_log.info("✓ Applied fix: Paper trading balance initialized")
                
                elif "Bot not in running state" in issue:
                    if self.bot:
                        self.bot.is_running = True
                        fixes_applied.append("Set bot to running state")
                        self.debug_log.info("✓ Applied fix: Bot set to running state")
                
            except Exception as e:
                self.debug_log.error(f"✗ Failed to apply fix for '{issue}': {e}")
        
        if fixes_applied:
            self.debug_log.info("Fixes applied:")
            for fix in fixes_applied:
                self.debug_log.info(f"  • {fix}")
        else:
            self.debug_log.info("No automatic fixes applied")
        
        return fixes_applied


async def main():
    """Main debug execution"""
    debugger = TradingExecutionDebugger()
    
    print("=" * 80)
    print("TRADING EXECUTION DEBUGGER")
    print("=" * 80)
    
    # Initialize bot
    if await debugger.initialize_bot(mode='paper'):
        # Run complete diagnosis
        await debugger.diagnose_complete_pipeline()
        
        # Identify issues
        issues = await debugger.identify_common_issues()
        
        # Apply fixes if issues found
        if issues:
            fixes = await debugger.apply_common_fixes(issues)
            
            if fixes:
                print("\nRe-running diagnosis after fixes...")
                await debugger.diagnose_complete_pipeline()
    
    print("\nDebug log saved to: trading_execution_debug.log")
    print("Review the log file for detailed analysis")


if __name__ == "__main__":
    asyncio.run(main())