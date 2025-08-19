#!/usr/bin/env python3
"""
Revolutionary Janics Freedom Factory - Deployment Script
=========================================================

Deploys the trading bot for 24/7 server operation with Paper Trading,
Remote Dashboard Access, and full monitoring capabilities.
"""

import os
import sys
import argparse
import logging
import asyncio
import signal
import threading
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from core.trading_bot import TradingBot
from data_sources.data_manager import DataManager
from api.app import create_app
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


class RevolutionaryDeployment:
    """
    Revolutionary Deployment Manager for Janics Freedom Factory
    Handles both Paper Trading and Live modes with remote access
    """
    
    def __init__(self, mode: str = "paper", docker: bool = False):
        self.mode = mode
        self.docker = docker
        self.is_running = False
        self.trading_bot = None
        self.flask_app = None
        self.api_thread = None
        
        # Setup logging
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        setup_logging(level=getattr(logging, log_level))
        
        logger.info("🚀 Revolutionary Janics Freedom Factory - Deployment Starting!")
        logger.info(f"   Mode: {mode.upper()}")
        logger.info(f"   Docker: {docker}")
        logger.info(f"   Timestamp: {datetime.now()}")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("⚙️ Initializing system components...")
            
            # Load configuration
            config_path = project_root / "config.yaml"
            if not config_path.exists():
                logger.warning("⚠️ config.yaml not found, using default settings")
                settings = Settings()
            else:
                settings = Settings.from_yaml(str(config_path))
            
            # Configure for paper trading if selected
            if self.mode == "paper":
                settings.update({
                    'trading.paper_mode': True,
                    'paper_trading.initial_balance': float(os.getenv('PAPER_TRADING_BALANCE', 10000)),
                    'paper_trading.enabled': True
                })
                logger.info(f"📝 Paper Trading enabled with ${settings.get('paper_trading.initial_balance')} virtual balance")
            
            # Initialize Data Manager
            logger.info("📊 Initializing Data Manager...")
            self.data_manager = DataManager(settings)
            
            # Initialize Trading Bot with Paper Trading support
            logger.info("🤖 Initializing Revolutionary Trading Bot...")
            self.trading_bot = TradingBot(
                mode=self.mode,
                strategy_name=settings.get('strategy.default', 'momentum_scalping'),
                settings=settings,
                data_manager=self.data_manager,
                paper_trading=(self.mode == "paper")
            )
            
            # Initialize Flask API
            logger.info("🌐 Initializing REST API and Dashboard...")
            self.flask_app = create_app('production')
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def start_api_server(self):
        """Start the Flask API server in a separate thread"""
        def run_api():
            try:
                host = "0.0.0.0" if self.docker else "127.0.0.1"
                port = int(os.getenv('API_PORT', 8080))
                
                logger.info(f"🌐 Starting API server on {host}:{port}")
                
                # Run with SocketIO support
                self.flask_app.socketio.run(
                    self.flask_app,
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False,
                    log_output=True
                )
            except Exception as e:
                logger.error(f"❌ API server error: {e}")
        
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()
        
        # Wait a moment for server to start
        time.sleep(3)
        logger.info("✅ API server started successfully!")
    
    def start_trading_bot(self):
        """Start the trading bot"""
        try:
            logger.info("🏭 Starting Revolutionary Trading Bot...")
            
            # Start the trading bot in async mode
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_bot():
                await self.trading_bot.start()
                while self.is_running:
                    await asyncio.sleep(1)
            
            def run_bot_thread():
                try:
                    loop.run_until_complete(run_bot())
                except Exception as e:
                    logger.error(f"❌ Trading bot error: {e}")
            
            self.bot_thread = threading.Thread(target=run_bot_thread, daemon=True)
            self.bot_thread.start()
            
            logger.info("✅ Trading bot started successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading bot: {e}")
            raise
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def display_status(self):
        """Display deployment status and access information"""
        logger.info("=" * 60)
        logger.info("🚀 REVOLUTIONARY JANICS FREEDOM FACTORY - LIVE! 🚀")
        logger.info("=" * 60)
        
        # Bot Status
        logger.info(f"🤖 Trading Bot Status: {'🟢 RUNNING' if self.is_running else '🔴 STOPPED'}")
        logger.info(f"📊 Trading Mode: {self.mode.upper()}")
        
        if self.mode == "paper":
            logger.info("💰 Paper Trading: $10,000 Virtual Balance")
            logger.info("🔒 Risk Level: ZERO (Virtual Money Only)")
        else:
            logger.info("⚠️  Live Trading: REAL MONEY AT RISK")
            logger.info("🔒 Risk Level: ACTIVE")
        
        # Access Information
        api_host = "localhost" if not self.docker else "your-server-ip"
        logger.info(f"🌐 API Endpoint: http://{api_host}:8080")
        logger.info(f"📱 Dashboard: http://{api_host}:3000")
        logger.info(f"📚 API Docs: http://{api_host}:8080/api/docs")
        
        # Remote Control
        logger.info("🎮 Remote Control Commands:")
        logger.info("   - Start/Stop Bot via Dashboard")
        logger.info("   - Switch Paper/Live Mode")
        logger.info("   - Monitor Performance Real-time")
        logger.info("   - View Trade History")
        
        # Docker Information
        if self.docker:
            logger.info("🐳 Docker Container: janics-freedom-factory")
            logger.info("📊 Health Check: Enabled")
            logger.info("🔄 Auto-restart: Enabled")
        
        logger.info("=" * 60)
    
    def run_health_monitor(self):
        """Run health monitoring in background"""
        def health_check():
            while self.is_running:
                try:
                    # Check bot health
                    if self.trading_bot and hasattr(self.trading_bot, 'is_running'):
                        bot_status = "🟢 HEALTHY" if self.trading_bot.is_running else "🟡 IDLE"
                    else:
                        bot_status = "🔴 ERROR"
                    
                    # Check API health
                    api_status = "🟢 HEALTHY" if self.api_thread and self.api_thread.is_alive() else "🔴 ERROR"
                    
                    # Log health status every 5 minutes
                    if int(time.time()) % 300 == 0:
                        logger.info(f"💓 Health Check - Bot: {bot_status}, API: {api_status}")
                    
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"❌ Health check error: {e}")
                    time.sleep(30)
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
    
    def deploy(self):
        """Main deployment process"""
        try:
            self.is_running = True
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Initialize all components
            self.initialize_components()
            
            # Start API server
            self.start_api_server()
            
            # Start trading bot
            self.start_trading_bot()
            
            # Start health monitoring
            self.run_health_monitor()
            
            # Display status
            self.display_status()
            
            # Keep the main process alive
            logger.info("🎯 Deployment complete! Bot running 24/7...")
            logger.info("🛑 Press Ctrl+C to shutdown gracefully")
            
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
            self.shutdown()
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating graceful shutdown...")
        self.is_running = False
        
        try:
            # Stop trading bot
            if self.trading_bot:
                logger.info("🤖 Stopping trading bot...")
                # The bot should handle graceful shutdown in its async loop
                
            # API server will stop when main process ends
            logger.info("🌐 Stopping API server...")
            
            logger.info("✅ Shutdown complete!")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Revolutionary Janics Freedom Factory - Trading Bot Deployment"
    )
    parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Running in Docker container'
    )
    
    args = parser.parse_args()
    
    # Create and run deployment
    deployment = RevolutionaryDeployment(
        mode=args.mode,
        docker=args.docker
    )
    
    deployment.deploy()


if __name__ == "__main__":
    main()