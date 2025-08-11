#!/usr/bin/env python3
"""
🚀 Start Enhanced Trading Bot
Wrapper um deinen bestehenden Bot mit Intelligence Features
"""

import asyncio
import logging
import sys
from pathlib import Path

# Projekt-Root
sys.path.insert(0, str(Path(__file__).parent))

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBotLauncher:
    def __init__(self):
        self.enhanced_logger = None
        self.intelligence_api_process = None
    
    async def setup_enhanced_logger(self):
        """Setup Enhanced Logger mit Mock Database"""
        try:
            from core.enhanced_decision_logger import create_enhanced_decision_logger
            
            # Mock Database Pool
            class MockDBPool:
                async def acquire(self):
                    return MockDBConnection()
                async def close(self):
                    pass

            class MockDBConnection:
                async def fetchrow(self, query, *args):
                    return None
                async def fetch(self, query, *args):
                    return []
                async def execute(self, query, *args):
                    pass
                async def fetchval(self, query, *args):
                    return 0
                def __aenter__(self):
                    return self
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            mock_pool = MockDBPool()
            
            self.enhanced_logger = await create_enhanced_decision_logger(
                db_pool=mock_pool,
                export_path="intelligence_exports/",
                dashboard_updates=True,
                learning_enabled=True
            )
            
            logger.info("✅ Enhanced Logger bereit")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced Logger Setup fehlgeschlagen: {e}")
            return False
    
    def start_intelligence_api(self):
        """Starte Intelligence API parallel"""
        import subprocess
        import os
        
        try:
            # Starte Intelligence API im Hintergrund
            self.intelligence_api_process = subprocess.Popen([
                sys.executable, 
                "run_intelligence_api.py",
                "--port", "8002"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info("✅ Intelligence API gestartet (Port 8002)")
            return True
            
        except Exception as e:
            logger.error(f"Intelligence API Start fehlgeschlagen: {e}")
            return False
    
    async def run_original_bot_with_enhancements(self, args):
        """Führe original Bot mit Enhancements aus"""
        try:
            # Import deine original TradingBotApplication
            from main import TradingBotApplication
            
            # Original App erstellen
            app = TradingBotApplication()
            
            # Standard Initialisierung
            app.setup_logging(args.verbose)
            app.initialize_components(args)
            
            # ENHANCEMENT: Enhanced Logger in Bot integrieren
            if self.enhanced_logger and hasattr(app.trading_bot, 'decision_logger'):
                # Original logger stoppen
                if hasattr(app.trading_bot.decision_logger, 'stop'):
                    try:
                        await app.trading_bot.decision_logger.stop()
                    except:
                        pass
                
                # Enhanced logger einsetzen
                app.trading_bot.decision_logger = self.enhanced_logger
                logger.info("🧠 Enhanced Logger in Trading Bot integriert!")
                
                # Test log decision
                test_decision = {
                    'strategy': args.strategy,
                    'mode': args.mode,
                    'enhancement': 'integrated'
                }
                
                test_context = {
                    'startup': True,
                    'enhanced_features': True,
                    'intelligence_api': True
                }
                
                await self.enhanced_logger.log_trading_decision_with_context(
                    decision_data=test_decision,
                    market_context=test_context,
                    strategy_reasoning=f"Enhanced bot started with {args.strategy} strategy in {args.mode} mode",
                    confidence_level=1.0
                )
                
                logger.info("✅ Integration Test Decision geloggt")
            
            # Enhanced startup info
            logger.info("=" * 60)
            logger.info("🚀 ENHANCED TRADING BOT GESTARTET")
            logger.info("=" * 60)
            logger.info(f"Original Mode: {args.mode}")
            logger.info(f"Original Strategy: {args.strategy}")
            logger.info(f"Enhanced Logger: ✅")
            logger.info(f"Intelligence API: http://localhost:8002/api/intelligence/demo")
            logger.info(f"Export Path: intelligence_exports/")
            logger.info("=" * 60)
            
            # Original Bot starten
            await app.run()
            
        except Exception as e:
            logger.error(f"Enhanced Bot Execution Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down Enhanced Bot...")
        
        if self.enhanced_logger:
            await self.enhanced_logger.stop()
            logger.info("Enhanced Logger gestoppt")
        
        if self.intelligence_api_process:
            self.intelligence_api_process.terminate()
            logger.info("Intelligence API gestoppt")

async def main():
    """Enhanced Main Entry Point"""
    import argparse
    
    # Parse arguments (gleiche wie original)
    parser = argparse.ArgumentParser(description="Enhanced Altcoin Trading Bot")
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], default='paper')
    parser.add_argument('--strategy', type=str, default='momentum')
    parser.add_argument('--config-profile', type=str, default='default')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--auto-strategy', action='store_true')
    parser.add_argument('--disable-ml', action='store_true')
    
    args = parser.parse_args()
    
    launcher = EnhancedBotLauncher()
    
    try:
        # Setup Enhanced Features
        logger.info("🚀 Starte Enhanced Trading Bot Setup...")
        
        enhanced_ok = await launcher.setup_enhanced_logger()
        api_ok = launcher.start_intelligence_api()
        
        logger.info(f"Setup Status: Enhanced Logger: {'✅' if enhanced_ok else '❌'}, API: {'✅' if api_ok else '❌'}")
        
        # Warte kurz damit API startet
        await asyncio.sleep(2)
        
        # Starte Enhanced Bot
        await launcher.run_original_bot_with_enhancements(args)
        
    except KeyboardInterrupt:
        logger.info("Beendigung durch Benutzer angefordert")
    except Exception as e:
        logger.error(f"Enhanced Bot Error: {e}")
    finally:
        await launcher.shutdown()

if __name__ == "__main__":
    asyncio.run(main())