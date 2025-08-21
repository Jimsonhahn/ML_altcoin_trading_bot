"""
Server Bot Wrapper - Erweitert den bestehenden Trading Bot für Server-Betrieb
=============================================================================

Dieser Wrapper nutzt die bestehende Bot-Funktionalität und fügt Server-Features hinzu.
"""

import os
import sys
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import TradingBotApplication
from core.trading_bot import TradingBot
from config.settings import Settings
from core.di_container import DIContainer
from utils.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class ServerBotWrapper:
    """Wrapper für den bestehenden Trading Bot mit Server-Features"""
    
    def __init__(self):
        self.bot_instance: Optional[TradingBotApplication] = None
        self.trading_bot: Optional[TradingBot] = None
        self.container: Optional[DIContainer] = None
        self.is_running = False
        self.current_mode = 'paper'
        self.start_time = None
        self.trade_history = []
        self.performance_metrics = {}
        
    def initialize(self, mode: str = 'paper', strategy: str = 'momentum'):
        """Initialisiert den Bot mit gewünschten Einstellungen"""
        try:
            # Erstelle Kommandozeilen-Argumente für bestehenden Bot
            class Args:
                def __init__(self, mode, strategy):
                    self.mode = mode
                    self.strategy = strategy
                    self.config_profile = 'default'
                    self.auto_strategy = False
                    self.disable_ml = False
                    self.symbols = None
                    self.capital = None
                    self.dry_run = False
                    self.verbose = False
                    
            args = Args(mode, strategy)
            
            # Erstelle Bot-Instanz
            self.bot_instance = TradingBotApplication()
            
            # Logging einrichten
            self.bot_instance.setup_logging(args.verbose)
            
            # Initialisiere Container und Komponenten
            self.bot_instance.initialize_components(args)
            self.container = self.bot_instance.di_container
            self.trading_bot = self.bot_instance.trading_bot
            
            self.current_mode = mode
            logger.info(f"Bot initialized in {mode} mode with {strategy} strategy")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            return False
    
    def start_bot(self, mode: str = 'paper', strategy: str = 'momentum') -> Dict[str, Any]:
        """Startet den Trading Bot"""
        try:
            if self.is_running:
                return {'success': False, 'message': 'Bot is already running'}
            
            # Initialisiere Bot mit gewünschten Einstellungen
            if not self.initialize(mode, strategy):
                return {'success': False, 'message': 'Failed to initialize bot'}
            
            # Starte Bot in separatem Thread
            self.is_running = True
            self.start_time = datetime.now()
            
            self.bot_thread = threading.Thread(target=self._run_bot)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            return {
                'success': True, 
                'message': f'Bot started in {mode} mode with {strategy} strategy',
                'mode': mode,
                'strategy': strategy
            }
            
        except Exception as e:
            self.is_running = False
            return {'success': False, 'message': f'Failed to start bot: {str(e)}'}
    
    def stop_bot(self) -> Dict[str, Any]:
        """Stoppt den Trading Bot sicher"""
        try:
            if not self.is_running:
                return {'success': False, 'message': 'Bot is not running'}
            
            self.is_running = False
            
            # Shutdown bot gracefully
            if self.bot_instance:
                self.bot_instance.shutdown()
            
            # Wait for thread to finish
            if hasattr(self, 'bot_thread'):
                self.bot_thread.join(timeout=10)
            
            return {'success': True, 'message': 'Bot stopped successfully'}
            
        except Exception as e:
            return {'success': False, 'message': f'Failed to stop bot: {str(e)}'}
    
    def _run_bot(self):
        """Interne Methode um Bot in Thread zu starten"""
        try:
            # Nutze die run() Methode des bestehenden Bots (async)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.bot_instance.run())
        except Exception as e:
            logger.error(f"Bot crashed: {str(e)}")
            self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Bot-Status zurück"""
        return {
            'running': self.is_running,
            'mode': self.current_mode,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'last_update': datetime.now().isoformat()
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Holt Portfolio-Zusammenfassung vom Trading Bot"""
        try:
            if not self.trading_bot:
                return {'error': 'Bot not initialized'}
            
            # Hole Daten vom bestehenden Bot
            positions = self.trading_bot.get_positions() if hasattr(self.trading_bot, 'get_positions') else []
            balance = self.trading_bot.get_balance() if hasattr(self.trading_bot, 'get_balance') else {}
            
            # Berechne Portfolio-Wert
            total_value = sum(balance.get('total', {}).values()) if isinstance(balance, dict) else 0
            
            return {
                'total_value': total_value,
                'total_balance': total_value,  # Alias für Kompatibilität
                'positions': len(positions),
                'daily_pnl': self._calculate_daily_pnl(),
                'win_rate': self._calculate_win_rate(),
                'active_trades': len([p for p in positions if p.get('status') == 'open']),
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {str(e)}")
            return {'error': str(e)}
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Gibt aktive Trades zurück"""
        try:
            if not self.trading_bot:
                return []
            
            positions = self.trading_bot.get_positions() if hasattr(self.trading_bot, 'get_positions') else []
            
            active_trades = []
            for pos in positions:
                if pos.get('status') == 'open':
                    active_trades.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'amount': pos.get('amount'),
                        'entry_price': pos.get('entry_price', 0),
                        'current_price': pos.get('current_price', 0),
                        'pnl': pos.get('pnl', 0),
                        'pnl_percent': pos.get('pnl_percent', 0),
                        'strategy': pos.get('strategy', 'unknown'),
                        'opened_at': pos.get('opened_at', '')
                    })
            
            return active_trades
            
        except Exception as e:
            logger.error(f"Failed to get active trades: {str(e)}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt Performance-Metriken zurück"""
        try:
            return {
                'total_pnl': self._calculate_total_pnl(),
                'daily_pnl': self._calculate_daily_pnl(),
                'weekly_pnl': self._calculate_weekly_pnl(),
                'win_rate': self._calculate_win_rate(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown(),
                'total_trades': len(self.trade_history),
                'winning_trades': len([t for t in self.trade_history if t.get('pnl', 0) > 0]),
                'losing_trades': len([t for t in self.trade_history if t.get('pnl', 0) < 0])
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    def switch_mode(self, new_mode: str) -> Dict[str, Any]:
        """Wechselt zwischen Paper und Live Mode"""
        try:
            if self.is_running:
                # Stop current bot
                self.stop_bot()
                time.sleep(2)  # Wait for clean shutdown
                
                # Start with new mode
                return self.start_bot(mode=new_mode, strategy=self.trading_bot.active_strategy if self.trading_bot else 'momentum')
            else:
                self.current_mode = new_mode
                return {'success': True, 'message': f'Mode switched to {new_mode}'}
                
        except Exception as e:
            return {'success': False, 'message': f'Failed to switch mode: {str(e)}'}
    
    # Helper methods for calculations
    def _calculate_daily_pnl(self) -> float:
        """Berechnet täglichen P&L"""
        try:
            if self.trading_bot and self.trading_bot.paper_trading and self.trading_bot.paper_engine:
                # Get daily P&L from paper trading engine
                return getattr(self.trading_bot.paper_engine, 'daily_pnl', 0.0)
            
            # Fallback: calculate from trade history
            today = datetime.now().date()
            daily_trades = [t for t in self.trade_history if 
                           datetime.fromisoformat(t.get('timestamp', '2023-01-01T00:00:00')).date() == today]
            return sum(t.get('pnl', 0) for t in daily_trades)
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def _calculate_weekly_pnl(self) -> float:
        """Berechnet wöchentlichen P&L"""
        return 0.0
    
    def _calculate_total_pnl(self) -> float:
        """Berechnet gesamten P&L"""
        return sum(t.get('pnl', 0) for t in self.trade_history)
    
    def _calculate_win_rate(self) -> float:
        """Berechnet Win Rate in Prozent"""
        if not self.trade_history:
            return 0.0
        winning = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        return (winning / len(self.trade_history)) * 100
    
    def _calculate_sharpe_ratio(self) -> float:
        """Berechnet Sharpe Ratio"""
        # Vereinfachte Implementierung
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Berechnet maximalen Drawdown"""
        return 0.0


# Globale Bot-Instanz für API
server_bot = ServerBotWrapper()