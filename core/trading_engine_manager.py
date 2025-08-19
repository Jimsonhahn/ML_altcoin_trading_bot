"""
Trading Engine Manager - Upgrade für echtes Exchange Paper Trading
================================================================

Erweitert das bestehende System um echte Exchange APIs während die 
bestehende Simulation als Fallback erhalten bleibt.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from core.paper_trading_engine import PaperTradingEngine
from utils.secret_manager import SecretManager
from core.trading_engine_improvements import (
    event_bus, resource_manager, security_auditor, performance_monitor,
    HealthChecker, APIResponse, retry_async_operation, get_request_id, log_with_context
)

logger = logging.getLogger(__name__)


class TradingEngineManager:
    """
    Manager für verschiedene Trading Engine Modi:
    - simulated: Bestehende virtuelle Engine (immer verfügbar)
    - real_paper: Echte Exchange Demo APIs
    - live: Echte Exchange Live APIs
    """
    
    def __init__(self, settings=None):
        self.settings = settings or {}
        
        # Bestehende simulierte Engine als Fallback (immer verfügbar)
        self.simulated_engine = PaperTradingEngine(
            initial_balance=self.settings.get('paper_trading.initial_balance', 10000.0)
        )
        
        # Neue echte Exchange Engines (nur wenn konfiguriert)
        self.real_paper_engine = None
        self.live_engine = None
        
        # Aktueller Modus (startet mit bestehendem System)
        self.current_mode = 'simulated'
        
        # Exchange Configuration
        self.exchange_config = None
        self.secret_manager = SecretManager('trading_engine')
        
        # Verfügbare Modi
        self.available_modes = {
            'simulated': True,  # Immer verfügbar
            'real_paper': False,  # Nur wenn Exchange APIs konfiguriert
            'live': False  # Nur wenn Live APIs konfiguriert
        }
        
        # Improvements Integration
        self.health_checker = HealthChecker(check_interval=60)
        self.performance_monitor = performance_monitor
        self.security_auditor = security_auditor
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        
        # Subscribe to events
        self.event_bus.subscribe('trade_executed', self._on_trade_executed)
        self.event_bus.subscribe('mode_switched', self._on_mode_switched)
        
        logger.info("🔧 TradingEngineManager initialized with simulated mode and improvements")
        
        # Versuche Exchange APIs zu laden falls bereits konfiguriert
        self._try_load_exchange_config()
    
    def _try_load_exchange_config(self):
        """Versucht bestehende Exchange Konfiguration zu laden"""
        try:
            # Lade Exchange Config aus SecretManager
            config = self.secret_manager.get_secret('exchange_config')
            if config:
                self.exchange_config = json.loads(config)
                self._initialize_real_engines()
                logger.info("✅ Existing exchange configuration loaded")
        except Exception as e:
            logger.info(f"ℹ️ No existing exchange config: {e}")
    
    def _initialize_real_engines(self):
        """Initialisiert echte Exchange APIs basierend auf Konfiguration"""
        if not self.exchange_config:
            return
        
        try:
            # Real Paper Trading Engine (Testnet/Demo)
            if self._has_testnet_credentials():
                from core.real_exchange_paper_engine import RealExchangePaperEngine
                self.real_paper_engine = RealExchangePaperEngine(
                    exchange=self.exchange_config.get('exchange', 'binance'),
                    api_key=self.exchange_config.get('testnet_api_key'),
                    api_secret=self.exchange_config.get('testnet_api_secret'),
                    testnet=True
                )
                self.available_modes['real_paper'] = True
                logger.info("✅ Real paper trading engine initialized (testnet)")
            
            # Live Trading Engine (nur wenn Live Keys vorhanden)
            if self._has_live_credentials():
                from core.live_exchange_engine import LiveExchangeEngine
                self.live_engine = LiveExchangeEngine(
                    exchange=self.exchange_config.get('exchange', 'binance'),
                    api_key=self.exchange_config.get('live_api_key'),
                    api_secret=self.exchange_config.get('live_api_secret'),
                    testnet=False
                )
                self.available_modes['live'] = True
                logger.info("✅ Live trading engine initialized")
                
        except ImportError as e:
            logger.warning(f"⚠️ Exchange engine classes not available: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange engines: {e}")
    
    def _has_testnet_credentials(self) -> bool:
        """Prüft ob Testnet Credentials vorhanden sind"""
        return (self.exchange_config and 
                self.exchange_config.get('testnet_api_key') and 
                self.exchange_config.get('testnet_api_secret'))
    
    def _has_live_credentials(self) -> bool:
        """Prüft ob Live Credentials vorhanden sind"""
        return (self.exchange_config and 
                self.exchange_config.get('live_api_key') and 
                self.exchange_config.get('live_api_secret'))
    
    def setup_exchange_credentials(self, config: Dict[str, str]) -> Dict[str, Any]:
        """
        Konfiguriert Exchange API Credentials
        
        Args:
            config: Dict mit exchange, api_key, api_secret, etc.
        
        Returns:
            Dict mit success status und message
        """
        try:
            # Validiere Konfiguration
            required_fields = ['exchange']
            if not all(field in config for field in required_fields):
                return {
                    'success': False,
                    'message': f'Missing required fields: {required_fields}'
                }
            
            # Speichere Konfiguration sicher
            self.exchange_config = config
            config_json = json.dumps(config)
            self.secret_manager.store_secret('exchange_config', config_json)
            
            # Initialisiere Exchange Engines
            self._initialize_real_engines()
            
            return {
                'success': True,
                'message': 'Exchange APIs configured successfully',
                'available_modes': self.available_modes
            }
            
        except Exception as e:
            logger.error(f"❌ Exchange setup failed: {e}")
            return {
                'success': False,
                'message': f'Exchange setup failed: {str(e)}'
            }
    
    def get_available_modes(self) -> Dict[str, Any]:
        """Liefert verfügbare Trading Modi mit Details"""
        return {
            'current_mode': self.current_mode,
            'available_modes': {
                'simulated': {
                    'available': True,
                    'description': 'Virtual trading with simulated data (your existing system)',
                    'risk': 'None',
                    'balance': self.simulated_engine.virtual_balance if self.simulated_engine else 0
                },
                'real_paper': {
                    'available': self.available_modes['real_paper'],
                    'description': 'Real exchange demo APIs with fake money',
                    'risk': 'None (demo account)',
                    'balance': self.real_paper_engine.get_balance() if self.real_paper_engine else 0
                },
                'live': {
                    'available': self.available_modes['live'],
                    'description': 'Real exchange APIs with real money',
                    'risk': 'High (real money)',
                    'balance': self.live_engine.get_balance() if self.live_engine else 0
                }
            }
        }
    
    @performance_monitor.performance_decorator('switch_mode')
    def switch_mode(self, new_mode: str, user: str = 'system') -> Dict[str, Any]:
        """
        Wechselt zwischen Trading Modi ohne System-Ausfall
        
        Args:
            new_mode: 'simulated', 'real_paper', oder 'live'
            
        Returns:
            Dict mit success status und message
        """
        valid_modes = ['simulated', 'real_paper', 'live']
        
        if new_mode not in valid_modes:
            return {
                'success': False,
                'message': f'Invalid mode: {new_mode}. Valid modes: {valid_modes}'
            }
        
        # Prüfe ob gewünschter Modus verfügbar ist
        if not self.available_modes.get(new_mode, False):
            if new_mode == 'real_paper':
                return {
                    'success': False,
                    'message': 'Real paper trading not configured. Add exchange testnet API keys first.'
                }
            elif new_mode == 'live':
                return {
                    'success': False,
                    'message': 'Live trading not configured. Add live API keys first.'
                }
            else:
                return {
                    'success': False,
                    'message': f'Mode {new_mode} not available'
                }
        
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Log security audit
        self.security_auditor.log_mode_switch(user, old_mode, new_mode, True)
        
        # Publish event
        asyncio.create_task(self.event_bus.publish('mode_switched', {
            'user': user,
            'old_mode': old_mode,
            'new_mode': new_mode,
            'timestamp': datetime.now().isoformat()
        }))
        
        logger.info(f"🔄 Trading mode switched: {old_mode} → {new_mode} by {user}")
        
        return {
            'success': True,
            'message': f'Successfully switched from {old_mode} to {new_mode}',
            'old_mode': old_mode,
            'new_mode': new_mode,
            'engine_status': self._get_engine_status(),
            'user': user
        }
    
    def _get_engine_status(self) -> Dict[str, str]:
        """Liefert Status aller Engines"""
        return {
            'simulated': 'active' if self.simulated_engine else 'inactive',
            'real_paper': 'active' if self.real_paper_engine else 'not_configured',
            'live': 'active' if self.live_engine else 'not_configured'
        }
    
    @retry_async_operation(max_retries=3, delay=1.0)
    async def execute_trade(self, symbol: str, side: str, size: float, 
                           strategy: str = "manual", user: str = 'system', **kwargs) -> Optional[Dict[str, Any]]:
        """
        Smart trade execution - nutzt beste verfügbare Engine
        
        Args:
            symbol: Trading pair (z.B. 'BTC/USDT')
            side: 'LONG' oder 'SHORT'
            size: Position size
            strategy: Strategy name
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Trade result dict oder None bei Fehler
        """
        try:
            if self.current_mode == 'simulated':
                # Nutze bestehendes simuliertes System
                logger.info(f"📝 Executing simulated trade: {side} {size} {symbol}")
                return await self.simulated_engine.execute_virtual_trade(
                    symbol=symbol,
                    side=side,
                    size=size,
                    strategy=strategy,
                    **kwargs
                )
                
            elif self.current_mode == 'real_paper' and self.real_paper_engine:
                # Nutze echte Exchange Demo APIs
                logger.info(f"🏦 Executing real paper trade: {side} {size} {symbol}")
                return await self.real_paper_engine.execute_demo_trade(
                    symbol=symbol,
                    side=side,
                    size=size,
                    strategy=strategy,
                    **kwargs
                )
                
            elif self.current_mode == 'live' and self.live_engine:
                # Nutze echte Exchange Live APIs
                logger.warning(f"💰 Executing LIVE trade: {side} {size} {symbol}")
                return await self.live_engine.execute_live_trade(
                    symbol=symbol,
                    side=side,
                    size=size,
                    strategy=strategy,
                    **kwargs
                )
            else:
                # Fallback zu Simulation
                logger.warning(f"⚠️ {self.current_mode} engine not available, falling back to simulation")
                result = await self.simulated_engine.execute_virtual_trade(
                    symbol=symbol,
                    side=side,
                    size=size,
                    strategy=strategy,
                    **kwargs
                )
            
            # Log successful trade
            if result:
                request_id = get_request_id()
                self.security_auditor.log_trade_execution(user, symbol, side, size, self.current_mode, True)
                
                # Publish trade event
                await self.event_bus.publish('trade_executed', {
                    'request_id': request_id,
                    'user': user,
                    'trade_id': result.id if hasattr(result, 'id') else 'unknown',
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'mode': self.current_mode
                })
                
                log_with_context(logger, 'info', f"Trade executed successfully", 
                                request_id=request_id, trade_id=result.id if hasattr(result, 'id') else 'unknown')
            
            return result
                
        except ValueError as e:
            # Validation errors
            request_id = get_request_id()
            log_with_context(logger, 'error', f"Trade validation failed: {e}", 
                            request_id=request_id, user=user)
            self.security_auditor.log_trade_execution(user, symbol, side, size, self.current_mode, False)
            raise
            
        except Exception as e:
            # Other errors
            request_id = get_request_id()
            log_with_context(logger, 'error', f"Trade execution failed: {e}", 
                            request_id=request_id, user=user, error=str(e))
            self.security_auditor.log_trade_execution(user, symbol, side, size, self.current_mode, False)
            
            # Publish failure event
            await self.event_bus.publish('trade_failed', {
                'request_id': request_id,
                'user': user,
                'symbol': symbol,
                'error': str(e),
                'mode': self.current_mode
            })
            
            return None
    
    async def close_trade(self, trade_id: str, exit_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Smart trade closing - nutzt aktuelle Engine
        """
        try:
            if self.current_mode == 'simulated':
                return await self.simulated_engine.close_virtual_trade(trade_id, exit_price)
            elif self.current_mode == 'real_paper' and self.real_paper_engine:
                return await self.real_paper_engine.close_demo_trade(trade_id, exit_price)
            elif self.current_mode == 'live' and self.live_engine:
                return await self.live_engine.close_live_trade(trade_id, exit_price)
            else:
                # Fallback
                return await self.simulated_engine.close_virtual_trade(trade_id, exit_price)
                
        except Exception as e:
            logger.error(f"❌ Trade closing failed: {e}")
            return None
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Liefert Portfolio Status der aktiven Engine
        """
        try:
            base_status = {
                'mode': self.current_mode,
                'engine_available': False,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.current_mode == 'simulated' and self.simulated_engine:
                status = self.simulated_engine.get_virtual_portfolio_status()
                status.update(base_status)
                status['engine_available'] = True
                return status
                
            elif self.current_mode == 'real_paper' and self.real_paper_engine:
                status = self.real_paper_engine.get_portfolio_status()
                status.update(base_status)
                status['engine_available'] = True
                return status
                
            elif self.current_mode == 'live' and self.live_engine:
                status = self.live_engine.get_portfolio_status()
                status.update(base_status)
                status['engine_available'] = True
                return status
            else:
                # Fallback zu Simulation
                if self.simulated_engine:
                    status = self.simulated_engine.get_virtual_portfolio_status()
                    status.update(base_status)
                    status['mode'] = f'{self.current_mode}_fallback_simulated'
                    status['engine_available'] = True
                    return status
                
            base_status['error'] = f'No engine available for mode: {self.current_mode}'
            return base_status
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio status: {e}")
            return {
                'mode': self.current_mode,
                'engine_available': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Liefert aktive Trades der aktuellen Engine"""
        try:
            if self.current_mode == 'simulated' and self.simulated_engine:
                # Konvertiere VirtualPosition zu Dict
                trades = []
                for pos in self.simulated_engine.virtual_positions.values():
                    trades.append({
                        'id': pos.id,
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'pnl': (pos.size * (pos.current_price - pos.entry_price) 
                                if pos.side == 'LONG' else 
                                pos.size * (pos.entry_price - pos.current_price)) - pos.fee,
                        'strategy': pos.strategy,
                        'timestamp': pos.timestamp.isoformat(),
                        'mode': 'simulated'
                    })
                return trades
                
            elif self.current_mode == 'real_paper' and self.real_paper_engine:
                return self.real_paper_engine.get_active_trades()
                
            elif self.current_mode == 'live' and self.live_engine:
                return self.live_engine.get_active_trades()
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting active trades: {e}")
            return []
    
    def reset_account(self, user: str = 'system') -> Dict[str, Any]:
        """Reset der aktuellen Engine (nur für Paper Trading Modi)"""
        try:
            # Security audit logging
            self.security_auditor.log_account_reset(user, self.current_mode)
            
            if self.current_mode == 'simulated' and self.simulated_engine:
                self.simulated_engine.reset_paper_account()
                result = {'success': True, 'message': 'Simulated account reset', 'user': user}
                
            elif self.current_mode == 'real_paper' and self.real_paper_engine:
                result = self.real_paper_engine.reset_demo_account()
                result['user'] = user
                
            else:
                result = {
                    'success': False,
                    'message': f'Account reset not available for mode: {self.current_mode}',
                    'user': user
                }
            
            # Publish reset event
            if result['success']:
                asyncio.create_task(self.event_bus.publish('account_reset', {
                    'user': user,
                    'mode': self.current_mode,
                    'timestamp': datetime.now().isoformat()
                }))
            
            return result
                
        except Exception as e:
            logger.error(f"❌ Account reset failed: {e}")
            return {'success': False, 'message': str(e)}
    
    async def _on_trade_executed(self, event: Dict[str, Any]):
        """Handle trade executed event"""
        try:
            logger.info(f"Trade executed event: {event['data']['trade_id']} in mode {event['data']['mode']}")
            
            # Emit dashboard update
            from api.websocket.events import emit_dashboard_update
            emit_dashboard_update({
                'type': 'trade_executed',
                'trade_id': event['data']['trade_id'],
                'symbol': event['data']['symbol'],
                'mode': event['data']['mode']
            })
        except Exception as e:
            logger.error(f"Error handling trade executed event: {e}")
    
    async def _on_mode_switched(self, event: Dict[str, Any]):
        """Handle mode switched event"""
        try:
            logger.info(f"Mode switched event: {event['data']['old_mode']} -> {event['data']['new_mode']}")
            
            # Emit dashboard update
            from api.websocket.events import emit_dashboard_update
            emit_dashboard_update({
                'type': 'mode_switched',
                'old_mode': event['data']['old_mode'],
                'new_mode': event['data']['new_mode']
            })
        except Exception as e:
            logger.error(f"Error handling mode switched event: {e}")