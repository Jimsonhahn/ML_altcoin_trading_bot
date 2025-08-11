#!/usr/bin/env python3
"""
Orchestrator Background Worker
==============================

Runs the orchestrator in the background and sends updates to the dashboard via WebSocket.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import orchestrator components
from core.strategy_orchestrator import StrategyDiscoveryEngine
from core.intelligent_orchestration_engine import IntelligentOrchestrationEngine
from core.strategy_health_monitor import HealthAndABTestingSystem
from core.orchestrator_portfolio_manager import create_portfolio_manager

# Import API app for WebSocket
from api.app import create_app
from api.websocket.socket_handlers import broadcast_to_channel

class OrchestratorWorker:
    """Background worker for orchestrator with WebSocket updates"""
    
    def __init__(self, trading_mode: str = "paper", update_interval: int = 30):
        self.trading_mode = trading_mode
        self.update_interval = update_interval
        self.running = False
        
        # Initialize orchestrator components
        self.orchestrator = None
        self.orchestration_engine = None
        self.health_system = None
        self.portfolio_manager = None
        
        # Flask app for WebSocket
        self.app = None
        self.socketio = None
        
        logger.info(f"🔧 Orchestrator Worker initialized in {trading_mode} mode")
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing orchestrator components...")
        
        # Initialize orchestrator
        self.orchestrator = StrategyDiscoveryEngine()
        self.orchestration_engine = IntelligentOrchestrationEngine()
        self.health_system = HealthAndABTestingSystem()
        
        # Initialize portfolio manager
        self.portfolio_manager = await create_portfolio_manager(self.trading_mode, 10000.0)
        
        # Create Flask app for WebSocket
        self.app, self.socketio = create_app()
        
        # Discover strategies
        self.discovered_strategies = await self.orchestrator.discover_all_strategies()
        logger.info(f"✅ Discovered {len(self.discovered_strategies)} strategies")
    
    async def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data (simulated for demo)"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        market_data = {}
        
        for symbol in symbols:
            # Generate realistic market data
            hours = 168  # 1 week of hourly data
            timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
            
            # Base price with trend
            base_price = {
                'BTC/USDT': 45000,
                'ETH/USDT': 2500,
                'SOL/USDT': 100,
                'BNB/USDT': 350
            }.get(symbol, 100)
            
            # Generate OHLCV data
            np.random.seed(hash(symbol) % 1000)
            trend = np.cumsum(np.random.randn(hours) * base_price * 0.002)
            prices = base_price + trend
            
            data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices + np.random.randn(hours) * base_price * 0.001,
                'high': prices + np.abs(np.random.randn(hours)) * base_price * 0.002,
                'low': prices - np.abs(np.random.randn(hours)) * base_price * 0.002,
                'close': prices,
                'volume': np.random.lognormal(20, 1, hours) * 1000
            })
            
            market_data[symbol] = data
        
        return market_data
    
    async def run_orchestration_cycle(self):
        """Run one orchestration cycle"""
        try:
            # Fetch market data
            market_data = await self.fetch_market_data()
            
            # Run orchestration
            decision = await self.orchestration_engine.orchestrate_strategies(
                market_data=market_data,
                risk_budget=self.portfolio_manager.cash_balance * 0.95
            )
            
            # Allocate capital
            allocations = await self.portfolio_manager.allocate_capital_to_strategies(
                decision.strategy_weights,
                self.portfolio_manager.cash_balance * 0.95
            )
            
            # Get portfolio state
            portfolio_state = await self.portfolio_manager.get_portfolio_state()
            
            # Monitor health
            health_updates = []
            for strategy, weight in decision.strategy_weights.items():
                if weight > 0:
                    # Mock performance data for demo
                    mock_performance = {
                        'trades': [
                            {'pnl': np.random.normal(0.002, 0.01), 
                             'entry_time': datetime.now(),
                             'exit_time': datetime.now()}
                        ],
                        'returns': list(np.random.normal(0.001, 0.02, 10)),
                        'error_rate': np.random.uniform(0, 0.05),
                        'avg_latency': np.random.uniform(0.05, 0.2),
                        'active_positions': np.random.randint(0, 5)
                    }
                    
                    health_result = await self.health_system.monitor_and_optimize(
                        strategy, mock_performance, market_data['BTC/USDT']
                    )
                    
                    health_updates.append({
                        'strategy': strategy,
                        'health_score': health_result['health_metrics'].overall_health_score,
                        'alerts': len(health_result['active_alerts']),
                        'emergency_stop': health_result['emergency_stop']
                    })
            
            # Prepare update data
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'market_regime': decision.market_regime,
                'market_volatility': decision.market_volatility,
                'portfolio': {
                    'mode': self.trading_mode,
                    'total_value': portfolio_state.total_value,
                    'cash_balance': portfolio_state.cash_balance,
                    'positions_value': portfolio_state.positions_value,
                    'total_pnl': portfolio_state.total_pnl,
                    'pnl_percent': (portfolio_state.total_pnl / self.portfolio_manager.initial_capital) * 100,
                    'win_rate': portfolio_state.win_rate,
                    'sharpe_ratio': portfolio_state.sharpe_ratio,
                    'max_drawdown': portfolio_state.max_drawdown,
                    'positions': portfolio_state.total_positions
                },
                'allocations': {
                    strategy: {
                        'weight': weight,
                        'capital': allocations.get(strategy, 0)
                    }
                    for strategy, weight in decision.strategy_weights.items()
                },
                'health_updates': health_updates,
                'risk_warnings': decision.risk_warnings
            }
            
            # Emit WebSocket updates
            self.emit_updates(update_data)
            
            return update_data
            
        except Exception as e:
            logger.error(f"Error in orchestration cycle: {e}")
            return None
    
    def emit_updates(self, data: Dict[str, Any]):
        """Emit updates via WebSocket"""
        try:
            # Emit to orchestrator channel
            broadcast_to_channel('orchestrator_updates', 'orchestrator_update', {
                'market_regime': data['market_regime'],
                'market_volatility': data['market_volatility'],
                'allocations': data['allocations'],
                'risk_warnings': data['risk_warnings']
            })
            
            # Emit to portfolio channel
            broadcast_to_channel('portfolio_updates', 'portfolio_update', {
                'portfolio': data['portfolio'],
                'timestamp': data['timestamp']
            })
            
            # Emit health alerts if any
            for health_update in data['health_updates']:
                if health_update['alerts'] > 0 or health_update['emergency_stop']:
                    broadcast_to_channel('health_alerts', 'health_alert', health_update)
            
        except Exception as e:
            logger.error(f"Error emitting updates: {e}")
    
    async def run(self):
        """Main worker loop"""
        await self.initialize()
        
        self.running = True
        cycle = 0
        
        logger.info("🎯 Starting orchestration worker loop...")
        
        while self.running:
            cycle += 1
            logger.info(f"🔄 Orchestration cycle #{cycle}")
            
            # Run orchestration
            result = await self.run_orchestration_cycle()
            
            if result:
                logger.info(f"✅ Cycle #{cycle} completed successfully")
            else:
                logger.warning(f"⚠️ Cycle #{cycle} had issues")
            
            # Wait for next cycle
            await asyncio.sleep(self.update_interval)
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        logger.info("🛑 Orchestrator worker stopping...")

def run_orchestrator_worker(mode: str = "paper"):
    """Run the orchestrator worker"""
    worker = OrchestratorWorker(trading_mode=mode, update_interval=30)
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.stop()
        logger.info("✅ Orchestrator worker stopped")

if __name__ == "__main__":
    # Get mode from command line or default to paper
    mode = sys.argv[1] if len(sys.argv) > 1 else "paper"
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║       ORCHESTRATOR BACKGROUND WORKER                     ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Mode: {mode.upper():^18}                       ║
    ║  Update Interval: 30 seconds                             ║
    ║                                                          ║
    ║  This worker runs the orchestrator and sends             ║
    ║  real-time updates to the dashboard via WebSocket        ║
    ║                                                          ║
    ║  Press Ctrl+C to stop                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    run_orchestrator_worker(mode)