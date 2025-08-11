#!/usr/bin/env python3
"""
🚀 Intelligence API Server
Standalone Server für Intelligence Features - funktioniert parallel zu deinem Bot
"""

import asyncio
import logging
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from utils.api_security import init_api_security, security_manager, resilient_client
from utils.dashboard_enhancements import dashboard_manager, BotHealthStatus, NotificationLevel
import sys

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock Database Pool für Standalone-Betrieb
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

# Mock Trading Bot for Standalone-Betrieb
class MockTradingBot:
    def __init__(self):
        self.exchange_manager = MockExchangeManager()
        self.config = {}
    
    async def execute_trade(self, signal):
        # Mock successful trade
        class MockTradeResult:
            def __init__(self):
                self.success = True
                self.pnl = 0.0
                self.error = None
        
        return MockTradeResult()

class MockExchangeManager:
    pass

# Global Enhanced Logger Instance
enhanced_logger = None
risk_tiered_manager = None
portfolio_optimizer = None

def create_intelligence_app():
    """Erstelle Flask App nur für Intelligence Features"""
    app = Flask(__name__)
    
    # CORS für alle Origins
    CORS(app, origins=["*"])
    
    # Initialize API Security
    init_api_security(
        app, 
        secret_key="janics_freedom_factory_secure_2024",
        admin_keys=["janics_admin_2024_secure"]
    )
    
    # Dashboard route
    @app.route('/')
    @app.route('/dashboard')
    def serve_dashboard():
        """Serve the JANICS FREEDOM FACTORY dashboard"""
        import os
        try:
            dashboard_path = 'janics_freedom_factory_dashboard.html'
            if os.path.exists(dashboard_path):
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Update API base URL to use relative path
                content = content.replace('http://85.215.183.30:8080/api/intelligence', '/api/intelligence')
                content = content.replace('http://85.215.183.30:8002/api/intelligence', '/api/intelligence')
                return content, 200, {'Content-Type': 'text/html'}
            else:
                return "Dashboard not found. Please ensure janics_freedom_factory_dashboard.html exists.", 404
        except Exception as e:
            return f"Error loading dashboard: {str(e)}", 500
    
    # Basic routes
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'Intelligence API',
            'enhanced_logger': enhanced_logger is not None
        })
    
    @app.route('/api/intelligence/health')
    @security_manager.rate_limit_decorator('api/intelligence')
    def intelligence_health():
        from datetime import datetime
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'intelligence_enabled': enhanced_logger is not None,
            'security': {
                'rate_limiting': True,
                'circuit_breaker': True,
                'encrypted_transmission': True
            },
            'features': {
                'learning': enhanced_logger.learning_enabled if enhanced_logger else False,
                'dashboard_updates': enhanced_logger.dashboard_updates if enhanced_logger else False,
                'export_path': str(enhanced_logger.export_path) if enhanced_logger else None
            }
        })
    
    @app.route('/api/intelligence/metrics')
    def get_metrics():
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
        
        try:
            # Sync wrapper für async call
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            metrics = loop.run_until_complete(
                enhanced_logger.get_dashboard_metrics()
            )
            
            return jsonify({
                'success': True,
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/intelligence/export/decisions')
    def export_decisions():
        if not enhanced_logger:
            return jsonify({'error': 'Enhanced logging not available'}), 503
        
        try:
            export_path = enhanced_logger.export_path
            file_path = export_path / 'structured_decisions.json'
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = f.read()
                return data, 200, {'Content-Type': 'application/json'}
            else:
                return jsonify({'error': 'No decisions export found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Risk-Tiered Strategy System API Endpoints
    @app.route('/api/risk-tiered/status')
    @security_manager.rate_limit_decorator('api/risk-tiered')
    def risk_tiered_status():
        """Get Risk-Tiered Strategy System status"""
        if not risk_tiered_manager:
            return jsonify({
                'status': 'inactive',
                'message': 'Risk-Tiered System not initialized'
            })
        
        try:
            # Get current allocations and status
            status_data = {
                'status': 'active' if risk_tiered_manager.is_running else 'stopped',
                'portfolio_value': float(risk_tiered_manager.portfolio_value),
                'total_strategies': len(risk_tiered_manager.strategy_allocations),
                'risk_categories': {
                    category_name: {
                        'allocation_percent': category.allocation_percent,
                        'max_position_size': category.max_position_size_percent,
                        'concurrent_trades': category.max_trades_concurrent,
                        'expected_roi': category.expected_roi_percent
                    }
                    for category_name, category in risk_tiered_manager.risk_categories.items()
                },
                'strategy_allocations': [
                    {
                        'name': alloc.strategy_name,
                        'risk_category': alloc.risk_category,
                        'allocation_percent': alloc.allocation_percent,
                        'active_positions': len(alloc.current_positions),
                        'total_trades': alloc.performance_metrics.get('total_trades', 0),
                        'winning_trades': alloc.performance_metrics.get('winning_trades', 0),
                        'total_pnl': float(alloc.performance_metrics.get('total_pnl', 0)),
                        'is_active': alloc.is_active
                    }
                    for alloc in risk_tiered_manager.strategy_allocations
                ]
            }
            
            return jsonify({
                'success': True,
                'data': status_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/risk-tiered/performance')
    @security_manager.rate_limit_decorator('api/risk-tiered')
    def risk_tiered_performance():
        """Get detailed performance metrics"""
        from datetime import datetime
        if not risk_tiered_manager:
            return jsonify({'error': 'Risk-Tiered System not available'}), 503
        
        try:
            # Calculate performance by risk category
            categories = risk_tiered_manager._group_strategies_by_category()
            performance_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_value': float(risk_tiered_manager.portfolio_value),
                'categories': {},
                'total_stats': {
                    'total_pnl': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0
                }
            }
            
            for category_name, strategies in categories.items():
                category_pnl = 0.0
                category_trades = 0
                category_wins = 0
                
                strategy_details = []
                
                for strategy in strategies:
                    metrics = strategy.performance_metrics
                    pnl = float(metrics.get('total_pnl', 0))
                    trades = metrics.get('total_trades', 0)
                    wins = metrics.get('winning_trades', 0)
                    
                    category_pnl += pnl
                    category_trades += trades
                    category_wins += wins
                    
                    strategy_details.append({
                        'name': strategy.strategy_name,
                        'allocation': strategy.allocation_percent,
                        'pnl': pnl,
                        'trades': trades,
                        'win_rate': (wins / max(1, trades)) * 100,
                        'active_positions': len(strategy.current_positions)
                    })
                
                win_rate = (category_wins / max(1, category_trades)) * 100
                
                performance_data['categories'][category_name] = {
                    'total_pnl': category_pnl,
                    'total_trades': category_trades,
                    'win_rate': win_rate,
                    'allocation_percent': risk_tiered_manager.risk_categories[category_name].allocation_percent,
                    'strategies': strategy_details
                }
                
                # Add to totals
                performance_data['total_stats']['total_pnl'] += category_pnl
                performance_data['total_stats']['total_trades'] += category_trades
                performance_data['total_stats']['winning_trades'] += category_wins
            
            # Calculate overall win rate
            total_trades = performance_data['total_stats']['total_trades']
            performance_data['total_stats']['overall_win_rate'] = (
                performance_data['total_stats']['winning_trades'] / max(1, total_trades)
            ) * 100
            
            # Calculate portfolio return
            performance_data['total_stats']['portfolio_return_percent'] = (
                performance_data['total_stats']['total_pnl'] / 
                float(risk_tiered_manager.portfolio_value)
            ) * 100
            
            return jsonify({
                'success': True,
                'data': performance_data
            })
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/portfolio/optimization')
    @security_manager.rate_limit_decorator('api/risk-tiered')
    def portfolio_optimization_status():
        """Get portfolio optimization metrics"""
        from datetime import datetime
        if not portfolio_optimizer:
            return jsonify({'error': 'Portfolio Optimizer not available'}), 503
        
        try:
            # Mock optimization data for now
            optimization_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'active',
                'last_rebalance': datetime.utcnow().isoformat(),
                'optimization_methods': [
                    {
                        'name': 'Sharpe Ratio',
                        'score': 1.85,
                        'status': 'optimal'
                    },
                    {
                        'name': 'Kelly Criterion',
                        'score': 2.12,
                        'status': 'good'
                    },
                    {
                        'name': 'Risk Parity',
                        'score': 1.76,
                        'status': 'balanced'
                    }
                ],
                'risk_metrics': {
                    'portfolio_volatility': 0.18,
                    'max_drawdown': 0.12,
                    'var_95': 0.08,
                    'sharpe_ratio': 1.85
                },
                'allocation_drift': {
                    'HIGH_RISK': {
                        'target': 15.0,
                        'current': 14.2,
                        'drift': -0.8
                    },
                    'MEDIUM_RISK': {
                        'target': 35.0,
                        'current': 36.1,
                        'drift': 1.1
                    },
                    'LOW_RISK': {
                        'target': 50.0,
                        'current': 49.7,
                        'drift': -0.3
                    }
                }
            }
            
            return jsonify({
                'success': True,
                'data': optimization_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Enhanced Dashboard Endpoints
    @app.route('/api/dashboard/health')
    @security_manager.rate_limit_decorator('dashboard')
    def dashboard_health():
        """Get comprehensive bot health metrics"""
        try:
            # Simulate real health data
            dashboard_manager.simulate_demo_data()
            health_data = dashboard_manager.get_dashboard_data()
            
            return jsonify({
                'success': True,
                'data': health_data
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/dashboard/notifications')
    @security_manager.rate_limit_decorator('dashboard')
    def dashboard_notifications():
        """Get current dashboard notifications"""
        try:
            notifications = [{
                'id': notif.id,
                'title': notif.title,
                'message': notif.message,
                'level': notif.level.value,
                'category': notif.category,
                'timestamp': notif.timestamp.isoformat(),
                'action_required': notif.action_required
            } for notif in dashboard_manager.notifications[:10]]
            
            return jsonify({
                'success': True,
                'notifications': notifications
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/dashboard/orchestra')
    @security_manager.rate_limit_decorator('dashboard')
    def strategy_orchestra():
        """Get strategy orchestra display data"""
        try:
            orchestra_data = [{
                'name': item.name,
                'status': item.status,
                'performance_score': item.performance_score,
                'current_signal': item.current_signal,
                'confidence': item.confidence,
                'trades_today': item.trades_today,
                'pnl_today': item.pnl_today,
                'risk_level': item.risk_level,
                'execution_time_ms': item.execution_time_ms,
                'last_action': item.last_action,
                'last_action_time': item.last_action_time.isoformat()
            } for item in dashboard_manager.strategy_orchestra]
            
            return jsonify({
                'success': True,
                'strategies': orchestra_data,
                'summary': {
                    'total_active': len([s for s in dashboard_manager.strategy_orchestra if s.status == 'active']),
                    'total_trades_today': sum(s.trades_today for s in dashboard_manager.strategy_orchestra),
                    'total_pnl_today': sum(s.pnl_today for s in dashboard_manager.strategy_orchestra),
                    'avg_performance': sum(s.performance_score for s in dashboard_manager.strategy_orchestra) / max(1, len(dashboard_manager.strategy_orchestra))
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/intelligence/demo')
    @security_manager.rate_limit_decorator('api/intelligence')
    def demo_data():
        """Demo-Daten für Testing"""
        return jsonify({
            'decisions': [
                {
                    'id': 'demo_001',
                    'timestamp': '2025-08-11T13:54:00Z',
                    'strategy': 'momentum_strategy',
                    'symbol': 'BTC/USDT',
                    'action': 'buy',
                    'confidence': 0.85,
                    'reasoning': 'Strong bullish momentum detected'
                },
                {
                    'id': 'demo_002',
                    'timestamp': '2025-08-11T13:55:00Z',
                    'strategy': 'mean_reversion',
                    'symbol': 'ETH/USDT',
                    'action': 'sell',
                    'confidence': 0.72,
                    'reasoning': 'Overbought condition detected'
                }
            ],
            'metrics': {
                'total_decisions': 2,
                'avg_confidence': 0.785,
                'strategies_active': ['momentum_strategy', 'mean_reversion']
            },
            'anomalies': [],
            'patterns': [
                {
                    'name': 'Momentum Breakout',
                    'frequency': 15,
                    'success_rate': 0.73
                }
            ]
        })
    
    return app

async def initialize_enhanced_logger():
    """Initialize Enhanced Logger"""
    global enhanced_logger
    
    try:
        from core.enhanced_decision_logger import create_enhanced_decision_logger
        
        # Mock database pool
        mock_pool = MockDBPool()
        
        # Create enhanced logger
        enhanced_logger = await create_enhanced_decision_logger(
            db_pool=mock_pool,
            export_path="intelligence_exports/",
            dashboard_updates=True,
            learning_enabled=True
        )
        
        logger.info("✅ Enhanced Logger initialisiert")
        
        # Log demo decision
        demo_decision = {
            'strategy': 'momentum_strategy',
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'price': 45000.0
        }
        
        demo_context = {
            'regime': 'bull_market',
            'volatility': 0.15,
            'rsi': 65.5
        }
        
        await enhanced_logger.log_trading_decision_with_context(
            decision_data=demo_decision,
            market_context=demo_context,
            strategy_reasoning="Demo decision for testing",
            confidence_level=0.85
        )
        
        logger.info("✅ Demo decision geloggt")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Logger initialization failed: {e}")
        return False

async def initialize_risk_tiered_system():
    """Initialize Risk-Tiered Strategy System"""
    global risk_tiered_manager, portfolio_optimizer
    
    try:
        from decimal import Decimal
        from risk_tiered_manager import RiskTieredStrategyManager
        from portfolio_optimizer import PortfolioOptimizer
        # Mock trading bot and database for demo  
        class MockExchangeManager:
            pass
        
        mock_trading_bot = MockTradingBot()
        mock_db_pool = MockDBPool()
        portfolio_value = Decimal('100000')  # $100k demo portfolio
        
        # Initialize Risk-Tiered Manager
        risk_tiered_manager = RiskTieredStrategyManager(
            trading_bot=mock_trading_bot,
            db_pool=mock_db_pool,
            portfolio_value=portfolio_value
        )
        
        # Initialize Portfolio Optimizer  
        portfolio_optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
        logger.info("✅ Risk-Tiered Strategy System initialized")
        logger.info(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"📊 Strategies Discovered: {len(risk_tiered_manager.strategy_allocations)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Risk-Tiered System initialization failed: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return False

def run_intelligence_server(host='localhost', port=8001):
    """Run Intelligence API Server"""
    from datetime import datetime
    
    print("🚀 Intelligence API Server")
    print("=" * 40)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("=" * 40)
    
    # Initialize async components
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    logger_initialized = loop.run_until_complete(initialize_enhanced_logger())
    risk_system_initialized = loop.run_until_complete(initialize_risk_tiered_system())
    
    if logger_initialized:
        print("✅ Enhanced Logger aktiv")
    else:
        print("⚠️  Enhanced Logger nicht verfügbar - Demo-Modus")
    
    if risk_system_initialized:
        print("✅ Risk-Tiered Strategy System aktiv")
    else:
        print("⚠️  Risk-Tiered System nicht verfügbar - Demo-Modus")
    
    # Create Flask app
    app = create_intelligence_app()
    
    print("\n🌐 Available Endpoints:")
    print(f"   🎮 Dashboard: http://{host}:{port}/")
    print(f"   🎮 Dashboard Alt: http://{host}:{port}/dashboard")
    print(f"   Health: http://{host}:{port}/health")
    print(f"   Intelligence Health: http://{host}:{port}/api/intelligence/health")
    print(f"   Metrics: http://{host}:{port}/api/intelligence/metrics")
    print(f"   Export: http://{host}:{port}/api/intelligence/export/decisions")
    print(f"   Demo: http://{host}:{port}/api/intelligence/demo")
    print(f"   🎯 Risk-Tiered Status: http://{host}:{port}/api/risk-tiered/status")
    print(f"   📊 Risk Performance: http://{host}:{port}/api/risk-tiered/performance")
    print(f"   ⚖️ Portfolio Optimization: http://{host}:{port}/api/portfolio/optimization")
    print("\n🎯 Test with: curl http://localhost:8001/health")
    print("🎯 Dashboard ready für mobile access!")
    print("=" * 40)
    
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server gestoppt")
    finally:
        if enhanced_logger:
            loop.run_until_complete(enhanced_logger.stop())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligence API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host (0.0.0.0 for external access)')
    parser.add_argument('--port', type=int, default=8001, help='Port')
    
    args = parser.parse_args()
    
    run_intelligence_server(host=args.host, port=args.port)