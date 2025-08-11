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

# Global Enhanced Logger Instance
enhanced_logger = None

def create_intelligence_app():
    """Erstelle Flask App nur für Intelligence Features"""
    app = Flask(__name__)
    
    # CORS für alle Origins
    CORS(app, origins=["*"])
    
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
    def intelligence_health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'intelligence_enabled': enhanced_logger is not None,
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
    
    @app.route('/api/intelligence/demo')
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
    
    if logger_initialized:
        print("✅ Enhanced Logger aktiv")
    else:
        print("⚠️  Enhanced Logger nicht verfügbar - Demo-Modus")
    
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