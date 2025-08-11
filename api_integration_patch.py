#!/usr/bin/env python3
"""
🚀 API Integration Patch für Intelligence Features
Füge diese Änderungen zu deiner bestehenden api/app.py hinzu
"""

# ===== SCHRITT 1: Neue Imports hinzufügen =====
# Nach den bestehenden imports in api/app.py:

from api.routes.intelligence import intelligence_bp, set_enhanced_logger

# ===== SCHRITT 2: Blueprint registrieren =====
# In der create_app() funktion, nach den anderen blueprint registrierungen:

def create_enhanced_app(config_name='development', enhanced_logger=None):
    """Create Flask app with enhanced intelligence features"""
    app = create_app(config_name)  # Deine bestehende create_app Funktion
    
    # Register Intelligence Blueprint
    app.register_blueprint(intelligence_bp)
    
    # Set enhanced logger for intelligence routes
    if enhanced_logger:
        set_enhanced_logger(enhanced_logger)
        logger.info("✅ Enhanced logger integrated with API")
    
    return app

# ===== SCHRITT 3: WebSocket Events erweitern =====
# Neue Datei: api/websocket/intelligence_events.py

#!/usr/bin/env python3
"""
WebSocket Events für Intelligence Features
"""

from flask_socketio import emit, join_room, leave_room
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

logger = logging.getLogger(__name__)

def register_intelligence_events(socketio, enhanced_logger=None):
    """Register intelligence-specific WebSocket events"""
    
    @socketio.on('join_intelligence', namespace='/intelligence')
    def handle_join_intelligence(data):
        """Client joins intelligence room für real-time updates"""
        room = 'intelligence_updates'
        join_room(room)
        
        if enhanced_logger:
            enhanced_logger.add_websocket_client(request.sid)
        
        emit('intelligence_connected', {
            'status': 'connected',
            'features': {
                'real_time_decisions': True,
                'anomaly_alerts': True,
                'pattern_updates': True
            }
        })
        
        logger.info(f"Client {request.sid} joined intelligence room")
    
    @socketio.on('leave_intelligence', namespace='/intelligence')
    def handle_leave_intelligence(data):
        """Client leaves intelligence room"""
        room = 'intelligence_updates'
        leave_room(room)
        
        if enhanced_logger:
            enhanced_logger.remove_websocket_client(request.sid)
        
        logger.info(f"Client {request.sid} left intelligence room")
    
    @socketio.on('get_live_metrics', namespace='/intelligence')
    def handle_get_live_metrics(data):
        """Get real-time metrics"""
        if enhanced_logger:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                metrics = loop.run_until_complete(
                    enhanced_logger.get_dashboard_metrics()
                )
                
                emit('live_metrics', {
                    'success': True,
                    'metrics': metrics
                })
                
            except Exception as e:
                emit('live_metrics', {
                    'success': False,
                    'error': str(e)
                })
        else:
            emit('live_metrics', {
                'success': False,
                'error': 'Enhanced logging not available'
            })
    
    @socketio.on('request_anomaly_check', namespace='/intelligence')
    def handle_anomaly_check(data):
        """Manual anomaly check request"""
        if enhanced_logger:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                anomalies = loop.run_until_complete(
                    enhanced_logger.detect_trading_anomalies()
                )
                
                emit('anomaly_results', {
                    'success': True,
                    'anomaly_count': len(anomalies),
                    'anomalies': anomalies
                })
                
            except Exception as e:
                emit('anomaly_results', {
                    'success': False,
                    'error': str(e)
                })


# ===== SCHRITT 4: Enhanced App Startup =====
# Neue Datei: run_enhanced_api.py

#!/usr/bin/env python3
"""
Enhanced API Server Startup
Startet die API mit Intelligence Features
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from api.app import create_app
from api_integration_patch import create_enhanced_app
from api.websocket.intelligence_events import register_intelligence_events
from core.enhanced_decision_logger import create_enhanced_decision_logger
from utils.database import create_db_pool

logger = logging.getLogger(__name__)

async def create_enhanced_api_server(config_name='production'):
    """Create enhanced API server with intelligence features"""
    try:
        # Create database pool
        db_pool = await create_db_pool()
        
        # Create enhanced decision logger
        enhanced_logger = await create_enhanced_decision_logger(
            db_pool=db_pool,
            export_path="intelligence_exports/",
            dashboard_updates=True,
            learning_enabled=True
        )
        
        # Create Flask app with enhancements
        app = create_enhanced_app(config_name, enhanced_logger)
        
        # Get SocketIO instance from app
        from api.app import socketio
        
        # Register intelligence WebSocket events
        register_intelligence_events(socketio, enhanced_logger)
        
        logger.info("✅ Enhanced API server created with intelligence features")
        
        return app, socketio, enhanced_logger
        
    except Exception as e:
        logger.error(f"Failed to create enhanced API server: {e}")
        raise

def run_enhanced_api(host='0.0.0.0', port=8000, debug=False):
    """Run the enhanced API server"""
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create enhanced server
        app, socketio, enhanced_logger = loop.run_until_complete(
            create_enhanced_api_server('production' if not debug else 'development')
        )
        
        # Start the server
        logger.info(f"🚀 Starting Enhanced API server on {host}:{port}")
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,  # Important: avoid reloader with async
            log_output=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Enhanced API server stopped by user")
    except Exception as e:
        logger.error(f"Enhanced API server error: {e}")
        raise
    finally:
        # Cleanup
        if 'enhanced_logger' in locals():
            try:
                loop.run_until_complete(enhanced_logger.stop())
            except:
                pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Trading Bot API')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_enhanced_api(host=args.host, port=args.port, debug=args.debug)


# ===== SCHRITT 5: Integration Test =====
# test_intelligence_api.py

#!/usr/bin/env python3
"""Test Intelligence API Integration"""

import requests
import json
import asyncio
from datetime import datetime

API_BASE = "http://localhost:8000/api/v1/intelligence"

def test_intelligence_endpoints():
    """Test all intelligence API endpoints"""
    
    print("🧪 Testing Intelligence API Integration...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Health Check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test metrics endpoint
    try:
        response = requests.get(f"{API_BASE}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Metrics: {len(metrics.get('metrics', {}))} categories")
        else:
            print(f"⚠️ Metrics: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
    
    # Test recent decisions
    try:
        response = requests.get(f"{API_BASE}/decisions/recent?limit=5")
        if response.status_code == 200:
            decisions = response.json()
            print(f"✅ Recent Decisions: {decisions.get('count', 0)} found")
        else:
            print(f"⚠️ Decisions: {response.status_code}")
    except Exception as e:
        print(f"❌ Decisions failed: {e}")
    
    # Test anomalies
    try:
        response = requests.get(f"{API_BASE}/anomalies")
        if response.status_code == 200:
            anomalies = response.json()
            print(f"✅ Anomalies: {anomalies.get('anomaly_count', 0)} detected")
        else:
            print(f"⚠️ Anomalies: {response.status_code}")
    except Exception as e:
        print(f"❌ Anomalies failed: {e}")
    
    print("🎉 Intelligence API test completed!")

if __name__ == "__main__":
    test_intelligence_endpoints()


"""
🎯 INTEGRATION ZUSAMMENFASSUNG:

NEUE DATEIEN ERSTELLT:
✅ api/routes/intelligence.py - Neue API Endpunkte
✅ api/websocket/intelligence_events.py - WebSocket Events
✅ run_enhanced_api.py - Enhanced API Startup
✅ test_intelligence_api.py - Integration Tests

ÄNDERUNGEN AN BESTEHENDEN DATEIEN:
1. api/app.py: Blueprint registrieren
2. main.py: Enhanced logger an API übergeben

NEUE API ENDPUNKTE:
- GET /api/v1/intelligence/health
- GET /api/v1/intelligence/metrics
- GET /api/v1/intelligence/decisions/recent
- GET /api/v1/intelligence/anomalies
- GET/POST /api/v1/intelligence/learning-report
- GET /api/v1/intelligence/export/<type>
- GET /api/v1/intelligence/patterns
- GET /api/v1/intelligence/insights/latest
- POST /api/v1/intelligence/log-decision
- GET/POST /api/v1/intelligence/config

WEBSOCKET EVENTS:
- join_intelligence
- live_metrics
- anomaly_alerts
- pattern_updates

STARTE ENHANCED API:
python run_enhanced_api.py --host 0.0.0.0 --port 8000

TESTE INTEGRATION:
python test_intelligence_api.py
"""