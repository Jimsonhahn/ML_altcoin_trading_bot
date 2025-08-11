"""
Flask REST API for Altcoin Trading Bot
======================================

Main Flask application with CORS, JWT authentication, and WebSocket support.
"""

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from flask_swagger_ui import get_swaggerui_blueprint
from datetime import timedelta
import os
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.secret_manager import SecretManager
from utils.error_handler import SecureErrorHandler
from api.middleware.auth import setup_jwt_callbacks
from api.middleware.error_handler import register_error_handlers
from api.routes import trading, monitoring, strategies
from api.websocket import socket_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Load configuration
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-prod')
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-change-in-prod')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    app.config['JWT_ALGORITHM'] = 'HS256'
    
    # CORS configuration - Erweitert um Port 3002
    app.config['CORS_ORIGINS'] = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://localhost:3002').split(',')
    
    # WebSocket configuration
    app.config['SOCKETIO_ASYNC_MODE'] = 'threading'
    
    # Initialize extensions with comprehensive CORS configuration
    CORS(app, 
         origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", 
                 "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"],
         supports_credentials=True,
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         expose_headers=["Content-Type", "Authorization"]
    )
    jwt = JWTManager(app)
    socketio = SocketIO(
        app, 
        cors_allowed_origins=[
            "http://localhost:3000", "http://localhost:3001", "http://localhost:3002",
            "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"
        ],
        logger=True,
        engineio_logger=True,
        async_mode='threading'
    )
    
    # Initialize SecretManager for JWT keys
    secret_manager = SecretManager('trading_bot_api')
    
    # Get or create JWT secret from SecretManager
    jwt_secret = secret_manager.get_secret('jwt_secret_key')
    if not jwt_secret:
        # Generate a secure secret key
        import secrets
        jwt_secret = secrets.token_urlsafe(32)
        secret_manager.store_secret('jwt_secret_key', jwt_secret)
    
    app.config['JWT_SECRET_KEY'] = jwt_secret
    
    # Setup JWT callbacks
    setup_jwt_callbacks(jwt)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register blueprints
    from api.routes import auth
    try:
        from api.routes import orchestrator_simple as orchestrator
        logger.info("✅ Using simplified orchestrator routes")
    except ImportError:
        logger.warning("⚠️ Simplified orchestrator not available")
        orchestrator = None
    
    app.register_blueprint(auth.bp, url_prefix='/auth')
    app.register_blueprint(trading.bp, url_prefix='/api/v1/trading')
    app.register_blueprint(monitoring.bp, url_prefix='/api/v1/monitoring')
    app.register_blueprint(strategies.bp, url_prefix='/api/v1/strategies')
    
    if orchestrator:
        app.register_blueprint(orchestrator.bp, url_prefix='/api/v1/orchestrator')
    
    # Setup Swagger UI
    SWAGGER_URL = '/api/docs'
    API_URL = '/api/v1/swagger.json'
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "Altcoin Trading Bot API",
            'validatorUrl': None,
            'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch']
        }
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    
    # Register WebSocket handlers
    socket_handlers.register_handlers(socketio)
    
    # Store socketio instance on app for access in other modules
    app.socketio = socketio
    
    # Handle preflight OPTIONS requests
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
            response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'altcoin-trading-bot-api',
            'version': '1.0.0'
        }), 200
    
    # Swagger JSON endpoint
    @app.route('/api/v1/swagger.json', methods=['GET'])
    def swagger_spec():
        """Return OpenAPI specification"""
        from api.schemas.openapi import get_openapi_spec
        return jsonify(get_openapi_spec()), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information"""
        return jsonify({
            'service': 'Altcoin Trading Bot API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'docs': '/api/docs',
                'trading': '/api/v1/trading',
                'monitoring': '/api/v1/monitoring',
                'strategies': '/api/v1/strategies'
            }
        }), 200
    
    logger.info(f"Flask app created with config: {config_name}")
    
    return app, socketio


def run_app():
    """Run the Flask application"""
    app, socketio = create_app()
    
    # Get configuration from environment
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    
    # Run with SocketIO
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_app()