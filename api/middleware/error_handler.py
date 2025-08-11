"""
API Error Handler Middleware
============================

Centralized error handling for the Flask API.
"""

from flask import jsonify, request
from werkzeug.exceptions import HTTPException
import logging
import traceback
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.error_handler import (
    SecureErrorHandler, TradingBotError, ValidationTradingError,
    NetworkTradingError, ExchangeTradingError, RateLimitTradingError
)

logger = logging.getLogger(__name__)
secure_handler = SecureErrorHandler('trading_bot_api')


def handle_validation_error(error: ValidationTradingError) -> tuple:
    """Handle validation errors"""
    response = secure_handler.handle_trading_error(
        error,
        endpoint=request.path,
        method=request.method
    )
    
    return jsonify({
        'error': 'Validation Error',
        'message': response.message,
        'field': error.additional_data.get('field'),
        'error_id': response.error_id,
        'details': response.details
    }), 400


def handle_network_error(error: NetworkTradingError) -> tuple:
    """Handle network errors"""
    response = secure_handler.handle_api_error(
        error,
        endpoint=request.path,
        status_code=error.additional_data.get('status_code', 500)
    )
    
    return jsonify({
        'error': 'Network Error',
        'message': response.message,
        'error_id': response.error_id,
        'retry_after': error.additional_data.get('retry_after')
    }), 503


def handle_exchange_error(error: ExchangeTradingError) -> tuple:
    """Handle exchange errors"""
    response = secure_handler.handle_trading_error(
        error,
        endpoint=request.path,
        exchange=error.additional_data.get('exchange')
    )
    
    return jsonify({
        'error': 'Exchange Error',
        'message': response.message,
        'exchange': error.additional_data.get('exchange'),
        'error_id': response.error_id
    }), 503


def handle_rate_limit_error(error: RateLimitTradingError) -> tuple:
    """Handle rate limit errors"""
    response = secure_handler.handle_api_error(
        error,
        endpoint=request.path,
        status_code=429
    )
    
    retry_after = error.additional_data.get('retry_after', 60)
    
    return jsonify({
        'error': 'Rate Limit Exceeded',
        'message': response.message,
        'retry_after': retry_after,
        'error_id': response.error_id
    }), 429


def handle_trading_bot_error(error: TradingBotError) -> tuple:
    """Handle generic trading bot errors"""
    response = secure_handler.handle_trading_error(
        error,
        endpoint=request.path,
        method=request.method
    )
    
    status_code = 500
    if error.category == 'validation':
        status_code = 400
    elif error.category == 'authentication':
        status_code = 401
    elif error.category == 'authorization':
        status_code = 403
    elif error.category in ['network', 'exchange']:
        status_code = 503
    
    return jsonify({
        'error': 'Trading Bot Error',
        'message': response.message,
        'category': error.category,
        'error_id': response.error_id
    }), status_code


def handle_http_exception(error: HTTPException) -> tuple:
    """Handle HTTP exceptions"""
    return jsonify({
        'error': error.name,
        'message': error.description,
        'status_code': error.code
    }), error.code


def handle_generic_exception(error: Exception) -> tuple:
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {error}", exc_info=True)
    
    response = secure_handler.handle_critical_error(
        error,
        context={
            'endpoint': request.path,
            'method': request.method,
            'traceback': traceback.format_exc()
        }
    )
    
    # In production, don't expose internal errors
    if request.environ.get('FLASK_ENV') == 'production':
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'error_id': response.error_id
        }), 500
    else:
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(error),
            'error_id': response.error_id,
            'traceback': traceback.format_exc()
        }), 500


def register_error_handlers(app):
    """Register all error handlers with the Flask app"""
    
    # Trading bot specific errors
    app.register_error_handler(ValidationTradingError, handle_validation_error)
    app.register_error_handler(NetworkTradingError, handle_network_error)
    app.register_error_handler(ExchangeTradingError, handle_exchange_error)
    app.register_error_handler(RateLimitTradingError, handle_rate_limit_error)
    app.register_error_handler(TradingBotError, handle_trading_bot_error)
    
    # HTTP exceptions
    app.register_error_handler(HTTPException, handle_http_exception)
    
    # Generic exceptions
    app.register_error_handler(Exception, handle_generic_exception)
    
    # Add before_request handler for request logging
    @app.before_request
    def log_request():
        """Log incoming requests"""
        logger.info(f"{request.method} {request.path} from {request.remote_addr}")
    
    # Add after_request handler for response logging
    @app.after_request
    def log_response(response):
        """Log outgoing responses"""
        logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response