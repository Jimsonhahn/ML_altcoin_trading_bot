"""
Monitoring API Routes
=====================

Handles system monitoring and health check endpoints.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required
import logging
import psutil
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.safety_manager import SafetyManager
from utils.error_handler import SecureErrorHandler
from api.middleware.auth import require_admin

logger = logging.getLogger(__name__)

bp = Blueprint('monitoring', __name__)


@bp.route('/health', methods=['GET'])
def health_check():
    """
    System health check
    ---
    tags:
      - Monitoring
    responses:
      200:
        description: System health status
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  enum: [healthy, degraded, unhealthy]
                timestamp:
                  type: string
                  format: date-time
                checks:
                  type: object
    """
    checks = {
        'api': 'healthy',
        'database': 'healthy',  # Add actual DB check
        'exchange_connection': 'healthy',  # Add actual exchange check
        'memory': 'healthy',
        'disk': 'healthy'
    }
    
    # Memory check
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        checks['memory'] = 'unhealthy'
    elif memory.percent > 80:
        checks['memory'] = 'degraded'
    
    # Disk check
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        checks['disk'] = 'unhealthy'
    elif disk.percent > 80:
        checks['disk'] = 'degraded'
    
    # Overall status
    if any(status == 'unhealthy' for status in checks.values()):
        overall_status = 'unhealthy'
    elif any(status == 'degraded' for status in checks.values()):
        overall_status = 'degraded'
    else:
        overall_status = 'healthy'
    
    return jsonify({
        'status': overall_status,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': checks
    }), 200 if overall_status == 'healthy' else 503


@bp.route('/metrics', methods=['GET'])
@jwt_required()
def get_metrics():
    """
    Get system metrics
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    responses:
      200:
        description: System metrics
        content:
          application/json:
            schema:
              type: object
              properties:
                cpu:
                  type: object
                  properties:
                    percent:
                      type: number
                    count:
                      type: integer
                memory:
                  type: object
                  properties:
                    total:
                      type: integer
                    available:
                      type: integer
                    percent:
                      type: number
                disk:
                  type: object
                  properties:
                    total:
                      type: integer
                    used:
                      type: integer
                    percent:
                      type: number
                process:
                  type: object
                  properties:
                    pid:
                      type: integer
                    uptime:
                      type: number
                    threads:
                      type: integer
    """
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    
    # Disk metrics
    disk = psutil.disk_usage('/')
    
    # Process metrics
    process = psutil.Process(os.getpid())
    process_info = {
        'pid': process.pid,
        'uptime': (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds(),
        'threads': process.num_threads(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(interval=0.1)
    }
    
    return jsonify({
        'cpu': {
            'percent': cpu_percent,
            'count': cpu_count
        },
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used
        },
        'disk': {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        },
        'process': process_info,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@bp.route('/logs', methods=['GET'])
@require_admin
def get_logs():
    """
    Get recent logs
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: level
        schema:
          type: string
          enum: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
      - in: query
        name: limit
        schema:
          type: integer
          default: 100
      - in: query
        name: module
        schema:
          type: string
    responses:
      200:
        description: Recent log entries
    """
    level = request.args.get('level', 'INFO')
    limit = request.args.get('limit', 100, type=int)
    module = request.args.get('module')
    
    # In production, this would query a centralized logging system
    # For now, return mock data
    logs = [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'module': module or 'trading_bot',
            'message': 'Sample log entry'
        }
    ]
    
    return jsonify({
        'logs': logs,
        'count': len(logs),
        'filters': {
            'level': level,
            'limit': limit,
            'module': module
        }
    }), 200


@bp.route('/errors', methods=['GET'])
@require_admin
def get_errors():
    """
    Get recent errors
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
      - in: query
        name: category
        schema:
          type: string
    responses:
      200:
        description: Recent errors
    """
    limit = request.args.get('limit', 50, type=int)
    category = request.args.get('category')
    
    # Get errors from SecureErrorHandler
    error_handler = SecureErrorHandler('trading_bot_api')
    errors = error_handler.get_recent_errors(limit=limit, category=category)
    
    return jsonify({
        'errors': [
            {
                'error_id': error.error_id,
                'timestamp': error.timestamp,
                'category': error.category,
                'severity': error.severity,
                'message': error.message,
                'details': error.details
            }
            for error in errors
        ],
        'count': len(errors)
    }), 200


@bp.route('/error-stats', methods=['GET'])
@jwt_required()
def get_error_statistics():
    """
    Get error statistics
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    responses:
      200:
        description: Error statistics
    """
    error_handler = SecureErrorHandler('trading_bot_api')
    stats = error_handler.get_error_statistics()
    
    return jsonify(stats), 200


@bp.route('/safety-status', methods=['GET'])
@jwt_required()
def get_safety_status():
    """
    Get safety manager status
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    responses:
      200:
        description: Safety manager status
    """
    from api.routes.trading import trading_bot
    
    if not trading_bot or not hasattr(trading_bot, 'safety_manager'):
        return jsonify({
            'killswitch_active': False,
            'daily_loss_limit': 0,
            'current_daily_loss': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'safety_checks': []
        }), 200
    
    safety_manager = trading_bot.safety_manager
    
    return jsonify({
        'killswitch_active': safety_manager.is_killswitch_active(),
        'daily_loss_limit': safety_manager.daily_loss_limit_percent,
        'current_daily_loss': safety_manager.daily_pnl_percent,
        'max_drawdown': safety_manager.max_drawdown_percent,
        'current_drawdown': safety_manager.current_drawdown_percent,
        'safety_checks': {
            'daily_loss_limit': safety_manager.daily_pnl_percent < safety_manager.daily_loss_limit_percent,
            'max_drawdown': safety_manager.current_drawdown_percent < safety_manager.max_drawdown_percent,
            'consecutive_losses': safety_manager.consecutive_losses < safety_manager.max_consecutive_losses
        },
        'consecutive_losses': safety_manager.consecutive_losses,
        'max_consecutive_losses': safety_manager.max_consecutive_losses
    }), 200


@bp.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    """
    Get active system alerts
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    responses:
      200:
        description: Active alerts
    """
    alerts = []
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        alerts.append({
            'id': 'cpu_high',
            'severity': 'critical',
            'message': f'CPU usage is critically high: {cpu_percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    elif cpu_percent > 80:
        alerts.append({
            'id': 'cpu_warning',
            'severity': 'warning',
            'message': f'CPU usage is high: {cpu_percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        alerts.append({
            'id': 'memory_high',
            'severity': 'critical',
            'message': f'Memory usage is critically high: {memory.percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    elif memory.percent > 80:
        alerts.append({
            'id': 'memory_warning',
            'severity': 'warning',
            'message': f'Memory usage is high: {memory.percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    # Check disk usage
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        alerts.append({
            'id': 'disk_high',
            'severity': 'critical',
            'message': f'Disk usage is critically high: {disk.percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    elif disk.percent > 80:
        alerts.append({
            'id': 'disk_warning',
            'severity': 'warning',
            'message': f'Disk usage is high: {disk.percent}%',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    # Check trading bot status
    from api.routes.trading import trading_bot
    if trading_bot and hasattr(trading_bot, 'safety_manager'):
        if trading_bot.safety_manager.is_killswitch_active():
            alerts.append({
                'id': 'killswitch_active',
                'severity': 'critical',
                'message': 'Trading killswitch is active',
                'timestamp': datetime.utcnow().isoformat()
            })
    
    return jsonify({
        'alerts': alerts,
        'count': len(alerts)
    }), 200


@bp.route('/system-info', methods=['GET'])
@jwt_required()
def get_system_info():
    """
    Get system information
    ---
    tags:
      - Monitoring
    security:
      - BearerAuth: []
    responses:
      200:
        description: System information
    """
    import platform
    
    return jsonify({
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation()
        },
        'api': {
            'version': '1.0.0',
            'environment': os.environ.get('FLASK_ENV', 'development')
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200