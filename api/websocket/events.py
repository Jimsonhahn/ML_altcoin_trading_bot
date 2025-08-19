"""
WebSocket Event Broadcasting
============================

Functions to broadcast events to WebSocket clients.
"""

from flask import current_app
from typing import Dict, Any
import logging
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.websocket.socket_handlers import broadcast_to_channel, broadcast_to_role, broadcast_to_user

logger = logging.getLogger(__name__)


def emit_trade_update(data: Dict[str, Any]):
    """Emit trading update to subscribed clients"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'trade_update',
            **data
        }
        
        # Broadcast to trading_updates channel
        broadcast_to_channel('trading_updates', 'trade_update', event_data)
        
        # Also broadcast to traders and admins
        broadcast_to_role('trader', 'trade_update', event_data)
        broadcast_to_role('admin', 'trade_update', event_data)
        
        logger.info(f"Trade update broadcasted: {data.get('event', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit trade update: {e}")


def emit_market_data(symbol: str, data: Dict[str, Any]):
    """Emit market data update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'market_data',
            'symbol': symbol,
            **data
        }
        
        # Broadcast to market_data channel
        broadcast_to_channel('market_data', 'market_data', event_data)
        
        logger.debug(f"Market data broadcasted for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to emit market data: {e}")


def emit_dashboard_update(data: Dict[str, Any]):
    """Emit dashboard update to all connected clients"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'dashboard_update',
            **data
        }
        
        # Broadcast to dashboard updates channel
        broadcast_to_channel('dashboard_updates', 'dashboard_update', event_data)
        broadcast_to_role('trader', 'dashboard_update', event_data)
        broadcast_to_role('admin', 'dashboard_update', event_data)
        
        logger.info(f"Dashboard update broadcasted: {data.get('type', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit dashboard update: {e}")


def emit_performance_update(data: Dict[str, Any]):
    """Emit performance metrics update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'performance_update',
            **data
        }
        
        # Broadcast to performance channel
        broadcast_to_channel('performance', 'performance_update', event_data)
        
        logger.info("Performance update broadcasted")
        
    except Exception as e:
        logger.error(f"Failed to emit performance update: {e}")


def emit_alert(alert_type: str, message: str, severity: str = 'info', target_user: str = None):
    """Emit alert to clients"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'alert',
            'alert_type': alert_type,
            'message': message,
            'severity': severity
        }
        
        if target_user:
            # Send to specific user
            broadcast_to_user(target_user, 'alert', event_data)
        else:
            # Broadcast to alerts channel
            broadcast_to_channel('alerts', 'alert', event_data)
            
            # Also send to admins for critical alerts
            if severity in ['critical', 'error']:
                broadcast_to_role('admin', 'alert', event_data)
        
        logger.info(f"Alert broadcasted: {alert_type} - {severity}")
        
    except Exception as e:
        logger.error(f"Failed to emit alert: {e}")


def emit_order_update(order_data: Dict[str, Any]):
    """Emit order status update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'order_update',
            **order_data
        }
        
        # Broadcast to trading_updates channel
        broadcast_to_channel('trading_updates', 'order_update', event_data)
        
        logger.info(f"Order update broadcasted: {order_data.get('order_id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit order update: {e}")


def emit_position_update(position_data: Dict[str, Any]):
    """Emit position update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'position_update',
            **position_data
        }
        
        # Broadcast to trading_updates channel
        broadcast_to_channel('trading_updates', 'position_update', event_data)
        
        logger.info(f"Position update broadcasted: {position_data.get('symbol', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit position update: {e}")


def emit_strategy_update(strategy_data: Dict[str, Any]):
    """Emit strategy update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'strategy_update',
            **strategy_data
        }
        
        # Broadcast to trading_updates channel
        broadcast_to_channel('trading_updates', 'strategy_update', event_data)
        
        logger.info(f"Strategy update broadcasted: {strategy_data.get('strategy', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit strategy update: {e}")


def emit_system_status(status_data: Dict[str, Any]):
    """Emit system status update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'system_status',
            **status_data
        }
        
        # Broadcast to all connected clients
        if hasattr(current_app, 'socketio'):
            current_app.socketio.emit('system_status', event_data)
        
        logger.info("System status broadcasted")
        
    except Exception as e:
        logger.error(f"Failed to emit system status: {e}")


def emit_error_event(error_data: Dict[str, Any]):
    """Emit error event to admins"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'error_event',
            **error_data
        }
        
        # Send to admins only
        broadcast_to_role('admin', 'error_event', event_data)
        
        logger.warning(f"Error event broadcasted: {error_data.get('error_id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit error event: {e}")


def emit_backtest_progress(progress_data: Dict[str, Any]):
    """Emit backtest progress update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'backtest_progress',
            **progress_data
        }
        
        # Send to user who initiated backtest
        if 'user' in progress_data:
            broadcast_to_user(progress_data['user'], 'backtest_progress', event_data)
        
        logger.info(f"Backtest progress broadcasted: {progress_data.get('progress', 0)}%")
        
    except Exception as e:
        logger.error(f"Failed to emit backtest progress: {e}")


def emit_bot_status_update(status_data: Dict[str, Any]):
    """Emit bot status update to all connected clients"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'bot_status_update',
            **status_data
        }
        
        # Broadcast to bot_status channel
        broadcast_to_channel('bot_status', 'bot_status_update', event_data)
        
        # Also broadcast to traders and admins
        broadcast_to_role('trader', 'bot_status_update', event_data)
        broadcast_to_role('admin', 'bot_status_update', event_data)
        
        logger.info(f"Bot status update broadcasted: {status_data.get('is_running', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to emit bot status update: {e}")


def emit_bot_performance_update(performance_data: Dict[str, Any]):
    """Emit bot performance metrics update"""
    try:
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'bot_performance_update',
            **performance_data
        }
        
        # Broadcast to bot_status channel
        broadcast_to_channel('bot_status', 'bot_performance_update', event_data)
        
        # Also broadcast to traders and admins
        broadcast_to_role('trader', 'bot_performance_update', event_data)
        broadcast_to_role('admin', 'bot_performance_update', event_data)
        
        logger.info("Bot performance update broadcasted")
        
    except Exception as e:
        logger.error(f"Failed to emit bot performance update: {e}")


def emit_connection_stats():
    """Emit connection statistics to admins"""
    try:
        from api.websocket.socket_handlers import get_active_connections
        
        connections = get_active_connections()
        
        stats = {
            'total_connections': len(connections),
            'users_online': len(set(conn['username'] for conn in connections.values())),
            'connections_by_role': {}
        }
        
        # Count connections by role
        for conn in connections.values():
            for role in conn['roles']:
                stats['connections_by_role'][role] = stats['connections_by_role'].get(role, 0) + 1
        
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'connection_stats',
            'stats': stats
        }
        
        # Send to admins
        broadcast_to_role('admin', 'connection_stats', event_data)
        
        logger.debug("Connection stats broadcasted")
        
    except Exception as e:
        logger.error(f"Failed to emit connection stats: {e}")