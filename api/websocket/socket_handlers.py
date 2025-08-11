"""
WebSocket Event Handlers
========================

Handles WebSocket connections and events for real-time updates.
"""

from flask_socketio import emit, join_room, leave_room, disconnect
from flask import request
import logging
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Store active connections
active_connections = {}


def register_handlers(socketio):
    """Register WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect(auth=None):
        """Handle client connection"""
        try:
            logger.info(f"Client connected: {request.sid}")
            
            # Store connection info (simplified for now)
            active_connections[request.sid] = {
                'username': 'dashboard',
                'roles': ['viewer'],
                'connected_at': None
            }
            
            # Join trading updates room
            join_room('trading_updates')
            
            # Send welcome message
            emit('connected', {
                'status': 'Connected to trading server',
                'session_id': request.sid
            })
            
            # Send initial bot status
            try:
                from api.services.bot_manager import bot_manager
                status = bot_manager.get_status()
                emit('bot_status_update', {'status': status['status']})
            except Exception as e:
                logger.warning(f"Could not send initial status: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            disconnect()
            return False
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        try:
            session_id = request.sid
            logger.info(f"Client disconnected: {session_id}")
            
            if session_id in active_connections:
                # Leave trading updates room
                leave_room('trading_updates')
                
                # Remove from active connections
                del active_connections[session_id]
            
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle subscription to specific events"""
        try:
            rooms = data.get('rooms', ['trading_updates']) if data else ['trading_updates']
            for room in rooms:
                join_room(room)
                logger.info(f"Client {request.sid} joined room: {room}")
            emit('subscribed', {'rooms': rooms})
        except Exception as e:
            logger.error(f"Error in subscribe: {e}")
    
    @socketio.on('request_status')
    def handle_status_request():
        """Handle status request"""
        try:
            from api.services.bot_manager import bot_manager
            status = bot_manager.get_status()
            emit('bot_status_update', {'status': status['status']})
        except Exception as e:
            logger.error(f"Error getting status: {e}")
    
    @socketio.on('subscribe_orchestrator')
    def handle_orchestrator_subscription():
        """Handle orchestrator-specific subscriptions"""
        try:
            join_room('orchestrator_updates')
            join_room('portfolio_updates')
            join_room('health_alerts')
            
            logger.info(f"Client {request.sid} subscribed to orchestrator updates")
            
            emit('orchestrator_subscribed', {
                'status': 'subscribed',
                'rooms': ['orchestrator_updates', 'portfolio_updates', 'health_alerts']
            })
        except Exception as e:
            logger.error(f"Error in orchestrator subscription: {e}")
    
    @socketio.on('request_orchestrator_status')
    def handle_orchestrator_status_request():
        """Handle orchestrator status request"""
        try:
            from api.routes.orchestrator import get_orchestrator_instances
            orchestrator, engine, health, portfolio = get_orchestrator_instances()
            
            # Get portfolio state
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            portfolio_state = loop.run_until_complete(portfolio.get_portfolio_state())
            
            emit('orchestrator_status_update', {
                'mode': portfolio.mode.value,
                'total_value': portfolio_state.total_value,
                'total_pnl': portfolio_state.total_pnl,
                'win_rate': portfolio_state.win_rate,
                'positions': portfolio_state.total_positions
            })
        except Exception as e:
            logger.error(f"Error getting orchestrator status: {e}")
    
    logger.info("WebSocket handlers registered successfully")


def broadcast_to_channel(channel: str, event: str, data: Dict[str, Any]):
    """Broadcast message to all clients in a channel"""
    try:
        from api.app import socketio
        socketio.emit(event, data, room=channel)
        logger.debug(f"Broadcasted {event} to channel {channel}")
    except Exception as e:
        logger.error(f"Error broadcasting to channel {channel}: {e}")


def broadcast_to_role(role: str, event: str, data: Dict[str, Any]):
    """Broadcast message to all clients with specific role"""
    try:
        from api.app import socketio
        # For simplicity, broadcast to trading_updates room
        # In production, you'd filter by actual roles
        socketio.emit(event, data, room='trading_updates')
        logger.debug(f"Broadcasted {event} to role {role}")
    except Exception as e:
        logger.error(f"Error broadcasting to role {role}: {e}")


def broadcast_to_user(user_id: str, event: str, data: Dict[str, Any]):
    """Broadcast message to specific user"""
    try:
        from api.app import socketio
        # Find user's session and emit directly
        for sid, conn_info in active_connections.items():
            if conn_info.get('username') == user_id:
                socketio.emit(event, data, room=sid)
                logger.debug(f"Broadcasted {event} to user {user_id}")
                break
    except Exception as e:
        logger.error(f"Error broadcasting to user {user_id}: {e}")