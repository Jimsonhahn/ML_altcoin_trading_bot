"""
Trading Mode Manager API Routes
===============================

Erweitert das bestehende System um Multi-Mode Trading Support
ohne bestehende Funktionalität zu beeinträchtigen.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.middleware.auth import require_admin
from api.websocket.events import emit_dashboard_update
from core.trading_engine_manager import TradingEngineManager

logger = logging.getLogger(__name__)

bp = Blueprint('trading_mode_manager', __name__)

# Global Trading Engine Manager Instance
# Wird vom bestehenden bot_manager integriert
trading_engine_manager = None


def initialize_trading_engine_manager(settings=None):
    """Initialisiert Trading Engine Manager (wird vom Hauptsystem aufgerufen)"""
    global trading_engine_manager
    try:
        trading_engine_manager = TradingEngineManager(settings)
        logger.info("✅ Trading Engine Manager initialized")
        return trading_engine_manager
    except Exception as e:
        logger.error(f"❌ Failed to initialize Trading Engine Manager: {e}")
        return None


@bp.route('/modes', methods=['GET'])
@jwt_required()
def get_available_trading_modes():
    """
    Zeigt alle verfügbaren Trading Modi mit Status
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    responses:
      200:
        description: Available trading modes
        content:
          application/json:
            schema:
              type: object
              properties:
                current_mode:
                  type: string
                  description: Currently active trading mode
                available_modes:
                  type: object
                  description: Details of all available modes
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': False,
                'message': 'Trading Engine Manager not initialized',
                'fallback_mode': 'simulated'
            }), 503
        
        modes_info = trading_engine_manager.get_available_modes()
        
        return jsonify({
            'success': True,
            'data': modes_info,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting trading modes: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get trading modes: {str(e)}'
        }), 500


@bp.route('/switch-mode', methods=['POST'])
@jwt_required()
def switch_trading_mode():
    """
    Wechselt zwischen Trading Modi (erweitert bestehende Funktionalität)
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              mode:
                type: string
                enum: [simulated, real_paper, live]
                description: Target trading mode
    responses:
      200:
        description: Mode switched successfully
      400:
        description: Invalid mode or switch not possible
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': False,
                'message': 'Trading Engine Manager not initialized - using fallback mode'
            }), 503
        
        data = request.json
        new_mode = data.get('mode')
        
        if not new_mode:
            return jsonify({
                'success': False,
                'message': 'Mode parameter required'
            }), 400
        
        # Versuche Mode-Wechsel
        user = get_jwt_identity()
        result = trading_engine_manager.switch_mode(new_mode, user=user)
        
        if result['success']:
            # Sende Update an Dashboard via WebSocket
            emit_dashboard_update({
                'type': 'mode_changed',
                'old_mode': result['old_mode'],
                'new_mode': result['new_mode'],
                'message': result['message'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"🔄 Trading mode switched: {result['old_mode']} → {result['new_mode']}")
            
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"❌ Error switching trading mode: {e}")
        return jsonify({
            'success': False,
            'message': f'Mode switch failed: {str(e)}'
        }), 500


@bp.route('/setup-exchange', methods=['POST'])
@jwt_required()
@require_admin
def setup_exchange_credentials():
    """
    Konfiguriert Exchange API Credentials für real_paper und live Modi
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              exchange:
                type: string
                enum: [binance, coinbase]
                description: Exchange name
              testnet_api_key:
                type: string
                description: Testnet/Demo API key
              testnet_api_secret:
                type: string
                description: Testnet/Demo API secret
              live_api_key:
                type: string
                description: Live API key (optional)
              live_api_secret:
                type: string
                description: Live API secret (optional)
    responses:
      200:
        description: Exchange credentials configured successfully
      400:
        description: Invalid configuration
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': False,
                'message': 'Trading Engine Manager not initialized'
            }), 503
        
        exchange_config = request.json
        
        # Validiere erforderliche Felder
        required_fields = ['exchange']
        if not all(field in exchange_config for field in required_fields):
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {required_fields}'
            }), 400
        
        # Setup Exchange Credentials
        result = trading_engine_manager.setup_exchange_credentials(exchange_config)
        
        if result['success']:
            logger.info(f"✅ Exchange credentials configured: {exchange_config.get('exchange')}")
            
            # Sende Update an Dashboard
            emit_dashboard_update({
                'type': 'exchange_configured',
                'exchange': exchange_config.get('exchange'),
                'available_modes': result.get('available_modes', {}),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"❌ Exchange setup failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Exchange setup failed: {str(e)}'
        }), 500


@bp.route('/portfolio-status', methods=['GET'])
@jwt_required()
def get_enhanced_portfolio_status():
    """
    Erweiterte Portfolio Status API - unterstützt alle Modi
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    responses:
      200:
        description: Enhanced portfolio status for current mode
    """
    try:
        if not trading_engine_manager:
            # Fallback zu bestehendem System
            return jsonify({
                'success': True,
                'mode': 'simulated_fallback',
                'message': 'Using fallback simulated mode',
                'data': {
                    'mode': 'SIMULATED (FALLBACK)',
                    'total_value': 10000,
                    'engine_available': True
                }
            }), 200
        
        portfolio_status = trading_engine_manager.get_portfolio_status()
        
        return jsonify({
            'success': True,
            'data': portfolio_status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting enhanced portfolio status: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get portfolio status: {str(e)}'
        }), 500


@bp.route('/active-trades', methods=['GET'])
@jwt_required()
def get_enhanced_active_trades():
    """
    Erweiterte Active Trades API - unterstützt alle Modi
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    responses:
      200:
        description: Enhanced active trades for current mode
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': True,
                'mode': 'simulated_fallback',
                'data': {
                    'trades': [],
                    'total_trades': 0,
                    'message': 'Using fallback mode - no active trades'
                }
            }), 200
        
        active_trades = trading_engine_manager.get_active_trades()
        
        return jsonify({
            'success': True,
            'data': {
                'trades': active_trades,
                'total_trades': len(active_trades),
                'mode': trading_engine_manager.current_mode
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting enhanced active trades: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get active trades: {str(e)}'
        }), 500


@bp.route('/execute-trade', methods=['POST'])
@jwt_required()
def execute_enhanced_trade():
    """
    Erweiterte Trade Execution - nutzt aktuelle Engine
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              symbol:
                type: string
                description: Trading pair (e.g., BTC/USDT)
              side:
                type: string
                enum: [LONG, SHORT]
                description: Trade direction
              size:
                type: number
                description: Position size
              strategy:
                type: string
                description: Strategy name
    responses:
      200:
        description: Trade executed successfully
      400:
        description: Invalid trade parameters
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': False,
                'message': 'Trading Engine Manager not initialized'
            }), 503
        
        trade_data = request.json
        
        # Validiere Trade Parameter
        required_fields = ['symbol', 'side', 'size']
        if not all(field in trade_data for field in required_fields):
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {required_fields}'
            }), 400
        
        # Execute Trade mit aktueller Engine
        import asyncio
        user = get_jwt_identity()
        trade_result = asyncio.run(trading_engine_manager.execute_trade(
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            size=trade_data['size'],
            strategy=trade_data.get('strategy', 'manual'),
            user=user
        ))
        
        if trade_result:
            # Sende Trade Update an Dashboard
            emit_dashboard_update({
                'type': 'trade_executed',
                'trade': {
                    'id': trade_result.id if hasattr(trade_result, 'id') else 'unknown',
                    'symbol': trade_data['symbol'],
                    'side': trade_data['side'],
                    'mode': trading_engine_manager.current_mode
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return jsonify({
                'success': True,
                'message': 'Trade executed successfully',
                'trade_id': trade_result.id if hasattr(trade_result, 'id') else 'unknown',
                'mode': trading_engine_manager.current_mode
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Trade execution failed'
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Enhanced trade execution failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Trade execution failed: {str(e)}'
        }), 500


@bp.route('/reset-account', methods=['POST'])
@jwt_required()
@require_admin
def reset_trading_account():
    """
    Reset Trading Account (nur für Paper Modi)
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    responses:
      200:
        description: Account reset successfully
      400:
        description: Reset not available for current mode
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': False,
                'message': 'Trading Engine Manager not initialized'
            }), 503
        
        user = get_jwt_identity()
        result = trading_engine_manager.reset_account(user=user)
        
        if result['success']:
            # Sende Reset Event an Dashboard
            emit_dashboard_update({
                'type': 'account_reset',
                'mode': trading_engine_manager.current_mode,
                'message': result['message'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"❌ Account reset failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Account reset failed: {str(e)}'
        }), 500


@bp.route('/exchange-info', methods=['GET'])
@jwt_required()
def get_exchange_info():
    """
    Liefert Exchange Informationen für real_paper und live Modi
    ---
    tags:
      - Trading Mode Manager
    security:
      - BearerAuth: []
    responses:
      200:
        description: Exchange information
    """
    try:
        if not trading_engine_manager:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'manager_not_initialized',
                    'message': 'Using simulated mode only'
                }
            }), 200
        
        info = {'exchanges': {}}
        
        # Real Paper Engine Info
        if (trading_engine_manager.real_paper_engine and 
            hasattr(trading_engine_manager.real_paper_engine, 'get_exchange_info')):
            import asyncio
            real_paper_info = asyncio.run(trading_engine_manager.real_paper_engine.get_exchange_info())
            info['exchanges']['real_paper'] = real_paper_info
        
        # Live Engine Info (wenn verfügbar)
        if (trading_engine_manager.live_engine and 
            hasattr(trading_engine_manager.live_engine, 'get_exchange_info')):
            import asyncio
            live_info = asyncio.run(trading_engine_manager.live_engine.get_exchange_info())
            info['exchanges']['live'] = live_info
        
        info['current_mode'] = trading_engine_manager.current_mode
        info['available_modes'] = trading_engine_manager.available_modes
        
        return jsonify({
            'success': True,
            'data': info,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting exchange info: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get exchange info: {str(e)}'
        }), 500


# Integration Helper für bestehende API
def get_trading_engine_manager():
    """Helper Funktion für Integration in bestehende APIs"""
    return trading_engine_manager