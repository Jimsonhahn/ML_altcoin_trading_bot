"""
Trading API Routes
==================

Handles all trading-related endpoints.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.trading_bot import TradingBot
from core.paper_trading_engine import PaperTradingEngine
from utils.validators import validate_trading_symbol, validate_amount, validate_order
from utils.error_handler import ValidationTradingError, ExchangeTradingError
from api.middleware.auth import require_trader, require_admin
from api.websocket.events import emit_trade_update
from api.services.bot_manager import bot_manager

logger = logging.getLogger(__name__)

bp = Blueprint('trading', __name__)

# Global trading bot instance (in production, use proper state management)
trading_bot = None


@bp.route('/status', methods=['GET'])
@jwt_required()
def get_trading_status():
    """
    Get accurate bot status with real-time verification
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Trading bot status
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                status:
                  type: object
                timestamp:
                  type: string
                verified:
                  type: boolean
    """
    try:
        # Get verified status from BotManager
        result = bot_manager.get_verified_status()
        
        # Add additional info
        result['status'].update({
            'api_version': '1.0.0',
            'available_strategies': ['momentum', 'mean_reversion', 'grid_trading', 'arbitrage', 'ml_strategy'],
            'available_symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT']
        })
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        # Return safe default status
        return jsonify({
            'success': False,
            'message': 'Failed to get trading status',
            'error': str(e),
            'status': {
                'is_running': False,
                'pid': None,
                'strategy': None,
                'mode': 'paper',
                'symbol': None,
                'start_time': None,
                'config': {},
                'last_update': datetime.now(timezone.utc).isoformat(),
                'performance': {
                    'total_pnl': 0,
                    'daily_pnl': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'active_positions': 0
                }
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'verified': False
        }), 200  # Return 200 even on error to avoid dashboard issues


@bp.route('/detailed-status', methods=['GET'])
@jwt_required()
def get_detailed_trading_status():
    """
    Get detailed trading bot status including trades and performance
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Detailed trading bot status
    """
    try:
        result = bot_manager.get_detailed_status()
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting detailed trading status: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get detailed trading status'
        }), 500


@bp.route('/force-stop', methods=['POST'])
@jwt_required()
@require_admin
def force_stop_bot():
    """
    Force stop bot and cleanup all processes
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Bot force stopped successfully
    """
    try:
        result = bot_manager.force_cleanup()
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error force stopping bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to force stop bot: {str(e)}'
        }), 500


@bp.route('/start', methods=['POST'])
@jwt_required()
def start_trading():
    """
    Start the trading bot with configuration
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - mode
              - strategy
              - symbol
            properties:
              mode:
                type: string
                enum: [live, paper]
                description: Trading mode
              strategy:
                type: string
                description: Trading strategy name
              symbol:
                type: string
                description: Trading pair (e.g., BTC/USDT)
              capital:
                type: number
                description: Initial capital
              risk_per_trade:
                type: number
                description: Risk percentage per trade
              strategy_params:
                type: object
                description: Strategy-specific parameters
    responses:
      200:
        description: Trading bot started successfully
      400:
        description: Invalid configuration
      409:
        description: Bot is already running
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No configuration provided'
            }), 400
        
        # Check if bot is REALLY running first
        if bot_manager.is_actually_running():
            return jsonify({
                'success': False,
                'message': 'Bot is already running',
                'pid': bot_manager.bot_status.get('pid'),
                'strategy': bot_manager.bot_status.get('strategy'),
                'mode': bot_manager.bot_status.get('mode')
            }), 400
        
        # Extract and validate configuration
        config = {
            'mode': data.get('mode', 'paper'),
            'strategy': data.get('strategy'),
            'symbol': data.get('symbol'),
            'capital': data.get('capital', 10000),
            'risk_per_trade': data.get('risk_per_trade', 0.02),
            'strategy_params': data.get('strategy_params', {})
        }
        
        # Validate required fields
        required_fields = ['strategy', 'symbol']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Additional validation
        valid_modes = ['paper', 'live']
        if config['mode'] not in valid_modes:
            return jsonify({
                'success': False,
                'message': f'Invalid mode. Must be one of: {", ".join(valid_modes)}'
            }), 400
        
        valid_strategies = ['momentum', 'mean_reversion', 'grid_trading', 'arbitrage', 'ml_strategy']
        if config['strategy'] not in valid_strategies:
            return jsonify({
                'success': False,
                'message': f'Invalid strategy. Must be one of: {", ".join(valid_strategies)}'
            }), 400
        
        # Try to start bot via BotManager
        result = bot_manager.start_bot(config)
        
        if result['success']:
            # Broadcast status update via WebSocket
            try:
                emit_bot_status_update({
                    'is_running': True,
                    'status': result['status'],
                    'event': 'bot_started',
                    'message': 'Bot started successfully'
                })
            except Exception as e:
                logger.warning(f"Failed to emit WebSocket event: {e}")
            
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to start trading bot: {str(e)}'
        }), 500


@bp.route('/stop', methods=['POST'])
@jwt_required()
def stop_trading():
    """
    Stop the trading bot
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Trading bot stopped successfully
      400:
        description: Bot is not running
    """
    try:
        result = bot_manager.stop_bot()
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error stopping trading bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to stop trading bot: {str(e)}'
        }), 500


@bp.route('/restart', methods=['POST'])
@jwt_required()
def restart_trading():
    """
    Restart the trading bot with optional new configuration
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    requestBody:
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              mode:
                type: string
                enum: [live, paper]
              strategy:
                type: string
              symbol:
                type: string
              capital:
                type: number
              risk_per_trade:
                type: number
              strategy_params:
                type: object
    responses:
      200:
        description: Trading bot restarted successfully
      400:
        description: Invalid configuration
    """
    try:
        data = request.json
        new_config = None
        
        if data:
            new_config = {
                'mode': data.get('mode'),
                'strategy': data.get('strategy'),
                'symbol': data.get('symbol'),
                'capital': data.get('capital'),
                'risk_per_trade': data.get('risk_per_trade'),
                'strategy_params': data.get('strategy_params', {})
            }
            # Remove None values
            new_config = {k: v for k, v in new_config.items() if v is not None}
        
        result = bot_manager.restart_bot(new_config)
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to restart trading bot: {str(e)}'
        }), 500


@bp.route('/positions', methods=['GET'])
@jwt_required()
def get_positions():
    """
    Get all open positions
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: List of open positions
    """
    global trading_bot
    
    if not trading_bot:
        return jsonify({'positions': []}), 200
    
    positions = trading_bot.position_manager.get_all_positions()
    current_prices = trading_bot.exchange.get_current_prices()
    
    position_list = []
    for symbol, pos in positions.items():
        current_price = current_prices.get(symbol, pos.get('entry_price', 0))
        
        position_list.append({
            'symbol': symbol,
            'amount': pos.get('amount', 0),
            'entry_price': pos.get('entry_price', 0),
            'current_price': current_price,
            'unrealized_pnl': pos.get('unrealized_pnl', 0),
            'realized_pnl': pos.get('realized_pnl', 0),
            'entry_time': pos.get('entry_time')
        })
    
    return jsonify({'positions': position_list}), 200


@bp.route('/orders', methods=['GET'])
@jwt_required()
def get_orders():
    """
    Get recent orders
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
      - in: query
        name: symbol
        schema:
          type: string
    responses:
      200:
        description: List of orders
    """
    global trading_bot
    
    if not trading_bot:
        return jsonify({'orders': []}), 200
    
    limit = request.args.get('limit', 50, type=int)
    symbol = request.args.get('symbol')
    
    # Get orders from order manager
    all_orders = trading_bot.order_manager.get_order_history()
    
    # Filter by symbol if provided
    if symbol:
        all_orders = [o for o in all_orders if o.get('symbol') == symbol]
    
    # Limit results
    orders = all_orders[-limit:]
    
    return jsonify({'orders': orders}), 200


@bp.route('/manual-order', methods=['POST'])
@require_trader
def create_manual_order():
    """
    Create a manual order
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - symbol
              - side
              - amount
            properties:
              symbol:
                type: string
              side:
                type: string
                enum: [buy, sell]
              amount:
                type: number
              order_type:
                type: string
                enum: [market, limit]
                default: market
              price:
                type: number
    responses:
      200:
        description: Order created
      400:
        description: Invalid order parameters
    """
    global trading_bot
    
    if not trading_bot or not trading_bot.running:
        raise ExchangeTradingError("Trading bot is not running")
    
    data = request.json
    
    try:
        # Validate order data
        validated_order = validate_order(data)
        
        # Create order based on type
        if validated_order.order_type == 'market':
            if validated_order.side == 'buy':
                order = trading_bot.order_manager.create_market_buy_order(
                    validated_order.symbol,
                    validated_order.amount
                )
            else:
                order = trading_bot.order_manager.create_market_sell_order(
                    validated_order.symbol,
                    validated_order.amount
                )
        else:  # limit order
            order = trading_bot.order_manager.create_limit_order(
                validated_order.symbol,
                validated_order.side,
                validated_order.amount,
                validated_order.price
            )
        
        # Update positions
        trading_bot.position_manager.update_position_from_order(order)
        
        # Emit WebSocket event
        emit_trade_update({
            'event': 'manual_order_created',
            'order': order,
            'user': get_jwt_identity()
        })
        
        return jsonify({
            'status': 'success',
            'order': order
        }), 200
        
    except ValidationTradingError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to create manual order: {e}")
        raise ExchangeTradingError(f"Failed to create order: {str(e)}")


@bp.route('/cancel-order/<order_id>', methods=['POST'])
@require_trader
def cancel_order(order_id: str):
    """
    Cancel an open order
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: order_id
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - symbol
            properties:
              symbol:
                type: string
    responses:
      200:
        description: Order cancelled
    """
    global trading_bot
    
    if not trading_bot:
        raise ExchangeTradingError("Trading bot is not running")
    
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        raise ValidationTradingError("Symbol is required", field="symbol")
    
    try:
        result = trading_bot.order_manager.cancel_order(order_id, symbol)
        
        if result:
            # Emit WebSocket event
            emit_trade_update({
                'event': 'order_cancelled',
                'order_id': order_id,
                'symbol': symbol,
                'user': get_jwt_identity()
            })
            
            return jsonify({
                'status': 'success',
                'message': f'Order {order_id} cancelled'
            }), 200
        else:
            raise ExchangeTradingError(f"Failed to cancel order {order_id}")
            
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise ExchangeTradingError(f"Failed to cancel order: {str(e)}")


@bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance():
    """
    Get trading performance metrics
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: period
        schema:
          type: string
          enum: [day, week, month, all]
          default: all
    responses:
      200:
        description: Performance metrics
    """
    global trading_bot
    
    if not trading_bot:
        return jsonify({
            'total_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0
        }), 200
    
    period = request.args.get('period', 'all')
    
    # Get performance metrics
    metrics = trading_bot.performance_tracker.get_performance_summary(period)
    
    return jsonify(metrics), 200


@bp.route('/logs', methods=['GET'])
@jwt_required()
def get_bot_logs():
    """Get recent bot logs"""
    try:
        logs = bot_manager.bot_status.get('logs', [])
        return jsonify({
            'success': True,
            'logs': logs
        }), 200
    except Exception as e:
        logger.error(f"Error getting bot logs: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'logs': []
        }), 500


@bp.route('/backtest', methods=['POST'])
@require_admin
def run_backtest():
    """
    Run a backtest
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - symbol
              - start_date
              - end_date
              - strategy
            properties:
              symbol:
                type: string
              start_date:
                type: string
                format: date
              end_date:
                type: string
                format: date
              strategy:
                type: string
              timeframe:
                type: string
                default: 1h
    responses:
      200:
        description: Backtest results
    """
    data = request.json
    
    try:
        # Validate inputs
        symbol = validate_trading_symbol(data.get('symbol')).symbol
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        strategy = data.get('strategy', 'momentum')
        timeframe = data.get('timeframe', '1h')
        
        # Create backtest bot
        from config.settings import Settings
        from data_sources.data_manager import DataManager
        
        settings = Settings('default')
        data_manager = DataManager(settings)
        
        backtest_bot = TradingBot(
            mode='backtest',
            strategy_name=strategy,
            settings=settings,
            data_manager=data_manager
        )
        
        # Run backtest
        backtest_bot.run_backtest(symbol, timeframe, start_date, end_date)
        
        # Get results
        results = backtest_bot.performance_tracker.get_backtest_results()
        
        return jsonify({
            'status': 'success',
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise ExchangeTradingError(f"Backtest failed: {str(e)}")


@bp.route('/mode', methods=['POST'])
@jwt_required()
def switch_trading_mode():
    """
    Switch between Paper and Live trading modes
    ---
    tags:
      - Trading
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
                enum: [paper, live]
                description: Trading mode to switch to
              initial_balance:
                type: number
                description: Initial balance for paper trading (optional)
    responses:
      200:
        description: Trading mode switched successfully
      400:
        description: Invalid mode or bot not running
    """
    try:
        data = request.json
        mode = data.get('mode')
        
        if mode not in ['paper', 'live']:
            return jsonify({
                'success': False,
                'message': 'Invalid mode. Must be "paper" or "live"'
            }), 400
        
        # Check if bot is running
        status = bot_manager.get_status()
        if not status['is_running']:
            return jsonify({
                'success': False,
                'message': 'Bot must be running to switch modes'
            }), 400
        
        # Get current config and update mode
        current_config = bot_manager.get_bot_config()
        new_config = current_config.copy() if current_config else {}
        
        # Set paper trading flag
        if mode == 'paper':
            new_config['paper_trading'] = True
            new_config['paper_trading_balance'] = data.get('initial_balance', 10000.0)
            message = f'Switched to Paper Trading mode with ${new_config["paper_trading_balance"]} virtual balance'
        else:
            new_config['paper_trading'] = False
            message = 'Switched to Live Trading mode'
        
        # Restart bot with new configuration
        result = bot_manager.restart_bot(new_config)
        
        if result['success']:
            # Emit mode change event via WebSocket
            emit_trade_update({
                'type': 'mode_changed',
                'mode': mode,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return jsonify({
                'success': True,
                'message': message,
                'mode': mode
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to switch mode: {result.get("message", "Unknown error")}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error switching trading mode: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to switch trading mode: {str(e)}'
        }), 500


@bp.route('/paper/status', methods=['GET'])
@jwt_required()
def get_paper_trading_status():
    """
    Get Paper Trading portfolio status
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Paper trading portfolio status
      404:
        description: Paper trading not active
    """
    try:
        # Check if bot is running and in paper mode
        status = bot_manager.get_status()
        if not status['is_running']:
            return jsonify({
                'success': False,
                'message': 'Bot is not running'
            }), 404
        
        # Get bot instance
        bot_instance = bot_manager.get_bot_instance()
        if not bot_instance or not hasattr(bot_instance, 'paper_engine') or not bot_instance.paper_engine:
            return jsonify({
                'success': False,
                'message': 'Paper trading is not active'
            }), 404
        
        # Get paper trading status
        paper_status = bot_instance.paper_engine.get_virtual_portfolio_status()
        
        return jsonify({
            'success': True,
            'data': paper_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting paper trading status: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get paper trading status: {str(e)}'
        }), 500


@bp.route('/paper/reset', methods=['POST'])
@jwt_required()
@require_admin
def reset_paper_account():
    """
    Reset Paper Trading account to initial state
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    responses:
      200:
        description: Paper account reset successfully
      404:
        description: Paper trading not active
    """
    try:
        # Get bot instance
        bot_instance = bot_manager.get_bot_instance()
        if not bot_instance or not hasattr(bot_instance, 'paper_engine') or not bot_instance.paper_engine:
            return jsonify({
                'success': False,
                'message': 'Paper trading is not active'
            }), 404
        
        # Reset paper account
        bot_instance.paper_engine.reset_paper_account()
        
        # Emit reset event
        emit_trade_update({
            'type': 'paper_account_reset',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return jsonify({
            'success': True,
            'message': 'Paper trading account reset successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error resetting paper account: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to reset paper account: {str(e)}'
        }), 500


@bp.route('/paper/history', methods=['GET'])
@jwt_required()
def get_paper_trade_history():
    """
    Get Paper Trading trade history
    ---
    tags:
      - Trading
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: limit
        schema:
          type: integer
          default: 50
        description: Number of trades to return
    responses:
      200:
        description: Paper trading history
      404:
        description: Paper trading not active
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Get bot instance
        bot_instance = bot_manager.get_bot_instance()
        if not bot_instance or not hasattr(bot_instance, 'paper_engine') or not bot_instance.paper_engine:
            return jsonify({
                'success': False,
                'message': 'Paper trading is not active'
            }), 404
        
        # Get trade history
        all_trades = bot_instance.paper_engine.trade_history
        
        # Convert to serializable format and limit
        trade_data = []
        for trade in all_trades[-limit:]:
            trade_dict = {
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_percentage': trade.pnl_percentage,
                'strategy': trade.strategy,
                'timestamp': trade.timestamp.isoformat(),
                'exit_timestamp': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                'duration_minutes': trade.duration_minutes,
                'fee': trade.fee
            }
            trade_data.append(trade_dict)
        
        return jsonify({
            'success': True,
            'data': {
                'trades': trade_data,
                'total_trades': len(all_trades),
                'showing': len(trade_data)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting paper trade history: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to get trade history: {str(e)}'
        }), 500