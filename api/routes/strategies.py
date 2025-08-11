"""
Strategy Management API Routes
===============================

Handles strategy configuration and management endpoints.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from strategies import STRATEGIES
from strategies.strategy_base import Strategy
from utils.validators import validate_config
from utils.error_handler import ValidationTradingError
from api.middleware.auth import require_admin, require_trader

logger = logging.getLogger(__name__)

bp = Blueprint('strategies', __name__)


@bp.route('/list', methods=['GET'])
@jwt_required()
def list_strategies():
    """
    List all available strategies
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    responses:
      200:
        description: List of available strategies
        content:
          application/json:
            schema:
              type: object
              properties:
                strategies:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      description:
                        type: string
                      parameters:
                        type: object
                      risk_level:
                        type: string
                        enum: [low, medium, high]
    """
    strategy_list = []
    
    for name, strategy_class in STRATEGIES.items():
        try:
            # Get strategy metadata
            strategy_info = {
                'name': name,
                'description': getattr(strategy_class, 'description', 'No description available'),
                'parameters': getattr(strategy_class, 'default_parameters', {}),
                'risk_level': getattr(strategy_class, 'risk_level', 'medium'),
                'timeframes': getattr(strategy_class, 'supported_timeframes', ['1h', '4h', '1d']),
                'markets': getattr(strategy_class, 'supported_markets', ['spot'])
            }
            
            strategy_list.append(strategy_info)
            
        except Exception as e:
            logger.error(f"Error loading strategy {name}: {e}")
    
    return jsonify({
        'strategies': strategy_list,
        'count': len(strategy_list)
    }), 200


@bp.route('/<strategy_name>', methods=['GET'])
@jwt_required()
def get_strategy_details(strategy_name: str):
    """
    Get detailed information about a specific strategy
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: strategy_name
        required: true
        schema:
          type: string
    responses:
      200:
        description: Strategy details
      404:
        description: Strategy not found
    """
    if strategy_name not in STRATEGIES:
        return jsonify({'error': 'Strategy not found'}), 404
    
    strategy_class = STRATEGIES[strategy_name]
    
    try:
        # Get detailed strategy information
        strategy_details = {
            'name': strategy_name,
            'description': getattr(strategy_class, 'description', 'No description available'),
            'parameters': getattr(strategy_class, 'default_parameters', {}),
            'risk_level': getattr(strategy_class, 'risk_level', 'medium'),
            'timeframes': getattr(strategy_class, 'supported_timeframes', ['1h', '4h', '1d']),
            'markets': getattr(strategy_class, 'supported_markets', ['spot']),
            'indicators': getattr(strategy_class, 'indicators_used', []),
            'min_history_required': getattr(strategy_class, 'min_history_required', 100),
            'backtesting_compatible': getattr(strategy_class, 'backtesting_compatible', True),
            'paper_trading_compatible': getattr(strategy_class, 'paper_trading_compatible', True),
            'live_trading_compatible': getattr(strategy_class, 'live_trading_compatible', True)
        }
        
        return jsonify(strategy_details), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy details for {strategy_name}: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@bp.route('/<strategy_name>/validate', methods=['POST'])
@jwt_required()
def validate_strategy_config(strategy_name: str):
    """
    Validate strategy configuration
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: strategy_name
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              config:
                type: object
    responses:
      200:
        description: Configuration is valid
      400:
        description: Invalid configuration
    """
    if strategy_name not in STRATEGIES:
        return jsonify({'error': 'Strategy not found'}), 404
    
    data = request.json
    config = data.get('config', {})
    
    try:
        strategy_class = STRATEGIES[strategy_name]
        
        # Create strategy instance to validate config
        strategy = strategy_class(config)
        
        return jsonify({
            'valid': True,
            'message': 'Configuration is valid',
            'normalized_config': strategy.config
        }), 200
        
    except Exception as e:
        logger.error(f"Strategy config validation failed: {e}")
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 400


@bp.route('/<strategy_name>/backtest', methods=['POST'])
@require_trader
def backtest_strategy(strategy_name: str):
    """
    Run a backtest for a specific strategy
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: strategy_name
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
              - start_date
              - end_date
            properties:
              symbol:
                type: string
              start_date:
                type: string
                format: date
              end_date:
                type: string
                format: date
              timeframe:
                type: string
                default: 1h
              config:
                type: object
    responses:
      200:
        description: Backtest results
    """
    if strategy_name not in STRATEGIES:
        return jsonify({'error': 'Strategy not found'}), 404
    
    data = request.json
    
    try:
        from utils.validators import validate_trading_symbol
        
        symbol = validate_trading_symbol(data.get('symbol')).symbol
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        timeframe = data.get('timeframe', '1h')
        config = data.get('config', {})
        
        # Add strategy name to config
        config['strategy_name'] = strategy_name
        
        # Run backtest
        from core.trading_bot import TradingBot
        from config.settings import Settings
        from data_sources.data_manager import DataManager
        
        settings = Settings('default')
        data_manager = DataManager(settings)
        
        bot = TradingBot(
            mode='backtest',
            strategy_name=strategy_name,
            settings=settings,
            data_manager=data_manager
        )
        
        # Override strategy config if provided
        if config:
            bot.current_active_strategy.config.update(config)
        
        # Run backtest
        bot.run_backtest(symbol, timeframe, start_date, end_date)
        
        # Get results
        results = bot.performance_tracker.get_backtest_results()
        
        return jsonify({
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Backtest failed for strategy {strategy_name}: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500


@bp.route('/<strategy_name>/signal', methods=['POST'])
@require_trader
def get_strategy_signal(strategy_name: str):
    """
    Get current signal from a strategy
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: strategy_name
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
              timeframe:
                type: string
                default: 1h
              config:
                type: object
    responses:
      200:
        description: Strategy signal
    """
    if strategy_name not in STRATEGIES:
        return jsonify({'error': 'Strategy not found'}), 404
    
    data = request.json
    
    try:
        from utils.validators import validate_trading_symbol
        
        symbol = validate_trading_symbol(data.get('symbol')).symbol
        timeframe = data.get('timeframe', '1h')
        config = data.get('config', {})
        
        # Create strategy instance
        strategy_class = STRATEGIES[strategy_name]
        strategy = strategy_class(config)
        
        # Get market data
        from data_sources.data_manager import DataManager
        from config.settings import Settings
        
        settings = Settings('default')
        data_manager = DataManager(settings)
        
        # Get recent data
        end_date = data_manager.get_current_time()
        start_date = end_date - data_manager.get_timedelta(days=30)
        
        ohlcv_data = data_manager.get_historical_data(
            symbol, timeframe, start_date, end_date
        )
        
        if ohlcv_data.empty:
            return jsonify({'error': 'No market data available'}), 400
        
        # Get current price
        current_price = ohlcv_data['close'].iloc[-1]
        
        # Calculate signal
        signal, signal_data = strategy.calculate_signal(symbol, ohlcv_data, current_price)
        
        return jsonify({
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal,
            'signal_data': signal_data,
            'current_price': current_price,
            'timestamp': end_date.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Signal calculation failed for strategy {strategy_name}: {e}")
        return jsonify({'error': f'Signal calculation failed: {str(e)}'}), 500


@bp.route('/active', methods=['GET'])
@jwt_required()
def get_active_strategies():
    """
    Get currently active strategies
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    responses:
      200:
        description: List of active strategies
    """
    from api.routes.trading import trading_bot
    
    if not trading_bot:
        return jsonify({'active_strategies': []}), 200
    
    # Get active strategy info
    if hasattr(trading_bot, 'strategy_router'):
        active_strategies = trading_bot.strategy_router.get_active_strategies()
    else:
        # Single strategy mode
        active_strategies = {
            trading_bot.strategy_name: {
                'name': trading_bot.strategy_name,
                'active': trading_bot.running,
                'config': trading_bot.current_active_strategy.config if trading_bot.current_active_strategy else {}
            }
        }
    
    return jsonify({
        'active_strategies': active_strategies,
        'count': len(active_strategies)
    }), 200


@bp.route('/performance', methods=['GET'])
@jwt_required()
def get_strategy_performance():
    """
    Get performance metrics for strategies
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: strategy
        schema:
          type: string
      - in: query
        name: period
        schema:
          type: string
          enum: [day, week, month, all]
          default: all
    responses:
      200:
        description: Strategy performance metrics
    """
    strategy_name = request.args.get('strategy')
    period = request.args.get('period', 'all')
    
    from api.routes.trading import trading_bot
    
    if not trading_bot:
        return jsonify({'performance': {}}), 200
    
    # Get performance data
    if strategy_name:
        # Performance for specific strategy
        performance = trading_bot.performance_tracker.get_strategy_performance(
            strategy_name, period
        )
    else:
        # Performance for all strategies
        performance = trading_bot.performance_tracker.get_all_strategies_performance(period)
    
    return jsonify({
        'performance': performance,
        'period': period,
        'strategy': strategy_name
    }), 200


@bp.route('/optimize', methods=['POST'])
@require_admin
def optimize_strategy():
    """
    Optimize strategy parameters
    ---
    tags:
      - Strategies
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - strategy
              - symbol
              - start_date
              - end_date
              - parameters
            properties:
              strategy:
                type: string
              symbol:
                type: string
              start_date:
                type: string
                format: date
              end_date:
                type: string
                format: date
              parameters:
                type: object
                description: Parameter ranges for optimization
              optimization_target:
                type: string
                enum: [return, sharpe, win_rate]
                default: sharpe
    responses:
      200:
        description: Optimization results
    """
    data = request.json
    
    try:
        strategy_name = data.get('strategy')
        symbol = data.get('symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        parameters = data.get('parameters', {})
        optimization_target = data.get('optimization_target', 'sharpe')
        
        if strategy_name not in STRATEGIES:
            return jsonify({'error': 'Strategy not found'}), 404
        
        # Run parameter optimization
        from utils.strategy_optimizer import StrategyOptimizer
        
        optimizer = StrategyOptimizer(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            optimization_target=optimization_target
        )
        
        results = optimizer.optimize(parameters)
        
        return jsonify({
            'strategy': strategy_name,
            'symbol': symbol,
            'optimization_target': optimization_target,
            'best_parameters': results['best_parameters'],
            'best_score': results['best_score'],
            'optimization_history': results['history']
        }), 200
        
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500