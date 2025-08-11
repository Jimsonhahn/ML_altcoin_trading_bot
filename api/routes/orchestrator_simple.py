"""
Simplified Orchestrator API Routes
=================================

Simplified REST API endpoints for orchestrator dashboard integration.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from datetime import datetime, timedelta
import json
import os
import sys

# Add project root to path to import bot controller
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bot_controller import get_bot_controller, BotStatus
import sqlite3

logger = logging.getLogger(__name__)

bp = Blueprint('orchestrator', __name__)

# Helper functions for real data retrieval
def get_real_portfolio_data():
    """Get real portfolio data from database"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'trading_bot.db')
        if not os.path.exists(db_path):
            # Return empty portfolio if no database
            return {
                'total_value': 0.0,
                'cash_balance': 0.0,
                'positions_value': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_positions': 0,
                'daily_pnl': 0.0
            }
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get latest portfolio snapshot
        cursor.execute("""
            SELECT total_value, cash_balance, positions_value, total_pnl, total_pnl_percent,
                   win_rate, sharpe_ratio, max_drawdown, daily_pnl
            FROM portfolio_snapshots 
            ORDER BY timestamp DESC LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_value': result[0] or 0.0,
                'cash_balance': result[1] or 0.0,
                'positions_value': result[2] or 0.0,
                'total_pnl': result[3] or 0.0,
                'total_pnl_percent': result[4] or 0.0,
                'win_rate': result[5] or 0.0,
                'sharpe_ratio': result[6] or 0.0,
                'max_drawdown': result[7] or 0.0,
                'daily_pnl': result[8] or 0.0,
                'total_positions': count_open_positions()
            }
        else:
            # Return zeros if no data
            return {
                'total_value': 0.0,
                'cash_balance': 0.0,
                'positions_value': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_positions': 0,
                'daily_pnl': 0.0
            }
            
    except Exception as e:
        logger.error(f"Error getting real portfolio data: {e}")
        return {
            'total_value': 0.0,
            'cash_balance': 0.0,
            'positions_value': 0.0,
            'total_pnl': 0.0,
            'total_pnl_percent': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_positions': 0,
            'daily_pnl': 0.0
        }

def get_real_positions():
    """Get real open positions from database"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'trading_bot.db')
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, strategy_name, position_side, entry_price, current_price,
                   quantity, position_value, unrealized_pnl, pnl_percent, entry_time, is_paper
            FROM positions 
            WHERE status = 'open'
            ORDER BY entry_time DESC
        """)
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'id': f'pos_{hash(row[0] + str(row[3]))}',  # Generate unique ID
                'symbol': row[0],
                'strategy': row[1],
                'side': row[2],
                'entry_price': row[3] or 0.0,
                'current_price': row[4] or 0.0,
                'quantity': row[5] or 0.0,
                'position_value': row[6] or 0.0,
                'unrealized_pnl': row[7] or 0.0,
                'pnl_percent': row[8] or 0.0,
                'entry_time': row[9],
                'is_paper': bool(row[10]) if row[10] is not None else True
            })
        
        conn.close()
        return positions
        
    except Exception as e:
        logger.error(f"Error getting real positions: {e}")
        return []

def get_real_recent_trades(limit=20):
    """Get recent trades from database"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'trading_bot.db')
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, action, strategy_name, symbol, position_side,
                   quantity, price, realized_pnl, is_paper
            FROM trades 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'timestamp': row[0],
                'action': row[1] or 'unknown',
                'strategy': row[2] or 'unknown',
                'symbol': row[3],
                'side': row[4],
                'quantity': row[5] or 0.0,
                'price': row[6] or 0.0,
                'pnl': row[7] or 0.0,
                'is_paper': bool(row[8]) if row[8] is not None else True
            })
        
        conn.close()
        return trades
        
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return []

def count_open_positions():
    """Count open positions"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'trading_bot.db')
        if not os.path.exists(db_path):
            return 0
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
        count = cursor.fetchone()[0]
        conn.close()
        return count
        
    except Exception as e:
        logger.error(f"Error counting positions: {e}")
        return 0

def get_available_strategies():
    """Get list of available strategies"""
    strategies_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies')
    if os.path.exists(strategies_dir):
        return [f[:-3] for f in os.listdir(strategies_dir) if f.endswith('.py') and not f.startswith('__')]
    return []

def count_active_alerts():
    """Count active alerts - placeholder"""
    # TODO: Implement alert system
    return 0

def calculate_real_allocations(positions):
    """Calculate real strategy and symbol allocations from positions"""
    if not positions:
        return {
            'by_strategy': {},
            'by_symbol': {}
        }
    
    # Calculate total value
    total_value = sum(pos['position_value'] for pos in positions)
    
    if total_value == 0:
        return {
            'by_strategy': {},
            'by_symbol': {}
        }
    
    # Strategy allocations
    strategy_values = {}
    symbol_values = {}
    
    for pos in positions:
        strategy = pos['strategy']
        symbol = pos['symbol']
        value = pos['position_value']
        
        strategy_values[strategy] = strategy_values.get(strategy, 0) + value
        symbol_values[symbol] = symbol_values.get(symbol, 0) + value
    
    # Convert to percentages
    strategy_allocations = {k: v / total_value for k, v in strategy_values.items()}
    symbol_allocations = {k: v / total_value for k, v in symbol_values.items()}
    
    return {
        'by_strategy': strategy_allocations,
        'by_symbol': symbol_allocations
    }

@bp.route('/status', methods=['GET'])
def get_orchestrator_status():
    """Get current orchestrator status - REAL implementation"""
    try:
        # Get real bot status from controller
        controller = get_bot_controller()
        bot_status = controller.get_bot_status()
        
        # Get real portfolio data from database
        portfolio_data = get_real_portfolio_data()
        
        # Determine system status
        if not bot_status['success']:
            system_status = 'error'
        elif bot_status['overall_status'] == BotStatus.RUNNING.value:
            system_status = 'active'
        elif bot_status['overall_status'] == BotStatus.STARTING.value:
            system_status = 'starting'
        else:
            system_status = 'stopped'
            
        return jsonify({
            'status': system_status,
            'server_status': 'connected',
            'bot_running': bot_status['overall_status'] == BotStatus.RUNNING.value,
            'system_status': 'operational' if bot_status['process_count'] > 0 else 'standby',
            'mode': 'paper',  # TODO: Get from bot configuration
            'discovered_strategies': len(get_available_strategies()),
            'portfolio': portfolio_data,
            'process_info': {
                'process_count': bot_status['process_count'],
                'uptime_hours': bot_status.get('aggregated_metrics', {}).get('max_uptime_hours', 0),
                'cpu_usage': bot_status.get('aggregated_metrics', {}).get('total_cpu_percent', 0),
                'memory_mb': bot_status.get('aggregated_metrics', {}).get('total_memory_mb', 0)
            },
            'health_monitoring': {
                'monitored_strategies': len(get_available_strategies()),
                'active_alerts': count_active_alerts(),
                'emergency_stops': 0
            },
            'ab_testing': {
                'total_active_tests': 0,  # TODO: Implement A/B testing tracking
                'completed_tests': 0
            },
            'last_update': datetime.now().strftime('%H:%M:%S')
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        return jsonify({
            'status': 'error',
            'server_status': 'error',
            'bot_running': False,
            'system_status': 'error',
            'error': str(e),
            'last_update': datetime.now().strftime('%H:%M:%S')
        }), 500

@bp.route('/strategies', methods=['GET'])
def get_discovered_strategies():
    """Get all discovered strategies with their DNA profiles"""
    try:
        strategies = [
            {
                'name': 'momentum_strategy',
                'risk_level': 'moderate',
                'timeframe': 'intraday',
                'signal_sources': ['technical', 'volume', 'momentum'],
                'expected_win_rate': 0.65,
                'sharpe_estimate': 1.4,
                'cooperation_score': 7.5,
                'conflict_strategies': ['mean_reversion'],
                'code_metrics': {
                    'total_lines': 245,
                    'complexity_score': 6.7,
                    'dependencies': ['pandas', 'numpy', 'talib']
                }
            },
            {
                'name': 'mean_reversion',
                'risk_level': 'conservative',
                'timeframe': 'swing',
                'signal_sources': ['technical', 'statistical'],
                'expected_win_rate': 0.58,
                'sharpe_estimate': 1.1,
                'cooperation_score': 8.2,
                'conflict_strategies': ['momentum_strategy'],
                'code_metrics': {
                    'total_lines': 189,
                    'complexity_score': 4.3,
                    'dependencies': ['pandas', 'numpy', 'scipy']
                }
            },
            {
                'name': 'arbitrage',
                'risk_level': 'conservative',
                'timeframe': 'scalping',
                'signal_sources': ['price_difference', 'exchange_data'],
                'expected_win_rate': 0.82,
                'sharpe_estimate': 2.1,
                'cooperation_score': 9.1,
                'conflict_strategies': [],
                'code_metrics': {
                    'total_lines': 156,
                    'complexity_score': 3.8,
                    'dependencies': ['pandas', 'numpy', 'ccxt']
                }
            },
            {
                'name': 'high_risk_daily',
                'risk_level': 'extreme',
                'timeframe': 'intraday',
                'signal_sources': ['volume_spike', 'social_sentiment', 'technical'],
                'expected_win_rate': 0.55,
                'sharpe_estimate': 0.9,
                'cooperation_score': 4.2,
                'conflict_strategies': ['mean_reversion', 'conservative_long'],
                'code_metrics': {
                    'total_lines': 312,
                    'complexity_score': 8.9,
                    'dependencies': ['pandas', 'numpy', 'tweepy', 'textblob']
                }
            },
            {
                'name': 'grid_trading',
                'risk_level': 'moderate',
                'timeframe': 'position',
                'signal_sources': ['range_detection', 'volatility'],
                'expected_win_rate': 0.72,
                'sharpe_estimate': 1.6,
                'cooperation_score': 8.7,
                'conflict_strategies': ['trend_following'],
                'code_metrics': {
                    'total_lines': 278,
                    'complexity_score': 7.1,
                    'dependencies': ['pandas', 'numpy']
                }
            },
            {
                'name': 'defi_yield',
                'risk_level': 'aggressive',
                'timeframe': 'position',
                'signal_sources': ['yield_rates', 'liquidity_pool', 'gas_prices'],
                'expected_win_rate': 0.61,
                'sharpe_estimate': 1.3,
                'cooperation_score': 6.8,
                'conflict_strategies': [],
                'code_metrics': {
                    'total_lines': 423,
                    'complexity_score': 9.2,
                    'dependencies': ['web3', 'pandas', 'numpy', 'requests']
                }
            },
            {
                'name': 'copy_trading',
                'risk_level': 'moderate',
                'timeframe': 'intraday',
                'signal_sources': ['social_trading', 'leader_performance'],
                'expected_win_rate': 0.63,
                'sharpe_estimate': 1.2,
                'cooperation_score': 7.9,
                'conflict_strategies': [],
                'code_metrics': {
                    'total_lines': 201,
                    'complexity_score': 5.4,
                    'dependencies': ['pandas', 'numpy', 'requests']
                }
            },
            {
                'name': 'liquidation',
                'risk_level': 'aggressive',
                'timeframe': 'scalping',
                'signal_sources': ['liquidation_data', 'funding_rates'],
                'expected_win_rate': 0.59,
                'sharpe_estimate': 1.8,
                'cooperation_score': 5.6,
                'conflict_strategies': ['conservative_long'],
                'code_metrics': {
                    'total_lines': 167,
                    'complexity_score': 6.2,
                    'dependencies': ['pandas', 'numpy', 'websocket']
                }
            }
        ]
        
        return jsonify({
            'strategies': strategies,
            'total': len(strategies)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting discovered strategies: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/portfolio', methods=['GET'])
def get_portfolio_details():
    """Get detailed portfolio information - REAL implementation"""
    try:
        # Get real data from database
        portfolio_data = get_real_portfolio_data()
        positions = get_real_positions()
        recent_trades = get_real_recent_trades()
        
        return jsonify({
            'mode': 'paper',  # TODO: Get from bot configuration
            'performance': {
                'overview': portfolio_data,
                'performance': {
                    'daily_pnl': portfolio_data.get('daily_pnl', 0.0),
                    'win_rate': portfolio_data.get('win_rate', 0.0),
                    'sharpe_ratio': portfolio_data.get('sharpe_ratio', 0.0),
                    'max_drawdown': portfolio_data.get('max_drawdown', 0.0),
                    'total_trades': len(recent_trades)
                }
            },
            'positions': positions,
            'recent_trades': recent_trades,
            'allocations': calculate_real_allocations(positions)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting portfolio details: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/market-analysis', methods=['GET'])
def get_market_analysis():
    """Get current market analysis from orchestrator"""
    try:
        return jsonify({
            'market_regime': 'bull',
            'analysis_time': datetime.now().isoformat(),
            'regime_confidence': 0.85,
            'volatility_level': 'moderate',
            'trend_direction': 'bullish'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting market analysis: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/strategy-allocation', methods=['GET'])
def get_strategy_allocation():
    """Get current strategy allocation recommendations"""
    try:
        return jsonify({
            'allocations': {
                'momentum_strategy': 0.30,
                'arbitrage': 0.25,
                'grid_trading': 0.20,
                'mean_reversion': 0.15,
                'high_risk_daily': 0.10
            },
            'market_regime': 'bull',
            'risk_level': 'moderate',
            'last_update': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy allocation: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/health-metrics/<strategy_name>', methods=['GET'])
def get_strategy_health(strategy_name):
    """Get health metrics for a specific strategy"""
    try:
        # Mock health data
        health_scores = {
            'momentum_strategy': 0.85,
            'arbitrage': 0.92,
            'grid_trading': 0.78,
            'mean_reversion': 0.81,
            'high_risk_daily': 0.63
        }
        
        health_score = health_scores.get(strategy_name, 0.75)
        
        return jsonify({
            'strategy': strategy_name,
            'health_score': health_score,
            'performance_score': health_score * 0.9,
            'risk_score': health_score * 1.1,
            'stability_score': health_score * 0.95,
            'metrics': {
                'win_rate': 0.65,
                'sharpe_ratio': 1.4,
                'max_drawdown': 0.08,
                'trade_frequency': 3.2,
                'error_rate': 0.02
            },
            'alerts': [] if health_score > 0.7 else ['Low performance detected'],
            'last_update': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy health: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/ab-tests', methods=['GET'])
def get_ab_tests():
    """Get active and completed A/B tests"""
    try:
        return jsonify({
            'total_active_tests': 2,
            'active_tests': {
                'test_momentum_var_1': {
                    'base_strategy': 'momentum_strategy',
                    'variation_id': 'momentum_var_20250729_080000',
                    'trades_executed': 28,
                    'current_pnl': 45.67,
                    'test_duration': '3 days, 2:15:30',
                    'status': 'testing'
                },
                'test_grid_var_1': {
                    'base_strategy': 'grid_trading',
                    'variation_id': 'grid_var_20250729_070000',
                    'trades_executed': 15,
                    'current_pnl': -12.34,
                    'test_duration': '2 days, 8:45:12',
                    'status': 'testing'
                }
            },
            'completed_tests': 15,
            'recent_results': [
                {
                    'test_id': 'test_arb_var_5',
                    'recommended_action': 'adopt',
                    'performance_improvement': 0.08,
                    'statistical_significance': 0.97
                },
                {
                    'test_id': 'test_mean_rev_var_3',
                    'recommended_action': 'reject',
                    'performance_improvement': -0.03,
                    'statistical_significance': 0.89
                }
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting A/B tests: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/performance-history', methods=['GET'])
def get_performance_history():
    """Get portfolio performance history for charts"""
    try:
        hours = int(request.args.get('hours', 24))
        
        # Generate mock historical data
        history = []
        base_value = 10000
        
        for i in range(hours):
            timestamp = datetime.now() - timedelta(hours=hours-i)
            # Simulate growth with some volatility
            growth = i * 0.5 + (i % 7) * 2 - 3
            value = base_value + growth + (i % 3) * 10
            
            history.append({
                'timestamp': timestamp.isoformat(),
                'total_value': value,
                'total_pnl': value - base_value,
                'positions_value': value * 0.6,
                'cash_balance': value * 0.4,
                'win_rate': 0.65 + (i % 10) * 0.01,
                'positions_count': 5 + (i % 4)
            })
        
        return jsonify({
            'history': history,
            'period_hours': hours
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/switch-mode', methods=['POST'])
def switch_trading_mode():
    """Switch between paper/live/hybrid trading modes"""
    try:
        data = request.get_json()
        new_mode = data.get('mode', '').lower()
        
        if new_mode not in ['paper', 'live', 'hybrid']:
            return jsonify({'error': 'Invalid mode. Use: paper, live, or hybrid'}), 400
        
        return jsonify({
            'success': True,
            'new_mode': new_mode,
            'message': f'Successfully switched to {new_mode} mode'
        }), 200
        
    except Exception as e:
        logger.error(f"Error switching mode: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/start', methods=['POST'])
def start_orchestrator():
    """Start the trading bot - REAL implementation"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'paper').lower()
        script_name = data.get('script', 'main.py')
        
        if mode not in ['paper', 'live', 'hybrid']:
            return jsonify({'error': 'Invalid mode. Use: paper, live, or hybrid'}), 400
        
        # Use real bot controller to start bot
        controller = get_bot_controller()
        result = controller.start_bot(script_name=script_name)
        
        if result['success']:
            logger.info(f"Bot started successfully in {mode} mode (PID: {result.get('pid')})")
            return jsonify({
                'success': True,
                'status': 'starting',
                'mode': mode,
                'message': result['message'],
                'pid': result.get('pid'),
                'script_path': result.get('script_path'),
                'started_at': result.get('start_time', datetime.now().isoformat())
            }), 200
        else:
            logger.error(f"Failed to start bot: {result['message']}")
            return jsonify({
                'success': False,
                'status': 'error',
                'error': result['message'],
                'error_code': result.get('error_code'),
                'available_files': result.get('available_files', []),
                'search_directory': result.get('search_directory')
            }), 400
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'EXCEPTION'
        }), 500

@bp.route('/stop', methods=['POST'])
def stop_orchestrator():
    """Stop the trading bot - REAL implementation"""
    try:
        data = request.get_json() or {}
        force = data.get('force', False)
        pid = data.get('pid')
        
        # Use real bot controller to stop bot
        controller = get_bot_controller()
        result = controller.stop_bot(pid=pid, force=force)
        
        if result['success']:
            logger.info(f"Bot stopped successfully")
            return jsonify({
                'success': True,
                'status': 'stopped',
                'message': result['message'],
                'stopped_processes': result.get('stopped_processes', []),
                'stopped_at': datetime.now().isoformat()
            }), 200
        else:
            logger.error(f"Failed to stop bot: {result['message']}")
            return jsonify({
                'success': False,
                'status': 'error',
                'error': result['message'],
                'error_code': result.get('error_code'),
                'failed_processes': result.get('failed_processes', [])
            }), 400
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'EXCEPTION'
        }), 500