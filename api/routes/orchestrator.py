"""
Orchestrator API Routes
======================

REST API endpoints for the self-discovering strategy orchestrator.
Provides real-time data for dashboard integration.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List
import json

# Import orchestrator components
from core.strategy_orchestrator import StrategyDiscoveryEngine
from core.intelligent_orchestration_engine import IntelligentOrchestrationEngine
from core.strategy_health_monitor import HealthAndABTestingSystem
from core.orchestrator_portfolio_manager import PortfolioManager, TradingMode

logger = logging.getLogger(__name__)

bp = Blueprint('orchestrator', __name__)

# Global instances (initialized on first request)
orchestrator_instance = None
orchestration_engine = None
health_system = None
portfolio_manager = None

def get_orchestrator_instances():
    """Get or create orchestrator instances"""
    global orchestrator_instance, orchestration_engine, health_system, portfolio_manager
    
    if orchestrator_instance is None:
        orchestrator_instance = StrategyDiscoveryEngine()
        orchestration_engine = IntelligentOrchestrationEngine()
        health_system = HealthAndABTestingSystem()
        
        # Get trading mode from config or environment
        import os
        trading_mode = os.environ.get('ORCHESTRATOR_MODE', 'paper')
        initial_capital = float(os.environ.get('ORCHESTRATOR_CAPITAL', '10000'))
        
        portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            mode=TradingMode(trading_mode)
        )
        
    return orchestrator_instance, orchestration_engine, health_system, portfolio_manager

@bp.route('/status', methods=['GET'])
@jwt_required()
def get_orchestrator_status():
    """Get current orchestrator status"""
    try:
        orchestrator, engine, health, portfolio = get_orchestrator_instances()
        
        # Get discovered strategies
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        discovered_strategies = loop.run_until_complete(orchestrator.discover_all_strategies())
        
        # Get portfolio state
        portfolio_state = loop.run_until_complete(portfolio.get_portfolio_state())
        
        # Get health system status
        health_status = health.get_system_status()
        
        return jsonify({
            'status': 'active',
            'mode': portfolio.mode.value,
            'discovered_strategies': len(discovered_strategies),
            'portfolio': {
                'total_value': portfolio_state.total_value,
                'cash_balance': portfolio_state.cash_balance,
                'positions_value': portfolio_state.positions_value,
                'total_pnl': portfolio_state.total_pnl,
                'total_pnl_percent': (portfolio_state.total_pnl / portfolio.initial_capital) * 100,
                'win_rate': portfolio_state.win_rate,
                'sharpe_ratio': portfolio_state.sharpe_ratio,
                'max_drawdown': portfolio_state.max_drawdown,
                'total_positions': portfolio_state.total_positions
            },
            'health_monitoring': health_status['health_monitoring'],
            'ab_testing': health_status['ab_testing']
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/strategies', methods=['GET'])
@jwt_required()
def get_discovered_strategies():
    """Get all discovered strategies with their DNA profiles"""
    try:
        orchestrator, _, _, _ = get_orchestrator_instances()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        discovered = loop.run_until_complete(orchestrator.discover_all_strategies())
        
        strategies = []
        for name, info in discovered.items():
            dna = info['dna']
            strategies.append({
                'name': name,
                'risk_level': dna.risk_level,
                'timeframe': dna.timeframe,
                'signal_sources': dna.signal_sources,
                'expected_win_rate': dna.expected_win_rate,
                'sharpe_estimate': dna.sharpe_estimate,
                'cooperation_score': dna.cooperation_score,
                'conflict_strategies': dna.conflict_strategies,
                'code_metrics': dna.code_metrics
            })
        
        return jsonify({
            'strategies': strategies,
            'total': len(strategies)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting discovered strategies: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/portfolio', methods=['GET'])
@jwt_required()
def get_portfolio_details():
    """Get detailed portfolio information"""
    try:
        _, _, _, portfolio = get_orchestrator_instances()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        portfolio_state = loop.run_until_complete(portfolio.get_portfolio_state())
        
        # Get performance summary
        performance = portfolio.get_performance_summary()
        
        # Get open positions
        positions = []
        for pos_id, pos in portfolio.positions.items():
            positions.append({
                'id': pos_id,
                'symbol': pos.symbol,
                'strategy': pos.strategy,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'position_value': pos.position_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_percent': pos.pnl_percent,
                'entry_time': pos.entry_time.isoformat(),
                'is_paper': pos.is_paper
            })
        
        # Get recent trades
        recent_trades = []
        for trade in portfolio.trade_history[-20:]:  # Last 20 trades
            recent_trades.append({
                'timestamp': trade['timestamp'].isoformat(),
                'action': trade['action'],
                'strategy': trade['strategy'],
                'symbol': trade['symbol'],
                'side': trade.get('side', 'N/A'),
                'quantity': trade.get('quantity', 0),
                'price': trade.get('price', 0),
                'pnl': trade.get('pnl', 0),
                'is_paper': trade.get('is_paper', True)
            })
        
        return jsonify({
            'mode': portfolio.mode.value,
            'performance': performance,
            'positions': positions,
            'recent_trades': recent_trades,
            'allocations': {
                'by_strategy': portfolio_state.strategy_allocations,
                'by_symbol': portfolio_state.symbol_allocations
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting portfolio details: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/market-analysis', methods=['GET'])
@jwt_required()
def get_market_analysis():
    """Get current market analysis from orchestrator"""
    try:
        _, engine, _, _ = get_orchestrator_instances()
        
        # Get latest market regime
        # In production, this would use real market data
        regime = engine.market_regime_detector.current_regime if hasattr(engine, 'market_regime_detector') else 'unknown'
        
        return jsonify({
            'market_regime': regime,
            'analysis_time': datetime.now().isoformat(),
            'regime_confidence': 0.85,  # Placeholder
            'volatility_level': 'moderate',  # Placeholder
            'trend_direction': 'neutral'  # Placeholder
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting market analysis: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/strategy-allocation', methods=['GET'])
@jwt_required()
def get_strategy_allocation():
    """Get current strategy allocation recommendations"""
    try:
        _, engine, _, portfolio = get_orchestrator_instances()
        
        # This would use real market data in production
        dummy_decision = {
            'strategy_weights': {
                'momentum_strategy': 0.3,
                'mean_reversion': 0.2,
                'arbitrage': 0.1,
                'defensive': 0.4
            },
            'market_regime': 'neutral',
            'risk_level': 'moderate'
        }
        
        return jsonify({
            'allocations': dummy_decision['strategy_weights'],
            'market_regime': dummy_decision['market_regime'],
            'risk_level': dummy_decision['risk_level'],
            'last_update': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy allocation: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/health-metrics/<strategy_name>', methods=['GET'])
@jwt_required()
def get_strategy_health(strategy_name):
    """Get health metrics for a specific strategy"""
    try:
        _, _, health, _ = get_orchestrator_instances()
        
        # Get strategy history if available
        if strategy_name in health.health_monitor.strategy_histories:
            history = health.health_monitor.strategy_histories[strategy_name]
            if history:
                latest = history[-1]
                
                return jsonify({
                    'strategy': strategy_name,
                    'health_score': latest.overall_health_score,
                    'performance_score': latest.performance_score,
                    'risk_score': latest.risk_score,
                    'stability_score': latest.stability_score,
                    'metrics': {
                        'win_rate': latest.win_rate,
                        'sharpe_ratio': latest.sharpe_ratio,
                        'max_drawdown': latest.max_drawdown,
                        'trade_frequency': latest.trade_frequency,
                        'error_rate': latest.error_rate
                    },
                    'alerts': latest.alerts,
                    'last_update': latest.timestamp.isoformat()
                }), 200
        
        return jsonify({
            'strategy': strategy_name,
            'health_score': 0.0,
            'message': 'No health data available yet'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting strategy health: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/ab-tests', methods=['GET'])
@jwt_required()
def get_ab_tests():
    """Get active and completed A/B tests"""
    try:
        _, _, health, _ = get_orchestrator_instances()
        
        ab_summary = health.ab_test_manager.get_active_tests_summary()
        
        return jsonify(ab_summary), 200
        
    except Exception as e:
        logger.error(f"Error getting A/B tests: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/performance-history', methods=['GET'])
@jwt_required()
def get_performance_history():
    """Get portfolio performance history for charts"""
    try:
        _, _, _, portfolio = get_orchestrator_instances()
        
        # Get time range from query params
        hours = int(request.args.get('hours', 24))
        
        # Get portfolio history
        history = []
        for state in portfolio.portfolio_history[-hours:]:
            history.append({
                'timestamp': state.timestamp.isoformat(),
                'total_value': state.total_value,
                'total_pnl': state.total_pnl,
                'positions_value': state.positions_value,
                'cash_balance': state.cash_balance,
                'win_rate': state.win_rate,
                'positions_count': state.total_positions
            })
        
        return jsonify({
            'history': history,
            'period_hours': hours
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/switch-mode', methods=['POST'])
@jwt_required()
def switch_trading_mode():
    """Switch between paper/live/hybrid trading modes"""
    try:
        data = request.get_json()
        new_mode = data.get('mode', '').lower()
        
        if new_mode not in ['paper', 'live', 'hybrid']:
            return jsonify({'error': 'Invalid mode. Use: paper, live, or hybrid'}), 400
        
        _, _, _, portfolio = get_orchestrator_instances()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            portfolio.switch_mode(TradingMode(new_mode), transfer_positions=data.get('transfer_positions', False))
        )
        
        return jsonify({
            'success': True,
            'new_mode': new_mode,
            'message': f'Successfully switched to {new_mode} mode'
        }), 200
        
    except Exception as e:
        logger.error(f"Error switching mode: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events for real-time updates
def emit_orchestrator_update(socketio, data):
    """Emit orchestrator update via WebSocket"""
    socketio.emit('orchestrator_update', data, namespace='/ws')

def emit_portfolio_update(socketio, data):
    """Emit portfolio update via WebSocket"""
    socketio.emit('portfolio_update', data, namespace='/ws')

def emit_health_alert(socketio, data):
    """Emit health alert via WebSocket"""
    socketio.emit('health_alert', data, namespace='/ws')