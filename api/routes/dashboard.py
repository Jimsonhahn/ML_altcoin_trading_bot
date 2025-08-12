"""
Dashboard API Routes
===================

REST API endpoints for the Janics Freedom Factory dashboard.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
import logging

from api.controllers import (
    DashboardStatusController,
    JanicsBotController,
    TradesController,
    PortfolioController,
    BotIntelligenceController,
    StrategySupermixController,
    AIAnalyticsController
)

logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('dashboard', __name__)

# Initialize controllers
status_controller = DashboardStatusController()
bot_controller = JanicsBotController()
trades_controller = TradesController()
portfolio_controller = PortfolioController()
bot_intelligence_controller = BotIntelligenceController()
strategy_controller = StrategySupermixController()
ai_analytics_controller = AIAnalyticsController()


# ====================================
# HEADER STATUS ENDPOINTS
# ====================================

@bp.route('/status/header', methods=['GET'])
def get_header_status():
    """Get header status indicators"""
    try:
        status = status_controller.get_header_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting header status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/status/system', methods=['GET'])
def get_system_metrics():
    """Get detailed system metrics"""
    try:
        metrics = status_controller.get_system_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# BOT CONTROL ENDPOINTS
# ====================================

@bp.route('/bot/start', methods=['POST'])
@jwt_required(optional=True)  # Optional JWT for development
def start_bot():
    """Start the trading bot"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'live')
        strategy = data.get('strategy')
        profile = data.get('profile')
        
        result = bot_controller.start_bot(mode=mode, strategy=strategy, profile=profile)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/bot/stop', methods=['POST'])
@jwt_required(optional=True)
def stop_bot():
    """Stop the trading bot"""
    try:
        result = bot_controller.stop_bot()
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/bot/restart', methods=['POST'])
@jwt_required(optional=True)
def restart_bot():
    """Restart the trading bot"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'live')
        strategy = data.get('strategy')
        profile = data.get('profile')
        
        result = bot_controller.restart_bot(mode=mode, strategy=strategy, profile=profile)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error restarting bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/bot/status', methods=['GET'])
def get_bot_status():
    """Get current bot process status"""
    try:
        status = bot_controller.get_bot_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/bot/logs', methods=['GET'])
def get_bot_logs():
    """Get bot logs"""
    try:
        lines = request.args.get('lines', 50, type=int)
        logs = bot_controller.get_bot_logs(lines=lines)
        return jsonify({'logs': logs}), 200
    except Exception as e:
        logger.error(f"Error getting bot logs: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# TRADES ENDPOINTS
# ====================================

@bp.route('/trades/active', methods=['GET'])
def get_active_trades():
    """Get active trades for the dashboard"""
    try:
        trades = trades_controller.get_active_trades()
        return jsonify(trades), 200
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/trades/history', methods=['GET'])
def get_trade_history():
    """Get trade history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = trades_controller.get_trade_history(limit=limit)
        return jsonify({'history': history}), 200
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/trades/<trade_id>/close', methods=['POST'])
@jwt_required(optional=True)
def close_trade(trade_id):
    """Close a specific trade"""
    try:
        result = trades_controller.close_trade(trade_id)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ====================================
# PORTFOLIO ENDPOINTS
# ====================================

@bp.route('/portfolio/wealth', methods=['GET'])
def get_wealth_data():
    """Get wealth accumulator data"""
    try:
        wealth = portfolio_controller.get_wealth_data()
        return jsonify(wealth), 200
    except Exception as e:
        logger.error(f"Error getting wealth data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/portfolio/breakdown', methods=['GET'])
def get_portfolio_breakdown():
    """Get detailed portfolio breakdown"""
    try:
        breakdown = portfolio_controller.get_portfolio_breakdown()
        return jsonify(breakdown), 200
    except Exception as e:
        logger.error(f"Error getting portfolio breakdown: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# BOT INTELLIGENCE ENDPOINTS
# ====================================

@bp.route('/bot/intelligence', methods=['GET'])
def get_bot_intelligence():
    """Get bot intelligence status"""
    try:
        intelligence = bot_intelligence_controller.get_bot_status()
        return jsonify(intelligence), 200
    except Exception as e:
        logger.error(f"Error getting bot intelligence: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/bot/learning/history', methods=['GET'])
def get_learning_history():
    """Get bot learning history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = bot_intelligence_controller.get_learning_history(limit=limit)
        return jsonify({'history': history}), 200
    except Exception as e:
        logger.error(f"Error getting learning history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/bot/learning/trigger', methods=['POST'])
@jwt_required(optional=True)
def trigger_learning():
    """Trigger a learning cycle"""
    try:
        result = bot_intelligence_controller.trigger_learning_cycle()
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error triggering learning: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ====================================
# STRATEGY SUPERMIX ENDPOINTS
# ====================================

@bp.route('/strategies/supermix', methods=['GET'])
def get_strategy_supermix():
    """Get risk-tiered strategy supermix status"""
    try:
        supermix = strategy_controller.get_strategy_supermix_status()
        return jsonify(supermix), 200
    except Exception as e:
        logger.error(f"Error getting strategy supermix: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/strategies/tier/<risk_level>/start', methods=['POST'])
@jwt_required(optional=True)
def start_risk_tier(risk_level):
    """Start strategies in a risk tier"""
    try:
        result = strategy_controller.start_risk_tier(risk_level)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error starting risk tier: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/strategies/tier/<risk_level>/stop', methods=['POST'])
@jwt_required(optional=True)
def stop_risk_tier(risk_level):
    """Stop strategies in a risk tier"""
    try:
        result = strategy_controller.stop_risk_tier(risk_level)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error stopping risk tier: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/strategies/tier/<risk_level>/allocation', methods=['PUT'])
@jwt_required(optional=True)
def adjust_tier_allocation(risk_level):
    """Adjust allocation for a risk tier"""
    try:
        data = request.get_json()
        allocation = data.get('allocation')
        
        if allocation is None:
            return jsonify({'success': False, 'message': 'Allocation value required'}), 400
        
        result = strategy_controller.adjust_allocation(risk_level, allocation)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        logger.error(f"Error adjusting allocation: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@bp.route('/strategies/<strategy_name>/details', methods=['GET'])
def get_strategy_details(strategy_name):
    """Get details for a specific strategy"""
    try:
        details = strategy_controller.get_strategy_details(strategy_name)
        return jsonify(details), 200
    except Exception as e:
        logger.error(f"Error getting strategy details: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# AI ANALYTICS ENDPOINTS
# ====================================

@bp.route('/ai/insights', methods=['GET'])
def get_ai_insights():
    """Get AI performance insights"""
    try:
        insights = ai_analytics_controller.get_performance_insights()
        return jsonify(insights), 200
    except Exception as e:
        logger.error(f"Error getting AI insights: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/ai/market-intelligence', methods=['GET'])
def get_market_intelligence():
    """Get market intelligence data"""
    try:
        intelligence = ai_analytics_controller.get_market_intelligence()
        return jsonify(intelligence), 200
    except Exception as e:
        logger.error(f"Error getting market intelligence: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/ai/sentiment', methods=['GET'])
def get_sentiment_intelligence():
    """Get market sentiment intelligence"""
    try:
        sentiment = ai_analytics_controller.get_sentiment_intelligence()
        return jsonify(sentiment), 200
    except Exception as e:
        logger.error(f"Error getting sentiment intelligence: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# DASHBOARD SUMMARY ENDPOINT
# ====================================

@bp.route('/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """Get complete dashboard data in one call"""
    try:
        summary = {
            'header_status': status_controller.get_header_status(),
            'active_trades': trades_controller.get_active_trades(),
            'wealth_data': portfolio_controller.get_wealth_data(),
            'bot_intelligence': bot_intelligence_controller.get_bot_status(),
            'strategy_supermix': strategy_controller.get_strategy_supermix_status(),
            'ai_insights': ai_analytics_controller.get_performance_insights()
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ====================================
# HEALTH CHECK
# ====================================

@bp.route('/health', methods=['GET'])
def dashboard_health():
    """Dashboard API health check"""
    return jsonify({
        'service': 'dashboard-api',
        'status': 'healthy',
        'timestamp': status_controller.get_header_status()['last_update']
    }), 200