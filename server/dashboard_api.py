"""
Dashboard API - Flask API für Mobile Dashboard Control
=====================================================

Bietet REST API Endpoints für das Mobile Dashboard.
"""

from flask import Flask, jsonify, request, render_template_string, send_file
from flask_cors import CORS
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging BEFORE using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from server.bot_wrapper import ServerBotWrapper
    server_bot = ServerBotWrapper()  # Erstelle Instanz hier statt beim Import
except Exception as e:
    logger.warning(f"Bot wrapper could not be imported: {e}")
    server_bot = None

try:
    from orchestrator import StrategyOrchestrator
except Exception as e:
    logger.warning(f"Strategy orchestrator could not be imported: {e}")
    StrategyOrchestrator = None

# Logger is already configured above

# Create Flask app
app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for mobile access

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'trading-bot-api'})

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Startet den Trading Bot"""
    try:
        if server_bot is None:
            return jsonify({'success': False, 'message': 'Bot system not available'}), 500
            
        data = request.get_json() or {}
        mode = data.get('mode', 'paper')
        strategy = data.get('strategy', 'smart_money_machine')  # Default zu Smart Money Machine
        
        result = server_bot.start_bot(mode=mode, strategy=strategy)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stoppt den Trading Bot"""
    try:
        if server_bot is None:
            return jsonify({'success': False, 'message': 'Bot system not available'}), 500
            
        result = server_bot.stop_bot()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bot/status', methods=['GET'])
def get_status():
    """Gibt Bot-Status zurück"""
    try:
        if server_bot is None:
            return jsonify({'running': False, 'mode': 'offline', 'message': 'Bot system not available'})
            
        status = server_bot.get_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/switch-mode', methods=['POST'])
def switch_mode():
    """Wechselt zwischen Paper und Live Mode"""
    try:
        if server_bot is None:
            return jsonify({'success': False, 'message': 'Bot system not available'}), 500
            
        data = request.get_json() or {}
        new_mode = data.get('mode', 'paper')
        
        result = server_bot.switch_mode(new_mode)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error switching mode: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Gibt Portfolio-Zusammenfassung zurück"""
    try:
        if server_bot is None:
            return jsonify({'total_balance': 0, 'message': 'Bot system not available'})
            
        portfolio = server_bot.get_portfolio_summary()
        return jsonify(portfolio)
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/active', methods=['GET'])
def get_active_trades():
    """Gibt aktive Trades zurück"""
    try:
        trades = server_bot.get_active_trades()
        return jsonify(trades)
        
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Gibt Performance-Metriken zurück"""
    try:
        metrics = server_bot.get_performance_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Gibt verfügbare Strategien zurück"""
    try:
        # Liste der verfügbaren Strategien
        strategies = [
            {'id': 'smart_money_machine', 'name': '💰 Smart Money Machine', 'risk': 'balanced', 'description': 'Intelligenter Portfolio-Split: 85% Safe + 15% High-Risk mit Leverage'},
            {'id': 'adaptive_auto_strategy', 'name': '🤖 Auto-Strategy (Sorglos)', 'risk': 'adaptive', 'description': 'Vollautomatische Strategie mit intelligentem Risk Management'},
            {'id': 'momentum', 'name': 'Momentum Strategy', 'risk': 'medium', 'description': 'Trend-folgende Strategie'},
            {'id': 'mean_reversion', 'name': 'Mean Reversion', 'risk': 'low', 'description': 'Rückkehr zum Mittelwert'},
            {'id': 'arbitrage', 'name': 'Arbitrage', 'risk': 'low', 'description': 'Preisunterschiede ausnutzen'},
            {'id': 'grid_trading', 'name': 'Grid Trading', 'risk': 'medium', 'description': 'Raster-basierter Handel'},
            {'id': 'candle_momentum', 'name': 'Candle Momentum', 'risk': 'medium', 'description': 'Kerzen-basierte Signale'},
            {'id': 'lazy_billionaire_strategy', 'name': 'Lazy Billionaire', 'risk': 'low', 'description': 'Entspanntes DCA Trading'},
            {'id': 'super_lazy_billionaire_strategy', 'name': 'Super Lazy Billionaire', 'risk': 'low', 'description': 'Ultra-entspanntes Trading'},
            {'id': 'high_risk_daily', 'name': 'High Risk Daily', 'risk': 'high', 'description': 'Aggressives Tagestrading'},
            {'id': 'ml_strategy', 'name': 'ML Strategy', 'risk': 'high', 'description': 'Machine Learning basiert'}
        ]
        return jsonify(strategies)
        
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Gibt aktuelle Konfiguration zurück"""
    try:
        config = {
            'mode': server_bot.current_mode,
            'is_running': server_bot.is_running,
            'available_modes': ['paper', 'live'],
            'auto_strategy_settings': {
                'daily_risk_limit': 100.0,
                'portfolio_scale_factor': 0.02,
                'min_daily_limit': 50.0,
                'max_daily_limit': 500.0
            },
            'risk_limits': {
                'max_position_size': 0.1,
                'max_open_positions': 5,
                'stop_loss_percent': 5
            }
        }
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-strategy/settings', methods=['GET'])
def get_auto_strategy_settings():
    """Gibt Auto-Strategy Einstellungen zurück"""
    try:
        settings = {
            'daily_risk_limit': 100.0,
            'current_daily_pnl': 0.0,
            'portfolio_value': 1000.0,
            'calculated_risk_limit': 100.0,
            'available_strategies': {
                'conservative': ['mean_reversion', 'arbitrage', 'lazy_billionaire_strategy'],
                'moderate': ['momentum', 'grid_trading', 'candle_momentum'],
                'aggressive': ['high_risk_daily', 'ml_strategy', 'optimized_candle_momentum']
            },
            'current_market_regime': 'mixed',
            'selected_strategy': 'momentum'
        }
        return jsonify(settings)
        
    except Exception as e:
        logger.error(f"Error getting auto-strategy settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-strategy/settings', methods=['POST'])
def update_auto_strategy_settings():
    """Aktualisiert Auto-Strategy Einstellungen"""
    try:
        data = request.get_json() or {}
        
        # Hier würden wir die Einstellungen im Bot aktualisieren
        # server_bot.update_auto_strategy_settings(data)
        
        return jsonify({
            'success': True,
            'message': 'Auto-Strategy settings updated',
            'settings': data
        })
        
    except Exception as e:
        logger.error(f"Error updating auto-strategy settings: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/emergency-stop', methods=['POST'])
def emergency_stop():
    """Not-Stop für den Bot"""
    try:
        # Stoppe Bot sofort
        result = server_bot.stop_bot()
        
        # Optional: Schließe alle offenen Positionen
        # server_bot.close_all_positions()
        
        return jsonify({
            'success': result.get('success', False),
            'message': 'Emergency stop executed',
            'details': result
        })
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/orchestrator/status', methods=['GET'])
def get_orchestrator_status():
    """Gibt Status des Strategy Orchestrators zurück"""
    try:
        # Get orchestrator status if available
        if hasattr(server_bot, 'orchestrator') and server_bot.orchestrator:
            status = server_bot.orchestrator.get_orchestrator_status()
        else:
            # Default status if orchestrator not initialized
            status = {
                'active_strategies': [],
                'market_regime': {
                    'volatility': 'medium',
                    'trend': 'sideways',
                    'volume': 'normal',
                    'correlation': 0.5,
                    'confidence': 0.0
                },
                'risk_allocation': {},
                'total_risk_used': 0,
                'performance_summary': {}
            }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orchestrator/rebalance', methods=['POST'])
def force_orchestrator_rebalance():
    """Erzwingt eine Neubalancierung der Strategien"""
    try:
        if hasattr(server_bot, 'orchestrator') and server_bot.orchestrator:
            # This would be async in production
            result = {
                'success': True,
                'message': 'Rebalancing triggered',
                'timestamp': datetime.now().isoformat()
            }
            # server_bot.orchestrator.force_rebalance()
        else:
            result = {
                'success': False,
                'message': 'Orchestrator not initialized'
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error forcing rebalance: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/smart-money-machine/status', methods=['GET'])
def get_smart_money_machine_status():
    """Gibt Portfolio-Split Status der Smart Money Machine zurück"""
    try:
        if (hasattr(server_bot, 'trading_bot') and server_bot.trading_bot and 
            hasattr(server_bot.trading_bot, 'current_strategy')):
            
            strategy = server_bot.trading_bot.current_strategy
            
            # Check if it's SmartMoneyMachine
            if hasattr(strategy, 'get_portfolio_status'):
                portfolio_status = strategy.get_portfolio_status()
                return jsonify(portfolio_status)
            else:
                return jsonify({
                    'error': 'Current strategy is not Smart Money Machine',
                    'strategy_type': type(strategy).__name__
                })
        else:
            return jsonify({
                'error': 'Bot not running or strategy not initialized',
                'default_status': {
                    'total_capital': 1000.0,
                    'safe_capital': 850.0,
                    'high_risk_capital': 150.0,
                    'safe_allocation': 0.85,
                    'high_risk_allocation': 0.15,
                    'safe_performance': {'total_pnl': 0.0, 'trades': 0, 'wins': 0, 'daily_pnl': 0.0},
                    'high_risk_performance': {'total_pnl': 0.0, 'trades': 0, 'wins': 0, 'daily_pnl': 0.0},
                    'daily_trades': {'safe': 0, 'high_risk': 0}
                }
            })
        
    except Exception as e:
        logger.error(f"Error getting smart money machine status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orchestrator/strategies', methods=['GET'])
def get_orchestrator_strategies():
    """Gibt detaillierte Informationen über alle Strategien zurück"""
    try:
        strategies_info = {
            'momentum': {
                'name': 'Momentum Strategy',
                'description': 'Follows market trends',
                'optimal_conditions': {'trend': 'bullish', 'volatility': 'medium'},
                'risk_level': 'medium',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'mean_reversion': {
                'name': 'Mean Reversion',
                'description': 'Trades against extremes',
                'optimal_conditions': {'trend': 'sideways', 'volatility': 'low'},
                'risk_level': 'low',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'arbitrage': {
                'name': 'Arbitrage',
                'description': 'Exploits price differences',
                'optimal_conditions': {'correlation': 'high', 'volatility': 'low'},
                'risk_level': 'low',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'grid_trading': {
                'name': 'Grid Trading',
                'description': 'Places orders at intervals',
                'optimal_conditions': {'trend': 'sideways', 'volatility': 'medium'},
                'risk_level': 'medium',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'high_risk_daily': {
                'name': 'High Risk Daily',
                'description': 'Aggressive day trading',
                'optimal_conditions': {'volatility': 'extreme', 'volume': 'high'},
                'risk_level': 'high',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'ml_strategy': {
                'name': 'Machine Learning',
                'description': 'AI-powered predictions',
                'optimal_conditions': {'any': True},
                'risk_level': 'medium',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            },
            'smart_money_machine': {
                'name': 'Smart Money Machine',
                'description': 'Portfolio-Split: 85% Safe + 15% High-Risk mit Leverage',
                'optimal_conditions': {'any': True},
                'risk_level': 'balanced',
                'performance': {'win_rate': 0.0, 'total_pnl': 0.0, 'confidence': 0.5}
            }
        }
        
        # Update with real performance data if available
        if hasattr(server_bot, 'orchestrator') and server_bot.orchestrator:
            for strategy_name, perf in server_bot.orchestrator.strategy_performance.items():
                if strategy_name in strategies_info:
                    strategies_info[strategy_name]['performance'] = {
                        'win_rate': perf.win_rate,
                        'total_pnl': perf.total_pnl,
                        'confidence': perf.confidence_score,
                        'total_trades': perf.total_trades
                    }
        
        return jsonify(strategies_info)
        
    except Exception as e:
        logger.error(f"Error getting orchestrator strategies: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Dashboard HTML Routes
@app.route('/')
def dashboard_home():
    """Haupt-Dashboard Route"""
    try:
        dashboard_path = Path(__file__).parent / 'mobile_dashboard.html'
        return send_file(str(dashboard_path))
    except Exception as e:
        return f"Dashboard not found: {str(e)}", 404

@app.route('/mobile_dashboard.html')
def mobile_dashboard():
    """Mobile Dashboard Route"""
    try:
        dashboard_path = Path(__file__).parent / 'mobile_dashboard.html'
        return send_file(str(dashboard_path))
    except Exception as e:
        return f"Mobile Dashboard not found: {str(e)}", 404

@app.route('/enhanced_dashboard.html')
def enhanced_dashboard():
    """Enhanced AI Dashboard Route"""
    try:
        dashboard_path = Path(__file__).parent / 'enhanced_dashboard.html'
        return send_file(str(dashboard_path))
    except Exception as e:
        return f"Enhanced Dashboard not found: {str(e)}", 404

@app.route('/dashboard')
def dashboard_redirect():
    """Redirect to main dashboard"""
    try:
        dashboard_path = Path(__file__).parent / 'mobile_dashboard.html'
        return send_file(str(dashboard_path))
    except Exception as e:
        return f"Dashboard not found: {str(e)}", 404


if __name__ == '__main__':
    # Starte Server
    app.run(
        host='0.0.0.0',  # Erreichbar von allen IPs
        port=5000,
        debug=False,
        threaded=True
    )