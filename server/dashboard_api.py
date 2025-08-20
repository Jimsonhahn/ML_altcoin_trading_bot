"""
Dashboard API - Flask API für Mobile Dashboard Control
=====================================================

Bietet REST API Endpoints für das Mobile Dashboard.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.bot_wrapper import server_bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        data = request.get_json() or {}
        mode = data.get('mode', 'paper')
        strategy = data.get('strategy', 'adaptive_auto_strategy')  # Default zu Auto-Strategy
        
        result = server_bot.start_bot(mode=mode, strategy=strategy)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stoppt den Trading Bot"""
    try:
        result = server_bot.stop_bot()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bot/status', methods=['GET'])
def get_status():
    """Gibt Bot-Status zurück"""
    try:
        status = server_bot.get_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/switch-mode', methods=['POST'])
def switch_mode():
    """Wechselt zwischen Paper und Live Mode"""
    try:
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


if __name__ == '__main__':
    # Starte Server
    app.run(
        host='0.0.0.0',  # Erreichbar von allen IPs
        port=5000,
        debug=False,
        threaded=True
    )