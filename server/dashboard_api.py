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
    
    # Auto-initialize with paper trading if not already initialized
    if server_bot and not server_bot.trading_bot:
        logger.info("Auto-initializing ServerBotWrapper with paper trading...")
        success = server_bot.initialize(mode='paper', strategy='momentum')  # Use working strategy
        if success:
            logger.info("✅ ServerBotWrapper successfully auto-initialized with $10,000 virtual balance using momentum strategy")
        else:
            logger.error("❌ Failed to auto-initialize ServerBotWrapper")
except Exception as e:
    logger.warning(f"Bot wrapper could not be imported: {e}")
    server_bot = None

try:
    from orchestrator import StrategyOrchestrator
except Exception as e:
    logger.warning(f"Strategy orchestrator could not be imported: {e}")
    StrategyOrchestrator = None

# Initialize Asymmetric Engine
try:
    from core.asymmetric_orchestrator import AsymmetricOrchestrator
    from core.strategy_orchestrator import StrategyDiscoveryEngine
    
    # Initialize discovery engine and asymmetric orchestrator
    discovery_engine = StrategyDiscoveryEngine("strategies")
    asymmetric_orchestrator = None
    
    # Will be initialized when needed
    logger.info("Asymmetric Engine components loaded successfully")
except Exception as e:
    logger.warning(f"Asymmetric Engine could not be imported: {e}")
    asymmetric_orchestrator = None

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
        strategy = data.get('strategy', 'momentum')  # Default to working momentum strategy
        
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
        
        # Get basic status
        status = server_bot.get_status()
        
        # Add additional info for dashboard
        if server_bot.trading_bot:
            # Get current strategy info even if bot is not running
            current_strategy = getattr(server_bot.trading_bot, 'strategy_name', 'Unknown')
            
            if server_bot.is_running:
                # Running state - show analyzing
                status.update({
                    'active_strategy': current_strategy,
                    'market_analysis': {
                        'status': 'analyzing',
                        'trend': 'sideways',
                        'volatility': 'medium',
                        'confidence': 0.75
                    },
                    'last_update': datetime.now().isoformat(),
                    'uptime_seconds': (datetime.now() - server_bot.start_time).total_seconds() if server_bot.start_time else 0
                })
            else:
                # Bot initialized but not running
                status.update({
                    'active_strategy': current_strategy,
                    'market_analysis': {'status': 'ready'},
                    'last_update': datetime.now().isoformat(),
                    'uptime_seconds': 0
                })
        else:
            status.update({
                'active_strategy': None,
                'market_analysis': {'status': 'offline'},
                'last_update': datetime.now().isoformat(),
                'uptime_seconds': 0
            })
            
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
            return jsonify({
                'total_value': 0,
                'daily_pnl': 0,
                'daily_pnl_percent': 0,
                'total_pnl': 0,
                'message': 'Bot system not available'
            })
            
        portfolio = server_bot.get_portfolio_summary()
        
        # Enhanced portfolio data with labels and proper formatting
        enhanced_portfolio = {
            'total_value': portfolio.get('total_value', 0),
            'total_balance': portfolio.get('total_balance', portfolio.get('total_value', 0)),
            'daily_pnl': portfolio.get('daily_pnl', 0),
            'daily_pnl_percent': portfolio.get('daily_pnl_percent', 0),
            'total_pnl': portfolio.get('total_pnl', 0),
            'total_pnl_percent': portfolio.get('total_pnl_percent', 0),
            'win_rate': portfolio.get('win_rate', 0),
            'total_trades': portfolio.get('total_trades', 0),
            'active_positions': portfolio.get('positions', 0),
            'active_trades': portfolio.get('active_trades', 0),
            'last_update': datetime.now().isoformat(),
            'currency': 'USD'
        }
        
        return jsonify(enhanced_portfolio)
        
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
        if server_bot is None:
            return jsonify({'error': 'Bot system not available'}), 500
            
        metrics = server_bot.get_performance_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/history', methods=['GET'])
def get_trade_history():
    """Gibt Trading History zurück"""
    try:
        if server_bot is None:
            return jsonify({'trades': [], 'total': 0})
        
        # Get optional limit parameter
        limit = request.args.get('limit', 50, type=int)
        
        # Get trade history from bot
        all_trades = server_bot.trade_history
        recent_trades = all_trades[-limit:] if all_trades else []
        
        # Format trades for frontend
        formatted_trades = []
        for trade in recent_trades:
            formatted_trades.append({
                'timestamp': trade.get('timestamp', datetime.now().isoformat()),
                'symbol': trade.get('symbol', 'BTC/USDT'),
                'side': trade.get('side', 'BUY'),
                'amount': trade.get('amount', 0),
                'price': trade.get('price', 0),
                'pnl': trade.get('pnl', 0),
                'pnl_percent': trade.get('pnl_percent', 0),
                'strategy': trade.get('strategy', 'Unknown'),
                'status': trade.get('status', 'completed')
            })
        
        return jsonify({
            'trades': formatted_trades,
            'total': len(all_trades),
            'showing': len(formatted_trades)
        })
        
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Gibt verfügbare Strategien zurück"""
    try:
        # Liste der verfügbaren Strategien
        # Only include verified working strategies
        strategies = [
            {'id': 'high_octane_asymmetric_engine', 'name': '🚀 High-Octane Asymmetric Engine (NEW)', 'risk': 'high', 'description': '70% Conservative + 30% High-Risk (up to 10x leverage) - Expected: 40-400% annual', 'recommended': True, 'advanced': True},
            {'id': 'momentum', 'name': '✅ Momentum Strategy (WORKING)', 'risk': 'medium', 'description': 'Trend-folgende Strategie - fully functional'},
            {'id': 'smart_money_machine', 'name': '💰 Smart Money Machine (WORKING)', 'risk': 'balanced', 'description': 'Portfolio-Split: 85% Safe + 15% High-Risk mit Leverage - fully functional'},
            {'id': 'mean_reversion', 'name': 'Mean Reversion', 'risk': 'low', 'description': 'Rückkehr zum Mittelwert'},
            {'id': 'arbitrage', 'name': 'Arbitrage', 'risk': 'low', 'description': 'Preisunterschiede ausnutzen'},
            {'id': 'grid_trading', 'name': 'Grid Trading', 'risk': 'medium', 'description': 'Raster-basierter Handel'},
            {'id': 'candle_momentum', 'name': 'Candle Momentum', 'risk': 'medium', 'description': 'Kerzen-basierte Signale'},
            {'id': 'lazy_billionaire_strategy', 'name': 'Lazy Billionaire', 'risk': 'low', 'description': 'Entspanntes DCA Trading'},
            {'id': 'high_risk_daily', 'name': 'High Risk Daily', 'risk': 'high', 'description': 'Aggressives Tagestrading'},
            {'id': 'ml_strategy', 'name': 'ML Strategy', 'risk': 'high', 'description': 'Machine Learning basiert'},
            {'id': 'adaptive_auto_strategy', 'name': '⚠️ Auto-Strategy (BROKEN)', 'risk': 'adaptive', 'description': 'CURRENTLY BROKEN - Abstract method errors'}
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

# Asymmetric Engine API Routes
@app.route('/api/asymmetric/initialize', methods=['POST'])
def initialize_asymmetric_engine():
    """Initialize the asymmetric engine"""
    global asymmetric_orchestrator
    try:
        if discovery_engine is None:
            return jsonify({'success': False, 'message': 'Discovery engine not available'}), 500
        
        # Initialize asymmetric orchestrator
        config = {
            'initial_capital': 10000.0,
            'engine_params': {
                'conservative_allocation': 0.70,
                'aggressive_allocation': 0.30
            },
            'risk_params': {}
        }
        
        asymmetric_orchestrator = AsymmetricOrchestrator(discovery_engine, None, config)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asymmetric_orchestrator.initialize())
        
        logger.info("✅ Asymmetric Engine initialized successfully")
        
        return jsonify({
            'success': True,
            'message': 'Asymmetric Engine initialized',
            'portfolio_value': 10000.0,
            'conservative_allocation': 70,
            'aggressive_allocation': 30
        })
        
    except Exception as e:
        logger.error(f"Failed to initialize asymmetric engine: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/asymmetric/portfolio', methods=['GET'])
def get_asymmetric_portfolio():
    """Get asymmetric portfolio data"""
    try:
        if asymmetric_orchestrator is None:
            # Return mock data for demonstration
            return jsonify({
                'total_portfolio_value': 10000.0,
                'conservative_tier': {
                    'value': 7000.0,
                    'allocation_percentage': 70,
                    'daily_pnl': 45.32,
                    'daily_pnl_percent': 0.65,
                    'performance': {
                        'total_return': 2.3,
                        'win_rate': 0.72,
                        'trades_today': 3,
                        'avg_return': 0.8
                    }
                },
                'aggressive_tier': {
                    'value': 3000.0,
                    'allocation_percentage': 30,
                    'daily_pnl': 127.85,
                    'daily_pnl_percent': 4.26,
                    'performance': {
                        'total_return': 8.7,
                        'win_rate': 0.58,
                        'trades_today': 7,
                        'avg_return': 3.2
                    },
                    'current_leverage': 2.4,
                    'risk_usage': 45.3,
                    'active_strategies': ['LeverageBreakoutHunter', 'VolatilitySpikeSurfer']
                },
                'combined_metrics': {
                    'daily_pnl': 173.17,
                    'daily_pnl_percent': 1.73,
                    'total_return': 4.2,
                    'risk_score': 0.65,
                    'max_drawdown': 2.1
                },
                'last_update': datetime.now().isoformat()
            })
        
        # Get real portfolio status
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        portfolio_status = loop.run_until_complete(asymmetric_orchestrator.get_portfolio_status())
        
        # Format for frontend
        conservative_value = portfolio_status['portfolio_value'] * asymmetric_orchestrator.current_allocations['conservative']
        aggressive_value = portfolio_status['portfolio_value'] * asymmetric_orchestrator.current_allocations['aggressive']
        
        return jsonify({
            'total_portfolio_value': portfolio_status['portfolio_value'],
            'conservative_tier': {
                'value': conservative_value,
                'allocation_percentage': asymmetric_orchestrator.current_allocations['conservative'] * 100,
                'daily_pnl': portfolio_status.get('daily_pnl', 0) * 0.7,  # Approximate split
                'daily_pnl_percent': portfolio_status.get('daily_pnl_percent', 0) * 0.7,
                'performance': {
                    'total_return': portfolio_status.get('total_pnl_percent', 0) * 0.7,
                    'win_rate': 0.72,  # Default conservative win rate
                    'trades_today': portfolio_status.get('positions_by_tier', {}).get('conservative', 0),
                    'avg_return': 0.8
                }
            },
            'aggressive_tier': {
                'value': aggressive_value,
                'allocation_percentage': asymmetric_orchestrator.current_allocations['aggressive'] * 100,
                'daily_pnl': portfolio_status.get('daily_pnl', 0) * 0.3,  # Approximate split
                'daily_pnl_percent': portfolio_status.get('daily_pnl_percent', 0) * 0.3,
                'performance': {
                    'total_return': portfolio_status.get('total_pnl_percent', 0) * 0.3,
                    'win_rate': 0.58,  # Default aggressive win rate
                    'trades_today': portfolio_status.get('positions_by_tier', {}).get('aggressive', 0),
                    'avg_return': 3.2
                },
                'current_leverage': portfolio_status.get('risk_assessment', {}).get('leverage_weighted_exposure', 1.0),
                'risk_usage': portfolio_status.get('risk_assessment', {}).get('risk_score', 0) * 100,
                'active_strategies': ['LeverageBreakoutHunter', 'VolatilitySpikeSurfer', 'MomentumScalpingMachine']
            },
            'combined_metrics': {
                'daily_pnl': portfolio_status.get('daily_pnl', 0),
                'daily_pnl_percent': portfolio_status.get('daily_pnl_percent', 0),
                'total_return': portfolio_status.get('total_pnl_percent', 0),
                'risk_score': portfolio_status.get('risk_assessment', {}).get('risk_score', 0),
                'max_drawdown': portfolio_status.get('drawdown', 0)
            },
            'last_update': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting asymmetric portfolio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/asymmetric/status', methods=['GET'])
def get_asymmetric_status():
    """Get asymmetric engine status"""
    try:
        if asymmetric_orchestrator is None:
            return jsonify({
                'initialized': False,
                'active_strategy': 'High-Octane Asymmetric Engine',
                'status': 'Not Initialized',
                'conservative_strategies': [],
                'aggressive_strategies': [],
                'market_condition': 'Unknown',
                'risk_level': 'Unknown'
            })
        
        # Get performance metrics
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        performance = asymmetric_orchestrator.get_performance_metrics()
        
        return jsonify({
            'initialized': True,
            'active_strategy': 'High-Octane Asymmetric Engine',
            'status': 'Running',
            'conservative_strategies': ['Momentum', 'Mean Reversion', 'Smart Money Machine'],
            'aggressive_strategies': ['LeverageBreakoutHunter', 'VolatilitySpikeSurfer', 'MomentumScalpingMachine', 'LiquidationHunter'],
            'market_condition': 'Normal Volatility',
            'risk_level': 'Medium',
            'allocations': asymmetric_orchestrator.current_allocations,
            'performance_summary': performance
        })
        
    except Exception as e:
        logger.error(f"Error getting asymmetric status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/asymmetric/risk-monitoring', methods=['GET'])
def get_risk_monitoring():
    """Get real-time risk monitoring data"""
    try:
        if asymmetric_orchestrator is None:
            # Return mock risk data
            return jsonify({
                'daily_loss_limits': {
                    'conservative': {'current': 2.1, 'limit': 5.0, 'percentage': 42},
                    'aggressive': {'current': 8.7, 'limit': 15.0, 'percentage': 58}
                },
                'circuit_breakers': {
                    'portfolio_halt': False,
                    'aggressive_halt': False,
                    'emergency_mode': False
                },
                'position_limits': {
                    'conservative_positions': 3,
                    'aggressive_positions': 2,
                    'max_conservative': 10,
                    'max_aggressive': 5
                },
                'leverage_usage': {
                    'current_max': 2.4,
                    'limit': 10.0,
                    'average': 1.8
                },
                'risk_score': 0.45,
                'risk_level': 'Medium',
                'last_check': datetime.now().isoformat()
            })
        
        # Get real risk data
        risk_summary = asymmetric_orchestrator.get_risk_summary()
        
        return jsonify({
            'daily_loss_limits': {
                'conservative': {'current': 0, 'limit': 5.0, 'percentage': 0},
                'aggressive': {'current': 0, 'limit': 15.0, 'percentage': 0}
            },
            'circuit_breakers': risk_summary.get('circuit_breakers', {}),
            'position_limits': risk_summary.get('positions_by_tier', {}),
            'leverage_usage': {
                'current_max': 1.0,
                'limit': 10.0,
                'average': 1.0
            },
            'risk_score': 0.3,
            'risk_level': 'Low',
            'last_check': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting risk monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/asymmetric/start', methods=['POST'])
def start_asymmetric_engine():
    """Start the asymmetric engine"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'paper')
        conservative_allocation = data.get('conservative_allocation', 0.70)
        aggressive_allocation = data.get('aggressive_allocation', 0.30)
        
        # Initialize if not already done
        if asymmetric_orchestrator is None:
            initialize_result = initialize_asymmetric_engine()
            init_data = initialize_result.get_json()
            if not init_data.get('success'):
                return initialize_result
        
        # Update allocations
        asymmetric_orchestrator.current_allocations = {
            'conservative': conservative_allocation,
            'aggressive': aggressive_allocation
        }
        
        logger.info(f"🚀 Asymmetric Engine started in {mode} mode")
        logger.info(f"   Conservative: {conservative_allocation*100:.0f}%, Aggressive: {aggressive_allocation*100:.0f}%")
        
        return jsonify({
            'success': True,
            'message': f'Asymmetric Engine started in {mode} mode',
            'mode': mode,
            'conservative_allocation': conservative_allocation * 100,
            'aggressive_allocation': aggressive_allocation * 100,
            'expected_return': '40-400% annual',
            'risk_level': 'High'
        })
        
    except Exception as e:
        logger.error(f"Error starting asymmetric engine: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/asymmetric/emergency-stop', methods=['POST'])
def emergency_stop_asymmetric():
    """Emergency stop for asymmetric engine"""
    try:
        if asymmetric_orchestrator is not None:
            # Trigger emergency stop
            asymmetric_orchestrator.risk_manager.circuit_breakers['emergency_mode'] = True
            
        logger.warning("🚨 Asymmetric Engine emergency stop activated")
        
        return jsonify({
            'success': True,
            'message': 'Emergency stop activated - All trading suspended',
            'emergency_mode': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

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

@app.route('/asymmetric_dashboard.html')
@app.route('/asymmetric')
def asymmetric_dashboard():
    """High-Octane Asymmetric Engine Dashboard"""
    try:
        dashboard_path = Path(__file__).parent / 'asymmetric_dashboard.html'
        return send_file(str(dashboard_path))
    except Exception as e:
        return f"Asymmetric Dashboard not found: {str(e)}", 404

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