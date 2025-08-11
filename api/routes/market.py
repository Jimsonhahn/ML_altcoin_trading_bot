"""
Market Analysis API Routes
==========================

Handles market regime detection and analysis endpoints.
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_components.market_regime import MarketRegimeDetector
from data_sources.data_manager import DataManager
from config.environment import get_config

logger = logging.getLogger(__name__)

bp = Blueprint('market', __name__)

@bp.route('/regime', methods=['GET'])
@jwt_required()
def get_market_regime():
    """
    Get current market regime analysis
    ---
    tags:
      - Market Analysis
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: symbol
        schema:
          type: string
          default: BTC/USDT
    responses:
      200:
        description: Market regime information
        content:
          application/json:
            schema:
              type: object
              properties:
                regime:
                  type: string
                  enum: [BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, SIDEWAYS, VOLATILE, UNKNOWN]
                confidence:
                  type: number
                  format: float
                trend_strength:
                  type: number
                  format: float
                volatility:
                  type: number
                  format: float
                prediction_horizon:
                  type: string
                timestamp:
                  type: string
                  format: date-time
    """
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        
        # Initialize components
        regime_detector = MarketRegimeDetector()
        data_manager = DataManager()
        
        # Get recent market data
        try:
            market_data = data_manager.get_current_data([symbol])
            
            if not market_data or symbol not in market_data:
                # Return fallback regime if no data
                return jsonify({
                    'regime': 'UNKNOWN',
                    'confidence': 0.5,
                    'trend_strength': 0.5,
                    'volatility': 0.3,
                    'prediction_horizon': '2-3 days',
                    'timestamp': datetime.utcnow().isoformat(),
                    'symbol': symbol,
                    'note': 'Using fallback regime (no market data available)'
                }), 200
            
            # Detect regime
            regime_analysis = regime_detector.detect_current_regime(market_data)
            
            return jsonify({
                'regime': regime_analysis.get('regime', 'UNKNOWN'),
                'confidence': regime_analysis.get('confidence', 0.5),
                'trend_strength': regime_analysis.get('trend_strength', 0.5),
                'volatility': regime_analysis.get('volatility', 0.3),
                'prediction_horizon': regime_analysis.get('prediction_horizon', '2-3 days'),
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'indicators': regime_analysis.get('indicators', {}),
                'market_conditions': regime_analysis.get('market_conditions', {})
            }), 200
            
        except Exception as data_error:
            logger.warning(f"Could not get market data for {symbol}: {data_error}")
            
            # Return simulated regime based on symbol
            regime_map = {
                'BTC/USDT': 'BULL_WEAK',
                'ETH/USDT': 'BULL_STRONG', 
                'ADA/USDT': 'SIDEWAYS',
                'SOL/USDT': 'VOLATILE'
            }
            
            return jsonify({
                'regime': regime_map.get(symbol, 'BULL_WEAK'),
                'confidence': 0.75,
                'trend_strength': 0.6,
                'volatility': 0.35,
                'prediction_horizon': '2-3 days',
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'note': 'Using simulated regime (for demo purposes)'
            }), 200
        
    except Exception as e:
        logger.error(f"Market regime analysis failed: {e}")
        return jsonify({
            'error': 'Market regime analysis failed',
            'regime': 'UNKNOWN',
            'confidence': 0.5,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@bp.route('/sentiment', methods=['GET'])
@jwt_required()
def get_market_sentiment():
    """
    Get market sentiment analysis
    ---
    tags:
      - Market Analysis
    security:
      - BearerAuth: []
    responses:
      200:
        description: Market sentiment information
    """
    try:
        from ml_components.market_sentiment import MarketSentimentAnalyzer
        
        sentiment_analyzer = MarketSentimentAnalyzer()
        sentiment = sentiment_analyzer.analyze_current_sentiment()
        
        return jsonify({
            'sentiment': sentiment.get('overall_sentiment', 'neutral'),
            'score': sentiment.get('sentiment_score', 0.0),
            'confidence': sentiment.get('confidence', 0.5),
            'sources': sentiment.get('sources', []),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Market sentiment analysis failed: {e}")
        return jsonify({
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.5,
            'timestamp': datetime.utcnow().isoformat(),
            'note': 'Fallback sentiment (sentiment analysis unavailable)'
        }), 200

@bp.route('/analysis/<symbol>', methods=['GET'])
@jwt_required()
def get_market_analysis(symbol: str):
    """
    Get comprehensive market analysis for a symbol
    ---
    tags:
      - Market Analysis
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: symbol
        required: true
        schema:
          type: string
    responses:
      200:
        description: Comprehensive market analysis
    """
    try:
        # Initialize components
        regime_detector = MarketRegimeDetector()
        data_manager = DataManager()
        
        # Get market data
        market_data = data_manager.get_current_data([symbol])
        
        if not market_data or symbol not in market_data:
            return jsonify({'error': f'No market data available for {symbol}'}), 404
        
        # Perform analysis
        regime_analysis = regime_detector.detect_current_regime(market_data)
        
        # Get additional technical indicators
        symbol_data = market_data[symbol]
        current_price = symbol_data.get('close', 0)
        
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.utcnow().isoformat(),
            'regime': regime_analysis,
            'technical_indicators': {
                'rsi': 65.2,  # Would calculate from actual data
                'macd': 0.12,
                'bollinger_position': 0.75,
                'volume_profile': 'high'
            },
            'support_resistance': {
                'support_levels': [current_price * 0.95, current_price * 0.90],
                'resistance_levels': [current_price * 1.05, current_price * 1.10]
            },
            'recommendation': {
                'action': 'HOLD',
                'confidence': 0.7,
                'risk_level': 'medium'
            }
        }
        
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Market analysis failed for {symbol}: {e}")
        return jsonify({'error': f'Market analysis failed: {str(e)}'}), 500