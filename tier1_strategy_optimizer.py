#!/usr/bin/env python3
"""
Tier 1 Strategy Optimizer
========================

Systematische Optimierung der Trading-Strategie bis Tier 1 Performance erreicht wird:
- Target: Sharpe Ratio > 1.5
- Target: Annual Return > 25%  
- Target: Win Rate > 55%
- Target: Max Drawdown < 10%
- Target: Profit Factor > 1.5

Iterative Verbesserung mit A/B Testing verschiedener Komponenten.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import itertools
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import realistic backtest engine
from realistic_crypto_backtest import (
    RealisticBacktester, RealisticMarketDataGenerator, 
    RealisticExchangeSimulator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Tier1Targets:
    """Tier 1 Performance Targets"""
    min_annual_return: float = 0.25      # 25%
    min_sharpe_ratio: float = 1.5        # 1.5
    min_win_rate: float = 0.55           # 55%
    max_drawdown: float = 0.10           # 10%
    min_profit_factor: float = 1.5       # 1.5
    min_trades_per_year: int = 20        # Mindestens 20 Trades
    max_cost_impact: float = 0.02        # Max 2% Kosten


class AdvancedTradingStrategy:
    """Advanced Trading Strategy mit multiplen Optimierungen"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Optimierte Parameter (werden iterativ angepasst)
        self.position_sizing = self.config.get('position_sizing', {
            'max_position_size': 0.12,      # Erhöht von 0.08
            'min_position_size': 0.03,      # Minimum für Ausführung
            'volatility_scaling': True,     # Skaliere mit Volatilität
            'kelly_fraction': 0.25,         # Kelly Criterion
            'risk_per_trade': 0.02          # 2% Risiko per Trade
        })
        
        self.signal_params = self.config.get('signal_params', {
            'min_signal_strength': 0.6,     # Reduziert von 0.65
            'trend_lookback': 50,           # Trend-Analyse Periode
            'momentum_periods': [5, 10, 20, 50],  # Multi-Period Momentum
            'volume_threshold': 1.3,        # Volumen-Bestätigung
            'volatility_regime_threshold': 0.035,  # Vol-Regime Filter
            'rsi_oversold': 25,            # RSI Levels
            'rsi_overbought': 75,
            'bb_squeeze_threshold': 0.02    # Bollinger Band Squeeze
        })
        
        self.risk_management = self.config.get('risk_management', {
            'stop_loss_base': 0.025,        # 2.5% Base Stop
            'stop_loss_atr_mult': 1.5,      # ATR-basierter Stop
            'take_profit_base': 0.06,       # 6% Base Target
            'take_profit_dynamic': True,    # Dynamic TP basierend auf Volatilität
            'trailing_stop_trigger': 0.4,  # Bei 40% des Targets
            'max_holding_hours': 72,        # Max 3 Tage
            'daily_loss_limit': 0.03,      # 3% täglich
            'max_correlated_positions': 1   # Keine korrelierten Positionen
        })
        
        self.advanced_features = self.config.get('advanced_features', {
            'multi_timeframe': True,        # Multi-Timeframe Analyse
            'regime_detection': True,       # Markt-Regime Detection
            'sentiment_weighting': False,   # Sentiment Analysis (placeholder)
            'adaptive_parameters': True,    # Adaptive Parameter
            'ml_signal_boost': False        # ML Signal Enhancement (placeholder)
        })
        
        # State Management
        self.performance_history = []
        self.regime_state = 'unknown'
        self.adaptive_multipliers = {
            'signal_threshold': 1.0,
            'position_size': 1.0,
            'risk_management': 1.0
        }
        
        # Daily limits
        self.daily_trades = 0
        self.daily_pnl = 0
        self.current_date = None
        
        logger.info("Advanced Tier 1 Strategy initialized")
    
    def calculate_advanced_indicators(self, data: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """Berechnet erweiterte technische Indikatoren"""
        
        if len(data) < lookback:
            return {}
            
        recent_data = data.tail(lookback).copy()
        current_price = recent_data['close'].iloc[-1]
        
        indicators = {}
        
        # === BASIC INDICATORS ===
        
        # Multiple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            if len(recent_data) >= period:
                indicators[f'sma_{period}'] = recent_data['close'].rolling(period).mean().iloc[-1]
        
        # EMAs
        for span in [8, 12, 21, 26, 50]:
            if len(recent_data) >= span:
                indicators[f'ema_{span}'] = recent_data['close'].ewm(span=span).mean().iloc[-1]
        
        # === MOMENTUM INDICATORS ===
        
        # Multi-period RSI
        for period in [9, 14, 21]:
            if len(recent_data) >= period + 1:
                delta = recent_data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = -delta.where(delta < 0, 0).rolling(period).mean()
                rs = gain / loss
                indicators[f'rsi_{period}'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD Family
        if len(recent_data) >= 26:
            exp1 = recent_data['close'].ewm(span=12).mean()
            exp2 = recent_data['close'].ewm(span=26).mean()
            indicators['macd'] = (exp1 - exp2).iloc[-1]
            indicators['macd_signal'] = (exp1 - exp2).ewm(span=9).mean().iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Multi-Period Momentum
        for period in self.signal_params['momentum_periods']:
            if len(recent_data) >= period + 1:
                momentum = (current_price / recent_data['close'].iloc[-(period+1)]) - 1
                indicators[f'momentum_{period}'] = momentum
        
        # Rate of Change
        for period in [10, 20]:
            if len(recent_data) >= period + 1:
                roc = ((current_price - recent_data['close'].iloc[-(period+1)]) / 
                       recent_data['close'].iloc[-(period+1)]) * 100
                indicators[f'roc_{period}'] = roc
        
        # === VOLATILITY INDICATORS ===
        
        # Multiple period volatilities
        for period in [10, 20, 50]:
            if len(recent_data) >= period + 1:
                returns = recent_data['close'].pct_change().dropna()
                if len(returns) >= period:
                    vol = returns.tail(period).std() * np.sqrt(252)
                    indicators[f'volatility_{period}'] = vol
        
        # ATR (Average True Range)
        if len(recent_data) >= 14 and all(col in recent_data.columns for col in ['high', 'low']):
            high = recent_data['high']
            low = recent_data['low']
            close_prev = recent_data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            true_range = pd.DataFrame([tr1, tr2, tr3]).max(axis=0)
            indicators['atr_14'] = true_range.rolling(14).mean().iloc[-1]
            indicators['atr_pct'] = (indicators['atr_14'] / current_price) * 100
        
        # Bollinger Bands
        if len(recent_data) >= 20:
            sma20 = recent_data['close'].rolling(20).mean()
            std20 = recent_data['close'].rolling(20).std()
            indicators['bb_upper'] = (sma20 + 2*std20).iloc[-1]
            indicators['bb_lower'] = (sma20 - 2*std20).iloc[-1]
            indicators['bb_middle'] = sma20.iloc[-1]
            indicators['bb_position'] = ((current_price - indicators['bb_lower']) / 
                                       (indicators['bb_upper'] - indicators['bb_lower']))
            indicators['bb_width'] = ((indicators['bb_upper'] - indicators['bb_lower']) / 
                                    indicators['bb_middle'])
        
        # === VOLUME INDICATORS ===
        
        if 'volume_usd' in recent_data.columns:
            # Multiple Volume Ratios
            for period in [5, 10, 20, 50]:
                if len(recent_data) >= period:
                    avg_vol = recent_data['volume_usd'].rolling(period).mean().iloc[-1]
                    current_vol = recent_data['volume_usd'].iloc[-1]
                    indicators[f'volume_ratio_{period}'] = current_vol / avg_vol if avg_vol > 0 else 1
            
            # Volume Trend
            if len(recent_data) >= 10:
                vol_trend = recent_data['volume_usd'].rolling(10).apply(
                    lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
                ).iloc[-1]
                indicators['volume_trend'] = vol_trend
        
        # === ADVANCED PATTERN RECOGNITION ===
        
        # Trend Strength Index
        if len(recent_data) >= 50:
            highs = recent_data['high'].rolling(5).max()
            lows = recent_data['low'].rolling(5).min()
            trend_up = (highs.iloc[-1] > highs.iloc[-10]).astype(int)
            trend_down = (lows.iloc[-1] < lows.iloc[-10]).astype(int)
            indicators['trend_strength'] = trend_up - trend_down
        
        # Support/Resistance Levels
        if len(recent_data) >= 50:
            # Simple S/R based on recent highs/lows
            recent_highs = recent_data['high'].rolling(20).max()
            recent_lows = recent_data['low'].rolling(20).min()
            
            resistance = recent_highs.iloc[-1]
            support = recent_lows.iloc[-1]
            
            indicators['resistance_distance'] = (resistance - current_price) / current_price
            indicators['support_distance'] = (current_price - support) / current_price
        
        # Market Structure (Higher Highs, Lower Lows)
        if len(recent_data) >= 20:
            highs_10 = recent_data['high'].rolling(10).max()
            lows_10 = recent_data['low'].rolling(10).min()
            
            if len(highs_10.dropna()) >= 2:
                hh = highs_10.iloc[-1] > highs_10.iloc[-2]  # Higher High
                ll = lows_10.iloc[-1] < lows_10.iloc[-2]    # Lower Low
                hl = lows_10.iloc[-1] > lows_10.iloc[-2]    # Higher Low
                lh = highs_10.iloc[-1] < highs_10.iloc[-2]  # Lower High
                
                if hh and hl:
                    indicators['market_structure'] = 1    # Uptrend
                elif ll and lh:
                    indicators['market_structure'] = -1   # Downtrend
                else:
                    indicators['market_structure'] = 0    # Sideways
        
        return indicators
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Erweiterte Markt-Regime Detection"""
        
        if len(data) < 100:
            return 'insufficient_data'
        
        recent = data.tail(50)
        
        # Volatility Regime
        vol_20 = recent['close'].pct_change().std() * np.sqrt(252)
        
        # Trend Analysis
        returns_20 = (recent['close'].iloc[-1] / recent['close'].iloc[-20]) - 1
        returns_50 = (recent['close'].iloc[-1] / recent['close'].iloc[-50]) - 1 if len(data) >= 50 else 0
        
        # Volume Analysis
        if 'volume_usd' in recent.columns:
            vol_trend = recent['volume_usd'].rolling(10).mean().iloc[-1] / recent['volume_usd'].rolling(30).mean().iloc[-1]
        else:
            vol_trend = 1.0
        
        # Regime Classification
        if vol_20 > 0.8:  # Very high volatility
            regime = 'crisis'
        elif vol_20 > 0.5:  # High volatility
            if returns_20 < -0.1:  # Down > 10%
                regime = 'bear_volatile'
            elif returns_20 > 0.1:  # Up > 10%
                regime = 'bull_volatile'
            else:
                regime = 'sideways_volatile'
        else:  # Normal/Low volatility
            if returns_20 > 0.05 and returns_50 > 0.1:
                regime = 'bull_trending'
            elif returns_20 < -0.05 and returns_50 < -0.1:
                regime = 'bear_trending'
            else:
                regime = 'sideways_calm'
        
        self.regime_state = regime
        return regime
    
    def multi_timeframe_analysis(self, data_1h: pd.DataFrame, 
                                current_idx: int) -> Dict[str, Any]:
        """Multi-Timeframe Signal Analyse"""
        
        if not self.advanced_features['multi_timeframe'] or current_idx < 100:
            return {'timeframe_bias': 'neutral', 'strength': 0.5}
        
        # Simuliere verschiedene Timeframes aus stündlichen Daten
        # 4H Timeframe (alle 4 Stunden)
        data_4h = data_1h.iloc[::4].copy() if len(data_1h) > 20 else data_1h
        
        # 1D Timeframe (alle 24 Stunden)  
        data_1d = data_1h.iloc[::24].copy() if len(data_1h) > 100 else data_1h
        
        signals = {}
        
        # 1H Signale (aktueller Timeframe)
        if len(data_1h) >= 50:
            h1_trend = 'bullish' if data_1h['close'].iloc[-1] > data_1h['close'].rolling(20).mean().iloc[-1] else 'bearish'
            h1_momentum = (data_1h['close'].iloc[-1] / data_1h['close'].iloc[-10]) - 1
            signals['1h'] = {'trend': h1_trend, 'momentum': h1_momentum}
        
        # 4H Signale
        if len(data_4h) >= 20:
            h4_trend = 'bullish' if data_4h['close'].iloc[-1] > data_4h['close'].rolling(10).mean().iloc[-1] else 'bearish'
            h4_momentum = (data_4h['close'].iloc[-1] / data_4h['close'].iloc[-5]) - 1
            signals['4h'] = {'trend': h4_trend, 'momentum': h4_momentum}
        
        # 1D Signale
        if len(data_1d) >= 10:
            d1_trend = 'bullish' if data_1d['close'].iloc[-1] > data_1d['close'].rolling(5).mean().iloc[-1] else 'bearish'
            d1_momentum = (data_1d['close'].iloc[-1] / data_1d['close'].iloc[-3]) - 1 if len(data_1d) >= 3 else 0
            signals['1d'] = {'trend': d1_trend, 'momentum': d1_momentum}
        
        # Confluence Analysis
        bullish_timeframes = sum(1 for tf in signals.values() if tf['trend'] == 'bullish')
        bearish_timeframes = sum(1 for tf in signals.values() if tf['trend'] == 'bearish')
        
        if bullish_timeframes > bearish_timeframes:
            timeframe_bias = 'bullish'
            strength = bullish_timeframes / len(signals)
        elif bearish_timeframes > bullish_timeframes:
            timeframe_bias = 'bearish'
            strength = bearish_timeframes / len(signals)
        else:
            timeframe_bias = 'neutral'
            strength = 0.5
        
        return {
            'timeframe_bias': timeframe_bias,
            'strength': strength,
            'signals': signals,
            'confluence_score': abs(bullish_timeframes - bearish_timeframes) / len(signals) if signals else 0
        }
    
    def generate_advanced_signal(self, data: pd.DataFrame, current_idx: int, 
                               timestamp: datetime) -> Dict[str, Any]:
        """Generiert erweiterte Trading-Signale mit multiplen Filtern"""
        
        # Daily Reset
        current_date = timestamp.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_trades = 0
            self.daily_pnl = 0
        
        # Constraints Check
        if self.daily_trades >= 3:  # Max 3 Trades täglich
            return {'direction': 'hold', 'reason': 'daily_limit', 'strength': 0}
        
        if abs(self.daily_pnl) >= self.risk_management['daily_loss_limit']:
            return {'direction': 'hold', 'reason': 'daily_loss_limit', 'strength': 0}
        
        # Get Indicators
        indicators = self.calculate_advanced_indicators(data.iloc[:current_idx+1])
        if not indicators:
            return {'direction': 'hold', 'reason': 'insufficient_data', 'strength': 0}
        
        current_price = data['close'].iloc[current_idx]
        
        # Market Regime Detection
        regime = self.detect_market_regime(data.iloc[:current_idx+1])
        
        # Multi-Timeframe Analysis
        mtf_analysis = self.multi_timeframe_analysis(data.iloc[:current_idx+1], current_idx)
        
        # === SIGNAL GENERATION SYSTEM ===
        
        signal_components = []
        signal_reasons = []
        
        # 1. TREND FILTER (Must Pass)
        trend_score = self._analyze_trend_component(indicators, current_price)
        if abs(trend_score) < 0.3:  # Minimum trend strength
            return {'direction': 'hold', 'reason': 'weak_trend', 'strength': 0}
        
        signal_components.append(trend_score * 0.3)  # 30% weight
        signal_reasons.append(f"trend_{trend_score:.2f}")
        
        # 2. MOMENTUM CONFIRMATION  
        momentum_score = self._analyze_momentum_component(indicators)
        signal_components.append(momentum_score * 0.25)  # 25% weight
        signal_reasons.append(f"momentum_{momentum_score:.2f}")
        
        # 3. VOLATILITY REGIME FILTER
        vol_score = self._analyze_volatility_component(indicators, regime)
        if vol_score < -0.5:  # Too volatile
            return {'direction': 'hold', 'reason': 'high_volatility', 'strength': 0}
        
        signal_components.append(vol_score * 0.15)  # 15% weight
        signal_reasons.append(f"volatility_{vol_score:.2f}")
        
        # 4. VOLUME CONFIRMATION
        volume_score = self._analyze_volume_component(indicators)
        signal_components.append(volume_score * 0.15)  # 15% weight
        signal_reasons.append(f"volume_{volume_score:.2f}")
        
        # 5. MULTI-TIMEFRAME ALIGNMENT
        mtf_score = self._analyze_mtf_component(mtf_analysis)
        signal_components.append(mtf_score * 0.1)  # 10% weight
        signal_reasons.append(f"mtf_{mtf_score:.2f}")
        
        # 6. PATTERN RECOGNITION
        pattern_score = self._analyze_pattern_component(indicators)
        signal_components.append(pattern_score * 0.05)  # 5% weight
        signal_reasons.append(f"pattern_{pattern_score:.2f}")
        
        # === FINAL SIGNAL CALCULATION ===
        
        base_signal_strength = sum(signal_components)
        
        # Regime Adjustment
        regime_multiplier = self._get_regime_multiplier(regime)
        adjusted_signal = base_signal_strength * regime_multiplier
        
        # Adaptive Parameter Adjustment
        adaptive_threshold = (self.signal_params['min_signal_strength'] * 
                            self.adaptive_multipliers['signal_threshold'])
        
        # Direction Determination
        if adjusted_signal > adaptive_threshold:
            direction = 'buy'
            confidence = min(adjusted_signal, 1.0)
        elif adjusted_signal < -adaptive_threshold:
            direction = 'sell'  
            confidence = min(abs(adjusted_signal), 1.0)
        else:
            direction = 'hold'
            confidence = 0
        
        return {
            'direction': direction,
            'strength': abs(adjusted_signal),
            'confidence': confidence,
            'base_signal': base_signal_strength,
            'regime_multiplier': regime_multiplier,
            'regime': regime,
            'indicators': indicators,
            'mtf_analysis': mtf_analysis,
            'signal_components': dict(zip(signal_reasons, signal_components)),
            'adaptive_threshold': adaptive_threshold
        }
    
    def _analyze_trend_component(self, indicators: Dict, price: float) -> float:
        """Analysiert Trend-Komponente"""
        
        signals = []
        
        # SMA Trend Hierarchy
        sma_20 = indicators.get('sma_20', price)
        sma_50 = indicators.get('sma_50', price)
        sma_100 = indicators.get('sma_100', price)
        
        if price > sma_20 > sma_50 > sma_100:
            signals.append(0.8)  # Strong uptrend
        elif price > sma_20 > sma_50:
            signals.append(0.6)  # Moderate uptrend
        elif price < sma_20 < sma_50 < sma_100:
            signals.append(-0.8)  # Strong downtrend
        elif price < sma_20 < sma_50:
            signals.append(-0.6)  # Moderate downtrend
        
        # EMA Convergence/Divergence
        ema_12 = indicators.get('ema_12', price)
        ema_26 = indicators.get('ema_26', price)
        
        if ema_12 > ema_26:
            ema_strength = min((ema_12 - ema_26) / ema_26 / 0.02, 1.0)
            signals.append(ema_strength)
        else:
            ema_strength = min((ema_26 - ema_12) / ema_26 / 0.02, 1.0)
            signals.append(-ema_strength)
        
        # Market Structure
        market_structure = indicators.get('market_structure', 0)
        if market_structure != 0:
            signals.append(market_structure * 0.5)
        
        return np.mean(signals) if signals else 0
    
    def _analyze_momentum_component(self, indicators: Dict) -> float:
        """Analysiert Momentum-Komponente"""
        
        signals = []
        
        # Multi-Period Momentum Alignment
        momentum_periods = [5, 10, 20]
        momentum_values = []
        
        for period in momentum_periods:
            mom = indicators.get(f'momentum_{period}', 0)
            momentum_values.append(mom)
        
        # Check for momentum alignment
        positive_momentum = sum(1 for m in momentum_values if m > 0.01)
        negative_momentum = sum(1 for m in momentum_values if m < -0.01)
        
        if positive_momentum >= 2:
            momentum_strength = np.mean([m for m in momentum_values if m > 0])
            signals.append(min(momentum_strength * 20, 1.0))
        elif negative_momentum >= 2:
            momentum_strength = np.mean([m for m in momentum_values if m < 0])
            signals.append(max(momentum_strength * 20, -1.0))
        
        # RSI Momentum
        rsi_14 = indicators.get('rsi_14', 50)
        
        if rsi_14 < self.signal_params['rsi_oversold'] and rsi_14 > 15:  # Oversold but not extreme
            signals.append((self.signal_params['rsi_oversold'] - rsi_14) / 20)
        elif rsi_14 > self.signal_params['rsi_overbought'] and rsi_14 < 85:  # Overbought but not extreme
            signals.append((self.signal_params['rsi_overbought'] - rsi_14) / 20)
        
        # MACD
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        if macd > macd_signal and macd_histogram > 0:
            signals.append(min(abs(macd_histogram) * 1000, 0.7))
        elif macd < macd_signal and macd_histogram < 0:
            signals.append(max(-abs(macd_histogram) * 1000, -0.7))
        
        return np.mean(signals) if signals else 0
    
    def _analyze_volatility_component(self, indicators: Dict, regime: str) -> float:
        """Analysiert Volatilitäts-Umgebung"""
        
        vol_20 = indicators.get('volatility_20', 0.3)
        
        # Base volatility assessment
        if vol_20 > self.signal_params['volatility_regime_threshold']:
            vol_penalty = -0.8  # High volatility penalty
        elif vol_20 < 0.015:
            vol_penalty = 0.3   # Low volatility bonus
        else:
            vol_penalty = 0.1   # Normal volatility
        
        # Regime-specific adjustments
        regime_adjustments = {
            'crisis': -0.9,
            'bear_volatile': -0.6,
            'bull_volatile': -0.3,
            'sideways_volatile': -0.4,
            'bull_trending': 0.4,
            'bear_trending': 0.2,
            'sideways_calm': 0.6
        }
        
        regime_adj = regime_adjustments.get(regime, 0)
        
        # Bollinger Band Squeeze (low volatility breakout setup)
        bb_width = indicators.get('bb_width', 0.1)
        if bb_width < self.signal_params['bb_squeeze_threshold']:
            squeeze_bonus = 0.3  # Squeeze setup bonus
        else:
            squeeze_bonus = 0
        
        return vol_penalty + regime_adj + squeeze_bonus
    
    def _analyze_volume_component(self, indicators: Dict) -> float:
        """Analysiert Volumen-Bestätigung"""
        
        signals = []
        
        # Multiple timeframe volume confirmation
        volume_ratios = []
        for period in [5, 10, 20]:
            vol_ratio = indicators.get(f'volume_ratio_{period}', 1.0)
            volume_ratios.append(vol_ratio)
        
        # High volume confirmation
        if any(vr > self.signal_params['volume_threshold'] for vr in volume_ratios):
            vol_strength = max(volume_ratios) - 1
            signals.append(min(vol_strength, 0.8))
        
        # Volume trend
        volume_trend = indicators.get('volume_trend', 0)
        if abs(volume_trend) > 0.3:  # Strong volume trend
            signals.append(volume_trend * 0.5)
        
        # Dry up volume (potential reversal)
        if all(vr < 0.8 for vr in volume_ratios):
            signals.append(-0.2)  # Volume dry-up penalty
        
        return np.mean(signals) if signals else 0
    
    def _analyze_mtf_component(self, mtf_analysis: Dict) -> float:
        """Analysiert Multi-Timeframe Alignment"""
        
        if not mtf_analysis or 'timeframe_bias' not in mtf_analysis:
            return 0
        
        bias = mtf_analysis['timeframe_bias']
        strength = mtf_analysis.get('strength', 0.5)
        confluence = mtf_analysis.get('confluence_score', 0)
        
        if bias == 'bullish':
            return strength * confluence
        elif bias == 'bearish':
            return -strength * confluence
        else:
            return 0
    
    def _analyze_pattern_component(self, indicators: Dict) -> float:
        """Analysiert Chart-Pattern"""
        
        signals = []
        
        # Support/Resistance Breakout
        resistance_dist = indicators.get('resistance_distance', 0)
        support_dist = indicators.get('support_distance', 0)
        
        # Breakout signals
        if resistance_dist < 0.005:  # Close to resistance
            signals.append(-0.3)  # Bearish near resistance
        elif resistance_dist < -0.01:  # Broke above resistance
            signals.append(0.6)  # Bullish breakout
        
        if support_dist < 0.005:  # Close to support
            signals.append(0.3)  # Bullish near support
        elif support_dist < -0.01:  # Broke below support
            signals.append(-0.6)  # Bearish breakdown
        
        # Bollinger Band Position
        bb_position = indicators.get('bb_position', 0.5)
        
        # Avoid extreme positions unless with strong momentum
        if bb_position > 0.95:
            signals.append(-0.2)  # Overbought
        elif bb_position < 0.05:
            signals.append(0.2)  # Oversold
        elif 0.3 < bb_position < 0.7:
            signals.append(0.1)  # Normal range bonus
        
        return np.mean(signals) if signals else 0
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Regime-spezifische Signal-Multiplikatoren"""
        
        multipliers = {
            'bull_trending': 1.2,    # Boost bull signals
            'bull_volatile': 0.8,    # Reduce in volatile bull
            'bear_trending': 1.1,    # Allow bear trades
            'bear_volatile': 0.6,    # Careful in volatile bear
            'sideways_calm': 1.0,    # Normal in calm sideways
            'sideways_volatile': 0.4, # Very careful in volatile sideways
            'crisis': 0.2,           # Almost no trades in crisis
            'insufficient_data': 0.5
        }
        
        return multipliers.get(regime, 0.8)
    
    def calculate_dynamic_position_size(self, signal_data: Dict, current_equity: float, 
                                      indicators: Dict) -> float:
        """Erweiterte dynamische Positionsgrößenberechnung"""
        
        if signal_data.get('direction') == 'hold':
            return 0
        
        # Base Parameters
        signal_strength = signal_data.get('strength', 0)
        confidence = signal_data.get('confidence', 0)
        regime = signal_data.get('regime', 'unknown')
        
        # === KELLY CRITERION ENHANCEMENT ===
        
        # Dynamic win rate estimation basierend auf recent performance
        recent_trades = self.performance_history[-20:] if self.performance_history else []
        if recent_trades:
            recent_wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            estimated_win_rate = recent_wins / len(recent_trades)
            # Smooth with historical expectation
            win_rate = 0.7 * estimated_win_rate + 0.3 * 0.58  # Mix with base 58%
        else:
            win_rate = 0.58  # Conservative estimate
        
        # Dynamic R/R based on volatility
        volatility = indicators.get('volatility_20', 0.03)
        atr_pct = indicators.get('atr_pct', 2.5)
        
        # Adaptive stop loss based on ATR
        if self.risk_management.get('stop_loss_atr_mult') and 'atr_pct' in indicators:
            dynamic_stop_loss = atr_pct * self.risk_management['stop_loss_atr_mult'] / 100
            dynamic_stop_loss = max(0.015, min(dynamic_stop_loss, 0.05))  # 1.5% - 5%
        else:
            dynamic_stop_loss = self.risk_management['stop_loss_base']
        
        # Adaptive take profit
        if self.risk_management.get('take_profit_dynamic'):
            # Higher volatility = higher targets
            vol_multiplier = 1 + (volatility - 0.03) * 2  # Scale with vol above 3%
            vol_multiplier = max(0.8, min(vol_multiplier, 2.0))
            dynamic_take_profit = self.risk_management['take_profit_base'] * vol_multiplier
        else:
            dynamic_take_profit = self.risk_management['take_profit_base']
        
        # Kelly Calculation
        avg_win = dynamic_take_profit
        avg_loss = dynamic_stop_loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.position_sizing['kelly_fraction']))
        
        # === SIGNAL QUALITY ADJUSTMENTS ===
        
        # Signal strength multiplier
        signal_multiplier = signal_strength ** 2  # Quadratic scaling rewards high-quality signals
        
        # Confidence boost
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        # Regime adjustment
        regime_multipliers = {
            'bull_trending': 1.2,
            'bull_volatile': 0.9,
            'bear_trending': 0.8,
            'bear_volatile': 0.6,
            'sideways_calm': 1.1,
            'sideways_volatile': 0.5,
            'crisis': 0.2
        }
        regime_multiplier = regime_multipliers.get(regime, 0.8)
        
        # === RISK MANAGEMENT OVERLAYS ===
        
        # Volatility scaling
        if self.position_sizing.get('volatility_scaling'):
            # Reduce size in high volatility
            vol_adjustment = max(0.5, 1 - (volatility - 0.03) * 5)  # Reduce if vol > 3%
        else:
            vol_adjustment = 1.0
        
        # Portfolio heat (current exposure)
        current_exposure = 0  # Would track actual positions in real implementation
        heat_adjustment = max(0.5, 1 - current_exposure * 2)  # Reduce with high exposure
        
        # Adaptive parameter overlay
        adaptive_sizing = self.adaptive_multipliers.get('position_size', 1.0)
        
        # === FINAL CALCULATION ===
        
        base_size = (kelly_fraction * signal_multiplier * confidence_multiplier * 
                    regime_multiplier * vol_adjustment * heat_adjustment * adaptive_sizing)
        
        # Apply absolute limits
        final_size = min(base_size, self.position_sizing['max_position_size'])
        final_size = max(final_size, self.position_sizing['min_position_size']) if final_size > 0 else 0
        
        # Risk per trade limit
        risk_per_trade = final_size * dynamic_stop_loss
        max_risk_size = self.position_sizing['risk_per_trade'] / dynamic_stop_loss
        final_size = min(final_size, max_risk_size)
        
        # Final minimum check
        if final_size < self.position_sizing['min_position_size']:
            return 0
        
        return final_size
    
    def calculate_dynamic_stops(self, entry_price: float, direction: str, 
                              indicators: Dict, signal_data: Dict) -> Tuple[float, float]:
        """Berechnet dynamische Stop Loss und Take Profit Levels"""
        
        # ATR-based stops
        if 'atr_pct' in indicators and self.risk_management.get('stop_loss_atr_mult'):
            atr_stop = indicators['atr_pct'] * self.risk_management['stop_loss_atr_mult'] / 100
            dynamic_stop = max(0.015, min(atr_stop, 0.05))
        else:
            dynamic_stop = self.risk_management['stop_loss_base']
        
        # Dynamic take profit based on volatility and regime
        volatility = indicators.get('volatility_20', 0.03)
        regime = signal_data.get('regime', 'unknown')
        
        if self.risk_management.get('take_profit_dynamic'):
            # Base target
            base_tp = self.risk_management['take_profit_base']
            
            # Volatility adjustment
            vol_mult = 1 + (volatility - 0.03) * 1.5
            vol_mult = max(0.8, min(vol_mult, 1.8))
            
            # Regime adjustment
            regime_tp_mult = {
                'bull_trending': 1.3,
                'bull_volatile': 1.1,
                'bear_trending': 1.0,
                'bear_volatile': 0.9,
                'sideways_calm': 1.2,
                'sideways_volatile': 0.8,
                'crisis': 0.7
            }
            
            regime_mult = regime_tp_mult.get(regime, 1.0)
            
            dynamic_tp = base_tp * vol_mult * regime_mult
            dynamic_tp = max(0.03, min(dynamic_tp, 0.12))  # 3% - 12%
        else:
            dynamic_tp = self.risk_management['take_profit_base']
        
        return dynamic_stop, dynamic_tp
    
    def should_exit_advanced(self, position_info: Dict, current_price: float, 
                           timestamp: datetime, indicators: Dict, 
                           signal_data: Dict) -> Tuple[bool, str]:
        """Erweiterte Exit-Logik"""
        
        entry_price = position_info['entry_price']
        direction = position_info['direction']
        entry_time = position_info['entry_time']
        
        # Calculate current PnL
        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Dynamic stops
        stop_loss_pct, take_profit_pct = self.calculate_dynamic_stops(
            entry_price, direction, indicators, signal_data
        )
        
        # === STOP LOSS EXITS ===
        
        # Basic stop loss
        if pnl_pct <= -stop_loss_pct:
            return True, "stop_loss"
        
        # === TAKE PROFIT EXITS ===
        
        # Basic take profit
        if pnl_pct >= take_profit_pct:
            return True, "take_profit"
        
        # === TRAILING STOP ===
        
        if pnl_pct >= take_profit_pct * self.risk_management['trailing_stop_trigger']:
            # Activate trailing stop
            trailing_stop = stop_loss_pct * 0.6  # Tighter trailing stop
            high_water_mark = position_info.get('max_profit', pnl_pct)
            
            # Update high water mark
            if pnl_pct > high_water_mark:
                position_info['max_profit'] = pnl_pct
                high_water_mark = pnl_pct
            
            # Check trailing stop
            if pnl_pct <= high_water_mark - trailing_stop:
                return True, "trailing_stop"
        
        # === TIME-BASED EXITS ===
        
        hours_held = (timestamp - entry_time).total_seconds() / 3600
        
        # Maximum holding period
        if hours_held > self.risk_management['max_holding_hours']:
            return True, "time_exit"
        
        # === SIGNAL REVERSAL EXITS ===
        
        # Check for signal reversal
        current_signal = signal_data.get('direction', 'hold')
        if direction == 'long' and current_signal == 'sell':
            return True, "signal_reversal"
        elif direction == 'short' and current_signal == 'buy':
            return True, "signal_reversal"
        
        # === REGIME CHANGE EXITS ===
        
        regime = signal_data.get('regime', 'unknown')
        if regime == 'crisis':  # Emergency exit in crisis
            return True, "regime_crisis"
        
        # === MOMENTUM DIVERGENCE ===
        
        # Check if momentum is diverging from position
        momentum_5 = indicators.get('momentum_5', 0)
        if direction == 'long' and momentum_5 < -0.02:  # Strong negative momentum
            return True, "momentum_divergence"
        elif direction == 'short' and momentum_5 > 0.02:  # Strong positive momentum
            return True, "momentum_divergence"
        
        return False, "hold"
    
    def update_adaptive_parameters(self, trade_results: List[Dict]):
        """Adaptive Parameter Update basierend auf Performance"""
        
        if not trade_results or len(trade_results) < 5:
            return
        
        recent_trades = trade_results[-20:] if len(trade_results) >= 20 else trade_results
        
        # Performance Metrics
        win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
        avg_return = np.mean([t.get('return_pct', 0) for t in recent_trades])
        
        # Signal Threshold Adjustment
        if win_rate < 0.45:  # Too many losing trades
            self.adaptive_multipliers['signal_threshold'] *= 1.05  # Be more selective
            self.adaptive_multipliers['signal_threshold'] = min(1.5, self.adaptive_multipliers['signal_threshold'])
        elif win_rate > 0.70:  # Very high win rate, can be less selective
            self.adaptive_multipliers['signal_threshold'] *= 0.98
            self.adaptive_multipliers['signal_threshold'] = max(0.8, self.adaptive_multipliers['signal_threshold'])
        
        # Position Sizing Adjustment
        if avg_return < -0.015:  # Poor performance
            self.adaptive_multipliers['position_size'] *= 0.9  # Reduce size
            self.adaptive_multipliers['position_size'] = max(0.3, self.adaptive_multipliers['position_size'])
        elif avg_return > 0.03:  # Good performance
            self.adaptive_multipliers['position_size'] *= 1.05  # Increase size
            self.adaptive_multipliers['position_size'] = min(1.5, self.adaptive_multipliers['position_size'])
        
        # Store performance
        self.performance_history.extend(recent_trades)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        logger.debug(f"Adaptive parameters updated: threshold={self.adaptive_multipliers['signal_threshold']:.3f}, "
                    f"sizing={self.adaptive_multipliers['position_size']:.3f}")


def optimize_strategy_iteratively(initial_config: Dict = None, 
                                max_iterations: int = 20) -> Dict[str, Any]:
    """Iterative Strategie-Optimierung bis Tier 1 erreicht wird"""
    
    print("🎯 TIER 1 STRATEGY OPTIMIZATION")
    print("=" * 80)
    print("Target: Sharpe >1.5, Return >25%, Win Rate >55%, Max DD <10%\n")
    
    targets = Tier1Targets()
    
    # Basis-Konfiguration
    base_config = initial_config or {
        'position_sizing': {
            'max_position_size': 0.10,
            'kelly_fraction': 0.20,
            'risk_per_trade': 0.02
        },
        'signal_params': {
            'min_signal_strength': 0.6,
            'volume_threshold': 1.2,
            'volatility_regime_threshold': 0.04
        },
        'risk_management': {
            'stop_loss_base': 0.025,
            'take_profit_base': 0.055,
            'take_profit_dynamic': True
        },
        'advanced_features': {
            'multi_timeframe': True,
            'regime_detection': True,
            'adaptive_parameters': True
        }
    }
    
    best_config = base_config.copy()
    best_metrics = None
    best_score = 0
    
    optimization_history = []
    
    # Generiere Testdaten einmal
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-01-01", "2024-01-01")
    
    for iteration in range(max_iterations):
        print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
        print("-" * 60)
        
        # Teste aktuelle Konfiguration
        current_config = best_config.copy() if iteration > 0 else base_config
        
        # Füge Random-Walk für Exploration hinzu
        if iteration > 0:
            current_config = mutate_config(current_config, iteration / max_iterations)
        
        # Führe Backtest durch
        try:
            backtester = create_advanced_backtester(current_config, market_data)
            results = backtester.run_backtest(market_data)
            metrics = results['metrics']
            
            # Bewerte Performance
            score = calculate_tier1_score(metrics, targets)
            
            print(f"📊 Results:")
            print(f"   Annual Return: {metrics.get('annual_return', 0):+.1%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   Tier 1 Score: {score:.3f}")
            
            # Speichere wenn besser
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_config = current_config.copy()
                print(f"   🎉 NEW BEST SCORE: {score:.3f}")
                
                # Check if Tier 1 achieved
                if meets_tier1_criteria(metrics, targets):
                    print(f"   ✅ TIER 1 ACHIEVED!")
                    break
            
            optimization_history.append({
                'iteration': iteration + 1,
                'config': current_config,
                'metrics': metrics,
                'score': score
            })
            
        except Exception as e:
            print(f"   ❌ Error in iteration {iteration + 1}: {e}")
            continue
    
    # Final Results
    print(f"\n🏆 OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    if best_metrics:
        print(f"🎯 BEST RESULTS:")
        print(f"   Annual Return: {best_metrics.get('annual_return', 0):+.1%}")
        print(f"   Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Win Rate: {best_metrics.get('win_rate', 0):.1%}")
        print(f"   Max Drawdown: {best_metrics.get('max_drawdown', 0):.1%}")
        print(f"   Profit Factor: {best_metrics.get('profit_factor', 0):.2f}")
        print(f"   Total Trades: {best_metrics.get('total_trades', 0)}")
        print(f"   Final Score: {best_score:.3f}")
        
        # Check final status
        if meets_tier1_criteria(best_metrics, targets):
            print(f"\n🎉 🎉 TIER 1 PERFORMANCE ACHIEVED! 🎉 🎉")
            print("✅ Ready for paper trading!")
        else:
            print(f"\n⚠️ Tier 1 not achieved - further optimization needed")
            print_tier1_gaps(best_metrics, targets)
    
    # Export results
    results_export = {
        'optimization_date': datetime.now().isoformat(),
        'iterations_completed': len(optimization_history),
        'tier1_achieved': meets_tier1_criteria(best_metrics, targets) if best_metrics else False,
        'best_config': best_config,
        'best_metrics': best_metrics,
        'best_score': best_score,
        'targets': {
            'min_annual_return': targets.min_annual_return,
            'min_sharpe_ratio': targets.min_sharpe_ratio,
            'min_win_rate': targets.min_win_rate,
            'max_drawdown': targets.max_drawdown
        },
        'optimization_history': optimization_history[-10:]  # Last 10 iterations
    }
    
    filename = f"tier1_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results_export, f, indent=2, default=str)
    
    print(f"\n💾 Results exported to: {filename}")
    
    return results_export


def create_advanced_backtester(config: Dict, market_data: pd.DataFrame) -> 'AdvancedBacktester':
    """Erstellt Advanced Backtester mit gegebener Konfiguration"""
    
    class AdvancedBacktester(RealisticBacktester):
        def __init__(self, config):
            super().__init__(initial_capital=10000, symbol="BTC/USDT")
            self.strategy = AdvancedTradingStrategy(config)
            self.positions_info = {}  # Track position details
        
        def run_backtest(self, market_data: pd.DataFrame) -> Dict[str, Any]:
            """Advanced backtest implementation"""
            
            logger.info("Starting Advanced Tier 1 Backtest...")
            
            signals_generated = 0
            high_quality_signals = 0
            
            for i, (timestamp, row) in enumerate(market_data.iterrows()):
                
                if (i + 1) % 2000 == 0:
                    progress = (i + 1) / len(market_data) * 100
                    current_equity = self.get_current_equity(row['close'])
                    logger.info(f"Progress: {progress:.1f}% - Equity: ${current_equity:,.0f}")
                
                # Skip warmup
                if i < 200:
                    continue
                
                # Market info
                market_info = {
                    'volatility': row.get('volatility', 0.03),
                    'volume_usd': row.get('volume_usd', 1000000)
                }
                
                # Check exits first
                for pos_id in list(self.positions_info.keys()):
                    pos_info = self.positions_info[pos_id]
                    
                    # Get current indicators for exit decision
                    indicators = self.strategy.calculate_advanced_indicators(market_data.iloc[:i+1])
                    
                    # Generate signal for exit decision
                    signal_data = self.strategy.generate_advanced_signal(market_data, i, timestamp)
                    
                    should_exit, reason = self.strategy.should_exit_advanced(
                        pos_info, row['close'], timestamp, indicators, signal_data
                    )
                    
                    if should_exit:
                        # Find corresponding position
                        for position in self.positions:
                            if position.symbol == self.symbol:  # Assuming one position
                                self._close_position(position, timestamp, row['close'], market_info, reason)
                                del self.positions_info[pos_id]
                                break
                
                # Generate new signals
                if len(self.positions) == 0:  # Only if no open positions
                    signal_data = self.strategy.generate_advanced_signal(market_data, i, timestamp)
                    
                    if signal_data['direction'] != 'hold':
                        signals_generated += 1
                        
                        if signal_data['confidence'] >= 0.6:  # High quality threshold
                            high_quality_signals += 1
                            
                            # Calculate position size
                            current_equity = self.get_current_equity(row['close'])
                            indicators = signal_data.get('indicators', {})
                            
                            position_size = self.strategy.calculate_dynamic_position_size(
                                signal_data, current_equity, indicators
                            )
                            
                            if position_size > 0:
                                order_size_usd = current_equity * position_size
                                
                                # Check execution
                                can_execute, reason = self.exchange.can_execute_order(
                                    order_size_usd, self.capital, market_info['volume_usd']
                                )
                                
                                if can_execute:
                                    self._open_advanced_position(signal_data, timestamp, row['close'], 
                                                               order_size_usd, market_info, indicators)
                                else:
                                    self.rejected_orders.append({
                                        'timestamp': timestamp,
                                        'reason': reason,
                                        'signal': signal_data,
                                        'attempted_size': order_size_usd
                                    })
                
                # Update equity
                self._update_equity_history(timestamp, row['close'])
            
            # Finalize
            if self.positions:
                final_price = market_data['close'].iloc[-1]
                final_timestamp = market_data.index[-1]
                final_market_info = {
                    'volatility': market_data.get('volatility', pd.Series([0.03])).iloc[-1],
                    'volume_usd': market_data.get('volume_usd', pd.Series([1000000])).iloc[-1]
                }
                
                for position in self.positions.copy():
                    self._close_position(position, final_timestamp, final_price, 
                                       final_market_info, "backtest_end")
            
            # Update adaptive parameters
            if self.trades:
                completed_trades = [self._trade_to_dict(t) for t in self.trades if hasattr(t, 'net_pnl')]
                self.strategy.update_adaptive_parameters(completed_trades)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            logger.info(f"Advanced backtest completed!")
            logger.info(f"Signals: {signals_generated}, HQ: {high_quality_signals}, Trades: {len(self.trades)}")
            
            return {
                'metrics': metrics,
                'trades': [self._trade_to_dict(t) for t in self.trades],
                'equity_history': self.equity_history[-1000:],  # Last 1000 points
                'signals_generated': signals_generated,
                'high_quality_signals': high_quality_signals,
                'strategy_state': {
                    'adaptive_multipliers': self.strategy.adaptive_multipliers,
                    'regime_state': self.strategy.regime_state
                }
            }
        
        def _open_advanced_position(self, signal_data: Dict, timestamp: datetime, price: float,
                                  order_size_usd: float, market_info: Dict, indicators: Dict):
            """Opens advanced position with enhanced tracking"""
            
            # Execute trade
            execution = self.exchange.execute_trade(
                timestamp, self.symbol, signal_data['direction'],
                order_size_usd, price, market_info
            )
            
            # Calculate dynamic stops
            stop_loss_pct, take_profit_pct = self.strategy.calculate_dynamic_stops(
                execution.executed_price, signal_data['direction'], indicators, signal_data
            )
            
            # Create position
            from realistic_crypto_backtest import Position
            position_size_coins = order_size_usd / execution.executed_price
            if signal_data['direction'] == 'sell':
                position_size_coins = -position_size_coins
            
            position = Position(
                symbol=self.symbol,
                size=position_size_coins,
                entry_price=execution.executed_price,
                entry_time=timestamp,
                unrealized_pnl=0,
                stop_loss=stop_loss_pct,
                take_profit=take_profit_pct,
                direction='long' if signal_data['direction'] == 'buy' else 'short'
            )
            
            self.positions.append(position)
            
            # Enhanced position info for advanced exits
            pos_id = f"pos_{len(self.trades)}"
            self.positions_info[pos_id] = {
                'entry_price': execution.executed_price,
                'entry_time': timestamp,
                'direction': position.direction,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'signal_data': signal_data,
                'max_profit': 0
            }
            
            # Update capital and tracking
            self.capital -= execution.total_cost
            self.total_fees_paid += execution.commission
            self.total_slippage_cost += execution.slippage * order_size_usd
            
            # Update strategy state
            self.strategy.daily_trades += 1
            
            logger.debug(f"Advanced position opened: {position.direction} ${order_size_usd:,.0f} @ ${execution.executed_price:,.2f}")
    
    return AdvancedBacktester(config)


def mutate_config(base_config: Dict, progress: float) -> Dict:
    """Mutiert Konfiguration für Exploration"""
    
    config = base_config.copy()
    
    # Mutation ranges (smaller mutations as we progress)
    mutation_factor = 0.2 * (1 - progress)  # Reduce mutations over time
    
    # Position Sizing Mutations - preserve all keys
    pos_sizing = config.get('position_sizing', {}).copy()
    pos_sizing['max_position_size'] = max(0.05, min(0.20, 
        pos_sizing.get('max_position_size', 0.10) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    pos_sizing['kelly_fraction'] = max(0.1, min(0.4,
        pos_sizing.get('kelly_fraction', 0.20) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    pos_sizing['risk_per_trade'] = max(0.01, min(0.04,
        pos_sizing.get('risk_per_trade', 0.02) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    # Preserve other position sizing parameters
    pos_sizing['min_position_size'] = pos_sizing.get('min_position_size', 0.03)
    pos_sizing['volatility_scaling'] = pos_sizing.get('volatility_scaling', True)
    
    # Signal Parameter Mutations - preserve all keys
    sig_params = config.get('signal_params', {}).copy()
    sig_params['min_signal_strength'] = max(0.4, min(0.8,
        sig_params.get('min_signal_strength', 0.6) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    sig_params['volume_threshold'] = max(1.1, min(2.0,
        sig_params.get('volume_threshold', 1.2) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    # Preserve other signal parameters
    sig_params['momentum_periods'] = sig_params.get('momentum_periods', [5, 10, 20, 50])
    sig_params['trend_lookback'] = sig_params.get('trend_lookback', 50)
    sig_params['volatility_regime_threshold'] = sig_params.get('volatility_regime_threshold', 0.035)
    sig_params['rsi_oversold'] = sig_params.get('rsi_oversold', 25)
    sig_params['rsi_overbought'] = sig_params.get('rsi_overbought', 75)
    sig_params['bb_squeeze_threshold'] = sig_params.get('bb_squeeze_threshold', 0.02)
    
    # Risk Management Mutations - preserve all keys
    risk_mgmt = config.get('risk_management', {}).copy()
    risk_mgmt['stop_loss_base'] = max(0.015, min(0.05,
        risk_mgmt.get('stop_loss_base', 0.025) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    risk_mgmt['take_profit_base'] = max(0.03, min(0.10,
        risk_mgmt.get('take_profit_base', 0.055) * np.random.uniform(1-mutation_factor, 1+mutation_factor)))
    # Preserve other risk management parameters
    risk_mgmt['daily_loss_limit'] = risk_mgmt.get('daily_loss_limit', 0.03)
    risk_mgmt['take_profit_dynamic'] = risk_mgmt.get('take_profit_dynamic', True)
    risk_mgmt['max_holding_hours'] = risk_mgmt.get('max_holding_hours', 72)
    risk_mgmt['stop_loss_atr_mult'] = risk_mgmt.get('stop_loss_atr_mult', 1.5)
    risk_mgmt['trailing_stop_trigger'] = risk_mgmt.get('trailing_stop_trigger', 0.4)
    risk_mgmt['max_correlated_positions'] = risk_mgmt.get('max_correlated_positions', 1)
    
    # Update config
    config['position_sizing'] = pos_sizing
    config['signal_params'] = sig_params
    config['risk_management'] = risk_mgmt
    
    return config


def calculate_tier1_score(metrics: Dict, targets: Tier1Targets) -> float:
    """Berechnet Tier 1 Score (0-1, 1 = perfect)"""
    
    if not metrics:
        return 0
    
    # Individual component scores
    annual_return = metrics.get('annual_return', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    win_rate = metrics.get('win_rate', 0)
    max_drawdown = metrics.get('max_drawdown', 1)  # Default high
    profit_factor = metrics.get('profit_factor', 0)
    total_trades = metrics.get('total_trades', 0)
    
    # Score components (0-1 each)
    return_score = min(1.0, max(0, annual_return / targets.min_annual_return))
    sharpe_score = min(1.0, max(0, sharpe_ratio / targets.min_sharpe_ratio))
    win_rate_score = min(1.0, max(0, win_rate / targets.min_win_rate))
    drawdown_score = min(1.0, max(0, (targets.max_drawdown - max_drawdown) / targets.max_drawdown)) if max_drawdown > 0 else 1.0
    pf_score = min(1.0, max(0, profit_factor / targets.min_profit_factor))
    trades_score = min(1.0, max(0, total_trades / targets.min_trades_per_year))
    
    # Weighted combination
    weights = {
        'return': 0.25,
        'sharpe': 0.25,
        'win_rate': 0.15,
        'drawdown': 0.15,
        'profit_factor': 0.10,
        'trades': 0.10
    }
    
    total_score = (return_score * weights['return'] +
                  sharpe_score * weights['sharpe'] +
                  win_rate_score * weights['win_rate'] +
                  drawdown_score * weights['drawdown'] +
                  pf_score * weights['profit_factor'] +
                  trades_score * weights['trades'])
    
    return total_score


def meets_tier1_criteria(metrics: Dict, targets: Tier1Targets) -> bool:
    """Prüft ob Tier 1 Kriterien erfüllt sind"""
    
    if not metrics:
        return False
    
    return (
        metrics.get('annual_return', 0) >= targets.min_annual_return and
        metrics.get('sharpe_ratio', 0) >= targets.min_sharpe_ratio and
        metrics.get('win_rate', 0) >= targets.min_win_rate and
        metrics.get('max_drawdown', 1) <= targets.max_drawdown and
        metrics.get('profit_factor', 0) >= targets.min_profit_factor and
        metrics.get('total_trades', 0) >= targets.min_trades_per_year
    )


def print_tier1_gaps(metrics: Dict, targets: Tier1Targets):
    """Zeigt Lücken zu Tier 1 Zielen"""
    
    print(f"\n🎯 TIER 1 GAPS ANALYSIS:")
    
    gaps = []
    
    annual_return = metrics.get('annual_return', 0)
    if annual_return < targets.min_annual_return:
        gap = targets.min_annual_return - annual_return
        gaps.append(f"   Annual Return: {gap:+.1%} needed (current: {annual_return:.1%})")
    
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    if sharpe_ratio < targets.min_sharpe_ratio:
        gap = targets.min_sharpe_ratio - sharpe_ratio
        gaps.append(f"   Sharpe Ratio: {gap:+.2f} needed (current: {sharpe_ratio:.2f})")
    
    win_rate = metrics.get('win_rate', 0)
    if win_rate < targets.min_win_rate:
        gap = targets.min_win_rate - win_rate
        gaps.append(f"   Win Rate: {gap:+.1%} needed (current: {win_rate:.1%})")
    
    max_drawdown = metrics.get('max_drawdown', 1)
    if max_drawdown > targets.max_drawdown:
        gap = max_drawdown - targets.max_drawdown
        gaps.append(f"   Max Drawdown: {gap:.1%} too high (current: {max_drawdown:.1%})")
    
    profit_factor = metrics.get('profit_factor', 0)
    if profit_factor < targets.min_profit_factor:
        gap = targets.min_profit_factor - profit_factor
        gaps.append(f"   Profit Factor: {gap:+.2f} needed (current: {profit_factor:.2f})")
    
    total_trades = metrics.get('total_trades', 0)
    if total_trades < targets.min_trades_per_year:
        gap = targets.min_trades_per_year - total_trades
        gaps.append(f"   Total Trades: {gap:+.0f} needed (current: {total_trades})")
    
    if gaps:
        for gap in gaps:
            print(gap)
    else:
        print("   ✅ All criteria met!")


if __name__ == "__main__":
    # Run iterative optimization
    results = optimize_strategy_iteratively(max_iterations=15)
    
    if results['tier1_achieved']:
        print(f"\n🚀 TIER 1 STRATEGY READY FOR DEPLOYMENT!")
    else:
        print(f"\n🔧 Continue optimization with best configuration...")