#!/usr/bin/env python3
"""
Enhanced Institutional Strategy - Schritt 1: Signal-Optimierung
Erhöhung der Trade-Frequenz bei Beibehaltung institutioneller Standards

Strategy Name: "Enhanced Institutional BTC Pro"
Version: 1.1 Signal-Enhanced
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedInstitutionalStrategy:
    """
    Enhanced Institutional Strategy - Schritt 1 Optimierung
    
    Name: "Enhanced Institutional BTC Pro"
    
    Signal-Optimierungen:
    1. Reduzierte Signal-Schwelle: 70% → 62%
    2. Erweiterte Signal-Quellen: 6 → 8 Strategien
    3. Adaptive Timeframes: Dynamische Anpassung
    4. Multi-Asset Ready: Vorbereitung für Portfolio-Diversifikation
    5. Enhanced Quality Scoring: Verbessertes Bewertungssystem
    """
    
    def __init__(self, initial_capital: float = 300000.0):
        self.strategy_name = "Enhanced Institutional BTC Pro"
        self.strategy_version = "1.1 Signal-Enhanced"
        self.risk_profile = "Conservative-Aggressive Plus"
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_position = 0.0
        self.cash_balance = initial_capital
        
        # Enhanced Signal Parameters
        self.trading_fee = 0.001
        self.min_trade_size = 0.001
        self.max_position_size = 0.45  # Leicht erhöht von 40% zu 45%
        self.max_drawdown_limit = 0.25
        self.min_signal_strength = 0.62  # Reduziert von 70% zu 62%
        
        # Enhanced Risk Management
        self.position_size_multiplier = 0.85  # Leicht erhöht von 0.8
        self.emergency_stop_enabled = True
        self.monthly_var_limit = 0.10  # Leicht erhöht von 8% zu 10%
        
        # Signal Enhancement
        self.adaptive_timeframes = True
        self.multi_strategy_weights = True
        self.quality_boost_enabled = True
        
        # Performance Tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.signal_performance_tracking = {}
        self.alerts = []
        
        # Dashboard Integration
        self.last_signal_time = None
        self.current_phase = "initialization"
        self.strategy_status = "active"
        self.signal_stats = {"generated": 0, "executed": 0, "quality_avg": 0}
        
        logger.info(f"{self.strategy_name} v{self.strategy_version} initialisiert")
        logger.info(f"Signal-Threshold: {self.min_signal_strength:.0%} | Max Position: {self.max_position_size:.0%}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Strategy-Info für Dashboard"""
        return {
            'name': self.strategy_name,
            'version': self.strategy_version,
            'risk_profile': self.risk_profile,
            'status': self.strategy_status,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_position_size': self.max_position_size,
            'min_signal_strength': self.min_signal_strength,
            'current_phase': self.current_phase,
            'last_signal_time': self.last_signal_time,
            'signal_stats': self.signal_stats,
            'alerts': self.alerts[-5:] if self.alerts else []
        }
    
    def generate_enhanced_signals(self, prices: List[Dict]) -> List[Dict]:
        """
        SCHRITT 1: Enhanced Signal Generation mit erhöhter Frequenz
        """
        signals = []
        df = pd.DataFrame(prices)
        
        # Extended technical indicators for more signal opportunities
        for window in [3, 5, 8, 10, 13, 20, 34, 50, 89, 144, 200]:  # Fibonacci numbers
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
        
        for span in [8, 12, 21, 26, 34, 50, 89]:
            df[f'ema_{span}'] = df['price'].ewm(span=span).mean()
        
        # Multiple MACD timeframes
        df['macd_fast'] = df['ema_8'] - df['ema_21']  # Faster MACD
        df['macd_standard'] = df['ema_12'] - df['ema_26']  # Standard MACD
        df['macd_slow'] = df['ema_21'] - df['ema_50']  # Slower MACD
        
        for macd in ['macd_fast', 'macd_standard', 'macd_slow']:
            df[f'{macd}_signal'] = df[macd].ewm(span=9).mean()
            df[f'{macd}_histogram'] = df[macd] - df[f'{macd}_signal']
        
        # Enhanced volatility analysis
        for window in [5, 8, 13, 20, 34, 50]:
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std()
            df[f'vol_zscore_{window}'] = ((df[f'volatility_{window}d'] - 
                                         df[f'volatility_{window}d'].rolling(50).mean()) / 
                                        df[f'volatility_{window}d'].rolling(50).std())
        
        # Multiple RSI periods for different timeframes
        for period in [9, 14, 21, 34, 50]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Enhanced momentum indicators
        for period in [3, 5, 8, 13, 20, 34, 50, 89]:
            df[f'momentum_{period}d'] = df['price'] / df['price'].shift(period)
            df[f'roc_{period}d'] = df['price'].pct_change(period)  # Rate of Change
        
        # Adaptive volume analysis
        for window in [5, 10, 20, 34, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            df[f'volume_zscore_{window}'] = ((df['volume'] - df[f'volume_sma_{window}']) / 
                                           df['volume'].rolling(window).std())
        
        # Multiple Bollinger Band timeframes
        for period in [13, 20, 34, 50]:
            df[f'bb_middle_{period}'] = df['price'].rolling(period).mean()
            bb_std = df['price'].rolling(period).std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
            df[f'bb_position_{period}'] = ((df['price'] - df[f'bb_lower_{period}']) / 
                                         (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']))
        
        # Adaptive analysis window (reduced from 200 to 144 for more signals)
        for i in range(144, len(df)):
            current = df.iloc[i]
            
            # === ENHANCED SIGNAL STRATEGIES ===
            
            # 1. MULTI-TIMEFRAME TREND STRATEGY
            trend_signals = []
            
            # Short-term trend
            if (current['sma_8'] > current['sma_13'] and 
                current['sma_13'] > current['sma_20'] and
                current['price'] > current['sma_8'] * 1.01):
                trend_signals.append(('long', 0.6))
            
            # Medium-term trend
            if (current['sma_20'] > current['sma_34'] and 
                current['sma_34'] > current['sma_50'] and
                current['price'] > current['sma_20'] * 1.015):
                trend_signals.append(('long', 0.7))
            
            # Long-term trend
            if (current['sma_50'] > current['sma_89'] and 
                current['sma_89'] > current['sma_144'] and
                current['price'] > current['sma_50'] * 1.02):
                trend_signals.append(('long', 0.8))
            
            # Aggregate trend signal
            if trend_signals:
                trend_direction = 'long'
                trend_signal = np.mean([signal for _, signal in trend_signals])
                trend_confidence = len(trend_signals) / 3  # Max 3 timeframes
            else:
                trend_direction = None
                trend_signal = 0
                trend_confidence = 0
            
            # 2. ENHANCED MOMENTUM STRATEGY
            momentum_signals = []
            
            # Fast momentum
            if current['momentum_3d'] > 1.01 and current['momentum_5d'] > 1.02:
                momentum_signals.append(('long', 0.5))
            
            # Medium momentum
            if current['momentum_13d'] > 1.03 and current['momentum_20d'] > 1.05:
                momentum_signals.append(('long', 0.6))
            
            # Slow momentum
            if current['momentum_34d'] > 1.08 and current['momentum_50d'] > 1.12:
                momentum_signals.append(('long', 0.7))
            
            # MACD confirmation
            macd_bullish = (current['macd_fast'] > current['macd_fast_signal'] and 
                          current['macd_standard'] > current['macd_standard_signal'])
            if macd_bullish:
                momentum_signals.append(('long', 0.6))
            
            if momentum_signals:
                momentum_direction = 'long'
                momentum_signal = np.mean([signal for _, signal in momentum_signals])
                momentum_confidence = len(momentum_signals) / 4
            else:
                momentum_direction = None
                momentum_signal = 0
                momentum_confidence = 0
            
            # 3. MULTI-TIMEFRAME MEAN REVERSION
            mean_reversion_signals = []
            
            # RSI multi-timeframe oversold
            rsi_oversold_short = current['rsi_9'] < 30 and current['rsi_14'] < 35
            rsi_oversold_medium = current['rsi_21'] < 40 and current['rsi_34'] < 45
            bb_oversold = current['bb_position_20'] < 0.1 and current['bb_position_34'] < 0.15
            
            if rsi_oversold_short and bb_oversold:
                mean_reversion_signals.append(('long', 0.7))
            elif rsi_oversold_medium and bb_oversold:
                mean_reversion_signals.append(('long', 0.6))
            
            # Volatility spike reversion
            if (current['vol_zscore_13'] > 2 and current['vol_zscore_20'] > 1.5 and
                current['phase'] in ['crypto_winter', 'post_etf_consolidation', 'summer_range']):
                mean_reversion_signals.append(('long', 0.5))
            
            if mean_reversion_signals:
                mean_reversion_direction = 'long'
                mean_reversion_signal = np.mean([signal for _, signal in mean_reversion_signals])
            else:
                mean_reversion_direction = None
                mean_reversion_signal = 0
            
            # 4. ENHANCED INSTITUTIONAL FLOW
            institutional_signals = []
            
            # Strong institutional buying
            if (current['institutional_flow'] > 0.15 and 
                current['whale_activity'] > 0.6 and
                current['volume_ratio_20'] > 1.2):
                institutional_signals.append(('long', 0.7))
            
            # Moderate institutional activity with volume confirmation
            if (current['institutional_flow'] > 0.1 and 
                current['volume_zscore_20'] > 1):
                institutional_signals.append(('long', 0.5))
            
            if institutional_signals:
                institutional_direction = 'long'
                institutional_signal = np.mean([signal for _, signal in institutional_signals])
            else:
                institutional_direction = None
                institutional_signal = 0
            
            # 5. ENHANCED SENTIMENT STRATEGY
            sentiment_signals = []
            
            # Strong contrarian signals
            if (current['sentiment'] < -0.2 and current['funding_rate'] < 0 and
                current['put_call_ratio'] > 1.2):
                sentiment_signals.append(('long', 0.6))
            
            # Moderate sentiment extremes
            if (current['sentiment'] < -0.1 and 
                current['phase'] in ['crypto_winter', 'post_etf_consolidation']):
                sentiment_signals.append(('long', 0.4))
            
            if sentiment_signals:
                sentiment_direction = 'long'
                sentiment_signal = np.mean([signal for _, signal in sentiment_signals])
            else:
                sentiment_direction = None
                sentiment_signal = 0
            
            # 6. VOLATILITY REGIME STRATEGY
            vol_signals = []
            
            # Low volatility expansion signal
            if (current['volatility_20d'] < 0.03 and 
                current['vol_zscore_20'] < -1 and
                current['volatility_regime'] < 0.035):
                vol_signals.append(('long', 0.5))
            
            # Volatility normalization after spike
            if (current['vol_zscore_13'] > 1.5 and current['vol_zscore_5'] < 1):
                vol_signals.append(('long', 0.4))
            
            if vol_signals:
                vol_direction = 'long'
                vol_signal = np.mean([signal for _, signal in vol_signals])
            else:
                vol_direction = None
                vol_signal = 0
            
            # 7. BREAKOUT STRATEGY (NEW)
            breakout_signals = []
            
            # Bollinger Band breakouts
            bb_breakout_up = (current['price'] > current['bb_upper_20'] and 
                            current['volume_ratio_10'] > 1.5)
            if bb_breakout_up:
                breakout_signals.append(('long', 0.6))
            
            # Moving average breakouts
            ma_breakout = (current['price'] > current['sma_50'] * 1.02 and 
                         current['volume_ratio_20'] > 1.3)
            if ma_breakout:
                breakout_signals.append(('long', 0.5))
            
            if breakout_signals:
                breakout_direction = 'long'
                breakout_signal = np.mean([signal for _, signal in breakout_signals])
            else:
                breakout_direction = None
                breakout_signal = 0
            
            # 8. MACRO MOMENTUM STRATEGY (NEW)
            macro_signals = []
            
            # Macro sentiment alignment
            if (current['macro_sentiment'] > 0.1 and 
                current['phase'] in ['etf_approval', 'election_rally', 'pre_election']):
                macro_signals.append(('long', 0.5))
            
            # Exchange flow signals
            if current.get('exchange_flow', 0) < -0.1:  # Outflows (bullish)
                macro_signals.append(('long', 0.4))
            
            if macro_signals:
                macro_direction = 'long'
                macro_signal = np.mean([signal for _, signal in macro_signals])
            else:
                macro_direction = None
                macro_signal = 0
            
            # === ENHANCED ENSEMBLE DECISION ===
            
            strategy_votes = []
            strategy_weights = []
            strategy_confidences = []
            
            # Dynamic weighting based on market phase
            if current['phase'] in ['etf_approval', 'election_rally']:
                # Favor momentum and breakouts in strong trends
                base_weights = {
                    'trend': 0.25, 'momentum': 0.25, 'breakout': 0.15, 'institutional': 0.15,
                    'mean_reversion': 0.05, 'sentiment': 0.05, 'volatility': 0.05, 'macro': 0.05
                }
            elif current['phase'] in ['crypto_winter', 'summer_range']:
                # Favor mean reversion and sentiment in ranging markets
                base_weights = {
                    'mean_reversion': 0.25, 'sentiment': 0.20, 'institutional': 0.20,
                    'trend': 0.15, 'momentum': 0.10, 'volatility': 0.05, 'breakout': 0.03, 'macro': 0.02
                }
            else:
                # Balanced approach
                base_weights = {
                    'trend': 0.22, 'momentum': 0.20, 'institutional': 0.18, 'mean_reversion': 0.15,
                    'breakout': 0.10, 'sentiment': 0.08, 'volatility': 0.04, 'macro': 0.03
                }
            
            # Add votes with enhanced criteria
            strategies = [
                ('trend', trend_direction, trend_signal, trend_confidence),
                ('momentum', momentum_direction, momentum_signal, momentum_confidence),
                ('mean_reversion', mean_reversion_direction, mean_reversion_signal, 1.0),
                ('institutional', institutional_direction, institutional_signal, 1.0),
                ('sentiment', sentiment_direction, sentiment_signal, 1.0),
                ('volatility', vol_direction, vol_signal, 1.0),
                ('breakout', breakout_direction, breakout_signal, 1.0),
                ('macro', macro_direction, macro_signal, 1.0)
            ]
            
            for strategy_name, direction, signal, confidence in strategies:
                if direction and signal > 0.4:  # Lower threshold for more signals
                    strategy_votes.append((direction, signal))
                    strategy_weights.append(base_weights[strategy_name] * confidence)
                    strategy_confidences.append(confidence)
            
            # Enhanced ensemble decision with lower threshold
            if len(strategy_votes) >= 2:  # Need at least 2 strategies
                total_weight = sum(strategy_weights)
                if total_weight > 0:
                    strategy_weights = [w / total_weight for w in strategy_weights]
                    
                    long_score = sum(weight * signal for (direction, signal), weight in 
                                   zip(strategy_votes, strategy_weights) if direction == 'long')
                    short_score = sum(weight * signal for (direction, signal), weight in 
                                    zip(strategy_votes, strategy_weights) if direction == 'short')
                    
                    final_signal_strength = max(long_score, short_score)
                    final_direction = 'long' if long_score > short_score else 'short'
                    
                    # Enhanced quality scoring
                    strategy_diversity = len(set(strategy_votes)) / len(strategy_votes)
                    confidence_avg = np.mean(strategy_confidences)
                    quality_score = (final_signal_strength + strategy_diversity + confidence_avg) / 3
                    
                    # Reduced signal filtering threshold
                    if final_signal_strength > self.min_signal_strength:
                        
                        # === ENHANCED POSITION SIZING ===
                        base_size = min(self.max_position_size, final_signal_strength * 0.6)
                        
                        # Enhanced phase-based sizing
                        phase_multipliers = {
                            'crypto_winter': 0.4,
                            'gradual_recovery': 0.7,
                            'etf_anticipation': 0.9,
                            'etf_approval': 1.1,
                            'post_etf_consolidation': 0.8,
                            'summer_range': 0.6,
                            'pre_election': 0.9,
                            'election_rally': 1.2
                        }
                        
                        phase_mult = phase_multipliers.get(current['phase'], 0.8)
                        
                        # Quality-based sizing boost
                        quality_mult = 0.8 + (quality_score * 0.4)  # 0.8 to 1.2 range
                        
                        # Volatility adjustment
                        if current['volatility_regime'] > 0.045:
                            vol_mult = 0.6
                        elif current['volatility_regime'] < 0.025:
                            vol_mult = 1.15
                        else:
                            vol_mult = 1.0
                        
                        # Volume confirmation
                        volume_mult = min(1.25, max(0.8, current.get('volume_ratio_20', 1.0) / 2))
                        
                        final_position_size = min(
                            self.max_position_size,
                            base_size * phase_mult * quality_mult * vol_mult * volume_mult * self.position_size_multiplier
                        )
                        
                        signals.append({
                            'date': current['date'],
                            'price': current['price'],
                            'signal_type': final_direction,
                            'signal_strength': final_signal_strength,
                            'position_size': final_position_size,
                            'phase': current['phase'],
                            'volatility_regime': current['volatility_regime'],
                            'strategy_count': len(strategy_votes),
                            'quality_score': quality_score,
                            'contributing_strategies': [direction for direction, _ in strategy_votes],
                            'confidence_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'standard',
                            'enhanced_features': {
                                'trend_confidence': trend_confidence,
                                'momentum_confidence': momentum_confidence,
                                'strategy_diversity': strategy_diversity,
                                'volume_confirmation': current.get('volume_ratio_20', 1.0)
                            },
                            'institutional_grade': True,
                            'enhanced_signal': True
                        })
                        
                        # Update signal stats
                        self.signal_stats["generated"] += 1
        
        return signals
    
    def execute_enhanced_backtest(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt Enhanced Backtest mit optimierten Signalen durch
        """
        logger.info(f"Führe {self.strategy_name} Enhanced Backtest durch...")
        
        prices = price_data['prices']
        signals = self.generate_enhanced_signals(prices)
        
        logger.info(f"Generiert: {len(signals)} enhanced Signale (Original: ~50)")
        
        # Update signal stats
        self.signal_stats["generated"] = len(signals)
        if signals:
            self.signal_stats["quality_avg"] = np.mean([s['quality_score'] for s in signals])
        
        # Execute backtest with same institutional risk management
        portfolio = []
        cash = self.initial_capital
        btc_position = 0.0
        
        signal_dict = {s['date']: s for s in signals}
        
        # Risk management state
        peak_portfolio_value = self.initial_capital
        current_drawdown = 0.0
        emergency_stop_triggered = False
        
        self.current_phase = "enhanced_backtesting"
        
        for i, price_data_point in enumerate(prices):
            date = price_data_point['date']
            current_price = price_data_point['price']
            
            # Portfolio value calculation
            portfolio_value = cash + (btc_position * current_price)
            
            # Strict drawdown control (unchanged)
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value
                current_drawdown = 0.0
                emergency_stop_triggered = False
            else:
                current_drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
                
                if current_drawdown >= self.max_drawdown_limit:
                    if not emergency_stop_triggered:
                        emergency_stop_triggered = True
                        self.alerts.append({
                            'timestamp': date,
                            'type': 'EMERGENCY_STOP',
                            'message': f'Enhanced strategy drawdown limit: {current_drawdown:.1%}',
                            'action': 'All positions liquidated'
                        })
                        
                        if btc_position > 0:
                            liquidation_price = current_price * 0.995
                            cash += btc_position * liquidation_price
                            btc_position = 0.0
            
            # Risk reduction factors
            if current_drawdown > 0.15:
                risk_reduction_factor = 0.3
            elif current_drawdown > 0.10:
                risk_reduction_factor = 0.5
            elif current_drawdown > 0.05:
                risk_reduction_factor = 0.7
            else:
                risk_reduction_factor = 1.0
            
            # Execute enhanced signals
            if date in signal_dict and not emergency_stop_triggered:
                signal = signal_dict[date]
                self.last_signal_time = date
                
                # Enhanced risk controls with quality consideration
                quality_factor = 0.7 + (signal['quality_score'] * 0.3)  # 0.7 to 1.0
                institutional_risk_factor = risk_reduction_factor * quality_factor
                
                adjusted_position_size = signal['position_size'] * institutional_risk_factor
                
                if signal['signal_type'] == 'long' and adjusted_position_size > 0.03:  # Lower minimum
                    target_allocation = adjusted_position_size
                    target_value = portfolio_value * target_allocation
                    target_btc = target_value / current_price
                    btc_to_buy = target_btc - btc_position
                    
                    if btc_to_buy > self.min_trade_size:
                        # Enhanced execution cost modeling
                        slippage = self._calculate_enhanced_slippage(signal, price_data_point)
                        execution_price = current_price * (1 + slippage + self.trading_fee)
                        cost = btc_to_buy * execution_price
                        
                        if cost <= cash * 0.95:
                            cash -= cost
                            btc_position += btc_to_buy
                            
                            self.trades.append({
                                'date': date,
                                'type': 'BUY',
                                'quantity': btc_to_buy,
                                'price': execution_price,
                                'cost': cost,
                                'signal_strength': signal['signal_strength'],
                                'position_size': target_allocation,
                                'phase': signal['phase'],
                                'quality_score': signal['quality_score'],
                                'confidence_level': signal['confidence_level'],
                                'risk_factor': institutional_risk_factor,
                                'slippage': slippage,
                                'enhanced_signal': True,
                                'strategy_count': signal['strategy_count']
                            })
                            
                            self.signal_stats["executed"] += 1
                
                elif signal['signal_type'] == 'short' and adjusted_position_size > 0.03:
                    btc_to_sell = btc_position * adjusted_position_size
                    
                    if btc_to_sell > self.min_trade_size:
                        slippage = self._calculate_enhanced_slippage(signal, price_data_point)
                        execution_price = current_price * (1 - slippage - self.trading_fee)
                        proceeds = btc_to_sell * execution_price
                        
                        cash += proceeds
                        btc_position -= btc_to_sell
                        
                        self.trades.append({
                            'date': date,
                            'type': 'SELL',
                            'quantity': btc_to_sell,
                            'price': execution_price,
                            'proceeds': proceeds,
                            'signal_strength': signal['signal_strength'],
                            'position_size': adjusted_position_size,
                            'phase': signal['phase'],
                            'quality_score': signal['quality_score'],
                            'confidence_level': signal['confidence_level'],
                            'risk_factor': institutional_risk_factor,
                            'slippage': slippage,
                            'enhanced_signal': True,
                            'strategy_count': signal['strategy_count']
                        })
                        
                        self.signal_stats["executed"] += 1
            
            # Portfolio snapshot
            portfolio_value = cash + btc_position * current_price
            portfolio.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'btc_position': btc_position,
                'btc_value': btc_position * current_price,
                'btc_price': current_price,
                'phase': price_data_point['phase'],
                'current_drawdown': current_drawdown,
                'emergency_stop': emergency_stop_triggered,
                'risk_reduction_factor': risk_reduction_factor,
                'allocation_pct': (btc_position * current_price) / portfolio_value if portfolio_value > 0 else 0
            })
            
            # Calculate daily return
            if len(portfolio) > 1:
                prev_value = portfolio[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                portfolio[-1]['daily_return'] = daily_return
                self.daily_returns.append(daily_return)
        
        self.equity_curve = portfolio
        self.current_phase = "enhanced_analysis"
        
        return {
            'strategy_info': self.get_strategy_info(),
            'portfolio_history': portfolio,
            'signals': signals,
            'trades': self.trades,
            'alerts': self.alerts,
            'signal_stats': self.signal_stats,
            'performance_metrics': self._calculate_enhanced_metrics(),
            'enhancement_summary': self._generate_enhancement_summary(signals)
        }
    
    def _calculate_enhanced_slippage(self, signal: Dict, price_data: Dict) -> float:
        """Enhanced slippage calculation mit quality adjustment"""
        base_slippage = 0.0003
        
        # Quality-based slippage reduction
        quality_factor = 0.7 + (signal['quality_score'] * 0.3)
        
        phase_multipliers = {
            'crypto_winter': 2.0, 'gradual_recovery': 1.2, 'etf_anticipation': 1.0,
            'etf_approval': 1.5, 'post_etf_consolidation': 1.1, 'summer_range': 0.8,
            'pre_election': 1.2, 'election_rally': 1.8
        }
        
        phase_mult = phase_multipliers.get(price_data['phase'], 1.0)
        volume_factor = max(0.6, min(1.5, 1.2 / signal.get('volume_confirmation', 1.0)))
        size_penalty = 1.0 + (signal['position_size'] * 1.5)
        vol_penalty = 1.0 + (price_data['volatility_regime'] * 3)
        
        total_slippage = (base_slippage * phase_mult * volume_factor * 
                         size_penalty * vol_penalty) / quality_factor
        
        return min(0.005, total_slippage)
    
    def _calculate_enhanced_metrics(self) -> Dict[str, Any]:
        """Enhanced performance metrics mit Signal-Analytics"""
        if len(self.daily_returns) < 2:
            return {}
        
        returns = np.array(self.daily_returns)
        equity_values = [p['portfolio_value'] for p in self.equity_curve]
        
        # Standard metrics
        final_value = equity_values[-1]
        total_return = (final_value / self.initial_capital) - 1
        days = len(equity_values)
        annual_return = ((final_value / self.initial_capital) ** (365 / days)) - 1
        
        daily_vol = np.std(returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)
        
        risk_free_rate = 0.025
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252) if len(negative_returns) > 1 else annual_vol
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        max_dd = self._calculate_max_drawdown(equity_values)
        calmar_ratio = annual_return / max(max_dd, 0.01)
        
        # Enhanced signal metrics
        if self.trades:
            quality_scores = [t.get('quality_score', 0) for t in self.trades]
            avg_quality = np.mean(quality_scores)
            
            confidence_levels = [t.get('confidence_level', 'standard') for t in self.trades]
            high_confidence_rate = sum(1 for c in confidence_levels if c == 'high') / len(confidence_levels)
            
            strategy_counts = [t.get('strategy_count', 0) for t in self.trades]
            avg_strategy_count = np.mean(strategy_counts)
            
            enhanced_trades = len([t for t in self.trades if t.get('enhanced_signal', False)])
            enhanced_rate = enhanced_trades / len(self.trades)
        else:
            avg_quality = 0
            high_confidence_rate = 0
            avg_strategy_count = 0
            enhanced_rate = 0
        
        return {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'annual_volatility': annual_vol,
            'total_trades': len(self.trades),
            'signal_generation_rate': self.signal_stats["generated"] / days,
            'signal_execution_rate': self.signal_stats["executed"] / max(self.signal_stats["generated"], 1),
            'avg_signal_quality': self.signal_stats["quality_avg"],
            'avg_trade_quality': avg_quality,
            'high_confidence_rate': high_confidence_rate,
            'avg_strategy_count': avg_strategy_count,
            'enhanced_trade_rate': enhanced_rate,
            'days_analyzed': days,
            'final_capital': final_value,
            'enhancement_active': True
        }
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return abs(max_drawdown)
    
    def _generate_enhancement_summary(self, signals: List[Dict]) -> Dict[str, Any]:
        """Generiert Summary der Signal-Enhancements"""
        if not signals:
            return {}
        
        quality_scores = [s['quality_score'] for s in signals]
        confidence_levels = [s['confidence_level'] for s in signals]
        strategy_counts = [s['strategy_count'] for s in signals]
        
        return {
            'total_signals_generated': len(signals),
            'avg_quality_score': np.mean(quality_scores),
            'quality_distribution': {
                'high': sum(1 for q in quality_scores if q > 0.8) / len(quality_scores),
                'medium': sum(1 for q in quality_scores if 0.6 <= q <= 0.8) / len(quality_scores),
                'standard': sum(1 for q in quality_scores if q < 0.6) / len(quality_scores)
            },
            'confidence_distribution': {
                'high': sum(1 for c in confidence_levels if c == 'high') / len(confidence_levels),
                'medium': sum(1 for c in confidence_levels if c == 'medium') / len(confidence_levels),
                'standard': sum(1 for c in confidence_levels if c == 'standard') / len(confidence_levels)
            },
            'avg_strategies_per_signal': np.mean(strategy_counts),
            'signal_frequency_improvement': f"+{(len(signals) / 50 - 1) * 100:.0f}%" if len(signals) > 50 else "No improvement"
        }


async def run_enhanced_step1():
    """
    SCHRITT 1: Signal-Optimierung ausführen
    """
    print("🔧 SCHRITT 1: SIGNAL-OPTIMIERUNG")
    print("=" * 80)
    print("Enhanced Strategy: Enhanced Institutional BTC Pro v1.1")
    print("Optimierung: Signal-Schwelle 70% → 62% | 6 → 8 Strategien | Adaptive Timeframes")
    print()
    
    strategy = EnhancedInstitutionalStrategy(initial_capital=300000.0)
    
    # Use same data as before for comparison
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("📊 Generiere Daten für Signal-Optimierung...")
    
    # Use institutional data generator from previous strategy
    from institutional_grade_btc_strategy import InstitutionalGradeBTCStrategy
    base_strategy = InstitutionalGradeBTCStrategy()
    price_data = base_strategy.generate_institutional_data(start_date, end_date)
    
    print(f"✅ {len(price_data['prices'])} Tage bereit für Enhanced Analysis")
    print()
    
    # Execute enhanced backtest
    print("⚡ Führe SCHRITT 1 Enhanced Backtest durch...")
    results = strategy.execute_enhanced_backtest(price_data)
    
    # Compare with original
    print("📊 SCHRITT 1 ERGEBNISSE - SIGNAL-OPTIMIERUNG")
    print("-" * 80)
    
    metrics = results['performance_metrics']
    enhancement = results['enhancement_summary']
    signal_stats = results['signal_stats']
    
    print(f"Strategy: {metrics['strategy_name']} v{metrics['strategy_version']}")
    print()
    print("🎯 SIGNAL-VERBESSERUNGEN:")
    print(f"   Generierte Signale:     {signal_stats['generated']:,} (Original: ~50)")
    print(f"   Ausgeführte Trades:     {signal_stats['executed']:,} (Original: ~3)")
    print(f"   Execution Rate:         {metrics['signal_execution_rate']:.1%}")
    print(f"   Ø Signal Quality:       {signal_stats['quality_avg']:.2f}")
    print(f"   Signal Frequency:       {enhancement.get('signal_frequency_improvement', 'N/A')}")
    print()
    
    print("📈 PERFORMANCE VERGLEICH:")
    print("                          Enhanced    |   Original")
    print("-" * 55)
    print(f"Annual Return:           {metrics['annual_return']:8.1%}   |    7.1%")
    print(f"Sharpe Ratio:            {metrics['sharpe_ratio']:8.2f}   |    0.85")
    print(f"Max Drawdown:            {metrics['max_drawdown']:8.1%}   |    5.1%")
    print(f"Total Trades:            {metrics['total_trades']:8,}   |    3")
    print(f"Volatilität:             {metrics['annual_volatility']:8.1%}   |    5.4%")
    print()
    
    # Target assessment for Step 1
    print("🎯 SCHRITT 1 ZIEL-BEWERTUNG:")
    print("-" * 80)
    
    signal_improvement = metrics['total_trades'] > 3
    return_improvement = metrics['annual_return'] > 0.071
    risk_maintained = metrics['max_drawdown'] <= 0.25
    quality_maintained = signal_stats['quality_avg'] > 0.5
    
    step1_score = 0
    if signal_improvement: step1_score += 30
    if return_improvement: step1_score += 25
    if risk_maintained: step1_score += 25
    if quality_maintained: step1_score += 20
    
    print(f"Mehr Trades generiert:   {'✅' if signal_improvement else '❌'} ({metrics['total_trades']} vs 3)")
    print(f"Return verbessert:       {'✅' if return_improvement else '❌'} ({metrics['annual_return']:.1%} vs 7.1%)")
    print(f"Risiko kontrolliert:     {'✅' if risk_maintained else '❌'} ({metrics['max_drawdown']:.1%} ≤ 25%)")
    print(f"Quality beibehalten:     {'✅' if quality_maintained else '❌'} ({signal_stats['quality_avg']:.2f} > 0.5)")
    print()
    print(f"SCHRITT 1 Score: {step1_score}/100")
    
    if step1_score >= 75:
        step1_status = "✅ ERFOLGREICH - Bereit für Schritt 2"
        next_action = "Proceed to Schritt 2: Return-Enhancement"
    elif step1_score >= 50:
        step1_status = "🔄 TEILWEISE ERFOLGREICH - Verbesserungen sichtbar"
        next_action = "Proceed to Schritt 2 mit Adjustments"
    else:
        step1_status = "❌ WEITERE OPTIMIERUNG ERFORDERLICH"
        next_action = "Iterate on Schritt 1 before proceeding"
    
    print(f"Status: {step1_status}")
    print(f"Nächste Aktion: {next_action}")
    print()
    
    # Export Step 1 results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"step1_signal_optimization_{timestamp}.json"
    
    export_data = {
        'step': 1,
        'optimization_focus': 'Signal Enhancement',
        'strategy_info': results['strategy_info'],
        'performance_metrics': metrics,
        'enhancement_summary': enhancement,
        'signal_statistics': signal_stats,
        'step1_assessment': {
            'score': step1_score,
            'status': step1_status,
            'improvements': {
                'signal_improvement': signal_improvement,
                'return_improvement': return_improvement,
                'risk_maintained': risk_maintained,
                'quality_maintained': quality_maintained
            },
            'next_action': next_action
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"💾 SCHRITT 1 Ergebnisse exportiert: {filename}")
    
    return metrics, step1_score, step1_status


if __name__ == "__main__":
    asyncio.run(run_enhanced_step1())