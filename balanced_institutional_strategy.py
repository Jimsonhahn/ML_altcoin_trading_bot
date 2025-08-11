#!/usr/bin/env python3
"""
Balanced Institutional Strategy - Schritt 1.1: Drawdown-Korrektur
Ausbalancierte Signal-Optimierung mit strikterer Risikokontrolle

Strategy Name: "Balanced Institutional BTC Elite"
Version: 1.2 Risk-Balanced
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


class BalancedInstitutionalStrategy:
    """
    Balanced Institutional Strategy - Schritt 1.1 Korrektur
    
    Name: "Balanced Institutional BTC Elite"
    
    Korrektur-Optimierungen:
    1. Signal-Schwelle optimal: 62% → 66% (Balance)
    2. Striktere Position-Limits: 45% → 35%
    3. Progressive Drawdown-Controls: Frühere Reduktion
    4. Quality-First Execution: Nur Top-Quality Signale
    5. Adaptive Risk Scaling: Dynamische Anpassung
    """
    
    def __init__(self, initial_capital: float = 300000.0):
        self.strategy_name = "Balanced Institutional BTC Elite"
        self.strategy_version = "1.3 Fine-Tuned"
        self.risk_profile = "Balanced Institutional"
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_position = 0.0
        self.cash_balance = initial_capital
        
        # Balanced Parameters (Korrektur)
        self.trading_fee = 0.001
        self.min_trade_size = 0.001
        self.max_position_size = 0.38  # STEP 1.2: 35% → 38% (mehr Exposure)
        self.max_drawdown_limit = 0.18  # STEP 1.2: 20% → 18% (noch strenger)
        self.min_signal_strength = 0.63  # STEP 1.2: 66% → 63% (mehr Trades)
        
        # Enhanced Risk Management
        self.position_size_multiplier = 0.75  # Konservativer: 0.85 → 0.75
        self.emergency_stop_enabled = True
        self.monthly_var_limit = 0.08  # Reduziert: 10% → 8%
        self.quality_threshold = 0.75  # Nur Top-Quality Signale
        
        # Progressive Controls
        self.progressive_dd_control = True
        self.adaptive_sizing = True
        self.quality_first_execution = True
        
        # Performance Tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.alerts = []
        
        # Dashboard Integration
        self.last_signal_time = None
        self.current_phase = "initialization"
        self.strategy_status = "active"
        self.signal_stats = {"generated": 0, "executed": 0, "quality_avg": 0, "top_quality": 0}
        
        logger.info(f"{self.strategy_name} v{self.strategy_version} initialisiert")
        logger.info(f"STEP 1.2: Signal={self.min_signal_strength:.0%} | Position={self.max_position_size:.0%} | DD={self.max_drawdown_limit:.0%}")
    
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
            'quality_threshold': self.quality_threshold,
            'current_phase': self.current_phase,
            'last_signal_time': self.last_signal_time,
            'signal_stats': self.signal_stats,
            'alerts': self.alerts[-5:] if self.alerts else []
        }
    
    def generate_balanced_signals(self, prices: List[Dict]) -> List[Dict]:
        """
        SCHRITT 1.1: Balanced Signal Generation mit Quality-First Ansatz
        """
        signals = []
        df = pd.DataFrame(prices)
        
        # Same technical indicators as enhanced version
        for window in [3, 5, 8, 10, 13, 20, 34, 50, 89, 144, 200]:
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
        
        for span in [8, 12, 21, 26, 34, 50, 89]:
            df[f'ema_{span}'] = df['price'].ewm(span=span).mean()
        
        # Multiple MACD timeframes
        df['macd_fast'] = df['ema_8'] - df['ema_21']
        df['macd_standard'] = df['ema_12'] - df['ema_26']
        df['macd_slow'] = df['ema_21'] - df['ema_50']
        
        for macd in ['macd_fast', 'macd_standard', 'macd_slow']:
            df[f'{macd}_signal'] = df[macd].ewm(span=9).mean()
        
        # Enhanced volatility analysis
        for window in [5, 8, 13, 20, 34, 50]:
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std()
            df[f'vol_zscore_{window}'] = ((df[f'volatility_{window}d'] - 
                                         df[f'volatility_{window}d'].rolling(50).mean()) / 
                                        df[f'volatility_{window}d'].rolling(50).std())
        
        # Multiple RSI periods
        for period in [9, 14, 21, 34, 50]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Enhanced momentum indicators
        for period in [3, 5, 8, 13, 20, 34, 50, 89]:
            df[f'momentum_{period}d'] = df['price'] / df['price'].shift(period)
        
        # Volume analysis
        for window in [5, 10, 20, 34, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Bollinger Bands
        for period in [13, 20, 34, 50]:
            df[f'bb_middle_{period}'] = df['price'].rolling(period).mean()
            bb_std = df['price'].rolling(period).std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
            df[f'bb_position_{period}'] = ((df['price'] - df[f'bb_lower_{period}']) / 
                                         (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']))
        
        # Conservative analysis window (increased for stability)
        for i in range(200, len(df)):  # Back to 200 for more stability
            current = df.iloc[i]
            
            # === BALANCED SIGNAL STRATEGIES ===
            
            # 1. ENHANCED TREND STRATEGY (Stricter Requirements)
            trend_signals = []
            
            # Only strong, confirmed trends
            strong_uptrend = (current['sma_20'] > current['sma_50'] * 1.02 and 
                            current['sma_50'] > current['sma_89'] * 1.02 and
                            current['sma_89'] > current['sma_144'] * 1.01 and
                            current['price'] > current['sma_20'] * 1.03)
            
            if strong_uptrend:
                trend_signals.append(('long', 0.8))
            
            # Medium trend with volume confirmation
            medium_uptrend = (current['sma_20'] > current['sma_50'] and 
                            current['sma_50'] > current['sma_89'] and
                            current['price'] > current['sma_20'] * 1.02 and
                            current.get('volume_ratio_20', 1.0) > 1.2)
            
            if medium_uptrend and not strong_uptrend:
                trend_signals.append(('long', 0.6))
            
            if trend_signals:
                trend_direction = 'long'
                trend_signal = max([signal for _, signal in trend_signals])  # Take strongest
                trend_confidence = len(trend_signals) / 2
            else:
                trend_direction = None
                trend_signal = 0
                trend_confidence = 0
            
            # 2. QUALITY-FOCUSED MOMENTUM STRATEGY
            momentum_signals = []
            
            # Multi-timeframe momentum with strict requirements
            short_mom_strong = (current['momentum_5d'] > 1.03 and current['momentum_8d'] > 1.04)
            medium_mom_strong = (current['momentum_20d'] > 1.08 and current['momentum_34d'] > 1.12)
            macd_strong = (current['macd_standard'] > current['macd_standard_signal'] and
                          current['macd_fast'] > current['macd_fast_signal'])
            
            if short_mom_strong and medium_mom_strong and macd_strong:
                momentum_signals.append(('long', 0.9))  # Very strong
            elif short_mom_strong and macd_strong:
                momentum_signals.append(('long', 0.7))
            elif medium_mom_strong:
                momentum_signals.append(('long', 0.6))
            
            if momentum_signals:
                momentum_direction = 'long'
                momentum_signal = max([signal for _, signal in momentum_signals])
                momentum_confidence = len(momentum_signals) / 3
            else:
                momentum_direction = None
                momentum_signal = 0
                momentum_confidence = 0
            
            # 3. HIGH-QUALITY MEAN REVERSION
            mean_reversion_signals = []
            
            # Strict oversold conditions with multiple confirmations
            extreme_oversold = (current['rsi_14'] < 25 and current['rsi_21'] < 30 and
                              current['bb_position_20'] < 0.05 and
                              current.get('vol_zscore_20', 0) > 1.5)
            
            moderate_oversold = (current['rsi_14'] < 35 and current['bb_position_20'] < 0.15 and
                               current['phase'] in ['crypto_winter', 'post_etf_consolidation'])
            
            if extreme_oversold:
                mean_reversion_signals.append(('long', 0.8))
            elif moderate_oversold and current.get('volume_ratio_20', 1.0) > 1.3:
                mean_reversion_signals.append(('long', 0.6))
            
            if mean_reversion_signals:
                mean_reversion_direction = 'long'
                mean_reversion_signal = max([signal for _, signal in mean_reversion_signals])
            else:
                mean_reversion_direction = None
                mean_reversion_signal = 0
            
            # 4. INSTITUTIONAL FLOW (Higher Standards)
            institutional_signals = []
            
            # Strong institutional activity with multiple confirmations
            strong_institutional = (current['institutional_flow'] > 0.2 and 
                                  current['whale_activity'] > 0.8 and
                                  current.get('volume_ratio_20', 1.0) > 1.5)
            
            if strong_institutional:
                institutional_signals.append(('long', 0.8))
            
            if institutional_signals:
                institutional_direction = 'long'
                institutional_signal = max([signal for _, signal in institutional_signals])
            else:
                institutional_direction = None
                institutional_signal = 0
            
            # 5. CONSERVATIVE SENTIMENT
            sentiment_signals = []
            
            # Only extreme contrarian signals
            extreme_bearish_sentiment = (current['sentiment'] < -0.3 and 
                                       current['funding_rate'] < -0.01 and
                                       current['put_call_ratio'] > 1.5)
            
            if extreme_bearish_sentiment and current['phase'] == 'crypto_winter':
                sentiment_signals.append(('long', 0.7))
            
            if sentiment_signals:
                sentiment_direction = 'long'
                sentiment_signal = max([signal for _, signal in sentiment_signals])
            else:
                sentiment_direction = None
                sentiment_signal = 0
            
            # === QUALITY-FIRST ENSEMBLE ===
            
            strategy_votes = []
            strategy_weights = []
            strategy_confidences = []
            
            # Conservative weighting favoring trend and momentum
            base_weights = {
                'trend': 0.35,
                'momentum': 0.30,
                'institutional': 0.20,
                'mean_reversion': 0.10,
                'sentiment': 0.05
            }
            
            # Add votes only for high-quality signals
            strategies = [
                ('trend', trend_direction, trend_signal, trend_confidence),
                ('momentum', momentum_direction, momentum_signal, momentum_confidence),
                ('mean_reversion', mean_reversion_direction, mean_reversion_signal, 1.0),
                ('institutional', institutional_direction, institutional_signal, 1.0),
                ('sentiment', sentiment_direction, sentiment_signal, 1.0)
            ]
            
            for strategy_name, direction, signal, confidence in strategies:
                if direction and signal >= 0.6:  # Higher quality threshold
                    strategy_votes.append((direction, signal))
                    strategy_weights.append(base_weights[strategy_name] * confidence)
                    strategy_confidences.append(confidence)
            
            # Ensemble decision with quality focus
            if len(strategy_votes) >= 2:  # Need at least 2 high-quality strategies
                total_weight = sum(strategy_weights)
                if total_weight > 0:
                    strategy_weights = [w / total_weight for w in strategy_weights]
                    
                    long_score = sum(weight * signal for (direction, signal), weight in 
                                   zip(strategy_votes, strategy_weights) if direction == 'long')
                    
                    final_signal_strength = long_score
                    final_direction = 'long'
                    
                    # Enhanced quality scoring with stricter standards
                    strategy_diversity = len(set(strategy_votes)) / len(strategy_votes)
                    confidence_avg = np.mean(strategy_confidences)
                    signal_strength_bonus = final_signal_strength if final_signal_strength > 0.7 else 0
                    
                    quality_score = (final_signal_strength * 0.4 + 
                                   strategy_diversity * 0.3 + 
                                   confidence_avg * 0.2 +
                                   signal_strength_bonus * 0.1)
                    
                    # Strict filtering: Higher signal strength AND quality thresholds
                    if (final_signal_strength > self.min_signal_strength and 
                        quality_score > self.quality_threshold):
                        
                        # === CONSERVATIVE POSITION SIZING ===
                        base_size = min(self.max_position_size, final_signal_strength * 0.4)  # More conservative
                        
                        # Conservative phase multipliers
                        phase_multipliers = {
                            'crypto_winter': 0.3,
                            'gradual_recovery': 0.6,
                            'etf_anticipation': 0.7,
                            'etf_approval': 0.8,  # More conservative
                            'post_etf_consolidation': 0.6,
                            'summer_range': 0.4,
                            'pre_election': 0.7,
                            'election_rally': 0.9   # More conservative
                        }
                        
                        phase_mult = phase_multipliers.get(current['phase'], 0.6)
                        
                        # Quality-based sizing (more conservative)
                        quality_mult = 0.6 + (quality_score * 0.4)  # 0.6 to 1.0 range
                        
                        # Conservative volatility adjustment
                        if current['volatility_regime'] > 0.045:
                            vol_mult = 0.4  # Much more conservative
                        elif current['volatility_regime'] < 0.025:
                            vol_mult = 1.0  # Less aggressive boost
                        else:
                            vol_mult = 0.8
                        
                        # Volume confirmation (conservative)
                        volume_mult = min(1.1, max(0.7, current.get('volume_ratio_20', 1.0) / 3))
                        
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
                            'confidence_level': 'ultra_high' if quality_score > 0.9 else 'high' if quality_score > 0.8 else 'medium',
                            'balanced_features': {
                                'trend_confidence': trend_confidence,
                                'momentum_confidence': momentum_confidence,
                                'strategy_diversity': strategy_diversity,
                                'quality_weighted_strength': quality_score,
                                'conservative_sizing': True
                            },
                            'institutional_grade': True,
                            'balanced_signal': True,
                            'top_quality': quality_score > 0.85
                        })
                        
                        # Update signal stats
                        self.signal_stats["generated"] += 1
                        if quality_score > 0.85:
                            self.signal_stats["top_quality"] += 1
        
        return signals
    
    def execute_balanced_backtest(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt Balanced Backtest mit strikter Risikokontrolle durch
        """
        logger.info(f"Führe {self.strategy_name} Balanced Backtest durch...")
        
        prices = price_data['prices']
        signals = self.generate_balanced_signals(prices)
        
        logger.info(f"Generiert: {len(signals)} balanced signals")
        
        # Update signal stats
        self.signal_stats["generated"] = len(signals)
        if signals:
            quality_scores = [s['quality_score'] for s in signals]
            self.signal_stats["quality_avg"] = np.mean(quality_scores)
            self.signal_stats["top_quality"] = len([s for s in signals if s.get('top_quality', False)])
        
        # Execute backtest with enhanced risk management
        portfolio = []
        cash = self.initial_capital
        btc_position = 0.0
        
        signal_dict = {s['date']: s for s in signals}
        
        # Enhanced risk management state
        peak_portfolio_value = self.initial_capital
        current_drawdown = 0.0
        emergency_stop_triggered = False
        consecutive_losses = 0
        
        self.current_phase = "balanced_backtesting"
        
        for i, price_data_point in enumerate(prices):
            date = price_data_point['date']
            current_price = price_data_point['price']
            
            # Portfolio value calculation
            portfolio_value = cash + (btc_position * current_price)
            
            # STRICT PROGRESSIVE DRAWDOWN CONTROL
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value
                current_drawdown = 0.0
                emergency_stop_triggered = False
                consecutive_losses = 0
            else:
                current_drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
                
                # Stricter emergency stop
                if current_drawdown >= self.max_drawdown_limit:
                    if not emergency_stop_triggered:
                        emergency_stop_triggered = True
                        self.alerts.append({
                            'timestamp': date,
                            'type': 'EMERGENCY_STOP',
                            'message': f'Balanced strategy DD limit: {current_drawdown:.1%}',
                            'action': 'All positions liquidated'
                        })
                        
                        if btc_position > 0:
                            liquidation_price = current_price * 0.995
                            cash += btc_position * liquidation_price
                            btc_position = 0.0
            
            # PROGRESSIVE RISK REDUCTION (Earlier Intervention)
            if current_drawdown > 0.12:  # 12% drawdown (vs 15% before)
                risk_reduction_factor = 0.2
                consecutive_losses += 1
            elif current_drawdown > 0.08:  # 8% drawdown (vs 10% before)
                risk_reduction_factor = 0.4
            elif current_drawdown > 0.04:  # 4% drawdown (vs 5% before)
                risk_reduction_factor = 0.6
            else:
                risk_reduction_factor = 1.0
            
            # Track consecutive losses for additional risk reduction
            if i > 0 and len(portfolio) > 1:
                yesterday_value = portfolio[-1]['portfolio_value']
                if portfolio_value < yesterday_value * 0.99:  # 1% daily loss
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            
            # Additional consecutive loss penalty
            if consecutive_losses > 3:
                risk_reduction_factor *= 0.5
            elif consecutive_losses > 2:
                risk_reduction_factor *= 0.7
            
            # Execute balanced signals (only if not in emergency stop)
            if date in signal_dict and not emergency_stop_triggered:
                signal = signal_dict[date]
                self.last_signal_time = date
                
                # Quality-first execution
                if not signal.get('top_quality', False) and current_drawdown > 0.05:
                    continue  # Skip non-top-quality signals during drawdown
                
                # Enhanced risk controls
                quality_boost = 1.0 + (signal['quality_score'] - 0.75) * 0.5
                institutional_risk_factor = risk_reduction_factor * quality_boost
                
                adjusted_position_size = signal['position_size'] * institutional_risk_factor
                
                if signal['signal_type'] == 'long' and adjusted_position_size > 0.02:  # Minimum 2%
                    target_allocation = adjusted_position_size
                    target_value = portfolio_value * target_allocation
                    target_btc = target_value / current_price
                    btc_to_buy = target_btc - btc_position
                    
                    if btc_to_buy > self.min_trade_size:
                        # Conservative execution cost modeling
                        slippage = self._calculate_conservative_slippage(signal, price_data_point)
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
                                'top_quality': signal.get('top_quality', False),
                                'risk_factor': institutional_risk_factor,
                                'slippage': slippage,
                                'balanced_signal': True,
                                'strategy_count': signal['strategy_count'],
                                'consecutive_losses': consecutive_losses
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
                'consecutive_losses': consecutive_losses,
                'allocation_pct': (btc_position * current_price) / portfolio_value if portfolio_value > 0 else 0
            })
            
            # Calculate daily return
            if len(portfolio) > 1:
                prev_value = portfolio[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                portfolio[-1]['daily_return'] = daily_return
                self.daily_returns.append(daily_return)
        
        self.equity_curve = portfolio
        self.current_phase = "balanced_analysis"
        
        return {
            'strategy_info': self.get_strategy_info(),
            'portfolio_history': portfolio,
            'signals': signals,
            'trades': self.trades,
            'alerts': self.alerts,
            'signal_stats': self.signal_stats,
            'performance_metrics': self._calculate_balanced_metrics(),
            'balance_summary': self._generate_balance_summary(signals)
        }
    
    def _calculate_conservative_slippage(self, signal: Dict, price_data: Dict) -> float:
        """Conservative slippage calculation"""
        base_slippage = 0.0002  # Even lower
        
        # Quality-based slippage reduction (stronger effect)
        quality_factor = 0.5 + (signal['quality_score'] * 0.5)  # 0.5 to 1.0
        
        phase_multipliers = {
            'crypto_winter': 1.5, 'gradual_recovery': 1.0, 'etf_anticipation': 0.9,
            'etf_approval': 1.2, 'post_etf_consolidation': 1.0, 'summer_range': 0.8,
            'pre_election': 1.0, 'election_rally': 1.3
        }
        
        phase_mult = phase_multipliers.get(price_data['phase'], 1.0)
        volume_factor = max(0.7, min(1.3, 1.0 / signal.get('volume_confirmation', 1.0)))
        size_penalty = 1.0 + (signal['position_size'] * 1.0)  # Reduced penalty
        vol_penalty = 1.0 + (price_data['volatility_regime'] * 2)  # Reduced penalty
        
        total_slippage = (base_slippage * phase_mult * volume_factor * 
                         size_penalty * vol_penalty) / quality_factor
        
        return min(0.003, total_slippage)  # Lower cap
    
    def _calculate_balanced_metrics(self) -> Dict[str, Any]:
        """Balanced performance metrics"""
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
        
        # Balanced specific metrics
        if self.trades:
            quality_scores = [t.get('quality_score', 0) for t in self.trades]
            avg_quality = np.mean(quality_scores)
            
            top_quality_trades = len([t for t in self.trades if t.get('top_quality', False)])
            top_quality_rate = top_quality_trades / len(self.trades)
            
            balanced_trades = len([t for t in self.trades if t.get('balanced_signal', False)])
            balanced_rate = balanced_trades / len(self.trades)
        else:
            avg_quality = 0
            top_quality_rate = 0
            balanced_rate = 0
        
        return {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'risk_profile': self.risk_profile,
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
            'top_quality_rate': top_quality_rate,
            'balanced_trade_rate': balanced_rate,
            'top_quality_signals': self.signal_stats["top_quality"],
            'days_analyzed': days,
            'final_capital': final_value,
            'balanced_optimization': True,
            'drawdown_limit': self.max_drawdown_limit,
            'quality_threshold': self.quality_threshold
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
    
    def _generate_balance_summary(self, signals: List[Dict]) -> Dict[str, Any]:
        """Generiert Balance Summary"""
        if not signals:
            return {}
        
        quality_scores = [s['quality_score'] for s in signals]
        top_quality_signals = [s for s in signals if s.get('top_quality', False)]
        
        return {
            'total_signals_generated': len(signals),
            'top_quality_signals': len(top_quality_signals),
            'top_quality_rate': len(top_quality_signals) / len(signals),
            'avg_quality_score': np.mean(quality_scores),
            'quality_distribution': {
                'ultra_high': sum(1 for q in quality_scores if q > 0.9) / len(quality_scores),
                'high': sum(1 for q in quality_scores if 0.8 <= q <= 0.9) / len(quality_scores),
                'medium': sum(1 for q in quality_scores if 0.75 <= q < 0.8) / len(quality_scores),
                'filtered_out': sum(1 for q in quality_scores if q < 0.75) / len(quality_scores)
            },
            'balance_improvements': {
                'conservative_sizing': True,
                'progressive_dd_control': True,
                'quality_first_execution': True,
                'stricter_risk_limits': True
            }
        }


async def run_balanced_step1_1():
    """
    SCHRITT 1.1: Balanced Signal-Optimierung mit Drawdown-Korrektur
    """
    print("🔧 SCHRITT 1.1: BALANCED SIGNAL-OPTIMIERUNG")
    print("=" * 80)
    print("Balanced Strategy: Balanced Institutional BTC Elite v1.2")
    print("Korrekturen: DD 25%→20% | Position 45%→35% | Signal 62%→66% | Quality-First")
    print()
    
    strategy = BalancedInstitutionalStrategy(initial_capital=300000.0)
    
    # Use same data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("📊 Generiere Daten für Balanced Analysis...")
    
    # Use institutional data
    from institutional_grade_btc_strategy import InstitutionalGradeBTCStrategy
    base_strategy = InstitutionalGradeBTCStrategy()
    price_data = base_strategy.generate_institutional_data(start_date, end_date)
    
    print(f"✅ {len(price_data['prices'])} Tage bereit für Balanced Analysis")
    print()
    
    # Execute balanced backtest
    print("⚡ Führe SCHRITT 1.1 Balanced Backtest durch...")
    results = strategy.execute_balanced_backtest(price_data)
    
    # Compare with previous versions
    print("📊 SCHRITT 1.1 ERGEBNISSE - BALANCED OPTIMIERUNG")
    print("-" * 80)
    
    metrics = results['performance_metrics']
    balance = results['balance_summary']
    signal_stats = results['signal_stats']
    
    print(f"Strategy: {metrics['strategy_name']} v{metrics['strategy_version']}")
    print(f"Risk Profile: {metrics['risk_profile']}")
    print()
    
    print("🎯 BALANCED IMPROVEMENTS:")
    print(f"   Generierte Signale:     {signal_stats['generated']:,}")
    print(f"   Top Quality Signale:    {signal_stats['top_quality']:,}")
    print(f"   Ausgeführte Trades:     {signal_stats['executed']:,}")
    print(f"   Top Quality Rate:       {balance['top_quality_rate']:.1%}")
    print(f"   Ø Signal Quality:       {signal_stats['quality_avg']:.2f}")
    print()
    
    print("📈 PERFORMANCE VERGLEICH:")
    print("                          Balanced   | Enhanced  | Original")
    print("-" * 65)
    print(f"Annual Return:           {metrics['annual_return']:8.1%}  | {37.6:8.1%}  | {7.1:8.1%}")
    print(f"Sharpe Ratio:            {metrics['sharpe_ratio']:8.2f}  | {1.36:8.2f}  | {0.85:8.2f}")
    print(f"Max Drawdown:            {metrics['max_drawdown']:8.1%}  | {26.1:8.1%}  | {5.1:8.1%}")
    print(f"Total Trades:            {metrics['total_trades']:8,}  | {2:8,}  | {3:8,}")
    print(f"Volatilität:             {metrics['annual_volatility']:8.1%}  | {25.9:8.1%}  | {5.4:8.1%}")
    print()
    
    # Target assessment for Step 1.1
    print("🎯 SCHRITT 1.1 ZIEL-BEWERTUNG:")
    print("-" * 80)
    
    signal_improvement = metrics['total_trades'] >= 3
    return_improvement = metrics['annual_return'] > 0.15  # 15% target
    risk_controlled = metrics['max_drawdown'] <= metrics['drawdown_limit']
    quality_maintained = signal_stats['quality_avg'] > 0.75
    
    step1_1_score = 0
    if signal_improvement: step1_1_score += 25
    if return_improvement: step1_1_score += 30
    if risk_controlled: step1_1_score += 30
    if quality_maintained: step1_1_score += 15
    
    print(f"Ausreichend Trades:      {'✅' if signal_improvement else '❌'} ({metrics['total_trades']} ≥ 3)")
    print(f"Return > 15%:            {'✅' if return_improvement else '❌'} ({metrics['annual_return']:.1%})")
    print(f"Drawdown kontrolliert:   {'✅' if risk_controlled else '❌'} ({metrics['max_drawdown']:.1%} ≤ {metrics['drawdown_limit']:.0%})")
    print(f"Quality hoch:            {'✅' if quality_maintained else '❌'} ({signal_stats['quality_avg']:.2f} > 0.75)")
    print()
    print(f"SCHRITT 1.1 Score: {step1_1_score}/100")
    
    if step1_1_score >= 75:
        step1_1_status = "✅ BALANCE ERREICHT - Bereit für Schritt 2"
        next_action = "Proceed to Schritt 2: Return-Enhancement"
    elif step1_1_score >= 60:
        step1_1_status = "🔄 GUTE BALANCE - Minor Tweaks möglich"
        next_action = "Proceed to Schritt 2 mit leichten Adjustments"
    else:
        step1_1_status = "❌ WEITERE BALANCE-OPTIMIERUNG ERFORDERLICH"
        next_action = "Iterate on balance parameters"
    
    print(f"Status: {step1_1_status}")
    print(f"Nächste Aktion: {next_action}")
    print()
    
    # Export Step 1.1 results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"step1_1_balanced_optimization_{timestamp}.json"
    
    export_data = {
        'step': '1.1',
        'optimization_focus': 'Balanced Signal Enhancement with Risk Control',
        'strategy_info': results['strategy_info'],
        'performance_metrics': metrics,
        'balance_summary': balance,
        'signal_statistics': signal_stats,
        'step1_1_assessment': {
            'score': step1_1_score,
            'status': step1_1_status,
            'improvements': {
                'signal_improvement': signal_improvement,
                'return_improvement': return_improvement,
                'risk_controlled': risk_controlled,
                'quality_maintained': quality_maintained
            },
            'next_action': next_action,
            'balance_achieved': step1_1_score >= 60
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"💾 SCHRITT 1.1 Ergebnisse exportiert: {filename}")
    
    return metrics, step1_1_score, step1_1_status


if __name__ == "__main__":
    asyncio.run(run_balanced_step1_1())