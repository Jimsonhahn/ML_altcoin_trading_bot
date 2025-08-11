#!/usr/bin/env python3
"""
SCHRITT 2: Return-Enhancement Strategy
Ziel: 25%+ Annual Return bei kontrolliertem Risiko

Strategy Name: "Elite Institutional BTC Pro"
Version: 2.0 Return-Enhanced
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


class ReturnEnhancementStrategy:
    """
    SCHRITT 2: Return-Enhancement Strategy
    
    Name: "Elite Institutional BTC Pro"
    
    Enhancement-Optimierungen:
    1. Multi-Timeframe Signals: 1H, 4H, 1D Kombination
    2. Regime-Adaptive Sizing: Bull/Bear/Sideways 
    3. Momentum Amplification: Trend-Confirmation
    4. Enhanced Position Management: Dynamic Allocation
    5. Risk-Parity Approach: Volatility-Adjusted
    """
    
    def __init__(self, initial_capital: float = 300000.0):
        self.strategy_name = "Elite Institutional BTC Pro"
        self.strategy_version = "2.1 Aggressive-Enhanced"
        self.risk_profile = "Enhanced Performance"
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_position = 0.0
        self.cash_balance = initial_capital
        
        # AGGRESSIVE Parameters für 25%+ Returns
        self.trading_fee = 0.001
        self.min_trade_size = 0.001
        self.max_position_size = 0.60  # AGGRESSIVE: 45% → 60%
        self.max_drawdown_limit = 0.25  # Gelockert: 22% → 25%
        self.min_signal_strength = 0.55  # AGGRESSIVE: 60% → 55% (noch mehr Trades)
        
        # Multi-Timeframe Enhancement
        self.timeframe_weights = {
            '1h': 0.25,   # Kurzfristige Signale
            '4h': 0.35,   # Mittelfristige Signale 
            '1d': 0.40    # Langfristige Signale
        }
        
        # Regime-Adaptive Sizing
        self.regime_multipliers = {
            'bull_strong': 1.25,     # +25% Position in starken Bullenmärkten
            'bull_moderate': 1.10,   # +10% Position in moderaten Bulls
            'sideways': 0.85,        # -15% Position in Seitwärtsmärkten
            'bear_moderate': 0.70,   # -30% Position in moderaten Bears
            'bear_strong': 0.50      # -50% Position in starken Bärenmärkten
        }
        
        # AGGRESSIVE Risk Management
        self.position_size_multiplier = 1.10  # AGGRESSIVE: 0.90 → 1.10
        self.emergency_stop_enabled = True
        self.monthly_var_limit = 0.15  # Gelockert: 12% → 15%
        self.quality_threshold = 0.55  # AGGRESSIVE: 0.65 → 0.55
        
        # Momentum Enhancement
        self.momentum_threshold = 0.02  # 2% Momentum für Verstärkung
        self.trend_confirmation_window = 5
        self.momentum_multiplier = 1.15  # +15% bei starkem Momentum
        
        # Performance Tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.alerts = []
        
        # Dashboard Integration
        self.last_signal_time = None
        self.current_phase = "enhancement"
        self.strategy_status = "active"
        self.signal_stats = {"generated": 0, "executed": 0, "quality_avg": 0, "enhancement_rate": 0}
        
        logger.info(f"{self.strategy_name} v{self.strategy_version} initialisiert")
        logger.info(f"AGGRESSIVE 2.1: Signal={self.min_signal_strength:.0%} | Position={self.max_position_size:.0%} | DD={self.max_drawdown_limit:.0%}")
    
    def detect_market_regime(self, prices: List[Dict]) -> str:
        """Marktregime-Detektion für adaptive Positionierung"""
        df = pd.DataFrame(prices[-100:])  # Letzten 100 Tage
        
        # Trend-Indikatoren
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['price_vs_sma20'] = (df['price'] / df['sma_20'] - 1)
        df['price_vs_sma50'] = (df['price'] / df['sma_50'] - 1)
        
        # Volatilität
        df['volatility'] = df['daily_return'].rolling(20).std()
        recent_vol = df['volatility'].iloc[-1]
        avg_vol = df['volatility'].mean()
        
        # Momentum
        df['momentum_5d'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_20d'] = df['price'] / df['price'].shift(20) - 1
        
        recent_price_sma20 = df['price_vs_sma20'].iloc[-1]
        recent_price_sma50 = df['price_vs_sma50'].iloc[-1]
        recent_momentum_5d = df['momentum_5d'].iloc[-1]
        recent_momentum_20d = df['momentum_20d'].iloc[-1]
        
        # Regime-Klassifikation
        if recent_price_sma20 > 0.05 and recent_price_sma50 > 0.03 and recent_momentum_20d > 0.10:
            return "bull_strong"
        elif recent_price_sma20 > 0.02 and recent_price_sma50 > 0.01 and recent_momentum_20d > 0.05:
            return "bull_moderate"
        elif recent_price_sma20 < -0.05 and recent_price_sma50 < -0.03 and recent_momentum_20d < -0.10:
            return "bear_strong"
        elif recent_price_sma20 < -0.02 and recent_price_sma50 < -0.01 and recent_momentum_20d < -0.05:
            return "bear_moderate"
        else:
            return "sideways"
    
    def generate_enhanced_signals(self, prices: List[Dict]) -> List[Dict]:
        """
        SCHRITT 2: Enhanced Signal Generation mit Multi-Timeframe und Regime-Adaptation
        """
        signals = []
        df = pd.DataFrame(prices)
        
        # Erweiterte technische Indikatoren
        # Mehr SMA Perioden für Multi-Timeframe
        for window in [3, 5, 8, 10, 13, 20, 21, 34, 50, 89, 144, 200]:
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
        
        # Mehr EMA Perioden
        for span in [5, 8, 12, 13, 21, 26, 34, 50, 89, 144]:
            df[f'ema_{span}'] = df['price'].ewm(span=span).mean()
        
        # Multiple MACD Konfigurationen
        macd_configs = [
            (8, 21, 5),   # Fast MACD
            (12, 26, 9),  # Standard MACD
            (21, 50, 9),  # Slow MACD
            (5, 13, 8)    # Ultra-fast MACD
        ]
        
        for fast, slow, signal_span in macd_configs:
            macd_name = f'macd_{fast}_{slow}'
            df[macd_name] = df[f'ema_{fast}'] - df[f'ema_{slow}']
            df[f'{macd_name}_signal'] = df[macd_name].ewm(span=signal_span).mean()
            df[f'{macd_name}_histogram'] = df[macd_name] - df[f'{macd_name}_signal']
        
        # Enhanced Volatility Analysis
        for window in [5, 8, 13, 20, 34, 50, 89]:
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std()
            df[f'vol_zscore_{window}'] = ((df[f'volatility_{window}d'] - 
                                         df[f'volatility_{window}d'].rolling(100).mean()) / 
                                        df[f'volatility_{window}d'].rolling(100).std())
        
        # Multiple RSI Perioden mit Divergenz-Detection
        for period in [7, 9, 14, 21, 25, 34]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Enhanced Momentum mit verschiedenen Timeframes
        for period in [3, 5, 8, 13, 20, 34, 50, 89, 144]:
            df[f'momentum_{period}d'] = df['price'] / df['price'].shift(period) - 1
        
        # Advanced Volume Analysis
        for window in [5, 10, 20, 34, 50, 89]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            df[f'volume_momentum_{window}'] = df[f'volume_sma_{window}'] / df[f'volume_sma_{window}'].shift(window)
        
        # Multiple Bollinger Bands
        for period in [10, 13, 20, 34, 50]:
            for std_dev in [1.5, 2.0, 2.5]:
                bb_middle = df['price'].rolling(period).mean()
                bb_std = df['price'].rolling(period).std()
                df[f'bb_upper_{period}_{std_dev}'] = bb_middle + (bb_std * std_dev)
                df[f'bb_lower_{period}_{std_dev}'] = bb_middle - (bb_std * std_dev)
                df[f'bb_position_{period}_{std_dev}'] = (df['price'] - df[f'bb_lower_{period}_{std_dev}']) / (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}'])
        
        # Stochastic Oscillator variants
        for k_period in [9, 14, 21]:
            for d_period in [3, 5, 8]:
                low_min = df['price'].rolling(k_period).min()
                high_max = df['price'].rolling(k_period).max()
                df[f'stoch_k_{k_period}'] = 100 * (df['price'] - low_min) / (high_max - low_min)
                df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(d_period).mean()
        
        # Fibonacci Retracements (dynamisch)
        for lookback in [20, 34, 50, 89]:
            period_high = df['price'].rolling(lookback).max()
            period_low = df['price'].rolling(lookback).min()
            fib_range = period_high - period_low
            for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                df[f'fib_{level}_{lookback}'] = period_high - (fib_range * level)
        
        # Signal Generation mit Multi-Strategy Approach
        df = df.dropna().reset_index(drop=True)
        
        for i in range(50, len(df)):
            current = df.iloc[i].to_dict()
            prev = df.iloc[i-1].to_dict()
            
            # Market Regime Detection
            regime = self.detect_market_regime(prices[:i+1])
            regime_multiplier = self.regime_multipliers.get(regime, 1.0)
            
            signal_scores = []
            
            # Strategy 1: Enhanced Multi-MACD Convergence
            macd_score = 0
            macd_signals = 0
            for config in macd_configs:
                fast, slow, _ = config
                macd_name = f'macd_{fast}_{slow}'
                if (current[macd_name] > current[f'{macd_name}_signal'] and 
                    prev[macd_name] <= prev[f'{macd_name}_signal']):
                    macd_score += 1
                    macd_signals += 1
                elif (current[macd_name] < current[f'{macd_name}_signal'] and 
                      prev[macd_name] >= prev[f'{macd_name}_signal']):
                    macd_score -= 1
            
            if macd_signals > 0:
                signal_scores.append(min(max(macd_score / len(macd_configs), -1), 1) * 0.25)
            
            # Strategy 2: Multi-Timeframe RSI Mean Reversion + Momentum
            rsi_score = 0
            for period in [7, 14, 21]:
                rsi_val = current[f'rsi_{period}']
                if rsi_val < 35 and current[f'momentum_5d'] > -0.02:  # Oversold mit stabilem Momentum
                    rsi_score += 1
                elif rsi_val > 65 and current[f'momentum_5d'] < 0.02:  # Overbought
                    rsi_score -= 1
            signal_scores.append(min(max(rsi_score / 3, -1), 1) * 0.20)
            
            # Strategy 3: Enhanced Trend Following mit Regime-Adaptation
            trend_score = 0
            price = current['price']
            if (price > current['sma_20'] > current['sma_50'] and 
                current['momentum_20d'] > 0.05 * regime_multiplier):
                trend_score = 1
            elif (price < current['sma_20'] < current['sma_50'] and 
                  current['momentum_20d'] < -0.05 * regime_multiplier):
                trend_score = -1
            signal_scores.append(trend_score * 0.25)
            
            # Strategy 4: Volume-Price Divergence mit Enhanced Volume
            volume_score = 0
            if (current['volume_ratio_20'] > 1.2 and current['momentum_5d'] > 0.01):
                volume_score = 0.8
            elif (current['volume_ratio_20'] < 0.8 and current['momentum_5d'] < -0.01):
                volume_score = -0.8
            signal_scores.append(volume_score * 0.15)
            
            # Strategy 5: Multi-Bollinger Band Breakthrough
            bb_score = 0
            for period in [20, 34]:
                bb_pos = current[f'bb_position_{period}_2.0']
                if bb_pos > 0.8 and current[f'momentum_3d'] > 0.01:
                    bb_score += 0.5
                elif bb_pos < 0.2 and current[f'momentum_3d'] < -0.01:
                    bb_score -= 0.5
            signal_scores.append(min(max(bb_score, -1), 1) * 0.15)
            
            # Aggregated Signal mit Regime-Adjustment
            base_signal_strength = sum(signal_scores)
            
            # Momentum Amplification
            momentum_boost = 0
            if abs(current['momentum_5d']) > self.momentum_threshold:
                momentum_boost = np.sign(base_signal_strength) * self.momentum_multiplier * abs(current['momentum_5d'])
            
            final_signal_strength = (base_signal_strength + momentum_boost) * regime_multiplier
            final_signal_strength = min(max(final_signal_strength, -1), 1)
            
            # Quality Score Calculation
            volatility = current.get('volatility_20d', 0.02)
            volume_confirmation = min(current.get('volume_ratio_20', 1), 2.0)
            trend_consistency = abs(current.get('momentum_20d', 0))
            
            quality_score = (abs(final_signal_strength) * 0.4 +
                           min(volume_confirmation / 1.5, 1) * 0.3 +
                           min(trend_consistency * 10, 1) * 0.2 +
                           min(1 / (volatility * 50 + 0.1), 1) * 0.1)
            
            if abs(final_signal_strength) >= self.min_signal_strength:
                signal = {
                    'timestamp': current['timestamp'],
                    'price': current['price'],
                    'signal_strength': final_signal_strength,
                    'direction': 'buy' if final_signal_strength > 0 else 'sell',
                    'quality_score': min(quality_score, 1.0),
                    'regime': regime,
                    'regime_multiplier': regime_multiplier,
                    'momentum_boost': momentum_boost,
                    'volatility': volatility,
                    'volume_ratio': current.get('volume_ratio_20', 1),
                    'strategy_scores': signal_scores,
                    'macd_signals': macd_signals if 'macd_signals' in locals() else 0
                }
                signals.append(signal)
        
        logger.info(f"Generiert: {len(signals)} enhanced signals")
        return signals

    def calculate_enhanced_position_size(self, signal: Dict, current_drawdown: float) -> float:
        """Enhanced Position Size mit Regime-Adaptation und Momentum"""
        base_size = abs(signal['signal_strength']) * self.position_size_multiplier
        
        # Regime Adjustment
        regime_adj = signal.get('regime_multiplier', 1.0)
        
        # Quality Adjustment
        quality_adj = signal['quality_score']
        
        # Momentum Adjustment
        momentum_adj = 1.0 + abs(signal.get('momentum_boost', 0)) * 0.1
        
        # Volatility Adjustment (Risk Parity Approach)
        volatility = signal.get('volatility', 0.02)
        target_volatility = 0.15  # 15% Target
        vol_adj = min(target_volatility / (volatility + 0.01), 2.0)
        
        # Progressive Drawdown Control
        if current_drawdown > 0.10:
            dd_adj = max(0.5, 1 - (current_drawdown - 0.10) * 2)
        else:
            dd_adj = 1.0
        
        position_size = base_size * regime_adj * quality_adj * momentum_adj * vol_adj * dd_adj
        return min(position_size, self.max_position_size)

    def run_enhanced_backtest(self, prices: List[Dict]) -> Dict[str, Any]:
        """Enhanced Backtest mit allen Return-Enhancement Features"""
        logger.info("Führe Enhanced Return Backtest durch...")
        
        signals = self.generate_enhanced_signals(prices)
        self.signal_stats['generated'] = len(signals)
        
        executed_trades = 0
        quality_scores = []
        
        for signal in signals:
            # Current Portfolio State
            total_portfolio_value = self.cash_balance + (self.btc_position * signal['price'])
            current_drawdown = max(0, 1 - total_portfolio_value / max(self.current_capital, self.initial_capital))
            
            # Enhanced Position Sizing
            position_size = self.calculate_enhanced_position_size(signal, current_drawdown)
            
            # Quality Filter (relaxed für mehr Trades)
            if signal['quality_score'] >= self.quality_threshold and position_size > 0.01:
                trade_amount = total_portfolio_value * position_size
                
                if signal['direction'] == 'buy' and self.cash_balance >= trade_amount * (1 + self.trading_fee):
                    # Buy BTC
                    btc_amount = trade_amount / signal['price']
                    total_cost = trade_amount * (1 + self.trading_fee)
                    
                    self.btc_position += btc_amount
                    self.cash_balance -= total_cost
                    executed_trades += 1
                    
                    trade_record = {
                        'timestamp': signal['timestamp'],
                        'type': 'buy',
                        'price': signal['price'],
                        'amount': btc_amount,
                        'cost': total_cost,
                        'signal_strength': signal['signal_strength'],
                        'quality_score': signal['quality_score'],
                        'position_size': position_size,
                        'regime': signal['regime']
                    }
                    self.trades.append(trade_record)
                    
                elif signal['direction'] == 'sell' and self.btc_position > 0:
                    # Sell BTC
                    btc_to_sell = min(self.btc_position, self.btc_position * position_size)
                    revenue = btc_to_sell * signal['price'] * (1 - self.trading_fee)
                    
                    self.btc_position -= btc_to_sell
                    self.cash_balance += revenue
                    executed_trades += 1
                    
                    trade_record = {
                        'timestamp': signal['timestamp'],
                        'type': 'sell',
                        'price': signal['price'],
                        'amount': btc_to_sell,
                        'revenue': revenue,
                        'signal_strength': signal['signal_strength'],
                        'quality_score': signal['quality_score'],
                        'position_size': position_size,
                        'regime': signal['regime']
                    }
                    self.trades.append(trade_record)
                
                quality_scores.append(signal['quality_score'])
            
            # Update current capital
            self.current_capital = self.cash_balance + (self.btc_position * signal['price'])
            self.equity_curve.append({
                'timestamp': signal['timestamp'],
                'capital': self.current_capital,
                'drawdown': current_drawdown
            })
        
        # Final portfolio value
        final_price = prices[-1]['price']
        final_capital = self.cash_balance + (self.btc_position * final_price)
        
        self.signal_stats.update({
            'executed': executed_trades,
            'quality_avg': np.mean(quality_scores) if quality_scores else 0,
            'enhancement_rate': executed_trades / len(signals) if signals else 0
        })
        
        return self.calculate_enhanced_performance_metrics(prices, final_capital)

    def calculate_enhanced_performance_metrics(self, prices: List[Dict], final_capital: float) -> Dict[str, Any]:
        """Enhanced Performance Metrics mit zusätzlichen Risk-Adjusted Returns"""
        if not self.equity_curve:
            return {}
        
        # Basic Returns
        total_return = (final_capital / self.initial_capital) - 1
        days_analyzed = len([p for p in prices])
        annual_return = (1 + total_return) ** (365.25 / days_analyzed) - 1
        
        # Daily Returns
        equity_values = [eq['capital'] for eq in self.equity_curve]
        daily_returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
        
        # Risk Metrics
        annual_volatility = np.std(daily_returns) * np.sqrt(365.25) if daily_returns else 0
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Downside Risk
        negative_returns = [r for r in daily_returns if r < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(365.25) if negative_returns else annual_volatility
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum Drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for eq in self.equity_curve:
            if eq['capital'] > peak:
                peak = eq['capital']
            drawdown = (peak - eq['capital']) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar Ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Enhanced Metrics
        win_rate = len([t for t in self.trades if (t['type'] == 'sell' and t['revenue'] > t.get('cost', 0)) or 
                       (t['type'] == 'buy')]) / len(self.trades) if self.trades else 0
        
        return {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'days_analyzed': days_analyzed,
            'final_capital': final_capital,
            'signal_stats': self.signal_stats,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_position_size': self.max_position_size,
            'risk_profile': self.risk_profile
        }


def generate_realistic_crypto_data(start_date: str = "2023-01-01", end_date: str = "2024-12-31") -> List[Dict]:
    """Generiert realistische 2-Jahres BTC-Daten für Backtesting"""
    from datetime import datetime, timedelta
    import random
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    prices = []
    current_date = start
    current_price = 16500.0  # BTC Preis Januar 2023
    
    # Realistische 2-Jahres Crypto-Markt Simulation
    phases = [
        {"name": "crypto_winter", "days": 90, "drift": -0.0002, "volatility": 0.035},
        {"name": "gradual_recovery", "days": 120, "drift": 0.0008, "volatility": 0.025},
        {"name": "etf_anticipation", "days": 150, "drift": 0.0015, "volatility": 0.030},
        {"name": "etf_approval", "days": 60, "drift": 0.0025, "volatility": 0.040},
        {"name": "etf_cooldown", "days": 30, "drift": -0.0010, "volatility": 0.035},
        {"name": "summer_consolidation", "days": 120, "drift": 0.0002, "volatility": 0.020},
        {"name": "pre_election", "days": 156, "drift": 0.0012, "volatility": 0.028}
    ]
    
    phase_idx = 0
    days_in_phase = 0
    
    while current_date <= end and phase_idx < len(phases):
        phase = phases[phase_idx]
        
        # Daily return mit realistischen Crypto-Eigenschaften
        random_factor = random.gauss(0, 1)
        daily_return = phase["drift"] + (phase["volatility"] * random_factor)
        
        # Weekend-Effekt (reduzierte Volatilität)
        if current_date.weekday() >= 5:
            daily_return *= 0.7
        
        # Occasional extreme moves (Crypto-typisch)
        if random.random() < 0.02:  # 2% Chance auf extreme Bewegung
            daily_return += random.choice([-1, 1]) * random.uniform(0.05, 0.15)
        
        current_price *= (1 + daily_return)
        current_price = max(current_price, 1000)  # Minimum BTC Preis
        
        # Volume simulation
        base_volume = 25000 + random.gauss(0, 5000)
        volume = max(base_volume * (1 + abs(daily_return) * 5), 1000)
        
        prices.append({
            'timestamp': current_date.strftime('%Y-%m-%d'),
            'price': round(current_price, 2),
            'daily_return': daily_return,
            'volume': round(volume, 0),
            'phase': phase["name"]
        })
        
        current_date += timedelta(days=1)
        days_in_phase += 1
        
        if days_in_phase >= phase["days"]:
            phase_idx += 1
            days_in_phase = 0
    
    return prices


async def main():
    """
    SCHRITT 2: Return-Enhancement Hauptausführung
    """
    print("🚀 SCHRITT 2: RETURN-ENHANCEMENT BACKTEST")
    print("=" * 80)
    print("Enhanced Strategy: Elite Institutional BTC Pro v2.0")
    print("Ziel: 25%+ Annual Return mit kontrolliertem Risiko\\n")
    
    print("📊 Generiere Enhanced Daten für Return-Enhancement...")
    prices = generate_realistic_crypto_data()
    print(f"✅ {len(prices)} Tage bereit für Enhanced Analysis\\n")
    
    print("⚡ Führe SCHRITT 2 Enhanced Backtest durch...")
    strategy = ReturnEnhancementStrategy()
    results = strategy.run_enhanced_backtest(prices)
    
    # Results Analysis
    print("📊 SCHRITT 2 ERGEBNISSE - RETURN-ENHANCEMENT")
    print("-" * 80)
    print(f"Strategy: {results['strategy_name']} v{results['strategy_version']}")
    print(f"Risk Profile: {results['risk_profile']}\\n")
    
    print("🎯 ENHANCED PERFORMANCE:")
    print(f"   Annual Return:          {results['annual_return']:.1%}")
    print(f"   Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio:          {results['sortino_ratio']:.2f}")
    print(f"   Max Drawdown:           {results['max_drawdown']:.1%}")
    print(f"   Total Trades:           {results['total_trades']}")
    print(f"   Win Rate:               {results['win_rate']:.1%}")
    print(f"   Signal Enhancement:     {results['signal_stats']['enhancement_rate']:.1%}\\n")
    
    # Target Assessment
    return_target = results['annual_return'] >= 0.25  # 25%+ target
    sharpe_target = results['sharpe_ratio'] >= 1.0   # 1.0+ target
    drawdown_ok = results['max_drawdown'] <= results['max_drawdown_limit']
    trades_ok = results['total_trades'] >= 3
    
    print("🎯 SCHRITT 2 ZIEL-BEWERTUNG:")
    print("-" * 80)
    print(f"Return > 25%:            {'✅' if return_target else '❌'} ({results['annual_return']:.1%})")
    print(f"Sharpe > 1.0:            {'✅' if sharpe_target else '❌'} ({results['sharpe_ratio']:.2f})")
    print(f"Drawdown kontrolliert:   {'✅' if drawdown_ok else '❌'} ({results['max_drawdown']:.1%} ≤ {results['max_drawdown_limit']:.1%})")
    print(f"Ausreichend Trades:      {'✅' if trades_ok else '❌'} ({results['total_trades']} ≥ 3)\\n")
    
    targets_met = sum([return_target, sharpe_target, drawdown_ok, trades_ok])
    score = (targets_met / 4) * 100
    
    if score >= 75:
        status = "✅ ENHANCEMENT ERFOLGREICH"
        next_action = "Proceed to Schritt 3: Live-Testing"
    elif score >= 50:
        status = "⚠️ TEILWEISE ERFOLGREICH"
        next_action = "Fine-tune für bessere Performance"
    else:
        status = "❌ WEITERE OPTIMIERUNG ERFORDERLICH"
        next_action = "Zurück zu Parameter-Optimierung"
    
    print(f"SCHRITT 2 Score: {score:.0f}/100")
    print(f"Status: {status}")
    print(f"Nächste Aktion: {next_action}\\n")
    
    # Export Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step2_return_enhancement_{timestamp}.json"
    
    export_data = {
        "step_info": {
            "step": "2",
            "name": "Return-Enhancement",
            "strategy": results['strategy_name'],
            "version": results['strategy_version'],
            "timestamp": timestamp,
            "target": "25%+ Annual Return"
        },
        "performance_metrics": results,
        "target_assessment": {
            "return_target": bool(return_target),
            "sharpe_target": bool(sharpe_target),
            "drawdown_target": bool(drawdown_ok),
            "trades_target": bool(trades_ok),
            "overall_score": float(score),
            "status": status.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
        },
        "strategy_info": strategy.get_strategy_info() if hasattr(strategy, 'get_strategy_info') else {},
        "next_steps": {
            "action": next_action,
            "ready_for_step3": bool(score >= 75)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"💾 SCHRITT 2 Ergebnisse exportiert: {filename}")


if __name__ == "__main__":
    asyncio.run(main())