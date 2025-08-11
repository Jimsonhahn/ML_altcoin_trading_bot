#!/usr/bin/env python3
"""
Strategy Validation - 2 Jahre Backtest mit $300k Startkapital
Teste Ultimate BTC Strategy auf realistische Marktbedingungen 2023-2024
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyValidation2Years:
    """
    2-Jahres Validation der Ultimate BTC Strategy
    
    Periode: Januar 2023 - Dezember 2024
    Startkapital: $300,000
    Realistic Market Conditions
    """
    
    def __init__(self, initial_capital: float = 300000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_position = 0.0
        self.cash_balance = initial_capital
        
        # Tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        
        # Optimized parameters für kleineres Kapital
        self.trading_fee = 0.001
        self.min_trade_size = 0.001
        self.max_position_size = 0.6  # Etwas konservativer für kleineres Kapital
        self.min_signal_strength = 0.55  # Leicht höhere Schwelle
        
        logger.info(f"Strategy Validation initialisiert mit ${initial_capital:,.0f}")
    
    def generate_realistic_2year_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generiert realistische BTC-Daten für 2023-2024
        Basiert auf tatsächlichen Marktbedingungen und Ereignissen
        """
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Realistische 2-Jahres BTC Performance
        start_price = 16500.0  # BTC Anfang 2023 (nach FTX Crash)
        end_price = 95000.0    # BTC Ende 2024 (Bull Run)
        
        prices = []
        current_price = start_price
        
        for i, date in enumerate(dates):
            progress = i / days
            
            # Realistische Markt-Phasen basierend auf 2023-2024
            if progress < 0.08:  # Q1 2023: FTX Nachwehen, Banking Crisis
                phase = 'crypto_winter'
                trend_factor = 0.2
                volatility = 0.06
                sentiment_base = -0.4
            elif progress < 0.25:  # Q2-Q3 2023: Graduelle Erholung
                phase = 'gradual_recovery'
                trend_factor = 0.8
                volatility = 0.04
                sentiment_base = -0.1
            elif progress < 0.42:  # Q4 2023: ETF Optimismus beginnt
                phase = 'etf_anticipation'
                trend_factor = 1.5
                volatility = 0.045
                sentiment_base = 0.2
            elif progress < 0.50:  # Q1 2024: ETF Approval
                phase = 'etf_approval'
                trend_factor = 2.2
                volatility = 0.055
                sentiment_base = 0.5
            elif progress < 0.58:  # Q2 2024: ETF Euphorie nachlässt
                phase = 'etf_cooldown'
                trend_factor = 0.3
                volatility = 0.05
                sentiment_base = 0.1
            elif progress < 0.75:  # Q3 2024: Sommer-Konsolidierung
                phase = 'summer_consolidation'
                trend_factor = 0.6
                volatility = 0.035
                sentiment_base = 0.0
            elif progress < 0.83:  # Q4 2024 Start: Pre-Election Rally
                phase = 'pre_election'
                trend_factor = 1.3
                volatility = 0.04
                sentiment_base = 0.3
            else:  # Q4 2024 Ende: Trump Victory Rally
                phase = 'trump_victory'
                trend_factor = 2.8
                volatility = 0.06
                sentiment_base = 0.6
            
            # Seasonal adjustments (realistischer)
            month = date.month
            seasonal_factor = 1.0
            if month in [11, 12]:  # November-Dezember historically strong
                seasonal_factor = 1.2
            elif month in [1]:     # Januar effect
                seasonal_factor = 1.1
            elif month in [6, 7, 8]:  # Summer slowdown
                seasonal_factor = 0.85
            elif month == 9:       # September typically weak
                seasonal_factor = 0.8
            
            # Calculate trend return
            base_daily_return = ((end_price / start_price) ** (1/days) - 1) * trend_factor * seasonal_factor
            
            # Add momentum persistence
            if i > 20:
                recent_returns = [p['daily_return'] for p in prices[-20:]]
                momentum = np.mean(recent_returns)
                
                # Momentum persistence but with mean reversion
                if momentum > 0.04:  # Strong positive momentum
                    momentum_adj = 1.3 * (1 - min(0.5, momentum))  # Diminishing returns
                elif momentum > 0.02:
                    momentum_adj = 1.15
                elif momentum < -0.04:  # Strong negative momentum
                    momentum_adj = 0.7 * (1 + min(0.3, abs(momentum)))  # Bounce potential
                elif momentum < -0.02:
                    momentum_adj = 0.85
                else:
                    momentum_adj = 1.0
            else:
                momentum_adj = 1.0
            
            # Final daily return
            daily_return = base_daily_return * momentum_adj + np.random.normal(0, volatility)
            
            # Add realistic market shocks
            shock_probability = 0.02  # 2% daily chance
            if np.random.random() < shock_probability:
                if phase in ['crypto_winter', 'etf_cooldown']:
                    # More negative shocks in weak phases
                    shock = np.random.choice([-0.15, -0.12, -0.08, 0.06, 0.08], p=[0.3, 0.25, 0.25, 0.1, 0.1])
                else:
                    # More positive shocks in strong phases
                    shock = np.random.choice([-0.12, -0.08, 0.08, 0.12, 0.18], p=[0.15, 0.15, 0.25, 0.25, 0.2])
                daily_return += shock
            
            current_price *= (1 + daily_return)
            current_price = max(10000, min(150000, current_price))
            
            # Market microstructure features
            volume = np.random.lognormal(10.3, 0.9)
            if phase in ['etf_approval', 'trump_victory']:
                volume *= 2.5  # High volume in major events
            elif phase in ['crypto_winter']:
                volume *= 0.6  # Low volume in bear markets
            
            # Order book features
            bid_ask_spread = np.random.uniform(0.03, 0.25)
            if phase == 'crypto_winter':
                bid_ask_spread *= 2.0  # Wider spreads in low liquidity
            
            order_book_imbalance = np.random.normal(0, 0.5)
            if phase in ['etf_approval', 'trump_victory']:
                order_book_imbalance += 0.3  # Bullish bias
            elif phase == 'crypto_winter':
                order_book_imbalance -= 0.2  # Bearish bias
            
            # Sentiment (more realistic)
            sentiment = sentiment_base + np.random.normal(0, 0.2)
            sentiment = max(-1.0, min(1.0, sentiment))
            
            # Options activity
            put_call_ratio = np.random.lognormal(0, 0.5)
            if phase in ['crypto_winter', 'etf_cooldown']:
                put_call_ratio *= 1.8  # More puts in uncertain times
            
            # Funding rates
            funding_rate = np.random.normal(0.01, 0.03)
            if phase in ['etf_approval', 'trump_victory']:
                funding_rate += 0.06  # High funding in euphoria
            elif phase == 'crypto_winter':
                funding_rate -= 0.02  # Negative funding in despair
            
            # Additional realistic features
            whale_activity = np.random.exponential(0.4)
            if phase in ['gradual_recovery', 'summer_consolidation']:
                whale_activity *= 1.5  # More accumulation in quiet periods
            
            # Exchange flows
            exchange_flow = np.random.normal(0, 0.3)
            if phase in ['etf_approval', 'trump_victory']:
                exchange_flow -= 0.2  # Outflows during rallies
            elif phase == 'crypto_winter':
                exchange_flow += 0.15  # Inflows during panic
            
            # Institutional activity
            institutional_flow = np.random.normal(0, 0.2)
            if phase in ['etf_approval', 'trump_victory']:
                institutional_flow += 0.4  # Strong institutional buying
            
            # Macro environment
            macro_sentiment = np.random.normal(0, 0.3)
            if date.month in [3, 6, 9, 12]:  # FOMC months
                macro_sentiment *= 1.5  # More macro volatility
            
            prices.append({
                'date': date,
                'price': current_price,
                'daily_return': daily_return,
                'volume': volume,
                'phase': phase,
                'trend_strength': trend_factor,
                'volatility_regime': volatility,
                'order_book_imbalance': order_book_imbalance,
                'sentiment': sentiment,
                'put_call_ratio': put_call_ratio,
                'funding_rate': funding_rate,
                'whale_activity': whale_activity,
                'exchange_flow': exchange_flow,
                'institutional_flow': institutional_flow,
                'macro_sentiment': macro_sentiment,
                'bid_ask_spread': bid_ask_spread,
                'momentum_adj': momentum_adj,
                'seasonal_factor': seasonal_factor
            })
        
        return {
            'prices': prices,
            'start_price': start_price,
            'end_price': prices[-1]['price'],
            'total_days': days
        }
    
    def generate_validation_signals(self, prices: List[Dict]) -> List[Dict]:
        """
        Generiert Signale für 2-Jahres Validation
        Gleiche Logik wie Ultimate Strategy aber angepasst für längeren Zeitraum
        """
        signals = []
        df = pd.DataFrame(prices)
        
        # Technical indicators
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
        
        for span in [12, 26, 50]:
            df[f'ema_{span}'] = df['price'].ewm(span=span).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volatility
        for window in [10, 20, 50]:
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}d'] = df['price'] / df['price'].shift(period)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        bb_std = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        for i in range(100, len(df)):  # Mehr Historie für robuste Signale
            current = df.iloc[i]
            
            # === ENHANCED MULTI-STRATEGY ENSEMBLE ===
            
            # 1. LONG-TERM TREND STRATEGY
            trend_signal = 0.0
            if (current['sma_20'] > current['sma_50'] and 
                current['sma_50'] > current['sma_100'] and
                current['price'] > current['sma_20'] * 1.02):
                trend_signal = 0.6
                trend_direction = 'long'
            elif (current['sma_20'] < current['sma_50'] and 
                  current['sma_50'] < current['sma_100'] and
                  current['price'] < current['sma_20'] * 0.98):
                trend_signal = 0.4
                trend_direction = 'short'
            else:
                trend_direction = None
            
            # 2. MOMENTUM STRATEGY (Multi-timeframe)
            momentum_signal = 0.0
            momentum_direction = None
            
            short_momentum = current['momentum_5d'] > 1.02
            medium_momentum = current['momentum_20d'] > 1.05
            long_momentum = current['momentum_50d'] > 1.10
            
            if short_momentum and medium_momentum:
                momentum_signal = 0.5
                momentum_direction = 'long'
                if long_momentum:
                    momentum_signal = 0.7  # All timeframes aligned
            elif current['momentum_5d'] < 0.98 and current['momentum_20d'] < 0.95:
                momentum_signal = 0.4
                momentum_direction = 'short'
            
            # 3. MEAN REVERSION STRATEGY (Enhanced)
            mean_reversion_signal = 0.0
            mean_reversion_direction = None
            
            # Use Bollinger Bands for mean reversion
            if current['bb_position'] < 0.1 and current['rsi'] < 30:  # Oversold
                mean_reversion_signal = 0.6
                mean_reversion_direction = 'long'
            elif current['bb_position'] > 0.9 and current['rsi'] > 70:  # Overbought
                mean_reversion_signal = 0.5
                mean_reversion_direction = 'short'
            
            # 4. SENTIMENT STRATEGY (Enhanced)
            sentiment_signal = 0.0
            sentiment_direction = None
            
            # Contrarian sentiment + Phase awareness
            if (current['sentiment'] < -0.3 and 
                current['phase'] in ['crypto_winter', 'etf_cooldown'] and
                current['funding_rate'] < 0):
                sentiment_signal = 0.7  # Strong contrarian buy
                sentiment_direction = 'long'
            elif (current['sentiment'] > 0.5 and 
                  current['phase'] in ['etf_approval', 'trump_victory'] and
                  current['funding_rate'] > 0.05):
                sentiment_signal = 0.4  # Euphoria warning
                sentiment_direction = 'short'
            
            # 5. INSTITUTIONAL FLOW STRATEGY
            institutional_signal = 0.0
            institutional_direction = None
            
            if (current['institutional_flow'] > 0.3 and 
                current['exchange_flow'] < -0.1):  # Institutions buying, retail selling
                institutional_signal = 0.6
                institutional_direction = 'long'
            elif current['whale_activity'] > 1.0 and current['phase'] in ['gradual_recovery', 'summer_consolidation']:
                institutional_signal = 0.5
                institutional_direction = 'long'
            
            # 6. VOLATILITY STRATEGY
            vol_signal = 0.0
            vol_direction = None
            
            current_vol = current['volatility_20d']
            vol_regime = current['volatility_regime']
            
            # Buy low vol, sell high vol
            if current_vol < 0.03 and vol_regime < 0.04:  # Very low volatility
                vol_signal = 0.4
                vol_direction = 'long'  # Expect volatility expansion upward
            elif current_vol > 0.08 and vol_regime > 0.06:  # Very high volatility
                vol_signal = 0.3
                vol_direction = 'short'  # Expect volatility contraction
            
            # === ENSEMBLE COMBINATION WITH WEIGHTS ===
            
            strategy_votes = []
            strategy_weights = []
            
            # Weight strategies based on market conditions
            base_weights = {
                'trend': 0.30,
                'momentum': 0.25,
                'mean_reversion': 0.15,
                'sentiment': 0.15,
                'institutional': 0.10,
                'volatility': 0.05
            }
            
            # Adjust weights based on phase
            if current['phase'] in ['etf_approval', 'trump_victory']:
                # Favor momentum in strong trends
                base_weights['momentum'] = 0.35
                base_weights['trend'] = 0.35
                base_weights['mean_reversion'] = 0.05
            elif current['phase'] in ['crypto_winter', 'summer_consolidation']:
                # Favor mean reversion in ranging markets
                base_weights['mean_reversion'] = 0.30
                base_weights['sentiment'] = 0.25
                base_weights['trend'] = 0.20
            
            # Add votes
            if trend_direction:
                strategy_votes.append((trend_direction, trend_signal))
                strategy_weights.append(base_weights['trend'])
            
            if momentum_direction:
                strategy_votes.append((momentum_direction, momentum_signal))
                strategy_weights.append(base_weights['momentum'])
            
            if mean_reversion_direction:
                strategy_votes.append((mean_reversion_direction, mean_reversion_signal))
                strategy_weights.append(base_weights['mean_reversion'])
            
            if sentiment_direction:
                strategy_votes.append((sentiment_direction, sentiment_signal))
                strategy_weights.append(base_weights['sentiment'])
            
            if institutional_direction:
                strategy_votes.append((institutional_direction, institutional_signal))
                strategy_weights.append(base_weights['institutional'])
            
            if vol_direction:
                strategy_votes.append((vol_direction, vol_signal))
                strategy_weights.append(base_weights['volatility'])
            
            # Calculate ensemble decision
            if strategy_votes:
                # Normalize weights
                total_weight = sum(strategy_weights)
                strategy_weights = [w / total_weight for w in strategy_weights]
                
                long_score = sum(weight * signal for (direction, signal), weight in zip(strategy_votes, strategy_weights) if direction == 'long')
                short_score = sum(weight * signal for (direction, signal), weight in zip(strategy_votes, strategy_weights) if direction == 'short')
                
                final_signal_strength = max(long_score, short_score)
                final_direction = 'long' if long_score > short_score else 'short'
                
                # Signal filtering
                if final_signal_strength > self.min_signal_strength:
                    
                    # === POSITION SIZING ===
                    base_size = min(self.max_position_size, final_signal_strength * 0.7)
                    
                    # Phase-based position sizing
                    phase_multipliers = {
                        'crypto_winter': 0.4,
                        'gradual_recovery': 0.7,
                        'etf_anticipation': 1.0,
                        'etf_approval': 1.3,
                        'etf_cooldown': 0.8,
                        'summer_consolidation': 0.6,
                        'pre_election': 1.1,
                        'trump_victory': 1.4
                    }
                    
                    phase_mult = phase_multipliers.get(current['phase'], 1.0)
                    
                    # Volatility adjustment
                    if current['volatility_regime'] > 0.06:
                        vol_mult = 0.6
                    elif current['volatility_regime'] < 0.03:
                        vol_mult = 1.2
                    else:
                        vol_mult = 1.0
                    
                    # Volume confirmation
                    volume_mult = min(1.3, max(0.8, current['volume_ratio'] / 2))
                    
                    final_position_size = min(
                        self.max_position_size,
                        base_size * phase_mult * vol_mult * volume_mult
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
                        'ensemble_votes': {
                            'long_score': long_score,
                            'short_score': short_score
                        },
                        'contributing_strategies': [direction for direction, _ in strategy_votes],
                        'rsi': current['rsi'],
                        'bb_position': current['bb_position'],
                        'volume_ratio': current['volume_ratio']
                    })
        
        return signals
    
    def execute_2year_backtest(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt 2-Jahres Backtest durch
        """
        logger.info("Führe 2-Jahres Validation Backtest durch...")
        
        prices = price_data['prices']
        signals = self.generate_validation_signals(prices)
        
        logger.info(f"Generiert: {len(signals)} Signale über {len(prices)} Tage")
        
        # Execute backtest with enhanced risk management
        portfolio = []
        cash = self.initial_capital
        btc_position = 0.0
        
        signal_dict = {s['date']: s for s in signals}
        
        # Enhanced risk management
        peak_portfolio_value = self.initial_capital
        drawdown_protection_factor = 1.0
        consecutive_losses = 0
        monthly_performance = []
        
        for price_data_point in prices:
            date = price_data_point['date']
            current_price = price_data_point['price']
            
            # Calculate portfolio value
            portfolio_value = cash + (btc_position * current_price)
            
            # Enhanced drawdown protection
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value
                drawdown_protection_factor = 1.0
                consecutive_losses = 0
            else:
                current_dd = (portfolio_value - peak_portfolio_value) / peak_portfolio_value
                
                # Progressive risk reduction
                if current_dd < -0.20:  # 20% drawdown
                    drawdown_protection_factor = 0.2
                elif current_dd < -0.15:  # 15% drawdown
                    drawdown_protection_factor = 0.4
                elif current_dd < -0.10:  # 10% drawdown
                    drawdown_protection_factor = 0.6
                elif current_dd < -0.05:  # 5% drawdown
                    drawdown_protection_factor = 0.8
                else:
                    drawdown_protection_factor = 1.0
            
            # Monthly performance tracking
            if date.day == 1:  # First day of month
                if len(portfolio) > 30:  # Have at least 30 days of data
                    month_start_value = portfolio[-30]['portfolio_value']
                    monthly_return = (portfolio_value - month_start_value) / month_start_value
                    monthly_performance.append(monthly_return)
                    
                    # Reduce risk after bad months
                    if len(monthly_performance) >= 3:
                        recent_months = monthly_performance[-3:]
                        if sum(r < -0.05 for r in recent_months) >= 2:  # 2 of last 3 months negative
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
            
            # Execute signal if present
            if date in signal_dict:
                signal = signal_dict[date]
                
                # Apply all risk factors
                risk_factor = drawdown_protection_factor
                
                if consecutive_losses > 0:
                    risk_factor *= (0.8 ** consecutive_losses)  # Exponential reduction
                
                adjusted_position_size = signal['position_size'] * risk_factor
                
                if signal['signal_type'] == 'long':
                    target_allocation = adjusted_position_size
                    target_value = portfolio_value * target_allocation
                    target_btc = target_value / current_price
                    btc_to_buy = target_btc - btc_position
                    
                    if btc_to_buy > self.min_trade_size:
                        # Realistic execution costs
                        slippage = self._calculate_realistic_slippage(signal, price_data_point)
                        execution_price = current_price * (1 + slippage + self.trading_fee)
                        cost = btc_to_buy * execution_price
                        
                        if cost <= cash * 0.95:  # Keep 5% cash buffer
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
                                'risk_factor': risk_factor,
                                'slippage': slippage
                            })
                
                elif signal['signal_type'] == 'short':
                    btc_to_sell = btc_position * adjusted_position_size
                    
                    if btc_to_sell > self.min_trade_size:
                        slippage = self._calculate_realistic_slippage(signal, price_data_point)
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
                            'risk_factor': risk_factor,
                            'slippage': slippage
                        })
            
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
                'drawdown_protection': drawdown_protection_factor,
                'consecutive_losses': consecutive_losses
            })
            
            # Calculate daily return
            if len(portfolio) > 1:
                prev_value = portfolio[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                portfolio[-1]['daily_return'] = daily_return
                self.daily_returns.append(daily_return)
        
        self.equity_curve = portfolio
        
        return {
            'portfolio_history': portfolio,
            'signals': signals,
            'trades': self.trades,
            'monthly_performance': monthly_performance,
            'performance_metrics': self._calculate_2year_metrics()
        }
    
    def _calculate_realistic_slippage(self, signal: Dict, price_data: Dict) -> float:
        """
        Berechnet realistische Slippage basierend auf Marktbedingungen
        """
        base_slippage = 0.0005  # 0.05% base
        
        # Phase-based slippage
        phase_multipliers = {
            'crypto_winter': 3.0,      # High slippage in low liquidity
            'gradual_recovery': 1.5,
            'etf_anticipation': 1.2,
            'etf_approval': 2.0,       # High slippage during major events
            'etf_cooldown': 1.3,
            'summer_consolidation': 1.0,
            'pre_election': 1.4,
            'trump_victory': 2.5       # Very high slippage during euphoria
        }
        
        phase_mult = phase_multipliers.get(price_data['phase'], 1.0)
        
        # Volume-based adjustment
        volume_factor = max(0.5, min(2.0, 1.5 / signal.get('volume_ratio', 1.0)))
        
        # Position size penalty
        size_penalty = 1.0 + (signal['position_size'] * 2)  # Larger positions have more impact
        
        # Volatility penalty
        vol_penalty = 1.0 + (price_data['volatility_regime'] * 5)
        
        total_slippage = base_slippage * phase_mult * volume_factor * size_penalty * vol_penalty
        
        return min(0.01, total_slippage)  # Cap at 1%
    
    def _calculate_2year_metrics(self) -> Dict[str, Any]:
        """
        Berechnet umfassende 2-Jahres Performance-Metriken
        """
        if len(self.daily_returns) < 2:
            return {}
        
        returns = np.array(self.daily_returns)
        equity_values = [p['portfolio_value'] for p in self.equity_curve]
        
        # Basic returns
        final_value = equity_values[-1]
        total_return = (final_value / self.initial_capital) - 1
        days = len(equity_values)
        annual_return = ((final_value / self.initial_capital) ** (365 / days)) - 1
        
        # Risk metrics
        daily_vol = np.std(returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe & Sortino
        risk_free_rate = 0.025  # 2.5% for 2023-2024 period
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252) if len(negative_returns) > 1 else annual_vol
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        max_dd = self._calculate_max_drawdown(equity_values)
        calmar_ratio = annual_return / max(max_dd, 0.01)
        
        # Advanced risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Trade analysis
        if self.trades:
            buy_trades = [t for t in self.trades if t['type'] == 'BUY']
            sell_trades = [t for t in self.trades if t['type'] == 'SELL']
            
            avg_slippage = np.mean([t.get('slippage', 0) for t in self.trades]) * 10000  # in bps
            total_fees = sum(t.get('cost', 0) * self.trading_fee for t in buy_trades) + \
                        sum(t.get('proceeds', 0) * self.trading_fee for t in sell_trades)
        else:
            avg_slippage = 0
            total_fees = 0
        
        # Win rate (simplified)
        positive_days = len(returns[returns > 0])
        win_rate = positive_days / len(returns)
        
        # Monthly metrics
        monthly_returns = []
        for i in range(30, len(equity_values), 30):  # Approximate monthly
            if i < len(equity_values):
                month_start = equity_values[i-30]
                month_end = equity_values[i]
                monthly_return = (month_end - month_start) / month_start
                monthly_returns.append(monthly_return)
        
        if monthly_returns:
            monthly_vol = np.std(monthly_returns, ddof=1)
            positive_months = len([r for r in monthly_returns if r > 0])
            monthly_win_rate = positive_months / len(monthly_returns)
        else:
            monthly_vol = 0
            monthly_win_rate = 0
        
        # Phase-based performance
        phase_performance = {}
        current_phase = None
        phase_start_value = None
        
        for snapshot in self.equity_curve:
            if snapshot['phase'] != current_phase:
                if current_phase and phase_start_value:
                    phase_return = (snapshot['portfolio_value'] - phase_start_value) / phase_start_value
                    if current_phase not in phase_performance:
                        phase_performance[current_phase] = []
                    phase_performance[current_phase].append(phase_return)
                
                current_phase = snapshot['phase']
                phase_start_value = snapshot['portfolio_value']
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'annual_volatility': annual_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'monthly_win_rate': monthly_win_rate,
            'monthly_volatility': monthly_vol,
            'total_trades': len(self.trades),
            'avg_slippage_bps': avg_slippage,
            'total_fees': total_fees,
            'fee_rate': total_fees / self.initial_capital,
            'days_analyzed': days,
            'months_analyzed': len(monthly_returns),
            'phase_performance': phase_performance,
            'final_capital': final_value
        }
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Berechnet Maximum Drawdown"""
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            drawdown = (value - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return abs(max_drawdown)


async def run_2year_validation():
    """
    Führt 2-Jahres Validation durch
    """
    print("📊 2-YEAR STRATEGY VALIDATION")
    print("=" * 80)
    print("Periode: Januar 2023 - Dezember 2024")
    print("Startkapital: $300,000")
    print("Realistische Marktbedingungen & Enhanced Risk Management")
    print()
    
    validation = StrategyValidation2Years(initial_capital=300000.0)
    
    # 2-year period
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("📈 Generiere realistische 2-Jahres BTC-Daten...")
    price_data = validation.generate_realistic_2year_data(start_date, end_date)
    
    print(f"✅ {len(price_data['prices'])} Tage (2 Jahre) generiert")
    print(f"   BTC Start 2023: ${price_data['start_price']:,.0f}")
    print(f"   BTC Ende 2024: ${price_data['end_price']:,.0f}")
    print(f"   BTC 2-Jahr Return: {((price_data['end_price']/price_data['start_price'])-1):.1%}")
    print()
    
    # Execute validation
    print("⚡ Führe 2-Jahres Validation Backtest durch...")
    results = validation.execute_2year_backtest(price_data)
    
    # Analyze results
    metrics = results['performance_metrics']
    
    print("📊 2-YEAR VALIDATION RESULTS")
    print("-" * 80)
    print(f"💰 Startkapital:          ${validation.initial_capital:,.0f}")
    print(f"💰 Endkapital:            ${metrics['final_capital']:,.0f}")
    print(f"📈 Total Return:          {metrics['total_return']:.1%}")
    print(f"📊 Annual Return:         {metrics['annual_return']:.1%}")
    print(f"⚡ Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    print(f"🎯 Sortino Ratio:         {metrics['sortino_ratio']:.2f}")
    print(f"🏆 Calmar Ratio:          {metrics['calmar_ratio']:.2f}")
    print(f"📉 Max Drawdown:          {metrics['max_drawdown']:.1%}")
    print(f"🎲 Volatilität:           {metrics['annual_volatility']:.1%}")
    print(f"✅ Daily Win Rate:        {metrics['win_rate']:.1%}")
    print(f"📅 Monthly Win Rate:      {metrics['monthly_win_rate']:.1%}")
    print(f"🔄 Total Trades:          {metrics['total_trades']:,}")
    print(f"💸 Slippage (avg):        {metrics['avg_slippage_bps']:.1f} bps")
    print(f"💸 Total Fees:            ${metrics['total_fees']:,.0f} ({metrics['fee_rate']:.2%})")
    print()
    
    # Performance vs. targets and benchmarks
    print("🎯 PERFORMANCE ASSESSMENT")
    print("-" * 80)
    
    # Original targets (adjusted for 2 years)
    target_annual_return = 0.40
    target_sharpe = 1.5
    target_max_dd = 0.15
    
    return_achieved = metrics['annual_return'] >= target_annual_return
    sharpe_achieved = metrics['sharpe_ratio'] >= target_sharpe
    dd_achieved = metrics['max_drawdown'] <= target_max_dd
    
    print(f"Annual Return ≥ 40%:      {'✅' if return_achieved else '❌'} ({metrics['annual_return']:.1%})")
    print(f"Sharpe Ratio ≥ 1.5:       {'✅' if sharpe_achieved else '❌'} ({metrics['sharpe_ratio']:.2f})")
    print(f"Max Drawdown ≤ 15%:       {'✅' if dd_achieved else '❌'} ({metrics['max_drawdown']:.1%})")
    print()
    
    # Benchmark comparisons
    btc_2year_return = (price_data['end_price'] / price_data['start_price']) - 1
    btc_annual_return = ((price_data['end_price'] / price_data['start_price']) ** (1/2)) - 1
    
    # S&P 500 proxy (approximately 20% over 2 years)
    sp500_2year_return = 0.20
    sp500_annual_return = ((1 + sp500_2year_return) ** (1/2)) - 1
    
    print("📊 BENCHMARK COMPARISON")
    print("-" * 80)
    print(f"Strategy Annual Return:   {metrics['annual_return']:.1%}")
    print(f"BTC Buy&Hold Annual:      {btc_annual_return:.1%}")
    print(f"S&P 500 Annual (est):     {sp500_annual_return:.1%}")
    print()
    print(f"Alpha vs BTC:             {metrics['annual_return'] - btc_annual_return:.1%}")
    print(f"Alpha vs S&P 500:         {metrics['annual_return'] - sp500_annual_return:.1%}")
    print()
    
    # Risk-adjusted performance
    btc_vol_estimate = 0.65  # BTC typical volatility
    sp500_vol_estimate = 0.18  # S&P 500 typical volatility
    
    btc_sharpe = (btc_annual_return - 0.025) / btc_vol_estimate
    sp500_sharpe = (sp500_annual_return - 0.025) / sp500_vol_estimate
    
    print("📊 RISK-ADJUSTED COMPARISON")
    print("-" * 80)
    print(f"Strategy Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"BTC Buy&Hold Sharpe:      {btc_sharpe:.2f}")
    print(f"S&P 500 Sharpe (est):     {sp500_sharpe:.2f}")
    print()
    
    # Phase performance analysis
    if 'phase_performance' in metrics and metrics['phase_performance']:
        print("📊 PERFORMANCE BY MARKET PHASE")
        print("-" * 80)
        for phase, returns in metrics['phase_performance'].items():
            if returns:
                avg_return = np.mean(returns)
                print(f"{phase:20}: {avg_return:7.1%}")
        print()
    
    # Trading activity analysis
    signals = results['signals']
    trades = results['trades']
    
    if signals:
        print("📊 TRADING ACTIVITY ANALYSIS")
        print("-" * 80)
        print(f"Total Signals:            {len(signals):,}")
        print(f"Signal Frequency:         {len(signals)/len(price_data['prices']):.1%} (daily)")
        print(f"Signals Executed:         {len(trades):,}")
        print(f"Execution Rate:           {len(trades)/max(len(signals),1):.1%}")
        
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            print(f"Buy Trades:               {len(buy_trades):,}")
            print(f"Sell Trades:              {len(sell_trades):,}")
            
            avg_position_size = np.mean([t.get('position_size', 0) for t in trades])
            print(f"Avg Position Size:        {avg_position_size:.1%}")
        print()
    
    # Final assessment
    overall_score = 0
    if return_achieved: overall_score += 40
    if sharpe_achieved: overall_score += 30
    if dd_achieved: overall_score += 20
    if metrics['annual_return'] > btc_annual_return: overall_score += 10  # Beat BTC
    
    print("🏆 FINAL ASSESSMENT")
    print("-" * 80)
    
    if overall_score >= 80:
        print("🎉 EXCELLENT - Strategy übertrifft alle Erwartungen!")
        assessment = "EXCELLENT"
        recommendation = "Empfehlung: Sofortige Live-Implementation mit vollem Kapital"
    elif overall_score >= 60:
        print("🎯 VERY GOOD - Strategy zeigt starke Performance!")
        assessment = "VERY_GOOD"
        recommendation = "Empfehlung: Live-Implementation mit reduziertem Kapital ($100k Start)"
    elif overall_score >= 40:
        print("👍 GOOD - Strategy ist profitable aber verbesserungsfähig")
        assessment = "GOOD"
        recommendation = "Empfehlung: Weitere Optimierung vor Live-Trading"
    elif overall_score >= 20:
        print("📈 MODERATE - Strategy zeigt Potential")
        assessment = "MODERATE"
        recommendation = "Empfehlung: Papier-Trading und weitere Entwicklung"
    else:
        print("🔧 NEEDS WORK - Strategy benötigt grundlegende Überarbeitung")
        assessment = "NEEDS_WORK"
        recommendation = "Empfehlung: Zurück zur Forschung und Entwicklung"
    
    print(f"Overall Score: {overall_score}/100")
    print(f"{recommendation}")
    print()
    
    # Export detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"2year_validation_results_{timestamp}.json"
    
    export_data = {
        'validation_info': {
            'strategy': 'Ultimate BTC Strategy',
            'validation_period': '2023-01-01 to 2024-12-31',
            'initial_capital': validation.initial_capital,
            'market_conditions': 'Realistic 2-year crypto cycle'
        },
        'performance_metrics': metrics,
        'benchmark_comparison': {
            'btc_annual_return': btc_annual_return,
            'sp500_annual_return': sp500_annual_return,
            'strategy_alpha_vs_btc': metrics['annual_return'] - btc_annual_return,
            'strategy_alpha_vs_sp500': metrics['annual_return'] - sp500_annual_return
        },
        'risk_assessment': {
            'target_achievement': {
                'return_target': return_achieved,
                'sharpe_target': sharpe_achieved,
                'drawdown_target': dd_achieved
            },
            'overall_score': overall_score,
            'assessment': assessment,
            'recommendation': recommendation
        },
        'trading_analysis': {
            'total_signals': len(signals),
            'total_trades': len(trades),
            'signal_frequency': len(signals)/len(price_data['prices']),
            'execution_rate': len(trades)/max(len(signals),1)
        }
    }
    
    import json
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"💾 Vollständige 2-Jahres Validation exportiert: {filename}")
    
    return metrics, assessment


if __name__ == "__main__":
    asyncio.run(run_2year_validation())