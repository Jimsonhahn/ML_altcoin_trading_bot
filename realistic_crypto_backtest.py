#!/usr/bin/env python3
"""
Realistic Crypto Trading Backtest Engine
========================================

Vollständig realistische Backtesting-Implementation mit:
- Echten Marktdaten oder sehr realistischen Simulationen
- Korrekten Trading-Kosten (Gebühren, Slippage, Spread)
- Liquiditätsbeschränkungen
- Proper Risk Management
- Multiple Marktzyklen
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Verschiedene Marktphasen"""
    BEAR = "bear"
    SIDEWAYS = "sideways" 
    BULL = "bull"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class TradeExecution:
    """Realistic trade execution details"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    size_usd: float
    price: float
    executed_price: float
    slippage: float
    spread: float
    commission: float
    total_cost: float
    execution_time_ms: int
    liquidity_impact: float


@dataclass
class Position:
    """Current position tracking"""
    symbol: str
    size: float  # in coins
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    direction: str  # 'long' or 'short'


class RealisticExchangeSimulator:
    """Simuliert realistische Exchange-Bedingungen"""
    
    def __init__(self):
        # Realistische Exchange Parameter (Binance-basiert)
        self.maker_fee = 0.00075    # 0.075% with BNB discount
        self.taker_fee = 0.00075    # 0.075% with BNB discount
        
        # Spread basierend auf Marktbedingungen
        self.spread_normal = 0.0001      # 0.01% in ruhigen Zeiten
        self.spread_volatile = 0.0005    # 0.05% bei hoher Volatilität
        self.spread_extreme = 0.001      # 0.1% bei extremen Bewegungen
        
        # Slippage Parameter
        self.slippage_tiers = {
            1000: 0.0005,    # $1k: 0.05%
            5000: 0.001,     # $5k: 0.1% 
            10000: 0.002,    # $10k: 0.2%
            25000: 0.005,    # $25k: 0.5%
            50000: 0.01,     # $50k: 1.0%
        }
        
        # Liquiditäts-Parameter
        self.max_order_of_volume = 0.02  # Max 2% des 1h Volumens
        self.min_order_size = 10         # $10 minimum
        
        # Ausführungszeiten (Millisekunden)
        self.execution_time_normal = 50
        self.execution_time_high_vol = 200
        
    def calculate_spread(self, price: float, volatility: float, volume: float) -> float:
        """Berechnet realistischen Bid-Ask Spread"""
        
        # Basis-Spread basierend auf Volatilität
        if volatility < 0.01:  # Niedrige Volatilität
            base_spread = self.spread_normal
        elif volatility < 0.03:  # Moderate Volatilität
            base_spread = self.spread_volatile
        else:  # Hohe Volatilität
            base_spread = self.spread_extreme
            
        # Volume Impact (niedriges Volumen = höherer Spread)
        volume_impact = max(1.0, 1000000 / max(volume, 1))  # Inverse Beziehung
        volume_multiplier = min(2.0, volume_impact / 500000)
        
        return base_spread * volume_multiplier
    
    def calculate_slippage(self, order_size_usd: float, volatility: float) -> float:
        """Berechnet Slippage basierend auf Order-Größe"""
        
        # Base slippage from tier
        base_slippage = 0.0001  # 0.01% minimum
        
        for size_threshold, slippage_rate in self.slippage_tiers.items():
            if order_size_usd >= size_threshold:
                base_slippage = slippage_rate
        
        # Volatility multiplier
        volatility_multiplier = 1 + (volatility * 10)  # Higher vol = more slippage
        
        # Large order penalty
        if order_size_usd > 50000:
            large_order_penalty = (order_size_usd - 50000) / 500000  # +0.1% per $50k
        else:
            large_order_penalty = 0
            
        return base_slippage * volatility_multiplier + large_order_penalty
    
    def calculate_liquidity_impact(self, order_size_usd: float, hourly_volume_usd: float) -> float:
        """Berechnet Market Impact basierend auf verfügbarer Liquidität"""
        
        if hourly_volume_usd <= 0:
            return 0.01  # 1% penalty für keine Liquiditätsdaten
            
        order_volume_ratio = order_size_usd / hourly_volume_usd
        
        # Progressive Impact
        if order_volume_ratio <= 0.001:  # <0.1% of volume
            return 0.0
        elif order_volume_ratio <= 0.005:  # 0.1-0.5%
            return order_volume_ratio * 0.1
        elif order_volume_ratio <= 0.02:   # 0.5-2%
            return order_volume_ratio * 0.2
        else:  # >2% - sehr schwer auszuführen
            return order_volume_ratio * 0.5
    
    def can_execute_order(self, order_size_usd: float, available_balance: float, 
                         hourly_volume_usd: float) -> Tuple[bool, str]:
        """Prüft ob Order ausgeführt werden kann"""
        
        # Minimum order size
        if order_size_usd < self.min_order_size:
            return False, f"Below minimum order size (${self.min_order_size})"
        
        # Balance check
        if order_size_usd > available_balance * 0.98:  # 2% reserve
            return False, "Insufficient balance"
        
        # Liquidity check
        if hourly_volume_usd > 0:
            volume_ratio = order_size_usd / hourly_volume_usd
            if volume_ratio > self.max_order_of_volume:
                return False, f"Order too large for available liquidity ({volume_ratio:.1%} of volume)"
        
        return True, "OK"
    
    def execute_trade(self, timestamp: datetime, symbol: str, side: str, 
                     order_size_usd: float, market_price: float, 
                     market_data: Dict[str, float]) -> TradeExecution:
        """Führt Trade mit realistischen Kosten aus"""
        
        volatility = market_data.get('volatility', 0.02)
        volume_usd = market_data.get('volume_usd', 1000000)
        
        # Kosten berechnen
        spread = self.calculate_spread(market_price, volatility, volume_usd)
        slippage = self.calculate_slippage(order_size_usd, volatility)
        liquidity_impact = self.calculate_liquidity_impact(order_size_usd, volume_usd)
        
        # Execution Price
        total_impact = spread + slippage + liquidity_impact
        if side == 'buy':
            executed_price = market_price * (1 + total_impact)
        else:
            executed_price = market_price * (1 - total_impact)
        
        # Commission
        commission = order_size_usd * self.taker_fee
        
        # Total costs
        total_cost = (total_impact * order_size_usd) + commission
        
        # Execution time (höher bei Volatilität)
        exec_time = self.execution_time_high_vol if volatility > 0.03 else self.execution_time_normal
        
        return TradeExecution(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size_usd=order_size_usd,
            price=market_price,
            executed_price=executed_price,
            slippage=slippage,
            spread=spread,
            commission=commission,
            total_cost=total_cost,
            execution_time_ms=exec_time,
            liquidity_impact=liquidity_impact
        )


class RealisticMarketDataGenerator:
    """Generiert sehr realistische Crypto-Marktdaten"""
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        
        # Historische BTC Parameter (2020-2024 Durchschnitt)
        self.base_params = {
            'daily_return_mean': 0.0003,      # ~11% jährlich
            'daily_volatility': 0.04,         # 4% täglich
            'trend_persistence': 0.95,        # Trends halten länger an
            'regime_change_prob': 0.02,       # 2% Chance täglich
            'weekend_effect': 0.85,           # Weniger Volumen am Wochenende
            'news_event_prob': 0.005,         # 0.5% Chance auf News Impact
            'correlation_with_stocks': 0.3    # Korrelation mit traditionellen Märkten
        }
        
    def generate_realistic_data(self, start_date: str = "2022-01-01", 
                              end_date: str = "2024-07-01", 
                              frequency: str = "1H") -> pd.DataFrame:
        """Generiert sehr realistische Marktdaten"""
        
        logger.info(f"Generiere realistische {self.symbol} Daten: {start_date} bis {end_date}")
        
        # Date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Initialize variables
        prices = []
        volumes = []
        current_price = 42000 if "BTC" in self.symbol else 3000  # Historical start
        current_regime = MarketRegime.SIDEWAYS
        regime_days = 0
        trend_momentum = 0
        
        # Market regime parameters
        regime_params = {
            MarketRegime.BULL: {
                'trend': 0.002/24,      # +0.2% daily
                'volatility': 0.035/np.sqrt(24),
                'volume_mult': 1.2,
                'duration_range': (30, 120)  # 30-120 days
            },
            MarketRegime.BEAR: {
                'trend': -0.0015/24,    # -0.15% daily  
                'volatility': 0.045/np.sqrt(24),
                'volume_mult': 1.4,
                'duration_range': (45, 180)
            },
            MarketRegime.SIDEWAYS: {
                'trend': 0,
                'volatility': 0.025/np.sqrt(24),
                'volume_mult': 1.0,
                'duration_range': (60, 200)
            },
            MarketRegime.CRASH: {
                'trend': -0.008/24,     # -0.8% daily
                'volatility': 0.08/np.sqrt(24),
                'volume_mult': 3.0,
                'duration_range': (3, 14)  # Short crashes
            },
            MarketRegime.RECOVERY: {
                'trend': 0.003/24,      # +0.3% daily
                'volatility': 0.04/np.sqrt(24),
                'volume_mult': 1.8,
                'duration_range': (20, 60)
            }
        }
        
        for i, timestamp in enumerate(dates):
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Regime Management
            regime_days += 1/24
            params = regime_params[current_regime]
            
            # Check for regime change
            if (regime_days > params['duration_range'][0] and 
                np.random.random() < self.base_params['regime_change_prob']/24):
                
                # Choose new regime based on current regime
                regime_transitions = {
                    MarketRegime.BULL: [MarketRegime.SIDEWAYS, MarketRegime.BEAR, MarketRegime.CRASH],
                    MarketRegime.BEAR: [MarketRegime.SIDEWAYS, MarketRegime.RECOVERY, MarketRegime.CRASH],
                    MarketRegime.SIDEWAYS: [MarketRegime.BULL, MarketRegime.BEAR],
                    MarketRegime.CRASH: [MarketRegime.RECOVERY, MarketRegime.BEAR],
                    MarketRegime.RECOVERY: [MarketRegime.BULL, MarketRegime.SIDEWAYS]
                }
                
                possible_regimes = regime_transitions[current_regime]
                current_regime = np.random.choice(possible_regimes)
                regime_days = 0
                params = regime_params[current_regime]
                
                logger.debug(f"Regime change to {current_regime.value} at {timestamp}")
            
            # Price Movement Components
            
            # 1. Trend component with momentum
            trend_component = params['trend']
            if current_regime != MarketRegime.CRASH:
                trend_momentum = trend_momentum * 0.99 + trend_component * 0.01  # Momentum buildup
                trend_component += trend_momentum * 0.5
            
            # 2. Random component with regime volatility
            base_volatility = params['volatility']
            random_component = np.random.normal(0, base_volatility)
            
            # 3. Intraday patterns
            intraday_patterns = {
                (0, 8): 0.9,     # Asian session - Lower activity
                (8, 16): 1.3,    # London session - Higher activity  
                (16, 24): 1.1    # US session - Moderate activity
            }
            
            intraday_mult = 1.0
            for (start_h, end_h), mult in intraday_patterns.items():
                if start_h <= hour < end_h:
                    intraday_mult = mult
                    break
            
            # 4. Weekend effect
            weekend_mult = self.base_params['weekend_effect'] if day_of_week >= 5 else 1.0
            
            # 5. News events (rare but significant)
            news_impact = 0
            if np.random.random() < self.base_params['news_event_prob']:
                # Random news can be positive or negative
                news_impact = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.08)
                logger.debug(f"News event: {news_impact:.2%} at {timestamp}")
            
            # 6. Mean reversion (especially in sideways markets)
            if current_regime == MarketRegime.SIDEWAYS:
                if "BTC" in self.symbol:
                    mean_price = 45000  # BTC mean
                else:
                    mean_price = 3200   # ETH mean
                    
                mean_reversion = -0.001 * (current_price - mean_price) / mean_price / 24
            else:
                mean_reversion = 0
            
            # Combine all components
            total_return = (trend_component + random_component + news_impact + mean_reversion) * intraday_mult * weekend_mult
            
            # Apply to price
            new_price = current_price * (1 + total_return)
            
            # Price floors (avoid unrealistic crashes)
            min_price = 15000 if "BTC" in self.symbol else 800
            new_price = max(new_price, min_price)
            
            current_price = new_price
            prices.append(current_price)
            
            # Volume Generation
            base_volume = 25000 if "BTC" in self.symbol else 50000  # Different base volumes
            
            # Volume correlates with price movement and regime
            volatility_volume = abs(total_return) * 1000000
            regime_volume = base_volume * params['volume_mult']
            intraday_volume = regime_volume * intraday_mult * weekend_mult
            
            # Random volume component
            random_volume = np.random.lognormal(0, 0.4) * base_volume * 0.3
            
            total_volume = intraday_volume + volatility_volume + random_volume
            volumes.append(total_volume)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.008)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.008)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Ensure OHLC logic is correct
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        # Add volume in USD
        df['volume_usd'] = df['volume'] * df['close']
        
        # Add calculated features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std() * np.sqrt(24)  # 24h rolling vol
        df['volume_sma20'] = df['volume_usd'].rolling(20).mean()
        df['volume_ratio'] = df['volume_usd'] / df['volume_sma20']
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Statistics
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        annual_vol = df['returns'].std() * np.sqrt(365*24)
        
        logger.info(f"✅ {len(df)} Datenpunkte generiert")
        logger.info(f"   Total Return: {total_return:.1%}")
        logger.info(f"   Annualized Volatility: {annual_vol:.1%}")
        logger.info(f"   Price Range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        logger.info(f"   Avg Daily Volume: ${df['volume_usd'].mean():,.0f}")
        
        return df


class RealisticTradingStrategy:
    """Realistische Trading-Strategie mit konservativen Parametern"""
    
    def __init__(self):
        # Sehr konservative Parameter für realistische Ergebnisse
        self.max_position_size = 0.08        # Max 8% per Trade
        self.min_signal_strength = 0.65      # Hohe Schwelle für Qualität
        self.stop_loss_pct = 0.025          # 2.5% Stop Loss
        self.take_profit_pct = 0.05         # 5% Take Profit (2:1 R/R)
        self.max_daily_trades = 2           # Max 2 Trades pro Tag
        self.cooldown_hours = 4             # 4h zwischen Trades
        
        # Risk Management
        self.max_daily_risk = 0.02          # Max 2% täglich riskieren
        self.max_drawdown_stop = 0.15       # Stop bei 15% Drawdown
        self.position_correlation_limit = 0.5 # Max Korrelation zwischen Positionen
        
        # State
        self.last_trade_time = None
        self.daily_trades = 0
        self.daily_risk_used = 0
        self.current_date = None
        
    def calculate_indicators(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Berechnet Trading-Indikatoren"""
        
        if len(data) < lookback:
            return {}
            
        recent_data = data.tail(lookback)
        current_price = recent_data['close'].iloc[-1]
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_10'] = recent_data['close'].rolling(10).mean().iloc[-1]
        indicators['sma_20'] = recent_data['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = recent_data['close'].rolling(50).mean().iloc[-1]
        
        # EMAs
        indicators['ema_12'] = recent_data['close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = recent_data['close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = recent_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = recent_data['close'].ewm(span=9).mean().iloc[-1]
        
        # Volatility
        indicators['volatility'] = recent_data['returns'].std() * np.sqrt(24)
        
        # Volume
        indicators['volume_ratio'] = recent_data['volume_ratio'].iloc[-1] if 'volume_ratio' in recent_data else 1.0
        
        # Momentum
        indicators['momentum_5'] = (current_price / recent_data['close'].iloc[-6]) - 1 if len(recent_data) >= 6 else 0
        indicators['momentum_10'] = (current_price / recent_data['close'].iloc[-11]) - 1 if len(recent_data) >= 11 else 0
        
        # Bollinger Bands
        sma20 = recent_data['close'].rolling(20).mean()
        std20 = recent_data['close'].rolling(20).std()
        indicators['bb_upper'] = (sma20 + 2*std20).iloc[-1]
        indicators['bb_lower'] = (sma20 - 2*std20).iloc[-1]
        indicators['bb_position'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators
    
    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generiert Trading-Signal mit mehreren Filtern"""
        
        # Daily reset
        current_date = timestamp.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_trades = 0
            self.daily_risk_used = 0
        
        # Check constraints
        if self.daily_trades >= self.max_daily_trades:
            return {'direction': 'hold', 'reason': 'daily_limit_reached'}
        
        if self.last_trade_time and (timestamp - self.last_trade_time).seconds / 3600 < self.cooldown_hours:
            return {'direction': 'hold', 'reason': 'cooldown_period'}
        
        if self.daily_risk_used >= self.max_daily_risk:
            return {'direction': 'hold', 'reason': 'daily_risk_limit'}
        
        # Get indicators
        indicators = self.calculate_indicators(data)
        if not indicators:
            return {'direction': 'hold', 'reason': 'insufficient_data'}
        
        current_price = data['close'].iloc[-1]
        
        # Signal Components
        signals = []
        
        # 1. Trend Filter (must be aligned)
        trend_bullish = (current_price > indicators['sma_20'] > indicators['sma_50'] and 
                        indicators['ema_12'] > indicators['ema_26'])
        trend_bearish = (current_price < indicators['sma_20'] < indicators['sma_50'] and 
                        indicators['ema_12'] < indicators['ema_26'])
        
        if not (trend_bullish or trend_bearish):
            return {'direction': 'hold', 'reason': 'no_clear_trend'}
        
        # 2. Momentum Filter
        momentum_positive = indicators['momentum_5'] > 0.01 and indicators['momentum_10'] > 0.005
        momentum_negative = indicators['momentum_5'] < -0.01 and indicators['momentum_10'] < -0.005
        
        # 3. RSI Filter (avoid extremes but not too strict)
        rsi_ok_for_long = 25 < indicators['rsi'] < 65
        rsi_ok_for_short = 35 < indicators['rsi'] < 75
        
        # 4. Volume Filter
        volume_confirmation = indicators['volume_ratio'] > 1.2  # At least 20% above average
        
        # 5. Volatility Filter (avoid too high volatility)
        volatility_ok = 0.01 < indicators['volatility'] < 0.06  # 1-6% daily vol
        
        # 6. Bollinger Band position
        bb_pos = indicators['bb_position']
        
        # Long Signal
        if (trend_bullish and momentum_positive and rsi_ok_for_long and 
            volume_confirmation and volatility_ok and 0.2 < bb_pos < 0.8):
            
            signal_strength = min(0.8, (
                0.3 * (1 if trend_bullish else 0) +
                0.25 * min(1, indicators['momentum_10'] * 20) +
                0.2 * min(1, (indicators['volume_ratio'] - 1)) +
                0.15 * (1 - abs(indicators['rsi'] - 50) / 50) +
                0.1 * (1 - indicators['volatility'] / 0.06)
            ))
            
            if signal_strength >= self.min_signal_strength:
                return {
                    'direction': 'buy',
                    'strength': signal_strength,
                    'confidence': signal_strength,
                    'indicators': indicators,
                    'reasons': ['trend_bullish', 'momentum_positive', 'rsi_ok', 'volume_confirmed']
                }
        
        # Short Signal  
        elif (trend_bearish and momentum_negative and rsi_ok_for_short and 
              volume_confirmation and volatility_ok and 0.2 < bb_pos < 0.8):
            
            signal_strength = min(0.8, (
                0.3 * (1 if trend_bearish else 0) +
                0.25 * min(1, abs(indicators['momentum_10']) * 20) +
                0.2 * min(1, (indicators['volume_ratio'] - 1)) +
                0.15 * (1 - abs(indicators['rsi'] - 50) / 50) +
                0.1 * (1 - indicators['volatility'] / 0.06)
            ))
            
            if signal_strength >= self.min_signal_strength:
                return {
                    'direction': 'sell',
                    'strength': signal_strength,
                    'confidence': signal_strength,
                    'indicators': indicators,
                    'reasons': ['trend_bearish', 'momentum_negative', 'rsi_ok', 'volume_confirmed']
                }
        
        return {'direction': 'hold', 'reason': 'no_quality_signal'}
    
    def calculate_position_size(self, signal_strength: float, current_equity: float, 
                              volatility: float) -> float:
        """Berechnet konservative Position Size"""
        
        # Base size from signal strength
        base_size = signal_strength * 0.1  # Max 8% even with perfect signal
        
        # Volatility adjustment (reduce size in volatile markets)
        vol_adjustment = max(0.3, 1 - (volatility - 0.02) * 10)  # Reduce if vol > 2%
        
        # Risk-based adjustment
        risk_per_trade = self.stop_loss_pct  # Risk is stop loss %
        max_position_for_risk = (self.max_daily_risk - self.daily_risk_used) / risk_per_trade
        
        # Final size
        final_size = min(base_size * vol_adjustment, max_position_for_risk, self.max_position_size)
        
        # Minimum size check
        if final_size < 0.02:  # Less than 2%
            return 0.0
        
        return final_size
    
    def should_exit(self, position: Position, current_price: float, 
                   timestamp: datetime, indicators: Dict[str, float]) -> Tuple[bool, str]:
        """Realistische Exit-Logik"""
        
        if position.direction == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"
        
        # Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit"
        
        # Time-based exit (max 48 hours)
        hours_held = (timestamp - position.entry_time).seconds / 3600
        if hours_held > 48:
            return True, "time_exit"
        
        # Trailing Stop (activates after 40% of target)
        if pnl_pct >= self.take_profit_pct * 0.4:
            trailing_stop = -self.stop_loss_pct * 0.7  # 70% of original stop
            if pnl_pct <= trailing_stop:
                return True, "trailing_stop"
        
        # Trend Reversal Exit
        if position.direction == 'long' and indicators:
            # Exit long if trend turns bearish
            if (indicators.get('ema_12', 0) < indicators.get('ema_26', 0) and
                indicators.get('momentum_5', 0) < -0.01):
                return True, "trend_reversal"
        
        elif position.direction == 'short' and indicators:
            # Exit short if trend turns bullish  
            if (indicators.get('ema_12', 0) > indicators.get('ema_26', 0) and
                indicators.get('momentum_5', 0) > 0.01):
                return True, "trend_reversal"
        
        return False, "hold"


class RealisticBacktester:
    """Hauptklasse für realistisches Backtesting"""
    
    def __init__(self, initial_capital: float = 10000, symbol: str = "BTC/USDT"):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.symbol = symbol
        
        # Components
        self.exchange = RealisticExchangeSimulator()
        self.strategy = RealisticTradingStrategy()
        
        # State
        self.positions: List[Position] = []
        self.trades: List[TradeExecution] = []
        self.equity_history: List[Dict] = []
        self.rejected_orders: List[Dict] = []
        
        # Statistics
        self.total_fees_paid = 0
        self.total_slippage_cost = 0
        self.max_equity = initial_capital
        self.max_drawdown = 0
        
    def run_backtest(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Führt kompletten Backtest durch"""
        
        logger.info(f"Starte realistischen Backtest für {self.symbol}")
        logger.info(f"Zeitraum: {market_data.index[0]} bis {market_data.index[-1]}")
        logger.info(f"Datenpunkte: {len(market_data):,}")
        
        signals_generated = 0
        trades_attempted = 0
        
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            
            # Progress update
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / len(market_data) * 100
                current_equity = self.get_current_equity(row['close'])
                logger.info(f"Progress: {progress:.1f}% - Equity: ${current_equity:,.0f}")
            
            # Skip initial warmup period
            if i < 100:
                continue
            
            # Prepare market data for this timestamp
            historical_data = market_data.iloc[:i+1]
            
            # Market data for execution
            market_info = {
                'volatility': row.get('volatility', 0.03),
                'volume_usd': row.get('volume_usd', 1000000)
            }
            
            # Check exits first
            for position in self.positions.copy():
                indicators = self.strategy.calculate_indicators(historical_data)
                should_exit, reason = self.strategy.should_exit(
                    position, row['close'], timestamp, indicators
                )
                
                if should_exit:
                    self._close_position(position, timestamp, row['close'], market_info, reason)
            
            # Generate new signals
            signal = self.strategy.generate_signal(historical_data, timestamp)
            
            if signal['direction'] != 'hold':
                signals_generated += 1
                
                # Check if we can open new position
                if len(self.positions) == 0:  # Only one position at a time
                    trades_attempted += 1
                    current_equity = self.get_current_equity(row['close'])
                    position_size = self.strategy.calculate_position_size(
                        signal['strength'], current_equity, market_info['volatility']
                    )
                    
                    if position_size > 0:
                        order_size_usd = current_equity * position_size
                        
                        # Check if order can be executed
                        can_execute, reason = self.exchange.can_execute_order(
                            order_size_usd, self.capital, market_info['volume_usd']
                        )
                        
                        if can_execute:
                            self._open_position(signal, timestamp, row['close'], 
                                              order_size_usd, market_info)
                        else:
                            self.rejected_orders.append({
                                'timestamp': timestamp,
                                'reason': reason,
                                'signal': signal,
                                'attempted_size': order_size_usd
                            })
            
            # Update equity history
            self._update_equity_history(timestamp, row['close'])
        
        # Close any remaining positions
        if self.positions:
            final_price = market_data['close'].iloc[-1]
            final_timestamp = market_data.index[-1]
            final_market_info = {
                'volatility': market_data['volatility'].iloc[-1],
                'volume_usd': market_data['volume_usd'].iloc[-1]
            }
            
            for position in self.positions.copy():
                self._close_position(position, final_timestamp, final_price, 
                                   final_market_info, "backtest_end")
        
        # Calculate final metrics
        metrics = self._calculate_metrics()
        
        logger.info("✅ Backtest abgeschlossen!")
        logger.info(f"Signale generiert: {signals_generated}")
        logger.info(f"Trades versucht: {trades_attempted}")
        logger.info(f"Trades ausgeführt: {len(self.trades)}")
        logger.info(f"Orders abgelehnt: {len(self.rejected_orders)}")
        
        return {
            'metrics': metrics,
            'trades': [self._trade_to_dict(t) for t in self.trades],
            'equity_history': self.equity_history,
            'rejected_orders': self.rejected_orders,
            'market_summary': {
                'start_price': market_data['close'].iloc[0],
                'end_price': market_data['close'].iloc[-1],
                'buy_hold_return': (market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1,
                'max_price': market_data['close'].max(),
                'min_price': market_data['close'].min(),
                'avg_volume': market_data['volume_usd'].mean()
            }
        }
    
    def _open_position(self, signal: Dict, timestamp: datetime, price: float,
                      order_size_usd: float, market_info: Dict):
        """Eröffnet neue Position"""
        
        # Execute trade
        execution = self.exchange.execute_trade(
            timestamp, self.symbol, signal['direction'],
            order_size_usd, price, market_info
        )
        
        # Create position
        position_size_coins = order_size_usd / execution.executed_price
        if signal['direction'] == 'sell':
            position_size_coins = -position_size_coins
        
        position = Position(
            symbol=self.symbol,
            size=position_size_coins,
            entry_price=execution.executed_price,
            entry_time=timestamp,
            unrealized_pnl=0,
            stop_loss=self.strategy.stop_loss_pct,
            take_profit=self.strategy.take_profit_pct,
            direction='long' if signal['direction'] == 'buy' else 'short'
        )
        
        self.positions.append(position)
        
        # Update capital
        self.capital -= execution.total_cost
        self.total_fees_paid += execution.commission
        self.total_slippage_cost += execution.slippage * order_size_usd
        
        # Update strategy state
        self.strategy.last_trade_time = timestamp
        self.strategy.daily_trades += 1
        self.strategy.daily_risk_used += self.strategy.stop_loss_pct * (order_size_usd / self.get_current_equity(price))
        
        logger.debug(f"Opened {position.direction} position: ${order_size_usd:,.0f} at ${execution.executed_price:,.2f}")
    
    def _close_position(self, position: Position, timestamp: datetime, price: float,
                       market_info: Dict, reason: str):
        """Schließt Position"""
        
        position_value_usd = abs(position.size) * price
        
        # Execute trade
        side = 'sell' if position.direction == 'long' else 'buy'
        execution = self.exchange.execute_trade(
            timestamp, self.symbol, side,
            position_value_usd, price, market_info
        )
        
        # Calculate PnL
        if position.direction == 'long':
            gross_pnl = (execution.executed_price - position.entry_price) * abs(position.size)
        else:
            gross_pnl = (position.entry_price - execution.executed_price) * abs(position.size)
        
        net_pnl = gross_pnl - execution.total_cost
        
        # Create trade record
        trade = TradeExecution(
            timestamp=execution.timestamp,
            symbol=execution.symbol,
            side=f"close_{position.direction}",
            size_usd=position_value_usd,
            price=price,
            executed_price=execution.executed_price,
            slippage=execution.slippage,
            spread=execution.spread,
            commission=execution.commission,
            total_cost=execution.total_cost,
            execution_time_ms=execution.execution_time_ms,
            liquidity_impact=execution.liquidity_impact
        )
        
        # Add trade-specific info
        trade.entry_time = position.entry_time
        trade.entry_price = position.entry_price
        trade.net_pnl = net_pnl
        trade.gross_pnl = gross_pnl
        trade.return_pct = net_pnl / (abs(position.size) * position.entry_price)
        trade.duration_hours = (timestamp - position.entry_time).total_seconds() / 3600
        trade.exit_reason = reason
        
        self.trades.append(trade)
        
        # Update capital
        self.capital += position_value_usd - execution.total_cost
        self.total_fees_paid += execution.commission
        self.total_slippage_cost += execution.slippage * position_value_usd
        
        # Remove position
        self.positions.remove(position)
        
        logger.debug(f"Closed {position.direction} position: PnL=${net_pnl:+,.0f} ({trade.return_pct:+.1%}) - {reason}")
    
    def _update_equity_history(self, timestamp: datetime, price: float):
        """Aktualisiert Equity History"""
        
        # Calculate unrealized PnL
        unrealized_pnl = 0
        for position in self.positions:
            if position.direction == 'long':
                unrealized_pnl += (price - position.entry_price) * abs(position.size)
            else:
                unrealized_pnl += (position.entry_price - price) * abs(position.size)
        
        total_equity = self.capital + unrealized_pnl
        
        # Update max equity and drawdown
        if total_equity > self.max_equity:
            self.max_equity = total_equity
        
        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        self.equity_history.append({
            'timestamp': timestamp,
            'price': price,
            'cash': self.capital,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'drawdown': current_drawdown,
            'positions_count': len(self.positions)
        })
    
    def get_current_equity(self, current_price: float) -> float:
        """Berechnet aktuelles Equity"""
        unrealized_pnl = 0
        for position in self.positions:
            if position.direction == 'long':
                unrealized_pnl += (current_price - position.entry_price) * abs(position.size)
            else:
                unrealized_pnl += (position.entry_price - current_price) * abs(position.size)
        
        return self.capital + unrealized_pnl
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Berechnet Performance-Metriken"""
        
        if not self.equity_history:
            return {}
        
        # Basic metrics
        final_equity = self.equity_history[-1]['total_equity']
        total_return = (final_equity / self.initial_capital) - 1
        
        # Time-based metrics
        start_time = self.equity_history[0]['timestamp']
        end_time = self.equity_history[-1]['timestamp']
        days = (end_time - start_time).days
        years = days / 365.25 if days > 0 else 1
        annual_return = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1
        
        # Returns for Sharpe calculation
        equity_values = [point['total_equity'] for point in self.equity_history]
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            
            # Sharpe ratio (3% risk-free rate)
            if len(returns) > 0 and returns.std() > 0:
                excess_returns = returns - (0.03 / 365)  # Daily risk-free rate
                sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(365)
            else:
                sharpe_ratio = 0
                
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(365)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Trade statistics
        if self.trades:
            completed_trades = [t for t in self.trades if hasattr(t, 'net_pnl')]
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t.net_pnl > 0]
                losing_trades = [t for t in completed_trades if t.net_pnl <= 0]
                
                win_rate = len(winning_trades) / len(completed_trades)
                avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
                
                # Profit factor
                total_wins = sum(t.net_pnl for t in winning_trades)
                total_losses = abs(sum(t.net_pnl for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Average trade duration
                avg_duration = np.mean([t.duration_hours for t in completed_trades])
                
                # Recovery factor
                recovery_factor = total_return / self.max_drawdown if self.max_drawdown > 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                avg_duration = 0
                recovery_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
            recovery_factor = 0
        
        # Cost analysis
        total_costs = self.total_fees_paid + self.total_slippage_cost
        cost_impact = total_costs / self.initial_capital
        
        return {
            # Returns
            'total_return': total_return,
            'annual_return': annual_return,
            'final_equity': final_equity,
            
            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'recovery_factor': recovery_factor,
            'volatility': returns.std() * np.sqrt(365) if 'returns' in locals() and len(returns) > 0 else 0,
            
            # Trading metrics
            'total_trades': len([t for t in self.trades if hasattr(t, 'net_pnl')]),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_duration_hours': avg_duration,
            
            # Cost analysis
            'total_fees_paid': self.total_fees_paid,
            'total_slippage_cost': self.total_slippage_cost,
            'total_costs': total_costs,
            'cost_impact_pct': cost_impact,
            
            # Execution statistics
            'rejected_orders': len(self.rejected_orders),
            'execution_rate': len(self.trades) / max(1, len(self.trades) + len(self.rejected_orders)),
            
            # Calmar ratio
            'calmar_ratio': annual_return / self.max_drawdown if self.max_drawdown > 0 else 0
        }
    
    def _trade_to_dict(self, trade: TradeExecution) -> Dict:
        """Konvertiert TradeExecution zu Dictionary"""
        result = {
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side,
            'size_usd': trade.size_usd,
            'price': trade.price,
            'executed_price': trade.executed_price,
            'slippage': trade.slippage,
            'spread': trade.spread,
            'commission': trade.commission,
            'total_cost': trade.total_cost,
            'execution_time_ms': trade.execution_time_ms,
            'liquidity_impact': trade.liquidity_impact
        }
        
        # Add trade-specific fields if available
        if hasattr(trade, 'net_pnl'):
            result.update({
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'net_pnl': trade.net_pnl,
                'gross_pnl': trade.gross_pnl,
                'return_pct': trade.return_pct,
                'duration_hours': trade.duration_hours,
                'exit_reason': trade.exit_reason
            })
        
        return result


def run_comprehensive_test():
    """Führt umfassenden realistischen Test durch"""
    
    print("🔬 REALISTIC CRYPTO BACKTEST ENGINE")
    print("=" * 80)
    print("Vollständig realistische Simulation mit echten Marktbedingungen\n")
    
    # Test Parameters
    test_configs = [
        {
            'name': 'Conservative Test',
            'capital': 10000,
            'symbol': 'BTC/USDT',
            'start_date': '2022-01-01',
            'end_date': '2024-01-01'
        },
        {
            'name': 'Smaller Capital Test', 
            'capital': 5000,
            'symbol': 'BTC/USDT',
            'start_date': '2023-01-01',
            'end_date': '2024-01-01'
        }
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n📋 RUNNING: {config['name']}")
        print("-" * 60)
        
        # Generate market data
        data_generator = RealisticMarketDataGenerator(config['symbol'])
        market_data = data_generator.generate_realistic_data(
            config['start_date'], config['end_date']
        )
        
        # Run backtest
        backtester = RealisticBacktester(config['capital'], config['symbol'])
        results = backtester.run_backtest(market_data)
        
        # Store results
        all_results[config['name']] = {
            'config': config,
            'results': results
        }
        
        # Print summary
        metrics = results['metrics']
        market = results['market_summary']
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Period: {config['start_date']} to {config['end_date']}")
        print(f"   Capital: ${config['capital']:,}")
        print(f"   Market Return: {market['buy_hold_return']:+.1%}")
        print(f"   Strategy Return: {metrics.get('annual_return', 0):+.1%}")
        print(f"   Total Return: {metrics.get('total_return', 0):+.1%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Total Costs: ${metrics.get('total_costs', 0):,.2f} ({metrics.get('cost_impact_pct', 0):.2%})")
        print(f"   Rejected Orders: {metrics.get('rejected_orders', 0)}")
    
    # Comparative Analysis
    print(f"\n📈 COMPARATIVE ANALYSIS:")
    print("=" * 80)
    
    for name, data in all_results.items():
        metrics = data['results']['metrics']
        config = data['config']
        market_return = data['results']['market_summary']['buy_hold_return']
        
        alpha = metrics.get('annual_return', 0) - market_return
        
        print(f"\n{name}:")
        print(f"  Strategy Alpha: {alpha:+.1%} vs Buy&Hold")
        print(f"  Risk-Adjusted Return: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Cost Impact: {metrics.get('cost_impact_pct', 0):.2%} of capital")
        print(f"  Execution Rate: {metrics.get('execution_rate', 0):.1%}")
        
        # Risk Assessment
        if metrics.get('max_drawdown', 0) < 0.1:
            risk_level = "LOW"
        elif metrics.get('max_drawdown', 0) < 0.2:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        print(f"  Risk Level: {risk_level}")
        
        # Performance Rating
        annual_return = metrics.get('annual_return', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        if annual_return > 0.20 and sharpe > 1.5:
            rating = "EXCELLENT"
        elif annual_return > 0.10 and sharpe > 1.0:
            rating = "GOOD"
        elif annual_return > 0.05 and sharpe > 0.5:
            rating = "ACCEPTABLE"
        elif annual_return > 0:
            rating = "POOR"
        else:
            rating = "LOSING"
        
        print(f"  Performance Rating: {rating}")
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"realistic_backtest_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results exported to: {filename}")
    
    # Final Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 40)
    
    best_result = max(all_results.values(), 
                     key=lambda x: x['results']['metrics'].get('sharpe_ratio', 0))
    best_metrics = best_result['results']['metrics']
    
    if best_metrics.get('annual_return', 0) > 0.15 and best_metrics.get('sharpe_ratio', 0) > 1.0:
        print("✅ Strategy shows promise with realistic expectations")
        print("   Consider paper trading for 3+ months before live deployment")
        print("   Start with small capital (< $5,000)")
    elif best_metrics.get('annual_return', 0) > 0.05:
        print("⚠️ Strategy is marginally profitable")
        print("   High risk relative to returns")
        print("   Extensive optimization needed before deployment")
    else:
        print("❌ Strategy is not profitable under realistic conditions")
        print("   Complete redesign recommended")
        print("   Focus on risk management and cost minimization")
    
    print(f"\n🔍 Key Learnings:")
    print("   1. Trading costs have significant impact on profitability")
    print("   2. Market conditions greatly affect strategy performance") 
    print("   3. Position sizing and risk management are critical")
    print("   4. Signal quality must be very high to overcome costs")
    print("   5. Realistic expectations: 10-25% annual returns are good")
    
    return all_results


if __name__ == "__main__":
    run_comprehensive_test()