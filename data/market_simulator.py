# data/market_simulator.py
"""
Advanced Market Data Simulator for Realistic Backtesting
Generates synthetic market data with realistic characteristics including:
- Volatility clustering
- Market regimes (bull/bear/sideways)
- News events and price shocks
- Liquidity patterns
- Multi-asset correlations
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class EventType(Enum):
    """Market event types"""
    NEWS_POSITIVE = "news_positive"
    NEWS_NEGATIVE = "news_negative"
    REGULATORY = "regulatory"
    TECHNICAL_BREAKOUT = "technical_breakout"
    WHALE_MOVEMENT = "whale_movement"
    EXCHANGE_EVENT = "exchange_event"
    MACRO_ECONOMIC = "macro_economic"


@dataclass
class MarketEvent:
    """Market event definition"""
    timestamp: datetime
    event_type: EventType
    impact_magnitude: float  # -1 to 1
    duration_hours: int
    affected_symbols: List[str]
    description: str


@dataclass
class RegimeParameters:
    """Market regime parameters"""
    trend_strength: float      # Daily trend drift
    volatility_base: float     # Base volatility
    volatility_clustering: float  # GARCH-like clustering
    mean_reversion: float      # Mean reversion strength
    momentum: float           # Momentum persistence
    shock_probability: float   # Probability of large moves
    correlation_strength: float # Inter-asset correlation


class MarketSimulator:
    """
    Advanced market data simulator with realistic market dynamics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Simulation parameters
        self.random_seed = self.config.get('random_seed', 42)
        self.enable_regimes = self.config.get('enable_regimes', True)
        self.enable_events = self.config.get('enable_events', True)
        self.enable_correlations = self.config.get('enable_correlations', True)
        self.enable_seasonality = self.config.get('enable_seasonality', True)
        
        # Market regime parameters
        self.regime_params = {
            MarketRegime.BULL: RegimeParameters(
                trend_strength=0.0008,     # 0.08% daily uptrend
                volatility_base=0.02,      # 2% daily volatility
                volatility_clustering=0.8,
                mean_reversion=0.1,
                momentum=0.3,
                shock_probability=0.02,
                correlation_strength=0.6
            ),
            MarketRegime.BEAR: RegimeParameters(
                trend_strength=-0.0005,    # -0.05% daily downtrend
                volatility_base=0.035,     # 3.5% daily volatility
                volatility_clustering=0.9,
                mean_reversion=0.05,
                momentum=0.4,
                shock_probability=0.04,
                correlation_strength=0.8
            ),
            MarketRegime.SIDEWAYS: RegimeParameters(
                trend_strength=0.0001,     # 0.01% daily trend
                volatility_base=0.015,     # 1.5% daily volatility
                volatility_clustering=0.6,
                mean_reversion=0.3,
                momentum=0.1,
                shock_probability=0.01,
                correlation_strength=0.4
            ),
            MarketRegime.VOLATILE: RegimeParameters(
                trend_strength=0.0002,     # 0.02% daily trend
                volatility_base=0.045,     # 4.5% daily volatility
                volatility_clustering=0.95,
                mean_reversion=0.05,
                momentum=0.2,
                shock_probability=0.06,
                correlation_strength=0.7
            ),
            MarketRegime.CRISIS: RegimeParameters(
                trend_strength=-0.002,     # -0.2% daily downtrend
                volatility_base=0.08,      # 8% daily volatility
                volatility_clustering=0.98,
                mean_reversion=0.02,
                momentum=0.5,
                shock_probability=0.1,
                correlation_strength=0.9
            )
        }
        
        # Asset correlation matrix (default crypto correlations)
        self.correlation_matrix = {
            'BTC/USDT': {'ETH/USDT': 0.7, 'SOL/USDT': 0.6, 'ADA/USDT': 0.5, 'XRP/USDT': 0.4},
            'ETH/USDT': {'BTC/USDT': 0.7, 'SOL/USDT': 0.8, 'ADA/USDT': 0.6, 'XRP/USDT': 0.5},
            'SOL/USDT': {'BTC/USDT': 0.6, 'ETH/USDT': 0.8, 'ADA/USDT': 0.7, 'XRP/USDT': 0.4},
            'ADA/USDT': {'BTC/USDT': 0.5, 'ETH/USDT': 0.6, 'SOL/USDT': 0.7, 'XRP/USDT': 0.6},
            'XRP/USDT': {'BTC/USDT': 0.4, 'ETH/USDT': 0.5, 'SOL/USDT': 0.4, 'ADA/USDT': 0.6}
        }
        
        # Initialize random state
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # State variables
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_duration = 0
        self.volatility_state = {}  # Per-symbol volatility state
        self.price_state = {}       # Per-symbol price state
        
        logger.info("MarketSimulator initialized")
    
    def generate_synthetic_data(self, 
                              symbols: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              frequency: str = '1h',
                              initial_prices: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic market data for multiple symbols
        """
        try:
            logger.info(f"Generating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Generate time index
            if frequency == '1h':
                freq_str = 'H'
            elif frequency == '1d':
                freq_str = 'D'
            elif frequency == '15m':
                freq_str = '15min'
            elif frequency == '5m':
                freq_str = '5min'
            else:
                freq_str = 'H'
            
            time_index = pd.date_range(start=start_date, end=end_date, freq=freq_str)
            
            # Initialize prices
            if initial_prices is None:
                initial_prices = {
                    'BTC/USDT': 45000,
                    'ETH/USDT': 3000,
                    'SOL/USDT': 100,
                    'ADA/USDT': 0.5,
                    'XRP/USDT': 0.6
                }
            
            # Initialize state
            for symbol in symbols:
                self.price_state[symbol] = initial_prices.get(symbol, 100.0)
                self.volatility_state[symbol] = self.regime_params[self.current_regime].volatility_base
            
            # Generate market events
            events = []
            if self.enable_events:
                events = self._generate_market_events(start_date, end_date, symbols)
            
            # Generate regime changes
            regime_schedule = []
            if self.enable_regimes:
                regime_schedule = self._generate_regime_schedule(start_date, end_date)
            
            # Generate data for each timestamp
            all_data = {symbol: [] for symbol in symbols}
            
            for i, timestamp in enumerate(time_index):
                # Update current regime
                current_regime = self._get_current_regime(timestamp, regime_schedule)
                
                # Get active events
                active_events = [e for e in events if self._is_event_active(e, timestamp)]
                
                # Generate correlated returns
                returns = self._generate_correlated_returns(symbols, current_regime, active_events, timestamp)
                
                # Update prices and generate OHLCV
                for symbol in symbols:
                    ohlcv = self._generate_ohlcv(symbol, returns[symbol], timestamp, frequency)
                    all_data[symbol].append(ohlcv)
            
            # Convert to DataFrames
            result = {}
            for symbol in symbols:
                df = pd.DataFrame(all_data[symbol])
                df['timestamp'] = time_index
                df.set_index('timestamp', inplace=True)
                
                # Add technical indicators for realism
                df = self._add_technical_indicators(df)
                
                result[symbol] = df
            
            logger.info(f"Generated {len(time_index)} data points for each symbol")
            return result
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def _generate_regime_schedule(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, MarketRegime]]:
        """Generate schedule of market regime changes"""
        try:
            schedule = []
            current_date = start_date
            
            # Regime transition probabilities
            transition_matrix = {
                MarketRegime.BULL: {
                    MarketRegime.BULL: 0.85,
                    MarketRegime.SIDEWAYS: 0.1,
                    MarketRegime.VOLATILE: 0.03,
                    MarketRegime.BEAR: 0.015,
                    MarketRegime.CRISIS: 0.005
                },
                MarketRegime.BEAR: {
                    MarketRegime.BEAR: 0.8,
                    MarketRegime.SIDEWAYS: 0.12,
                    MarketRegime.VOLATILE: 0.05,
                    MarketRegime.BULL: 0.02,
                    MarketRegime.CRISIS: 0.01
                },
                MarketRegime.SIDEWAYS: {
                    MarketRegime.SIDEWAYS: 0.7,
                    MarketRegime.BULL: 0.15,
                    MarketRegime.BEAR: 0.1,
                    MarketRegime.VOLATILE: 0.04,
                    MarketRegime.CRISIS: 0.01
                },
                MarketRegime.VOLATILE: {
                    MarketRegime.VOLATILE: 0.6,
                    MarketRegime.SIDEWAYS: 0.2,
                    MarketRegime.BEAR: 0.1,
                    MarketRegime.BULL: 0.08,
                    MarketRegime.CRISIS: 0.02
                },
                MarketRegime.CRISIS: {
                    MarketRegime.CRISIS: 0.7,
                    MarketRegime.BEAR: 0.2,
                    MarketRegime.VOLATILE: 0.08,
                    MarketRegime.SIDEWAYS: 0.02,
                    MarketRegime.BULL: 0.0
                }
            }
            
            current_regime = self.current_regime
            schedule.append((current_date, current_regime))
            
            # Generate regime changes
            min_regime_duration = timedelta(days=5)  # Minimum 5 days per regime
            max_regime_duration = timedelta(days=60)  # Maximum 60 days per regime
            
            while current_date < end_date:
                # Determine regime duration
                if current_regime == MarketRegime.CRISIS:
                    duration = timedelta(days=random.randint(3, 21))  # Crisis shorter
                else:
                    duration = timedelta(days=random.randint(7, 45))
                
                next_date = current_date + duration
                
                if next_date >= end_date:
                    break
                
                # Choose next regime based on transition probabilities
                transitions = transition_matrix[current_regime]
                regimes = list(transitions.keys())
                probabilities = list(transitions.values())
                
                next_regime = np.random.choice(regimes, p=probabilities)
                schedule.append((next_date, next_regime))
                
                current_date = next_date
                current_regime = next_regime
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error generating regime schedule: {e}")
            return [(start_date, self.current_regime)]
    
    def _generate_market_events(self, start_date: datetime, end_date: datetime, symbols: List[str]) -> List[MarketEvent]:
        """Generate random market events"""
        try:
            events = []
            current_date = start_date
            
            # Event frequency (average events per month)
            events_per_month = {
                EventType.NEWS_POSITIVE: 3,
                EventType.NEWS_NEGATIVE: 3,
                EventType.REGULATORY: 1,
                EventType.TECHNICAL_BREAKOUT: 4,
                EventType.WHALE_MOVEMENT: 6,
                EventType.EXCHANGE_EVENT: 1,
                EventType.MACRO_ECONOMIC: 2
            }
            
            total_days = (end_date - start_date).days
            total_months = total_days / 30.44
            
            for event_type, monthly_freq in events_per_month.items():
                expected_events = int(total_months * monthly_freq)
                
                for _ in range(expected_events):
                    # Random timestamp
                    random_days = random.randint(0, total_days)
                    event_timestamp = start_date + timedelta(days=random_days)
                    
                    # Event parameters
                    if event_type in [EventType.NEWS_POSITIVE, EventType.TECHNICAL_BREAKOUT]:
                        impact = random.uniform(0.1, 0.8)
                    elif event_type in [EventType.NEWS_NEGATIVE, EventType.REGULATORY]:
                        impact = random.uniform(-0.8, -0.1)
                    elif event_type == EventType.WHALE_MOVEMENT:
                        impact = random.uniform(-0.3, 0.3)
                    elif event_type == EventType.EXCHANGE_EVENT:
                        impact = random.uniform(-0.5, 0.2)
                    else:  # MACRO_ECONOMIC
                        impact = random.uniform(-0.4, 0.4)
                    
                    duration = random.randint(1, 48)  # 1-48 hours
                    
                    # Affected symbols (some events affect all, some specific)
                    if event_type in [EventType.REGULATORY, EventType.MACRO_ECONOMIC]:
                        affected = symbols  # All symbols
                    else:
                        # Random subset
                        num_affected = random.randint(1, len(symbols))
                        affected = random.sample(symbols, num_affected)
                    
                    event = MarketEvent(
                        timestamp=event_timestamp,
                        event_type=event_type,
                        impact_magnitude=impact,
                        duration_hours=duration,
                        affected_symbols=affected,
                        description=f"{event_type.value.replace('_', ' ').title()} event"
                    )
                    
                    events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Generated {len(events)} market events")
            return events
            
        except Exception as e:
            logger.error(f"Error generating market events: {e}")
            return []
    
    def _get_current_regime(self, timestamp: datetime, regime_schedule: List[Tuple[datetime, MarketRegime]]) -> MarketRegime:
        """Get current market regime for given timestamp"""
        current_regime = self.current_regime
        
        for regime_time, regime in regime_schedule:
            if timestamp >= regime_time:
                current_regime = regime
            else:
                break
        
        return current_regime
    
    def _is_event_active(self, event: MarketEvent, timestamp: datetime) -> bool:
        """Check if event is active at given timestamp"""
        event_end = event.timestamp + timedelta(hours=event.duration_hours)
        return event.timestamp <= timestamp <= event_end
    
    def _generate_correlated_returns(self, symbols: List[str], regime: MarketRegime, 
                                   active_events: List[MarketEvent], timestamp: datetime) -> Dict[str, float]:
        """Generate correlated returns for all symbols"""
        try:
            # Get regime parameters
            params = self.regime_params[regime]
            
            # Base returns (uncorrelated)
            base_returns = {}
            
            for symbol in symbols:
                # Update volatility with clustering
                prev_vol = self.volatility_state.get(symbol, params.volatility_base)
                
                # GARCH-like volatility clustering
                vol_shock = np.random.normal(0, 0.01)
                new_vol = (params.volatility_clustering * prev_vol + 
                          (1 - params.volatility_clustering) * params.volatility_base + vol_shock)
                new_vol = max(0.005, min(0.15, new_vol))  # Clamp volatility
                
                self.volatility_state[symbol] = new_vol
                
                # Generate return with trend, mean reversion, and momentum
                prev_price = self.price_state[symbol]
                
                # Trend component
                trend_return = params.trend_strength
                
                # Mean reversion (toward long-term average)
                # Simplified: revert toward initial price
                initial_price = prev_price  # Could be improved
                mean_reversion_return = -params.mean_reversion * np.log(prev_price / initial_price)
                
                # Random shock
                random_return = np.random.normal(0, new_vol)
                
                # Large shock probability
                if np.random.random() < params.shock_probability:
                    shock_magnitude = np.random.normal(0, new_vol * 3)
                    random_return += shock_magnitude
                
                # Combine components
                base_return = trend_return + mean_reversion_return + random_return
                base_returns[symbol] = base_return
            
            # Apply correlations
            if self.enable_correlations and len(symbols) > 1:
                correlated_returns = self._apply_correlations(base_returns, params.correlation_strength)
            else:
                correlated_returns = base_returns.copy()
            
            # Apply event impacts
            for event in active_events:
                event_impact = self._calculate_event_impact(event, timestamp)
                
                for symbol in symbols:
                    if symbol in event.affected_symbols:
                        correlated_returns[symbol] += event_impact
            
            # Apply seasonality
            if self.enable_seasonality:
                for symbol in symbols:
                    seasonal_factor = self._calculate_seasonal_factor(timestamp)
                    correlated_returns[symbol] *= seasonal_factor
            
            return correlated_returns
            
        except Exception as e:
            logger.error(f"Error generating correlated returns: {e}")
            return {symbol: 0.0 for symbol in symbols}
    
    def _apply_correlations(self, base_returns: Dict[str, float], correlation_strength: float) -> Dict[str, float]:
        """Apply correlations between assets"""
        try:
            symbols = list(base_returns.keys())
            
            if len(symbols) <= 1:
                return base_returns
            
            # Create correlation matrix
            n = len(symbols)
            corr_matrix = np.eye(n)
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        base_corr = self.correlation_matrix.get(symbol1, {}).get(symbol2, 0.3)
                        corr_matrix[i, j] = base_corr * correlation_strength
            
            # Ensure positive definiteness
            try:
                # Cholesky decomposition
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                # If not positive definite, use nearest positive definite matrix
                eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
                eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
                corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                L = np.linalg.cholesky(corr_matrix)
            
            # Apply correlation
            uncorrelated = np.array([base_returns[symbol] for symbol in symbols])
            correlated = L @ uncorrelated
            
            return {symbol: correlated[i] for i, symbol in enumerate(symbols)}
            
        except Exception as e:
            logger.error(f"Error applying correlations: {e}")
            return base_returns
    
    def _calculate_event_impact(self, event: MarketEvent, timestamp: datetime) -> float:
        """Calculate event impact at given timestamp"""
        try:
            # Time since event start
            time_since_start = (timestamp - event.timestamp).total_seconds() / 3600
            
            # Impact decay function (exponential decay)
            decay_rate = 0.2  # Impact halves every 5 hours
            impact_multiplier = np.exp(-decay_rate * time_since_start)
            
            return event.impact_magnitude * impact_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating event impact: {e}")
            return 0.0
    
    def _calculate_seasonal_factor(self, timestamp: datetime) -> float:
        """Calculate seasonal adjustment factor"""
        try:
            # Hour of day effect (lower volume at night UTC)
            hour = timestamp.hour
            if 6 <= hour <= 18:  # Day hours
                hour_factor = 1.0
            else:  # Night hours
                hour_factor = 0.8
            
            # Day of week effect (lower volume on weekends)
            weekday = timestamp.weekday()
            if weekday < 5:  # Monday-Friday
                day_factor = 1.0
            else:  # Weekend
                day_factor = 0.7
            
            # Month effect (January effect, etc.)
            month = timestamp.month
            if month in [1, 11, 12]:  # Year-end effects
                month_factor = 1.1
            else:
                month_factor = 1.0
            
            return hour_factor * day_factor * month_factor
            
        except Exception as e:
            logger.error(f"Error calculating seasonal factor: {e}")
            return 1.0
    
    def _generate_ohlcv(self, symbol: str, return_rate: float, timestamp: datetime, frequency: str) -> Dict[str, Any]:
        """Generate OHLCV data from return rate"""
        try:
            # Update price
            prev_price = self.price_state[symbol]
            new_price = prev_price * (1 + return_rate)
            self.price_state[symbol] = new_price
            
            # Generate intraperiod movements for OHLC
            volatility = self.volatility_state[symbol]
            
            # Number of micro-movements within the period
            if frequency == '1h':
                micro_periods = 12  # 5-minute movements
            elif frequency == '1d':
                micro_periods = 24  # Hourly movements
            else:
                micro_periods = 6
            
            # Generate micro-movements
            micro_returns = np.random.normal(0, volatility / np.sqrt(micro_periods), micro_periods)
            micro_returns[-1] = return_rate - np.sum(micro_returns[:-1])  # Ensure total return matches
            
            # Calculate OHLC
            prices = [prev_price]
            for micro_return in micro_returns:
                new_micro_price = prices[-1] * (1 + micro_return)
                prices.append(new_micro_price)
            
            open_price = prev_price
            close_price = new_price
            high_price = max(prices)
            low_price = min(prices)
            
            # Generate volume (correlated with volatility and price movement)
            base_volume = 1000000  # Base volume
            volatility_factor = 1 + volatility * 10  # Higher volatility = higher volume
            movement_factor = 1 + abs(return_rate) * 5  # Larger moves = higher volume
            random_factor = np.random.lognormal(0, 0.3)  # Random variation
            
            volume = base_volume * volatility_factor * movement_factor * random_factor
            
            return {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
        except Exception as e:
            logger.error(f"Error generating OHLCV for {symbol}: {e}")
            return {
                'open': 100.0,
                'high': 100.0,
                'low': 100.0,
                'close': 100.0,
                'volume': 1000000.0
            }
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR
            df['atr'] = self._calculate_atr(df, 14)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(50, index=prices.index)  # Neutral RSI
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            return atr
        except:
            return pd.Series(0.02, index=df.index)  # Default ATR
    
    def add_market_stress_scenarios(self, data: Dict[str, pd.DataFrame], 
                                  scenario_type: str = "flash_crash") -> Dict[str, pd.DataFrame]:
        """Add specific market stress scenarios to existing data"""
        try:
            logger.info(f"Adding {scenario_type} scenario to data")
            
            stressed_data = {}
            
            for symbol, df in data.items():
                stressed_df = df.copy()
                
                if scenario_type == "flash_crash":
                    # Insert flash crash scenario
                    crash_start = len(df) // 3  # 1/3 through the data
                    crash_duration = 24  # 24 periods
                    
                    # Severe price drop
                    for i in range(crash_start, min(crash_start + crash_duration, len(df))):
                        if i < len(stressed_df):
                            crash_magnitude = 0.3 * np.exp(-(i - crash_start) / 8)  # Exponential recovery
                            stressed_df.iloc[i, stressed_df.columns.get_loc('close')] *= (1 - crash_magnitude)
                            stressed_df.iloc[i, stressed_df.columns.get_loc('low')] *= (1 - crash_magnitude * 1.2)
                            stressed_df.iloc[i, stressed_df.columns.get_loc('volume')] *= (1 + crash_magnitude * 5)
                
                elif scenario_type == "regulatory_shock":
                    # Regulatory announcement effect
                    shock_start = len(df) // 2
                    shock_duration = 72  # 3 days
                    
                    for i in range(shock_start, min(shock_start + shock_duration, len(df))):
                        if i < len(stressed_df):
                            shock_magnitude = 0.15 * (1 - (i - shock_start) / shock_duration)
                            stressed_df.iloc[i, stressed_df.columns.get_loc('close')] *= (1 - shock_magnitude)
                            stressed_df.iloc[i, stressed_df.columns.get_loc('volume')] *= (1 + shock_magnitude * 3)
                
                elif scenario_type == "extreme_volatility":
                    # Extreme volatility period
                    vol_start = len(df) // 4
                    vol_duration = 168  # 1 week
                    
                    for i in range(vol_start, min(vol_start + vol_duration, len(df))):
                        if i < len(stressed_df):
                            vol_multiplier = 3.0
                            price_change = np.random.normal(0, 0.05) * vol_multiplier
                            stressed_df.iloc[i, stressed_df.columns.get_loc('close')] *= (1 + price_change)
                            stressed_df.iloc[i, stressed_df.columns.get_loc('volume')] *= (1 + abs(price_change) * 10)
                
                # Recalculate OHLC consistency
                stressed_df = self._ensure_ohlc_consistency(stressed_df)
                
                # Recalculate technical indicators
                stressed_df = self._add_technical_indicators(stressed_df)
                
                stressed_data[symbol] = stressed_df
            
            return stressed_data
            
        except Exception as e:
            logger.error(f"Error adding stress scenario: {e}")
            return data
    
    def _ensure_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC data consistency"""
        try:
            for i in range(len(df)):
                open_price = df.iloc[i]['open']
                close_price = df.iloc[i]['close']
                high_price = df.iloc[i]['high']
                low_price = df.iloc[i]['low']
                
                # Ensure high is highest and low is lowest
                actual_high = max(open_price, close_price, high_price)
                actual_low = min(open_price, close_price, low_price)
                
                df.iloc[i, df.columns.get_loc('high')] = actual_high
                df.iloc[i, df.columns.get_loc('low')] = actual_low
            
            return df
            
        except Exception as e:
            logger.error(f"Error ensuring OHLC consistency: {e}")
            return df
    
    def export_simulation_config(self, filepath: str = None) -> str:
        """Export simulation configuration"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"market_simulation_config_{timestamp}.json"
            
            config = {
                'random_seed': self.random_seed,
                'enable_regimes': self.enable_regimes,
                'enable_events': self.enable_events,
                'enable_correlations': self.enable_correlations,
                'enable_seasonality': self.enable_seasonality,
                'regime_parameters': {
                    regime.value: {
                        'trend_strength': params.trend_strength,
                        'volatility_base': params.volatility_base,
                        'volatility_clustering': params.volatility_clustering,
                        'mean_reversion': params.mean_reversion,
                        'momentum': params.momentum,
                        'shock_probability': params.shock_probability,
                        'correlation_strength': params.correlation_strength
                    } for regime, params in self.regime_params.items()
                },
                'correlation_matrix': self.correlation_matrix,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Simulation config exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting simulation config: {e}")
            return ""


# Factory function
def create_market_simulator(config: Dict[str, Any] = None) -> MarketSimulator:
    """Create and return MarketSimulator instance"""
    return MarketSimulator(config)


# Example usage
def generate_test_data():
    """Generate test data for demonstration"""
    simulator = MarketSimulator()
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = simulator.generate_synthetic_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        frequency='1h'
    )
    
    # Add stress scenario
    stressed_data = simulator.add_market_stress_scenarios(data, "flash_crash")
    
    return stressed_data