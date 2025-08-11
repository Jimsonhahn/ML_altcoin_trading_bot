# risk/position_calculator.py
"""
Advanced Position Calculator with Kelly Criterion, VaR, and Dynamic Sizing
Implements sophisticated risk management for optimal position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Strategy performance statistics"""
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    
    @classmethod
    def from_returns(cls, returns: pd.Series) -> 'StrategyStats':
        """Calculate strategy stats from returns series"""
        if len(returns) == 0:
            return cls(0.5, 0.01, -0.01, 0, 1.0, 0.0, 0.0, 0.02)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
        avg_win = wins.mean() if len(wins) > 0 else 0.01
        avg_loss = losses.mean() if len(losses) > 0 else -0.01
        
        total_trades = len(returns)
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 1.0
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(252)
        
        return cls(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility
        )


@dataclass
class MarketConditions:
    """Current market conditions for position sizing"""
    symbol: str
    current_price: float
    atr: float  # Average True Range
    atr_ratio: float  # ATR / Price
    btc_correlation: float
    eth_correlation: float
    volume_ratio: float  # Current volume vs average
    market_regime: str  # bull, bear, sideways, volatile
    volatility_percentile: float  # Current volatility vs historical


@dataclass
class PositionSize:
    """Position sizing result"""
    symbol: str
    raw_size: float  # Kelly/base calculation
    adjusted_size: float  # After all adjustments
    max_size: float  # Maximum allowed
    risk_per_trade: float
    expected_return: float
    confidence: float
    reasoning: List[str]  # Explanation of adjustments
    
    @property
    def size_ratio(self) -> float:
        """Ratio of adjusted to raw size"""
        return self.adjusted_size / self.raw_size if self.raw_size > 0 else 0


class PositionCalculator:
    """
    Advanced Position Calculator implementing multiple risk management techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get('risk', {})
        
        # Risk parameters
        self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_total_risk = self.risk_config.get('max_total_risk', 0.10)  # 10%
        self.confidence_level = self.risk_config.get('var_confidence', 0.95)  # 95% VaR
        self.lookback_days = self.risk_config.get('lookback_days', 252)  # 1 year
        
        # Kelly parameters
        self.kelly_fraction = self.risk_config.get('kelly_fraction', 0.25)  # 25% of Kelly
        self.min_trades_for_kelly = self.risk_config.get('min_trades_for_kelly', 30)
        
        # Position limits
        self.max_position_size = self.risk_config.get('max_position_size', 0.20)  # 20% of equity
        self.min_position_size = self.risk_config.get('min_position_size', 0.001)  # 0.1% of equity
        
        # Correlation limits
        self.max_corr_adjustment = self.risk_config.get('max_correlation_adjustment', 0.5)
        
        logger.info("PositionCalculator initialized with advanced risk management")
    
    def calculate_position_size(self, 
                              strategy_stats: StrategyStats,
                              market_conditions: MarketConditions,
                              current_equity: float,
                              active_positions: Dict[str, float] = None) -> PositionSize:
        """
        Calculate optimal position size using multiple risk management techniques
        """
        try:
            active_positions = active_positions or {}
            reasoning = []
            
            # 1. Kelly Criterion base calculation
            kelly_size = self._calculate_kelly_size(strategy_stats, reasoning)
            
            # 2. VaR adjustment
            var_size = self._calculate_var_adjusted_size(
                kelly_size, strategy_stats, market_conditions, reasoning
            )
            
            # 3. Volatility adjustment
            vol_adjusted_size = self._apply_volatility_adjustment(
                var_size, market_conditions, reasoning
            )
            
            # 4. Correlation adjustment
            corr_adjusted_size = self._apply_correlation_adjustment(
                vol_adjusted_size, market_conditions, active_positions, reasoning
            )
            
            # 5. Market regime adjustment
            regime_adjusted_size = self._apply_regime_adjustment(
                corr_adjusted_size, market_conditions, reasoning
            )
            
            # 6. Apply hard limits
            final_size = self._apply_position_limits(
                regime_adjusted_size, current_equity, reasoning
            )
            
            # Calculate risk metrics
            risk_per_trade = final_size * market_conditions.atr_ratio
            expected_return = self._calculate_expected_return(strategy_stats, final_size)
            confidence = self._calculate_confidence(strategy_stats, market_conditions)
            
            return PositionSize(
                symbol=market_conditions.symbol,
                raw_size=kelly_size,
                adjusted_size=final_size,
                max_size=self.max_position_size,
                risk_per_trade=risk_per_trade,
                expected_return=expected_return,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Return minimal safe position
            return PositionSize(
                symbol=market_conditions.symbol,
                raw_size=self.min_position_size,
                adjusted_size=self.min_position_size,
                max_size=self.max_position_size,
                risk_per_trade=self.min_position_size * 0.02,
                expected_return=0.0,
                confidence=0.0,
                reasoning=[f"Error in calculation: {str(e)}"]
            )
    
    def _calculate_kelly_size(self, strategy_stats: StrategyStats, reasoning: List[str]) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            if strategy_stats.total_trades < self.min_trades_for_kelly:
                reasoning.append(f"Insufficient trades ({strategy_stats.total_trades}) for Kelly, using conservative size")
                return self.min_position_size * 2
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win / abs(avg_loss))
            #       p = probability of winning
            #       q = probability of losing (1-p)
            
            b = abs(strategy_stats.avg_win / strategy_stats.avg_loss) if strategy_stats.avg_loss != 0 else 1
            p = strategy_stats.win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b if b != 0 else 0
            
            # Apply Kelly fraction to avoid over-leveraging
            kelly_size = kelly_fraction * self.kelly_fraction
            
            # Ensure positive and reasonable
            kelly_size = max(0, min(kelly_size, self.max_position_size))
            
            reasoning.append(f"Kelly size: {kelly_size:.3f} (raw: {kelly_fraction:.3f}, fraction: {self.kelly_fraction})")
            
            return kelly_size
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            reasoning.append(f"Kelly calculation error, using min size")
            return self.min_position_size
    
    def _calculate_var_adjusted_size(self, 
                                   base_size: float, 
                                   strategy_stats: StrategyStats, 
                                   market_conditions: MarketConditions, 
                                   reasoning: List[str]) -> float:
        """Adjust position size based on Value at Risk"""
        try:
            # Estimate daily VaR using normal distribution assumption
            # VaR = z_score * volatility * position_size
            z_score = stats.norm.ppf(1 - self.confidence_level)  # e.g., -1.645 for 95%
            
            # Use strategy volatility and market ATR
            combined_volatility = np.sqrt(
                strategy_stats.volatility**2 + (market_conditions.atr_ratio * np.sqrt(252))**2
            )
            
            # Calculate maximum position size to stay within VaR limit
            max_var_size = self.max_risk_per_trade / (abs(z_score) * combined_volatility)
            
            var_adjusted_size = min(base_size, max_var_size)
            
            if var_adjusted_size < base_size:
                reduction = (1 - var_adjusted_size / base_size) * 100
                reasoning.append(f"VaR adjustment: -{reduction:.1f}% (volatility: {combined_volatility:.3f})")
            
            return var_adjusted_size
            
        except Exception as e:
            logger.error(f"Error in VaR calculation: {e}")
            reasoning.append("VaR calculation error, using base size")
            return base_size
    
    def _apply_volatility_adjustment(self, 
                                   base_size: float, 
                                   market_conditions: MarketConditions, 
                                   reasoning: List[str]) -> float:
        """Adjust position size based on current volatility conditions"""
        try:
            # Reduce size in high volatility environments
            if market_conditions.volatility_percentile > 0.8:  # Top 20% volatility
                volatility_factor = 0.5  # Reduce by 50%
                reasoning.append(f"High volatility adjustment: -50% (percentile: {market_conditions.volatility_percentile:.2f})")
            elif market_conditions.volatility_percentile > 0.6:  # Top 40% volatility
                volatility_factor = 0.75  # Reduce by 25%
                reasoning.append(f"Elevated volatility adjustment: -25% (percentile: {market_conditions.volatility_percentile:.2f})")
            else:
                volatility_factor = 1.0
            
            # Additional ATR-based adjustment
            if market_conditions.atr_ratio > 0.05:  # Very high ATR
                atr_factor = 0.8
                reasoning.append(f"High ATR adjustment: -20% (ATR ratio: {market_conditions.atr_ratio:.3f})")
            elif market_conditions.atr_ratio > 0.03:  # High ATR
                atr_factor = 0.9
                reasoning.append(f"Elevated ATR adjustment: -10% (ATR ratio: {market_conditions.atr_ratio:.3f})")
            else:
                atr_factor = 1.0
            
            combined_factor = volatility_factor * atr_factor
            return base_size * combined_factor
            
        except Exception as e:
            logger.error(f"Error in volatility adjustment: {e}")
            return base_size
    
    def _apply_correlation_adjustment(self, 
                                    base_size: float, 
                                    market_conditions: MarketConditions, 
                                    active_positions: Dict[str, float], 
                                    reasoning: List[str]) -> float:
        """Adjust position size based on correlation with existing positions"""
        try:
            if not active_positions:
                return base_size
            
            # Check correlation with BTC and ETH (major market drivers)
            max_correlation = max(
                abs(market_conditions.btc_correlation),
                abs(market_conditions.eth_correlation)
            )
            
            # Calculate exposure to correlated assets
            btc_exposure = active_positions.get('BTC/USDT', 0) + active_positions.get('BTCUSDT', 0)
            eth_exposure = active_positions.get('ETH/USDT', 0) + active_positions.get('ETHUSDT', 0)
            
            correlation_exposure = (
                btc_exposure * abs(market_conditions.btc_correlation) +
                eth_exposure * abs(market_conditions.eth_correlation)
            )
            
            # Reduce size if high correlation and significant exposure
            if max_correlation > 0.7 and correlation_exposure > 0.1:
                correlation_factor = 1 - (max_correlation * self.max_corr_adjustment)
                reasoning.append(f"High correlation adjustment: -{(1-correlation_factor)*100:.1f}% (corr: {max_correlation:.2f})")
                return base_size * correlation_factor
            elif max_correlation > 0.5 and correlation_exposure > 0.05:
                correlation_factor = 1 - (max_correlation * self.max_corr_adjustment * 0.5)
                reasoning.append(f"Moderate correlation adjustment: -{(1-correlation_factor)*100:.1f}% (corr: {max_correlation:.2f})")
                return base_size * correlation_factor
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error in correlation adjustment: {e}")
            return base_size
    
    def _apply_regime_adjustment(self, 
                               base_size: float, 
                               market_conditions: MarketConditions, 
                               reasoning: List[str]) -> float:
        """Adjust position size based on market regime"""
        try:
            regime_factors = {
                'bull': 1.2,      # Increase size in bull markets
                'bear': 0.6,      # Reduce size in bear markets
                'sideways': 0.8,  # Slightly reduce in sideways markets
                'volatile': 0.5,  # Significantly reduce in volatile markets
                'extreme_fear': 0.3,  # Minimal size in extreme fear
                'unknown': 0.8    # Conservative in unknown conditions
            }
            
            regime_factor = regime_factors.get(market_conditions.market_regime, 0.8)
            
            if regime_factor != 1.0:
                change = (regime_factor - 1) * 100
                reasoning.append(f"Market regime adjustment ({market_conditions.market_regime}): {change:+.1f}%")
            
            return base_size * regime_factor
            
        except Exception as e:
            logger.error(f"Error in regime adjustment: {e}")
            return base_size
    
    def _apply_position_limits(self, 
                             base_size: float, 
                             current_equity: float, 
                             reasoning: List[str]) -> float:
        """Apply final position size limits"""
        try:
            # Apply maximum position size limit
            if base_size > self.max_position_size:
                reasoning.append(f"Max position limit applied: {self.max_position_size:.3f}")
                base_size = self.max_position_size
            
            # Apply minimum position size limit
            if base_size < self.min_position_size:
                reasoning.append(f"Min position limit applied: {self.min_position_size:.3f}")
                base_size = self.min_position_size
            
            # Ensure we have enough equity
            dollar_amount = base_size * current_equity
            if dollar_amount < 10:  # Minimum $10 trade
                reasoning.append("Trade too small (<$10), setting to minimum")
                base_size = 10 / current_equity
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error applying position limits: {e}")
            return self.min_position_size
    
    def _calculate_expected_return(self, strategy_stats: StrategyStats, position_size: float) -> float:
        """Calculate expected return for the position"""
        try:
            # Expected return = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            expected_return_per_trade = (
                strategy_stats.win_rate * strategy_stats.avg_win +
                (1 - strategy_stats.win_rate) * strategy_stats.avg_loss
            )
            
            return expected_return_per_trade * position_size
            
        except Exception as e:
            logger.error(f"Error calculating expected return: {e}")
            return 0.0
    
    def _calculate_confidence(self, strategy_stats: StrategyStats, market_conditions: MarketConditions) -> float:
        """Calculate confidence score for the position sizing"""
        try:
            confidence_factors = []
            
            # Strategy track record confidence
            if strategy_stats.total_trades >= 100:
                confidence_factors.append(0.9)
            elif strategy_stats.total_trades >= 50:
                confidence_factors.append(0.7)
            elif strategy_stats.total_trades >= 20:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Sharpe ratio confidence
            if strategy_stats.sharpe_ratio > 2.0:
                confidence_factors.append(0.9)
            elif strategy_stats.sharpe_ratio > 1.0:
                confidence_factors.append(0.7)
            elif strategy_stats.sharpe_ratio > 0.5:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Market conditions confidence
            if market_conditions.market_regime in ['bull', 'sideways']:
                confidence_factors.append(0.8)
            elif market_conditions.market_regime in ['bear']:
                confidence_factors.append(0.6)
            else:  # volatile, extreme_fear, unknown
                confidence_factors.append(0.4)
            
            # Volume confidence
            if market_conditions.volume_ratio > 0.8:
                confidence_factors.append(0.8)
            elif market_conditions.volume_ratio > 0.5:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # Return average confidence
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def calculate_portfolio_var(self, 
                              positions: Dict[str, float], 
                              correlations: np.ndarray, 
                              volatilities: np.ndarray) -> float:
        """Calculate portfolio-level Value at Risk"""
        try:
            if len(positions) == 0:
                return 0.0
            
            # Convert positions to numpy array
            weights = np.array(list(positions.values()))
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(correlations, weights * volatilities**2))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - self.confidence_level)
            portfolio_var = abs(z_score) * portfolio_volatility
            
            return portfolio_var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def get_risk_metrics(self, position_size: PositionSize, current_equity: float) -> Dict[str, Any]:
        """Get comprehensive risk metrics for a position"""
        try:
            dollar_amount = position_size.adjusted_size * current_equity
            
            return {
                'position_size_pct': position_size.adjusted_size * 100,
                'dollar_amount': dollar_amount,
                'risk_per_trade_pct': position_size.risk_per_trade * 100,
                'expected_return_pct': position_size.expected_return * 100,
                'confidence_score': position_size.confidence,
                'size_efficiency': position_size.size_ratio,
                'adjustments_applied': len(position_size.reasoning) - 1,  # -1 for Kelly base
                'reasoning': position_size.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}


# Convenience functions for easy integration
def create_market_conditions(symbol: str, 
                           current_price: float, 
                           market_data: pd.DataFrame, 
                           correlations: Dict[str, float] = None,
                           market_regime: str = 'unknown') -> MarketConditions:
    """Create MarketConditions from market data"""
    try:
        correlations = correlations or {}
        
        # Calculate ATR
        if len(market_data) >= 14:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_ratio = atr / current_price
        else:
            atr = current_price * 0.02  # 2% fallback
            atr_ratio = 0.02
        
        # Calculate volume ratio
        if 'volume' in market_data.columns and len(market_data) >= 20:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Calculate volatility percentile
        if len(market_data) >= 252:
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std()
            current_vol = volatility.iloc[-1]
            vol_percentile = (volatility <= current_vol).sum() / len(volatility)
        else:
            vol_percentile = 0.5
        
        return MarketConditions(
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            atr_ratio=atr_ratio,
            btc_correlation=correlations.get('BTC', 0.0),
            eth_correlation=correlations.get('ETH', 0.0),
            volume_ratio=volume_ratio,
            market_regime=market_regime,
            volatility_percentile=vol_percentile
        )
        
    except Exception as e:
        logger.error(f"Error creating market conditions: {e}")
        # Return safe defaults
        return MarketConditions(
            symbol=symbol,
            current_price=current_price,
            atr=current_price * 0.02,
            atr_ratio=0.02,
            btc_correlation=0.0,
            eth_correlation=0.0,
            volume_ratio=1.0,
            market_regime='unknown',
            volatility_percentile=0.5
        )