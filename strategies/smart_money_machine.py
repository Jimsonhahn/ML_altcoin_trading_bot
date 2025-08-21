"""
Smart Money Machine - Intelligente Portfolio-Split-Strategy
=========================================================

Diese Strategie teilt das Portfolio intelligent auf:
- 10-20% High-Risk mit Leverage für maximale Rendite
- 80-90% Safe Portfolio für stabile Gewinne

Das Ergebnis: Eine echte "Money Machine" die sowohl sicher als auch profitabel ist.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Direkter Import um zirkuläre Imports zu vermeiden
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import pandas as pd

# Basis Strategy Interface
class Strategy(ABC):
    """Basis Strategy Klasse"""
    
    def __init__(self, exchange_manager, config: dict):
        self.exchange_manager = exchange_manager
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Berechnet Trading-Signal"""
        pass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Portfolio-Aufteilung zwischen Safe und High-Risk"""
    total_capital: float
    safe_allocation: float = 0.85  # 85% sicher
    high_risk_allocation: float = 0.15  # 15% high-risk
    
    @property
    def safe_capital(self) -> float:
        return self.total_capital * self.safe_allocation
    
    @property
    def high_risk_capital(self) -> float:
        return self.total_capital * self.high_risk_allocation
    
    def adjust_allocation(self, performance_ratio: float):
        """Passt Allocation basierend auf Performance an"""
        if performance_ratio > 1.2:  # High-risk outperforming
            self.high_risk_allocation = min(0.25, self.high_risk_allocation + 0.02)
        elif performance_ratio < 0.8:  # High-risk underperforming
            self.high_risk_allocation = max(0.05, self.high_risk_allocation - 0.02)
        
        self.safe_allocation = 1.0 - self.high_risk_allocation


@dataclass
class TradeConfig:
    """Konfiguration für Trade-Execution"""
    strategy_type: str  # 'safe' or 'high_risk'
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    leverage: float = 1.0
    max_daily_trades: int = 10
    confidence_threshold: float = 0.7


class SmartMoneyMachine(Strategy):
    """
    Intelligente Money Machine mit Portfolio-Split
    
    Features:
    - Safe Portfolio (85%): Konservative, profitable Strategien
    - High-Risk Portfolio (15%): Aggressive Strategies mit Leverage
    - Dynamische Allocation basierend auf Performance
    - Intelligent Risk Management für beide Portfolios
    """
    
    def __init__(self, exchange_manager, config: dict):
        super().__init__(exchange_manager, config)
        
        # Portfolio Setup
        total_capital = config.get('total_capital', 1000.0)
        self.portfolio_allocation = PortfolioAllocation(
            total_capital=total_capital,
            safe_allocation=config.get('safe_allocation', 0.85),
            high_risk_allocation=config.get('high_risk_allocation', 0.15)
        )
        
        # Trade Configurations
        self.safe_config = TradeConfig(
            strategy_type='safe',
            max_position_size=self.portfolio_allocation.safe_capital * 0.1,  # Max 10% per trade
            stop_loss_pct=2.0,  # 2% Stop Loss
            take_profit_pct=4.0,  # 4% Take Profit
            leverage=1.0,  # No leverage for safe
            max_daily_trades=5,
            confidence_threshold=0.8  # High confidence required
        )
        
        self.high_risk_config = TradeConfig(
            strategy_type='high_risk',
            max_position_size=self.portfolio_allocation.high_risk_capital * 0.3,  # Max 30% per trade
            stop_loss_pct=5.0,  # 5% Stop Loss (wider for leverage)
            take_profit_pct=15.0,  # 15% Take Profit (higher target)
            leverage=config.get('max_leverage', 3.0),  # 3x Leverage
            max_daily_trades=15,
            confidence_threshold=0.6  # Lower confidence ok for high-risk
        )
        
        # Performance Tracking
        self.safe_performance = {
            'total_pnl': 0.0,
            'trades': 0,
            'wins': 0,
            'daily_pnl': 0.0
        }
        
        self.high_risk_performance = {
            'total_pnl': 0.0,
            'trades': 0,
            'wins': 0,
            'daily_pnl': 0.0
        }
        
        # Daily Limits
        self.daily_trades = {'safe': 0, 'high_risk': 0}
        self.last_reset_date = datetime.now().date()
        
        # Market Analysis
        self.market_conditions = {}
        
        logger.info(f"Smart Money Machine initialized: Safe ${self.portfolio_allocation.safe_capital:.2f} | "
                   f"High-Risk ${self.portfolio_allocation.high_risk_capital:.2f}")
    
    def calculate_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Hauptlogik: Generiert Signale für beide Portfolio-Teile
        """
        try:
            # Daily Reset
            self._check_daily_reset()
            
            # Market Analysis
            market_analysis = self._analyze_market_conditions(symbol, timeframe)
            
            # Generate signals for both portfolios
            safe_signal = self._generate_safe_signal(symbol, timeframe, market_analysis)
            high_risk_signal = self._generate_high_risk_signal(symbol, timeframe, market_analysis)
            
            # Select best signal based on conditions and limits
            selected_signal = self._select_optimal_signal(safe_signal, high_risk_signal, market_analysis)
            
            # Apply final risk checks
            final_signal = self._apply_risk_management(selected_signal, symbol)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in Smart Money Machine: {str(e)}")
            return self._create_hold_signal(symbol, f"Error: {str(e)}")
    
    def _generate_safe_signal(self, symbol: str, timeframe: str, market_analysis: Dict) -> Dict[str, Any]:
        """Generiert sicheres, konservatives Signal"""
        
        if self.daily_trades['safe'] >= self.safe_config.max_daily_trades:
            return self._create_hold_signal(symbol, "Safe daily limit reached")
        
        try:
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=50)
            if df is None or len(df) < 20:
                return self._create_hold_signal(symbol, "Insufficient data for safe analysis")
            
            # Conservative indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20)
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1] if len(df) >= 50 else sma_20
            rsi = df['rsi'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # Safe Trading Logic (sehr konservativ)
            confidence = 0.0
            signal_strength = 0
            action = 'HOLD'
            reason = "Analyzing safe opportunities..."
            
            # Bullish conditions (very conservative)
            if (sma_20 > sma_50 and  # Uptrend
                current_price > sma_20 and  # Above trend
                rsi < 65 and  # Not overbought
                rsi > 45 and  # Not oversold
                current_price < bb_upper * 0.98 and  # Not at upper band
                market_analysis.get('volatility', 'high') in ['low', 'medium']):  # Low/medium volatility
                
                action = 'BUY'
                confidence = 0.85
                signal_strength = 2
                reason = "Conservative uptrend entry with low volatility"
            
            # Bearish conditions (for short or avoid)
            elif (sma_20 < sma_50 and  # Downtrend
                  current_price < sma_20 and  # Below trend
                  rsi > 35 and  # Not oversold
                  rsi < 55):  # Not overbought
                
                # In safe mode, we mostly avoid shorts unless very confident
                if market_analysis.get('trend_strength', 0) > 0.8:
                    action = 'SELL'
                    confidence = 0.75
                    signal_strength = 1
                    reason = "Strong downtrend - conservative short"
                else:
                    reason = "Downtrend detected - staying safe in cash"
            
            # Mean reversion opportunities (safe)
            elif (current_price <= bb_lower * 1.02 and  # Near lower bollinger
                  rsi < 35 and  # Oversold
                  market_analysis.get('support_level', 0) > 0):  # Near support
                
                action = 'BUY'
                confidence = 0.80
                signal_strength = 2
                reason = "Safe mean reversion opportunity"
            
            return {
                'action': action,
                'symbol': symbol,
                'position_size': self.safe_config.max_position_size if action != 'HOLD' else 0,
                'stop_loss_pct': self.safe_config.stop_loss_pct,
                'take_profit_pct': self.safe_config.take_profit_pct,
                'leverage': self.safe_config.leverage,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'reason': reason,
                'strategy_type': 'safe',
                'portfolio_allocation': 'safe'
            }
            
        except Exception as e:
            logger.error(f"Error generating safe signal: {str(e)}")
            return self._create_hold_signal(symbol, f"Safe signal error: {str(e)}")
    
    def _generate_high_risk_signal(self, symbol: str, timeframe: str, market_analysis: Dict) -> Dict[str, Any]:
        """Generiert aggressive High-Risk Signal mit Leverage"""
        
        if self.daily_trades['high_risk'] >= self.high_risk_config.max_daily_trades:
            return self._create_hold_signal(symbol, "High-risk daily limit reached")
        
        try:
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=100)
            if df is None or len(df) < 30:
                return self._create_hold_signal(symbol, "Insufficient data for high-risk analysis")
            
            # Aggressive indicators
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'] = df['ema_12'] - df['ema_26']
            df['volume_sma'] = df['volume'].rolling(20).mean()
            
            current_price = df['close'].iloc[-1]
            ema_12 = df['ema_12'].iloc[-1]
            ema_26 = df['ema_26'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            prev_macd = df['macd'].iloc[-2]
            volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
            
            # High-Risk Trading Logic (aggressiv)
            confidence = 0.0
            signal_strength = 0
            action = 'HOLD'
            reason = "Scanning for high-risk opportunities..."
            
            # Aggressive Bullish Momentum
            if (ema_12 > ema_26 and  # Bullish trend
                current_price > ema_12 and  # Above fast EMA
                macd > prev_macd and  # MACD increasing
                volume_ratio > 1.2 and  # Volume spike
                rsi > 50 and rsi < 80):  # Momentum zone
                
                action = 'BUY'
                confidence = 0.75
                signal_strength = 3
                reason = "Aggressive momentum breakout with volume"
            
            # Aggressive Bearish Momentum
            elif (ema_12 < ema_26 and  # Bearish trend
                  current_price < ema_12 and  # Below fast EMA
                  macd < prev_macd and  # MACD decreasing
                  volume_ratio > 1.1 and  # Volume confirmation
                  rsi < 50 and rsi > 20):  # Momentum zone
                
                action = 'SELL'
                confidence = 0.70
                signal_strength = 3
                reason = "Aggressive momentum breakdown"
            
            # Volatility Breakout (high-risk favorite)
            elif (market_analysis.get('volatility', 'low') == 'high' and
                  volume_ratio > 1.5 and  # Strong volume
                  abs(df['close'].pct_change().iloc[-1]) > 0.02):  # 2% move
                
                # Direction based on price action
                if df['close'].iloc[-1] > df['close'].iloc[-2]:
                    action = 'BUY'
                    reason = "Volatility breakout - bullish"
                else:
                    action = 'SELL'
                    reason = "Volatility breakout - bearish"
                
                confidence = 0.85
                signal_strength = 4
            
            # Oversold/Overbought Extremes (contrarian high-risk)
            elif rsi < 25 and volume_ratio > 1.1:  # Extreme oversold with volume
                action = 'BUY'
                confidence = 0.65
                signal_strength = 2
                reason = "Extreme oversold bounce play"
            
            elif rsi > 75 and volume_ratio > 1.1:  # Extreme overbought with volume
                action = 'SELL'
                confidence = 0.65
                signal_strength = 2
                reason = "Extreme overbought rejection play"
            
            return {
                'action': action,
                'symbol': symbol,
                'position_size': self.high_risk_config.max_position_size if action != 'HOLD' else 0,
                'stop_loss_pct': self.high_risk_config.stop_loss_pct,
                'take_profit_pct': self.high_risk_config.take_profit_pct,
                'leverage': self.high_risk_config.leverage,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'reason': reason,
                'strategy_type': 'high_risk',
                'portfolio_allocation': 'high_risk'
            }
            
        except Exception as e:
            logger.error(f"Error generating high-risk signal: {str(e)}")
            return self._create_hold_signal(symbol, f"High-risk signal error: {str(e)}")
    
    def _select_optimal_signal(self, safe_signal: Dict, high_risk_signal: Dict, market_analysis: Dict) -> Dict[str, Any]:
        """Wählt das optimale Signal basierend auf Marktbedingungen"""
        
        # If both are HOLD, return HOLD
        if safe_signal['action'] == 'HOLD' and high_risk_signal['action'] == 'HOLD':
            return safe_signal
        
        # Market condition based selection
        volatility = market_analysis.get('volatility', 'medium')
        
        # High volatility favors high-risk (if good signal)
        if volatility == 'high' and high_risk_signal['confidence'] > 0.7:
            return high_risk_signal
        
        # Low volatility favors safe strategies
        elif volatility == 'low' and safe_signal['confidence'] > 0.7:
            return safe_signal
        
        # Medium volatility - choose higher confidence
        elif volatility == 'medium':
            if safe_signal['confidence'] > high_risk_signal['confidence']:
                return safe_signal
            elif high_risk_signal['confidence'] > self.high_risk_config.confidence_threshold:
                return high_risk_signal
            else:
                return safe_signal
        
        # Default to higher confidence signal
        if safe_signal['confidence'] > high_risk_signal['confidence']:
            return safe_signal
        elif high_risk_signal['confidence'] > self.high_risk_config.confidence_threshold:
            return high_risk_signal
        else:
            return self._create_hold_signal(safe_signal['symbol'], "No confident signals available")
    
    def _apply_risk_management(self, signal: Dict, symbol: str) -> Dict[str, Any]:
        """Anwendung finaler Risk-Management-Regeln"""
        
        if signal['action'] == 'HOLD':
            return signal
        
        strategy_type = signal.get('strategy_type', 'safe')
        
        # Check daily limits
        if self.daily_trades[strategy_type] >= (
            self.safe_config.max_daily_trades if strategy_type == 'safe' 
            else self.high_risk_config.max_daily_trades
        ):
            return self._create_hold_signal(symbol, f"{strategy_type} daily limit reached")
        
        # Portfolio allocation check
        if strategy_type == 'safe':
            max_allocation = self.portfolio_allocation.safe_capital
        else:
            max_allocation = self.portfolio_allocation.high_risk_capital
        
        # Adjust position size if needed
        if signal['position_size'] > max_allocation:
            signal['position_size'] = max_allocation * 0.8  # Use 80% of available
            signal['reason'] += " (position size adjusted for risk)"
        
        # Minimum position size check
        min_position = 10.0  # $10 minimum
        if signal['position_size'] < min_position:
            return self._create_hold_signal(symbol, "Position size below minimum")
        
        return signal
    
    def _analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analysiert aktuelle Marktbedingungen"""
        try:
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=50)
            if df is None or len(df) < 20:
                return {'volatility': 'medium', 'trend_strength': 0.5}
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            if volatility < 0.015:
                vol_regime = 'low'
            elif volatility < 0.03:
                vol_regime = 'medium'
            else:
                vol_regime = 'high'
            
            # Trend strength
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            trend_strength = abs(current_price - sma_20) / sma_20
            
            # Support/Resistance levels
            recent_low = df['low'].tail(20).min()
            recent_high = df['high'].tail(20).max()
            support_distance = (current_price - recent_low) / current_price
            resistance_distance = (recent_high - current_price) / current_price
            
            return {
                'volatility': vol_regime,
                'trend_strength': trend_strength,
                'support_level': support_distance,
                'resistance_level': resistance_distance,
                'current_price': current_price,
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {'volatility': 'medium', 'trend_strength': 0.5}
    
    def _check_daily_reset(self):
        """Prüft und resettet Daily Limits"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(f"Daily reset - Safe P&L: ${self.safe_performance['daily_pnl']:.2f}, "
                       f"High-Risk P&L: ${self.high_risk_performance['daily_pnl']:.2f}")
            
            self.daily_trades = {'safe': 0, 'high_risk': 0}
            self.safe_performance['daily_pnl'] = 0.0
            self.high_risk_performance['daily_pnl'] = 0.0
            self.last_reset_date = today
            
            # Adjust allocation based on performance
            self._adjust_portfolio_allocation()
    
    def _adjust_portfolio_allocation(self):
        """Passt Portfolio-Allocation basierend auf Performance an"""
        if self.safe_performance['trades'] > 10 and self.high_risk_performance['trades'] > 10:
            safe_avg = (self.safe_performance['total_pnl'] / self.safe_performance['trades']) if self.safe_performance['trades'] > 0 else 0
            high_risk_avg = (self.high_risk_performance['total_pnl'] / self.high_risk_performance['trades']) if self.high_risk_performance['trades'] > 0 else 0
            
            if high_risk_avg > 0 and safe_avg > 0:
                performance_ratio = high_risk_avg / safe_avg
                self.portfolio_allocation.adjust_allocation(performance_ratio)
                
                logger.info(f"Portfolio allocation adjusted: Safe {self.portfolio_allocation.safe_allocation:.1%}, "
                           f"High-Risk {self.portfolio_allocation.high_risk_allocation:.1%}")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Berechnet RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Berechnet Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def _create_hold_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Erstellt HOLD Signal"""
        return {
            'action': 'HOLD',
            'symbol': symbol,
            'position_size': 0,
            'confidence': 0.0,
            'reason': reason,
            'strategy_type': 'smart_money_machine',
            'portfolio_allocation': 'none'
        }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Portfolio-Status zurück"""
        return {
            'total_capital': self.portfolio_allocation.total_capital,
            'safe_capital': self.portfolio_allocation.safe_capital,
            'high_risk_capital': self.portfolio_allocation.high_risk_capital,
            'safe_allocation': self.portfolio_allocation.safe_allocation,
            'high_risk_allocation': self.portfolio_allocation.high_risk_allocation,
            'safe_performance': self.safe_performance.copy(),
            'high_risk_performance': self.high_risk_performance.copy(),
            'daily_trades': self.daily_trades.copy()
        }
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Aktualisiert Performance nach einem Trade"""
        strategy_type = trade_result.get('strategy_type', 'safe')
        pnl = trade_result.get('pnl', 0.0)
        
        if strategy_type == 'safe':
            self.safe_performance['total_pnl'] += pnl
            self.safe_performance['daily_pnl'] += pnl
            self.safe_performance['trades'] += 1
            if pnl > 0:
                self.safe_performance['wins'] += 1
            self.daily_trades['safe'] += 1
            
        elif strategy_type == 'high_risk':
            self.high_risk_performance['total_pnl'] += pnl
            self.high_risk_performance['daily_pnl'] += pnl
            self.high_risk_performance['trades'] += 1
            if pnl > 0:
                self.high_risk_performance['wins'] += 1
            self.daily_trades['high_risk'] += 1
        
        logger.info(f"Performance updated for {strategy_type}: P&L ${pnl:.2f}")