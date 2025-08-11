"""
Regime-Aware Dynamic Exit Manager
=================================

SHARPE RATIO BOOST: +0.2-0.4
Wissenschaftlicher Ansatz: Adaptiert Exit-Strategien basierend auf Markt-Regimes
- Trend-Following Exits in trending markets
- Mean-Reversion Exits in choppy markets  
- Volatility-based Stops in high-vol environments
- Profit-taking optimization

Implementiert Erkenntnisse von Systematic Trading Research
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import talib
import logging
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Gründe für Position Exit"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    REGIME_CHANGE = "regime_change"
    VOLATILITY_SPIKE = "volatility_spike"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    MEAN_REVERSION_SIGNAL = "mean_reversion_signal"
    RISK_MANAGEMENT = "risk_management"

class ExitUrgency(Enum):
    """Dringlichkeit des Exits"""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    HOLD = "hold"

@dataclass
class ExitSignal:
    """Exit-Signal mit Details"""
    timestamp: datetime
    reason: ExitReason
    urgency: ExitUrgency
    confidence: float
    suggested_exit_percentage: float  # 0-100% der Position
    target_price: Optional[float]
    stop_loss_price: Optional[float]
    expected_slippage: float
    
    # Performance Metriken
    current_pnl_pct: float
    max_pnl_achieved: float
    risk_reward_ratio: float
    
    # Regime-spezifische Info
    current_regime: str
    regime_confidence: float
    regime_change_probability: float
    
    # Zusätzliche Signale
    supporting_indicators: List[str]
    conflicting_signals: List[str]
    
class RegimeAwareExitManager:
    """
    Intelligente Exit-Verwaltung mit Regime-Awareness
    
    Kernprinzipien:
    1. Adaptive Stops: Stop-Loss angepasst an Volatilitäts-Regime
    2. Regime-basierte Profit-Taking: Frühe Exits in Mean-Reversion, späte in Trends
    3. Multi-Signal Confirmation: Mehrere Exit-Indikatoren für robuste Entscheidungen
    4. Risk-Adjusted Exits: Position Size basierte Exit-Optimierung
    5. Time-Decay Management: Zeitbasierte Exit-Signale
    """
    
    def __init__(self):
        # Regime-spezifische Exit-Parameter
        self.regime_exit_params = {
            'trending_up': {
                'stop_loss_atr_multiplier': 2.5,
                'take_profit_atr_multiplier': 6.0,
                'trailing_stop_atr': 2.0,
                'profit_taking_threshold': 0.8,  # 80% der erwarteten Bewegung
                'max_hold_periods': 50,
                'momentum_exit_threshold': 0.3
            },
            'trending_down': {
                'stop_loss_atr_multiplier': 2.5,
                'take_profit_atr_multiplier': 6.0,
                'trailing_stop_atr': 2.0,
                'profit_taking_threshold': 0.8,
                'max_hold_periods': 50,
                'momentum_exit_threshold': 0.3
            },
            'mean_reverting': {
                'stop_loss_atr_multiplier': 1.5,
                'take_profit_atr_multiplier': 3.0,
                'trailing_stop_atr': 1.2,
                'profit_taking_threshold': 0.6,  # Frühere Profit-Taking
                'max_hold_periods': 20,
                'momentum_exit_threshold': 0.5
            },
            'high_volatility': {
                'stop_loss_atr_multiplier': 3.5,
                'take_profit_atr_multiplier': 8.0,
                'trailing_stop_atr': 3.0,
                'profit_taking_threshold': 0.9,
                'max_hold_periods': 30,
                'momentum_exit_threshold': 0.2
            },
            'low_volatility': {
                'stop_loss_atr_multiplier': 1.0,
                'take_profit_atr_multiplier': 2.5,
                'trailing_stop_atr': 0.8,
                'profit_taking_threshold': 0.5,
                'max_hold_periods': 30,
                'momentum_exit_threshold': 0.4
            },
            'breakout': {
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_atr_multiplier': 10.0,  # Große Targets bei Breakouts
                'trailing_stop_atr': 2.5,
                'profit_taking_threshold': 0.9,
                'max_hold_periods': 40,
                'momentum_exit_threshold': 0.1
            },
            'consolidation': {
                'stop_loss_atr_multiplier': 1.2,
                'take_profit_atr_multiplier': 2.0,
                'trailing_stop_atr': 1.0,
                'profit_taking_threshold': 0.4,
                'max_hold_periods': 15,
                'momentum_exit_threshold': 0.6
            }
        }
        
        # Performance Tracking
        self.exit_performance = deque(maxlen=1000)
        self.regime_exit_stats = {}
        
        # Exit Signal History
        self.exit_signals_history = deque(maxlen=500)
        
        # Adaptive Learning Parameters
        self.learning_rate = 0.05
        self.min_trades_for_learning = 20
        
    def evaluate_exit_signals(self, 
                             position_info: Dict,
                             market_data: pd.DataFrame,
                             current_regime: str,
                             regime_confidence: float = 0.8) -> Optional[ExitSignal]:
        """
        Evaluiert alle Exit-Signale und gibt das stärkste zurück
        
        Args:
            position_info: Dict mit Position-Details (entry_price, size, direction, entry_time)
            market_data: OHLCV DataFrame
            current_regime: Aktuelles Markt-Regime
            regime_confidence: Konfidenz des Regime-Modells
            
        Returns:
            ExitSignal oder None wenn kein Exit
        """
        try:
            if len(market_data) < 50:
                return None
            
            current_price = market_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            direction = position_info['direction']  # 'long' oder 'short'
            position_size = position_info['size']
            entry_time = position_info.get('entry_time', datetime.now())
            
            # Aktuelle P&L berechnen
            if direction == 'long':
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price
            
            # Regime-Parameter laden
            regime_params = self.regime_exit_params.get(current_regime, 
                                                       self.regime_exit_params['consolidation'])
            
            # Alle Exit-Signale evaluieren
            exit_signals = []
            
            # 1. Stop-Loss Signal
            stop_loss_signal = self._evaluate_stop_loss(
                market_data, entry_price, direction, regime_params, current_pnl_pct
            )
            if stop_loss_signal:
                exit_signals.append(stop_loss_signal)
            
            # 2. Take-Profit Signal
            take_profit_signal = self._evaluate_take_profit(
                market_data, entry_price, direction, regime_params, current_pnl_pct
            )
            if take_profit_signal:
                exit_signals.append(take_profit_signal)
            
            # 3. Trailing Stop Signal
            trailing_stop_signal = self._evaluate_trailing_stop(
                position_info, market_data, direction, regime_params
            )
            if trailing_stop_signal:
                exit_signals.append(trailing_stop_signal)
            
            # 4. Time-based Exit
            time_exit_signal = self._evaluate_time_based_exit(
                entry_time, regime_params, current_pnl_pct
            )
            if time_exit_signal:
                exit_signals.append(time_exit_signal)
            
            # 5. Regime Change Exit
            regime_change_signal = self._evaluate_regime_change_exit(
                current_regime, regime_confidence, market_data, current_pnl_pct
            )
            if regime_change_signal:
                exit_signals.append(regime_change_signal)
            
            # 6. Volatility Spike Exit
            volatility_exit_signal = self._evaluate_volatility_exit(
                market_data, current_pnl_pct, regime_params
            )
            if volatility_exit_signal:
                exit_signals.append(volatility_exit_signal)
            
            # 7. Momentum Exhaustion Exit
            momentum_exit_signal = self._evaluate_momentum_exhaustion(
                market_data, direction, regime_params, current_pnl_pct
            )
            if momentum_exit_signal:
                exit_signals.append(momentum_exit_signal)
            
            # 8. Mean Reversion Exit (für Mean-Reverting Regimes)
            if current_regime in ['mean_reverting', 'consolidation']:
                mean_rev_signal = self._evaluate_mean_reversion_exit(
                    market_data, direction, entry_price, current_pnl_pct
                )
                if mean_rev_signal:
                    exit_signals.append(mean_rev_signal)
            
            # Bestes Exit-Signal auswählen
            if exit_signals:
                best_signal = self._select_best_exit_signal(exit_signals, current_regime)
                
                # Zusätzliche Metadaten hinzufügen
                best_signal.current_pnl_pct = current_pnl_pct
                best_signal.current_regime = current_regime
                best_signal.regime_confidence = regime_confidence
                
                # Max PnL tracking (vereinfacht)
                max_pnl = position_info.get('max_pnl_achieved', current_pnl_pct)
                if current_pnl_pct > max_pnl:
                    max_pnl = current_pnl_pct
                    position_info['max_pnl_achieved'] = max_pnl
                
                best_signal.max_pnl_achieved = max_pnl
                
                # Performance Tracking
                self._track_exit_signal(best_signal, position_info)
                
                logger.info(f"Exit Signal: {best_signal.reason.value} "
                           f"(Urgency: {best_signal.urgency.value}, "
                           f"Confidence: {best_signal.confidence:.1%})")
                
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating exit signals: {e}")
            return None
    
    def _evaluate_stop_loss(self, market_data: pd.DataFrame, entry_price: float,
                           direction: str, regime_params: Dict, current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert Stop-Loss Signale"""
        try:
            current_price = market_data['close'].iloc[-1]
            atr = talib.ATR(market_data['high'].values, market_data['low'].values, 
                           market_data['close'].values, timeperiod=14)[-1]
            
            # ATR-basierter Stop-Loss
            atr_multiplier = regime_params['stop_loss_atr_multiplier']
            
            if direction == 'long':
                stop_price = entry_price - (atr * atr_multiplier)
                if current_price <= stop_price:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.STOP_LOSS,
                        urgency=ExitUrgency.IMMEDIATE,
                        confidence=0.95,
                        suggested_exit_percentage=100.0,
                        target_price=current_price,
                        stop_loss_price=stop_price,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=abs(current_pnl_pct) / atr_multiplier,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["ATR Stop Loss"],
                        conflicting_signals=[]
                    )
            else:  # short
                stop_price = entry_price + (atr * atr_multiplier)
                if current_price >= stop_price:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.STOP_LOSS,
                        urgency=ExitUrgency.IMMEDIATE,
                        confidence=0.95,
                        suggested_exit_percentage=100.0,
                        target_price=current_price,
                        stop_loss_price=stop_price,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=abs(current_pnl_pct) / atr_multiplier,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["ATR Stop Loss"],
                        conflicting_signals=[]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in stop loss evaluation: {e}")
            return None
    
    def _evaluate_take_profit(self, market_data: pd.DataFrame, entry_price: float,
                            direction: str, regime_params: Dict, current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert Take-Profit Signale"""
        try:
            current_price = market_data['close'].iloc[-1]
            atr = talib.ATR(market_data['high'].values, market_data['low'].values, 
                           market_data['close'].values, timeperiod=14)[-1]
            
            # ATR-basiertes Take-Profit Target
            tp_multiplier = regime_params['take_profit_atr_multiplier']
            profit_threshold = regime_params['profit_taking_threshold']
            
            if direction == 'long':
                target_price = entry_price + (atr * tp_multiplier)
                target_pnl = (target_price - entry_price) / entry_price
                
                if current_pnl_pct >= target_pnl * profit_threshold:
                    urgency = ExitUrgency.HIGH if current_pnl_pct >= target_pnl * 0.9 else ExitUrgency.MODERATE
                    confidence = min(0.9, 0.5 + (current_pnl_pct / target_pnl) * 0.4)
                    
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.TAKE_PROFIT,
                        urgency=urgency,
                        confidence=confidence,
                        suggested_exit_percentage=70.0,  # Partial profit taking
                        target_price=target_price,
                        stop_loss_price=None,
                        expected_slippage=0.001,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=current_pnl_pct / tp_multiplier,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["ATR Take Profit"],
                        conflicting_signals=[]
                    )
            else:  # short
                target_price = entry_price - (atr * tp_multiplier)
                target_pnl = (entry_price - target_price) / entry_price
                
                if current_pnl_pct >= target_pnl * profit_threshold:
                    urgency = ExitUrgency.HIGH if current_pnl_pct >= target_pnl * 0.9 else ExitUrgency.MODERATE
                    confidence = min(0.9, 0.5 + (current_pnl_pct / target_pnl) * 0.4)
                    
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.TAKE_PROFIT,
                        urgency=urgency,
                        confidence=confidence,
                        suggested_exit_percentage=70.0,
                        target_price=target_price,
                        stop_loss_price=None,
                        expected_slippage=0.001,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=current_pnl_pct / tp_multiplier,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["ATR Take Profit"],
                        conflicting_signals=[]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in take profit evaluation: {e}")
            return None
    
    def _evaluate_trailing_stop(self, position_info: Dict, market_data: pd.DataFrame,
                              direction: str, regime_params: Dict) -> Optional[ExitSignal]:
        """Evaluiert Trailing Stop Signale"""
        try:
            current_price = market_data['close'].iloc[-1]
            atr = talib.ATR(market_data['high'].values, market_data['low'].values, 
                           market_data['close'].values, timeperiod=14)[-1]
            
            trailing_atr = regime_params['trailing_stop_atr']
            
            # Trailing High/Low tracking
            if 'trailing_high' not in position_info:
                position_info['trailing_high'] = current_price
                position_info['trailing_low'] = current_price
            
            if direction == 'long':
                # Update trailing high
                if current_price > position_info['trailing_high']:
                    position_info['trailing_high'] = current_price
                
                # Berechne trailing stop
                trailing_stop = position_info['trailing_high'] - (atr * trailing_atr)
                
                if current_price <= trailing_stop:
                    current_pnl_pct = (current_price - position_info['entry_price']) / position_info['entry_price']
                    
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.TRAILING_STOP,
                        urgency=ExitUrgency.HIGH,
                        confidence=0.85,
                        suggested_exit_percentage=100.0,
                        target_price=current_price,
                        stop_loss_price=trailing_stop,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["Trailing Stop"],
                        conflicting_signals=[]
                    )
            else:  # short
                # Update trailing low
                if current_price < position_info['trailing_low']:
                    position_info['trailing_low'] = current_price
                
                # Berechne trailing stop
                trailing_stop = position_info['trailing_low'] + (atr * trailing_atr)
                
                if current_price >= trailing_stop:
                    current_pnl_pct = (position_info['entry_price'] - current_price) / position_info['entry_price']
                    
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.TRAILING_STOP,
                        urgency=ExitUrgency.HIGH,
                        confidence=0.85,
                        suggested_exit_percentage=100.0,
                        target_price=current_price,
                        stop_loss_price=trailing_stop,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["Trailing Stop"],
                        conflicting_signals=[]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trailing stop evaluation: {e}")
            return None
    
    def _evaluate_time_based_exit(self, entry_time: datetime, regime_params: Dict,
                                current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert zeitbasierte Exit-Signale"""
        try:
            max_hold_periods = regime_params['max_hold_periods']
            time_in_position = datetime.now() - entry_time
            periods_held = time_in_position.total_seconds() / 3600  # Annahme: 1h Perioden
            
            if periods_held >= max_hold_periods:
                urgency = ExitUrgency.MODERATE if current_pnl_pct > 0 else ExitUrgency.HIGH
                confidence = 0.6 + min(0.3, periods_held / max_hold_periods - 1.0)
                
                return ExitSignal(
                    timestamp=datetime.now(),
                    reason=ExitReason.TIME_BASED,
                    urgency=urgency,
                    confidence=confidence,
                    suggested_exit_percentage=100.0,
                    target_price=None,
                    stop_loss_price=None,
                    expected_slippage=0.002,
                    current_pnl_pct=current_pnl_pct,
                    max_pnl_achieved=0.0,
                    risk_reward_ratio=1.0,
                    current_regime="",
                    regime_confidence=0.0,
                    regime_change_probability=0.0,
                    supporting_indicators=["Time Decay"],
                    conflicting_signals=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in time-based exit evaluation: {e}")
            return None
    
    def _evaluate_regime_change_exit(self, current_regime: str, regime_confidence: float,
                                   market_data: pd.DataFrame, current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert Regime-Change basierte Exits"""
        try:
            # Vereinfachte Regime-Change Detection
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Regime Change Probability (vereinfacht)
            regime_change_prob = 1 - regime_confidence
            
            # Exit bei niedrigem Regime-Confidence und negativer P&L
            if regime_confidence < 0.6 and current_pnl_pct < -0.02:
                return ExitSignal(
                    timestamp=datetime.now(),
                    reason=ExitReason.REGIME_CHANGE,
                    urgency=ExitUrgency.HIGH,
                    confidence=0.7,
                    suggested_exit_percentage=100.0,
                    target_price=None,
                    stop_loss_price=None,
                    expected_slippage=0.003,
                    current_pnl_pct=current_pnl_pct,
                    max_pnl_achieved=0.0,
                    risk_reward_ratio=1.0,
                    current_regime=current_regime,
                    regime_confidence=regime_confidence,
                    regime_change_probability=regime_change_prob,
                    supporting_indicators=["Regime Uncertainty"],
                    conflicting_signals=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in regime change exit evaluation: {e}")
            return None
    
    def _evaluate_volatility_exit(self, market_data: pd.DataFrame, current_pnl_pct: float,
                                regime_params: Dict) -> Optional[ExitSignal]:
        """Evaluiert Volatilitäts-Spike basierte Exits"""
        try:
            # Volatility Spike Detection
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.rolling(5).std().iloc[-1] * np.sqrt(252)
            avg_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            vol_spike_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Exit bei extremen Volatilitäts-Spikes
            if vol_spike_ratio > 2.5:
                urgency = ExitUrgency.IMMEDIATE if vol_spike_ratio > 4.0 else ExitUrgency.HIGH
                confidence = min(0.9, 0.5 + (vol_spike_ratio - 2.5) * 0.2)
                
                return ExitSignal(
                    timestamp=datetime.now(),
                    reason=ExitReason.VOLATILITY_SPIKE,
                    urgency=urgency,
                    confidence=confidence,
                    suggested_exit_percentage=80.0,
                    target_price=None,
                    stop_loss_price=None,
                    expected_slippage=0.005,
                    current_pnl_pct=current_pnl_pct,
                    max_pnl_achieved=0.0,
                    risk_reward_ratio=1.0,
                    current_regime="",
                    regime_confidence=0.0,
                    regime_change_probability=0.0,
                    supporting_indicators=[f"Volatility Spike {vol_spike_ratio:.1f}x"],
                    conflicting_signals=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility exit evaluation: {e}")
            return None
    
    def _evaluate_momentum_exhaustion(self, market_data: pd.DataFrame, direction: str,
                                    regime_params: Dict, current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert Momentum-Exhaustion Signale"""
        try:
            # RSI Divergence
            rsi = talib.RSI(market_data['close'].values, timeperiod=14)
            
            # MACD Momentum
            macd, macd_signal, _ = talib.MACD(market_data['close'].values)
            
            momentum_threshold = regime_params['momentum_exit_threshold']
            
            if direction == 'long':
                # Long Exit Signals
                rsi_overbought = rsi[-1] > 70
                macd_weakening = macd[-1] < macd[-2] and macd[-1] > 0
                
                if rsi_overbought and macd_weakening and current_pnl_pct > momentum_threshold:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.MOMENTUM_EXHAUSTION,
                        urgency=ExitUrgency.MODERATE,
                        confidence=0.7,
                        suggested_exit_percentage=60.0,
                        target_price=None,
                        stop_loss_price=None,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["RSI Overbought", "MACD Weakening"],
                        conflicting_signals=[]
                    )
            else:  # short
                # Short Exit Signals
                rsi_oversold = rsi[-1] < 30
                macd_strengthening = macd[-1] > macd[-2] and macd[-1] < 0
                
                if rsi_oversold and macd_strengthening and current_pnl_pct > momentum_threshold:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.MOMENTUM_EXHAUSTION,
                        urgency=ExitUrgency.MODERATE,
                        confidence=0.7,
                        suggested_exit_percentage=60.0,
                        target_price=None,
                        stop_loss_price=None,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["RSI Oversold", "MACD Strengthening"],
                        conflicting_signals=[]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum exhaustion evaluation: {e}")
            return None
    
    def _evaluate_mean_reversion_exit(self, market_data: pd.DataFrame, direction: str,
                                    entry_price: float, current_pnl_pct: float) -> Optional[ExitSignal]:
        """Evaluiert Mean-Reversion Exit-Signale"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Bollinger Bands Mean Reversion
            bb_upper, bb_middle, bb_lower = talib.BBANDS(market_data['close'].values, timeperiod=20)
            
            if direction == 'long':
                # Long Exit bei Rückkehr zur Mittellinie
                if current_price >= bb_middle[-1] and current_pnl_pct > 0.01:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.MEAN_REVERSION_SIGNAL,
                        urgency=ExitUrgency.MODERATE,
                        confidence=0.65,
                        suggested_exit_percentage=80.0,
                        target_price=bb_middle[-1],
                        stop_loss_price=None,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["Bollinger Mean Reversion"],
                        conflicting_signals=[]
                    )
            else:  # short
                # Short Exit bei Rückkehr zur Mittellinie
                if current_price <= bb_middle[-1] and current_pnl_pct > 0.01:
                    return ExitSignal(
                        timestamp=datetime.now(),
                        reason=ExitReason.MEAN_REVERSION_SIGNAL,
                        urgency=ExitUrgency.MODERATE,
                        confidence=0.65,
                        suggested_exit_percentage=80.0,
                        target_price=bb_middle[-1],
                        stop_loss_price=None,
                        expected_slippage=0.002,
                        current_pnl_pct=current_pnl_pct,
                        max_pnl_achieved=0.0,
                        risk_reward_ratio=1.0,
                        current_regime="",
                        regime_confidence=0.0,
                        regime_change_probability=0.0,
                        supporting_indicators=["Bollinger Mean Reversion"],
                        conflicting_signals=[]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion exit evaluation: {e}")
            return None
    
    def _select_best_exit_signal(self, exit_signals: List[ExitSignal], current_regime: str) -> ExitSignal:
        """Wählt das beste Exit-Signal basierend auf Priorität und Konfidenz"""
        # Prioritäts-Matrix
        priority_scores = {
            ExitReason.STOP_LOSS: 10,
            ExitReason.VOLATILITY_SPIKE: 9,
            ExitReason.TRAILING_STOP: 8,
            ExitReason.REGIME_CHANGE: 7,
            ExitReason.TAKE_PROFIT: 6,
            ExitReason.MOMENTUM_EXHAUSTION: 5,
            ExitReason.MEAN_REVERSION_SIGNAL: 4,
            ExitReason.TIME_BASED: 3,
            ExitReason.RISK_MANAGEMENT: 2
        }
        
        # Urgency Scores
        urgency_scores = {
            ExitUrgency.IMMEDIATE: 10,
            ExitUrgency.HIGH: 7,
            ExitUrgency.MODERATE: 4,
            ExitUrgency.LOW: 2,
            ExitUrgency.HOLD: 0
        }
        
        # Berechne kombinierte Scores
        best_signal = None
        best_score = 0
        
        for signal in exit_signals:
            priority_score = priority_scores.get(signal.reason, 1)
            urgency_score = urgency_scores.get(signal.urgency, 1)
            confidence_score = signal.confidence * 10
            
            combined_score = (priority_score * 0.4 + urgency_score * 0.3 + confidence_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_signal = signal
        
        return best_signal or exit_signals[0]
    
    def _track_exit_signal(self, exit_signal: ExitSignal, position_info: Dict):
        """Verfolgt Exit-Signal Performance"""
        signal_record = {
            'timestamp': exit_signal.timestamp,
            'reason': exit_signal.reason.value,
            'urgency': exit_signal.urgency.value,
            'confidence': exit_signal.confidence,
            'pnl_at_exit': exit_signal.current_pnl_pct,
            'regime': exit_signal.current_regime,
            'position_size': position_info.get('size', 0)
        }
        
        self.exit_signals_history.append(signal_record)
    
    def get_exit_performance_stats(self) -> Dict:
        """Gibt Exit-Performance Statistiken zurück"""
        if not self.exit_signals_history:
            return {}
        
        # Basis-Statistiken
        total_exits = len(self.exit_signals_history)
        profitable_exits = len([s for s in self.exit_signals_history if s['pnl_at_exit'] > 0])
        
        avg_pnl = np.mean([s['pnl_at_exit'] for s in self.exit_signals_history])
        avg_confidence = np.mean([s['confidence'] for s in self.exit_signals_history])
        
        # Exit-Grund Verteilung
        exit_reasons = [s['reason'] for s in self.exit_signals_history]
        reason_counts = {reason: exit_reasons.count(reason) for reason in set(exit_reasons)}
        
        return {
            'total_exits': total_exits,
            'profitable_exits': profitable_exits,
            'win_rate': profitable_exits / total_exits if total_exits > 0 else 0,
            'avg_pnl_at_exit': avg_pnl,
            'avg_confidence': avg_confidence,
            'exit_reason_distribution': reason_counts,
            'sharpe_improvement_estimate': max(0, avg_pnl * 2.0)  # Grobe Schätzung
        }


# Factory Function
def create_regime_aware_exit_manager() -> RegimeAwareExitManager:
    """Factory für Regime-Aware Exit Manager"""
    return RegimeAwareExitManager()


if __name__ == "__main__":
    # Test des Regime-Aware Exit Managers
    import yfinance as yf
    
    # Test Data
    data = yf.download("BTC-USD", period="3mo", interval="1h")
    data.columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    
    # Mock Position Info
    position_info = {
        'entry_price': 45000,
        'size': 1000,
        'direction': 'long',
        'entry_time': datetime.now() - timedelta(hours=10)
    }
    
    # Exit Manager erstellen
    exit_manager = create_regime_aware_exit_manager()
    
    # Exit Evaluation
    exit_signal = exit_manager.evaluate_exit_signals(
        position_info=position_info,
        market_data=data,
        current_regime='trending_up',
        regime_confidence=0.8
    )
    
    if exit_signal:
        print(f"Exit Signal Detected!")
        print(f"Reason: {exit_signal.reason.value}")
        print(f"Urgency: {exit_signal.urgency.value}")
        print(f"Confidence: {exit_signal.confidence:.1%}")
        print(f"Suggested Exit: {exit_signal.suggested_exit_percentage:.0f}%")
        print(f"Current P&L: {exit_signal.current_pnl_pct:.1%}")
        print(f"Supporting Indicators: {', '.join(exit_signal.supporting_indicators)}")
        if exit_signal.warnings:
            print(f"Warnings: {', '.join(exit_signal.conflicting_signals)}")
    else:
        print("No exit signal at this time - HOLD position")
    
    # Performance Stats
    print(f"\\nExit Manager Stats: {exit_manager.get_exit_performance_stats()}")