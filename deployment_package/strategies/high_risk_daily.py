#!/usr/bin/env python3
"""
High-Risk Daily Trading Strategy
===============================

Aggressive daily trading strategy with strict budget controls:
- 30€ daily budget limit (hard stop)
- Target: 50-100% returns on winning trades
- 4-6 hour maximum hold times
- Multi-signal entry system
- Complete isolation from other strategies
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import uuid

from .strategy_base import Strategy
from core.risk_limiter import get_risk_limiter, DailyRiskLimiter
from core.volume_detector import create_volume_detector, VolumeDetector
from core.social_sentiment import create_sentiment_analyzer, SocialSentimentAnalyzer
from utils.high_risk_logger import get_high_risk_logger, HighRiskLogger

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Combined trading signal from multiple sources"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    sources: List[str]
    volume_score: float
    sentiment_score: float
    technical_score: float
    combined_score: float
    metadata: Dict[str, Any]

@dataclass
class ActivePosition:
    """Active trading position"""
    trade_id: str
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    quantity: float
    value: float
    budget_allocated: float
    confidence: float
    entry_signal: str
    current_pnl: float
    target_exit_time: datetime
    stop_loss: float
    profit_targets: List[float]
    metadata: Dict[str, Any]

class HighRiskDailyStrategy(Strategy):
    """
    Extreme high-risk daily trading strategy
    
    Key Features:
    - 30€ daily budget with hard limits
    - Multi-signal entry system (volume + sentiment + technical)
    - Aggressive profit targets (50-100%+)
    - Time-based exits (4-6 hours max)
    - Complete trade isolation
    - Real-time risk monitoring
    """
    
    def __init__(self, params: Dict[str, Any] = None, ml_components: Optional[Any] = None):
        super().__init__(params, ml_components)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize risk limiter
        self.risk_limiter = get_risk_limiter(self.config['risk_management']['daily_budget'])
        
        # Initialize detectors
        self.volume_detector = create_volume_detector(self.config.get('volume_config', {}))
        self.sentiment_analyzer = create_sentiment_analyzer(self.config.get('sentiment_config', {}))
        
        # Initialize logger
        self.hr_logger = get_high_risk_logger()
        
        # Active positions tracking
        self.active_positions: Dict[str, ActivePosition] = {}
        
        # Performance tracking
        self.daily_stats = {
            'signals_generated': 0,
            'signals_acted': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # Configuration parameters
        self.max_positions = self.config['risk_management']['position_limits']['max_positions']
        self.min_position_size = self.config['risk_management']['position_limits']['min_position_size']
        self.max_position_size = self.config['risk_management']['position_limits']['max_position_size']
        
        # Entry signal weights
        self.volume_weight = self.config['entry_signals']['volume_spike']['weight']
        self.sentiment_weight = self.config['entry_signals']['social_sentiment']['weight']
        self.technical_weight = self.config['entry_signals']['technical_breakout']['weight']
        
        # Exit parameters
        self.max_hold_hours = self.config['exit_conditions']['time_based']['max_hold_hours']
        self.profit_targets = [
            self.config['exit_conditions']['profit_targets']['target_1'],
            self.config['exit_conditions']['profit_targets']['target_2'],
            self.config['exit_conditions']['profit_targets']['target_3']
        ]
        self.stop_loss_pct = self.config['exit_conditions']['stop_losses']['initial_stop']
        
        self.hr_logger.log_strategy_init({
            'strategy': 'HighRiskDaily',
            'daily_budget': self.risk_limiter.daily_budget,
            'max_positions': self.max_positions,
            'profit_targets': self.profit_targets,
            'max_hold_hours': self.max_hold_hours
        })
        
        logger.info(f"🔥 High-Risk Daily Strategy initialized")
        logger.info(f"💰 Daily budget: {self.risk_limiter.daily_budget}€")
        logger.info(f"📊 Max positions: {self.max_positions}")
        logger.info(f"🎯 Profit targets: {self.profit_targets}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        try:
            config_path = "config/high_risk.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'risk_management': {
                'daily_budget': 30.0,
                'position_limits': {
                    'max_positions': 3,
                    'min_position_size': 2.0,
                    'max_position_size': 15.0
                }
            },
            'entry_signals': {
                'volume_spike': {'weight': 0.4, 'min_spike_ratio': 3.0},
                'social_sentiment': {'weight': 0.3, 'min_sentiment': 0.3},
                'technical_breakout': {'weight': 0.3, 'breakout_threshold': 0.02}
            },
            'exit_conditions': {
                'time_based': {'max_hold_hours': 6},
                'profit_targets': {'target_1': 0.25, 'target_2': 0.50, 'target_3': 1.00},
                'stop_losses': {'initial_stop': 0.15}
            }
        }
    
    async def calculate_signal(self, symbol: str, data: pd.DataFrame, 
                             current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate high-risk trading signal
        
        This is the main entry point for signal generation
        """
        try:
            # Check if trading is allowed
            can_trade, reason = self.risk_limiter.can_trade()
            if not can_trade:
                self.hr_logger.log_signal_ignored(
                    {'symbol': symbol}, 
                    f"Risk limiter: {reason}"
                )
                return 'HOLD', {'reason': reason, 'confidence': 0.0}
            
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                return 'HOLD', {'reason': 'max_positions_reached', 'confidence': 0.0}
            
            # Generate multi-source signals
            signals = await self._generate_combined_signals([symbol])
            
            if not signals:
                return 'HOLD', {'reason': 'no_signals_generated', 'confidence': 0.0}
            
            # Get best signal for this symbol
            symbol_signals = [s for s in signals if s.symbol == symbol]
            if not symbol_signals:
                return 'HOLD', {'reason': 'no_signal_for_symbol', 'confidence': 0.0}
            
            best_signal = max(symbol_signals, key=lambda x: x.combined_score)
            
            # Check minimum confidence
            min_confidence = 0.7
            if best_signal.confidence < min_confidence:
                self.hr_logger.log_signal_ignored(
                    {'symbol': symbol, 'confidence': best_signal.confidence},
                    f"Low confidence: {best_signal.confidence:.2f} < {min_confidence}"
                )
                return 'HOLD', {'reason': 'low_confidence', 'confidence': best_signal.confidence}
            
            # Log signal generation
            self.daily_stats['signals_generated'] += 1
            self.hr_logger.log_signal_generated({
                'symbol': symbol,
                'type': best_signal.signal_type,
                'confidence': best_signal.confidence,
                'sources': best_signal.sources,
                'combined_score': best_signal.combined_score,
                'metadata': best_signal.metadata
            })
            
            if best_signal.signal_type in ['BUY', 'SELL']:
                return best_signal.signal_type, {
                    'confidence': best_signal.confidence,
                    'sources': best_signal.sources,
                    'volume_score': best_signal.volume_score,
                    'sentiment_score': best_signal.sentiment_score,
                    'technical_score': best_signal.technical_score,
                    'combined_score': best_signal.combined_score,
                    'metadata': best_signal.metadata,
                    'strategy': 'high_risk_daily'
                }
            
            return 'HOLD', {'reason': 'hold_signal', 'confidence': best_signal.confidence}
            
        except Exception as e:
            logger.error(f"Error calculating signal for {symbol}: {e}")
            self.hr_logger.log_risk_event('signal_calculation_error', {
                'symbol': symbol,
                'error': str(e)
            })
            return 'HOLD', {'reason': 'error', 'error': str(e), 'confidence': 0.0}
    
    async def _generate_combined_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate combined signals from all sources"""
        all_signals = []
        
        try:
            # Parallel signal generation
            volume_task = self.volume_detector.detect_volume_spikes(symbols)
            sentiment_task = self.sentiment_analyzer.analyze_sentiment(symbols)
            
            # Wait for all tasks
            volume_spikes = await volume_task
            sentiment_signals = await sentiment_task
            
            # Process each symbol
            for symbol in symbols:
                signal = await self._combine_signals_for_symbol(
                    symbol, volume_spikes, sentiment_signals
                )
                
                if signal:
                    all_signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating combined signals: {e}")
            self.hr_logger.log_risk_event('signal_generation_error', {'error': str(e)})
        
        return all_signals
    
    async def _combine_signals_for_symbol(self, symbol: str, volume_spikes: List, 
                                        sentiment_signals: List) -> Optional[TradingSignal]:
        """Combine all signal sources for a single symbol"""
        
        # Find relevant signals
        symbol_volume = [v for v in volume_spikes if v.symbol == symbol]
        symbol_sentiment = [s for s in sentiment_signals if s.symbol == symbol]
        
        # Calculate individual scores
        volume_score = 0.0
        sentiment_score = 0.0
        technical_score = 0.0
        
        sources = []
        metadata = {}
        
        # Volume score
        if symbol_volume:
            best_volume = max(symbol_volume, key=lambda x: x.confidence)
            volume_score = best_volume.confidence * best_volume.spike_ratio / 5.0  # Normalize
            sources.append('volume')
            metadata['volume'] = {
                'spike_ratio': best_volume.spike_ratio,
                'confidence': best_volume.confidence,
                'breakout_detected': best_volume.breakout_detected
            }
        
        # Sentiment score
        if symbol_sentiment:
            best_sentiment = max(symbol_sentiment, key=lambda x: x.confidence)
            sentiment_score = abs(best_sentiment.sentiment_score) * best_sentiment.confidence
            sources.append('sentiment')
            metadata['sentiment'] = {
                'sentiment_score': best_sentiment.sentiment_score,
                'confidence': best_sentiment.confidence,
                'momentum_score': best_sentiment.momentum_score
            }
        
        # Technical score (simplified - could be enhanced)
        technical_score = self._calculate_technical_score(symbol)
        if technical_score > 0.3:
            sources.append('technical')
            metadata['technical'] = {'score': technical_score}
        
        # Combine scores
        combined_score = (
            volume_score * self.volume_weight +
            sentiment_score * self.sentiment_weight +
            technical_score * self.technical_weight
        )
        
        # Determine signal type and confidence
        if combined_score >= 0.6:
            signal_type = 'BUY'  # Simplified - mainly long bias for this strategy
            confidence = min(combined_score, 1.0)
        else:
            signal_type = 'HOLD'
            confidence = combined_score
        
        # Must have at least one strong source
        if not sources or combined_score < 0.4:
            return None
        
        return TradingSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            sources=sources,
            volume_score=volume_score,
            sentiment_score=sentiment_score,
            technical_score=technical_score,
            combined_score=combined_score,
            metadata=metadata
        )
    
    def _calculate_technical_score(self, symbol: str) -> float:
        """Calculate technical analysis score (simplified implementation)"""
        # In a real implementation, this would analyze:
        # - Chart patterns (triangles, flags, etc.)
        # - Support/resistance levels
        # - Technical indicators
        # - Breakout confirmations
        
        # For demo, return random score based on symbol
        random.seed(hash(symbol) + int(datetime.now().timestamp() / 3600))
        return random.uniform(0.2, 0.8)
    
    async def execute_trade(self, signal: str, symbol: str, current_price: float, 
                          signal_data: Dict[str, Any]) -> bool:
        """Execute high-risk trade"""
        
        if signal == 'HOLD':
            return False
        
        try:
            # Calculate position size
            confidence = signal_data.get('confidence', 0.5)
            position_value = self._calculate_position_size(confidence)
            
            # Reserve budget
            if not self.risk_limiter.reserve_budget(position_value):
                self.hr_logger.log_risk_event('budget_reservation_failed', {
                    'symbol': symbol,
                    'requested_amount': position_value
                })
                return False
            
            # Calculate quantity
            quantity = position_value / current_price
            
            # Create position
            trade_id = str(uuid.uuid4())[:8]
            
            position = ActivePosition(
                trade_id=trade_id,
                symbol=symbol,
                side=signal,
                entry_time=datetime.now(),
                entry_price=current_price,
                quantity=quantity,
                value=position_value,
                budget_allocated=position_value,
                confidence=confidence,
                entry_signal=','.join(signal_data.get('sources', [])),
                current_pnl=0.0,
                target_exit_time=datetime.now() + timedelta(hours=self.max_hold_hours),
                stop_loss=current_price * (1 - self.stop_loss_pct) if signal == 'BUY' else current_price * (1 + self.stop_loss_pct),
                profit_targets=[
                    current_price * (1 + target) if signal == 'BUY' else current_price * (1 - target)
                    for target in self.profit_targets
                ],
                metadata=signal_data
            )
            
            # Store position
            self.active_positions[trade_id] = position
            
            # Log trade
            self.hr_logger.log_trade_entry({
                'trade_id': trade_id,
                'symbol': symbol,
                'side': signal,
                'quantity': quantity,
                'price': current_price,
                'value': position_value,
                'commission': position_value * 0.001,  # Assume 0.1% commission
                'budget_used': position_value,
                'remaining_budget': self.risk_limiter.get_status()['remaining_budget'],
                'confidence': confidence,
                'entry_signal': position.entry_signal,
                'metadata': signal_data
            })
            
            # Update stats
            self.daily_stats['signals_acted'] += 1
            self.daily_stats['trades_executed'] += 1
            
            logger.info(f"🔥 HIGH-RISK TRADE: {symbol} {signal} {quantity:.6f} @ ${current_price:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.hr_logger.log_risk_event('trade_execution_error', {
                'symbol': symbol,
                'signal': signal,
                'error': str(e)
            })
            return False
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk limits"""
        
        # Base size from remaining budget
        remaining_budget = self.risk_limiter.get_status()['remaining_budget']
        
        # Scale by confidence (0.5 to 1.0 range)
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        # Calculate size
        position_size = remaining_budget * confidence_multiplier * 0.8  # Use max 80% of remaining
        
        # Apply limits
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))
        position_size = min(position_size, remaining_budget)
        
        return position_size
    
    async def manage_positions(self, current_prices: Dict[str, float]):
        """Manage active positions - check exits"""
        
        positions_to_close = []
        
        for trade_id, position in self.active_positions.items():
            current_price = current_prices.get(position.symbol)
            if current_price is None:
                continue
            
            # Update current P&L
            if position.side == 'BUY':
                position.current_pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL
                position.current_pnl = (position.entry_price - current_price) * position.quantity
            
            # Check exit conditions
            should_exit, exit_reason = self._should_exit_position(position, current_price)
            
            if should_exit:
                positions_to_close.append((trade_id, position, current_price, exit_reason))
        
        # Close positions
        for trade_id, position, exit_price, exit_reason in positions_to_close:
            await self._close_position(trade_id, position, exit_price, exit_reason)
    
    def _should_exit_position(self, position: ActivePosition, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited"""
        
        # Time-based exit
        if datetime.now() >= position.target_exit_time:
            return True, 'time_limit'
        
        # Stop loss
        if position.side == 'BUY' and current_price <= position.stop_loss:
            return True, 'stop_loss'
        elif position.side == 'SELL' and current_price >= position.stop_loss:
            return True, 'stop_loss'
        
        # Profit targets
        pnl_pct = position.current_pnl / position.value
        
        for i, target in enumerate(self.profit_targets):
            if position.side == 'BUY' and pnl_pct >= target:
                return True, f'profit_target_{i+1}'
            elif position.side == 'SELL' and pnl_pct >= target:
                return True, f'profit_target_{i+1}'
        
        return False, ''
    
    async def _close_position(self, trade_id: str, position: ActivePosition, 
                            exit_price: float, exit_reason: str):
        """Close active position"""
        
        try:
            # Calculate final P&L
            if position.side == 'BUY':
                final_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                final_pnl = (position.entry_price - exit_price) * position.quantity
            
            final_pnl_pct = final_pnl / position.value
            commission = position.value * 0.002  # 0.1% entry + 0.1% exit
            net_pnl = final_pnl - commission
            
            # Update risk limiter
            self.risk_limiter.release_budget(position.budget_allocated, net_pnl)
            
            # Log trade exit
            hold_duration = (datetime.now() - position.entry_time).total_seconds() / 3600
            
            self.hr_logger.log_trade_exit({
                'trade_id': trade_id,
                'symbol': position.symbol,
                'quantity': position.quantity,
                'price': exit_price,
                'value': position.quantity * exit_price,
                'commission': commission,
                'pnl': net_pnl,
                'pnl_pct': final_pnl_pct,
                'remaining_budget': self.risk_limiter.get_status()['remaining_budget'],
                'entry_signal': position.entry_signal,
                'exit_reason': exit_reason,
                'hold_duration': hold_duration,
                'original_side': position.side,
                'metadata': position.metadata
            })
            
            # Update daily stats
            self.daily_stats['total_pnl'] += net_pnl
            self.daily_stats['best_trade'] = max(self.daily_stats['best_trade'], net_pnl)
            self.daily_stats['worst_trade'] = min(self.daily_stats['worst_trade'], net_pnl)
            
            # Remove from active positions
            del self.active_positions[trade_id]
            
            logger.info(f"🔥 POSITION CLOSED: {position.symbol} P&L: {net_pnl:+.2f}€ ({final_pnl_pct:+.1%}) - {exit_reason}")
            
        except Exception as e:
            logger.error(f"Error closing position {trade_id}: {e}")
            self.hr_logger.log_risk_event('position_close_error', {
                'trade_id': trade_id,
                'error': str(e)
            })
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        risk_status = self.risk_limiter.get_status()
        
        return {
            'name': 'High-Risk Daily Trading Strategy',
            'version': '1.0.0',
            'description': 'Aggressive daily trading with 30€ budget limit',
            'risk_level': 'EXTREME',
            'daily_budget': risk_status['daily_budget'],
            'remaining_budget': risk_status['remaining_budget'],
            'budget_used': risk_status['spent_budget'],
            'is_locked': risk_status['is_locked'],
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'daily_stats': self.daily_stats,
            'profit_targets': self.profit_targets,
            'max_hold_hours': self.max_hold_hours,
            'stop_loss_pct': self.stop_loss_pct,
            'last_update': datetime.now().isoformat()
        }
    
    def get_daily_summary(self) -> str:
        """Get formatted daily summary"""
        info = self.get_strategy_info()
        
        summary = f"""
🔥 HIGH-RISK DAILY STRATEGY SUMMARY
{'='*40}
💰 Budget: {info['budget_used']:.2f}€ / {info['daily_budget']:.2f}€
📊 P&L: {self.daily_stats['total_pnl']:+.2f}€
🎯 Trades: {self.daily_stats['trades_executed']}
📡 Active Positions: {info['active_positions']}
🏆 Best Trade: {self.daily_stats['best_trade']:+.2f}€
📉 Worst Trade: {self.daily_stats['worst_trade']:+.2f}€
🔒 Status: {'LOCKED' if info['is_locked'] else 'ACTIVE'}
        """.strip()
        
        return summary

# Register strategy in the system
def get_high_risk_daily_strategy(params: Dict[str, Any] = None) -> HighRiskDailyStrategy:
    """Factory function for strategy creation"""
    return HighRiskDailyStrategy(params)