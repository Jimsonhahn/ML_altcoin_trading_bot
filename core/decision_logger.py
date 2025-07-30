#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Decision Logger
Thread-safe und async Logger für alle Orchestrator-Entscheidungen

Diese Klasse:
- Loggt alle Orchestrator-Entscheidungen in die Datenbank
- Speichert Trade-Performance mit vollem Kontext
- Zeichnet Market States auf
- Ist thread-safe und async-kompatibel
- Implementiert Connection Pooling und Error Handling
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from collections import deque
import hashlib

import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorDecision:
    """Orchestrator decision data structure"""
    decision_type: str  # 'strategy_allocation', 'risk_adjustment', etc.
    strategy_name: Optional[str] = None
    old_allocation: Optional[float] = None
    new_allocation: Optional[float] = None
    market_regime: Optional[str] = None
    volatility_level: Optional[float] = None
    confidence_score: Optional[float] = None
    trigger_source: Optional[str] = None
    trigger_data: Optional[Dict[str, Any]] = None
    decision_reasoning: Optional[str] = None
    expected_impact: Optional[float] = None
    portfolio_value_before: Optional[float] = None
    risk_score_before: Optional[float] = None
    session_id: Optional[str] = None

@dataclass
class TradePerformance:
    """Trade performance data structure"""
    trade_id: str
    strategy_name: str
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    entry_timestamp: datetime
    
    # Optional fields for trade completion
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    trade_status: str = 'open'  # 'open', 'closed', 'cancelled'
    pnl_absolute: Optional[float] = None
    pnl_percentage: Optional[float] = None
    fees_paid: Optional[float] = None
    slippage: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    position_size_usd: Optional[float] = None
    
    # Context
    market_regime_at_entry: Optional[str] = None
    volatility_at_entry: Optional[float] = None
    volume_profile: Optional[Dict[str, Any]] = None
    technical_context: Optional[Dict[str, Any]] = None
    
    # ML data
    ml_confidence: Optional[float] = None
    ml_features: Optional[Dict[str, Any]] = None
    ml_model_version: Optional[str] = None
    
    # Strategy specific
    strategy_parameters: Optional[Dict[str, Any]] = None
    signal_strength: Optional[float] = None
    correlation_with_other_trades: Optional[float] = None
    
    # Portfolio context
    portfolio_heat: Optional[float] = None
    correlation_risk: Optional[float] = None
    portfolio_value_at_entry: Optional[float] = None
    
    # Exit reasoning
    exit_reason: Optional[str] = None
    exit_signal_strength: Optional[float] = None
    trade_quality_score: Optional[float] = None
    lessons_learned: Optional[str] = None
    
    session_id: Optional[str] = None

@dataclass
class MarketState:
    """Market state snapshot data structure"""
    data_source: str
    market_type: str = 'spot'
    
    # Market metrics
    total_market_cap: Optional[float] = None
    btc_dominance: Optional[float] = None
    fear_greedy_index: Optional[int] = None
    
    # Regime
    detected_regime: str = 'neutral'
    regime_confidence: Optional[float] = None
    regime_duration_hours: Optional[int] = None
    previous_regime: Optional[str] = None
    
    # Volatility
    vix_crypto: Optional[float] = None
    realized_volatility_24h: Optional[float] = None
    implied_volatility: Optional[float] = None
    volatility_percentile: Optional[float] = None
    
    # Volume
    total_volume_24h: Optional[float] = None
    volume_ma_ratio: Optional[float] = None
    unusual_volume_detected: bool = False
    volume_spike_threshold: Optional[float] = None
    
    # Price action
    major_support_levels: Optional[List[float]] = None
    major_resistance_levels: Optional[List[float]] = None
    trend_strength: Optional[float] = None
    trend_direction: Optional[str] = None
    
    # Correlations
    btc_correlation: Optional[Dict[str, float]] = None
    traditional_markets_correlation: Optional[Dict[str, float]] = None
    
    # Sentiment
    social_sentiment_score: Optional[float] = None
    news_sentiment_score: Optional[float] = None
    funding_rates: Optional[Dict[str, float]] = None
    
    # Technical indicators
    rsi_composite: Optional[float] = None
    macd_signal: Optional[str] = None
    bollinger_position: Optional[float] = None
    
    # Risk metrics
    systemic_risk_score: Optional[float] = None
    tail_risk_indicator: Optional[float] = None
    leverage_ratio: Optional[float] = None
    
    # ML insights
    anomaly_score: Optional[float] = None
    predicted_next_regime: Optional[str] = None
    regime_change_probability: Optional[float] = None

class DecisionLogger:
    """
    Thread-safe and async-compatible decision logger
    
    Features:
    - Async database operations with connection pooling
    - Thread-safe batch processing
    - Automatic retry logic with exponential backoff
    - Data validation and sanitization
    - Performance monitoring
    - Graceful error handling
    """
    
    def __init__(self, 
                 db_pool: Pool,
                 batch_size: int = 100,
                 flush_interval: int = 30,
                 max_retries: int = 3,
                 session_id: Optional[str] = None):
        """
        Initialize DecisionLogger
        
        Args:
            db_pool: AsyncPG connection pool
            batch_size: Number of records to batch before flushing
            flush_interval: Seconds between automatic flushes
            max_retries: Maximum retry attempts for failed operations
            session_id: Trading session identifier
        """
        self.db_pool = db_pool
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.session_id = session_id or str(uuid.uuid4())
        
        # Thread-safe queues for batching
        self._decision_queue = deque(maxlen=1000)
        self._performance_queue = deque(maxlen=1000)
        self._market_state_queue = deque(maxlen=100)
        
        # Threading
        self._lock = threading.RLock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance tracking
        self._total_logged = 0
        self._total_errors = 0
        self._last_flush_time = datetime.utcnow()
        
        logger.info(f"DecisionLogger initialized with session_id: {self.session_id}")

    async def start(self):
        """Start the logger and background flush task"""
        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())
        logger.info("DecisionLogger started")

    async def stop(self):
        """Stop the logger and flush remaining data"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining data
        await self.flush_all()
        logger.info("DecisionLogger stopped")

    async def log_orchestrator_decision(self, decision: OrchestratorDecision) -> bool:
        """
        Log an orchestrator decision
        
        Args:
            decision: OrchestratorDecision instance
            
        Returns:
            bool: Success status
        """
        try:
            # Validate decision
            if not self._validate_decision(decision):
                logger.error("Invalid decision data")
                return False
            
            # Set session ID if not provided
            if not decision.session_id:
                decision.session_id = self.session_id
            
            # Add to queue
            with self._lock:
                self._decision_queue.append(decision)
                
            # Auto-flush if batch size reached
            if len(self._decision_queue) >= self.batch_size:
                await self._flush_decisions()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log orchestrator decision: {e}")
            self._total_errors += 1
            return False

    async def log_trade_performance(self, trade: TradePerformance) -> bool:
        """
        Log trade performance data
        
        Args:
            trade: TradePerformance instance
            
        Returns:
            bool: Success status
        """
        try:
            # Validate trade data
            if not self._validate_trade(trade):
                logger.error("Invalid trade data")
                return False
            
            # Set session ID if not provided
            if not trade.session_id:
                trade.session_id = self.session_id
            
            # Add to queue
            with self._lock:
                self._performance_queue.append(trade)
                
            # Auto-flush if batch size reached
            if len(self._performance_queue) >= self.batch_size:
                await self._flush_performance()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade performance: {e}")
            self._total_errors += 1
            return False

    async def log_market_state(self, market_state: MarketState) -> bool:
        """
        Log market state snapshot
        
        Args:
            market_state: MarketState instance
            
        Returns:
            bool: Success status
        """
        try:
            # Validate market state
            if not self._validate_market_state(market_state):
                logger.error("Invalid market state data")
                return False
            
            # Add to queue
            with self._lock:
                self._market_state_queue.append(market_state)
                
            # Auto-flush if batch size reached (smaller for market states)
            if len(self._market_state_queue) >= min(self.batch_size // 4, 25):
                await self._flush_market_states()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log market state: {e}")
            self._total_errors += 1
            return False

    async def update_trade_exit(self, trade_id: str, 
                               exit_price: float,
                               exit_timestamp: Optional[datetime] = None,
                               exit_reason: str = 'manual',
                               trade_quality_score: Optional[float] = None,
                               lessons_learned: Optional[str] = None) -> bool:
        """
        Update trade with exit information
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_timestamp: Exit time (defaults to now)
            exit_reason: Reason for exit
            trade_quality_score: Quality score (0-1)
            lessons_learned: Post-trade analysis
            
        Returns:
            bool: Success status
        """
        try:
            if exit_timestamp is None:
                exit_timestamp = datetime.utcnow()
                
            async with self.db_pool.acquire() as conn:
                # Get trade data for PnL calculation
                trade_data = await conn.fetchrow("""
                    SELECT entry_price, quantity, side, fees_paid
                    FROM strategy_performance 
                    WHERE trade_id = $1
                """, trade_id)
                
                if not trade_data:
                    logger.error(f"Trade {trade_id} not found")
                    return False
                
                # Calculate PnL
                entry_price = float(trade_data['entry_price'])
                quantity = float(trade_data['quantity'])
                side = trade_data['side']
                fees_paid = float(trade_data['fees_paid'] or 0)
                
                if side == 'long':
                    pnl_absolute = (exit_price - entry_price) * quantity - fees_paid
                else:  # short
                    pnl_absolute = (entry_price - exit_price) * quantity - fees_paid
                
                pnl_percentage = (pnl_absolute / (entry_price * quantity)) * 100
                
                # Update trade
                await conn.execute("""
                    UPDATE strategy_performance 
                    SET exit_price = $1,
                        exit_timestamp = $2,
                        trade_status = 'closed',
                        pnl_absolute = $3,
                        pnl_percentage = $4,
                        exit_reason = $5,
                        trade_quality_score = $6,
                        lessons_learned = $7
                    WHERE trade_id = $8
                """, exit_price, exit_timestamp, pnl_absolute, pnl_percentage,
                    exit_reason, trade_quality_score, lessons_learned, trade_id)
                
                logger.info(f"Trade {trade_id} updated with exit data - PnL: {pnl_percentage:.2f}%")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update trade exit: {e}")
            return False

    async def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent orchestrator decisions"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM orchestrator_decisions 
                    WHERE session_id = $1 
                    ORDER BY timestamp DESC 
                    LIMIT $2
                """, self.session_id, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent decisions: {e}")
            return []

    async def get_strategy_performance_summary(self, strategy_name: str,
                                             days: int = 30) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        try:
            async with self.db_pool.acquire() as conn:
                since_date = datetime.utcnow() - timedelta(days=days)
                
                row = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE trade_status = 'closed') as closed_trades,
                        AVG(pnl_percentage) FILTER (WHERE trade_status = 'closed') as avg_return,
                        STDDEV(pnl_percentage) FILTER (WHERE trade_status = 'closed') as return_volatility,
                        COUNT(*) FILTER (WHERE pnl_percentage > 0 AND trade_status = 'closed') as winning_trades,
                        SUM(pnl_absolute) FILTER (WHERE trade_status = 'closed') as total_pnl,
                        AVG(duration_minutes) FILTER (WHERE trade_status = 'closed') as avg_duration,
                        MAX(pnl_percentage) FILTER (WHERE trade_status = 'closed') as best_trade,
                        MIN(pnl_percentage) FILTER (WHERE trade_status = 'closed') as worst_trade
                    FROM strategy_performance 
                    WHERE strategy_name = $1 
                        AND timestamp >= $2
                        AND session_id = $3
                """, strategy_name, since_date, self.session_id)
                
                if row:
                    result = dict(row)
                    # Calculate additional metrics
                    if result['closed_trades'] and result['closed_trades'] > 0:
                        result['win_rate'] = (result['winning_trades'] / result['closed_trades']) * 100
                        if result['return_volatility'] and result['return_volatility'] > 0:
                            result['sharpe_ratio'] = result['avg_return'] / result['return_volatility']
                    
                    return result
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get strategy performance summary: {e}")
            return {}

    async def flush_all(self):
        """Flush all queued data to database"""
        try:
            await self._flush_decisions()
            await self._flush_performance()
            await self._flush_market_states()
            logger.info("All queues flushed successfully")
        except Exception as e:
            logger.error(f"Failed to flush all queues: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        with self._lock:
            return {
                'session_id': self.session_id,
                'total_logged': self._total_logged,
                'total_errors': self._total_errors,
                'queue_sizes': {
                    'decisions': len(self._decision_queue),
                    'performance': len(self._performance_queue),
                    'market_states': len(self._market_state_queue)
                },
                'last_flush_time': self._last_flush_time.isoformat(),
                'running': self._running
            }

    # Private methods
    async def _background_flush(self):
        """Background task to periodically flush queues"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background flush error: {e}")

    async def _flush_decisions(self):
        """Flush orchestrator decisions to database"""
        if not self._decision_queue:
            return
            
        with self._lock:
            decisions = list(self._decision_queue)
            self._decision_queue.clear()
        
        for attempt in range(self.max_retries):
            try:
                async with self.db_pool.acquire() as conn:
                    for decision in decisions:
                        await conn.execute("""
                            INSERT INTO orchestrator_decisions (
                                session_id, decision_type, strategy_name, old_allocation, 
                                new_allocation, market_regime, volatility_level, confidence_score,
                                trigger_source, trigger_data, decision_reasoning, expected_impact,
                                portfolio_value_before, risk_score_before
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        """, decision.session_id, decision.decision_type, decision.strategy_name,
                            decision.old_allocation, decision.new_allocation, decision.market_regime,
                            decision.volatility_level, decision.confidence_score, decision.trigger_source,
                            json.dumps(decision.trigger_data) if decision.trigger_data else None,
                            decision.decision_reasoning, decision.expected_impact,
                            decision.portfolio_value_before, decision.risk_score_before)
                
                self._total_logged += len(decisions)
                self._last_flush_time = datetime.utcnow()
                logger.debug(f"Flushed {len(decisions)} decisions to database")
                return
                
            except Exception as e:
                logger.error(f"Failed to flush decisions (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # Re-queue failed items
                    with self._lock:
                        self._decision_queue.extendleft(reversed(decisions))
                    self._total_errors += len(decisions)
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _flush_performance(self):
        """Flush trade performance to database"""
        if not self._performance_queue:
            return
            
        with self._lock:
            trades = list(self._performance_queue)
            self._performance_queue.clear()
        
        for attempt in range(self.max_retries):
            try:
                async with self.db_pool.acquire() as conn:
                    for trade in trades:
                        await conn.execute("""
                            INSERT INTO strategy_performance (
                                trade_id, strategy_name, symbol, exchange, side, entry_price,
                                quantity, entry_timestamp, exit_price, exit_timestamp, trade_status,
                                pnl_absolute, pnl_percentage, fees_paid, slippage, stop_loss,
                                take_profit, max_drawdown, risk_reward_ratio, position_size_usd,
                                market_regime_at_entry, volatility_at_entry, volume_profile,
                                technical_context, ml_confidence, ml_features, ml_model_version,
                                strategy_parameters, signal_strength, correlation_with_other_trades,
                                portfolio_heat, correlation_risk, portfolio_value_at_entry,
                                exit_reason, exit_signal_strength, trade_quality_score,
                                lessons_learned, session_id
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                                $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
                                $29, $30, $31, $32, $33, $34, $35, $36, $37, $38
                            )
                        """, trade.trade_id, trade.strategy_name, trade.symbol, trade.exchange,
                            trade.side, trade.entry_price, trade.quantity, trade.entry_timestamp,
                            trade.exit_price, trade.exit_timestamp, trade.trade_status,
                            trade.pnl_absolute, trade.pnl_percentage, trade.fees_paid, trade.slippage,
                            trade.stop_loss, trade.take_profit, trade.max_drawdown, trade.risk_reward_ratio,
                            trade.position_size_usd, trade.market_regime_at_entry, trade.volatility_at_entry,
                            json.dumps(trade.volume_profile) if trade.volume_profile else None,
                            json.dumps(trade.technical_context) if trade.technical_context else None,
                            trade.ml_confidence, json.dumps(trade.ml_features) if trade.ml_features else None,
                            trade.ml_model_version, json.dumps(trade.strategy_parameters) if trade.strategy_parameters else None,
                            trade.signal_strength, trade.correlation_with_other_trades, trade.portfolio_heat,
                            trade.correlation_risk, trade.portfolio_value_at_entry, trade.exit_reason,
                            trade.exit_signal_strength, trade.trade_quality_score, trade.lessons_learned,
                            trade.session_id)
                
                self._total_logged += len(trades)
                self._last_flush_time = datetime.utcnow()
                logger.debug(f"Flushed {len(trades)} trades to database")
                return
                
            except Exception as e:
                logger.error(f"Failed to flush performance (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # Re-queue failed items
                    with self._lock:
                        self._performance_queue.extendleft(reversed(trades))
                    self._total_errors += len(trades)
                else:
                    await asyncio.sleep(2 ** attempt)

    async def _flush_market_states(self):
        """Flush market states to database"""
        if not self._market_state_queue:
            return
            
        with self._lock:
            states = list(self._market_state_queue)
            self._market_state_queue.clear()
        
        for attempt in range(self.max_retries):
            try:
                async with self.db_pool.acquire() as conn:
                    for state in states:
                        await conn.execute("""
                            INSERT INTO market_states (
                                data_source, market_type, total_market_cap, btc_dominance,
                                fear_greedy_index, detected_regime, regime_confidence,
                                regime_duration_hours, previous_regime, vix_crypto,
                                realized_volatility_24h, implied_volatility, volatility_percentile,
                                total_volume_24h, volume_ma_ratio, unusual_volume_detected,
                                volume_spike_threshold, major_support_levels, major_resistance_levels,
                                trend_strength, trend_direction, btc_correlation,
                                traditional_markets_correlation, social_sentiment_score,
                                news_sentiment_score, funding_rates, rsi_composite, macd_signal,
                                bollinger_position, systemic_risk_score, tail_risk_indicator,
                                leverage_ratio, anomaly_score, predicted_next_regime,
                                regime_change_probability
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                                $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
                                $29, $30, $31, $32, $33, $34, $35
                            )
                        """, state.data_source, state.market_type, state.total_market_cap,
                            state.btc_dominance, state.fear_greedy_index, state.detected_regime,
                            state.regime_confidence, state.regime_duration_hours, state.previous_regime,
                            state.vix_crypto, state.realized_volatility_24h, state.implied_volatility,
                            state.volatility_percentile, state.total_volume_24h, state.volume_ma_ratio,
                            state.unusual_volume_detected, state.volume_spike_threshold,
                            state.major_support_levels, state.major_resistance_levels,
                            state.trend_strength, state.trend_direction,
                            json.dumps(state.btc_correlation) if state.btc_correlation else None,
                            json.dumps(state.traditional_markets_correlation) if state.traditional_markets_correlation else None,
                            state.social_sentiment_score, state.news_sentiment_score,
                            json.dumps(state.funding_rates) if state.funding_rates else None,
                            state.rsi_composite, state.macd_signal, state.bollinger_position,
                            state.systemic_risk_score, state.tail_risk_indicator, state.leverage_ratio,
                            state.anomaly_score, state.predicted_next_regime, state.regime_change_probability)
                
                self._total_logged += len(states)
                self._last_flush_time = datetime.utcnow()
                logger.debug(f"Flushed {len(states)} market states to database")
                return
                
            except Exception as e:
                logger.error(f"Failed to flush market states (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # Re-queue failed items
                    with self._lock:
                        self._market_state_queue.extendleft(reversed(states))
                    self._total_errors += len(states)
                else:
                    await asyncio.sleep(2 ** attempt)

    def _validate_decision(self, decision: OrchestratorDecision) -> bool:
        """Validate orchestrator decision data"""
        if not decision.decision_type:
            return False
        if decision.confidence_score is not None and not (0 <= decision.confidence_score <= 1):
            return False
        if decision.old_allocation is not None and not (0 <= decision.old_allocation <= 1):
            return False
        if decision.new_allocation is not None and not (0 <= decision.new_allocation <= 1):
            return False
        return True

    def _validate_trade(self, trade: TradePerformance) -> bool:
        """Validate trade performance data"""
        if not trade.trade_id or not trade.strategy_name or not trade.symbol:
            return False
        if trade.entry_price <= 0 or trade.quantity <= 0:
            return False
        if trade.side not in ['long', 'short']:
            return False
        if trade.exit_price is not None and trade.exit_price <= 0:
            return False
        return True

    def _validate_market_state(self, state: MarketState) -> bool:
        """Validate market state data"""
        if not state.data_source or not state.detected_regime:
            return False
        if state.regime_confidence is not None and not (0 <= state.regime_confidence <= 1):
            return False
        if state.fear_greedy_index is not None and not (0 <= state.fear_greedy_index <= 100):
            return False
        return True

# Context manager for easy usage
@asynccontextmanager
async def decision_logger_context(db_pool: Pool, **kwargs):
    """Context manager for DecisionLogger"""
    logger_instance = DecisionLogger(db_pool, **kwargs)
    try:
        await logger_instance.start()
        yield logger_instance
    finally:
        await logger_instance.stop()

# Example usage
async def example_usage():
    """Example of how to use DecisionLogger"""
    # This would typically be initialized with your actual database pool
    # db_pool = await asyncpg.create_pool(...)
    
    # async with decision_logger_context(db_pool) as decision_logger:
    #     # Log a decision
    #     decision = OrchestratorDecision(
    #         decision_type='strategy_allocation',
    #         strategy_name='momentum_strategy',
    #         old_allocation=0.25,
    #         new_allocation=0.30,
    #         market_regime='bull',
    #         confidence_score=0.85,
    #         decision_reasoning='Increased allocation due to strong momentum signals'
    #     )
    #     await decision_logger.log_orchestrator_decision(decision)
    #     
    #     # Log a trade
    #     trade = TradePerformance(
    #         trade_id='trade_123',
    #         strategy_name='momentum_strategy',
    #         symbol='BTC/USDT',
    #         exchange='binance',
    #         side='long',
    #         entry_price=45000.0,
    #         quantity=0.1,
    #         entry_timestamp=datetime.utcnow()
    #     )
    #     await decision_logger.log_trade_performance(trade)
    #     
    #     # Log market state
    #     market_state = MarketState(
    #         data_source='binance',
    #         detected_regime='bull',
    #         regime_confidence=0.8,
    #         total_market_cap=2000000000000.0,
    #         btc_dominance=0.45
    #     )
    #     await decision_logger.log_market_state(market_state)
    
    print("Example usage completed (commented out - requires actual DB pool)")

if __name__ == "__main__":
    asyncio.run(example_usage())