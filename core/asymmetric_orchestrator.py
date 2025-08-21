"""
Asymmetric Orchestrator - Integration Layer
==========================================

Integrates the High-Octane Asymmetric Engine with existing orchestrator
and risk management systems for seamless operation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from dataclasses import asdict

from .strategy_orchestrator import StrategyDiscoveryEngine, LearningOrchestrator
from .asymmetric_risk_manager import AsymmetricRiskManager, RiskLevel, AlertSeverity
from strategies.high_octane_asymmetric_engine import HighOctaneAsymmetricEngine
from strategies.strategy_base import Strategy

logger = logging.getLogger(__name__)


class AsymmetricOrchestrator:
    """
    Master orchestrator that combines:
    - Conservative foundation (existing orchestrator)
    - High-octane aggressive strategies
    - Advanced risk management
    - Performance-based rebalancing
    """
    
    def __init__(self, 
                 discovery_engine: StrategyDiscoveryEngine,
                 learning_orchestrator: Optional[LearningOrchestrator] = None,
                 config: Dict[str, Any] = None):
        
        self.discovery_engine = discovery_engine
        self.learning_orchestrator = learning_orchestrator
        self.config = config or {}
        
        # Initialize components
        self.asymmetric_engine = HighOctaneAsymmetricEngine(config.get('engine_params', {}))
        self.risk_manager = AsymmetricRiskManager(config.get('risk_params', {}))
        
        # Conservative strategies (from discovery engine)
        self.conservative_strategies: Dict[str, Strategy] = {}
        
        # Performance tracking
        self.portfolio_value = config.get('initial_capital', 10000.0)
        self.peak_portfolio_value = self.portfolio_value
        self.performance_history = []
        
        # Allocation tracking
        self.current_allocations = {
            'conservative': 0.70,
            'aggressive': 0.30
        }
        
        # Active positions
        self.active_positions: Dict[str, Dict] = {}
        self.position_counter = 0
        
        # Strategy performance
        self.strategy_performance = {
            'conservative_total': {'trades': 0, 'wins': 0, 'total_pnl': 0.0},
            'aggressive_total': {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        }
        
        # Last rebalance
        self.last_rebalance = datetime.now()
        self.rebalance_frequency = timedelta(days=1)  # Daily rebalancing
        
        logger.info("🎯 Asymmetric Orchestrator initialized")
        logger.info(f"   Portfolio: ${self.portfolio_value:,.2f}")
        logger.info(f"   Conservative: {self.current_allocations['conservative']*100:.0f}%")
        logger.info(f"   Aggressive: {self.current_allocations['aggressive']*100:.0f}%")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Discover strategies
            await self.discovery_engine.discover_all_strategies()
            
            # Load conservative strategies
            await self._load_conservative_strategies()
            
            # Initialize learning orchestrator if available
            if self.learning_orchestrator:
                await self.learning_orchestrator.initialize()
            
            logger.info("✅ Asymmetric Orchestrator ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Asymmetric Orchestrator: {e}")
            raise
    
    async def generate_trading_signals(self, market_data: pd.DataFrame, 
                                     symbol: str) -> List[Dict[str, Any]]:
        """
        Generate trading signals from both conservative and aggressive tiers
        
        Args:
            market_data: Market data for analysis
            symbol: Trading symbol
            
        Returns:
            List of validated trading signals
        """
        signals = []
        
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Check for rebalancing
            await self._check_rebalancing()
            
            # 1. Get conservative signal
            conservative_signal = await self._get_conservative_signal(
                market_data, symbol, current_price
            )
            
            if conservative_signal:
                signals.append(conservative_signal)
            
            # 2. Get aggressive signals from asymmetric engine
            aggressive_action, aggressive_data = self.asymmetric_engine.calculate_signal(
                symbol, market_data, current_price
            )
            
            if aggressive_action != 'HOLD':
                aggressive_signal = {
                    'action': aggressive_action,
                    'symbol': symbol,
                    'tier': 'aggressive',
                    'engine': 'asymmetric',
                    **aggressive_data
                }
                signals.append(aggressive_signal)
            
            # 3. Validate and prioritize signals
            validated_signals = await self._validate_and_prioritize_signals(signals)
            
            # 4. Apply risk management
            final_signals = await self._apply_risk_management(validated_signals)
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []
    
    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a validated trading signal
        
        Args:
            signal: Validated trading signal
            
        Returns:
            Trade execution result
        """
        try:
            # Calculate position size with risk management
            position_size = self.risk_manager.calculate_position_size(
                signal, signal['tier'], self.portfolio_value
            )
            
            # Validate trade
            trade_data = {
                **signal,
                'position_size': position_size,
                'portfolio_value': self.portfolio_value,
                'entry_time': datetime.now(),
                'entry_price': signal.get('current_price', 0)
            }
            
            is_valid, issues = self.risk_manager.validate_trade(trade_data)
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {issues}")
                return {
                    'success': False,
                    'reason': 'Risk validation failed',
                    'issues': issues
                }
            
            # Execute trade (simulation)
            position_id = f"pos_{self.position_counter}"
            self.position_counter += 1
            
            # Store position
            self.active_positions[position_id] = {
                **trade_data,
                'position_id': position_id,
                'status': 'open',
                'unrealized_pnl': 0.0
            }
            
            # Update performance tracking
            tier = signal.get('tier', 'conservative')
            self.strategy_performance[f'{tier}_total']['trades'] += 1
            
            logger.info(f"📈 Trade executed: {signal['action']} {signal['symbol']} "
                       f"({tier}, {position_size:.3f} size)")
            
            return {
                'success': True,
                'position_id': position_id,
                'position_size': position_size,
                'tier': tier,
                'entry_price': trade_data['entry_price']
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'reason': f'Execution error: {str(e)}'
            }
    
    async def monitor_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Monitor all active positions and generate exit signals
        
        Args:
            current_prices: Current market prices
            
        Returns:
            List of position updates/exits
        """
        position_updates = []
        
        try:
            for position_id, position in list(self.active_positions.items()):
                if position['status'] != 'open':
                    continue
                
                symbol = position['symbol']
                current_price = current_prices.get(symbol)
                
                if current_price is None:
                    continue
                
                # Update position data
                position['current_price'] = current_price
                
                # Calculate unrealized P&L
                entry_price = position['entry_price']
                action = position['action']
                leverage = position.get('leverage', 1.0)
                position_size = position['position_size']
                
                if action == 'BUY':
                    pnl_percent = ((current_price - entry_price) / entry_price) * leverage
                else:  # SELL
                    pnl_percent = ((entry_price - current_price) / entry_price) * leverage
                
                position['unrealized_pnl_percent'] = pnl_percent
                position['unrealized_pnl'] = pnl_percent * position_size * self.portfolio_value
                
                # Check for exit conditions
                should_exit, exit_reason = await self._check_exit_conditions(position)
                
                if should_exit:
                    # Close position
                    exit_result = await self._close_position(position_id, exit_reason)
                    position_updates.append(exit_result)
                else:
                    # Monitor for risk alerts
                    alerts = self.risk_manager.monitor_position(position_id, position)
                    
                    for alert in alerts:
                        if alert.auto_execute and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                            exit_result = await self._close_position(position_id, alert.message)
                            position_updates.append(exit_result)
                            break
            
            return position_updates
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return []
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            # Calculate current portfolio value
            total_unrealized = sum(pos.get('unrealized_pnl', 0) for pos in self.active_positions.values())
            current_value = self.portfolio_value + total_unrealized
            
            # Update peak value
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
            
            # Calculate daily P&L
            daily_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.active_positions.values()
                           if pos.get('entry_time', datetime.now()).date() == datetime.now().date())
            
            # Get risk assessment
            portfolio_data = {
                'total_value': current_value,
                'peak_value': self.peak_portfolio_value
            }
            portfolio_risk = self.risk_manager.assess_portfolio_risk(portfolio_data)
            
            # Asymmetric engine performance
            engine_performance = self.asymmetric_engine.get_performance_summary()
            
            return {
                'portfolio_value': current_value,
                'daily_pnl': daily_pnl,
                'daily_pnl_percent': (daily_pnl / self.portfolio_value) * 100,
                'total_pnl': current_value - self.portfolio_value,
                'total_pnl_percent': ((current_value - self.portfolio_value) / self.portfolio_value) * 100,
                'peak_value': self.peak_portfolio_value,
                'drawdown': ((self.peak_portfolio_value - current_value) / self.peak_portfolio_value) * 100,
                
                'allocations': self.current_allocations,
                'active_positions': len(self.active_positions),
                'positions_by_tier': {
                    'conservative': len([p for p in self.active_positions.values() if p.get('tier') == 'conservative']),
                    'aggressive': len([p for p in self.active_positions.values() if p.get('tier') == 'aggressive'])
                },
                
                'risk_assessment': {
                    'overall_risk_level': portfolio_risk.risk_level.value,
                    'risk_score': portfolio_risk.overall_risk_score,
                    'value_at_risk_1day': portfolio_risk.value_at_risk_1day,
                    'total_exposure': portfolio_risk.total_exposure,
                    'leverage_weighted_exposure': portfolio_risk.leverage_weighted_exposure
                },
                
                'strategy_performance': self.strategy_performance,
                'engine_performance': engine_performance,
                
                'last_rebalance': self.last_rebalance.isoformat(),
                'next_rebalance': (self.last_rebalance + self.rebalance_frequency).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {
                'portfolio_value': self.portfolio_value,
                'error': str(e)
            }
    
    async def _load_conservative_strategies(self):
        """Load conservative strategies from discovery engine"""
        try:
            # Get conservative strategies
            conservative_dnas = self.discovery_engine.get_strategies_by_criteria(
                risk_level='conservative'
            )
            
            # Add moderate strategies as well
            moderate_dnas = self.discovery_engine.get_strategies_by_criteria(
                risk_level='moderate'
            )
            
            all_conservative = conservative_dnas + moderate_dnas
            
            logger.info(f"Loaded {len(all_conservative)} conservative strategies")
            
            # For now, we'll use a simple strategy selection
            # In production, this would instantiate actual strategy classes
            self.conservative_strategies = {
                dna.name: dna for dna in all_conservative[:5]  # Top 5 strategies
            }
            
        except Exception as e:
            logger.error(f"Error loading conservative strategies: {e}")
    
    async def _get_conservative_signal(self, market_data: pd.DataFrame, 
                                     symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Get signal from conservative strategies"""
        try:
            # Use learning orchestrator if available
            if self.learning_orchestrator:
                # Create mock strategy signals for orchestrator
                strategy_signals = {}
                for strategy_name in self.conservative_strategies.keys():
                    # Simple momentum signal for demonstration
                    sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                    signal_strength = (current_price - sma_20) / sma_20
                    
                    strategy_signals[strategy_name] = {
                        'signal_strength': signal_strength,
                        'confidence': abs(signal_strength),
                        'action': 'BUY' if signal_strength > 0.02 else 'SELL' if signal_strength < -0.02 else 'HOLD'
                    }
                
                # Get orchestrator decision
                market_data_dict = {
                    'regime': 'normal',
                    'volatility': market_data['close'].pct_change().std(),
                    'indicators': {'rsi': 50, 'volume_ratio': 1.0}
                }
                
                portfolio_state = {
                    'total_value': self.portfolio_value,
                    'risk_score': 0.3
                }
                
                allocation_decision = await self.learning_orchestrator.make_allocation_decision(
                    market_data_dict, portfolio_state, strategy_signals
                )
                
                # Convert to signal format
                allocations = allocation_decision.get('allocations', {})
                if allocations:
                    best_strategy = max(allocations.items(), key=lambda x: x[1])
                    if best_strategy[1] > 0.1:  # Minimum 10% allocation
                        return {
                            'action': 'BUY',  # Simplified
                            'symbol': symbol,
                            'tier': 'conservative',
                            'engine': 'orchestrator',
                            'strategy': best_strategy[0],
                            'confidence': allocation_decision.get('confidence', 0.5),
                            'position_size': 0.02,  # 2% position
                            'stop_loss_pct': 0.02,
                            'take_profit_pct': 0.04,
                            'leverage': 1.0
                        }
            
            else:
                # Simple conservative signal
                if len(market_data) >= 50:
                    sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                    sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
                    
                    if current_price > sma_20 > sma_50:
                        return {
                            'action': 'BUY',
                            'symbol': symbol,
                            'tier': 'conservative',
                            'engine': 'simple',
                            'strategy': 'conservative_momentum',
                            'confidence': 0.7,
                            'position_size': 0.02,
                            'stop_loss_pct': 0.02,
                            'take_profit_pct': 0.04,
                            'leverage': 1.0
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting conservative signal: {e}")
            return None
    
    async def _validate_and_prioritize_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and prioritize signals"""
        if not signals:
            return []
        
        # Score signals
        scored_signals = []
        for signal in signals:
            score = signal.get('confidence', 0.5)
            
            # Tier bonus
            if signal.get('tier') == 'conservative':
                score *= 1.1  # Slight preference for conservative
            
            # Engine bonus
            if signal.get('engine') == 'asymmetric':
                score *= 1.2  # Preference for asymmetric engine
            
            scored_signals.append((score, signal))
        
        # Sort by score and return top signals
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return [signal for score, signal in scored_signals[:3]]  # Top 3 signals
    
    async def _apply_risk_management(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply risk management to signals"""
        approved_signals = []
        
        for signal in signals:
            # Risk validation
            is_valid, issues = self.risk_manager.validate_trade(signal)
            
            if is_valid:
                approved_signals.append(signal)
            else:
                logger.debug(f"Signal rejected by risk management: {issues}")
        
        return approved_signals
    
    async def _check_exit_conditions(self, position: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if position should be exited"""
        
        # Take profit
        if position['unrealized_pnl_percent'] >= position.get('take_profit_pct', 0.10):
            return True, "Take profit hit"
        
        # Stop loss
        if position['unrealized_pnl_percent'] <= -position.get('stop_loss_pct', 0.05):
            return True, "Stop loss hit"
        
        # Time limit
        if position.get('time_limit'):
            entry_time = position.get('entry_time', datetime.now())
            time_limit = timedelta(hours=position['time_limit'])
            if datetime.now() - entry_time > time_limit:
                return True, "Time limit reached"
        
        return False, ""
    
    async def _close_position(self, position_id: str, reason: str) -> Dict[str, Any]:
        """Close a position"""
        try:
            position = self.active_positions[position_id]
            
            # Calculate final P&L
            final_pnl = position.get('unrealized_pnl', 0)
            final_pnl_percent = position.get('unrealized_pnl_percent', 0)
            
            # Update portfolio value
            self.portfolio_value += final_pnl
            
            # Update performance tracking
            tier = position.get('tier', 'conservative')
            self.strategy_performance[f'{tier}_total']['total_pnl'] += final_pnl_percent
            
            if final_pnl > 0:
                self.strategy_performance[f'{tier}_total']['wins'] += 1
            
            # Update risk manager performance
            self.risk_manager.update_performance(tier, final_pnl_percent)
            
            # Update asymmetric engine performance
            if tier == 'aggressive':
                trade_result = {
                    'pnl': final_pnl_percent,
                    'risk_tier': tier,
                    'strategy_name': position.get('strategy', 'unknown')
                }
                self.asymmetric_engine.update_performance(trade_result)
            
            # Mark position as closed
            position['status'] = 'closed'
            position['exit_time'] = datetime.now()
            position['exit_reason'] = reason
            position['final_pnl'] = final_pnl
            
            logger.info(f"💰 Position closed: {position_id} - {reason} - "
                       f"P&L: {final_pnl:+.2f} ({final_pnl_percent:+.2%})")
            
            return {
                'position_id': position_id,
                'action': 'closed',
                'reason': reason,
                'final_pnl': final_pnl,
                'final_pnl_percent': final_pnl_percent,
                'tier': tier
            }
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return {
                'position_id': position_id,
                'action': 'error',
                'error': str(e)
            }
    
    async def _check_rebalancing(self):
        """Check if rebalancing is needed"""
        if datetime.now() - self.last_rebalance >= self.rebalance_frequency:
            await self._rebalance_allocations()
    
    async def _rebalance_allocations(self):
        """Rebalance portfolio allocations based on performance"""
        try:
            # Get performance metrics
            conservative_perf = self.strategy_performance['conservative_total']
            aggressive_perf = self.strategy_performance['aggressive_total']
            
            # Calculate performance ratios
            if conservative_perf['trades'] >= 10 and aggressive_perf['trades'] >= 10:
                conservative_avg = conservative_perf['total_pnl'] / conservative_perf['trades']
                aggressive_avg = aggressive_perf['total_pnl'] / aggressive_perf['trades']
                
                if aggressive_avg > conservative_avg * 1.5:  # Aggressive doing much better
                    self.current_allocations['aggressive'] = min(0.40, self.current_allocations['aggressive'] + 0.05)
                elif conservative_avg > aggressive_avg * 1.5:  # Conservative doing better
                    self.current_allocations['aggressive'] = max(0.20, self.current_allocations['aggressive'] - 0.05)
                
                self.current_allocations['conservative'] = 1.0 - self.current_allocations['aggressive']
            
            self.last_rebalance = datetime.now()
            
            logger.info(f"📊 Portfolio rebalanced - Conservative: {self.current_allocations['conservative']*100:.0f}%, "
                       f"Aggressive: {self.current_allocations['aggressive']*100:.0f}%")
            
        except Exception as e:
            logger.error(f"Error rebalancing allocations: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        return self.risk_manager.get_risk_summary()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'strategy_performance': self.strategy_performance,
            'portfolio_metrics': {
                'current_value': self.portfolio_value,
                'peak_value': self.peak_portfolio_value,
                'total_return': ((self.portfolio_value / 10000.0) - 1) * 100,  # Assuming $10k start
                'max_drawdown': ((self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value) * 100
            },
            'allocation_history': self.current_allocations,
            'position_summary': {
                'total_positions': len(self.active_positions),
                'open_positions': len([p for p in self.active_positions.values() if p['status'] == 'open']),
                'closed_positions': len([p for p in self.active_positions.values() if p['status'] == 'closed'])
            }
        }