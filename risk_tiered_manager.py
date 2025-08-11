#!/usr/bin/env python3
"""
🎯 Risk-Tiered Strategy Supermix Manager
Parallel execution of ALL strategies with intelligent risk allocation
"""

import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path

# Import your existing strategy base
from strategies.strategy_base import Strategy as StrategyBase
# Mock imports for demo mode
try:
    from strategies import *
    from core.safety_manager import SafetyManager
    from core.interfaces import TradingBot, DatabasePool
    from utils.exceptions import TradingError
except ImportError:
    # Demo mode fallbacks
    class SafetyManager:
        async def can_open_position(self):
            return True
    
    class TradingBot:
        pass
    
    class DatabasePool:
        pass
    
    class TradingError(Exception):
        pass

@dataclass
class RiskCategory:
    """Risk category configuration"""
    name: str
    allocation_percent: float
    max_position_size_percent: float
    max_trades_concurrent: int
    target_timeframe_hours: tuple
    expected_roi_percent: tuple
    max_loss_per_trade_percent: float
    rebalance_frequency_hours: int

@dataclass
class StrategyAllocation:
    """Individual strategy allocation within risk category"""
    strategy_name: str
    strategy_class: type
    risk_category: str
    allocation_percent: float
    current_positions: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    is_active: bool = True

class RiskTieredStrategyManager:
    """
    🚀 Multi-Strategy Risk-Tiered Execution Manager
    
    Executes ALL available strategies in parallel with intelligent risk allocation:
    - HIGH_RISK: 15% portfolio, frequent small trades
    - MEDIUM_RISK: 35% portfolio, moderate swing trades  
    - LOW_RISK: 50% portfolio, large conservative trades
    """
    
    def __init__(self, 
                 trading_bot: TradingBot,
                 db_pool: DatabasePool,
                 portfolio_value: Decimal,
                 config_path: str = "config/risk_tiered_config.json"):
        
        self.trading_bot = trading_bot
        self.db_pool = db_pool
        self.portfolio_value = portfolio_value
        self.config_path = config_path
        
        # Initialize components
        self.safety_manager = SafetyManager()
        self.logger = logging.getLogger(__name__)
        
        # Risk categories with dynamic allocation
        self.risk_categories = self._initialize_risk_categories()
        
        # Strategy discovery and allocation
        self.strategy_allocations = self._discover_and_allocate_strategies()
        
        # Execution state
        self.execution_threads = {}
        self.is_running = False
        self.performance_tracker = {}
        
        # Load configuration
        self._load_configuration()
        
        self.logger.info("🎯 Risk-Tiered Strategy Manager initialized")
        self.logger.info(f"💰 Portfolio Value: ${self.portfolio_value:,.2f}")
        self.logger.info(f"📊 Strategies Discovered: {len(self.strategy_allocations)}")
    
    def _initialize_risk_categories(self) -> Dict[str, RiskCategory]:
        """Initialize risk categories with optimal configurations"""
        return {
            'LOW_RISK': RiskCategory(
                name='LOW_RISK_STEADY_GROWTH',
                allocation_percent=50.0,  # 50% of portfolio
                max_position_size_percent=8.0,  # Max 8% per position
                max_trades_concurrent=3,
                target_timeframe_hours=(24, 720),  # 1-30 days
                expected_roi_percent=(0.5, 3.0),
                max_loss_per_trade_percent=2.0,
                rebalance_frequency_hours=24
            ),
            
            'MEDIUM_RISK': RiskCategory(
                name='MEDIUM_RISK_BALANCED_GROWTH',
                allocation_percent=35.0,  # 35% of portfolio
                max_position_size_percent=5.0,  # Max 5% per position
                max_trades_concurrent=5,
                target_timeframe_hours=(1, 72),  # 1 hour - 3 days
                expected_roi_percent=(1.0, 8.0),
                max_loss_per_trade_percent=3.0,
                rebalance_frequency_hours=8
            ),
            
            'HIGH_RISK': RiskCategory(
                name='HIGH_RISK_AGGRESSIVE_GROWTH',
                allocation_percent=15.0,  # 15% of portfolio
                max_position_size_percent=2.0,  # Max 2% per position
                max_trades_concurrent=10,
                target_timeframe_hours=(0.02, 4),  # 1 min - 4 hours
                expected_roi_percent=(2.0, 15.0),
                max_loss_per_trade_percent=1.5,
                rebalance_frequency_hours=2
            )
        }
    
    def _discover_and_allocate_strategies(self) -> List[StrategyAllocation]:
        """
        Auto-discover all available strategies and allocate to risk categories
        """
        strategy_allocations = []
        
        # Define strategy risk mappings based on analysis
        strategy_risk_mapping = {
            # LOW RISK STRATEGIES (50% allocation)
            'DefensiveVolatilityStrategy': ('LOW_RISK', 20.0),
            'SmartRebalancingStrategy': ('LOW_RISK', 15.0),
            'MeanReversionStrategy': ('LOW_RISK', 10.0),
            'ArbitrageStrategy': ('LOW_RISK', 5.0),
            
            # MEDIUM RISK STRATEGIES (35% allocation)
            'MomentumStrategy': ('MEDIUM_RISK', 12.0),
            'CandleBodyMomentumStrategy': ('MEDIUM_RISK', 10.0),
            'GridTradingStrategy': ('MEDIUM_RISK', 8.0),
            'TrendFollowingStrategy': ('MEDIUM_RISK', 5.0),
            
            # HIGH RISK STRATEGIES (15% allocation)  
            'MLStrategy': ('HIGH_RISK', 5.0),
            'EnhancedHighRiskStrategy': ('HIGH_RISK', 4.0),
            'HighRiskDailyStrategy': ('HIGH_RISK', 3.0),
            'LiquidationStrategy': ('HIGH_RISK', 2.0),
            'UltimateBTCStrategy': ('HIGH_RISK', 1.0)
        }
        
        # Import and validate all strategies
        for strategy_name, (risk_category, allocation) in strategy_risk_mapping.items():
            try:
                # Dynamic import from strategies module
                strategy_class = getattr(__import__('strategies', fromlist=[strategy_name]), strategy_name, None)
                
                if strategy_class and issubclass(strategy_class, StrategyBase):
                    strategy_allocation = StrategyAllocation(
                        strategy_name=strategy_name,
                        strategy_class=strategy_class,
                        risk_category=risk_category,
                        allocation_percent=allocation,
                        performance_metrics={
                            'total_trades': 0,
                            'winning_trades': 0,
                            'total_pnl': Decimal('0'),
                            'sharpe_ratio': 0.0,
                            'max_drawdown': 0.0
                        }
                    )
                    strategy_allocations.append(strategy_allocation)
                    self.logger.info(f"✅ {strategy_name} allocated to {risk_category}: {allocation}%")
                else:
                    self.logger.warning(f"⚠️ Strategy {strategy_name} not found or invalid")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load strategy {strategy_name}: {e}")
        
        return strategy_allocations
    
    def calculate_position_size(self, 
                              strategy_allocation: StrategyAllocation,
                              market_volatility: float,
                              portfolio_health: float) -> Decimal:
        """
        🎯 Dynamic position sizing based on risk category and current conditions
        
        Args:
            strategy_allocation: Strategy allocation configuration
            market_volatility: Current market volatility (0-1)
            portfolio_health: Portfolio health score (0-1)
            
        Returns:
            Calculated position size in base currency
        """
        risk_category = self.risk_categories[strategy_allocation.risk_category]
        
        # Base allocation from portfolio
        base_allocation = self.portfolio_value * Decimal(str(strategy_allocation.allocation_percent / 100))
        
        # Risk category maximum position size
        max_position = self.portfolio_value * Decimal(str(risk_category.max_position_size_percent / 100))
        
        # Volatility adjustment (reduce size in high volatility)
        volatility_multiplier = Decimal(str(max(0.2, 1.0 - market_volatility)))
        
        # Portfolio health adjustment (reduce size if portfolio unhealthy)
        health_multiplier = Decimal(str(max(0.1, portfolio_health)))
        
        # Strategy performance adjustment
        performance_metrics = strategy_allocation.performance_metrics
        win_rate = (performance_metrics.get('winning_trades', 0) / 
                   max(1, performance_metrics.get('total_trades', 1)))
        
        performance_multiplier = Decimal(str(min(2.0, max(0.2, win_rate * 2))))
        
        # Calculate final position size
        calculated_size = min(
            base_allocation * volatility_multiplier * health_multiplier * performance_multiplier,
            max_position
        )
        
        self.logger.debug(f"📏 Position size for {strategy_allocation.strategy_name}: "
                         f"${calculated_size:,.2f} (base: ${base_allocation:,.2f})")
        
        return calculated_size
    
    async def execute_parallel_strategies(self):
        """
        🚀 Execute all strategies in parallel with risk-based allocation
        """
        self.logger.info("🚀 Starting parallel strategy execution...")
        self.is_running = True
        
        # Create execution tasks for each strategy
        strategy_tasks = []
        
        for strategy_allocation in self.strategy_allocations:
            if not strategy_allocation.is_active:
                continue
                
            task = asyncio.create_task(
                self._execute_strategy_loop(strategy_allocation)
            )
            strategy_tasks.append(task)
            
        # Start portfolio optimizer
        optimizer_task = asyncio.create_task(self._portfolio_optimization_loop())
        
        # Start performance tracker
        tracker_task = asyncio.create_task(self._performance_tracking_loop())
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(
                *strategy_tasks,
                optimizer_task,
                tracker_task,
                return_exceptions=True
            )
        except Exception as e:
            self.logger.error(f"❌ Error in parallel strategy execution: {e}")
        finally:
            self.is_running = False
    
    async def _execute_strategy_loop(self, strategy_allocation: StrategyAllocation):
        """Execute individual strategy in continuous loop"""
        strategy_name = strategy_allocation.strategy_name
        risk_category = self.risk_categories[strategy_allocation.risk_category]
        
        self.logger.info(f"🎯 Starting {strategy_name} execution loop ({strategy_allocation.risk_category})")
        
        # Initialize strategy instance
        try:
            strategy_instance = strategy_allocation.strategy_class(
                db_pool=self.db_pool,
                exchange_manager=self.trading_bot.exchange_manager,
                config=self.trading_bot.config
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {strategy_name}: {e}")
            return
        
        while self.is_running:
            try:
                # Check if we can open new positions
                current_positions = len(strategy_allocation.current_positions)
                if current_positions >= risk_category.max_trades_concurrent:
                    await asyncio.sleep(30)  # Wait before checking again
                    continue
                
                # Calculate position size
                market_volatility = await self._get_market_volatility()
                portfolio_health = await self._get_portfolio_health()
                
                position_size = self.calculate_position_size(
                    strategy_allocation,
                    market_volatility, 
                    portfolio_health
                )
                
                # Execute strategy signal
                signal = await strategy_instance.generate_signal(
                    symbol='BTCUSDT',  # Primary symbol, can be expanded
                    position_size=position_size
                )
                
                if signal and signal.action in ['BUY', 'SELL']:
                    await self._execute_strategy_signal(strategy_allocation, signal)
                
                # Sleep based on strategy timeframe
                sleep_time = self._calculate_sleep_time(risk_category)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"❌ Error in {strategy_name} execution: {e}")
                await asyncio.sleep(60)  # Error recovery sleep
    
    async def _execute_strategy_signal(self, strategy_allocation: StrategyAllocation, signal):
        """Execute trading signal with risk management"""
        strategy_name = strategy_allocation.strategy_name
        
        try:
            # Risk checks
            if not await self.safety_manager.can_open_position():
                self.logger.warning(f"🛑 Safety manager blocked {strategy_name} trade")
                return
            
            # Execute trade through trading bot
            trade_result = await self.trading_bot.execute_trade(signal)
            
            if trade_result.success:
                # Track position
                position = {
                    'strategy': strategy_name,
                    'signal': signal,
                    'trade_result': trade_result,
                    'opened_at': datetime.now(),
                    'position_size': signal.position_size
                }
                
                strategy_allocation.current_positions.append(position)
                
                # Update performance metrics
                await self._update_performance_metrics(strategy_allocation, trade_result)
                
                self.logger.info(f"✅ {strategy_name} trade executed: {signal.action} {signal.symbol}")
            else:
                self.logger.warning(f"❌ {strategy_name} trade failed: {trade_result.error}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute signal for {strategy_name}: {e}")
    
    async def _portfolio_optimization_loop(self):
        """Continuous portfolio optimization and rebalancing"""
        self.logger.info("📊 Starting portfolio optimization loop...")
        
        while self.is_running:
            try:
                await self._rebalance_strategies()
                await self._adjust_risk_allocations()
                await self._compound_growth_accelerator()
                
                await asyncio.sleep(3600)  # Rebalance every hour
                
            except Exception as e:
                self.logger.error(f"❌ Error in portfolio optimization: {e}")
                await asyncio.sleep(300)
    
    async def _rebalance_strategies(self):
        """Rebalance strategy allocations based on performance"""
        self.logger.info("⚖️ Rebalancing strategy allocations...")
        
        for category_name, strategies in self._group_strategies_by_category().items():
            # Calculate performance scores
            performance_scores = {}
            for strategy_allocation in strategies:
                metrics = strategy_allocation.performance_metrics
                
                # Calculate composite performance score
                win_rate = (metrics.get('winning_trades', 0) / 
                           max(1, metrics.get('total_trades', 1)))
                pnl_ratio = float(metrics.get('total_pnl', 0)) / max(1, float(self.portfolio_value))
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                
                performance_score = (win_rate * 0.4 + pnl_ratio * 0.4 + sharpe_ratio * 0.2)
                performance_scores[strategy_allocation.strategy_name] = performance_score
            
            # Redistribute allocations based on performance
            total_category_allocation = self.risk_categories[category_name].allocation_percent
            total_performance = sum(performance_scores.values()) or 1
            
            for strategy_allocation in strategies:
                strategy_performance = performance_scores[strategy_allocation.strategy_name]
                new_allocation = (strategy_performance / total_performance) * total_category_allocation
                
                # Smooth allocation changes (max 20% change per rebalance)
                old_allocation = strategy_allocation.allocation_percent
                max_change = abs(old_allocation * 0.2)
                allocation_change = max(-max_change, min(max_change, new_allocation - old_allocation))
                
                strategy_allocation.allocation_percent = old_allocation + allocation_change
                
                self.logger.info(f"📊 {strategy_allocation.strategy_name} allocation: "
                               f"{old_allocation:.1f}% → {strategy_allocation.allocation_percent:.1f}%")
    
    async def _compound_growth_accelerator(self):
        """Reinvest profits based on risk category performance"""
        self.logger.info("🚀 Applying compound growth acceleration...")
        
        total_profits = Decimal('0')
        
        for strategy_allocation in self.strategy_allocations:
            strategy_pnl = strategy_allocation.performance_metrics.get('total_pnl', Decimal('0'))
            if strategy_pnl > 0:
                total_profits += strategy_pnl
        
        if total_profits > 0:
            # Reinvestment rates by risk category
            reinvestment_rates = {
                'HIGH_RISK': 0.3,    # 30% of profits back to high risk
                'MEDIUM_RISK': 0.5,  # 50% of profits back to medium risk
                'LOW_RISK': 0.2      # 20% of profits back to low risk
            }
            
            for category_name, rate in reinvestment_rates.items():
                reinvestment_amount = total_profits * Decimal(str(rate))
                
                # Distribute among strategies in this category
                category_strategies = [s for s in self.strategy_allocations 
                                     if s.risk_category == category_name]
                
                if category_strategies:
                    per_strategy_amount = reinvestment_amount / len(category_strategies)
                    
                    for strategy_allocation in category_strategies:
                        # Convert to allocation percentage increase
                        allocation_increase = float(per_strategy_amount / self.portfolio_value * 100)
                        strategy_allocation.allocation_percent += allocation_increase
            
            self.logger.info(f"💰 Compound growth applied: ${total_profits:,.2f} profits reinvested")
    
    def _group_strategies_by_category(self) -> Dict[str, List[StrategyAllocation]]:
        """Group strategies by risk category"""
        categories = {}
        for strategy_allocation in self.strategy_allocations:
            category = strategy_allocation.risk_category
            if category not in categories:
                categories[category] = []
            categories[category].append(strategy_allocation)
        return categories
    
    def _calculate_sleep_time(self, risk_category: RiskCategory) -> int:
        """Calculate sleep time based on risk category timeframe"""
        min_hours, max_hours = risk_category.target_timeframe_hours
        avg_hours = (min_hours + max_hours) / 2
        
        # Convert to seconds, minimum 30 seconds
        sleep_seconds = max(30, int(avg_hours * 3600 / 10))  # Check 10 times per average timeframe
        
        return sleep_seconds
    
    async def _get_market_volatility(self) -> float:
        """Get current market volatility score (0-1)"""
        # Implement volatility calculation
        # For now, return moderate volatility
        return 0.3
    
    async def _get_portfolio_health(self) -> float:
        """Get portfolio health score (0-1)"""
        # Implement portfolio health calculation
        # For now, return good health
        return 0.8
    
    async def _update_performance_metrics(self, strategy_allocation: StrategyAllocation, trade_result):
        """Update strategy performance metrics"""
        metrics = strategy_allocation.performance_metrics
        
        metrics['total_trades'] += 1
        
        if trade_result.pnl > 0:
            metrics['winning_trades'] += 1
        
        metrics['total_pnl'] += trade_result.pnl
        
        # Calculate new Sharpe ratio, max drawdown, etc.
        # Simplified implementation
        win_rate = metrics['winning_trades'] / metrics['total_trades']
        metrics['sharpe_ratio'] = win_rate * 2.0  # Simplified Sharpe
    
    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.is_running:
            try:
                await self._log_performance_summary()
                await asyncio.sleep(1800)  # Log every 30 minutes
            except Exception as e:
                self.logger.error(f"❌ Error in performance tracking: {e}")
                await asyncio.sleep(60)
    
    async def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        self.logger.info("📊 PERFORMANCE SUMMARY")
        self.logger.info("=" * 50)
        
        total_pnl = Decimal('0')
        total_trades = 0
        
        for category_name, strategies in self._group_strategies_by_category().items():
            category_pnl = Decimal('0')
            category_trades = 0
            
            self.logger.info(f"\n🎯 {category_name}:")
            
            for strategy_allocation in strategies:
                metrics = strategy_allocation.performance_metrics
                strategy_pnl = metrics.get('total_pnl', Decimal('0'))
                strategy_trades = metrics.get('total_trades', 0)
                
                category_pnl += strategy_pnl
                category_trades += strategy_trades
                
                win_rate = (metrics.get('winning_trades', 0) / max(1, strategy_trades)) * 100
                
                self.logger.info(f"  📈 {strategy_allocation.strategy_name}:")
                self.logger.info(f"    Trades: {strategy_trades}, Win Rate: {win_rate:.1f}%, P&L: ${strategy_pnl:,.2f}")
            
            total_pnl += category_pnl
            total_trades += category_trades
            
            self.logger.info(f"  💰 Category Total: ${category_pnl:,.2f} ({category_trades} trades)")
        
        portfolio_return = float(total_pnl / self.portfolio_value * 100)
        
        self.logger.info(f"\n🚀 TOTAL PORTFOLIO:")
        self.logger.info(f"   Total P&L: ${total_pnl:,.2f}")
        self.logger.info(f"   Total Trades: {total_trades}")
        self.logger.info(f"   Portfolio Return: {portfolio_return:.2f}%")
        self.logger.info("=" * 50)
    
    def _load_configuration(self):
        """Load risk-tiered configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                # Override default configurations with loaded values
                if 'risk_categories' in config:
                    for category_name, category_config in config['risk_categories'].items():
                        if category_name in self.risk_categories:
                            category = self.risk_categories[category_name]
                            for key, value in category_config.items():
                                if hasattr(category, key):
                                    setattr(category, key, value)
                
                self.logger.info(f"✅ Configuration loaded from {self.config_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load configuration: {e}, using defaults")
    
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                'risk_categories': {},
                'strategy_allocations': {}
            }
            
            for name, category in self.risk_categories.items():
                config['risk_categories'][name] = {
                    'allocation_percent': category.allocation_percent,
                    'max_position_size_percent': category.max_position_size_percent,
                    'max_trades_concurrent': category.max_trades_concurrent,
                    'max_loss_per_trade_percent': category.max_loss_per_trade_percent,
                    'rebalance_frequency_hours': category.rebalance_frequency_hours
                }
            
            for strategy_allocation in self.strategy_allocations:
                config['strategy_allocations'][strategy_allocation.strategy_name] = {
                    'allocation_percent': strategy_allocation.allocation_percent,
                    'is_active': strategy_allocation.is_active
                }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"✅ Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
    
    async def stop(self):
        """Stop all strategy execution"""
        self.logger.info("🛑 Stopping Risk-Tiered Strategy Manager...")
        self.is_running = False
        
        # Save final configuration
        self.save_configuration()
        
        self.logger.info("✅ Risk-Tiered Strategy Manager stopped")

# Example usage integration
async def main():
    """Example usage of Risk-Tiered Strategy Manager"""
    # This would be integrated into your main.py
    
    from core.trading_bot import TradingBot
    from core.database import create_db_pool
    
    # Initialize your existing components
    db_pool = await create_db_pool()
    trading_bot = TradingBot(db_pool=db_pool)
    
    # Initialize Risk-Tiered Manager
    portfolio_value = Decimal('100000')  # $100k portfolio
    manager = RiskTieredStrategyManager(
        trading_bot=trading_bot,
        db_pool=db_pool, 
        portfolio_value=portfolio_value
    )
    
    try:
        # Start parallel strategy execution
        await manager.execute_parallel_strategies()
    except KeyboardInterrupt:
        await manager.stop()
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())