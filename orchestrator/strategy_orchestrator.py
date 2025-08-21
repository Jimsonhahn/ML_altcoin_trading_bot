"""
Multi-Strategy Orchestrator - Das Gehirn des Trading Bots
========================================================

Dieser Orchestrator verwaltet und koordiniert mehrere Trading-Strategien gleichzeitig,
analysiert deren Performance und wählt intelligent die beste Kombination für aktuelle Marktbedingungen.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance-Tracking für individuelle Strategien"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5  # 0-1 Konfidenz-Score
    market_conditions: Dict[str, float] = field(default_factory=dict)
    
    def update_metrics(self, trade_result: Dict[str, Any]):
        """Aktualisiert Performance-Metriken nach einem Trade"""
        self.total_trades += 1
        pnl = trade_result.get('pnl', 0)
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.avg_profit = ((self.avg_profit * (self.winning_trades - 1)) + pnl) / self.winning_trades
        else:
            self.losing_trades += 1
            self.avg_loss = ((self.avg_loss * (self.losing_trades - 1)) + abs(pnl)) / self.losing_trades
        
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.last_update = datetime.now()
        self._update_confidence_score()
    
    def _update_confidence_score(self):
        """Berechnet Konfidenz-Score basierend auf Performance"""
        if self.total_trades < 10:
            # Nicht genug Daten - bleibe bei 0.5
            return
        
        # Faktoren für Konfidenz
        win_rate_factor = self.win_rate * 0.4
        profit_factor = min((self.avg_profit / (self.avg_loss + 0.01)) * 0.1, 0.3)
        consistency_factor = min(self.total_trades / 100, 0.3) * 0.3
        
        self.confidence_score = win_rate_factor + profit_factor + consistency_factor


@dataclass
class MarketRegime:
    """Aktuelles Marktregime mit Konfidenz-Scores"""
    timestamp: datetime = field(default_factory=datetime.now)
    volatility: str = "medium"  # low, medium, high, extreme
    trend: str = "sideways"  # bullish, bearish, sideways
    volume: str = "normal"  # low, normal, high
    correlation: float = 0.5  # Inter-asset correlation
    regime_confidence: float = 0.0  # Konfidenz in die Regime-Erkennung
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'volatility': self.volatility,
            'trend': self.trend,
            'volume': self.volume,
            'correlation': self.correlation,
            'confidence': self.regime_confidence,
            'timestamp': self.timestamp.isoformat()
        }


class StrategyOrchestrator:
    """
    Intelligenter Orchestrator der mehrere Trading-Strategien koordiniert
    """
    
    def __init__(self, exchange_manager, config: Dict[str, Any]):
        self.exchange_manager = exchange_manager
        self.config = config
        
        # Strategy Management
        self.available_strategies: Dict[str, Any] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        
        # Orchestra Configuration
        self.max_concurrent_strategies = config.get('max_concurrent_strategies', 3)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        self.rebalance_interval = config.get('rebalance_interval', 300)  # 5 minutes
        self.performance_window = config.get('performance_window', 24)  # hours
        
        # Risk Management
        self.total_risk_limit = config.get('total_risk_limit', 1000)  # Total $ risk across all strategies
        self.strategy_risk_allocation: Dict[str, float] = {}
        
        # Market Analysis
        self.current_market_regime = MarketRegime()
        self.market_regime_history: List[MarketRegime] = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.is_running = False
        self.last_rebalance = datetime.now()
        
        # Initialize available strategies
        self._initialize_strategies()
        
        logger.info(f"Strategy Orchestrator initialized with {len(self.available_strategies)} strategies")
    
    def _initialize_strategies(self):
        """Initialisiert verfügbare Strategien"""
        # Strategy definitions with their optimal market conditions
        strategy_configs = {
            'momentum': {
                'optimal_conditions': {'trend': 'bullish', 'volatility': 'medium'},
                'risk_weight': 0.8,
                'min_confidence': 0.4
            },
            'mean_reversion': {
                'optimal_conditions': {'trend': 'sideways', 'volatility': 'low'},
                'risk_weight': 0.6,
                'min_confidence': 0.5
            },
            'arbitrage': {
                'optimal_conditions': {'correlation': 'high', 'volatility': 'low'},
                'risk_weight': 0.4,
                'min_confidence': 0.6
            },
            'grid_trading': {
                'optimal_conditions': {'trend': 'sideways', 'volatility': 'medium'},
                'risk_weight': 0.7,
                'min_confidence': 0.5
            },
            'candle_momentum': {
                'optimal_conditions': {'trend': 'bullish', 'volume': 'high'},
                'risk_weight': 0.9,
                'min_confidence': 0.4
            },
            'high_risk_daily': {
                'optimal_conditions': {'volatility': 'extreme', 'volume': 'high'},
                'risk_weight': 1.2,
                'min_confidence': 0.7
            },
            'ml_strategy': {
                'optimal_conditions': {'any': True},  # ML adapts to all conditions
                'risk_weight': 1.0,
                'min_confidence': 0.6
            },
            'adaptive_auto_strategy': {
                'optimal_conditions': {'any': True},  # Meta-strategy
                'risk_weight': 0.8,
                'min_confidence': 0.3
            },
            'smart_money_machine': {
                'optimal_conditions': {'any': True},  # Portfolio-split strategy works in all conditions
                'risk_weight': 1.0,
                'min_confidence': 0.4
            }
        }
        
        for strategy_name, config in strategy_configs.items():
            self.available_strategies[strategy_name] = config
            self.strategy_performance[strategy_name] = StrategyPerformance(strategy_name)
    
    async def start(self):
        """Startet den Strategy Orchestrator"""
        logger.info("Starting Strategy Orchestrator...")
        self.is_running = True
        
        # Start concurrent tasks
        await asyncio.gather(
            self._orchestration_loop(),
            self._market_analysis_loop(),
            self._performance_monitoring_loop()
        )
    
    async def stop(self):
        """Stoppt den Orchestrator"""
        logger.info("Stopping Strategy Orchestrator...")
        self.is_running = False
        self.executor.shutdown(wait=True)
    
    async def _orchestration_loop(self):
        """Haupt-Orchestrierungs-Loop"""
        while self.is_running:
            try:
                # 1. Analyze current market conditions
                await self._update_market_regime()
                
                # 2. Check if rebalancing is needed
                if self._should_rebalance():
                    await self._rebalance_strategies()
                
                # 3. Execute active strategies
                await self._execute_active_strategies()
                
                # 4. Monitor and adjust risk
                await self._monitor_risk()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _market_analysis_loop(self):
        """Kontinuierliche Marktanalyse"""
        while self.is_running:
            try:
                # Analyze market for each major pair
                symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
                
                for symbol in symbols:
                    await self._analyze_market_conditions(symbol)
                
                # Update regime confidence
                self._calculate_regime_confidence()
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error in market analysis: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self):
        """Überwacht Strategy Performance"""
        while self.is_running:
            try:
                # Update performance metrics for each strategy
                for strategy_name in self.active_strategies:
                    self._update_strategy_performance(strategy_name)
                
                # Log performance summary
                self._log_performance_summary()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(300)
    
    def _should_rebalance(self) -> bool:
        """Prüft ob Strategien neu ausbalanciert werden sollten"""
        # Time-based rebalancing
        time_since_rebalance = (datetime.now() - self.last_rebalance).seconds
        if time_since_rebalance < self.rebalance_interval:
            return False
        
        # Performance-based rebalancing
        for strategy in self.active_strategies:
            perf = self.strategy_performance[strategy]
            if perf.confidence_score < self.min_confidence_threshold:
                logger.info(f"Strategy {strategy} confidence below threshold: {perf.confidence_score:.2f}")
                return True
        
        # Market regime change
        if self._has_market_regime_changed():
            logger.info("Market regime change detected - rebalancing needed")
            return True
        
        return False
    
    async def _rebalance_strategies(self):
        """Rebalanciert aktive Strategien basierend auf Marktbedingungen"""
        logger.info("Rebalancing strategies...")
        
        # 1. Score all available strategies
        strategy_scores = await self._score_strategies_for_current_market()
        
        # 2. Sort by score
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Select top strategies
        new_active_strategies = set()
        for strategy_name, score in sorted_strategies[:self.max_concurrent_strategies]:
            if score >= self.min_confidence_threshold:
                new_active_strategies.add(strategy_name)
        
        # 4. Update active strategies
        strategies_to_add = new_active_strategies - self.active_strategies
        strategies_to_remove = self.active_strategies - new_active_strategies
        
        # Remove underperforming strategies
        for strategy in strategies_to_remove:
            logger.info(f"Removing strategy: {strategy}")
            self.active_strategies.remove(strategy)
        
        # Add new strategies
        for strategy in strategies_to_add:
            logger.info(f"Adding strategy: {strategy}")
            self.active_strategies.add(strategy)
        
        # 5. Reallocate risk budget
        self._allocate_risk_budget()
        
        self.last_rebalance = datetime.now()
        
        logger.info(f"Rebalancing complete. Active strategies: {list(self.active_strategies)}")
    
    async def _score_strategies_for_current_market(self) -> Dict[str, float]:
        """Bewertet Strategien für aktuelle Marktbedingungen"""
        scores = {}
        
        for strategy_name, config in self.available_strategies.items():
            # Base score from historical performance
            perf = self.strategy_performance[strategy_name]
            base_score = perf.confidence_score
            
            # Market condition match score
            market_match_score = self._calculate_market_match_score(
                config['optimal_conditions'],
                self.current_market_regime
            )
            
            # Combined score (weighted average)
            combined_score = (base_score * 0.6) + (market_match_score * 0.4)
            
            # Apply risk weight
            risk_adjusted_score = combined_score * (2 - config['risk_weight'])
            
            scores[strategy_name] = risk_adjusted_score
            
            logger.debug(f"Strategy {strategy_name} score: {risk_adjusted_score:.3f} "
                        f"(perf: {base_score:.3f}, market: {market_match_score:.3f})")
        
        return scores
    
    def _calculate_market_match_score(self, optimal_conditions: Dict, current_regime: MarketRegime) -> float:
        """Berechnet wie gut eine Strategie zu aktuellen Marktbedingungen passt"""
        if optimal_conditions.get('any', False):
            return 0.7  # Universal strategies get decent score
        
        score = 0.0
        factors = 0
        
        # Check each condition
        if 'trend' in optimal_conditions:
            factors += 1
            if optimal_conditions['trend'] == current_regime.trend:
                score += 1.0
            elif current_regime.trend == 'sideways':
                score += 0.5  # Partial match for sideways
        
        if 'volatility' in optimal_conditions:
            factors += 1
            if optimal_conditions['volatility'] == current_regime.volatility:
                score += 1.0
            elif abs(self._volatility_to_number(optimal_conditions['volatility']) - 
                    self._volatility_to_number(current_regime.volatility)) == 1:
                score += 0.5  # Adjacent volatility levels
        
        if 'volume' in optimal_conditions:
            factors += 1
            if optimal_conditions['volume'] == current_regime.volume:
                score += 1.0
        
        if 'correlation' in optimal_conditions:
            factors += 1
            if optimal_conditions['correlation'] == 'high' and current_regime.correlation > 0.7:
                score += 1.0
            elif optimal_conditions['correlation'] == 'low' and current_regime.correlation < 0.3:
                score += 1.0
        
        return score / factors if factors > 0 else 0.5
    
    def _volatility_to_number(self, vol_str: str) -> int:
        """Konvertiert Volatilitäts-String zu Zahl für Vergleiche"""
        mapping = {'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
        return mapping.get(vol_str, 2)
    
    def _allocate_risk_budget(self):
        """Verteilt Risiko-Budget auf aktive Strategien"""
        if not self.active_strategies:
            return
        
        # Calculate total weight
        total_weight = sum(
            self.available_strategies[s]['risk_weight'] 
            for s in self.active_strategies
        )
        
        # Allocate proportionally
        for strategy in self.active_strategies:
            weight = self.available_strategies[strategy]['risk_weight']
            allocation = (weight / total_weight) * self.total_risk_limit
            self.strategy_risk_allocation[strategy] = allocation
            
            logger.info(f"Risk allocation for {strategy}: ${allocation:.2f}")
    
    async def _execute_active_strategies(self):
        """Führt alle aktiven Strategien aus"""
        if not self.active_strategies:
            return
        
        # Execute strategies concurrently
        tasks = []
        for strategy_name in self.active_strategies:
            task = self._execute_single_strategy(strategy_name)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for strategy_name, result in zip(self.active_strategies, results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy_name} execution failed: {result}")
            elif result and result.get('executed'):
                logger.info(f"Strategy {strategy_name} executed: {result}")
    
    async def _execute_single_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Führt eine einzelne Strategie aus"""
        try:
            # Get risk allocation for this strategy
            risk_budget = self.strategy_risk_allocation.get(strategy_name, 0)
            
            if risk_budget <= 0:
                return {'executed': False, 'reason': 'No risk budget allocated'}
            
            # Here you would call the actual strategy implementation
            # For now, we'll simulate
            result = {
                'executed': True,
                'strategy': strategy_name,
                'risk_used': risk_budget * 0.1,  # Use 10% of allocated risk
                'timestamp': datetime.now().isoformat()
            }
            
            # Update performance tracking
            # self.strategy_performance[strategy_name].update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
            return {'executed': False, 'error': str(e)}
    
    async def _update_market_regime(self):
        """Aktualisiert das aktuelle Marktregime"""
        try:
            # Get market data for analysis
            symbol = 'BTC/USDT'  # Primary market indicator
            timeframe = '1h'
            
            ohlcv = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=100)
            if ohlcv is None or len(ohlcv) < 50:
                return
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate market metrics
            volatility = self._calculate_volatility(df)
            trend = self._calculate_trend(df)
            volume_profile = self._calculate_volume_profile(df)
            
            # Update regime
            self.current_market_regime = MarketRegime(
                volatility=volatility,
                trend=trend,
                volume=volume_profile,
                correlation=await self._calculate_correlation(),
                regime_confidence=0.8  # Placeholder
            )
            
            # Store in history
            self.market_regime_history.append(self.current_market_regime)
            if len(self.market_regime_history) > 100:
                self.market_regime_history.pop(0)
            
            logger.info(f"Market regime updated: {self.current_market_regime.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error updating market regime: {str(e)}")
    
    def _calculate_volatility(self, df: pd.DataFrame) -> str:
        """Berechnet Volatilitäts-Regime"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(24)  # Daily volatility
        
        if volatility < 0.02:
            return 'low'
        elif volatility < 0.04:
            return 'medium'
        elif volatility < 0.08:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """Berechnet Trend-Richtung"""
        # Simple trend using SMA crossover
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        if len(df) < 50:
            return 'sideways'
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if sma_20 > sma_50 and current_price > sma_20:
            return 'bullish'
        elif sma_20 < sma_50 and current_price < sma_20:
            return 'bearish'
        else:
            return 'sideways'
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> str:
        """Berechnet Volumen-Profil"""
        current_volume = df['volume'].iloc[-24:].mean()  # Last 24 hours
        avg_volume = df['volume'].mean()
        
        ratio = current_volume / avg_volume
        
        if ratio < 0.7:
            return 'low'
        elif ratio < 1.3:
            return 'normal'
        else:
            return 'high'
    
    async def _calculate_correlation(self) -> float:
        """Berechnet Inter-Asset Korrelation"""
        # Simplified - would calculate correlation between major pairs
        return 0.5
    
    def _has_market_regime_changed(self) -> bool:
        """Prüft ob sich das Marktregime signifikant geändert hat"""
        if len(self.market_regime_history) < 2:
            return False
        
        current = self.market_regime_history[-1]
        previous = self.market_regime_history[-2]
        
        # Check for significant changes
        if current.volatility != previous.volatility:
            return True
        if current.trend != previous.trend:
            return True
        if abs(current.correlation - previous.correlation) > 0.3:
            return True
        
        return False
    
    async def _monitor_risk(self):
        """Überwacht und managed Gesamt-Risiko"""
        total_risk_used = sum(self.strategy_risk_allocation.values())
        
        if total_risk_used > self.total_risk_limit * 0.9:
            logger.warning(f"Risk limit approaching: ${total_risk_used:.2f} / ${self.total_risk_limit:.2f}")
            # Reduce positions or stop new trades
            await self._reduce_risk_exposure()
    
    async def _reduce_risk_exposure(self):
        """Reduziert Risiko-Exposure wenn Limits erreicht werden"""
        # Find worst performing strategy
        worst_strategy = None
        worst_performance = float('inf')
        
        for strategy in self.active_strategies:
            perf = self.strategy_performance[strategy]
            if perf.confidence_score < worst_performance:
                worst_performance = perf.confidence_score
                worst_strategy = strategy
        
        if worst_strategy and worst_performance < 0.4:
            logger.warning(f"Removing underperforming strategy: {worst_strategy}")
            self.active_strategies.remove(worst_strategy)
            self._allocate_risk_budget()
    
    def _update_strategy_performance(self, strategy_name: str):
        """Aktualisiert Performance-Metriken einer Strategie"""
        # This would fetch actual trade results from database
        # For now, we'll simulate
        pass
    
    def _log_performance_summary(self):
        """Loggt Performance-Zusammenfassung aller Strategien"""
        summary = []
        for strategy_name, perf in self.strategy_performance.items():
            if perf.total_trades > 0:
                summary.append(f"{strategy_name}: WR={perf.win_rate:.1%}, PnL=${perf.total_pnl:.2f}, Conf={perf.confidence_score:.2f}")
        
        if summary:
            logger.info("Strategy Performance Summary: " + " | ".join(summary))
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Status des Orchestrators zurück"""
        return {
            'active_strategies': list(self.active_strategies),
            'market_regime': self.current_market_regime.to_dict(),
            'risk_allocation': self.strategy_risk_allocation,
            'total_risk_used': sum(self.strategy_risk_allocation.values()),
            'performance_summary': {
                name: {
                    'win_rate': perf.win_rate,
                    'total_pnl': perf.total_pnl,
                    'confidence': perf.confidence_score,
                    'trades': perf.total_trades
                }
                for name, perf in self.strategy_performance.items()
                if perf.total_trades > 0
            }
        }
    
    async def force_rebalance(self):
        """Erzwingt eine sofortige Neubalancierung"""
        logger.info("Force rebalancing triggered")
        self.last_rebalance = datetime.min  # Force rebalance on next cycle
        await self._rebalance_strategies()