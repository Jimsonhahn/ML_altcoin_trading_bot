"""
Lazy Billionaire Strategy - Master-Strategie für vollautomatisches Portfolio-Management
Orchestriert alle Sub-Strategien basierend auf Marktbedingungen und Kapitalallokation
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from strategies.strategy_base import Strategy
from core.strategy_router import StrategyRouter, MarketPhase, StrategyAllocation
from core.market_analyzer import MarketAnalyzer
from core.position import Position
from utils.exceptions import StrategyError
from utils.risk_management import RiskManager


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für die Lazy Billionaire Strategy"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    active_strategies: int
    successful_rebalances: int
    emergency_stops: int
    last_rebalance: datetime


class LazyBillionaireStrategy(Strategy):
    """
    Master-Strategie die automatisch zwischen verschiedenen Trading-Strategien wechselt
    basierend auf Marktbedingungen und optimaler Kapitalallokation.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.logger = logging.getLogger(__name__)
        
        # Konfiguration
        self.initial_capital = params.get('initial_capital', 300000)  # 300k EUR
        self.risk_per_trade = params.get('risk_per_trade', 0.02)  # 2% pro Trade
        self.max_drawdown_limit = params.get('max_drawdown_limit', 0.15)  # 15% Max Drawdown
        self.rebalance_threshold = params.get('rebalance_threshold', 0.05)  # 5% Threshold
        self.profile = params.get('profile', 'default')  # Gewichtungsprofil
        
        # Komponenten
        self.strategy_router: Optional[StrategyRouter] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Status
        self.current_allocations: Dict[str, StrategyAllocation] = {}
        self.active_positions: Dict[str, Position] = {}
        self.performance_history: List[PerformanceMetrics] = []
        self.last_rebalance = datetime.now()
        self.emergency_mode = False
        
        # Performance Tracking
        self.start_capital = self.initial_capital
        self.current_capital = self.initial_capital
        self.high_water_mark = self.initial_capital
        self.total_trades = 0
        self.successful_trades = 0
        
        # Sub-Strategien Registry
        self.sub_strategies: Dict[str, Strategy] = {}
        
        self.logger.info(f"Lazy Billionaire Strategy initialized with {self.initial_capital:,.2f} EUR")

    def initialize(self, strategy_router: StrategyRouter, market_analyzer: MarketAnalyzer, 
                  risk_manager: RiskManager) -> None:
        """Initialisiert die Master-Strategie mit ihren Komponenten"""
        self.strategy_router = strategy_router
        self.market_analyzer = market_analyzer
        self.risk_manager = risk_manager
        
        self.logger.info("Lazy Billionaire Strategy components initialized")

    async def execute(self, market_data: Dict[str, Any]) -> List[Position]:
        """
        Hauptausführung der Lazy Billionaire Strategy
        """
        try:
            # 1. Marktanalyse durchführen
            market_condition = await self._analyze_market_conditions(market_data)
            
            # 2. Prüfe ob Notfall-Modus aktiviert werden muss
            if await self._should_enter_emergency_mode(market_condition):
                return await self._handle_emergency_mode()
            
            # 3. Prüfe ob Rebalancing nötig ist
            if await self._should_rebalance(market_condition):
                await self._rebalance_portfolio(market_condition)
            
            # 4. Führe aktive Strategien aus
            new_positions = await self._execute_active_strategies(market_data)
            
            # 5. Risikomanagement anwenden
            filtered_positions = await self._apply_risk_management(new_positions)
            
            # 6. Performance tracken
            await self._update_performance_metrics()
            
            return filtered_positions
            
        except Exception as e:
            self.logger.error(f"Error in Lazy Billionaire Strategy execution: {e}")
            return []

    async def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict:
        """Analysiert die aktuellen Marktbedingungen"""
        try:
            if self.market_analyzer:
                return await self.market_analyzer.get_market_metrics()
            else:
                # Fallback: Verwende übergebene Marktdaten
                return {
                    'price_change_24h': market_data.get('price_change_24h', 0),
                    'volatility': market_data.get('volatility', 0.2),
                    'volume': market_data.get('volume', 0),
                    'fear_greed_index': 50,
                    'trend_strength': 0.3,
                    'confidence': 0.5
                }
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {'price_change_24h': 0, 'volatility': 0.2, 'confidence': 0.3}

    async def _should_enter_emergency_mode(self, market_condition: Dict) -> bool:
        """Prüft ob der Notfall-Modus aktiviert werden sollte"""
        try:
            # Extreme Fear Index
            if market_condition.get('fear_greed_index', 50) <= 15:
                self.logger.warning("Extreme fear detected - considering emergency mode")
                return True
            
            # Maximaler Drawdown erreicht
            current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark
            if current_drawdown >= self.max_drawdown_limit:
                self.logger.warning(f"Max drawdown limit reached: {current_drawdown:.2%}")
                return True
            
            # Extreme Volatilität
            if market_condition.get('volatility', 0) > 0.5:
                self.logger.warning("Extreme volatility detected")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
            return False

    async def _handle_emergency_mode(self) -> List[Position]:
        """Behandelt den Notfall-Modus"""
        try:
            if not self.emergency_mode:
                self.logger.critical("EMERGENCY MODE ACTIVATED - Moving to stablecoins")
                self.emergency_mode = True
                
                # Schließe alle aktiven Positionen
                await self._close_all_positions()
                
                # Parke alles in Stablecoins
                await self._park_in_stablecoins()
                
                # Benachrichtige über Notfall-Modus
                await self._notify_emergency_mode()
            
            return []  # Keine neuen Positionen im Notfall-Modus
            
        except Exception as e:
            self.logger.error(f"Error handling emergency mode: {e}")
            return []

    async def _should_rebalance(self, market_condition: Dict) -> bool:
        """Prüft ob ein Rebalancing durchgeführt werden sollte"""
        try:
            # Zeit-basiertes Rebalancing (alle 30 Minuten)
            time_since_last = datetime.now() - self.last_rebalance
            if time_since_last >= timedelta(minutes=30):
                return True
            
            # Marktbedingungen haben sich signifikant geändert
            if self.strategy_router:
                return await self.strategy_router.rebalance_strategies(self.current_capital, force=False)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance conditions: {e}")
            return False

    async def _rebalance_portfolio(self, market_condition: Dict) -> None:
        """Führt das Portfolio-Rebalancing durch"""
        try:
            if not self.strategy_router:
                return
            
            self.logger.info("Starting portfolio rebalancing...")
            
            # Analysiere Marktphase
            current_phase = await self.strategy_router.analyze_market_phase()
            
            # Berechne neue Allokationen
            new_allocations = await self.strategy_router.rebalance_strategies(
                self.current_capital, force=True
            )
            
            if new_allocations:
                # Aktualisiere aktive Strategien
                await self._update_active_strategies(new_allocations)
                
                # Speichere neue Allokationen
                self.current_allocations = new_allocations
                self.last_rebalance = datetime.now()
                
                self.logger.info(f"Portfolio rebalanced for {current_phase.phase.value} market phase")
                
                # Logge neue Allokationen
                await self._log_allocation_changes(new_allocations)
            
        except Exception as e:
            self.logger.error(f"Error during portfolio rebalancing: {e}")

    async def _execute_active_strategies(self, market_data: Dict[str, Any]) -> List[Position]:
        """Führt alle aktiven Sub-Strategien aus"""
        try:
            all_positions = []
            
            for strategy_name, allocation in self.current_allocations.items():
                if allocation.is_active and strategy_name in self.sub_strategies:
                    try:
                        strategy = self.sub_strategies[strategy_name]
                        
                        # Führe Sub-Strategie aus
                        positions = await strategy.execute(market_data)
                        
                        # Skaliere Positionen basierend auf Allokation
                        scaled_positions = await self._scale_positions(positions, allocation)
                        
                        all_positions.extend(scaled_positions)
                        
                    except Exception as e:
                        self.logger.error(f"Error executing strategy {strategy_name}: {e}")
                        continue
            
            return all_positions
            
        except Exception as e:
            self.logger.error(f"Error executing active strategies: {e}")
            return []

    async def _apply_risk_management(self, positions: List[Position]) -> List[Position]:
        """Wendet Risikomanagement auf die Positionen an"""
        try:
            if not self.risk_manager:
                return positions
            
            filtered_positions = []
            
            for position in positions:
                # Prüfe Positionsgröße
                if await self._validate_position_size(position):
                    # Prüfe Korrelation zu existierenden Positionen
                    if await self._validate_position_correlation(position):
                        # Prüfe Gesamtrisiko
                        if await self._validate_total_risk(position):
                            filtered_positions.append(position)
                        else:
                            self.logger.warning(f"Position rejected due to total risk: {position.symbol}")
                    else:
                        self.logger.warning(f"Position rejected due to correlation: {position.symbol}")
                else:
                    self.logger.warning(f"Position rejected due to size: {position.symbol}")
            
            return filtered_positions
            
        except Exception as e:
            self.logger.error(f"Error applying risk management: {e}")
            return positions

    async def _update_performance_metrics(self) -> None:
        """Aktualisiert die Performance-Metriken"""
        try:
            # Berechne aktuelles Portfolio-Wert
            portfolio_value = await self._calculate_portfolio_value()
            
            # Aktualisiere High Water Mark
            if portfolio_value > self.high_water_mark:
                self.high_water_mark = portfolio_value
            
            # Berechne Performance-Metriken
            total_return = (portfolio_value - self.start_capital) / self.start_capital
            current_drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
            
            # Erstelle Performance-Snapshot
            metrics = PerformanceMetrics(
                total_return=total_return,
                sharpe_ratio=await self._calculate_sharpe_ratio(),
                max_drawdown=current_drawdown,
                volatility=await self._calculate_portfolio_volatility(),
                active_strategies=len(self.current_allocations),
                successful_rebalances=len([m for m in self.performance_history if m.successful_rebalances > 0]),
                emergency_stops=len([m for m in self.performance_history if m.emergency_stops > 0]),
                last_rebalance=self.last_rebalance
            )
            
            self.performance_history.append(metrics)
            
            # Logge Performance
            if len(self.performance_history) % 10 == 0:  # Alle 10 Updates
                await self._log_performance_summary()
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _scale_positions(self, positions: List[Position], allocation: StrategyAllocation) -> List[Position]:
        """Skaliert Positionen basierend auf Kapitalallokation"""
        try:
            scaled_positions = []
            
            for position in positions:
                # Berechne skalierte Positionsgröße
                scale_factor = allocation.capital_amount / self.current_capital
                scaled_size = position.size * scale_factor
                
                # Erstelle skalierte Position
                scaled_position = Position(
                    symbol=position.symbol,
                    side=position.side,
                    size=scaled_size,
                    entry_price=position.entry_price,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    strategy=f"lazy_billionaire_{allocation.strategy_name}"
                )
                
                scaled_positions.append(scaled_position)
            
            return scaled_positions
            
        except Exception as e:
            self.logger.error(f"Error scaling positions: {e}")
            return positions

    async def _validate_position_size(self, position: Position) -> bool:
        """Validiert die Positionsgröße"""
        try:
            # Maximale Positionsgröße: 5% des Portfolios
            max_position_value = self.current_capital * 0.05
            position_value = position.size * position.entry_price
            
            return position_value <= max_position_value
            
        except Exception as e:
            self.logger.error(f"Error validating position size: {e}")
            return False

    async def _validate_position_correlation(self, position: Position) -> bool:
        """Validiert die Korrelation zu existierenden Positionen"""
        try:
            # Vereinfachte Korrelationsprüfung
            # In Realität würde man historische Korrelationen verwenden
            
            similar_positions = [p for p in self.active_positions.values() 
                               if p.symbol.split('/')[0] == position.symbol.split('/')[0]]
            
            # Maximal 3 Positionen im gleichen Basisasset
            return len(similar_positions) < 3
            
        except Exception as e:
            self.logger.error(f"Error validating position correlation: {e}")
            return True

    async def _validate_total_risk(self, position: Position) -> bool:
        """Validiert das Gesamtrisiko"""
        try:
            # Berechne potenzieller Verlust
            potential_loss = position.size * abs(position.entry_price - (position.stop_loss or 0))
            
            # Maximaler Verlust: 2% des Portfolios pro Position
            max_loss = self.current_capital * self.risk_per_trade
            
            return potential_loss <= max_loss
            
        except Exception as e:
            self.logger.error(f"Error validating total risk: {e}")
            return False

    async def _calculate_portfolio_value(self) -> float:
        """Berechnet den aktuellen Portfolio-Wert"""
        try:
            # Vereinfachte Berechnung
            # In Realität würde man aktuelle Marktpreise verwenden
            total_value = self.current_capital
            
            for position in self.active_positions.values():
                # Berechne unrealisierten P&L
                unrealized_pnl = position.calculate_unrealized_pnl()
                total_value += unrealized_pnl
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.current_capital

    async def _calculate_sharpe_ratio(self) -> float:
        """Berechnet die Sharpe Ratio"""
        try:
            if len(self.performance_history) < 10:
                return 0.0
            
            # Berechne Returns
            returns = [m.total_return for m in self.performance_history[-30:]]  # Letzte 30 Messungen
            
            if not returns:
                return 0.0
            
            # Berechne Sharpe Ratio (vereinfacht)
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            risk_free_rate = 0.02  # 2% Risikofreier Zinssatz
            
            if volatility == 0:
                return 0.0
            
            return (avg_return - risk_free_rate) / volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_portfolio_volatility(self) -> float:
        """Berechnet die Portfolio-Volatilität"""
        try:
            if len(self.performance_history) < 10:
                return 0.0
            
            returns = [m.total_return for m in self.performance_history[-30:]]
            
            if not returns:
                return 0.0
            
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            
            return variance ** 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    async def _log_performance_summary(self) -> None:
        """Loggt eine Performance-Zusammenfassung"""
        try:
            if not self.performance_history:
                return
            
            latest_metrics = self.performance_history[-1]
            
            self.logger.info("=== LAZY BILLIONAIRE PERFORMANCE SUMMARY ===")
            self.logger.info(f"Total Return: {latest_metrics.total_return:.2%}")
            self.logger.info(f"Sharpe Ratio: {latest_metrics.sharpe_ratio:.3f}")
            self.logger.info(f"Max Drawdown: {latest_metrics.max_drawdown:.2%}")
            self.logger.info(f"Volatility: {latest_metrics.volatility:.2%}")
            self.logger.info(f"Active Strategies: {latest_metrics.active_strategies}")
            self.logger.info(f"Portfolio Value: {await self._calculate_portfolio_value():,.2f} EUR")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")

    async def _close_all_positions(self) -> None:
        """Schließt alle aktiven Positionen"""
        try:
            self.logger.info("Closing all active positions...")
            
            for position_id, position in self.active_positions.items():
                try:
                    # Schließe Position (vereinfacht)
                    # In Realität würde man die Exchange API verwenden
                    self.logger.info(f"Closing position {position.symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error closing position {position.symbol}: {e}")
            
            self.active_positions.clear()
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")

    async def _park_in_stablecoins(self) -> None:
        """Parkt das gesamte Kapital in Stablecoins"""
        try:
            self.logger.info("Parking capital in stablecoins...")
            
            # Implementierung würde hier das Kapital in USDT/USDC parken
            # Für Demo-Zwecke setzen wir nur den Status
            
            self.current_allocations = {
                "stablecoin_parking": StrategyAllocation(
                    strategy_name="stablecoin_parking",
                    weight=1.0,
                    capital_amount=self.current_capital,
                    max_allocation=self.current_capital,
                    min_allocation=self.current_capital,
                    is_active=True
                )
            }
            
            self.logger.info(f"All capital ({self.current_capital:,.2f} EUR) parked in stablecoins")
            
        except Exception as e:
            self.logger.error(f"Error parking in stablecoins: {e}")

    async def _notify_emergency_mode(self) -> None:
        """Benachrichtigt über den Notfall-Modus"""
        try:
            message = f"🚨 EMERGENCY MODE ACTIVATED 🚨\n"
            message += f"Portfolio: {self.current_capital:,.2f} EUR\n"
            message += f"All positions closed and capital moved to stablecoins\n"
            message += f"Timestamp: {datetime.now().isoformat()}"
            
            self.logger.critical(message)
            
            # Hier würde normalerweise eine Benachrichtigung an den User gesendet
            # (E-Mail, SMS, Push-Notification, etc.)
            
        except Exception as e:
            self.logger.error(f"Error sending emergency notification: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Status der Lazy Billionaire Strategy zurück"""
        try:
            portfolio_value = asyncio.run(self._calculate_portfolio_value())
            
            return {
                "strategy": "lazy_billionaire",
                "status": "emergency" if self.emergency_mode else "active",
                "capital": {
                    "initial": self.start_capital,
                    "current": portfolio_value,
                    "high_water_mark": self.high_water_mark
                },
                "performance": {
                    "total_return": (portfolio_value - self.start_capital) / self.start_capital,
                    "current_drawdown": (self.high_water_mark - portfolio_value) / self.high_water_mark,
                    "sharpe_ratio": asyncio.run(self._calculate_sharpe_ratio())
                },
                "allocations": {
                    name: {
                        "weight": alloc.weight,
                        "capital": alloc.capital_amount,
                        "active": alloc.is_active
                    } for name, alloc in self.current_allocations.items()
                },
                "active_positions": len(self.active_positions),
                "last_rebalance": self.last_rebalance.isoformat(),
                "emergency_mode": self.emergency_mode
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    async def _update_active_strategies(self, allocations: Dict[str, StrategyAllocation]) -> None:
        """Aktualisiert die aktiven Sub-Strategien"""
        try:
            # Hier würde normalerweise die Logik stehen um Sub-Strategien zu aktivieren/deaktivieren
            # Für Demo-Zwecke loggen wir nur die Änderungen
            
            for strategy_name, allocation in allocations.items():
                if allocation.is_active:
                    self.logger.info(f"Strategy {strategy_name} activated with {allocation.capital_amount:,.2f} EUR")
                else:
                    self.logger.info(f"Strategy {strategy_name} deactivated")
                    
        except Exception as e:
            self.logger.error(f"Error updating active strategies: {e}")

    async def _log_allocation_changes(self, allocations: Dict[str, StrategyAllocation]) -> None:
        """Loggt Änderungen in der Kapitalallokation"""
        try:
            self.logger.info("=== CAPITAL ALLOCATION UPDATE ===")
            total_allocated = sum(alloc.capital_amount for alloc in allocations.values())
            
            for strategy_name, allocation in allocations.items():
                percentage = (allocation.capital_amount / total_allocated) * 100
                self.logger.info(f"{strategy_name}: {allocation.capital_amount:,.2f} EUR ({percentage:.1f}%)")
            
            self.logger.info(f"Total Allocated: {total_allocated:,.2f} EUR")
            self.logger.info("=" * 35)
            
        except Exception as e:
            self.logger.error(f"Error logging allocation changes: {e}")

    def stop(self) -> None:
        """Stoppt die Lazy Billionaire Strategy"""
        try:
            self.logger.info("Stopping Lazy Billionaire Strategy...")
            
            # Schließe alle Positionen
            asyncio.run(self._close_all_positions())
            
            # Logge finale Performance
            asyncio.run(self._log_performance_summary())
            
            self.logger.info("Lazy Billionaire Strategy stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy: {e}")