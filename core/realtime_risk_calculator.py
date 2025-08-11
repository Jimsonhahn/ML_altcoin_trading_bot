"""
Real-time Risk Calculator
========================

Professionelle Echtzeit-Risikoberechnung mit:
- Kontinuierliches Monitoring
- Event-driven Updates  
- Performance-optimierte Berechnung
- Multi-threaded Processing
"""
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd

from core.interfaces import global_event_bus

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Container für Risikokennzahlen"""
    timestamp: datetime
    current_drawdown: float
    max_drawdown: float
    portfolio_var: float  # Value at Risk
    sharpe_ratio: float
    total_exposure: float
    position_concentration: float
    correlation_risk: float
    liquidity_risk: float
    
    # Performance Metrics
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    
    # Position Metrics
    open_positions_count: int
    largest_position_size: float
    avg_position_age: float
    
    # Risk Flags
    is_risk_limit_breached: bool = False
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    warnings: List[str] = field(default_factory=list)

class RealTimeRiskCalculator:
    """
    Echtzeit-Risikokalkulationsengine
    """
    
    def __init__(self, 
                 update_interval: float = 1.0,  # Sekunden
                 max_drawdown_limit: float = 0.15,
                 var_confidence: float = 0.95,
                 lookback_window: int = 252):
        
        self.update_interval = update_interval
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        self.lookback_window = lookback_window
        
        # Threading
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Data Storage
        self._price_history = {}  # symbol -> deque of prices
        self._pnl_history = deque(maxlen=lookback_window)
        self._portfolio_values = deque(maxlen=lookback_window)
        
        # Current State
        self._current_positions = {}
        self._current_portfolio_value = 0.0
        self._peak_portfolio_value = 0.0
        self._initial_capital = 0.0
        
        # Risk Callbacks
        self._risk_callbacks: List[Callable[[RiskMetrics], None]] = []
        
        # Metrics Cache
        self._last_metrics: Optional[RiskMetrics] = None
        self._metrics_cache_time = 0
        
        # Event Subscriptions
        self._setup_event_handlers()
        
        logger.info(f"RealTimeRiskCalculator initialized with {update_interval}s interval")
    
    def _setup_event_handlers(self):
        """Setup Event Bus Handlers"""
        global_event_bus.subscribe("position_update", self._on_position_update)
        global_event_bus.subscribe("price_update", self._on_price_update)
        global_event_bus.subscribe("portfolio_update", self._on_portfolio_update)
        global_event_bus.subscribe("trade_executed", self._on_trade_executed)
    
    def start_monitoring(self, initial_capital: float):
        """Starte Echtzeit-Monitoring"""
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.warning("Risk monitoring already running")
                return
            
            self._initial_capital = initial_capital
            self._current_portfolio_value = initial_capital
            self._peak_portfolio_value = initial_capital
            self._stop_monitoring.clear()
            
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="RiskMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
            
            logger.info("Real-time risk monitoring started")
    
    def stop_monitoring(self):
        """Stoppe Echtzeit-Monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            logger.info("Real-time risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Haupt-Monitoring Loop"""
        last_calculation = 0
        
        while not self._stop_monitoring.is_set():
            try:
                current_time = time.time()
                
                # Rate Limiting - berechne nur bei Bedarf
                if current_time - last_calculation >= self.update_interval:
                    metrics = self._calculate_risk_metrics()
                    
                    if metrics:
                        self._process_risk_metrics(metrics)
                        last_calculation = current_time
                
                # Kurz schlafen um CPU zu schonen
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def _calculate_risk_metrics(self) -> Optional[RiskMetrics]:
        """Berechne aktuelle Risikokennzahlen"""
        with self._lock:
            try:
                now = datetime.now()
                
                # Portfolio Value Updates
                unrealized_pnl = self._calculate_unrealized_pnl()
                current_value = self._initial_capital + unrealized_pnl
                
                # Update Portfolio History
                self._portfolio_values.append(current_value)
                self._current_portfolio_value = current_value
                
                # Update Peak Value
                if current_value > self._peak_portfolio_value:
                    self._peak_portfolio_value = current_value
                
                # Drawdown Calculation
                current_drawdown = (self._peak_portfolio_value - current_value) / self._peak_portfolio_value
                max_drawdown = self._calculate_max_drawdown()
                
                # VaR Calculation
                portfolio_var = self._calculate_var()
                
                # Sharpe Ratio
                sharpe_ratio = self._calculate_sharpe_ratio()
                
                # Position Metrics
                position_metrics = self._calculate_position_metrics()
                
                # Risk Assessment
                risk_level, warnings = self._assess_risk_level(current_drawdown, portfolio_var)
                
                metrics = RiskMetrics(
                    timestamp=now,
                    current_drawdown=current_drawdown,
                    max_drawdown=max_drawdown,
                    portfolio_var=portfolio_var,
                    sharpe_ratio=sharpe_ratio,
                    total_exposure=position_metrics['total_exposure'],
                    position_concentration=position_metrics['concentration'],
                    correlation_risk=position_metrics['correlation_risk'],
                    liquidity_risk=position_metrics['liquidity_risk'],
                    total_pnl=current_value - self._initial_capital,
                    daily_pnl=self._calculate_daily_pnl(),
                    unrealized_pnl=unrealized_pnl,
                    open_positions_count=len(self._current_positions),
                    largest_position_size=position_metrics['largest_position'],
                    avg_position_age=position_metrics['avg_age'],
                    is_risk_limit_breached=current_drawdown > self.max_drawdown_limit,
                    risk_level=risk_level,
                    warnings=warnings
                )
                
                self._last_metrics = metrics
                self._metrics_cache_time = time.time()
                
                return metrics
                
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {e}")
                return None
    
    def _calculate_unrealized_pnl(self) -> float:
        """Berechne aktuellen unrealisierten P&L"""
        total_unrealized = 0.0
        
        for symbol, position in self._current_positions.items():
            if position['quantity'] != 0:
                # Hole aktuellen Preis
                current_price = self._get_current_price(symbol)
                if current_price:
                    entry_price = position['avg_price']
                    quantity = position['quantity']
                    
                    if position['side'] == 'long':
                        unrealized = (current_price - entry_price) * quantity
                    else:  # short
                        unrealized = (entry_price - current_price) * quantity
                    
                    total_unrealized += unrealized
        
        return total_unrealized
    
    def _calculate_max_drawdown(self) -> float:
        """Berechne maximalen historischen Drawdown"""
        if len(self._portfolio_values) < 2:
            return 0.0
        
        values = np.array(self._portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        
        return float(np.max(drawdown))
    
    def _calculate_var(self) -> float:
        """Berechne Value at Risk"""
        if len(self._pnl_history) < 30:
            return 0.0
        
        returns = np.array(self._pnl_history)
        return float(np.percentile(returns, (1 - self.var_confidence) * 100))
    
    def _calculate_sharpe_ratio(self) -> float:
        """Berechne Sharpe Ratio"""
        if len(self._pnl_history) < 30:
            return 0.0
        
        returns = np.array(self._pnl_history)
        return float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    
    def _calculate_position_metrics(self) -> Dict[str, float]:
        """Berechne Position-spezifische Metriken"""
        if not self._current_positions:
            return {
                'total_exposure': 0.0,
                'concentration': 0.0,
                'correlation_risk': 0.0,
                'liquidity_risk': 0.0,
                'largest_position': 0.0,
                'avg_age': 0.0
            }
        
        exposures = []
        largest_position = 0.0
        total_exposure = 0.0
        position_ages = []
        
        for symbol, position in self._current_positions.items():
            exposure = abs(position['quantity'] * position['avg_price'])
            exposures.append(exposure)
            total_exposure += exposure
            
            if exposure > largest_position:
                largest_position = exposure
            
            # Position Age in hours
            if 'entry_time' in position:
                age = (datetime.now() - position['entry_time']).total_seconds() / 3600
                position_ages.append(age)
        
        # Concentration (HHI)
        if total_exposure > 0:
            concentration = sum((exp / total_exposure) ** 2 for exp in exposures)
        else:
            concentration = 0.0
        
        # Average Position Age
        avg_age = np.mean(position_ages) if position_ages else 0.0
        
        # Simplified Risk Metrics
        correlation_risk = min(concentration * 0.5, 1.0)  # Simplified
        liquidity_risk = min(len(self._current_positions) * 0.1, 1.0)  # Simplified
        
        return {
            'total_exposure': total_exposure,
            'concentration': concentration,
            'correlation_risk': correlation_risk,
            'liquidity_risk': liquidity_risk,
            'largest_position': largest_position,
            'avg_age': avg_age
        }
    
    def _calculate_daily_pnl(self) -> float:
        """Berechne Tages-P&L"""
        if len(self._portfolio_values) < 2:
            return 0.0
        
        return float(self._portfolio_values[-1] - self._portfolio_values[-2])
    
    def _assess_risk_level(self, drawdown: float, var: float) -> tuple[str, List[str]]:
        """Bewerte Risiko-Level"""
        warnings = []
        
        # Drawdown Checks
        if drawdown > self.max_drawdown_limit:
            warnings.append(f"Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_limit:.1%}")
            risk_level = "CRITICAL"
        elif drawdown > self.max_drawdown_limit * 0.7:
            warnings.append(f"Drawdown {drawdown:.1%} approaching limit")
            risk_level = "HIGH"
        elif drawdown > self.max_drawdown_limit * 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # VaR Checks
        if abs(var) > self._current_portfolio_value * 0.05:
            warnings.append(f"High VaR detected: {var:.2f}")
            if risk_level in ["LOW", "MEDIUM"]:
                risk_level = "HIGH"
        
        # Position Concentration
        position_metrics = self._calculate_position_metrics()
        if position_metrics['concentration'] > 0.5:
            warnings.append("High position concentration detected")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        return risk_level, warnings
    
    def _process_risk_metrics(self, metrics: RiskMetrics):
        """Verarbeite berechnete Risikometriken"""
        # Publish Event
        global_event_bus.publish("risk_metrics_update", {
            'metrics': metrics,
            'timestamp': metrics.timestamp.isoformat()
        })
        
        # Call Registered Callbacks
        for callback in self._risk_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in risk callback: {e}")
        
        # Log Critical Events
        if metrics.is_risk_limit_breached:
            logger.critical(f"RISK LIMIT BREACHED: Drawdown {metrics.current_drawdown:.1%}")
            global_event_bus.publish("risk_limit_breached", {
                'drawdown': metrics.current_drawdown,
                'limit': self.max_drawdown_limit
            })
        
        if metrics.risk_level == "CRITICAL":
            logger.critical(f"CRITICAL RISK LEVEL: {metrics.warnings}")
        elif metrics.risk_level == "HIGH":
            logger.warning(f"HIGH RISK LEVEL: {metrics.warnings}")
    
    # Event Handlers
    def _on_position_update(self, data: Dict[str, Any]):
        """Handle Position Updates"""
        with self._lock:
            symbol = data.get('symbol')
            if symbol:
                self._current_positions[symbol] = {
                    'quantity': data.get('quantity', 0),
                    'avg_price': data.get('avg_price', 0),
                    'side': data.get('side', 'long'),
                    'entry_time': data.get('entry_time', datetime.now())
                }
    
    def _on_price_update(self, data: Dict[str, Any]):
        """Handle Price Updates"""
        symbol = data.get('symbol')
        price = data.get('price')
        
        if symbol and price:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=1000)
            self._price_history[symbol].append(price)
    
    def _on_portfolio_update(self, data: Dict[str, Any]):
        """Handle Portfolio Value Updates"""
        value = data.get('total_value')
        if value:
            with self._lock:
                self._current_portfolio_value = value
    
    def _on_trade_executed(self, data: Dict[str, Any]):
        """Handle Trade Execution"""
        pnl = data.get('realized_pnl')
        if pnl is not None:
            self._pnl_history.append(pnl)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Hole aktuellen Preis für Symbol"""
        if symbol in self._price_history and self._price_history[symbol]:
            return self._price_history[symbol][-1]
        return None
    
    # Public API
    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Hole aktuelle Risikometriken"""
        # Cache für 1 Sekunde
        if self._last_metrics and (time.time() - self._metrics_cache_time) < 1.0:
            return self._last_metrics
        
        return self._calculate_risk_metrics()
    
    def add_risk_callback(self, callback: Callable[[RiskMetrics], None]):
        """Registriere Risk Callback"""
        self._risk_callbacks.append(callback)
    
    def remove_risk_callback(self, callback: Callable[[RiskMetrics], None]):
        """Entferne Risk Callback"""
        if callback in self._risk_callbacks:
            self._risk_callbacks.remove(callback)
    
    def update_position(self, symbol: str, quantity: float, avg_price: float, side: str = 'long'):
        """Manuelles Position Update"""
        global_event_bus.publish("position_update", {
            'symbol': symbol,
            'quantity': quantity,
            'avg_price': avg_price,
            'side': side,
            'entry_time': datetime.now()
        })
    
    def update_price(self, symbol: str, price: float):
        """Manuelles Price Update"""
        global_event_bus.publish("price_update", {
            'symbol': symbol,
            'price': price
        })

# Singleton Instance
_risk_calculator_instance = None

def get_risk_calculator() -> RealTimeRiskCalculator:
    """Hole globale Risk Calculator Instance"""
    global _risk_calculator_instance
    if _risk_calculator_instance is None:
        _risk_calculator_instance = RealTimeRiskCalculator()
    return _risk_calculator_instance

def initialize_risk_monitoring(initial_capital: float):
    """Initialisiere Risk Monitoring"""
    calculator = get_risk_calculator()
    calculator.start_monitoring(initial_capital)
    logger.info(f"Risk monitoring initialized with {initial_capital} capital")

def shutdown_risk_monitoring():
    """Shutdown Risk Monitoring"""
    global _risk_calculator_instance
    if _risk_calculator_instance:
        _risk_calculator_instance.stop_monitoring()
        _risk_calculator_instance = None

if __name__ == "__main__":
    # Test Setup
    logging.basicConfig(level=logging.INFO)
    
    calc = RealTimeRiskCalculator(update_interval=0.5)
    calc.start_monitoring(initial_capital=10000)
    
    # Test Position Update
    calc.update_position("BTC/USDT", 0.1, 45000, "long")
    calc.update_price("BTC/USDT", 46000)
    
    time.sleep(2)
    
    metrics = calc.get_current_metrics()
    if metrics:
        print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
        print(f"Total P&L: {metrics.total_pnl:.2f}")
        print(f"Risk Level: {metrics.risk_level}")
    
    calc.stop_monitoring()