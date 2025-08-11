"""
Trading Bot Interfaces
=====================

Abstrakte Interfaces zur Vermeidung zirkulärer Dependencies
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd

class ITradingBot(ABC):
    """Interface für Trading Bot"""
    
    @abstractmethod
    def get_current_capital(self) -> float:
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        pass
    
    @abstractmethod
    def cancel_all_orders(self):
        pass
    
    @abstractmethod
    def stop_trading(self):
        pass

class ISafetyManager(ABC):
    """Interface für Safety Manager"""
    
    @abstractmethod
    def check_drawdown(self) -> bool:
        pass
    
    @abstractmethod
    def emergency_stop(self, reason: str):
        pass
    
    @abstractmethod
    def is_safe_to_trade(self) -> bool:
        pass

class IDataManager(ABC):
    """Interface für Data Manager"""
    
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str) -> Dict:
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        pass

class IStrategy(ABC):
    """Interface für Trading Strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass

class IMarketAnalyzer(ABC):
    """Interface für Market Analyzer"""
    
    @abstractmethod
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_market_regime(self) -> str:
        pass

class IRiskManager(ABC):
    """Interface für Risk Manager"""
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: float, capital: float) -> float:
        pass
    
    @abstractmethod
    def check_risk_limits(self, position: Dict) -> bool:
        pass

class IPositionManager(ABC):
    """Interface für Position Manager"""
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    def update_position(self, symbol: str, data: Dict):
        pass

class IOrderManager(ABC):
    """Interface für Order Manager"""
    
    @abstractmethod
    def place_order(self, order: Dict) -> Dict:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

class IStrategyRouter(ABC):
    """Interface für Strategy Router"""
    
    @abstractmethod
    def get_current_strategy(self) -> str:
        pass
    
    @abstractmethod
    def switch_strategy(self, new_strategy: str):
        pass
    
    @abstractmethod
    def get_allocation_weights(self) -> Dict[str, float]:
        pass

class EventBus:
    """Event Bus für lose gekoppelte Kommunikation"""
    
    def __init__(self):
        self._subscribers = {}
    
    def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Event handler error: {e}")
    
    def unsubscribe(self, event_type: str, callback):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

# Globaler Event Bus
global_event_bus = EventBus()
