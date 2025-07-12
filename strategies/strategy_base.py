"""
Base Strategy Class
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Any
import pandas as pd

class Signal(Enum):
    """Trading Signal Enum"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Strategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate_signal(self, symbol: str, data: pd.DataFrame, 
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate trading signal

        Returns:
            Tuple of (signal_string, signal_data_dict)
            Example: ('BUY', {'confidence': 0.8, 'reason': 'momentum'})
        """
        pass
