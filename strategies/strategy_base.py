"""
Base Strategy Class und Signal Enum
Definiert die Basis für alle Trading-Strategien
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading Signal Enum"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(ABC):
    """Abstrakte Basisklasse für alle Trading-Strategien"""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.name = self.__class__.__name__
        self.max_position_size = self.params.get('max_position_size', 0.1)
        self.stop_loss = self.params.get('stop_loss', 0.05)
        self.take_profit = self.params.get('take_profit', 0.1)
        self.trades = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0
        }

    @abstractmethod
    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        """Berechnet das Trading-Signal"""
        pass

    def calculate_position_size(self, symbol: str, signal: Signal,
                               confidence: float, current_price: float,
                               balance: float) -> float:
        """Berechnet die Positionsgröße"""
        if signal == Signal.HOLD:
            return 0.0
        base_position = balance * self.max_position_size * confidence
        return min(base_position, balance * 0.95)
