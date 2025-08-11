"""
Core Trading Bot Components
===========================

Zentrale Module für den Trading Bot:
- TradingBot: Hauptklasse für Trading-Operations
- StrategyRouter: Intelligente Strategieverteilung
- MarketAnalyzer: Marktanalyse und Regime-Erkennung
- RiskManager: Risikomanagement und Sicherheit
- SafetyManager: Notfall- und Drawdown-Schutz
"""

# Hauptkomponenten
from .trading_bot import TradingBot
from .strategy_router import StrategyRouter
from .market_analyzer import MarketAnalyzer
from .risk_manager import RiskManager
from .safety_manager import SafetyManager

# Exchange und Data Management
from .exchange import ExchangeManager
# Backward compatibility alias
Exchange = ExchangeManager
from .position import Position

# Backtesting
from .backtest_engine import BacktestEngine

__version__ = "1.0.0"
__all__ = [
    'TradingBot',
    'StrategyRouter', 
    'MarketAnalyzer',
    'RiskManager',
    'SafetyManager',
    'ExchangeManager',
    'Exchange',  # Backward compatibility
    'Position',
    'BacktestEngine'
]