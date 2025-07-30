"""
Strategies Package - Konsolidierte Strategie-Registry
"""

import logging
from typing import Dict, Type, Optional

# Basis-Strategieklasse
from .strategy_base import Strategy

logger = logging.getLogger(__name__)

# Registry für alle verfügbaren Strategien
STRATEGIES: Dict[str, Type[Strategy]] = {}

def register_strategy(name: str, strategy_class: Type[Strategy]):
    """
    Registriert eine Strategie in der globalen Registry
    
    Args:
        name: Name der Strategie
        strategy_class: Strategieklasse
    """
    STRATEGIES[name] = strategy_class
    logger.debug(f"Strategie registriert: {name}")

def get_strategy(name: str) -> Optional[Type[Strategy]]:
    """
    Holt eine Strategie aus der Registry
    
    Args:
        name: Name der Strategie
        
    Returns:
        Strategieklasse oder None wenn nicht gefunden
    """
    return STRATEGIES.get(name)

def list_strategies() -> list:
    """
    Gibt Liste aller verfügbaren Strategien zurück
    
    Returns:
        Liste der Strategienamen
    """
    return list(STRATEGIES.keys())

# Lade alle verfügbaren Strategien
def _load_strategies():
    """Lädt alle verfügbaren Strategien und registriert sie"""
    strategies_to_load = [
        ('momentum', 'MomentumStrategy'),
        ('mean_reversion', 'MeanReversionStrategy'),
        ('arbitrage', 'ArbitrageStrategy'),
        ('grid_trading', 'GridTradingStrategy'),
        ('defi_yield', 'DeFiYieldStrategy'),
        ('ultimate_btc', 'UltimateBtcStrategy'),
        ('profitable_btc', 'ProfitableBtcStrategy'),
        ('lazy_billionaire', 'LazyBillionaireStrategy'),
        ('super_lazy_billionaire', 'SuperLazyBillionaireStrategy'),
        ('ml_strategy', 'MLStrategy'),
        ('autopilot', 'AutopilotStrategy'),
        ('copy_trading', 'CopyTradingStrategy'),
        ('liquidation', 'LiquidationStrategy'),
        # Neue defensive Strategien
        ('advanced_portfolio', 'AdvancedPortfolioStrategy'),
        ('defensive_volatility', 'DefensiveVolatilityStrategy'),
        ('smart_rebalancing', 'SmartRebalancingStrategy'),
        # Candle momentum strategy
        ('candle_momentum', 'CandleMomentumStrategy'),
        # Exact TradingView candle body momentum strategy
        ('candle_body_momentum', 'CandleBodyMomentumStrategy'),
        # Optimized candle momentum strategy  
        ('optimized_candle_momentum', 'OptimizedCandleMomentumStrategy'),
        # High-risk daily trading strategy
        ('high_risk_daily', 'HighRiskDailyStrategy')
    ]
    
    for strategy_name, class_name in strategies_to_load:
        try:
            # Import des Strategie-Moduls
            if strategy_name == 'optimized_candle_momentum':
                module = __import__(f'strategies.optimized_candle_momentum', fromlist=[class_name])
            elif strategy_name == 'high_risk_daily':
                module = __import__(f'strategies.high_risk_daily', fromlist=[class_name])
            else:
                module = __import__(f'strategies.{strategy_name}', fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            
            # Validierung der Strategieklasse
            if not issubclass(strategy_class, Strategy):
                logger.warning(f"Klasse {class_name} ist keine gültige Strategie-Subklasse")
                continue
            
            # Registrierung in der Registry
            register_strategy(strategy_name, strategy_class)
            
        except ImportError as e:
            logger.debug(f"Strategie {strategy_name} nicht verfügbar: {e}")
        except AttributeError as e:
            logger.warning(f"Klasse {class_name} nicht gefunden in {strategy_name}: {e}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Strategie {strategy_name}: {e}")

# Strategien beim Import laden
_load_strategies()

# Log verfügbare Strategien
logger.info(f"Verfügbare Strategien: {list_strategies()}")

# Exports
__all__ = [
    'Strategy',
    'STRATEGIES',
    'register_strategy',
    'get_strategy',
    'list_strategies'
]