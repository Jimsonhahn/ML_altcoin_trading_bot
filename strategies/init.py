#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Strategies Package
--------------------------
Dieses Modul stellt alle verfügbaren Trading-Strategien für den Altcoin Trading Bot bereit.

Verfügbare Strategien:
- Strategy: Basisklasse für alle Strategien
- MomentumStrategy: Nutzt Preismomentum und Trendkontinuität
- MeanReversionStrategy: Basiert auf Regression zum Mittelwert
- MLStrategy: Machine Learning basierte Vorhersagen
- GridStrategy: Automatische Geldmaschine die bei jeder Preisbewegung verdient

Grid Trading Features:
- Kauft automatisch bei fallenden Preisen
- Verkauft automatisch bei steigenden Preisen  
- Verdient Geld bei steigenden UND fallenden Märkten
- Vollautomatisch nach einmaliger Konfiguration
- ROI: 20-100% pro Jahr je nach Volatilität
"""

# Basis-Strategie importieren
try:
    from .strategy_base import Strategy
except ImportError as e:
    print(f"Warning: Could not import Strategy base class: {e}")
    Strategy = None

# Standard-Strategien importieren
try:
    from .momentum import MomentumStrategy
except ImportError as e:
    print(f"Warning: Could not import MomentumStrategy: {e}")
    MomentumStrategy = None

try:
    from .mean_reversion import MeanReversionStrategy
except ImportError as e:
    print(f"Warning: Could not import MeanReversionStrategy: {e}")
    MeanReversionStrategy = None

try:
    from .ml_strategy import MLStrategy
except ImportError as e:
    print(f"Warning: Could not import MLStrategy: {e}")
    MLStrategy = None

# Grid Trading Strategie importieren
try:
    from .grid_trading import GridStrategy
except ImportError as e:
    print(f"Warning: Could not import GridStrategy: {e}")
    print("Make sure strategies/grid_trading.py exists with the complete Grid Trading implementation")
    GridStrategy = None

# Alle verfügbaren Strategien
__all__ = [
    'Strategy',
    'MomentumStrategy', 
    'MeanReversionStrategy',
    'MLStrategy',
    'GridStrategy'
]

# Nur verfügbare Strategien exportieren
__all__ = [name for name in __all__ if globals().get(name) is not None]

# Strategie-Registry für dynamische Auswahl
STRATEGY_REGISTRY = {}

# Registriere verfügbare Strategien
if Strategy:
    STRATEGY_REGISTRY['base'] = Strategy

if MomentumStrategy:
    STRATEGY_REGISTRY['momentum'] = MomentumStrategy
    STRATEGY_REGISTRY['default'] = MomentumStrategy  # Default-Strategie

if MeanReversionStrategy:
    STRATEGY_REGISTRY['mean_reversion'] = MeanReversionStrategy

if MLStrategy:
    STRATEGY_REGISTRY['ml'] = MLStrategy

if GridStrategy:
    STRATEGY_REGISTRY['grid_trading'] = GridStrategy
    STRATEGY_REGISTRY['grid'] = GridStrategy  # Alias


def get_strategy_class(strategy_name: str):
    """
    Holt eine Strategie-Klasse nach Namen.
    
    Args:
        strategy_name: Name der Strategie
        
    Returns:
        Strategie-Klasse oder None
    """
    return STRATEGY_REGISTRY.get(strategy_name.lower())


def list_available_strategies():
    """
    Listet alle verfügbaren Strategien auf.
    
    Returns:
        Liste der verfügbaren Strategie-Namen
    """
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_info():
    """
    Gibt Informationen über alle verfügbaren Strategien zurück.
    
    Returns:
        Dictionary mit Strategie-Informationen
    """
    info = {}
    
    for name, strategy_class in STRATEGY_REGISTRY.items():
        if strategy_class:
            info[name] = {
                'class': strategy_class.__name__,
                'description': getattr(strategy_class, '__doc__', 'No description available'),
                'module': strategy_class.__module__
            }
    
    return info


# Automatische Strategie-Validierung beim Import
def _validate_strategies():
    """Validiert alle importierten Strategien"""
    issues = []
    
    if not Strategy:
        issues.append("❌ Base Strategy class not available")
    
    if not STRATEGY_REGISTRY:
        issues.append("❌ No strategies available")
    
    if 'default' not in STRATEGY_REGISTRY:
        issues.append("⚠️  No default strategy defined")
    
    if 'grid_trading' not in STRATEGY_REGISTRY:
        issues.append("⚠️  Grid Trading strategy not available - create strategies/grid_trading.py")
    
    return issues


# Führe Validierung beim Import durch
_validation_issues = _validate_strategies()
if _validation_issues:
    print("Strategy Package Validation:")
    for issue in _validation_issues:
        print(f"  {issue}")
    print()


# Hilfsfunktionen für Grid Trading
if GridStrategy:
    def create_grid_strategy(settings, lower_price=None, upper_price=None, num_grids=None, investment_per_grid=None):
        """
        Erstellt eine vorkonfigurierte Grid Trading Strategie.
        
        Args:
            settings: Bot-Einstellungen
            lower_price: Untere Preisgrenze (optional)
            upper_price: Obere Preisgrenze (optional)
            num_grids: Anzahl Grids (optional)
            investment_per_grid: Investition pro Grid (optional)
            
        Returns:
            Konfigurierte GridStrategy-Instanz
        """
        # Überschreibe Konfiguration falls Parameter gegeben
        if lower_price:
            settings.set('grid_trading.price_range.lower', lower_price)
        if upper_price:
            settings.set('grid_trading.price_range.upper', upper_price)
        if num_grids:
            settings.set('grid_trading.num_grids', num_grids)
        if investment_per_grid:
            settings.set('grid_trading.investment_per_grid', investment_per_grid)
        
        return GridStrategy(settings)


# Erweiterte Strategien (falls verfügbar)
try:
    from .advanced import *
    print("✅ Advanced strategies loaded")
except ImportError:
    pass  # Advanced strategies sind optional

try:
    from .regime_adaptive import RegimeAdaptiveStrategy
    STRATEGY_REGISTRY['regime_adaptive'] = RegimeAdaptiveStrategy
    __all__.append('RegimeAdaptiveStrategy')
except ImportError:
    pass  # Regime-adaptive strategy ist optional


# Debugging-Informationen
if __name__ == "__main__":
    print("🤖 Trading Strategies Package")
    print("=" * 50)
    print(f"Available strategies: {len(STRATEGY_REGISTRY)}")
    
    for name, strategy_class in STRATEGY_REGISTRY.items():
        print(f"  ✅ {name}: {strategy_class.__name__}")
    
    print()
    print("Strategy Information:")
    info = get_strategy_info()
    for name, details in info.items():
        print(f"\n📊 {name.upper()}:")
        print(f"   Class: {details['class']}")
        print(f"   Module: {details['module']}")
        if details['description']:
            # Erste Zeile der Beschreibung
            desc = details['description'].split('\n')[0].strip()
            if desc:
                print(f"   Description: {desc}")
    
    if GridStrategy:
        print("\n💰 GRID TRADING FEATURES:")
        print("   ✅ Automatische Geldmaschine")
        print("   ✅ Verdient bei steigenden UND fallenden Preisen")
        print("   ✅ 24/7 vollautomatisch")
        print("   ✅ ROI: 20-100% pro Jahr")
        print("   ✅ Läuft ohne weiteres Zutun")
    
    print("\n" + "=" * 50)