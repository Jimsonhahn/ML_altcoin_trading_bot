#!/usr/bin/env python3
"""
LightGBM Dependencies Fix
========================

Behebt LightGBM Import-Probleme professionell durch:
1. Lazy Loading für ML-Komponenten
2. Graceful Fallbacks wenn LightGBM fehlt
3. Alternative Implementierungen
"""
import os
import sys
import warnings
from typing import Optional, Any

class LightGBMWrapper:
    """
    Professioneller Wrapper für LightGBM mit Fallback-Mechanismen
    """
    
    def __init__(self):
        self._lgb = None
        self._available = False
        self._init_lightgbm()
    
    def _init_lightgbm(self):
        """Versuche LightGBM zu importieren mit Fallback"""
        try:
            import lightgbm as lgb
            self._lgb = lgb
            self._available = True
            print("✅ LightGBM successfully loaded")
        except ImportError as e:
            self._available = False
            warnings.warn(
                "LightGBM not available. ML features will use fallback implementations. "
                f"To fix: pip install lightgbm (Error: {e})",
                UserWarning
            )
        except OSError as e:
            self._available = False
            if "libomp.dylib" in str(e):
                warnings.warn(
                    "LightGBM OpenMP library missing. On macOS, run: brew install libomp",
                    UserWarning
                )
            else:
                warnings.warn(f"LightGBM system error: {e}", UserWarning)
    
    @property
    def available(self) -> bool:
        """Prüfe ob LightGBM verfügbar ist"""
        return self._available
    
    def get_lgb(self) -> Optional[Any]:
        """Hole LightGBM Modul oder None"""
        return self._lgb if self._available else None
    
    def create_model(self, **kwargs):
        """Erstelle LightGBM Model mit Fallback"""
        if self._available:
            return self._lgb.LGBMRegressor(**kwargs)
        else:
            # Fallback zu sklearn
            try:
                from sklearn.ensemble import RandomForestRegressor
                warnings.warn("Using RandomForestRegressor as LightGBM fallback", UserWarning)
                return RandomForestRegressor(n_estimators=100, random_state=42)
            except ImportError:
                raise ImportError(
                    "Neither LightGBM nor scikit-learn available. "
                    "Install with: pip install lightgbm scikit-learn"
                )

# Globale LightGBM Instance
_lgb_wrapper = LightGBMWrapper()

def get_lightgbm_wrapper() -> LightGBMWrapper:
    """Hole globale LightGBM Wrapper Instance"""
    return _lgb_wrapper

def is_lightgbm_available() -> bool:
    """Prüfe ob LightGBM verfügbar ist"""
    return _lgb_wrapper.available

def create_ml_model(**kwargs):
    """Factory für ML Models mit automatischem Fallback"""
    return _lgb_wrapper.create_model(**kwargs)

# Fix für existierende ML-Komponenten
def patch_ml_components():
    """Patche existierende ML-Komponenten für Fallback-Support"""
    
    # Patch ml_components/market_regime.py
    regime_path = 'ml_components/market_regime.py'
    if os.path.exists(regime_path):
        with open(regime_path, 'r') as f:
            content = f.read()
        
        # Füge Fallback-Import hinzu
        if 'import lightgbm' in content and 'LightGBMWrapper' not in content:
            fallback_import = """
# LightGBM mit Fallback
from fix_lightgbm_dependencies import get_lightgbm_wrapper, is_lightgbm_available
_lgb_wrapper = get_lightgbm_wrapper()

"""
            # Ersetze direkten LightGBM Import
            content = content.replace(
                'import lightgbm as lgb',
                fallback_import + '# import lightgbm as lgb  # Replaced with fallback'
            )
            
            # Ersetze LightGBM Usage
            content = content.replace(
                'lgb.LGBMRegressor',
                '_lgb_wrapper.create_model'
            )
            
            # Backup und Update
            with open(f"{regime_path}.backup", 'w') as f:
                f.write(open(regime_path, 'r').read())
            
            with open(regime_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Patched {regime_path} with LightGBM fallback")
    
    # Patch weitere ML-Files
    ml_files = [
        'ml_components/feature_extraction.py',
        'ml_components/market_sentiment.py',
        'strategies/ml_strategy.py'
    ]
    
    for ml_file in ml_files:
        if os.path.exists(ml_file):
            with open(ml_file, 'r') as f:
                content = f.read()
            
            if 'lightgbm' in content and 'LightGBMWrapper' not in content:
                # Füge Fallback-Header hinzu
                fallback_header = """# ML Model Fallback Support
from fix_lightgbm_dependencies import is_lightgbm_available, create_ml_model
import warnings

if not is_lightgbm_available():
    warnings.warn(f"LightGBM not available in {__file__}. Using fallback.", UserWarning)

"""
                content = fallback_header + content
                
                with open(ml_file, 'w') as f:
                    f.write(content)
                
                print(f"✅ Added fallback support to {ml_file}")

def create_ml_requirements():
    """Erstelle erweiterte Requirements mit Fallbacks"""
    
    requirements_ml = """# ML Dependencies mit Fallbacks
# Primäre ML Library
lightgbm>=3.3.0

# Fallback Libraries
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# Alternative ML Libraries (bei Bedarf)
xgboost>=1.5.0
catboost>=1.0.0

# Für macOS LightGBM Support
# Run: brew install libomp

# Für Linux LightGBM Support  
# Meist automatisch verfügbar

# Für Windows LightGBM Support
# Visual Studio Build Tools erforderlich
"""
    
    with open('requirements-ml-fallback.txt', 'w') as f:
        f.write(requirements_ml)
    
    print("✅ Created requirements-ml-fallback.txt")

def test_ml_setup():
    """Teste ML Setup mit allen Fallbacks"""
    print("\n🔍 Testing ML Setup...")
    
    # Test LightGBM
    wrapper = get_lightgbm_wrapper()
    print(f"LightGBM Available: {wrapper.available}")
    
    # Test Model Creation
    try:
        model = create_ml_model(n_estimators=10)
        print(f"✅ Model created: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
    
    # Test scikit-learn Fallback
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✅ Scikit-learn fallback available")
    except ImportError:
        print("❌ Scikit-learn fallback not available")
    
    # Test numpy/pandas
    try:
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas available")
    except ImportError as e:
        print(f"❌ NumPy/Pandas missing: {e}")

if __name__ == "__main__":
    print("🔧 Fixing LightGBM Dependencies...")
    
    # 1. Patch existierende ML-Komponenten
    patch_ml_components()
    
    # 2. Erstelle ML Requirements
    create_ml_requirements()
    
    # 3. Teste Setup
    test_ml_setup()
    
    print("\n✅ LightGBM Dependencies Fix completed!")
    print("📋 Next steps:")
    print("   1. For macOS: brew install libomp")
    print("   2. pip install -r requirements-ml-fallback.txt")
    print("   3. Test imports: python -c 'from fix_lightgbm_dependencies import test_ml_setup; test_ml_setup()'")