#!/usr/bin/env python3
"""
Bot Import-Audit Script
Testet systematisch alle kritischen Imports und Abhängigkeiten
"""
import os
import sys
import importlib.util
from pathlib import Path

def test_all_imports():
    """Teste alle kritischen Imports"""
    failed_imports = []
    successful_imports = []
    
    # Kritische Module zum Testen
    critical_modules = [
        'main',
        'core.trading_bot',
        'core.strategy_router', 
        'core.market_analyzer',
        'core.exchange',
        'strategies.super_lazy_billionaire_strategy',
        'strategies.momentum',
        'strategies.mean_reversion',
        'strategies.ml_strategy',
        'utils.logger',
        'utils.notifier',
        'config.settings',
        'data_sources.data_manager',
        'ml_components.market_regime'
    ]
    
    print("🔍 Testing Critical Imports...")
    print("=" * 50)
    
    for module in critical_modules:
        try:
            # Versuche direkten Import
            __import__(module)
            successful_imports.append(module)
            print(f"✅ {module}")
        except ImportError as e:
            failed_imports.append(f"{module}: {str(e)}")
            print(f"❌ {module}: {str(e)}")
        except Exception as e:
            failed_imports.append(f"{module}: {str(e)}")
            print(f"⚠️  {module}: {str(e)}")
    
    return failed_imports, successful_imports

def check_init_files():
    """Prüfe __init__.py Files"""
    print("\n🔍 Checking __init__.py Files...")
    print("=" * 50)
    
    required_dirs = [
        'core', 'strategies', 'utils', 'config', 
        'data_sources', 'ml_components', 'tests'
    ]
    
    missing_inits = []
    existing_inits = []
    
    for dir_name in required_dirs:
        init_path = Path(dir_name) / '__init__.py'
        if init_path.exists():
            existing_inits.append(str(init_path))
            print(f"✅ {init_path}")
        else:
            missing_inits.append(str(init_path))
            print(f"❌ Missing: {init_path}")
    
    return missing_inits, existing_inits

def check_circular_dependencies():
    """Suche nach zirkulären Abhängigkeiten"""
    print("\n🔍 Checking for Circular Dependencies...")
    print("=" * 50)
    
    # Einfache Heuristik: Suche nach gegenseitigen Imports
    import_map = {}
    potential_cycles = []
    
    # Durchsuche alle Python-Files
    for root, dirs, files in os.walk('.'):
        # Skip bestimmte Verzeichnisse
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'build']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extrahiere Imports
                    imports = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('from ') and ' import ' in line:
                            module = line.split('from ')[1].split(' import')[0].strip()
                            imports.append(module)
                        elif line.startswith('import '):
                            module = line.split('import ')[1].split(' as ')[0].split('.')[0].strip()
                            imports.append(module)
                    
                    # Speichere relative Module
                    relative_path = file_path.replace('./', '').replace('.py', '').replace('/', '.')
                    import_map[relative_path] = [imp for imp in imports if not imp.startswith('_') and '.' in imp]
                    
                except Exception as e:
                    continue
    
    # Suche nach potentiellen Zyklen
    for module, imports in import_map.items():
        for imported in imports:
            if imported in import_map:
                # Prüfe ob imported module zurück zu uns importiert
                if any(imp.startswith(module.split('.')[0]) for imp in import_map[imported]):
                    potential_cycles.append(f"{module} <-> {imported}")
    
    if potential_cycles:
        print("⚠️  Potential Circular Dependencies:")
        for cycle in potential_cycles:
            print(f"   {cycle}")
    else:
        print("✅ No obvious circular dependencies found")
    
    return potential_cycles

def check_main_py_strategy_factory():
    """Prüfe Strategy Factory in main.py"""
    print("\n🔍 Checking Strategy Factory...")
    print("=" * 50)
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        checks = {
            'has_strategy_import': 'from strategies' in content,
            'has_strategy_mapping': 'strategies' in content and '{' in content,
            'has_lazy_billionaire': 'lazy_billionaire' in content or 'super_lazy_billionaire' in content,
            'has_momentum': 'momentum' in content,
            'has_mean_reversion': 'mean_reversion' in content
        }
        
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check}: {result}")
        
        return checks
        
    except FileNotFoundError:
        print("❌ main.py not found!")
        return {'main_py_exists': False}

def generate_dependency_report():
    """Generiere Abhängigkeits-Report"""
    print("\n" + "="*60)
    print("📊 DEPENDENCY AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    failed_imports, successful_imports = test_all_imports()
    missing_inits, existing_inits = check_init_files()
    potential_cycles = check_circular_dependencies()
    strategy_factory = check_main_py_strategy_factory()
    
    # Berechne Score
    total_score = 0
    max_score = 100
    
    # Import Score (40 Punkte)
    import_score = (len(successful_imports) / (len(successful_imports) + len(failed_imports))) * 40 if (successful_imports or failed_imports) else 0
    total_score += import_score
    
    # Init Files Score (20 Punkte)  
    init_score = (len(existing_inits) / (len(existing_inits) + len(missing_inits))) * 20 if (existing_inits or missing_inits) else 0
    total_score += init_score
    
    # Circular Dependencies Score (20 Punkte)
    cycle_score = 20 if len(potential_cycles) == 0 else max(0, 20 - len(potential_cycles) * 5)
    total_score += cycle_score
    
    # Strategy Factory Score (20 Punkte)
    factory_score = sum(strategy_factory.values()) * 4 if strategy_factory else 0
    total_score += factory_score
    
    print(f"\n📊 FINAL SCORE: {total_score:.1f}/100")
    print(f"   Import Health: {import_score:.1f}/40")
    print(f"   Init Files: {init_score:.1f}/20") 
    print(f"   No Cycles: {cycle_score:.1f}/20")
    print(f"   Strategy Factory: {factory_score:.1f}/20")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if failed_imports:
        print("   🔧 Fix failed imports first (critical)")
    if missing_inits:
        print("   🔧 Create missing __init__.py files")
    if potential_cycles:
        print("   🔧 Resolve circular dependencies")
    if factory_score < 15:
        print("   🔧 Improve strategy factory in main.py")
    
    return {
        'total_score': total_score,
        'failed_imports': failed_imports,
        'missing_inits': missing_inits,
        'potential_cycles': potential_cycles,
        'strategy_factory_issues': {k: v for k, v in strategy_factory.items() if not v}
    }

if __name__ == "__main__":
    # Führe Audit durch
    report = generate_dependency_report()
    
    # Speichere Report
    import json
    with open('import_audit_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: import_audit_report.json")