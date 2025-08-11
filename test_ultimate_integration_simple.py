#!/usr/bin/env python3
"""
Simple Ultimate BTC Strategy Integration Test
Direkter Test ohne komplexe Imports
"""

import sys
sys.path.append('.')

def test_file_structure():
    """Test ob alle Files vorhanden sind"""
    import os
    
    files_to_check = [
        'strategies/ultimate_btc_strategy.py',
        'strategies/__init__.py',
        'dashboard/src/components/StrategySelector.js'
    ]
    
    print("📁 FILE STRUCTURE TEST:")
    all_exist = True
    
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_strategy_registry():
    """Test ob Strategy in Registry eingetragen ist"""
    try:
        with open('strategies/__init__.py', 'r') as f:
            content = f.read()
        
        print("\n📋 STRATEGY REGISTRY TEST:")
        
        # Check import
        has_import = 'from .ultimate_btc_strategy import UltimateBTCStrategy' in content
        print(f"   {'✅' if has_import else '❌'} Import statement present")
        
        # Check registry entry
        has_registry = '"ultimate_btc": UltimateBTCStrategy' in content
        print(f"   {'✅' if has_registry else '❌'} Registry entry present")
        
        # Check position (should be first)
        is_first = content.find('"ultimate_btc"') < content.find('"super_lazy_billionaire"')
        print(f"   {'✅' if is_first else '❌'} Listed first in registry")
        
        return has_import and has_registry
        
    except Exception as e:
        print(f"   ❌ Error reading registry: {e}")
        return False

def test_dashboard_integration():
    """Test Dashboard Integration"""
    try:
        with open('dashboard/src/components/StrategySelector.js', 'r') as f:
            content = f.read()
        
        print("\n🖥️ DASHBOARD INTEGRATION TEST:")
        
        # Check icon handling
        has_icon = "case 'ultimate_btc':" in content
        print(f"   {'✅' if has_icon else '❌'} Ultimate BTC icon case present")
        
        # Check dropdown emoji
        has_emoji = "strategy.name === 'ultimate_btc' ? '🏆 '" in content
        print(f"   {'✅' if has_emoji else '❌'} Dropdown emoji present")
        
        # Check special display section
        has_display = "selectedStrategy === 'ultimate_btc'" in content
        print(f"   {'✅' if has_display else '❌'} Special display section present")
        
        # Check performance highlights
        has_performance = "177.8%" in content and "2.14" in content
        print(f"   {'✅' if has_performance else '❌'} Performance highlights present")
        
        return has_icon and has_emoji and has_display and has_performance
        
    except Exception as e:
        print(f"   ❌ Error reading dashboard file: {e}")
        return False

def test_api_endpoints():
    """Test API Endpoints"""
    try:
        with open('api/routes/strategies.py', 'r') as f:
            content = f.read()
        
        print("\n🌐 API ENDPOINTS TEST:")
        
        # Check strategies import
        has_import = "from strategies import STRATEGIES" in content
        print(f"   {'✅' if has_import else '❌'} STRATEGIES import present")
        
        # Check list endpoint
        has_list = "def list_strategies():" in content
        print(f"   {'✅' if has_list else '❌'} List strategies endpoint present")
        
        # Check detail endpoint  
        has_detail = "def get_strategy_details(strategy_name: str):" in content
        print(f"   {'✅' if has_detail else '❌'} Strategy details endpoint present")
        
        return has_import and has_list and has_detail
        
    except Exception as e:
        print(f"   ❌ Error reading API file: {e}")
        return False

def main():
    """Haupttest"""
    print("🏆 ULTIMATE BTC STRATEGY - SIMPLE INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Strategy Registry", test_strategy_registry), 
        ("Dashboard Integration", test_dashboard_integration),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n🎯 INTEGRATION TEST RESULTS:")
    print(f"   Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
    
    if passed == total:
        print("\n🎉 INTEGRATION COMPLETE!")
        print("✅ Ultimate BTC Strategy erfolgreich in Dashboard integriert!")
        print("\n📋 NÄCHSTE SCHRITTE:")
        print("   1. API Server starten: python api/app.py")
        print("   2. Dashboard starten: cd dashboard && npm start")
        print("   3. Ultimate BTC Strategy aus Dropdown wählen")
        print("   4. Parameter konfigurieren")
        print("   5. Paper-Trading starten")
    else:
        print(f"\n⚠️ {total-passed} TESTS FAILED")
        print("Integration noch nicht vollständig abgeschlossen.")

if __name__ == "__main__":
    main()