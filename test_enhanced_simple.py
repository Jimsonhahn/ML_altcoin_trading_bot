#!/usr/bin/env python3
"""
🧪 Test Enhanced Decision Logger - Simple Version
Ohne Dependencies auf dein komplettes Bot-System
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock Database Pool für Testing
class MockDBPool:
    async def acquire(self):
        return MockDBConnection()
    
    async def close(self):
        pass

class MockDBConnection:
    async def fetchrow(self, query, *args):
        return None
    
    async def fetch(self, query, *args):
        return []
    
    async def execute(self, query, *args):
        pass
    
    def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

async def test_enhanced_logger():
    """Test Enhanced Decision Logger komplett isoliert"""
    print("🧪 Testing Enhanced Decision Logger (Simple Version)...")
    
    try:
        # Create export directory
        export_path = Path("intelligence_exports_test")
        export_path.mkdir(exist_ok=True)
        
        # Import Enhanced Logger
        from core.enhanced_decision_logger import EnhancedDecisionLogger
        
        # Mock database pool
        mock_pool = MockDBPool()
        
        # Create enhanced logger
        enhanced_logger = EnhancedDecisionLogger(
            db_pool=mock_pool,
            export_path=str(export_path),
            dashboard_updates=True,
            learning_enabled=True
        )
        
        print("✅ Enhanced Logger erstellt")
        
        # Start logger
        await enhanced_logger.start()
        print("✅ Logger gestartet")
        
        # Test decision logging
        test_decision_data = {
            'strategy': 'momentum_strategy',
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 45000.0,
            'reasoning': 'Strong momentum signals detected'
        }
        
        market_context = {
            'regime': 'bull_market',
            'volatility': 0.15,
            'volume': 'high',
            'sentiment': 'positive',
            'rsi': 65.5,
            'macd_signal': 'bullish'
        }
        
        decision_id = await enhanced_logger.log_trading_decision_with_context(
            decision_data=test_decision_data,
            market_context=market_context,
            strategy_reasoning="Strong momentum detected with RSI > 60 and bullish MACD",
            confidence_level=0.85
        )
        
        print(f"✅ Decision geloggt: {decision_id}")
        
        # Test metrics generation
        metrics = await enhanced_logger.get_dashboard_metrics()
        print(f"✅ Metrics generiert: {len(metrics)} categories")
        
        # Test anomaly detection
        anomalies = await enhanced_logger.detect_trading_anomalies()
        print(f"✅ Anomaly detection: {len(anomalies)} anomalies")
        
        # Test learning report
        report = await enhanced_logger.generate_ai_learning_report(days=1)
        print(f"✅ Learning report: {len(report)} sections")
        
        # Check exported files
        exported_files = list(export_path.glob("*.json*"))
        print(f"✅ Exported files: {len(exported_files)} files")
        
        for file in exported_files:
            print(f"   📄 {file.name} ({file.stat().st_size} bytes)")
        
        # Stop logger
        await enhanced_logger.stop()
        print("✅ Logger gestoppt")
        
        print("🎉 Enhanced Logger Test ERFOLGREICH!")
        return True
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_intelligence_api_routes():
    """Test Intelligence API Routes"""
    print("\n🧪 Testing Intelligence API Routes...")
    
    try:
        from api.routes.intelligence import intelligence_bp
        print("✅ Intelligence Blueprint importiert")
        
        # Check available routes
        rules = []
        for rule in intelligence_bp.url_map.iter_rules():
            rules.append(f"{rule.methods} {rule.rule}")
        
        print(f"✅ API Routes: {len(rules)} endpoints")
        for rule in rules[:5]:  # Show first 5
            print(f"   🔗 {rule}")
        
        print("✅ API Routes Test erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ API Routes Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test dass alle Dateien existieren"""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        'core/enhanced_decision_logger.py',
        'api/routes/intelligence.py',
        'integration_patch_main.py',
        'api_integration_patch.py'
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} FEHLT!")
            all_good = False
    
    return all_good

async def main():
    """Haupttest-Funktion"""
    print("🚀 Enhanced Integration Test Suite")
    print("=" * 50)
    
    # Test 1: File Structure
    print("\n1️⃣ FILE STRUCTURE TEST")
    files_ok = test_file_structure()
    
    # Test 2: Enhanced Logger
    print("\n2️⃣ ENHANCED LOGGER TEST")
    logger_ok = await test_enhanced_logger() if files_ok else False
    
    # Test 3: API Routes
    print("\n3️⃣ INTELLIGENCE API TEST")
    api_ok = await test_intelligence_api_routes() if files_ok else False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY:")
    print(f"   Files: {'✅' if files_ok else '❌'}")
    print(f"   Enhanced Logger: {'✅' if logger_ok else '❌'}")
    print(f"   Intelligence API: {'✅' if api_ok else '❌'}")
    
    if files_ok and logger_ok and api_ok:
        print("\n🎉 ALLE TESTS ERFOLGREICH! 🎉")
        print("Du kannst jetzt mit der Integration fortfahren!")
        return True
    else:
        print("\n⚠️  EINIGE TESTS FEHLGESCHLAGEN")
        print("Bitte behebe die Probleme bevor du weiter machst.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)