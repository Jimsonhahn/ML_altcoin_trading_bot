#!/usr/bin/env python3
"""
Safety Systems Audit
Prüft Drawdown-Schutz, Emergency Stop und Sicherheitssysteme
"""
import os
import json
from pathlib import Path

def check_risk_protection():
    """Prüfe Risikoschutz-Systeme"""
    print("🔍 Checking Risk Protection Systems...")
    print("=" * 50)
    
    protection_features = {
        'has_emergency_stop': False,
        'has_drawdown_limits': False,
        'has_api_fallback': False,
        'has_killswitch': False,
        'has_stop_loss': False,
        'has_position_limits': False,
        'has_timeout_protection': False
    }
    
    safety_files_found = []
    critical_features = []
    
    # Suche nach Sicherheitsfeatures in allen Python-Files
    for root, dirs, files in os.walk('.'):
        # Skip bestimmte Verzeichnisse
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'build', '.venv']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    # Emergency Stop
                    if 'emergency_stop' in content or 'emergency stop' in content:
                        protection_features['has_emergency_stop'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Emergency stop in {file_path}")
                    
                    # Drawdown Limits
                    if ('max_drawdown' in content and 'stop' in content) or 'drawdown_limit' in content:
                        protection_features['has_drawdown_limits'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Drawdown limits in {file_path}")
                    
                    # API Fallback
                    if ('api' in content and 'fallback' in content) or ('connection' in content and 'lost' in content):
                        protection_features['has_api_fallback'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"API fallback in {file_path}")
                    
                    # Killswitch
                    if 'killswitch' in content or 'kill_switch' in content or 'force_stop' in content:
                        protection_features['has_killswitch'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Killswitch in {file_path}")
                    
                    # Stop Loss
                    if 'stop_loss' in content or 'stoploss' in content:
                        protection_features['has_stop_loss'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Stop loss in {file_path}")
                    
                    # Position Limits
                    if 'max_position' in content or 'position_limit' in content:
                        protection_features['has_position_limits'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Position limits in {file_path}")
                    
                    # Timeout Protection
                    if ('timeout' in content and 'protect' in content) or 'connection_timeout' in content:
                        protection_features['has_timeout_protection'] = True
                        if file_path not in safety_files_found:
                            safety_files_found.append(file_path)
                            critical_features.append(f"Timeout protection in {file_path}")
                
                except Exception as e:
                    continue
    
    # Ausgabe der Ergebnisse
    print(f"\n📄 Found {len(safety_files_found)} files with safety features:")
    for file_path in safety_files_found:
        print(f"   📁 {file_path}")
    
    print(f"\n🔍 Safety Features Analysis:")
    for feature, found in protection_features.items():
        status = "✅" if found else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    return protection_features, critical_features

def check_safety_manager():
    """Prüfe spezifisch den Safety Manager"""
    print("\n🔍 Checking Safety Manager Implementation...")
    print("=" * 50)
    
    safety_manager_analysis = {
        'has_safety_manager': False,
        'has_real_time_monitoring': False,
        'has_automatic_shutdown': False,
        'has_position_liquidation': False,
        'safety_manager_features': []
    }
    
    safety_manager_path = 'core/safety_manager.py'
    
    if os.path.exists(safety_manager_path):
        safety_manager_analysis['has_safety_manager'] = True
        print(f"✅ Safety Manager found: {safety_manager_path}")
        
        with open(safety_manager_path, 'r') as f:
            content = f.read()
        
        # Analyze Safety Manager Features
        features = {
            'real_time_monitoring': ['monitor', 'real_time', 'continuous'],
            'automatic_shutdown': ['shutdown', 'stop_bot', 'terminate'],
            'position_liquidation': ['liquidate', 'close_positions', 'close_all'],
            'drawdown_check': ['drawdown', 'loss_limit', 'max_loss'],
            'api_health_check': ['api_health', 'connection_check', 'ping'],
            'emergency_protocols': ['emergency', 'critical', 'urgent']
        }
        
        found_features = []
        for feature_name, keywords in features.items():
            if any(keyword.lower() in content.lower() for keyword in keywords):
                found_features.append(feature_name)
                print(f"✅ {feature_name.replace('_', ' ').title()}")
            else:
                print(f"❌ {feature_name.replace('_', ' ').title()}")
        
        safety_manager_analysis['safety_manager_features'] = found_features
        
        # Spezifische Checks
        if 'real_time' in content.lower() or 'monitor' in content.lower():
            safety_manager_analysis['has_real_time_monitoring'] = True
        
        if 'shutdown' in content.lower() or 'stop' in content.lower():
            safety_manager_analysis['has_automatic_shutdown'] = True
        
        if 'liquidate' in content.lower() or 'close_positions' in content.lower():
            safety_manager_analysis['has_position_liquidation'] = True
    
    else:
        print(f"❌ Safety Manager not found: {safety_manager_path}")
    
    return safety_manager_analysis

def check_risk_manager_integration():
    """Prüfe Risk Manager Integration"""
    print("\n🔍 Checking Risk Manager Integration...")
    print("=" * 50)
    
    risk_integration = {
        'has_risk_manager': False,
        'integrated_with_trading_bot': False,
        'has_risk_metrics': False,
        'has_real_time_risk_calc': False
    }
    
    risk_manager_path = 'core/risk_manager.py'
    
    if os.path.exists(risk_manager_path):
        risk_integration['has_risk_manager'] = True
        print(f"✅ Risk Manager found: {risk_manager_path}")
        
        with open(risk_manager_path, 'r') as f:
            risk_content = f.read()
        
        # Check für Risk Metrics
        risk_metrics = ['var', 'sharpe', 'drawdown', 'volatility', 'risk_per_trade']
        found_metrics = [metric for metric in risk_metrics if metric in risk_content.lower()]
        
        if found_metrics:
            risk_integration['has_risk_metrics'] = True
            print(f"✅ Risk metrics found: {', '.join(found_metrics)}")
        else:
            print("❌ No risk metrics found")
        
        # Check für Real-time calculation
        if 'real_time' in risk_content.lower() or 'continuous' in risk_content.lower():
            risk_integration['has_real_time_risk_calc'] = True
            print("✅ Real-time risk calculation")
        else:
            print("❌ No real-time risk calculation")
        
        # Check Integration mit Trading Bot
        if os.path.exists('core/trading_bot.py'):
            with open('core/trading_bot.py', 'r') as f:
                bot_content = f.read()
            
            if 'risk_manager' in bot_content.lower():
                risk_integration['integrated_with_trading_bot'] = True
                print("✅ Integrated with Trading Bot")
            else:
                print("❌ Not integrated with Trading Bot")
    else:
        print(f"❌ Risk Manager not found: {risk_manager_path}")
    
    return risk_integration

def check_emergency_procedures():
    """Prüfe Emergency Procedures"""
    print("\n🔍 Checking Emergency Procedures...")
    print("=" * 50)
    
    emergency_procedures = {
        'has_graceful_shutdown': False,
        'has_position_backup': False,
        'has_state_recovery': False,
        'has_notification_system': False,
        'has_manual_override': False
    }
    
    # Check für Emergency Procedures in verschiedenen Files
    emergency_keywords = {
        'graceful_shutdown': ['graceful', 'shutdown', 'cleanup'],
        'position_backup': ['backup', 'save_state', 'persist'],
        'state_recovery': ['recovery', 'restore', 'resume'],
        'notification_system': ['notify', 'alert', 'telegram', 'email'],
        'manual_override': ['manual', 'override', 'force']
    }
    
    files_to_check = [
        'core/trading_bot.py',
        'core/safety_manager.py',
        'utils/notifier.py',
        'main.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Checking {file_path}:")
            
            for procedure, keywords in emergency_keywords.items():
                if any(keyword in content for keyword in keywords):
                    emergency_procedures[procedure] = True
                    print(f"✅ {procedure.replace('_', ' ').title()}")
                else:
                    print(f"❌ {procedure.replace('_', ' ').title()}")
    
    return emergency_procedures

def analyze_safety_testing():
    """Analysiere Safety Testing"""
    print("\n🔍 Analyzing Safety Testing...")
    print("=" * 50)
    
    safety_testing = {
        'has_safety_tests': False,
        'has_crash_simulation': False,
        'has_api_failure_tests': False,
        'has_drawdown_tests': False,
        'test_coverage': 0
    }
    
    # Check Tests-Verzeichnis
    test_files = []
    if os.path.exists('tests'):
        test_files = [f for f in os.listdir('tests') if f.endswith('.py')]
        print(f"Found {len(test_files)} test files")
    
    safety_test_keywords = ['safety', 'emergency', 'drawdown', 'crash', 'failure', 'risk']
    
    for test_file in test_files:
        test_path = os.path.join('tests', test_file)
        try:
            with open(test_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Analyzing {test_file}:")
            
            if any(keyword in content for keyword in safety_test_keywords):
                safety_testing['has_safety_tests'] = True
                print("✅ Safety tests found")
            
            if 'crash' in content or 'simulation' in content:
                safety_testing['has_crash_simulation'] = True
                print("✅ Crash simulation tests")
            
            if 'api' in content and 'failure' in content:
                safety_testing['has_api_failure_tests'] = True
                print("✅ API failure tests")
            
            if 'drawdown' in content:
                safety_testing['has_drawdown_tests'] = True
                print("✅ Drawdown tests")
                
        except Exception as e:
            continue
    
    # Berechne Test Coverage
    tested_features = sum([
        safety_testing['has_safety_tests'],
        safety_testing['has_crash_simulation'],
        safety_testing['has_api_failure_tests'],
        safety_testing['has_drawdown_tests']
    ])
    safety_testing['test_coverage'] = (tested_features / 4) * 100
    
    print(f"\n📊 Safety Test Coverage: {safety_testing['test_coverage']:.1f}%")
    
    return safety_testing

def generate_safety_systems_score():
    """Generiere Safety Systems Score"""
    print("\n" + "="*60)
    print("📊 SAFETY SYSTEMS AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    protection, critical_features = check_risk_protection()
    safety_mgr = check_safety_manager()
    risk_mgr = check_risk_manager_integration()
    emergency = check_emergency_procedures()
    testing = analyze_safety_testing()
    
    # Berechne Scores
    total_score = 0
    max_score = 100
    
    # Basic Protection Features (30 Punkte)
    protection_score = sum(protection.values()) * (30 / len(protection))
    total_score += protection_score
    
    # Safety Manager (25 Punkte)
    safety_score = 0
    if safety_mgr['has_safety_manager']:
        safety_score += 15
    if safety_mgr['has_real_time_monitoring']:
        safety_score += 5
    if safety_mgr['has_automatic_shutdown']:
        safety_score += 3
    if safety_mgr['has_position_liquidation']:
        safety_score += 2
    total_score += safety_score
    
    # Risk Manager Integration (20 Punkte)
    risk_score = sum(risk_mgr.values()) * (20 / len(risk_mgr))
    total_score += risk_score
    
    # Emergency Procedures (15 Punkte)
    emergency_score = sum(emergency.values()) * (15 / len(emergency))
    total_score += emergency_score
    
    # Safety Testing (10 Punkte)
    testing_score = testing['test_coverage'] / 10  # 100% coverage = 10 points
    total_score += testing_score
    
    print(f"\n📊 SAFETY SYSTEMS SCORE: {total_score:.1f}/100")
    print(f"   Basic Protection: {protection_score:.1f}/30")
    print(f"   Safety Manager: {safety_score:.1f}/25")
    print(f"   Risk Manager: {risk_score:.1f}/20")
    print(f"   Emergency Procedures: {emergency_score:.1f}/15")
    print(f"   Safety Testing: {testing_score:.1f}/10")
    
    # Kritische Bewertung
    print(f"\n🚨 CRITICAL SAFETY ASSESSMENT:")
    critical_missing = []
    
    if not protection['has_emergency_stop']:
        critical_missing.append("❌ CRITICAL: No emergency stop system")
    
    if not protection['has_drawdown_limits']:
        critical_missing.append("❌ CRITICAL: No drawdown protection")
    
    if not safety_mgr['has_safety_manager']:
        critical_missing.append("❌ CRITICAL: No safety manager")
    
    if not protection['has_killswitch']:
        critical_missing.append("❌ CRITICAL: No manual killswitch")
    
    if critical_missing:
        print("   🚨 IMMEDIATE ACTION REQUIRED:")
        for missing in critical_missing:
            print(f"      {missing}")
    else:
        print("   ✅ Basic critical safety systems are present")
    
    # Detaillierte Features
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    print(f"\nBasic Protection Features:")
    for key, value in protection.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nSafety Manager:")
    for key, value in safety_mgr.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nRisk Manager Integration:")
    for key, value in risk_mgr.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nEmergency Procedures:")
    for key, value in emergency.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    # Priority Recommendations
    print(f"\n💡 PRIORITY RECOMMENDATIONS:")
    
    if total_score < 50:
        print("   🚨 URGENT: Safety systems are critically insufficient!")
        print("   🔧 Implement emergency stop system immediately")
        print("   🔧 Add drawdown protection with automatic shutdown")
        print("   🔧 Create comprehensive safety manager")
    
    elif total_score < 70:
        print("   ⚠️  Safety systems need significant improvement")
        print("   🔧 Enhance real-time monitoring")
        print("   🔧 Add API failure handling")
        print("   🔧 Implement graceful shutdown procedures")
    
    else:
        print("   ✅ Safety systems are well-implemented")
        print("   🔧 Consider adding more comprehensive testing")
        print("   🔧 Enhance notification systems")
    
    # Specific TODOs
    priority_todos = []
    if not protection['has_emergency_stop']:
        priority_todos.append("URGENT: Create emergency stop system")
    if not protection['has_drawdown_limits']:
        priority_todos.append("URGENT: Implement drawdown protection")
    if not safety_mgr['has_safety_manager']:
        priority_todos.append("HIGH: Create core/safety_manager.py")
    if not protection['has_killswitch']:
        priority_todos.append("HIGH: Add manual killswitch")
    if not emergency['has_graceful_shutdown']:
        priority_todos.append("MEDIUM: Implement graceful shutdown")
    
    print(f"\n🎯 PRIORITY TODO LIST:")
    for i, todo in enumerate(priority_todos, 1):
        print(f"   {i}. {todo}")
    
    return {
        'total_score': total_score,
        'protection_features': protection,
        'safety_manager': safety_mgr,
        'risk_manager': risk_mgr,
        'emergency_procedures': emergency,
        'safety_testing': testing,
        'critical_features': critical_features,
        'critical_missing': critical_missing,
        'priority_todos': priority_todos
    }

if __name__ == "__main__":
    # Führe Safety Systems Audit durch
    report = generate_safety_systems_score()
    
    # Speichere Report
    with open('safety_systems_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: safety_systems_audit.json")