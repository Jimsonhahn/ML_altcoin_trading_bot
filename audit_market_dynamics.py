#!/usr/bin/env python3
"""
Marktphasen & Strategiewechsel Audit
Analysiert die Dynamik und Adaption des Bots
"""
import os
import json
from pathlib import Path

def check_market_adaptability():
    """Prüfe Market-Adaptive Features"""
    print("🔍 Checking Market Adaptability...")
    print("=" * 50)
    
    checks = {
        'has_market_analyzer': False,
        'has_phase_detection': False,
        'has_dynamic_weights': False,
        'has_smooth_transitions': False,
        'has_multi_timeframe': False,
        'has_confidence_scores': False,
        'has_regime_detection': False
    }
    
    # Check core/market_analyzer.py
    market_analyzer_path = 'core/market_analyzer.py'
    if os.path.exists(market_analyzer_path):
        checks['has_market_analyzer'] = True
        print(f"✅ {market_analyzer_path} exists")
        
        with open(market_analyzer_path, 'r') as f:
            content = f.read()
            
        # Check für spezifische Features
        if 'detect_market_phase' in content:
            checks['has_phase_detection'] = True
            print("✅ Phase detection found")
        else:
            print("❌ No phase detection method")
            
        if 'confidence' in content and 'score' in content:
            checks['has_confidence_scores'] = True
            print("✅ Confidence scoring found")
        else:
            print("❌ No confidence scoring")
            
        if 'timeframe' in content and ('1h' in content or '4h' in content):
            checks['has_multi_timeframe'] = True
            print("✅ Multi-timeframe analysis found")
        else:
            print("❌ No multi-timeframe analysis")
    else:
        print(f"❌ {market_analyzer_path} not found")
    
    # Check core/strategy_router.py
    strategy_router_path = 'core/strategy_router.py'
    if os.path.exists(strategy_router_path):
        print(f"✅ {strategy_router_path} exists")
        
        with open(strategy_router_path, 'r') as f:
            content = f.read()
            
        if 'calculate_allocations' in content or 'dynamic' in content:
            checks['has_dynamic_weights'] = True
            print("✅ Dynamic allocation found")
        else:
            print("❌ No dynamic allocation")
            
        if 'transition' in content and ('smooth' in content or 'gradual' in content):
            checks['has_smooth_transitions'] = True
            print("✅ Smooth transitions found")
        else:
            print("❌ No smooth transitions")
    else:
        print(f"❌ {strategy_router_path} not found")
    
    # Check ML-enhanced regime detection
    regime_files = [
        'ml_components/market_regime.py',
        'core/advanced_market_regime_detector.py'
    ]
    
    for file_path in regime_files:
        if os.path.exists(file_path):
            checks['has_regime_detection'] = True
            print(f"✅ Regime detection: {file_path}")
            break
    
    if not checks['has_regime_detection']:
        print("❌ No advanced regime detection")
    
    return checks

def analyze_strategy_switching_logic():
    """Analysiere Strategiewechsel-Logik"""
    print("\n🔍 Analyzing Strategy Switching Logic...")
    print("=" * 50)
    
    switching_analysis = {
        'has_switching_rules': False,
        'switching_speed': 'unknown',
        'has_hysteresis': False,
        'has_manual_override': False,
        'switching_frequency': 'unknown'
    }
    
    # Suche nach Switching-Logic in verschiedenen Files
    files_to_check = [
        'core/strategy_router.py',
        'main.py', 
        'core/trading_bot.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            print(f"\n📄 Analyzing {file_path}:")
            
            # Check für Switching Rules
            if 'switch' in content.lower() and 'strategy' in content.lower():
                switching_analysis['has_switching_rules'] = True
                print("✅ Strategy switching logic found")
            
            # Check für Speed Control
            if 'gradual' in content or 'transition_speed' in content:
                switching_analysis['switching_speed'] = 'gradual'
                print("✅ Gradual switching detected")
            elif 'immediate' in content or 'instant' in content:
                switching_analysis['switching_speed'] = 'immediate'
                print("⚠️  Immediate switching (risky)")
            
            # Check für Hysterese
            if 'hysteresis' in content or 'buffer' in content:
                switching_analysis['has_hysteresis'] = True
                print("✅ Hysteresis/buffering found")
            
            # Check für Manual Override
            if 'manual' in content and 'override' in content:
                switching_analysis['has_manual_override'] = True
                print("✅ Manual override capability")
    
    return switching_analysis

def evaluate_market_phase_coverage():
    """Bewerte Marktphasen-Abdeckung"""
    print("\n🔍 Evaluating Market Phase Coverage...")
    print("=" * 50)
    
    phase_coverage = {
        'bull_market': False,
        'bear_market': False,
        'sideways_market': False,
        'high_volatility': False,
        'low_liquidity': False,
        'flash_crash': False,
        'total_phases': 0
    }
    
    # Suche nach Marktphasen-Definitionen
    search_files = [
        'core/market_analyzer.py',
        'ml_components/market_regime.py',
        'strategies/super_lazy_billionaire_strategy.py'
    ]
    
    for file_path in search_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Checking {file_path}:")
            
            phases = ['bull', 'bear', 'sideways', 'volatile', 'liquidity', 'crash']
            for phase in phases:
                if phase in content:
                    phase_key = f"{phase}_market" if phase in ['bull', 'bear', 'sideways'] else f"{phase}_{'volatility' if phase == 'volatile' else 'liquidity' if phase == 'liquidity' else 'crash'}"
                    if phase_key in phase_coverage:
                        phase_coverage[phase_key] = True
                        print(f"✅ {phase.title()} market handling found")
    
    phase_coverage['total_phases'] = sum(1 for v in phase_coverage.values() if isinstance(v, bool) and v)
    
    return phase_coverage

def check_backtesting_market_conditions():
    """Prüfe Backtest-Abdeckung verschiedener Marktbedingungen"""
    print("\n🔍 Checking Backtest Market Conditions...")
    print("=" * 50)
    
    backtest_coverage = {
        'has_multiple_periods': False,
        'has_crisis_testing': False,
        'has_different_volatilities': False,
        'total_backtest_files': 0
    }
    
    # Check data/backtest_results/
    backtest_dir = 'data/backtest_results'
    if os.path.exists(backtest_dir):
        backtest_files = [f for f in os.listdir(backtest_dir) if os.path.isdir(os.path.join(backtest_dir, f))]
        backtest_coverage['total_backtest_files'] = len(backtest_files)
        print(f"✅ Found {len(backtest_files)} backtest result sets")
        
        if len(backtest_files) > 5:
            backtest_coverage['has_multiple_periods'] = True
            print("✅ Multiple test periods available")
    
    # Check for specific market condition tests
    test_files = [
        'core/enhanced_backtesting.py',
        'core/ml_enhanced_backtesting.py'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            if 'crisis' in content or '2022' in content or 'crash' in content:
                backtest_coverage['has_crisis_testing'] = True
                print(f"✅ Crisis/crash testing in {file_path}")
            
            if 'volatility' in content and 'different' in content:
                backtest_coverage['has_different_volatilities'] = True
                print(f"✅ Volatility testing in {file_path}")
    
    return backtest_coverage

def generate_market_dynamics_score():
    """Generiere Market Dynamics Score"""
    print("\n" + "="*60)
    print("📊 MARKET DYNAMICS AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    adaptability = check_market_adaptability()
    switching = analyze_strategy_switching_logic()
    phase_coverage = evaluate_market_phase_coverage()
    backtest_coverage = check_backtesting_market_conditions()
    
    # Berechne Scores
    total_score = 0
    max_score = 100
    
    # Adaptability Score (40 Punkte)
    adaptability_score = sum(adaptability.values()) * (40 / len(adaptability))
    total_score += adaptability_score
    
    # Switching Logic Score (30 Punkte)
    switching_score = 0
    if switching['has_switching_rules']:
        switching_score += 10
    if switching['switching_speed'] == 'gradual':
        switching_score += 10
    elif switching['switching_speed'] == 'immediate':
        switching_score += 5  # Weniger Punkte für immediate
    if switching['has_hysteresis']:
        switching_score += 5
    if switching['has_manual_override']:
        switching_score += 5
    total_score += switching_score
    
    # Phase Coverage Score (20 Punkte)
    phase_score = (phase_coverage['total_phases'] / 6) * 20
    total_score += phase_score
    
    # Backtest Coverage Score (10 Punkte)
    backtest_score = 0
    if backtest_coverage['has_multiple_periods']:
        backtest_score += 4
    if backtest_coverage['has_crisis_testing']:
        backtest_score += 3
    if backtest_coverage['has_different_volatilities']:
        backtest_score += 3
    total_score += backtest_score
    
    print(f"\n📊 MARKET DYNAMICS SCORE: {total_score:.1f}/100")
    print(f"   Adaptability: {adaptability_score:.1f}/40")
    print(f"   Switching Logic: {switching_score:.1f}/30")
    print(f"   Phase Coverage: {phase_score:.1f}/20")
    print(f"   Backtest Coverage: {backtest_score:.1f}/10")
    
    # Detaillierte Bewertung
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"Market Adaptability:")
    for key, value in adaptability.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nStrategy Switching:")
    for key, value in switching.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}")
        else:
            print(f"   📊 {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nMarket Phase Coverage ({phase_coverage['total_phases']}/6):")
    for key, value in phase_coverage.items():
        if isinstance(value, bool) and key != 'total_phases':
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if adaptability_score < 30:
        print("   🔧 Implement dynamic market phase detection")
        print("   🔧 Add confidence scoring for phase transitions")
        print("   🔧 Implement multi-timeframe analysis")
    
    if switching_score < 20:
        print("   🔧 Add smooth strategy transitions (no hard switches)")
        print("   🔧 Implement hysteresis to prevent whipsaws")
        print("   🔧 Add manual override capability")
    
    if phase_score < 15:
        print("   🔧 Extend market phase detection to cover:")
        print("       - Flash crash scenarios")
        print("       - Low liquidity conditions")
        print("       - High volatility periods")
    
    if backtest_score < 7:
        print("   🔧 Test bot performance across different market conditions:")
        print("       - Bull market 2021")
        print("       - Bear market 2022")
        print("       - Sideways markets")
        print("       - Crisis periods")
    
    return {
        'total_score': total_score,
        'adaptability': adaptability,
        'switching': switching,
        'phase_coverage': phase_coverage,
        'backtest_coverage': backtest_coverage,
        'critical_missing': [
            'Smooth transitions' if not switching.get('has_hysteresis') else None,
            'Multi-phase detection' if phase_coverage['total_phases'] < 4 else None,
            'Crisis testing' if not backtest_coverage.get('has_crisis_testing') else None
        ]
    }

if __name__ == "__main__":
    # Führe Market Dynamics Audit durch
    report = generate_market_dynamics_score()
    
    # Speichere Report
    with open('market_dynamics_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: market_dynamics_audit.json")