#!/usr/bin/env python3
"""
Capital Management Audit
Prüft zentrale Kapitalverwaltung und Position-Tracking
"""
import os
import json
from pathlib import Path

def check_capital_management():
    """Prüfe zentrale Kapitalverwaltung"""
    print("🔍 Checking Capital Management...")
    print("=" * 50)
    
    issues = []
    features = []
    
    capital_files = [
        'core/portfolio_manager.py',
        'risk/portfolio_manager.py', 
        'core/position.py',
        'core/order_manager.py'
    ]
    
    capital_management = {
        'has_portfolio_manager': False,
        'has_position_tracking': False,
        'has_capital_allocation': False,
        'has_position_limits': False,
        'has_collision_detection': False,
        'has_risk_management': False
    }
    
    # Check für Portfolio Manager
    for file_path in capital_files:
        if os.path.exists(file_path):
            capital_management['has_portfolio_manager'] = True
            features.append(f"Portfolio Manager: {file_path}")
            print(f"✅ Portfolio manager found: {file_path}")
            break
    
    if not capital_management['has_portfolio_manager']:
        issues.append("Kein zentraler Portfolio Manager gefunden")
        print("❌ No central Portfolio Manager found")
    
    # Check core/trading_bot.py für Capital Tracking
    if os.path.exists('core/trading_bot.py'):
        with open('core/trading_bot.py', 'r') as f:
            content = f.read()
        
        print(f"\n📄 Analyzing core/trading_bot.py:")
        
        # Capital Tracking
        if 'track_capital' in content or 'capital_allocation' in content:
            capital_management['has_capital_allocation'] = True
            features.append("Capital tracking in trading_bot.py")
            print("✅ Capital tracking found")
        else:
            issues.append("Kein Kapital-Tracking implementiert")
            print("❌ No capital tracking")
        
        # Position Limits
        if 'max_positions' in content or 'position_limit' in content:
            capital_management['has_position_limits'] = True
            features.append("Position limits defined")
            print("✅ Position limits found")
        else:
            issues.append("Keine Position-Limits definiert")
            print("❌ No position limits")
        
        # Risk Management
        if 'risk' in content.lower() and ('manager' in content or 'check' in content):
            capital_management['has_risk_management'] = True
            features.append("Risk management integration")
            print("✅ Risk management found")
        else:
            issues.append("Keine Risk Management Integration")
            print("❌ No risk management")
    
    # Check für Position Tracking
    position_files = ['core/position.py', 'core/order_manager.py']
    for file_path in position_files:
        if os.path.exists(file_path):
            capital_management['has_position_tracking'] = True
            features.append(f"Position tracking: {file_path}")
            print(f"✅ Position tracking: {file_path}")
            break
    
    # Check für Collision Detection
    collision_keywords = ['collision', 'duplicate', 'overlap', 'conflict']
    for root, dirs, files in os.walk('core'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                    
                    if any(keyword in content for keyword in collision_keywords):
                        capital_management['has_collision_detection'] = True
                        features.append(f"Collision detection in {file}")
                        print(f"✅ Collision detection in {file}")
                        break
                except:
                    continue
    
    if not capital_management['has_collision_detection']:
        issues.append("Keine Collision-Detection für doppelte Trades")
        print("❌ No collision detection for duplicate trades")
    
    return issues, features, capital_management

def analyze_strategy_coordination():
    """Analysiere Koordination zwischen Strategien"""
    print("\n🔍 Analyzing Strategy Coordination...")
    print("=" * 50)
    
    coordination_analysis = {
        'strategies_share_capital': False,
        'strategies_communicate': False,
        'has_central_coordinator': False,
        'independent_strategies': True
    }
    
    # Check strategy files
    strategy_files = []
    if os.path.exists('strategies'):
        strategy_files = [f for f in os.listdir('strategies') if f.endswith('.py') and f != '__init__.py']
    
    print(f"Found {len(strategy_files)} strategy files")
    
    # Analyze strategy independence
    strategy_coordination_features = []
    
    for strategy_file in strategy_files[:5]:  # Check first 5 strategies
        file_path = os.path.join('strategies', strategy_file)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            print(f"\n📄 Analyzing {strategy_file}:")
            
            # Check für shared resources
            if 'shared' in content.lower() or 'common' in content.lower():
                coordination_analysis['strategies_share_capital'] = True
                strategy_coordination_features.append(f"Shared resources in {strategy_file}")
                print("✅ Shared resource usage detected")
            
            # Check für communication
            if 'signal' in content.lower() or 'message' in content.lower() or 'coordinate' in content.lower():
                coordination_analysis['strategies_communicate'] = True
                strategy_coordination_features.append(f"Inter-strategy communication in {strategy_file}")
                print("✅ Inter-strategy communication detected")
            
            # Check für central coordination
            if 'router' in content.lower() or 'coordinator' in content.lower():
                coordination_analysis['has_central_coordinator'] = True
                strategy_coordination_features.append(f"Central coordination in {strategy_file}")
                print("✅ Central coordination detected")
        
        except Exception as e:
            print(f"❌ Error analyzing {strategy_file}: {e}")
            continue
    
    # Override independence if coordination found
    if (coordination_analysis['strategies_share_capital'] or 
        coordination_analysis['strategies_communicate'] or 
        coordination_analysis['has_central_coordinator']):
        coordination_analysis['independent_strategies'] = False
    
    return coordination_analysis, strategy_coordination_features

def check_position_size_management():
    """Prüfe Position Size Management"""
    print("\n🔍 Checking Position Size Management...")
    print("=" * 50)
    
    position_management = {
        'has_kelly_criterion': False,
        'has_risk_per_trade': False,
        'has_position_sizing': False,
        'has_capital_limits': False,
        'has_diversification': False
    }
    
    # Files to check for position sizing
    files_to_check = [
        'risk/kelly_criterion_optimizer.py',
        'risk/position_calculator.py',
        'strategies/super_lazy_billionaire_strategy.py',
        'core/risk_manager.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Analyzing {file_path}:")
            
            # Kelly Criterion
            if 'kelly' in content:
                position_management['has_kelly_criterion'] = True
                print("✅ Kelly Criterion found")
            
            # Risk per trade
            if 'risk_per_trade' in content or 'risk per trade' in content:
                position_management['has_risk_per_trade'] = True
                print("✅ Risk per trade management found")
            
            # Position sizing
            if 'position_size' in content or 'calculate_size' in content:
                position_management['has_position_sizing'] = True
                print("✅ Position sizing calculation found")
            
            # Capital limits
            if 'capital_limit' in content or 'max_capital' in content:
                position_management['has_capital_limits'] = True
                print("✅ Capital limits found")
            
            # Diversification
            if 'diversification' in content or 'correlation' in content:
                position_management['has_diversification'] = True
                print("✅ Diversification management found")
    
    return position_management

def evaluate_capital_efficiency():
    """Bewerte Kapitaleffizienz"""
    print("\n🔍 Evaluating Capital Efficiency...")
    print("=" * 50)
    
    efficiency_metrics = {
        'tracks_utilization': False,
        'has_rebalancing': False,
        'monitors_performance': False,
        'optimizes_allocation': False
    }
    
    # Check für Capital Utilization Tracking
    search_terms = {
        'tracks_utilization': ['utilization', 'usage', 'allocation_percentage'],
        'has_rebalancing': ['rebalance', 'reallocation', 'redistribution'],
        'monitors_performance': ['performance', 'roi', 'returns'],
        'optimizes_allocation': ['optimize', 'optimal', 'efficient']
    }
    
    # Search in relevant files
    relevant_files = [
        'core/trading_bot.py',
        'core/strategy_router.py', 
        'risk/portfolio_manager.py',
        'strategies/super_lazy_billionaire_strategy.py'
    ]
    
    for file_path in relevant_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Checking {file_path}:")
            
            for metric, terms in search_terms.items():
                if any(term in content for term in terms):
                    efficiency_metrics[metric] = True
                    print(f"✅ {metric.replace('_', ' ').title()}")
                else:
                    print(f"❌ {metric.replace('_', ' ').title()}")
    
    return efficiency_metrics

def generate_capital_management_score():
    """Generiere Capital Management Score"""
    print("\n" + "="*60)
    print("📊 CAPITAL MANAGEMENT AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    issues, features, capital_mgmt = check_capital_management()
    coordination, coord_features = analyze_strategy_coordination()
    position_mgmt = check_position_size_management()
    efficiency = evaluate_capital_efficiency()
    
    # Berechne Scores
    total_score = 0
    max_score = 100
    
    # Basic Capital Management (40 Punkte)
    basic_score = sum(capital_mgmt.values()) * (40 / len(capital_mgmt))
    total_score += basic_score
    
    # Strategy Coordination (25 Punkte)
    coord_score = 0
    if coordination['has_central_coordinator']:
        coord_score += 10
    if coordination['strategies_share_capital']:
        coord_score += 8
    if coordination['strategies_communicate']:
        coord_score += 7
    total_score += coord_score
    
    # Position Management (25 Punkte)
    position_score = sum(position_mgmt.values()) * (25 / len(position_mgmt))
    total_score += position_score
    
    # Capital Efficiency (10 Punkte)
    efficiency_score = sum(efficiency.values()) * (10 / len(efficiency))
    total_score += efficiency_score
    
    print(f"\n📊 CAPITAL MANAGEMENT SCORE: {total_score:.1f}/100")
    print(f"   Basic Management: {basic_score:.1f}/40")
    print(f"   Strategy Coordination: {coord_score:.1f}/25")
    print(f"   Position Management: {position_score:.1f}/25")
    print(f"   Capital Efficiency: {efficiency_score:.1f}/10")
    
    # Detaillierte Bewertung
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    print(f"\nBasic Capital Management:")
    for key, value in capital_mgmt.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nStrategy Coordination:")
    for key, value in coordination.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nPosition Management:")
    for key, value in position_mgmt.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nCapital Efficiency:")
    for key, value in efficiency.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    # Issues & Features
    if issues:
        print(f"\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
    
    if features:
        print(f"\n✅ POSITIVE FEATURES:")
        for feature in features:
            print(f"   • {feature}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if basic_score < 25:
        print("   🔧 CRITICAL: Implement central portfolio manager")
        print("   🔧 Add position tracking and capital allocation")
        print("   🔧 Implement collision detection for duplicate trades")
    
    if coord_score < 15:
        print("   🔧 Add central strategy coordinator")
        print("   🔧 Implement shared capital pool management")
        print("   🔧 Add inter-strategy communication")
    
    if position_score < 15:
        print("   🔧 Implement Kelly Criterion for position sizing")
        print("   🔧 Add risk-per-trade limits")
        print("   🔧 Include diversification management")
    
    if efficiency_score < 7:
        print("   🔧 Track capital utilization metrics")
        print("   🔧 Implement automatic rebalancing")
        print("   🔧 Add performance monitoring per allocation")
    
    # Priority TODOs
    priority_todos = []
    if not capital_mgmt['has_portfolio_manager']:
        priority_todos.append("Create core/portfolio_manager.py")
    if not capital_mgmt['has_collision_detection']:
        priority_todos.append("Implement duplicate trade prevention")
    if not position_mgmt['has_kelly_criterion']:
        priority_todos.append("Add Kelly Criterion position sizing")
    if not coordination['has_central_coordinator']:
        priority_todos.append("Create central strategy coordinator")
    
    print(f"\n🎯 PRIORITY TODO LIST:")
    for i, todo in enumerate(priority_todos, 1):
        print(f"   {i}. {todo}")
    
    return {
        'total_score': total_score,
        'basic_management': capital_mgmt,
        'coordination': coordination,
        'position_management': position_mgmt,
        'efficiency': efficiency,
        'issues': issues,
        'features': features,
        'priority_todos': priority_todos
    }

if __name__ == "__main__":
    # Führe Capital Management Audit durch
    report = generate_capital_management_score()
    
    # Speichere Report
    with open('capital_management_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: capital_management_audit.json")