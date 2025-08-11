#!/usr/bin/env python3
"""
Testing Framework Audit
Prüft Qualität und Vollständigkeit des Testing-Systems
"""
import os
import json
from pathlib import Path

def check_testing_framework():
    """Prüfe Testing Framework"""
    print("🔍 Checking Testing Framework...")
    print("=" * 50)
    
    testing_quality = {
        'has_unit_tests': False,
        'has_integration_tests': False,
        'has_backtesting': False,
        'has_test_data': False,
        'has_ci_cd': False,
        'has_performance_tracking': False,
        'has_mocking': False,
        'has_coverage_reporting': False
    }
    
    test_files = []
    test_categories = []
    
    # Check tests/ directory
    if os.path.exists('tests/'):
        testing_quality['has_unit_tests'] = True
        test_files = [f for f in os.listdir('tests/') if f.endswith('.py')]
        test_categories.append(f"Unit tests: {len(test_files)} files")
        print(f"✅ Tests directory found with {len(test_files)} test files")
        
        # Analyze test types
        for test_file in test_files:
            test_path = os.path.join('tests/', test_file)
            try:
                with open(test_path, 'r') as f:
                    content = f.read().lower()
                
                if 'integration' in test_file.lower() or 'integration' in content:
                    testing_quality['has_integration_tests'] = True
                    test_categories.append(f"Integration test: {test_file}")
                
                if 'mock' in content or 'patch' in content:
                    testing_quality['has_mocking'] = True
                    test_categories.append(f"Mocking in: {test_file}")
                
                if 'coverage' in content:
                    testing_quality['has_coverage_reporting'] = True
                    test_categories.append(f"Coverage in: {test_file}")
                    
            except Exception as e:
                continue
    else:
        print("❌ No tests/ directory found")
    
    # Check für Backtesting
    backtesting_files = [
        'core/backtest.py',
        'core/backtest_engine.py',
        'core/enhanced_backtesting.py',
        'core/ml_enhanced_backtesting.py'
    ]
    
    for file_path in backtesting_files:
        if os.path.exists(file_path):
            testing_quality['has_backtesting'] = True
            test_categories.append(f"Backtesting: {file_path}")
            print(f"✅ Backtesting found: {file_path}")
            break
    
    if not testing_quality['has_backtesting']:
        print("❌ No backtesting system found")
    
    # Check für Test Data
    test_data_dirs = ['data/test/', 'tests/data/', 'data/backtest_results/']
    for data_dir in test_data_dirs:
        if os.path.exists(data_dir):
            testing_quality['has_test_data'] = True
            test_categories.append(f"Test data: {data_dir}")
            print(f"✅ Test data found: {data_dir}")
            break
    
    if not testing_quality['has_test_data']:
        print("❌ No dedicated test data found")
    
    # Check für CI/CD
    ci_files = ['.github/workflows/', '.gitlab-ci.yml', '.travis.yml', 'Jenkinsfile']
    for ci_file in ci_files:
        if os.path.exists(ci_file):
            testing_quality['has_ci_cd'] = True
            test_categories.append(f"CI/CD: {ci_file}")
            print(f"✅ CI/CD found: {ci_file}")
            break
    
    if not testing_quality['has_ci_cd']:
        print("❌ No CI/CD configuration found")
    
    # Check für Performance Tracking
    if os.path.exists('data/backtest_results/'):
        results = os.listdir('data/backtest_results/')
        if len(results) > 0:
            testing_quality['has_performance_tracking'] = True
            test_categories.append(f"Performance tracking: {len(results)} result sets")
            print(f"✅ Performance tracking: {len(results)} backtest result sets")
    
    return testing_quality, test_files, test_categories

def analyze_test_coverage():
    """Analysiere Test Coverage"""
    print("\n🔍 Analyzing Test Coverage...")
    print("=" * 50)
    
    coverage_analysis = {
        'core_module_coverage': 0,
        'strategy_coverage': 0,
        'utils_coverage': 0,
        'integration_coverage': 0,
        'total_estimated_coverage': 0
    }
    
    # Map core modules to test files
    core_modules = {
        'trading_bot': 'test_core.py',
        'strategy_router': 'test_strategies.py',
        'market_analyzer': 'test_core.py',
        'risk_manager': 'test_security.py',
        'safety_manager': 'test_security.py'
    }
    
    tested_core_modules = 0
    total_core_modules = len(core_modules)
    
    for module, test_file in core_modules.items():
        module_path = f'core/{module}.py'
        test_path = f'tests/{test_file}'
        
        if os.path.exists(module_path) and os.path.exists(test_path):
            with open(test_path, 'r') as f:
                test_content = f.read()
            
            if module in test_content:
                tested_core_modules += 1
                print(f"✅ {module} is tested in {test_file}")
            else:
                print(f"❌ {module} not found in {test_file}")
        else:
            print(f"⚠️  Missing: {module_path} or {test_path}")
    
    coverage_analysis['core_module_coverage'] = (tested_core_modules / total_core_modules) * 100
    
    # Strategy Coverage
    if os.path.exists('strategies/'):
        strategy_files = [f for f in os.listdir('strategies/') if f.endswith('.py') and f != '__init__.py']
        tested_strategies = 0
        
        if os.path.exists('tests/test_strategies.py'):
            with open('tests/test_strategies.py', 'r') as f:
                strategy_test_content = f.read()
            
            for strategy_file in strategy_files:
                strategy_name = strategy_file.replace('.py', '')
                if strategy_name in strategy_test_content:
                    tested_strategies += 1
                    print(f"✅ Strategy {strategy_name} is tested")
                else:
                    print(f"❌ Strategy {strategy_name} not tested")
        
        coverage_analysis['strategy_coverage'] = (tested_strategies / len(strategy_files)) * 100 if strategy_files else 0
    
    # Utils Coverage
    if os.path.exists('utils/'):
        utils_files = [f for f in os.listdir('utils/') if f.endswith('.py') and not f.startswith('test_')]
        tested_utils = 0
        
        # Check verschiedene Test-Files für Utils
        utils_test_files = ['test_core.py', 'test_integration.py', 'test_security.py']
        
        for utils_file in utils_files:
            utils_name = utils_file.replace('.py', '')
            tested = False
            
            for test_file in utils_test_files:
                test_path = f'tests/{test_file}'
                if os.path.exists(test_path):
                    with open(test_path, 'r') as f:
                        test_content = f.read()
                    if utils_name in test_content:
                        tested_utils += 1
                        tested = True
                        print(f"✅ Utils {utils_name} tested in {test_file}")
                        break
            
            if not tested:
                print(f"❌ Utils {utils_name} not tested")
        
        coverage_analysis['utils_coverage'] = (tested_utils / len(utils_files)) * 100 if utils_files else 0
    
    # Integration Tests Coverage
    integration_tests = ['test_integration.py', 'test_end_to_end.py']
    integration_score = 0
    
    for test_file in integration_tests:
        if os.path.exists(f'tests/{test_file}'):
            integration_score += 50
            print(f"✅ Integration test found: {test_file}")
    
    coverage_analysis['integration_coverage'] = integration_score
    
    # Berechne Gesamt-Coverage
    coverage_analysis['total_estimated_coverage'] = (
        coverage_analysis['core_module_coverage'] * 0.4 +
        coverage_analysis['strategy_coverage'] * 0.3 +
        coverage_analysis['utils_coverage'] * 0.2 +
        coverage_analysis['integration_coverage'] * 0.1
    )
    
    return coverage_analysis

def check_backtest_reproducibility():
    """Prüfe Backtest Reproduzierbarkeit"""
    print("\n🔍 Checking Backtest Reproducibility...")
    print("=" * 50)
    
    reproducibility = {
        'has_standardized_process': False,
        'has_versioned_configs': False,
        'has_deterministic_results': False,
        'has_comparison_tools': False,
        'has_automated_reports': False
    }
    
    # Check für standardisierte Prozesse
    process_files = [
        'core/backtest_engine.py',
        'scripts/run_comprehensive_backtest.py',
        'core/enhanced_backtesting.py'
    ]
    
    for file_path in process_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            if 'seed' in content.lower() or 'random_state' in content.lower():
                reproducibility['has_deterministic_results'] = True
                print(f"✅ Deterministic results in {file_path}")
            
            if 'config' in content.lower() and 'save' in content.lower():
                reproducibility['has_versioned_configs'] = True
                print(f"✅ Config versioning in {file_path}")
            
            if 'standard' in content.lower() or 'consistent' in content.lower():
                reproducibility['has_standardized_process'] = True
                print(f"✅ Standardized process in {file_path}")
    
    # Check für Comparison Tools
    comparison_indicators = ['compare', 'benchmark', 'analysis']
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules']]
        for file in files:
            if any(indicator in file.lower() for indicator in comparison_indicators):
                reproducibility['has_comparison_tools'] = True
                print(f"✅ Comparison tools: {file}")
                break
        if reproducibility['has_comparison_tools']:
            break
    
    # Check für Automated Reports
    if os.path.exists('reports/') or os.path.exists('data/backtest_results/'):
        report_files = []
        
        if os.path.exists('reports/'):
            report_files.extend([f for f in os.listdir('reports/') if f.endswith('.py')])
        
        if report_files:
            reproducibility['has_automated_reports'] = True
            print(f"✅ Automated reports: {len(report_files)} report generators")
    
    return reproducibility

def analyze_test_quality():
    """Analysiere Test-Qualität"""
    print("\n🔍 Analyzing Test Quality...")
    print("=" * 50)
    
    quality_metrics = {
        'test_isolation': 0,
        'assertion_quality': 0,
        'edge_case_coverage': 0,
        'error_handling_tests': 0,
        'performance_tests': 0
    }
    
    if not os.path.exists('tests/'):
        print("❌ No tests directory for quality analysis")
        return quality_metrics
    
    test_files = [f for f in os.listdir('tests/') if f.endswith('.py')]
    total_quality_score = 0
    analyzed_files = 0
    
    for test_file in test_files:
        test_path = os.path.join('tests/', test_file)
        try:
            with open(test_path, 'r') as f:
                content = f.read()
            
            print(f"\n📄 Analyzing {test_file}:")
            file_quality = 0
            
            # Test Isolation
            if 'setup' in content.lower() and 'teardown' in content.lower():
                quality_metrics['test_isolation'] += 1
                file_quality += 1
                print("✅ Test isolation (setup/teardown)")
            
            # Assertion Quality
            assertion_keywords = ['assert', 'assertEqual', 'assertTrue', 'assertRaises']
            if any(keyword in content for keyword in assertion_keywords):
                quality_metrics['assertion_quality'] += 1
                file_quality += 1
                print("✅ Quality assertions")
            
            # Edge Case Coverage
            edge_case_keywords = ['edge', 'boundary', 'limit', 'extreme', 'corner']
            if any(keyword in content.lower() for keyword in edge_case_keywords):
                quality_metrics['edge_case_coverage'] += 1
                file_quality += 1
                print("✅ Edge case coverage")
            
            # Error Handling Tests
            error_keywords = ['exception', 'error', 'fail', 'raises']
            if any(keyword in content.lower() for keyword in error_keywords):
                quality_metrics['error_handling_tests'] += 1
                file_quality += 1
                print("✅ Error handling tests")
            
            # Performance Tests
            performance_keywords = ['time', 'performance', 'speed', 'benchmark']
            if any(keyword in content.lower() for keyword in performance_keywords):
                quality_metrics['performance_tests'] += 1
                file_quality += 1
                print("✅ Performance tests")
            
            total_quality_score += file_quality
            analyzed_files += 1
            
        except Exception as e:
            print(f"❌ Error analyzing {test_file}: {e}")
            continue
    
    # Normalisiere Scores
    if analyzed_files > 0:
        for metric in quality_metrics:
            quality_metrics[metric] = (quality_metrics[metric] / analyzed_files) * 100
    
    average_quality = total_quality_score / (analyzed_files * 5) * 100 if analyzed_files > 0 else 0
    
    print(f"\n📊 Average test quality: {average_quality:.1f}%")
    
    return quality_metrics, average_quality

def generate_testing_score():
    """Generiere Testing Framework Score"""
    print("\n" + "="*60)
    print("📊 TESTING FRAMEWORK AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    testing_framework, test_files, test_categories = check_testing_framework()
    coverage = analyze_test_coverage()
    reproducibility = check_backtest_reproducibility()
    quality_metrics, average_quality = analyze_test_quality()
    
    # Berechne Scores
    total_score = 0
    max_score = 100
    
    # Basic Testing Framework (30 Punkte)
    framework_score = sum(testing_framework.values()) * (30 / len(testing_framework))
    total_score += framework_score
    
    # Test Coverage (25 Punkte)
    coverage_score = coverage['total_estimated_coverage'] * 0.25
    total_score += coverage_score
    
    # Reproducibility (20 Punkte)
    repro_score = sum(reproducibility.values()) * (20 / len(reproducibility))
    total_score += repro_score
    
    # Test Quality (25 Punkte)
    quality_score = average_quality * 0.25
    total_score += quality_score
    
    print(f"\n📊 TESTING FRAMEWORK SCORE: {total_score:.1f}/100")
    print(f"   Framework Foundation: {framework_score:.1f}/30")
    print(f"   Test Coverage: {coverage_score:.1f}/25")
    print(f"   Reproducibility: {repro_score:.1f}/20")
    print(f"   Test Quality: {quality_score:.1f}/25")
    
    # Detaillierte Bewertung
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    print(f"\nTesting Framework Foundation:")
    for key, value in testing_framework.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nTest Coverage:")
    print(f"   📊 Core Modules: {coverage['core_module_coverage']:.1f}%")
    print(f"   📊 Strategies: {coverage['strategy_coverage']:.1f}%")
    print(f"   📊 Utils: {coverage['utils_coverage']:.1f}%")
    print(f"   📊 Integration: {coverage['integration_coverage']:.1f}%")
    print(f"   📊 Total Estimated: {coverage['total_estimated_coverage']:.1f}%")
    
    print(f"\nReproducibility:")
    for key, value in reproducibility.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nTest Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"   📊 {key.replace('_', ' ').title()}: {value:.1f}%")
    
    # Test Categories
    if test_categories:
        print(f"\n📋 Found Test Categories:")
        for category in test_categories:
            print(f"   • {category}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if framework_score < 20:
        print("   🔧 CRITICAL: Establish basic testing framework")
        print("   🔧 Create unit tests for core modules")
        print("   🔧 Set up CI/CD pipeline")
        print("   🔧 Add integration tests")
    
    if coverage_score < 15:
        print("   🔧 Increase test coverage:")
        print("       - Core modules (trading_bot, strategy_router)")
        print("       - All strategies")
        print("       - Critical utilities")
    
    if repro_score < 12:
        print("   🔧 Improve reproducibility:")
        print("       - Add deterministic seeds to backtests")
        print("       - Version control test configurations")
        print("       - Create automated comparison tools")
    
    if quality_score < 15:
        print("   🔧 Enhance test quality:")
        print("       - Add proper setup/teardown")
        print("       - Include edge case testing")
        print("       - Test error handling paths")
        print("       - Add performance benchmarks")
    
    # Specific Test Scenarios
    print(f"\n🎯 PRIORITY TEST SCENARIOS TO IMPLEMENT:")
    
    missing_tests = []
    if coverage['core_module_coverage'] < 80:
        missing_tests.append("Core module unit tests")
    if coverage['strategy_coverage'] < 70:
        missing_tests.append("Strategy backtests across different market conditions")
    if not reproducibility['has_deterministic_results']:
        missing_tests.append("Deterministic backtest framework")
    if not testing_framework['has_integration_tests']:
        missing_tests.append("End-to-end integration tests")
    
    for i, test in enumerate(missing_tests, 1):
        print(f"   {i}. {test}")
    
    # Market Condition Tests
    if total_score < 70:
        print(f"\n📋 RECOMMENDED MARKET CONDITION TESTS:")
        market_tests = [
            "Bull market (2020-2021)",
            "Bear market (2022)",
            "Sideways market (consolidation periods)",
            "Flash crash scenarios",
            "High volatility periods",
            "Low liquidity conditions"
        ]
        
        for test in market_tests:
            print(f"   • {test}")
    
    return {
        'total_score': total_score,
        'testing_framework': testing_framework,
        'coverage': coverage,
        'reproducibility': reproducibility,
        'quality_metrics': quality_metrics,
        'test_files': test_files,
        'test_categories': test_categories,
        'missing_tests': missing_tests
    }

if __name__ == "__main__":
    # Führe Testing Framework Audit durch
    report = generate_testing_score()
    
    # Speichere Report
    with open('testing_framework_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: testing_framework_audit.json")