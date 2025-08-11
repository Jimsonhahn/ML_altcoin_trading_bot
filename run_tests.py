"""
Test Runner for Trading Bot
Runs all tests and provides comprehensive reporting
"""

import sys
import subprocess
import time
from pathlib import Path

def run_component_tests():
    """Run individual component tests"""
    print("🔧 Running Component Tests")
    print("-" * 30)
    
    component_tests = [
        ("Secret Manager", "python -m utils.secret_manager"),
        ("Secure HTTP", "python -c \"from utils.secure_http import create_secure_session; s = create_secure_session(); print('✅ Secure HTTP works')\""),
        ("Validators", "python -m utils.test_validators"),
        ("Error Handler", "python -m utils.test_error_handler"),
        ("Validation Integration", "python -m utils.validation_integration_example"),
        ("Integrated Trading", "python -m utils.integrated_trading_example"),
    ]
    
    results = []
    
    for test_name, command in component_tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {test_name}: PASSED")
                results.append((test_name, True, None))
            else:
                print(f"❌ {test_name}: FAILED")
                print(f"   Error: {result.stderr}")
                results.append((test_name, False, result.stderr))
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name}: TIMEOUT")
            results.append((test_name, False, "Timeout"))
        except Exception as e:
            print(f"💥 {test_name}: EXCEPTION - {e}")
            results.append((test_name, False, str(e)))
    
    return results

def run_integration_tests():
    """Run integration tests using pytest"""
    print("\n🔗 Running Integration Tests")
    print("-" * 30)
    
    try:
        # Check if pytest is available
        result = subprocess.run([sys.executable, "-c", "import pytest"], capture_output=True)
        if result.returncode != 0:
            print("⚠️ pytest not available, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)
        
        # Run integration tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_integration.py",
            "-v",
            "--tb=short",
            "--no-header"
        ], capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Integration tests timed out")
        return False
    except Exception as e:
        print(f"💥 Integration tests failed: {e}")
        return False

def run_system_health_checks():
    """Run system health checks"""
    print("\n🏥 Running System Health Checks")
    print("-" * 30)
    
    health_checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        health_checks.append(("Python Version", True))
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}")
        health_checks.append(("Python Version", False))
    
    # Check required packages
    required_packages = [
        "pydantic", "cryptography", "keyring", "certifi", "requests"
    ]
    
    for package in required_packages:
        try:
            result = subprocess.run([
                sys.executable, "-c", f"import {package}"
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ Package {package}: Available")
                health_checks.append((f"Package {package}", True))
            else:
                print(f"❌ Package {package}: Missing")
                health_checks.append((f"Package {package}", False))
                
        except Exception as e:
            print(f"💥 Package {package}: Error - {e}")
            health_checks.append((f"Package {package}", False))
    
    # Check file permissions
    important_files = [
        "utils/secret_manager.py",
        "utils/secure_http.py", 
        "utils/validators.py",
        "utils/error_handler.py"
    ]
    
    for file_path in important_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"✅ File {file_path}: Exists")
            health_checks.append((f"File {file_path}", True))
        else:
            print(f"❌ File {file_path}: Missing")
            health_checks.append((f"File {file_path}", False))
    
    return health_checks

def generate_test_report(component_results, integration_success, health_checks):
    """Generate comprehensive test report"""
    print("\n📊 Test Report")
    print("=" * 50)
    
    # Component test summary
    component_passed = sum(1 for _, success, _ in component_results if success)
    component_total = len(component_results)
    
    print(f"🔧 Component Tests: {component_passed}/{component_total} passed")
    for test_name, success, error in component_results:
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
        if not success and error:
            print(f"      Error: {error[:100]}...")
    
    # Integration test summary
    print(f"\n🔗 Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    # Health check summary
    health_passed = sum(1 for _, success in health_checks if success)
    health_total = len(health_checks)
    
    print(f"\n🏥 Health Checks: {health_passed}/{health_total} passed")
    for check_name, success in health_checks:
        status = "✅" if success else "❌"
        print(f"   {status} {check_name}")
    
    # Overall summary
    total_tests = component_total + 1 + health_total  # +1 for integration
    total_passed = component_passed + (1 if integration_success else 0) + health_passed
    
    print(f"\n📈 Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if total_passed == total_tests:
        print("   🎉 All tests passed! Your trading bot is ready for deployment.")
    else:
        print("   ⚠️ Some tests failed. Review the errors above before deployment.")
        if health_passed < health_total:
            print("   🔧 Fix system health issues first.")
        if not integration_success:
            print("   🔗 Check integration test failures.")
        if component_passed < component_total:
            print("   🔧 Fix component test failures.")
    
    return total_passed == total_tests

def main():
    """Main test runner"""
    print("🚀 Trading Bot Test Suite")
    print("=" * 50)
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run all test categories
        component_results = run_component_tests()
        integration_success = run_integration_tests()
        health_checks = run_system_health_checks()
        
        # Generate report
        all_passed = generate_test_report(component_results, integration_success, health_checks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ Total runtime: {duration:.2f} seconds")
        print(f"⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if all_passed:
            print("\n🎉 SUCCESS: All tests passed!")
            return 0
        else:
            print("\n❌ FAILURE: Some tests failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Test run interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test run failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())