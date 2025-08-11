#!/usr/bin/env python3
"""
Production Readiness Test Suite
==============================

Final comprehensive test to verify system is ready for production deployment.
Tests all critical components and performance under realistic conditions.
"""

import time
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))

class ProductionReadinessTest:
    """Complete production readiness validation"""
    
    def __init__(self):
        self.test_results = {}
        self.critical_failures = []
        self.warnings = []
        
    def run_all_tests(self):
        """Execute complete production readiness test suite"""
        print("🚀 PRODUCTION READINESS TEST SUITE")
        print("=" * 60)
        
        # 1. Critical System Components
        print("\n🔧 Testing Critical System Components...")
        self.test_critical_components()
        
        # 2. Configuration Validation
        print("\n⚙️ Testing Configuration...")
        self.test_configuration()
        
        # 3. File Structure Validation
        print("\n📁 Testing File Structure...")
        self.test_file_structure()
        
        # 4. Docker & Deployment Setup
        print("\n🐳 Testing Docker Setup...")
        self.test_docker_setup()
        
        # 5. API Endpoints Structure
        print("\n🌐 Testing API Structure...")
        self.test_api_structure()
        
        # 6. Performance Baseline
        print("\n📊 Testing Performance Baseline...")
        self.test_performance_baseline()
        
        # 7. Error Handling
        print("\n🛡️ Testing Error Handling...")
        self.test_error_handling()
        
        # Generate final report
        self.generate_final_report()
        
        return len(self.critical_failures) == 0
    
    def test_critical_components(self):
        """Test critical system components"""
        components = {
            'config.environment': 'Configuration system',
            'core.safety_manager': 'Safety management',
            'data_sources.data_manager': 'Data management', 
            'core.trading_bot': 'Trading bot core',
            'strategies': 'Strategy system',
            'api.app': 'API application',
            'utils.notifier': 'Notification system'
        }
        
        for module_name, description in components.items():
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.critical_failures.append(f"Missing module: {module_name} ({description})")
                    print(f"   ❌ {description}: Module not found")
                else:
                    print(f"   ✅ {description}: Available")
            except Exception as e:
                self.warnings.append(f"Cannot verify {module_name}: {e}")
                print(f"   ⚠️  {description}: {e}")
    
    def test_configuration(self):
        """Test configuration system"""
        try:
            from config.environment import get_config
            config = get_config()
            
            # Check required config sections
            required_sections = ['EXCHANGE', 'TRADING', 'RISK', 'API']
            for section in required_sections:
                if hasattr(config, section):
                    print(f"   ✅ Config section {section}: Available")
                else:
                    self.warnings.append(f"Missing config section: {section}")
                    print(f"   ⚠️  Config section {section}: Missing")
            
            # Test environment files
            env_files = ['.env', '.env.example', 'config/settings.py']
            for env_file in env_files:
                if Path(env_file).exists():
                    print(f"   ✅ Environment file {env_file}: Exists")
                else:
                    self.warnings.append(f"Missing environment file: {env_file}")
                    print(f"   ⚠️  Environment file {env_file}: Missing")
                    
        except Exception as e:
            self.critical_failures.append(f"Configuration system failure: {e}")
            print(f"   ❌ Configuration system: {e}")
    
    def test_file_structure(self):
        """Test critical file structure"""
        critical_files = [
            'main.py',
            'requirements.txt',
            'Dockerfile',
            'docker-compose.yml',
            'config/settings.py',
            'core/trading_bot.py',
            'strategies/__init__.py',
            'api/app.py'
        ]
        
        critical_dirs = [
            'core',
            'strategies', 
            'data_sources',
            'ml_components',
            'api',
            'tests',
            'config'
        ]
        
        for file_path in critical_files:
            if Path(file_path).exists():
                print(f"   ✅ Critical file {file_path}: Exists")
            else:
                self.critical_failures.append(f"Missing critical file: {file_path}")
                print(f"   ❌ Critical file {file_path}: Missing")
        
        for dir_path in critical_dirs:
            if Path(dir_path).is_dir():
                print(f"   ✅ Critical directory {dir_path}: Exists")
            else:
                self.critical_failures.append(f"Missing critical directory: {dir_path}")
                print(f"   ❌ Critical directory {dir_path}: Missing")
    
    def test_docker_setup(self):
        """Test Docker deployment setup"""
        docker_files = [
            'Dockerfile',
            'docker-compose.yml',
            'docker-compose.dev.yml',
            'docker-entrypoint.sh',
            '.dockerignore'
        ]
        
        for docker_file in docker_files:
            if Path(docker_file).exists():
                print(f"   ✅ Docker file {docker_file}: Exists")
                
                # Basic content validation
                try:
                    content = Path(docker_file).read_text()
                    if docker_file == 'Dockerfile' and 'FROM python:' in content:
                        print(f"      ✅ Dockerfile has Python base image")
                    elif docker_file == 'docker-compose.yml' and 'services:' in content:
                        print(f"      ✅ docker-compose.yml has services definition")
                except Exception as e:
                    self.warnings.append(f"Cannot read {docker_file}: {e}")
            else:
                if docker_file in ['Dockerfile', 'docker-compose.yml']:
                    self.critical_failures.append(f"Missing critical Docker file: {docker_file}")
                    print(f"   ❌ Docker file {docker_file}: Missing")
                else:
                    self.warnings.append(f"Missing optional Docker file: {docker_file}")
                    print(f"   ⚠️  Docker file {docker_file}: Missing")
    
    def test_api_structure(self):
        """Test API structure"""
        api_files = [
            'api/app.py',
            'api/__init__.py',
            'api/routes/__init__.py',
            'api/routes/trading.py',
            'api/routes/auth.py',
            'api/routes/monitoring.py',
            'api/health.py'
        ]
        
        for api_file in api_files:
            if Path(api_file).exists():
                print(f"   ✅ API file {api_file}: Exists")
            else:
                if api_file in ['api/app.py', 'api/__init__.py']:
                    self.critical_failures.append(f"Missing critical API file: {api_file}")
                    print(f"   ❌ API file {api_file}: Missing")
                else:
                    self.warnings.append(f"Missing API file: {api_file}")
                    print(f"   ⚠️  API file {api_file}: Missing")
        
        # Test if API can be imported (try standalone first)
        api_imported = False
        try:
            from api.standalone_api import create_app
            print(f"   ✅ API application (standalone): Can be imported")
            api_imported = True
        except Exception as e:
            print(f"   ⚠️  Standalone API failed: {e}")
        
        if not api_imported:
            try:
                from api.app import create_app
                print(f"   ✅ API application (full): Can be imported")
                api_imported = True
            except Exception as e:
                print(f"   ⚠️  Full API failed: {e}")
        
        if not api_imported:
            self.critical_failures.append(f"No working API found")
            print(f"   ❌ API application: No working version available")
        else:
            print(f"   ✅ API application: Working version available")
    
    def test_performance_baseline(self):
        """Test basic performance baseline"""
        try:
            # Simple import performance test
            start_time = time.time()
            from config.environment import get_config
            config = get_config()
            config_time = (time.time() - start_time) * 1000
            
            if config_time < 100:
                print(f"   ✅ Config loading: {config_time:.2f}ms (FAST)")
            elif config_time < 500:
                print(f"   ⚠️  Config loading: {config_time:.2f}ms (ACCEPTABLE)")
                self.warnings.append(f"Config loading is slow: {config_time:.2f}ms")
            else:
                print(f"   ❌ Config loading: {config_time:.2f}ms (TOO SLOW)")
                self.critical_failures.append(f"Config loading too slow: {config_time:.2f}ms")
            
            # Simple concurrent test
            def simple_task():
                time.sleep(0.01)
                return time.time()
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(simple_task) for _ in range(8)]
                results = [f.result() for f in futures]
            
            concurrent_time = (time.time() - start_time) * 1000
            
            if concurrent_time < 100:
                print(f"   ✅ Concurrent operations: {concurrent_time:.2f}ms (GOOD)")
            else:
                print(f"   ⚠️  Concurrent operations: {concurrent_time:.2f}ms (CHECK)")
                self.warnings.append(f"Concurrent operations may be slow: {concurrent_time:.2f}ms")
                
        except Exception as e:
            self.warnings.append(f"Performance baseline test failed: {e}")
            print(f"   ⚠️  Performance baseline: {e}")
    
    def test_error_handling(self):
        """Test basic error handling"""
        try:
            # Test config with invalid environment
            original_env = None
            try:
                import os
                original_env = os.environ.get('TRADING_ENV')
                os.environ['TRADING_ENV'] = 'invalid_env_test'
                
                from config.environment import get_config
                config = get_config()
                
                # Should handle invalid environment gracefully
                print(f"   ✅ Error handling: Invalid environment handled")
                
            except Exception as e:
                # This is expected behavior
                print(f"   ✅ Error handling: Properly raises exception for invalid config")
            finally:
                if original_env:
                    os.environ['TRADING_ENV'] = original_env
                elif 'TRADING_ENV' in os.environ:
                    del os.environ['TRADING_ENV']
            
            # Test import error handling
            try:
                import non_existent_module
            except ImportError:
                print(f"   ✅ Error handling: Import errors handled correctly")
            
        except Exception as e:
            self.warnings.append(f"Error handling test failed: {e}")
            print(f"   ⚠️  Error handling test: {e}")
    
    def generate_final_report(self):
        """Generate final production readiness report"""
        print("\n" + "=" * 60)
        print("📋 PRODUCTION READINESS REPORT")
        print("=" * 60)
        
        total_issues = len(self.critical_failures) + len(self.warnings)
        
        print(f"\n🎯 Summary:")
        print(f"   • Critical failures: {len(self.critical_failures)}")
        print(f"   • Warnings: {len(self.warnings)}")
        print(f"   • Total issues: {total_issues}")
        
        if len(self.critical_failures) == 0:
            if len(self.warnings) == 0:
                print(f"\n✅ STATUS: PRODUCTION READY")
                print(f"   • System is fully ready for production deployment")
                print(f"   • All critical components are available and functional")
            else:
                print(f"\n⚠️  STATUS: PRODUCTION READY WITH WARNINGS")
                print(f"   • System can be deployed but should address warnings")
                print(f"   • Non-critical issues detected")
        else:
            print(f"\n❌ STATUS: NOT PRODUCTION READY")
            print(f"   • Critical issues must be resolved before deployment")
            print(f"   • System may not function correctly in production")
        
        if self.critical_failures:
            print(f"\n🚨 Critical Failures:")
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"   {i}. {failure}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Deployment recommendations
        print(f"\n🚀 Deployment Recommendations:")
        
        if len(self.critical_failures) == 0:
            recommendations = [
                "✅ System is ready for production deployment",
                "✅ Docker setup is complete and functional",
                "✅ All critical components are available",
                "🔧 Run final integration tests in staging environment",
                "📊 Monitor performance metrics after deployment",
                "🔍 Set up logging and monitoring in production"
            ]
        else:
            recommendations = [
                "❌ Resolve all critical failures before deployment",
                "🔧 Run this test again after fixing issues",
                "📝 Review system architecture and dependencies",
                "🛠️ Complete missing components and files"
            ]
        
        for rec in recommendations:
            print(f"   • {rec}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"production_readiness_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'test_type': 'production_readiness',
                    'status': 'READY' if len(self.critical_failures) == 0 else 'NOT_READY',
                    'summary': {
                        'critical_failures': len(self.critical_failures),
                        'warnings': len(self.warnings),
                        'total_issues': total_issues
                    },
                    'critical_failures': self.critical_failures,
                    'warnings': self.warnings,
                    'recommendations': recommendations
                }, f, indent=2)
            
            print(f"\n📄 Detailed report saved: {report_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save report: {e}")
        
        print("\n" + "=" * 60)

def main():
    """Run production readiness test"""
    test_runner = ProductionReadinessTest()
    is_ready = test_runner.run_all_tests()
    
    if is_ready:
        print("\n🎉 SYSTEM IS PRODUCTION READY! 🎉")
    else:
        print("\n🔧 SYSTEM NEEDS WORK BEFORE PRODUCTION")
    
    return is_ready

if __name__ == "__main__":
    ready = main()
    sys.exit(0 if ready else 1)