#!/usr/bin/env python3
"""
🔍 Janics Freedom Factory - Connection Diagnostic Tool
Vollständige Diagnose aller Connection-Probleme
"""

import requests
import subprocess
import psutil
import time
import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime

class ConnectionDiagnostic:
    def __init__(self):
        self.dashboard_url = "http://localhost:3000"  # Frontend
        self.api_url = "http://localhost:8080"        # Backend API  
        self.bot_path = os.getcwd()                   # Bot Directory
        
    def diagnose_connection_issues(self) -> Dict[str, Any]:
        """Vollständige Diagnose aller Connection-Probleme"""
        
        print("🔍 JANICS FREEDOM FACTORY - CONNECTION DIAGNOSTIC")
        print("=" * 50)
        
        results = {}
        
        # 1. Prüfe Backend API
        api_status = self.check_backend_api()
        print(f"Backend API: {'✅ RUNNING' if api_status else '❌ DOWN'}")
        results['api'] = api_status
        
        # 2. Prüfe Bot Process
        bot_status = self.check_bot_process()
        print(f"Bot Process: {'✅ RUNNING' if bot_status else '❌ NOT RUNNING'}")
        results['bot'] = bot_status
        
        # 3. Prüfe Database
        db_status = self.check_database()
        print(f"Database: {'✅ CONNECTED' if db_status else '❌ DISCONNECTED'}")
        results['database'] = db_status
        
        # 4. Prüfe Frontend-Backend Communication
        frontend_status = self.check_frontend_backend()
        print(f"Frontend↔Backend: {'✅ OK' if frontend_status else '❌ FAILED'}")
        results['frontend'] = frontend_status
        
        # 5. Teste API Endpoints
        api_endpoints = self.test_api_endpoints()
        print(f"API Endpoints: {'✅ OK' if api_endpoints['success'] else '❌ FAILED'}")
        results['api_endpoints'] = api_endpoints
        
        return results
    
    def check_backend_api(self) -> bool:
        """Prüfe ob Backend API läuft"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"   API Error: {e}")
            return False
    
    def check_bot_process(self) -> bool:
        """Prüfe ob Bot-Process läuft"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    # Look for common bot patterns
                    if any(pattern in cmdline.lower() for pattern in [
                        'main.py', 'trading_bot.py', 'altcoin_trading_bot'
                    ]):
                        # Exclude this diagnostic script itself
                        if 'debug_connection.py' not in cmdline:
                            return True
            return False
        except Exception as e:
            print(f"   Bot Process Error: {e}")
            return False
    
    def check_database(self) -> bool:
        """Prüfe Database Connection"""
        try:
            db_path = os.path.join(self.bot_path, 'db', 'trading_bot.db')
            return os.path.exists(db_path)
        except Exception as e:
            print(f"   Database Error: {e}")
            return False
    
    def check_frontend_backend(self) -> bool:
        """Prüfe Frontend-Backend Communication"""
        try:
            # Check if frontend is accessible
            response = requests.get(self.dashboard_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"   Frontend Error: {e}")
            return False
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Teste wichtige API Endpoints"""
        endpoints = {
            '/api/v1/orchestrator/status': 'GET',
            '/api/v1/orchestrator/portfolio': 'GET', 
            '/api/v1/orchestrator/strategies': 'GET'
        }
        
        results = {'success': True, 'endpoints': {}}
        
        for endpoint, method in endpoints.items():
            try:
                url = f"{self.api_url}{endpoint}"
                if method == 'GET':
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, timeout=5)
                
                success = response.status_code in [200, 201]
                results['endpoints'][endpoint] = {
                    'status': response.status_code,
                    'success': success,
                    'response_size': len(response.text)
                }
                
                if not success:
                    results['success'] = False
                    
            except Exception as e:
                results['endpoints'][endpoint] = {
                    'success': False,
                    'error': str(e)
                }
                results['success'] = False
        
        return results
    
    def start_backend_api(self):
        """Starte Backend API"""
        try:
            print("🚀 Starting Backend API...")
            api_path = os.path.join(self.bot_path, 'api', 'app.py')
            if os.path.exists(api_path):
                subprocess.Popen([
                    sys.executable, api_path
                ], cwd=os.path.dirname(api_path))
                print("   Backend API started")
            else:
                print(f"   ❌ API script not found: {api_path}")
        except Exception as e:
            print(f"   ❌ Failed to start API: {e}")
    
    def start_trading_bot(self):
        """Starte Trading Bot"""
        try:
            print("🤖 Starting Trading Bot...")
            # Try different common bot entry points
            bot_scripts = ['main.py', 'main_fixed.py', 'trading_bot.py']
            
            for script in bot_scripts:
                script_path = os.path.join(self.bot_path, script)
                if os.path.exists(script_path):
                    subprocess.Popen([
                        sys.executable, script_path
                    ], cwd=self.bot_path)
                    print(f"   Trading bot started: {script}")
                    return
            
            print("   ❌ No bot script found")
        except Exception as e:
            print(f"   ❌ Failed to start bot: {e}")
    
    def fix_connection_issues(self):
        """Automatische Reparatur der Connection-Issues"""
        
        print("\n🔧 FIXING CONNECTION ISSUES...")
        
        # 1. Starte Backend API falls nicht läuft
        if not self.check_backend_api():
            self.start_backend_api()
            time.sleep(5)  # Wait for startup
            
        # 2. Starte Bot falls nicht läuft
        if not self.check_bot_process():
            self.start_trading_bot()
            time.sleep(3)  # Wait for startup
            
        print("\n⏰ Waiting 10 seconds for services to initialize...")
        time.sleep(10)
        
        # 3. Teste erneut
        print("\n🔍 RE-TESTING AFTER FIXES...")
        final_status = self.diagnose_connection_issues()
        
        return final_status
    
    def get_system_info(self):
        """Sammle System-Informationen"""
        print("\n📊 SYSTEM INFORMATION")
        print("-" * 30)
        
        # Python Version
        print(f"Python: {sys.version}")
        
        # Current Directory
        print(f"Working Dir: {os.getcwd()}")
        
        # Available Files
        try:
            py_files = [f for f in os.listdir('.') if f.endswith('.py')][:10]
            print(f"Python Files: {py_files}")
        except:
            print("Python Files: Error reading directory")
        
        # System Resources
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            print(f"CPU: {cpu_percent}%")
            print(f"Memory: {memory.percent}% used")
        except:
            print("System Resources: Error reading")
        
        # Network Ports
        try:
            connections = psutil.net_connections()
            listening_ports = [conn.laddr.port for conn in connections 
                             if conn.status == 'LISTEN' and conn.laddr.port in [3000, 8080, 5000]]
            print(f"Listening Ports: {listening_ports}")
        except:
            print("Network Ports: Error reading")

def main():
    """Main diagnostic function"""
    diagnostic = ConnectionDiagnostic()
    
    # Show system info
    diagnostic.get_system_info()
    
    # Initial diagnosis
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting diagnostic...")
    results = diagnostic.diagnose_connection_issues()
    
    # Show results summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("-" * 30)
    for component, status in results.items():
        if component == 'api_endpoints':
            success = status.get('success', False)
            print(f"{component}: {'✅ PASS' if success else '❌ FAIL'}")
        else:
            print(f"{component}: {'✅ PASS' if status else '❌ FAIL'}")
    
    # Auto-fix if needed
    if not all([results.get('api', False), results.get('bot', False)]):
        print(f"\n🔧 Issues detected, attempting auto-fix...")
        diagnostic.fix_connection_issues()

if __name__ == "__main__":
    main()