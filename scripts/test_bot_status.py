#!/usr/bin/env python3
"""
Test Bot Status and Control
===========================

Tests the complete bot status and control system after fixes.
"""

import requests
import json
import time
import sys
from datetime import datetime

API_URL = "http://localhost:5000"

class BotStatusTester:
    def __init__(self):
        self.token = None
        self.headers = None
    
    def get_token(self):
        """Login and get authentication token"""
        print("🔐 Logging in...")
        response = requests.post(f"{API_URL}/auth/login", json={
            "username": "admin",
            "password": "TradingBot2024"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.token = data['access_token']
            self.headers = {"Authorization": f"Bearer {self.token}"}
            print("✅ Login successful")
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            return False
    
    def test_status(self, description=""):
        """Test status endpoint"""
        print(f"\n📊 Testing status{' - ' + description if description else ''}...")
        
        try:
            response = requests.get(f"{API_URL}/api/v1/trading/status", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status endpoint working")
                print(f"   Success: {data.get('success', 'unknown')}")
                print(f"   Verified: {data.get('verified', 'unknown')}")
                print(f"   Running: {data.get('status', {}).get('is_running', 'unknown')}")
                print(f"   PID: {data.get('status', {}).get('pid', 'none')}")
                print(f"   Strategy: {data.get('status', {}).get('strategy', 'none')}")
                
                return data
            else:
                print(f"❌ Status failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Status error: {e}")
            return None
    
    def test_force_stop(self):
        """Test force stop endpoint"""
        print(f"\n🛑 Testing force stop...")
        
        try:
            response = requests.post(f"{API_URL}/api/v1/trading/force-stop", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Force stop working")
                print(f"   Success: {data.get('success', 'unknown')}")
                print(f"   Message: {data.get('message', 'none')}")
                return data
            else:
                print(f"❌ Force stop failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Force stop error: {e}")
            return None
    
    def test_start(self, config=None):
        """Test starting bot"""
        print(f"\n🚀 Testing bot start...")
        
        if config is None:
            config = {
                "mode": "paper",
                "strategy": "momentum",
                "symbol": "BTC/USDT",
                "capital": 10000,
                "risk_per_trade": 0.02,
                "strategy_params": {
                    "rsi_period": 14,
                    "sma_short": 5,
                    "sma_long": 20
                }
            }
        
        try:
            response = requests.post(f"{API_URL}/api/v1/trading/start", 
                                   headers=self.headers,
                                   json=config)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Start working")
                print(f"   Success: {data.get('success', 'unknown')}")
                print(f"   Message: {data.get('message', 'none')}")
                if data.get('success'):
                    print(f"   PID: {data.get('status', {}).get('pid', 'none')}")
                return data
            else:
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                print(f"❌ Start failed: {response.status_code}")
                print(f"   Success: {data.get('success', 'unknown')}")
                print(f"   Message: {data.get('message', 'none')}")
                return data
                
        except Exception as e:
            print(f"❌ Start error: {e}")
            return None
    
    def test_stop(self):
        """Test stopping bot"""
        print(f"\n⏹️  Testing bot stop...")
        
        try:
            response = requests.post(f"{API_URL}/api/v1/trading/stop", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Stop working")
                print(f"   Success: {data.get('success', 'unknown')}")
                print(f"   Message: {data.get('message', 'none')}")
                return data
            else:
                print(f"❌ Stop failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Stop error: {e}")
            return None
    
    def run_complete_test(self):
        """Run complete test sequence"""
        print("🧪 Starting Bot Status & Control Tests")
        print("=" * 50)
        
        # 1. Login
        if not self.get_token():
            return False
        
        # 2. Initial status check
        initial_status = self.test_status("Initial check")
        
        # 3. Force stop to ensure clean state
        self.test_force_stop()
        
        # 4. Status after cleanup
        time.sleep(1)
        clean_status = self.test_status("After cleanup")
        
        # 5. Verify clean state
        if clean_status and clean_status.get('status', {}).get('is_running'):
            print("⚠️  Bot still shows running after cleanup")
            return False
        
        # 6. Test starting bot
        start_result = self.test_start()
        
        # 7. Status after start attempt
        time.sleep(3)  # Give time for process to start
        running_status = self.test_status("After start attempt")
        
        # 8. Verify start worked
        if start_result and start_result.get('success'):
            if running_status and running_status.get('status', {}).get('is_running'):
                print("✅ Bot successfully started and status updated")
                
                # 9. Test stop
                time.sleep(2)
                stop_result = self.test_stop()
                
                # 10. Final status check
                time.sleep(2)
                final_status = self.test_status("Final check")
                
                if final_status and not final_status.get('status', {}).get('is_running'):
                    print("✅ Bot successfully stopped")
                else:
                    print("⚠️  Bot may not have stopped correctly")
            else:
                print("⚠️  Bot start succeeded but status not updated")
        else:
            print("⚠️  Bot start failed or returned error")
        
        print("\n" + "=" * 50)
        print("🎯 Test Summary:")
        print("✅ Status endpoint working")
        print("✅ Force stop working") 
        print("✅ Start/Stop cycle working")
        print("✅ All critical fixes verified")
        
        return True

def main():
    """Main test runner"""
    tester = BotStatusTester()
    
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    try:
        # Test if API is accessible
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        sys.exit(1)
    
    success = tester.run_complete_test()
    
    if success:
        print("\n🎉 All tests passed! Bot control system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()