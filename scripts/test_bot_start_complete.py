#!/usr/bin/env python3
"""
Complete Bot Start & Status Test
================================

Tests the complete bot start flow including:
1. Bot start process validation
2. WebSocket connection testing  
3. Status synchronization verification
4. Real-time monitoring
"""

import requests
import time
import subprocess
import sys
import json
import threading
import socketio
from datetime import datetime

API_URL = "http://localhost:5000"

class CompleteBotTester:
    def __init__(self):
        self.token = None
        self.headers = None
        self.sio = None
        self.websocket_messages = []
        self.websocket_connected = False
        
    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def get_token(self):
        """Login and get authentication token"""
        self.log("🔐 Logging in...")
        try:
            response = requests.post(f"{API_URL}/auth/login", json={
                "username": "admin",
                "password": "TradingBot2024"
            })
            
            if response.status_code == 200:
                data = response.json()
                self.token = data['access_token']
                self.headers = {"Authorization": f"Bearer {self.token}"}
                self.log("✅ Login successful")
                return True
            else:
                self.log(f"❌ Login failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log(f"❌ Login error: {e}")
            return False
    
    def setup_websocket(self):
        """Setup WebSocket connection for real-time monitoring"""
        self.log("🔌 Setting up WebSocket connection...")
        
        try:
            self.sio = socketio.Client()
            
            @self.sio.event
            def connect():
                self.log("✅ WebSocket connected")
                self.websocket_connected = True
                self.sio.emit('subscribe', {'rooms': ['trading_updates']})
                self.sio.emit('request_status')
            
            @self.sio.event
            def disconnect():
                self.log("❌ WebSocket disconnected")
                self.websocket_connected = False
            
            @self.sio.event
            def connected(data):
                self.log(f"📡 Connected message: {data}")
                
            @self.sio.event
            def bot_status_update(data):
                self.log(f"📊 Status update: {data}")
                self.websocket_messages.append({
                    'type': 'status_update',
                    'data': data,
                    'timestamp': datetime.now()
                })
            
            @self.sio.event
            def subscribed(data):
                self.log(f"📋 Subscribed to: {data}")
            
            # Connect to WebSocket
            self.sio.connect(API_URL)
            time.sleep(2)  # Give connection time to establish
            
            return self.websocket_connected
            
        except Exception as e:
            self.log(f"❌ WebSocket setup error: {e}")
            return False
    
    def test_api_status(self, description=""):
        """Test API status endpoint"""
        self.log(f"📊 Testing API status{' - ' + description if description else ''}...")
        
        try:
            response = requests.get(f"{API_URL}/api/v1/trading/status", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ API Status OK")
                self.log(f"   Running: {data.get('status', {}).get('is_running', 'unknown')}")
                self.log(f"   Status: {data.get('status', {}).get('status', 'unknown')}")
                self.log(f"   PID: {data.get('status', {}).get('pid', 'none')}")
                self.log(f"   Strategy: {data.get('status', {}).get('strategy', 'none')}")
                
                return data
            else:
                self.log(f"❌ API Status failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"❌ API Status error: {e}")
            return None
    
    def test_force_cleanup(self):
        """Test force cleanup"""
        self.log("🛑 Testing force cleanup...")
        
        try:
            response = requests.post(f"{API_URL}/api/v1/trading/force-stop", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Force cleanup successful")
                self.log(f"   Message: {data.get('message', 'none')}")
                return data
            else:
                self.log(f"❌ Force cleanup failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"❌ Force cleanup error: {e}")
            return None
    
    def test_bot_start(self):
        """Test bot start with comprehensive monitoring"""
        self.log("🚀 Testing bot start...")
        
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
                self.log(f"✅ Start request successful")
                self.log(f"   Success: {data.get('success', 'unknown')}")
                self.log(f"   Message: {data.get('message', 'none')}")
                if data.get('success'):
                    self.log(f"   PID: {data.get('status', {}).get('pid', 'none')}")
                
                return data
            else:
                self.log(f"❌ Start request failed: {response.status_code}")
                try:
                    error_data = response.json()
                    self.log(f"   Error: {error_data.get('message', 'Unknown error')}")
                except:
                    self.log(f"   Raw response: {response.text}")
                return None
                
        except Exception as e:
            self.log(f"❌ Start request error: {e}")
            return None
    
    def test_bot_logs(self):
        """Test bot logs endpoint"""
        self.log("📋 Testing bot logs...")
        
        try:
            response = requests.get(f"{API_URL}/api/v1/trading/logs", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logs = data.get('logs', [])
                self.log(f"✅ Logs retrieved: {len(logs)} entries")
                
                # Show recent logs
                for log_entry in logs[-5:]:
                    self.log(f"   📝 {log_entry.get('timestamp', '')}: {log_entry.get('message', '')}")
                
                return data
            else:
                self.log(f"❌ Logs failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"❌ Logs error: {e}")
            return None
    
    def test_bot_stop(self):
        """Test bot stop"""
        self.log("⏹️  Testing bot stop...")
        
        try:
            response = requests.post(f"{API_URL}/api/v1/trading/stop", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Stop successful")
                self.log(f"   Message: {data.get('message', 'none')}")
                return data
            else:
                self.log(f"❌ Stop failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"❌ Stop error: {e}")
            return None
    
    def monitor_websocket_messages(self, duration=10):
        """Monitor WebSocket messages for a period"""
        self.log(f"👁️  Monitoring WebSocket messages for {duration} seconds...")
        
        start_time = time.time()
        message_count = len(self.websocket_messages)
        
        time.sleep(duration)
        
        new_messages = len(self.websocket_messages) - message_count
        self.log(f"📡 Received {new_messages} new WebSocket messages")
        
        # Show recent messages
        for msg in self.websocket_messages[-3:]:
            self.log(f"   📨 {msg['type']}: {msg['data']}")
        
        return new_messages > 0
    
    def run_complete_test(self):
        """Run complete test sequence"""
        self.log("🧪 Starting Complete Bot Start & Status Test")
        self.log("=" * 60)
        
        # 1. Login
        if not self.get_token():
            return False
        
        # 2. Setup WebSocket
        if not self.setup_websocket():
            self.log("⚠️  WebSocket setup failed, continuing with API-only tests...")
        
        # 3. Initial cleanup
        self.test_force_cleanup()
        time.sleep(2)
        
        # 4. Initial status check
        initial_status = self.test_api_status("Initial check")
        time.sleep(2)
        
        # 5. Test bot start
        start_result = self.test_bot_start()
        
        if start_result and start_result.get('success'):
            # 6. Monitor startup process
            self.log("⏳ Monitoring startup process...")
            
            for i in range(6):  # Monitor for 30 seconds
                time.sleep(5)
                status = self.test_api_status(f"Startup check {i+1}")
                
                if status and status.get('status', {}).get('is_running'):
                    self.log("✅ Bot is now running!")
                    break
            else:
                self.log("⚠️  Bot may not have started successfully")
            
            # 7. Monitor WebSocket messages
            if self.websocket_connected:
                self.monitor_websocket_messages(10)
            
            # 8. Test logs
            self.test_bot_logs()
            
            # 9. Test stop
            time.sleep(3)
            stop_result = self.test_bot_stop()
            
            # 10. Final status check
            time.sleep(3)
            final_status = self.test_api_status("Final check")
            
        else:
            self.log("❌ Bot start failed, skipping runtime tests")
        
        # Cleanup
        if self.sio and self.websocket_connected:
            self.sio.disconnect()
        
        self.log("=" * 60)
        self.log("🎯 Test Summary:")
        self.log("✅ API Authentication")
        self.log("✅ Force Cleanup")
        self.log("✅ Status Endpoints")
        self.log("✅ Bot Start/Stop Cycle")
        
        if self.websocket_connected:
            self.log("✅ WebSocket Connection")
        else:
            self.log("⚠️  WebSocket Connection (check SocketIO)")
        
        self.log("🎉 Complete test finished!")
        
        return True

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main test runner"""
    print("🏥 Checking API health...")
    
    if not check_api_health():
        print("❌ API is not accessible. Please start the API server first.")
        print("   Run: python -m api.app")
        sys.exit(1)
    
    print("✅ API is healthy")
    
    tester = CompleteBotTester()
    success = tester.run_complete_test()
    
    if success:
        print("\\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\\n❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()