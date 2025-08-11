#!/usr/bin/env python3
"""
WebSocket Test Script
====================

Tests WebSocket functionality with authentication.
"""

import socketio
import json
import requests
import sys
import time

def test_websocket_connection():
    """Test WebSocket connection with authentication"""
    
    # First, get an authentication token
    print("🔐 Getting authentication token...")
    
    try:
        auth_response = requests.post(
            'http://localhost:5000/auth/login',
            json={'username': 'admin', 'password': 'TradingBot2024'},
            headers={'Content-Type': 'application/json'}
        )
        
        if auth_response.status_code != 200:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            print(f"Response: {auth_response.text}")
            return False
            
        token_data = auth_response.json()
        access_token = token_data['access_token']
        print(f"✅ Got access token: {access_token[:50]}...")
        
    except Exception as e:
        print(f"❌ Failed to get auth token: {e}")
        return False
    
    # Now test WebSocket connection
    print("\n🌐 Testing WebSocket connection...")
    
    sio = socketio.Client()
    
    @sio.event
    def connect():
        print("✅ WebSocket connected!")
        
    @sio.event
    def connected(data):
        print(f"✅ Connection confirmed: {data}")
        
        # Test subscription
        print("📡 Testing subscription...")
        sio.emit('subscribe', {'channels': ['trading_updates', 'market_data']})
        
    @sio.event
    def subscription_confirmed(data):
        print(f"✅ Subscription confirmed: {data}")
        
        # Test ping/pong
        print("🏓 Testing ping/pong...")
        sio.emit('ping')
        
    @sio.event
    def pong(data):
        print(f"✅ Pong received: {data}")
        
        # Test status request
        print("📊 Requesting status...")
        sio.emit('get_status')
        
    @sio.event
    def status_update(data):
        print(f"✅ Status received: {data}")
        
        # Test complete
        print("✅ All WebSocket tests passed!")
        sio.disconnect()
        
    @sio.event
    def error(data):
        print(f"❌ WebSocket error: {data}")
        
    @sio.event
    def disconnect():
        print("👋 WebSocket disconnected")
    
    try:
        # Connect with authentication
        sio.connect(
            'http://localhost:5000',
            auth={'token': access_token},
            wait_timeout=10
        )
        
        # Wait for events
        time.sleep(5)
        
        if sio.connected:
            sio.disconnect()
            
        return True
        
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

def test_websocket_broadcast():
    """Test WebSocket broadcast functionality"""
    print("\n📢 Testing WebSocket broadcast...")
    
    try:
        # Test broadcast to all users
        response = requests.post(
            'http://localhost:5000/api/v1/trading/broadcast-test',
            headers={'Content-Type': 'application/json'},
            json={'message': 'Test broadcast', 'type': 'info'}
        )
        
        if response.status_code == 200:
            print("✅ Broadcast test endpoint works")
        else:
            print(f"⚠️ Broadcast endpoint returned: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Broadcast test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting WebSocket Tests")
    print("="*50)
    
    success = test_websocket_connection()
    
    if success:
        test_websocket_broadcast()
        print("\n✅ WebSocket testing completed successfully!")
    else:
        print("\n❌ WebSocket testing failed!")
        sys.exit(1)