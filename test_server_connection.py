#!/usr/bin/env python3
"""
Strato Server Connection Test
Tests SSH/HTTP connectivity to the Strato server
"""

import asyncio
import requests
import subprocess
import sys
from datetime import datetime

# Server Configuration
SERVER_IP = "85.215.183.30"
SSH_USER = "trading"  # Adjust if different
HTTP_PORT = 8000
API_PORT = 8002

def test_ping():
    """Test basic network connectivity"""
    print("🌐 Testing basic network connectivity...")
    
    try:
        # Test ping (works on most systems)
        result = subprocess.run(['ping', '-c', '1', SERVER_IP], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ Ping successful to {SERVER_IP}")
            return True
        else:
            print(f"❌ Ping failed to {SERVER_IP}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Ping timeout to {SERVER_IP}")
        return False
    except FileNotFoundError:
        # Fallback for systems without ping
        print("⚠️ Ping command not available, skipping...")
        return True

def test_ssh_connection():
    """Test SSH connectivity"""
    print(f"🔑 Testing SSH connection to {SSH_USER}@{SERVER_IP}...")
    
    try:
        # Test SSH connection
        result = subprocess.run([
            'ssh', '-o', 'ConnectTimeout=10', 
            '-o', 'BatchMode=yes',  # Non-interactive
            f'{SSH_USER}@{SERVER_IP}', 'echo "SSH OK"'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0 and "SSH OK" in result.stdout:
            print(f"✅ SSH connection successful")
            return True
        else:
            print(f"❌ SSH connection failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ SSH connection timeout")
        return False
    except FileNotFoundError:
        print("⚠️ SSH command not available")
        return False

def test_http_endpoints():
    """Test HTTP endpoints"""
    print(f"🌐 Testing HTTP endpoints...")
    
    endpoints = [
        (f"http://{SERVER_IP}:{HTTP_PORT}", "Dashboard"),
        (f"http://{SERVER_IP}:{HTTP_PORT}/health", "Health Check"),
        (f"http://{SERVER_IP}:{API_PORT}", "API Server"),
        (f"http://{SERVER_IP}:{API_PORT}/health", "API Health")
    ]
    
    results = []
    
    for url, name in endpoints:
        try:
            print(f"   Testing {name}: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ {name} - OK ({response.status_code})")
                results.append(True)
            else:
                print(f"   ⚠️ {name} - HTTP {response.status_code}")
                results.append(False)
                
        except requests.exceptions.ConnectTimeout:
            print(f"   ⏰ {name} - Timeout")
            results.append(False)
        except requests.exceptions.ConnectionError:
            print(f"   ❌ {name} - Connection refused")
            results.append(False)
        except Exception as e:
            print(f"   ❌ {name} - Error: {e}")
            results.append(False)
    
    return any(results)

def get_server_status():
    """Get detailed server status via SSH"""
    print(f"📊 Getting server status...")
    
    commands = [
        ('uptime', 'System uptime'),
        ('ps aux | grep python | head -5', 'Python processes'),
        ('netstat -tlnp | grep :800', 'Port status'),
        ('df -h /', 'Disk space'),
        ('free -h', 'Memory usage')
    ]
    
    for cmd, description in commands:
        print(f"   {description}:")
        
        try:
            result = subprocess.run([
                'ssh', '-o', 'ConnectTimeout=10',
                f'{SSH_USER}@{SERVER_IP}', cmd
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"   ✅ {output}")
            else:
                print(f"   ❌ Command failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
def test_bot_directory():
    """Check if bot directory exists and has latest files"""
    print(f"📁 Checking bot directory...")
    
    commands = [
        ('ls -la /home/trading/altcoin_trading_bot/', 'Bot directory'),
        ('cd /home/trading/altcoin_trading_bot && git log --oneline -3', 'Recent commits'),
        ('cd /home/trading/altcoin_trading_bot && git status', 'Git status'),
        ('cd /home/trading/altcoin_trading_bot && ls -la *.py | head -10', 'Python files')
    ]
    
    for cmd, description in commands:
        print(f"   {description}:")
        
        try:
            result = subprocess.run([
                'ssh', '-o', 'ConnectTimeout=10',
                f'{SSH_USER}@{SERVER_IP}', cmd
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                # Show only first few lines to avoid clutter
                lines = output.split('\n')[:3]
                for line in lines:
                    print(f"   ✅ {line}")
            else:
                print(f"   ❌ {description} - Command failed")
                
        except Exception as e:
            print(f"   ❌ {description} - Error: {e}")

def main():
    """Main test execution"""
    print("🎯 STRATO SERVER CONNECTION TEST")
    print("=" * 50)
    print(f"Server: {SERVER_IP}")
    print(f"SSH User: {SSH_USER}")
    print(f"Dashboard Port: {HTTP_PORT}")
    print(f"API Port: {API_PORT}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Basic connectivity
    if test_ping():
        tests_passed += 1
    
    # Test 2: SSH connection
    if test_ssh_connection():
        tests_passed += 1
        
        # Additional SSH tests if connection works
        get_server_status()
        test_bot_directory()
    
    # Test 3: HTTP endpoints
    if test_http_endpoints():
        tests_passed += 1
    
    print()
    print("=" * 50)
    print(f"SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Server is ready.")
    elif tests_passed >= 2:
        print("⚠️ Some tests failed, but server appears accessible.")
        print("You can proceed with manual deployment.")
    else:
        print("❌ Multiple tests failed. Check server accessibility.")
        
    print()
    print("Next steps:")
    if tests_passed >= 1:
        print("1. Run update script: ./strato_update_bot.sh")
        print("2. Or use Windows script: ./strato_update_windows.ps1")
        print(f"3. Access dashboard: http://{SERVER_IP}:{HTTP_PORT}")
    else:
        print("1. Verify server IP and credentials")
        print("2. Check network/VPN connection")
        print("3. Ensure SSH key is configured")

if __name__ == "__main__":
    main()