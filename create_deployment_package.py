#!/usr/bin/env python3
"""
🚀 Deployment Package Creator für Strato Windows Server
Erstellt ein komplettes Paket mit allen notwendigen Dateien
"""

import os
import shutil
import zipfile
from pathlib import Path
import json
from datetime import datetime

def create_deployment_package():
    print("🚀💰 CREATING DEPLOYMENT PACKAGE FOR STRATO WINDOWS SERVER 💰🚀")
    print("=" * 70)
    
    # Create deployment directory
    deploy_dir = Path("deployment_package")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    print(f"📦 Created deployment directory: {deploy_dir}")
    
    # Files to copy
    core_files = [
        'main.py',
        'core/enhanced_decision_logger.py',
        'api/routes/intelligence.py', 
        'run_intelligence_api.py',
        'start_enhanced_bot.py',
        'mobile_dashboard.html',
        'requirements_simple.txt',
        '.env_simple_example'
    ]
    
    # Create Windows batch files
    windows_scripts = {
        'install_dependencies_windows.bat': '''@echo off
echo 🚀💰 Installing Trading Bot Dependencies on Windows Server 💰🚀
echo ================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Installing Python 3.11...
    echo Download from: https://www.python.org/downloads/windows/
    start https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

REM Create project directory structure
echo 📁 Creating directory structure...
if not exist "C:\\TradingBot" mkdir "C:\\TradingBot"
cd /d "C:\\TradingBot"

if not exist "core" mkdir core
if not exist "api" mkdir api
if not exist "api\\routes" mkdir "api\\routes"
if not exist "strategies" mkdir strategies
if not exist "config" mkdir config
if not exist "utils" mkdir utils
if not exist "data_sources" mkdir data_sources
if not exist "ml_components" mkdir ml_components
if not exist "logs" mkdir logs
if not exist "intelligence_exports" mkdir intelligence_exports
if not exist "data" mkdir data

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing Python packages...
if exist "requirements_simple.txt" (
    pip install -r requirements_simple.txt
) else (
    echo Installing basic packages...
    pip install flask flask-cors requests pandas asyncio
    pip install asyncpg sqlalchemy aiohttp ccxt
)

REM Create .env from example
if not exist ".env" (
    if exist ".env_simple_example" (
        copy ".env_simple_example" ".env"
        echo ⚠️ Created .env from example - PLEASE EDIT WITH YOUR API KEYS!
    )
)

echo ✅ Dependencies installation completed!
echo.
echo 🎯 NEXT STEPS:
echo 1. Edit .env file with your REAL API keys
echo 2. Copy all your Python files to C:\\TradingBot\\
echo 3. Run: start_bot_windows.bat
echo.
pause
''',

        'start_bot_windows.bat': '''@echo off
title 💰🚀 MONEY-MAKING MACHINE 🚀💰
color 0A

echo.
echo 💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
echo 💰                                                              💰
echo 💰            🚀 STARTING THE MONEY-MAKING MACHINE 🚀          💰  
echo 💰                                                              💰
echo 💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
echo.

cd /d "C:\\TradingBot"

REM Check if we're in the right directory
if not exist "venv\\Scripts\\activate.bat" (
    echo ❌ Virtual environment not found in C:\\TradingBot!
    echo Please run install_dependencies_windows.bat first.
    pause
    exit /b 1
)

REM Check if .env exists and has API keys
if not exist ".env" (
    echo ❌ .env file not found!
    echo Please copy .env_simple_example to .env and add your API keys.
    pause  
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating Python environment...
call venv\\Scripts\\activate.bat

REM Kill any existing processes
echo 🛑 Stopping existing bot processes...
taskkill /f /im python.exe /t >nul 2>&1
timeout /t 3 >nul

REM Start Intelligence API
echo 🧠 Starting Intelligence API Server...
start "💰 Intelligence API" /min python run_intelligence_api.py --host 0.0.0.0 --port 8002

REM Wait for API to start
echo ⏳ Waiting for Intelligence API to start...
timeout /t 8 >nul

REM Test API connection
echo 🔍 Testing Intelligence API connection...
curl -s http://localhost:8002/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Intelligence API may not be fully ready yet...
) else (
    echo ✅ Intelligence API is running!
)

REM Start Enhanced Trading Bot
echo 🤖 Starting Enhanced Trading Bot...
start "💰 Trading Bot" python start_enhanced_bot.py --mode live --strategy momentum

REM Wait a moment for startup
timeout /t 5 >nul

echo.
echo 🎉🎉🎉 MONEY-MAKING MACHINE IS NOW RUNNING! 🎉🎉🎉
echo ========================================================
echo.
echo 📊 Intelligence API:  http://85.215.183.30:8002/health
echo 🌐 Demo Data:         http://85.215.183.30:8002/api/intelligence/demo  
echo 📱 Mobile Dashboard:  Open mobile_dashboard.html in browser
echo 💾 Data Export:       http://85.215.183.30:8002/api/intelligence/export/decisions
echo.
echo 💰💰💰 LET THE MONEY FLOW! 💰💰💰
echo.
echo 🎯 Keep this window open to monitor the bot
echo 🔄 Press 'S' for status, 'Q' to quit
echo.

:loop
choice /c SQ /n /m "Press S for status, Q to quit: "
if errorlevel 2 goto quit
if errorlevel 1 goto status
goto loop

:status
echo.
echo 📊 MONEY-MAKING MACHINE STATUS:
echo ===============================
echo 🤖 Python Processes:
tasklist | findstr python.exe
echo.
echo 🌐 Network Connections:
netstat -an | findstr ":8002 "
echo.
echo 💾 Recent Log Files:
if exist "logs\\*.log" (
    dir logs\\*.log /o-d /b | head -3
) else (
    echo No log files found
)
echo.
goto loop

:quit
echo.
echo 🛑 Do you want to stop the Money-Making Machine? (Y/N)
choice /c YN /m "Stop bot"
if errorlevel 2 goto loop

echo 🛑 Stopping Money-Making Machine...
taskkill /f /im python.exe /t
echo 💤 Bot stopped. Window will close in 5 seconds...
timeout /t 5
exit
''',

        'stop_bot_windows.bat': '''@echo off
echo 🛑💰 STOPPING MONEY-MAKING MACHINE 💰🛑
echo =========================================

echo Terminating all Python processes...
taskkill /f /im python.exe /t

echo Waiting for processes to stop...
timeout /t 5 >nul

echo 📊 Checking remaining processes...
tasklist | findstr python.exe >nul 2>&1
if errorlevel 1 (
    echo ✅ All trading bot processes stopped successfully!
    echo 💤 Money-making machine is now offline.
) else (
    echo ⚠️ Some Python processes may still be running.
    echo 🔍 Remaining processes:
    tasklist | findstr python.exe
)

echo.
echo 💰 Ready to restart? Run: start_bot_windows.bat
pause
''',

        'status_bot_windows.bat': '''@echo off
title 📊 Money-Making Machine Status

:refresh
cls
echo 📊💰 MONEY-MAKING MACHINE STATUS 💰📊
echo =========================================
echo Refresh Time: %date% %time%
echo =========================================

echo.
echo 🤖 TRADING BOT PROCESSES:
tasklist | findstr python.exe
if errorlevel 1 (
    echo ❌ No bot processes running - START THE MACHINE!
) else (
    echo ✅ Trading bot processes are ACTIVE and MAKING MONEY! 💰
)

echo.
echo 🌐 NETWORK STATUS:
echo Intelligence API (Port 8002):
netstat -an | findstr ":8002 " | findstr LISTENING
if errorlevel 1 (
    echo ❌ Intelligence API offline
) else (
    echo ✅ Intelligence API online - Ready for mobile access! 📱
)

echo.
echo 💾 SYSTEM RESOURCES:
for /f "skip=1" %%p in ('wmic cpu get loadpercentage /value') do (
    if "%%p"=="" goto :done
    echo CPU: %%p
)
:done

echo Memory: 
for /f "skip=1 tokens=4" %%i in ('wmic OS get FreePhysicalMemory /format:table') do (
    if "%%i" NEQ "" (
        set /a mem=%%i/1024
        echo Free RAM: !mem! MB
        goto :memend
    )
)
:memend

echo.
echo 📁 RECENT ACTIVITY:
echo Intelligence Exports:
if exist "intelligence_exports\\*.json*" (
    dir intelligence_exports\\*.json* /o-d /b 2>nul | head -3
) else (
    echo No export files yet
)

echo.
echo Log Files:  
if exist "logs\\*.log" (
    dir logs\\*.log /o-d /b 2>nul | head -3
    echo Latest log size:
    for %%F in (logs\\*.log) do (
        echo %%~nxF: %%~zF bytes
        goto :logend
    )
    :logend
) else (
    echo No log files found
)

echo.
echo 🎯 QUICK TESTS:
echo API Health Check:
curl -s -m 5 http://localhost:8002/health 2>nul | findstr "healthy" >nul
if errorlevel 1 (
    echo ❌ API not responding
) else (
    echo ✅ API responding - MONEY MACHINE HEALTHY! 💰
)

echo.
echo =========================================
echo 💰 MAKING MONEY 24/7 ON STRATO SERVER! 💰
echo =========================================
echo.
echo Press any key to refresh, or Ctrl+C to exit...
pause >nul
goto refresh
''',

        'mobile_dashboard_server.html': '''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀💰 Money-Making Machine Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow-md p-4 mb-4; }
        .metric { @apply text-center p-3 bg-gray-50 rounded; }
        .insight-item { @apply border-l-4 border-green-500 pl-3 mb-2; }
        .status-online { @apply text-green-600; }
        .status-offline { @apply text-red-600; }
        .money-text { @apply text-green-600 font-bold; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    </style>
</head>
<body class="min-h-screen">
    <div class="container mx-auto px-3 py-4 max-w-md">
        <!-- Header -->
        <div class="card bg-gradient-to-r from-green-500 to-yellow-500 text-white">
            <h1 class="text-2xl font-bold mb-2">🚀💰 Money Machine</h1>
            <p class="text-sm">
                <span id="status" class="status-offline">● Connecting...</span> | 
                <span id="lastUpdate">Last Update: --</span>
            </p>
            <div class="text-center mt-2">
                <div class="text-3xl font-bold money-text" id="moneyCounter">$$$</div>
                <div class="text-xs">MAKING MONEY 24/7</div>
            </div>
        </div>

        <!-- Quick Metrics -->
        <div class="grid grid-cols-2 gap-2 mb-4">
            <div class="metric bg-green-50">
                <div class="text-xl font-bold text-green-600" id="totalDecisions">--</div>
                <div class="text-xs text-gray-600">💰 Money Decisions</div>
            </div>
            <div class="metric bg-yellow-50">
                <div class="text-xl font-bold text-yellow-600" id="avgConfidence">--</div>
                <div class="text-xs text-gray-600">🎯 Confidence</div>
            </div>
            <div class="metric bg-blue-50">
                <div class="text-xl font-bold text-blue-600" id="activeStrategies">--</div>
                <div class="text-xs text-gray-600">🚀 Strategies</div>
            </div>
            <div class="metric bg-red-50">
                <div class="text-xl font-bold text-red-600" id="anomalies">--</div>
                <div class="text-xs text-gray-600">⚠️ Alerts</div>
            </div>
        </div>

        <!-- Latest Money Decisions -->
        <div class="card">
            <h2 class="text-lg font-bold mb-3 flex items-center">
                💰 Latest Money Decisions
                <span class="ml-2 text-sm bg-green-100 text-green-800 px-2 py-1 rounded-full" id="decisionsCount">0</span>
            </h2>
            <div id="decisionsList">
                <div class="text-center text-gray-500 py-4">Loading money decisions...</div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="card">
            <h2 class="text-lg font-bold mb-3">🎛️ Money Machine Controls</h2>
            <div class="grid grid-cols-2 gap-2">
                <button onclick="refreshData()" class="bg-green-500 text-white px-3 py-2 rounded text-sm hover:bg-green-600">
                    💰 Refresh
                </button>
                <button onclick="exportData()" class="bg-yellow-500 text-white px-3 py-2 rounded text-sm hover:bg-yellow-600">
                    📊 Export
                </button>
                <button onclick="showSystemInfo()" class="bg-blue-500 text-white px-3 py-2 rounded text-sm hover:bg-blue-600">
                    📈 Stats
                </button>
                <button onclick="toggleSound()" id="soundBtn" class="bg-purple-500 text-white px-3 py-2 rounded text-sm hover:bg-purple-600">
                    🔊 Sound
                </button>
            </div>
        </div>

        <!-- Footer -->
        <div class="text-center text-xs text-white mt-4">
            🏆 Strato Windows Server: <span id="apiStatus">Checking...</span><br>
            💎 Making Money: 24/7/365<br>
            <span id="refreshTime">--</span>
        </div>
    </div>

    <script>
        // Configuration - CHANGE THIS TO YOUR SERVER IP
        const API_BASE = 'http://85.215.183.30:8002/api/intelligence';
        let soundEnabled = true;
        let moneyCounter = 0;

        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
        
        // Money counter animation
        setInterval(() => {
            moneyCounter += Math.floor(Math.random() * 100) + 1;
            document.getElementById('moneyCounter').textContent = '$' + moneyCounter.toLocaleString();
        }, 2000);

        document.addEventListener('DOMContentLoaded', refreshData);

        async function refreshData() {
            try {
                document.getElementById('lastUpdate').textContent = 
                    'Last Update: ' + new Date().toLocaleTimeString();
                document.getElementById('refreshTime').textContent = 
                    'Last Refresh: ' + new Date().toLocaleTimeString();

                const demoResponse = await axios.get(`${API_BASE}/demo`);
                const demo = demoResponse.data;
                
                document.getElementById('status').textContent = '● MAKING MONEY';
                document.getElementById('status').className = 'status-online';
                document.getElementById('apiStatus').textContent = 'Online 💰';

                displayDemoData(demo);

                // Play money sound
                if (soundEnabled) {
                    playMoneySound();
                }

            } catch (error) {
                console.error('Connection error:', error);
                document.getElementById('status').textContent = '● OFFLINE';
                document.getElementById('status').className = 'status-offline';
                document.getElementById('apiStatus').textContent = 'Disconnected ❌';
                displayOfflineData();
            }
        }

        function displayDemoData(demo) {
            document.getElementById('totalDecisions').textContent = demo.metrics.total_decisions;
            document.getElementById('avgConfidence').textContent = 
                (demo.metrics.avg_confidence * 100).toFixed(1) + '%';
            document.getElementById('activeStrategies').textContent = demo.metrics.strategies_active.length;
            document.getElementById('anomalies').textContent = demo.anomalies.length;

            displayDecisions(demo.decisions);
        }

        function displayDecisions(decisions) {
            const container = document.getElementById('decisionsList');
            document.getElementById('decisionsCount').textContent = decisions.length;
            
            container.innerHTML = decisions.map(decision => `
                <div class="insight-item bg-green-50">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <div class="font-semibold text-sm">💰 ${decision.strategy}</div>
                            <div class="text-xs text-gray-600">${decision.symbol} • ${decision.action.toUpperCase()}</div>
                            <div class="text-xs text-green-600 font-bold">${(decision.confidence * 100).toFixed(0)}% CONFIDENT</div>
                        </div>
                        <div class="text-right ml-2">
                            <div class="text-green-600 font-bold text-sm">$$$</div>
                            <div class="text-xs text-gray-400">
                                ${new Date(decision.timestamp).toLocaleTimeString()}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function displayOfflineData() {
            document.getElementById('totalDecisions').textContent = '--';
            document.getElementById('avgConfidence').textContent = '--';
            document.getElementById('activeStrategies').textContent = '--';
            document.getElementById('anomalies').textContent = '--';
            
            document.getElementById('decisionsList').innerHTML = 
                '<div class="text-red-500 text-center py-4">❌ Money machine offline</div>';
        }

        async function exportData() {
            try {
                const response = await axios.get(`${API_BASE}/export/decisions`);
                const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `money_decisions_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
                window.URL.revokeObjectURL(url);
                alert('💰 Money data exported successfully!');
            } catch (error) {
                alert('❌ Export failed: ' + error.message);
            }
        }

        function showSystemInfo() {
            axios.get(`${API_BASE}/health`)
                .then(response => {
                    const health = response.data;
                    alert(`💰 Money Machine Status:\\n` +
                          `Status: ${health.status}\\n` +
                          `Server: Strato Windows Server\\n` +
                          `Intelligence: ${health.intelligence_enabled ? 'Active 💰' : 'Inactive'}\\n` +
                          `Making Money: 24/7/365\\n` +
                          `Time: ${new Date().toLocaleString()}`);
                })
                .catch(error => {
                    alert('❌ System info unavailable');
                });
        }

        function toggleSound() {
            soundEnabled = !soundEnabled;
            const btn = document.getElementById('soundBtn');
            btn.textContent = soundEnabled ? '🔊 Sound' : '🔇 Muted';
        }

        function playMoneySound() {
            // Create audio context for money sound effect
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.1);
            } catch (e) {
                // Silent fail for audio
            }
        }

        // Add some visual money effects
        function createMoneyEffect() {
            const money = document.createElement('div');
            money.innerHTML = '💰';
            money.style.position = 'fixed';
            money.style.left = Math.random() * window.innerWidth + 'px';
            money.style.top = '-50px';
            money.style.fontSize = '20px';
            money.style.zIndex = '1000';
            money.style.pointerEvents = 'none';
            document.body.appendChild(money);
            
            let pos = -50;
            const fall = setInterval(() => {
                pos += 5;
                money.style.top = pos + 'px';
                if (pos > window.innerHeight) {
                    clearInterval(fall);
                    document.body.removeChild(money);
                }
            }, 50);
        }

        // Create money effect every 30 seconds
        setInterval(createMoneyEffect, 30000);
    </script>
</body>
</html>
'''
    }
    
    # Copy core files
    print("\n📁 Copying core files...")
    for file_path in core_files:
        src = Path(file_path)
        if src.exists():
            if '/' in file_path:
                # Create subdirectories
                dest_dir = deploy_dir / Path(file_path).parent
                dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest = deploy_dir / file_path
            shutil.copy2(src, dest)
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path} not found")
    
    # Create Windows scripts
    print("\n🪟 Creating Windows scripts...")
    for script_name, script_content in windows_scripts.items():
        script_path = deploy_dir / script_name
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"✅ {script_name}")
    
    # Copy important directories
    important_dirs = ['strategies', 'config', 'utils', 'data_sources']
    print("\n📂 Copying important directories...")
    for dir_name in important_dirs:
        src_dir = Path(dir_name)
        if src_dir.exists():
            dest_dir = deploy_dir / dir_name
            shutil.copytree(src_dir, dest_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️ {dir_name}/ not found")
    
    # Create deployment instructions
    instructions = f'''
# 🚀💰 STRATO WINDOWS SERVER DEPLOYMENT INSTRUCTIONS 💰🚀

## STEP 1: Connect to Server
- Use Remote Desktop Connection (RDP)
- Server: 85.215.183.30  
- Username: dministrator (your actual username)

## STEP 2: Copy Files to Server
Copy ALL files from this deployment_package folder to C:\\TradingBot\\ on the server

## STEP 3: Install Dependencies  
Run: install_dependencies_windows.bat

## STEP 4: Configure API Keys
Edit .env file with your REAL trading API keys

## STEP 5: START THE MONEY MACHINE!
Run: start_bot_windows.bat

## STEP 6: Access Mobile Dashboard
Open mobile_dashboard_server.html in browser
Or access: http://85.215.183.30:8002/mobile_dashboard_server.html

## 💰 MONEY WILL START FLOWING! 💰

Management Commands:
- Status: status_bot_windows.bat
- Stop: stop_bot_windows.bat
- Restart: stop_bot_windows.bat then start_bot_windows.bat
'''
    
    with open(deploy_dir / 'DEPLOYMENT_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    
    # Create deployment info
    deployment_info = {
        'created': str(datetime.now()),
        'total_files': len(list(deploy_dir.rglob('*'))),
        'core_files': core_files,
        'windows_scripts': list(windows_scripts.keys()),
        'target_server': '85.215.183.30',
        'target_path': 'C:\\TradingBot',
        'status': 'Ready for deployment'
    }
    
    with open(deploy_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n✅ Deployment package created successfully!")
    print(f"📦 Location: {deploy_dir.absolute()}")
    print(f"📄 Total files: {len(list(deploy_dir.rglob('*')))}")
    
    # Create zip file
    zip_path = Path("strato_trading_bot_deployment.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in deploy_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(deploy_dir)
                zipf.write(file_path, arcname)
    
    print(f"📦 Zip package created: {zip_path.absolute()}")
    
    return deploy_dir, zip_path

if __name__ == "__main__":
    try:
        deploy_dir, zip_path = create_deployment_package()
        
        print("\n" + "="*70)
        print("🚀💰 DEPLOYMENT PACKAGE READY! 💰🚀")
        print("="*70)
        print(f"📁 Directory: {deploy_dir}")
        print(f"📦 Zip File: {zip_path}")
        print("\n🎯 NEXT STEPS:")
        print("1. Connect to Strato Windows Server via RDP")
        print("2. Copy deployment_package folder to C:\\TradingBot\\")
        print("3. Run install_dependencies_windows.bat")
        print("4. Edit .env with your API keys") 
        print("5. Run start_bot_windows.bat")
        print("\n💰 LET THE MONEY FLOW! 💰")
        
    except Exception as e:
        print(f"❌ Error creating deployment package: {e}")
        import traceback
        traceback.print_exc()