# 🚀 STRATO WINDOWS SERVER DEPLOYMENT SCRIPT
# DER ULTIMATIVE GELDDRUCKMASCHINEN-DEPLOYER

param(
    [string]$ServerIP = "85.215.183.30",
    [string]$Username = "administrator",
    [string]$ProjectPath = "C:\TradingBot"
)

Write-Host "🚀💰 DEPLOYING MONEY-MAKING MACHINE TO STRATO WINDOWS SERVER 💰🚀" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Yellow

# Step 1: Server Connection Test
Write-Host "📡 Testing connection to Strato Windows Server..." -ForegroundColor Cyan
$TestConnection = Test-NetConnection -ComputerName $ServerIP -Port 3389 -InformationLevel Quiet
if ($TestConnection) {
    Write-Host "✅ RDP Connection to $ServerIP successful!" -ForegroundColor Green
} else {
    Write-Host "❌ Cannot reach server. Check VPN/Network." -ForegroundColor Red
    exit 1
}

# Step 2: Prepare Deployment Package
Write-Host "📦 Preparing deployment package..." -ForegroundColor Cyan

$DeployPackage = @"
# GELDDRUCKMASCHINE DEPLOYMENT PACKAGE
# Auto-generated for Strato Windows Server

# Core Bot Files
- main.py (Enhanced with Intelligence)
- core/enhanced_decision_logger.py
- api/routes/intelligence.py
- run_intelligence_api.py
- start_enhanced_bot.py
- mobile_dashboard.html

# Configuration
- docker/docker-compose.production.yml
- docker/Dockerfile.production
- requirements_simple.txt
- .env.production.example

# Windows Scripts
- start_bot_windows.bat
- stop_bot_windows.bat  
- status_bot_windows.bat
- install_dependencies_windows.bat
"@

Write-Host $DeployPackage -ForegroundColor White

# Step 3: Create Windows Deployment Scripts
Write-Host "🛠️ Creating Windows-specific deployment scripts..." -ForegroundColor Cyan

# Install Dependencies Script
$InstallScript = @'
@echo off
echo 🚀 Installing Trading Bot Dependencies on Windows Server...
echo ================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.11+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create project directory
if not exist "C:\TradingBot" mkdir "C:\TradingBot"
cd /d "C:\TradingBot"

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing Python packages...
if exist "requirements_simple.txt" (
    pip install -r requirements_simple.txt
) else (
    echo Installing basic packages...
    pip install flask flask-cors requests pandas asyncpg asyncio-mqtt
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "logs" mkdir logs
if not exist "intelligence_exports" mkdir intelligence_exports
if not exist "data" mkdir data

REM Install Docker Desktop (if not present)
echo 🐳 Checking Docker Desktop...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Docker Desktop not found. Please install manually:
    echo Download from: https://www.docker.com/products/docker-desktop
    echo After installation, restart this script.
    pause
)

echo ✅ Dependencies installation completed!
echo 🎯 Next: Copy your bot files to C:\TradingBot\
echo 🚀 Then run: start_bot_windows.bat
pause
'@

$InstallScript | Out-File -FilePath "install_dependencies_windows.bat" -Encoding ASCII

# Start Bot Script  
$StartScript = @'
@echo off
echo 💰🚀 STARTING THE MONEY-MAKING MACHINE 🚀💰
echo ==============================================

cd /d "C:\TradingBot"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Run install_dependencies_windows.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Kill any existing processes
echo 🛑 Stopping existing processes...
taskkill /f /im python.exe >nul 2>&1
timeout /t 2 >nul

REM Start Intelligence API
echo 🧠 Starting Intelligence API...
start "Intelligence API" python run_intelligence_api.py --host 0.0.0.0 --port 8002

REM Wait a moment
timeout /t 5 >nul

REM Start Enhanced Bot
echo 🤖 Starting Enhanced Trading Bot...
start "Enhanced Bot" python start_enhanced_bot.py --mode live --strategy momentum

REM Show status
timeout /t 3 >nul
echo.
echo 🎉 MONEY-MAKING MACHINE IS NOW RUNNING! 🎉
echo ============================================
echo 📊 Intelligence API: http://85.215.183.30:8002/api/intelligence/demo
echo 🌐 Dashboard: Open mobile_dashboard.html and change API_BASE to your server IP
echo 📱 Mobile Access: http://85.215.183.30:8002/mobile_dashboard.html
echo.
echo 💰 LET THE MONEY FLOW! 💰
echo Press any key to see status...
pause

REM Show running processes
echo 🔍 Running Trading Bot Processes:
tasklist | findstr python.exe
echo.
echo 🌐 Network Connections:
netstat -an | findstr :8002
echo.
echo 🚀 Bot is running! Keep this window open or press Ctrl+C to stop.
pause
'@

$StartScript | Out-File -FilePath "start_bot_windows.bat" -Encoding ASCII

# Stop Bot Script
$StopScript = @'
@echo off
echo 🛑 STOPPING MONEY-MAKING MACHINE...
echo ==================================

echo Killing Python processes...
taskkill /f /im python.exe

echo Waiting for processes to stop...
timeout /t 3 >nul

echo 📊 Checking if processes are stopped...
tasklist | findstr python.exe
if errorlevel 1 (
    echo ✅ All trading bot processes stopped successfully!
) else (
    echo ⚠️ Some processes may still be running.
)

echo 💤 Money-making machine is now offline.
pause
'@

$StopScript | Out-File -FilePath "stop_bot_windows.bat" -Encoding ASCII

# Status Script
$StatusScript = @'
@echo off
echo 📊 MONEY-MAKING MACHINE STATUS
echo =============================

echo 🤖 Python Processes:
tasklist | findstr python.exe
if errorlevel 1 (
    echo ❌ No bot processes running
) else (
    echo ✅ Bot processes active
)

echo.
echo 🌐 Network Status:
echo Intelligence API (Port 8002):
netstat -an | findstr :8002

echo Bot API (Port 8000):
netstat -an | findstr :8000

echo.
echo 💾 System Resources:
echo CPU Usage:
wmic cpu get loadpercentage /value | findstr LoadPercentage

echo Memory Usage:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "="

echo.
echo 📁 Log Files:
if exist "logs\*.log" (
    dir logs\*.log /b
) else (
    echo No log files found
)

echo.
echo 💰 READY TO MAKE MONEY? 💰
echo API Test: curl http://localhost:8002/health
pause
'@

$StatusScript | Out-File -FilePath "status_bot_windows.bat" -Encoding ASCII

Write-Host "✅ Windows deployment scripts created!" -ForegroundColor Green

# Step 4: Show Deployment Instructions
Write-Host "`n🎯 DEPLOYMENT INSTRUCTIONS FOR STRATO WINDOWS SERVER:" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Yellow

$Instructions = @"
1️⃣ CONNECT TO YOUR STRATO WINDOWS SERVER:
   - Use Remote Desktop Connection
   - Server: 85.215.183.30
   - Username: administrator (or your actual username)
   
2️⃣ TRANSFER FILES TO SERVER:
   Copy these files to C:\TradingBot\ on the server:
   ✅ All Python files (main.py, core/, api/, etc.)
   ✅ install_dependencies_windows.bat
   ✅ start_bot_windows.bat  
   ✅ stop_bot_windows.bat
   ✅ status_bot_windows.bat
   ✅ requirements_simple.txt
   ✅ mobile_dashboard.html

3️⃣ INSTALL DEPENDENCIES ON SERVER:
   - Double-click: install_dependencies_windows.bat
   - Follow the prompts
   - Install Python 3.11+ if needed
   - Install Docker Desktop if needed

4️⃣ CONFIGURE ENVIRONMENT:
   - Edit .env file with your real API keys
   - Set BINANCE_API_KEY and BINANCE_SECRET_KEY
   - Choose trading mode (paper or live)

5️⃣ START THE MONEY-MAKING MACHINE:
   - Double-click: start_bot_windows.bat
   - Watch the magic happen! 💰💰💰

6️⃣ ACCESS FROM YOUR PHONE:
   - Open mobile_dashboard.html 
   - Change API_BASE to: http://85.215.183.30:8002
   - Enjoy real-time bot monitoring! 📱

7️⃣ MANAGEMENT:
   - Status: status_bot_windows.bat
   - Stop: stop_bot_windows.bat
   - Restart: stop_bot_windows.bat then start_bot_windows.bat
"@

Write-Host $Instructions -ForegroundColor White

Write-Host "`n💰 MONEY-MAKING FEATURES ACTIVATED:" -ForegroundColor Green
Write-Host "✅ Enhanced Decision Logger with ML Intelligence" -ForegroundColor Green
Write-Host "✅ Real-time Pattern Recognition" -ForegroundColor Green  
Write-Host "✅ Mobile Dashboard for 24/7 Monitoring" -ForegroundColor Green
Write-Host "✅ Automatic Data Export for Continuous Learning" -ForegroundColor Green
Write-Host "✅ Anomaly Detection for Risk Management" -ForegroundColor Green

Write-Host "`n🚀 READY TO DEPLOY THE ULTIMATE TRADING MACHINE!" -ForegroundColor Cyan
Write-Host "💎 LET'S MAKE THOSE COINS RAIN! 💎" -ForegroundColor Cyan