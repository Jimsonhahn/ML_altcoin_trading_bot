@echo off
title 💰🚀 MONEY-MAKING MACHINE 🚀💰
color 0A

echo.
echo 💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
echo 💰                                                              💰
echo 💰            🚀 STARTING THE MONEY-MAKING MACHINE 🚀          💰  
echo 💰                                                              💰
echo 💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰
echo.

cd /d "C:\TradingBot"

REM Check if we're in the right directory
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found in C:\TradingBot!
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
call venv\Scripts\activate.bat

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
if exist "logs\*.log" (
    dir logs\*.log /o-d /b | head -3
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
