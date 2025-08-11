@echo off
REM 🚀 Start Simple Trading Bot on Windows Server
REM Für 24/7 Betrieb auf Windows Server 2022

echo 🚀 Starting Bulletproof Simple Trading Bot...
echo =============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv_simple" (
    echo 📦 Creating virtual environment...
    python -m venv venv_simple
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv_simple\Scripts\activate.bat

REM Install requirements
echo 📥 Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements_simple.txt

REM Create data directories
echo 📁 Creating data directories...
if not exist "sync_data" mkdir sync_data
if not exist "logs" mkdir logs

REM Set environment variables
set FLASK_APP=simple_bulletproof_trading_bot.py
set FLASK_ENV=production

REM Start the bot
echo 🚀 Starting Trading Bot...
echo Dashboard will be available at: http://85.215.183.30:5000
echo API available at: http://85.215.183.30:5000/api/health
echo Press Ctrl+C to stop
echo.

python simple_bulletproof_trading_bot.py

pause