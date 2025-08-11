@echo off
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
if not exist "C:\TradingBot" mkdir "C:\TradingBot"
cd /d "C:\TradingBot"

if not exist "core" mkdir core
if not exist "api" mkdir api
if not exist "api\routes" mkdir "api\routes"
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
echo 2. Copy all your Python files to C:\TradingBot\
echo 3. Run: start_bot_windows.bat
echo.
pause
