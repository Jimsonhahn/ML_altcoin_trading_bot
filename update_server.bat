@echo off
REM 🔄 QUICK UPDATE SCRIPT FOR JANICS FREEDOM FACTORY
echo.
echo 🔄 Updating Janics Freedom Factory on Windows Server...
echo.

REM Navigate to project directory
if exist "ML_altcoin_trading_bot" (
    cd ML_altcoin_trading_bot
) else (
    echo ❌ Project directory not found!
    echo Please run deploy_windows.ps1 first
    pause
    exit /b 1
)

REM Pull latest changes
echo 📥 Pulling latest changes from GitHub...
git pull origin main

if %errorlevel% neq 0 (
    echo ⚠️ Git pull had issues, but continuing...
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Update packages
echo 📦 Updating Python packages...
pip install -r requirements.txt

REM Test API
echo 🧪 Testing API...
python -c "from api.app import create_app; app, socketio = create_app(); print('✅ API ready!')"

if %errorlevel% neq 0 (
    echo ❌ API test failed! Check for errors above.
    pause
    exit /b 1
)

echo.
echo ✅ Update completed successfully!
echo.
echo 🚀 To start the server, run:
echo python api/app.py
echo.
pause