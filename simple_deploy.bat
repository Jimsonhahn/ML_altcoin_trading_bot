@echo off
echo.
echo ======================================
echo   JANICS FREEDOM FACTORY DEPLOYMENT
echo ======================================
echo.

REM Check if git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git not found! Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Check if python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Clone or update repository
if exist "ML_altcoin_trading_bot" (
    echo Project directory found. Updating...
    cd ML_altcoin_trading_bot
    git pull origin main
) else (
    echo Cloning repository from GitHub...
    git clone https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
    cd ML_altcoin_trading_bot
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install packages
echo Installing Python packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create data directories
echo Creating data directories...
if not exist "data\trades" mkdir data\trades
if not exist "data\portfolio" mkdir data\portfolio  
if not exist "data\intelligence" mkdir data\intelligence
if not exist "data\ml" mkdir data\ml
if not exist "data\strategy_performance" mkdir data\strategy_performance
if not exist "data\ai" mkdir data\ai
if not exist "logs" mkdir logs

REM Create environment file
if not exist ".env.production" (
    echo Creating environment file...
    echo # Flask Configuration > .env.production
    echo FLASK_PORT=8080 >> .env.production
    echo FLASK_HOST=0.0.0.0 >> .env.production
    echo FLASK_ENV=production >> .env.production
    echo FLASK_DEBUG=False >> .env.production
    echo. >> .env.production
    echo # API Configuration >> .env.production
    echo API_PORT=8080 >> .env.production
    echo. >> .env.production  
    echo # Security - CHANGE THESE! >> .env.production
    echo SECRET_KEY=change-this-secret-key >> .env.production
    echo JWT_SECRET_KEY=change-this-jwt-secret >> .env.production
    echo. >> .env.production
    echo # Trading >> .env.production
    echo TRADING_MODE=paper >> .env.production
    echo EXCHANGE_NAME=binance >> .env.production
)

REM Test API
echo Testing API components...
python -c "from api.app import create_app; app, socketio = create_app(); print('API test successful!')"

if %errorlevel% neq 0 (
    echo API test failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo ======================================
echo   DEPLOYMENT SUCCESSFUL!
echo ======================================
echo.
echo Current directory: %cd%
echo.
echo TO START THE SERVER:
echo   python api/app.py
echo.
echo SERVER WILL BE AVAILABLE AT:
echo   http://localhost:8080
echo   http://[YOUR-SERVER-IP]:8080
echo.
echo DASHBOARD ENDPOINTS:
echo   http://[YOUR-SERVER-IP]:8080/health
echo   http://[YOUR-SERVER-IP]:8080/api/v1/dashboard/status/header
echo.
echo NEXT STEPS:
echo 1. Edit .env.production with your settings
echo 2. Open Windows Firewall for port 8080
echo 3. Run: python api/app.py
echo.
pause