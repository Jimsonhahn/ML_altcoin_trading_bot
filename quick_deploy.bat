@echo off
echo.
echo 🚀 JANICS FREEDOM FACTORY REVOLUTIONARY DASHBOARD DEPLOYMENT 🚀
echo Das ultimative Trading Terminal der Welt wird deployed!
echo.

REM Change to project directory
cd /d C:\Users\Administrator\ML_altcoin_trading_bot

REM Pull latest updates
echo ⬇️ Pulling revolutionary dashboard updates...
git pull origin main

REM Check if git pull was successful
if %errorlevel% neq 0 (
    echo ❌ Failed to pull updates!
    pause
    exit /b 1
)

echo ✅ Updates pulled successfully!

REM Install Python dependencies
echo 📦 Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Install and build dashboard
echo 🎨 Setting up revolutionary dashboard...
cd dashboard
call npm install --legacy-peer-deps
call npm run build
cd ..

REM Stop existing processes
echo 🛑 Stopping existing services...
taskkill /F /IM python.exe 2>NUL
timeout /t 3 /nobreak >NUL

REM Start Flask server
echo 🚀 Starting Flask backend server...
start "Flask Backend" /MIN python app.py

REM Wait for server to start
timeout /t 8 /nobreak >NUL

echo.
echo 🎉 DEPLOYMENT COMPLETE! 🎉
echo 🏭 The Ultimate Janics Freedom Factory Dashboard is now running!
echo.
echo 🌐 Access your revolutionary trading terminal at:
echo    http://85.215.183.30:5000
echo.
echo 🚀 Das ultimative Trading Terminal der Welt ist bereit!
echo 💰 Time to make some serious money with style!
echo.
pause