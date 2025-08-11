@echo off
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
