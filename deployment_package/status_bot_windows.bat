@echo off
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
if exist "intelligence_exports\*.json*" (
    dir intelligence_exports\*.json* /o-d /b 2>nul | head -3
) else (
    echo No export files yet
)

echo.
echo Log Files:  
if exist "logs\*.log" (
    dir logs\*.log /o-d /b 2>nul | head -3
    echo Latest log size:
    for %%F in (logs\*.log) do (
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
