# Windows Server 2022 Setup Script für Trading Bot
# Als Administrator in PowerShell ausführen

Write-Host "🚀 Trading Bot Setup für Windows Server 2022..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Blue

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ Dieses Skript muss als Administrator ausgeführt werden!" -ForegroundColor Red
    Write-Host "Rechtsklick auf PowerShell → 'Als Administrator ausführen'" -ForegroundColor Yellow
    exit 1
}

# Set execution policy
Write-Host "📋 PowerShell Execution Policy setzen..." -ForegroundColor Cyan
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Force

# Create project directory
Write-Host "📁 Projekt-Verzeichnis erstellen..." -ForegroundColor Cyan
$ProjectPath = "C:\TradingBot"
New-Item -ItemType Directory -Path $ProjectPath -Force
New-Item -ItemType Directory -Path "$ProjectPath\api" -Force
New-Item -ItemType Directory -Path "$ProjectPath\dashboard" -Force
New-Item -ItemType Directory -Path "$ProjectPath\core" -Force
New-Item -ItemType Directory -Path "$ProjectPath\strategies" -Force
New-Item -ItemType Directory -Path "$ProjectPath\data" -Force
New-Item -ItemType Directory -Path "$ProjectPath\logs" -Force
New-Item -ItemType Directory -Path "$ProjectPath\config" -Force
New-Item -ItemType Directory -Path "$ProjectPath\scripts" -Force

Write-Host "✅ Verzeichnisse erstellt: $ProjectPath" -ForegroundColor Green

# Install Chocolatey (Package Manager für Windows)
Write-Host "🍫 Chocolatey Package Manager installieren..." -ForegroundColor Cyan
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
try {
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "✅ Chocolatey installiert" -ForegroundColor Green
} catch {
    Write-Host "❌ Chocolatey Installation fehlgeschlagen: $($_.Exception.Message)" -ForegroundColor Red
}

# Refresh environment variables
refreshenv

# Install Python 3.11
Write-Host "🐍 Python 3.11 installieren..." -ForegroundColor Cyan
try {
    choco install python311 -y
    Write-Host "✅ Python 3.11 installiert" -ForegroundColor Green
} catch {
    Write-Host "❌ Python Installation fehlgeschlagen" -ForegroundColor Red
}

# Install Node.js 18
Write-Host "📦 Node.js 18 installieren..." -ForegroundColor Cyan
try {
    choco install nodejs --version=18.19.0 -y
    Write-Host "✅ Node.js 18 installiert" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js Installation fehlgeschlagen" -ForegroundColor Red
}

# Install Git
Write-Host "📚 Git installieren..." -ForegroundColor Cyan
try {
    choco install git -y
    Write-Host "✅ Git installiert" -ForegroundColor Green
} catch {
    Write-Host "❌ Git Installation fehlgeschlagen" -ForegroundColor Red
}

# Install additional tools
Write-Host "🔧 Zusätzliche Tools installieren..." -ForegroundColor Cyan
choco install 7zip -y
choco install notepadplusplus -y
choco install vcredist-all -y

# Refresh PATH
Write-Host "🔄 Umgebungsvariablen aktualisieren..." -ForegroundColor Cyan
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Create Python virtual environment
Write-Host "🐍 Python Virtual Environment erstellen..." -ForegroundColor Cyan
Set-Location $ProjectPath
python -m venv venv
Write-Host "✅ Virtual Environment erstellt: $ProjectPath\venv" -ForegroundColor Green

# Activate virtual environment and install packages
Write-Host "📦 Python Pakete installieren..." -ForegroundColor Cyan
& "$ProjectPath\venv\Scripts\Activate.ps1"

# Create requirements.txt
$RequirementsContent = @"
# Core FastAPI and async
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0

# Data processing
pandas==2.1.4
numpy==1.25.2
scipy==1.11.4

# Machine Learning
scikit-learn==1.3.2
lightgbm==4.1.0

# Trading APIs
ccxt==4.1.90
python-binance==1.0.19
requests==2.31.0
aiohttp==3.9.1

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
aiosqlite==0.19.0

# Utilities
python-telegram-bot==20.7
python-dotenv==1.0.0
pyyaml==6.0.1
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0
psutil==5.9.6

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
"@

$RequirementsContent | Out-File -FilePath "$ProjectPath\requirements.txt" -Encoding UTF8
Write-Host "✅ requirements.txt erstellt" -ForegroundColor Green

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install -r "$ProjectPath\requirements.txt"

# Configure Windows Firewall
Write-Host "🔒 Windows Firewall konfigurieren..." -ForegroundColor Cyan

# Allow Python through firewall
New-NetFirewallRule -DisplayName "Python Trading Bot" -Direction Inbound -Protocol TCP -LocalPort 8000,8001 -Action Allow
New-NetFirewallRule -DisplayName "React Dashboard" -Direction Inbound -Protocol TCP -LocalPort 3000,3001,3002,3003 -Action Allow
New-NetFirewallRule -DisplayName "WebSocket" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow

Write-Host "✅ Firewall Regeln hinzugefügt" -ForegroundColor Green

# Install IIS (falls nicht bereits installiert)
Write-Host "🌐 IIS Features aktivieren..." -ForegroundColor Cyan
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole, IIS-WebServer, IIS-CommonHttpFeatures, IIS-HttpErrors, IIS-HttpRedirect, IIS-ApplicationDevelopment, IIS-NetFxExtensibility45, IIS-HealthAndDiagnostics, IIS-HttpLogging, IIS-Security, IIS-RequestFiltering, IIS-Performance, IIS-WebServerManagementTools, IIS-ManagementConsole, IIS-IIS6ManagementCompatibility, IIS-Metabase, IIS-ASPNET45 -All

# Create Windows Service Scripts (wird später verwendet)
Write-Host "📝 Service-Skripte erstellen..." -ForegroundColor Cyan

# Create start script
$StartScript = @"
@echo off
echo 🚀 Starting Trading Bot Services...
cd /d C:\TradingBot
call venv\Scripts\activate.bat
start "Trading Bot API" python -m uvicorn main:app --host 0.0.0.0 --port 8000
start "Intelligence API" python config\server_intelligence_api.py
cd dashboard
start "React Dashboard" npm start
echo ✅ All services started
pause
"@

$StartScript | Out-File -FilePath "$ProjectPath\scripts\start_services.bat" -Encoding ASCII

# Create stop script
$StopScript = @"
@echo off
echo 🛑 Stopping Trading Bot Services...
taskkill /f /im python.exe
taskkill /f /im node.exe
echo ✅ All services stopped
pause
"@

$StopScript | Out-File -FilePath "$ProjectPath\scripts\stop_services.bat" -Encoding ASCII

# Create status script
$StatusScript = @"
@echo off
echo 📊 Trading Bot Service Status
echo ================================
netstat -an | findstr ":8000 "
netstat -an | findstr ":8001 "
netstat -an | findstr ":3000 "
echo.
echo 🔥 Active Python Processes:
tasklist | findstr python.exe
echo.
echo 🌐 Active Node Processes:
tasklist | findstr node.exe
pause
"@

$StatusScript | Out-File -FilePath "$ProjectPath\scripts\status.bat" -Encoding ASCII

# System Information
Write-Host "`n📊 System Information:" -ForegroundColor Yellow
Write-Host "Windows Version: $(Get-ComputerInfo | Select-Object -ExpandProperty WindowsProductName)" -ForegroundColor White
Write-Host "Available Memory: $([math]::Round((Get-ComputerInfo).TotalPhysicalMemory/1GB, 2)) GB" -ForegroundColor White
Write-Host "Free Disk Space: $([math]::Round((Get-PSDrive C).Free/1GB, 2)) GB" -ForegroundColor White

# Final status
Write-Host "`n🎉 Trading Bot Setup Komplett!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Blue
Write-Host "✅ Projekt Verzeichnis: C:\TradingBot" -ForegroundColor Green
Write-Host "✅ Python 3.11 + Virtual Environment" -ForegroundColor Green
Write-Host "✅ Node.js 18 für React Dashboard" -ForegroundColor Green
Write-Host "✅ Windows Firewall konfiguriert" -ForegroundColor Green
Write-Host "✅ IIS aktiviert" -ForegroundColor Green
Write-Host "✅ Service-Skripte erstellt" -ForegroundColor Green

Write-Host "`n📋 Nächste Schritte:" -ForegroundColor Yellow
Write-Host "1. Projekt-Dateien nach C:\TradingBot kopieren" -ForegroundColor White
Write-Host "2. .env Datei mit API-Keys konfigurieren" -ForegroundColor White
Write-Host "3. Services starten: C:\TradingBot\scripts\start_services.bat" -ForegroundColor White
Write-Host "4. Dashboard öffnen: http://localhost:3000" -ForegroundColor White

Write-Host "`n🚀 Ready für Intelligence Trading Bot Deployment!" -ForegroundColor Cyan