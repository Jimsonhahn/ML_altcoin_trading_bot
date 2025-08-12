# Janics Freedom Factory - Windows Strato Server Deployment
# PowerShell Script für Windows Server

Write-Host "Starting Janics Freedom Factory Deployment on Windows Server..." -ForegroundColor Green

# 1. Check if Git is available
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git not found! Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# 2. Check if Python is available
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.8+" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# 3. Navigate to project directory or clone if needed
Write-Host "Setting up project directory..." -ForegroundColor Yellow

if (Test-Path "ML_altcoin_trading_bot") {
    Write-Host "Project directory found. Updating..." -ForegroundColor Green
    Set-Location ML_altcoin_trading_bot
    
    # Pull latest changes
    Write-Host "Pulling latest changes from GitHub..." -ForegroundColor Yellow
    git pull origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Code updated successfully!" -ForegroundColor Green
    } else {
        Write-Host "Git pull had issues. Continuing anyway..." -ForegroundColor Yellow
    }
} else {
    Write-Host "Cloning repository from GitHub..." -ForegroundColor Yellow
    git clone https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
    Set-Location ML_altcoin_trading_bot
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Repository cloned successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to clone repository!" -ForegroundColor Red
        exit 1
    }
}

# 4. Create Virtual Environment if needed
Write-Host "Setting up Python Virtual Environment..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment created!" -ForegroundColor Green
    } else {
        Write-Host "Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
}

# 5. Activate Virtual Environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} catch {
    Write-Host "Could not activate virtual environment. Continuing..." -ForegroundColor Yellow
}

# 6. Install Requirements
Write-Host "Installing Python packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "Some packages might have issues. Continuing..." -ForegroundColor Yellow
}

# 7. Create necessary data directories
Write-Host "Creating data directories..." -ForegroundColor Yellow
$directories = @(
    "data\trades",
    "data\portfolio", 
    "data\intelligence",
    "data\ml",
    "data\strategy_performance",
    "data\ai",
    "logs"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# 8. Setup Environment File
Write-Host "Setting up environment configuration..." -ForegroundColor Yellow

if (!(Test-Path ".env.production")) {
    Write-Host "Creating .env.production file..." -ForegroundColor Yellow
    
    $envContent = @"
# Flask Configuration
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
FLASK_ENV=production
FLASK_DEBUG=False

# API Configuration
API_PORT=8080
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002

# Security - CHANGE THESE VALUES!
SECRET_KEY=change-this-secret-key-for-production
JWT_SECRET_KEY=change-this-jwt-secret-key-for-production

# Trading Configuration
TRADING_MODE=paper
EXCHANGE_NAME=binance

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Logging
LOG_LEVEL=INFO
"@

    $envContent | Out-File -FilePath ".env.production" -Encoding utf8
    Write-Host "Environment file created!" -ForegroundColor Green
    Write-Host "Please edit .env.production with your actual configuration!" -ForegroundColor Yellow
}

# 9. Test API Import
Write-Host "Testing API components..." -ForegroundColor Yellow

$testScript = @"
import sys
sys.path.append('.')
try:
    from api.app import create_app
    from api.controllers import *
    app, socketio = create_app()
    print('All API components loaded successfully!')
except Exception as e:
    print(f'Error loading API: {e}')
    sys.exit(1)
"@

$testScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Host "API test successful!" -ForegroundColor Green
} else {
    Write-Host "API test failed!" -ForegroundColor Red
    Write-Host "Check the error above and fix any issues." -ForegroundColor Yellow
    exit 1
}

# 10. Display server information
Write-Host ""
Write-Host "DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

Write-Host ""
Write-Host "SERVER INFORMATION:" -ForegroundColor Cyan
Write-Host "Project Directory: $(Get-Location)" -ForegroundColor White
Write-Host "Python Version: $(python --version)" -ForegroundColor White
Write-Host "Git Status:" -ForegroundColor White
git status --short

Write-Host ""
Write-Host "TO START THE SERVER:" -ForegroundColor Cyan
Write-Host "Run this command:" -ForegroundColor Yellow
Write-Host "python api/app.py" -ForegroundColor Green

Write-Host ""
Write-Host "SERVER WILL BE AVAILABLE AT:" -ForegroundColor Cyan
Write-Host "- Local: http://localhost:8080" -ForegroundColor White
Write-Host "- Network: http://[YOUR-SERVER-IP]:8080" -ForegroundColor White

Write-Host ""
Write-Host "DASHBOARD ENDPOINTS:" -ForegroundColor Cyan
Write-Host "- Health: http://[YOUR-SERVER-IP]:8080/health" -ForegroundColor White
Write-Host "- Dashboard: http://[YOUR-SERVER-IP]:8080/api/v1/dashboard" -ForegroundColor White
Write-Host "- Status: http://[YOUR-SERVER-IP]:8080/api/v1/dashboard/status/header" -ForegroundColor White

Write-Host ""
Write-Host "BOT CONTROL:" -ForegroundColor Cyan
Write-Host "- Start Bot: POST http://[YOUR-SERVER-IP]:8080/api/v1/dashboard/bot/start" -ForegroundColor White
Write-Host "- Stop Bot: POST http://[YOUR-SERVER-IP]:8080/api/v1/dashboard/bot/stop" -ForegroundColor White
Write-Host "- Bot Status: GET http://[YOUR-SERVER-IP]:8080/api/v1/dashboard/bot/status" -ForegroundColor White

Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Edit .env.production with your actual configuration" -ForegroundColor White
Write-Host "2. Run: python api/app.py" -ForegroundColor White
Write-Host "3. Open browser to test dashboard endpoints" -ForegroundColor White
Write-Host "4. Configure firewall to allow port 8080" -ForegroundColor White

Write-Host ""
Write-Host "TO UPDATE LATER:" -ForegroundColor Cyan
Write-Host "git pull origin main" -ForegroundColor Green
Write-Host "pip install -r requirements.txt" -ForegroundColor Green
Write-Host "python api/app.py" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "Janics Freedom Factory is ready to launch!" -ForegroundColor Green