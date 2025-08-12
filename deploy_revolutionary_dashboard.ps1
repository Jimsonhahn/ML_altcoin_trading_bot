# ==============================================================================
# 🚀 JANICS FREEDOM FACTORY REVOLUTIONARY DASHBOARD DEPLOYMENT 🚀
# Das ultimative Trading Terminal der Welt auf Strato Server deployen!
# ==============================================================================

Write-Host "🏭 Starting Janics Freedom Factory Revolutionary Dashboard Deployment..." -ForegroundColor Green

# Navigate to project directory
Write-Host "📁 Navigating to project directory..." -ForegroundColor Yellow
cd C:\Users\Administrator\ML_altcoin_trading_bot

# Pull latest revolutionary updates from GitHub
Write-Host "⬇️ Pulling revolutionary dashboard updates from GitHub..." -ForegroundColor Yellow
git pull origin main

# Check if pull was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pulled revolutionary dashboard updates!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to pull updates from GitHub!" -ForegroundColor Red
    exit 1
}

# Install any new Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install Node.js dependencies for the revolutionary dashboard
Write-Host "🎨 Installing Node.js dependencies for revolutionary dashboard..." -ForegroundColor Yellow
cd dashboard
npm install --legacy-peer-deps

# Build the revolutionary dashboard
Write-Host "🏗️ Building the ultimate Janics Freedom Factory dashboard..." -ForegroundColor Yellow
npm run build

# Go back to main directory
cd ..

# Stop existing services gracefully
Write-Host "🛑 Stopping existing services..." -ForegroundColor Yellow
taskkill /F /IM python.exe 2>$null
Start-Sleep -Seconds 3

# Start the Flask backend server
Write-Host "🚀 Starting Flask backend server..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "app.py" -WindowStyle Minimized

# Wait for server to initialize
Start-Sleep -Seconds 5

# Start the revolutionary dashboard (if using separate Node server)
Write-Host "🏭 Starting revolutionary dashboard frontend..." -ForegroundColor Yellow
cd dashboard
Start-Process -FilePath "npm" -ArgumentList "start" -WindowStyle Minimized

# Go back to main directory
cd ..

# Test if services are running
Write-Host "🔍 Testing services..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test Flask backend
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/dashboard/summary" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Flask backend is running successfully!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Flask backend may need more time to start..." -ForegroundColor Yellow
}

# Test dashboard frontend (if running on separate port)
try {
    $dashResponse = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 10
    if ($dashResponse.StatusCode -eq 200) {
        Write-Host "✅ Revolutionary dashboard frontend is running!" -ForegroundColor Green
    }
} catch {
    Write-Host "ℹ️ Dashboard may be integrated into Flask app..." -ForegroundColor Blue
}

Write-Host ""
Write-Host "🎉 DEPLOYMENT COMPLETE! 🎉" -ForegroundColor Green
Write-Host "🏭 The Ultimate Janics Freedom Factory Dashboard is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access your revolutionary trading terminal at:" -ForegroundColor Cyan
Write-Host "   Flask Backend: http://85.215.183.30:5000" -ForegroundColor White
Write-Host "   Dashboard: http://85.215.183.30:3000 (if separate)" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Das ultimative Trading Terminal der Welt ist bereit!" -ForegroundColor Magenta
Write-Host "💰 Time to make some serious money with style! 💰" -ForegroundColor Gold

# Show running processes
Write-Host ""
Write-Host "🔍 Currently running processes:" -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*"} | Select-Object Id, ProcessName, CPU | Format-Table