# Windows Server Deployment Guide - Mobile Dashboard

## Schnell-Start für Mobile Dashboard Control

### 1. Dependencies installieren
```powershell
# Python Virtual Environment erstellen
python -m venv venv
.\venv\Scripts\Activate.ps1

# Trading Bot Dependencies
pip install -r requirements.txt
pip install lightgbm xgboost

# Windows Service Support (optional für 24/7)
pip install pywin32
```

### 2. Dashboard API starten
```powershell
# Einfacher Start
python server/start_server.py

# Dashboard läuft auf:
# http://localhost:5000 (lokal)
# http://DEINE-SERVER-IP:5000 (mobile)
```

### 3. Firewall für Mobile-Zugriff konfigurieren
```powershell
# Als Administrator ausführen
New-NetFirewallRule -DisplayName "Trading Bot Dashboard" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

# Server IP herausfinden
ipconfig
```

### 4️⃣ Für Server-Deployment (Linux)

**Option A: Mit Windows Terminal/PowerShell SSH**
```powershell
# SSH zum Server
ssh ubuntu@deine-server-ip

# Dann die Linux-Befehle ausführen
git clone https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
cd ML_altcoin_trading_bot
chmod +x setup_production.sh
./setup_production.sh
```

**Option B: Mit PuTTY**
1. PuTTY öffnen
2. Server-IP eingeben
3. Mit Benutzername/Passwort einloggen
4. Linux-Befehle ausführen

### 5️⃣ .env.production Konfiguration

Öffne `.env.production` mit einem Texteditor und fülle aus:

```env
# Server Configuration
SERVER_HOST=deine-server-ip
SERVER_USER=ubuntu
SERVER_PORT=22

# API Configuration  
API_PORT=8080
FLASK_ENV=production
FLASK_DEBUG=False

# Security (WICHTIG: Ändere diese!)
SECRET_KEY=generiere-einen-zufaelligen-key
JWT_SECRET_KEY=generiere-einen-anderen-key

# Trading Configuration
TRADING_MODE=paper  # Ändere zu 'live' wenn bereit
EXCHANGE_NAME=binance

# Exchange API Keys (für Live Trading)
EXCHANGE_API_KEY=dein-exchange-api-key
EXCHANGE_API_SECRET=dein-exchange-api-secret
```

### 6️⃣ Deployment vom Windows PC

**Automatisch mit Git:**
```powershell
git add .
git commit -m "Update configuration"
git push origin main

# Dann auf Server: 
# ./update_production.sh
```

**Manuell mit SCP (Windows 10/11):**
```powershell
scp .env.production ubuntu@server-ip:~/ML_altcoin_trading_bot/
```

## Windows-spezifische Tools

### Empfohlene Software:
- **Git Bash**: Für Unix-ähnliche Befehle
- **Windows Terminal**: Modernes Terminal mit SSH
- **VSCode**: Code-Editor mit Git-Integration
- **PuTTY**: SSH-Client für ältere Windows-Versionen
- **WinSCP**: Grafisches Tool für Dateitransfer

### Bot lokal auf Windows testen:
```powershell
# Start-Script für Windows
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
$env:FLASK_PORT=8080
python api\app.py
```

### Status prüfen (lokal):
```powershell
# In neuem Terminal/Tab
Invoke-WebRequest -Uri http://localhost:8080/api/v1/orchestrator/status | Select-Object -Expand Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## Troubleshooting

### SSL-Zertifikat Fehler:
```powershell
# Temporär für Tests
$env:PYTHONHTTPSVERIFY=0
```

### Port bereits belegt:
```powershell
# Prozess auf Port 8080 finden
netstat -ano | findstr :8080

# Prozess beenden (PID ersetzen)
taskkill /PID <PID> /F
```