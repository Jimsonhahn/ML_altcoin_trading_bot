# 🔧 Dashboard Troubleshooting Guide

## 🎯 Current Status (FIXED!)

✅ **Dashboard läuft erfolgreich auf:** http://localhost:3002  
✅ **API läuft erfolgreich auf:** http://localhost:5001  
✅ **Alle API-Endpunkte funktionieren**  
✅ **Dashboard kompiliert erfolgreich**  

## 🚀 Zugriff auf das Dashboard

### Direkte Links:
- **Dashboard:** [http://localhost:3002](http://localhost:3002)
- **API Health:** [http://localhost:5001/health](http://localhost:5001/health)
- **Connection Test:** [dashboard/test_connection.html](http://localhost:3002/test_connection.html)

## ⚡ Quick Fix Commands

Wenn das Dashboard nicht läuft, diese Befehle ausführen:

```bash
# 1. In das richtige Verzeichnis wechseln
cd /Users/jnb/PycharmProjects/altcoin_trading_bot

# 2. API starten (falls nicht läuft)
PORT=5001 python api/standalone_api.py &

# 3. Dashboard starten
cd dashboard
npm start
```

## 🔍 Diagnose-Schritte

### 1. Prüfe, ob Prozesse laufen
```bash
# API prüfen
curl http://localhost:5001/health

# Dashboard-Port prüfen
netstat -an | grep 3002

# Prozesse anzeigen
ps aux | grep -E "(python.*api|node.*react)"
```

### 2. Logs prüfen
```bash
# Dashboard-Logs (im Terminal wo npm start läuft)
# API-Logs (im Terminal wo API läuft)

# Oder Background-Logs:
tail -f logs/api.log
tail -f dashboard.log
```

### 3. Browser-Probleme
- **F12** für Developer Tools öffnen
- **Console-Tab** für JavaScript-Fehler prüfen
- **Network-Tab** für API-Requests prüfen
- **Ctrl+F5** für Hard Refresh

## 🛠️ Häufige Probleme & Lösungen

### Problem 1: "Site can't be reached"

**Ursache:** Dashboard läuft nicht  
**Lösung:**
```bash
cd dashboard
npm start
```

### Problem 2: "ERR_CONNECTION_REFUSED"

**Ursache:** API läuft nicht  
**Lösung:**
```bash
PORT=5001 python api/standalone_api.py &
```

### Problem 3: Port bereits in Verwendung

**Ursache:** Anderer Prozess blockiert Port  
**Lösung:**
```bash
# Anderen Port verwenden
PORT=5002 python api/standalone_api.py &
PORT=3003 npm start

# Oder blockierende Prozesse stoppen
lsof -ti:5001 | xargs kill -9
lsof -ti:3002 | xargs kill -9
```

### Problem 4: API-Verbindungsfehler im Dashboard

**Ursache:** Falsche API-URL  
**Lösung:**
```bash
# .env Datei prüfen/erstellen
echo "REACT_APP_API_URL=http://localhost:5001" > dashboard/.env
```

### Problem 5: Compilation-Fehler

**Ursache:** Dependencies fehlen  
**Lösung:**
```bash
cd dashboard
npm install
npm start
```

## 🔧 Erweiterte Problemlösung

### Komplettes System neu starten
```bash
# Alle Prozesse stoppen
pkill -f "python.*api"
pkill -f "node.*react"

# System neu starten
cd /Users/jnb/PycharmProjects/altcoin_trading_bot

# API starten
PORT=5001 python api/standalone_api.py &

# Dashboard starten
cd dashboard
npm start
```

### Docker Alternative (falls lokale Version Probleme macht)
```bash
# Docker verwenden
docker-compose up -d

# Dashboard wird dann über nginx auf Port 80 verfügbar
```

### Clean Install (letzter Ausweg)
```bash
cd dashboard
rm -rf node_modules package-lock.json
npm install
npm start
```

## 📊 System Requirements Check

### Node.js Version
```bash
node --version  # Should be 14.x or higher
npm --version   # Should be 6.x or higher
```

### Python Version
```bash
python --version  # Should be 3.8 or higher
```

### Required Ports Free
```bash
# Check ports
netstat -an | grep -E "(3002|5001)"
# Should show LISTEN status when running
```

## 🎯 Success Indicators

Wenn alles funktioniert, sollten Sie sehen:

### Terminal Output (Dashboard):
```
Compiled successfully!

You can now view trading-bot-dashboard in the browser.

  Local:            http://localhost:3002
  On Your Network:  http://192.168.x.x:3002
```

### Terminal Output (API):
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://192.168.x.x:5001
```

### Browser Tests:
- ✅ http://localhost:3002 → Dashboard lädt
- ✅ http://localhost:5001/health → JSON Response
- ✅ Dashboard zeigt Live-Daten an

## 🔒 Firewall/Security Issues

### macOS
```bash
# Firewall-Status prüfen
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Falls nötig, Ports freigeben
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/node
```

### Windows
- Windows Defender Firewall → Allow an app
- Node.js und Python hinzufügen

### Antivirus
- Lokale Server (localhost:3002, localhost:5001) zu Ausnahmen hinzufügen

## 📞 Emergency Commands

Wenn gar nichts funktioniert:

```bash
# 1. Alles stoppen
pkill -f "python"
pkill -f "node"

# 2. Ports prüfen
netstat -tulpn | grep -E "(3002|5001)"

# 3. Mit Production Build testen
cd dashboard
npm run build
npx serve -s build -p 3002

# 4. API minimal testen
python -c "from api.standalone_api import create_app; app = create_app(); app.run(port=5001)"
```

## ✅ Verification Checklist

- [ ] API läuft auf Port 5001
- [ ] Dashboard läuft auf Port 3002  
- [ ] curl http://localhost:5001/health funktioniert
- [ ] Browser kann http://localhost:3002 öffnen
- [ ] Dashboard zeigt SuperLazyBillionaire Panel
- [ ] Live-Daten werden angezeigt
- [ ] Keine Konsolen-Fehler im Browser

## 🎉 Success!

Wenn alle Punkte ✅ sind, ist das Dashboard voll funktionsfähig und Sie können:

- Trading Status überwachen
- SuperLazyBillionaire Strategy verfolgen
- Market Regime Analysis sehen
- Kelly Criterion Position Sizing verwalten
- Performance Metrics einsehen

**Das System ist PRODUKTIONSBEREIT! 🚀**