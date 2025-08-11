# Trading Bot Dashboard

Ein modernes React-Dashboard für den Altcoin Trading Bot mit Real-time Updates, Dark/Light Mode und responsivem Design.

## 🚀 Schnellstart

### Gesamtes System starten:
```bash
./start_all.sh
```

### Nur Dashboard starten:
```bash
cd dashboard
npm start
```

### System stoppen:
```bash
./stop_all.sh
```

## 🔗 URLs

- **Dashboard**: http://localhost:3001
- **API**: http://localhost:5000
- **API Health**: http://localhost:5000/health
- **API Docs**: http://localhost:5000/api/docs

## 🔑 Login

**Standard-Credentials:**
- **Username**: `admin` oder `trader`
- **Password**: `password`

## 🛠️ Entwicklung

### Dependencies installieren:
```bash
npm install
```

### Development Server starten:
```bash
npm start
```

### Production Build:
```bash
npm run build
```

## 📊 Features

- **🎮 Trading Controls**: Bot starten/stoppen/pausieren
- **📊 Live Positions**: Real-time Position-Management
- **⚙️ Risk Parameters**: Risiko-Einstellungen anpassen
- **📈 Performance Analytics**: Charts und Metriken
- **🔔 Alert Management**: Benachrichtigungen verwalten
- **📜 Trade History**: Erweiterte Filter und Export
- **🌓 Dark/Light Mode**: Responsive Design
- **🔗 WebSocket**: Live-Updates für alle Komponenten

## 🔧 Konfiguration

### Environment Variables (.env):
```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=http://localhost:5000
PORT=3001
BROWSER=none
```

## 🐛 Troubleshooting

### Dashboard nicht erreichbar:
```bash
# Prüfe ob Port 3001 frei ist
lsof -i :3001

# Starte Dashboard neu
cd dashboard
npm start
```

### API nicht erreichbar:
```bash
# Prüfe API Health
curl http://localhost:5000/health

# Starte API neu
python -m api.app
```

### Port-Konflikte:
```bash
# Andere Ports verwenden
PORT=3002 npm start
```

## 📝 Logs

- **API Logs**: `api.log`
- **Dashboard Logs**: `dashboard/dashboard.log`