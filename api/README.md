# Trading Bot REST API

Eine vollständige Flask-basierte REST API für den Altcoin Trading Bot mit JWT-Authentifizierung, WebSocket-Support und Swagger-Dokumentation.

## Features

### 🔐 Sicherheit
- JWT-basierte Authentifizierung mit SecretManager
- Rollenbasierte Zugriffssteuerung (Admin, Trader)
- Sichere Passwort-Hashing mit bcrypt
- CORS-Unterstützung für Cross-Origin-Requests
- Umfassende Fehlerbehandlung mit Datenbereinigung

### 📊 Trading-Funktionen
- Bot-Status und -Kontrolle
- Positions- und Order-Management
- Manuelle Order-Erstellung
- Performance-Metriken
- Backtest-Funktionen

### 🎯 Strategy-Management
- Verfügbare Strategien auflisten
- Strategiekonfiguration validieren
- Signale in Echtzeit generieren
- Backtest-Funktionen
- Parameter-Optimierung

### 📈 Monitoring
- System-Health-Checks
- Metriken (CPU, Memory, Disk)
- Fehler-Statistiken
- Aktive Alerts
- Log-Verwaltung

### 🔄 Echtzeit-Updates
- WebSocket-Verbindungen
- Live-Trading-Updates
- Marktdaten-Streaming
- Alert-Benachrichtigungen
- System-Status-Updates

## Installation

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# API starten
python api/app.py
```

## Konfiguration

### Umgebungsvariablen

```bash
# Flask-Konfiguration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
FLASK_ENV=production

# CORS-Konfiguration
CORS_ORIGINS=http://localhost:3000,https://your-frontend.com

# JWT-Konfiguration (optional - wird automatisch generiert)
JWT_SECRET_KEY=your-secret-key
```

### Standard-Benutzer

Die API erstellt automatisch einen Admin-Benutzer:
- **Username**: admin
- **Password**: admin123
- **Rollen**: admin, trader

⚠️ **Wichtig**: Ändern Sie das Standard-Passwort nach der ersten Anmeldung!

## API-Dokumentation

### Swagger UI
```
http://localhost:5000/api/docs
```

### OpenAPI-Spezifikation
```
http://localhost:5000/api/v1/swagger.json
```

## Endpoints

### Authentication
- `POST /auth/login` - Benutzer-Anmeldung
- `POST /auth/refresh` - Token erneuern
- `POST /auth/logout` - Benutzer-Abmeldung
- `GET /auth/profile` - Benutzerprofil
- `POST /auth/change-password` - Passwort ändern

### Trading
- `GET /api/v1/trading/status` - Bot-Status
- `POST /api/v1/trading/start` - Bot starten
- `POST /api/v1/trading/stop` - Bot stoppen
- `GET /api/v1/trading/positions` - Positionen abrufen
- `GET /api/v1/trading/orders` - Orders abrufen
- `POST /api/v1/trading/manual-order` - Manuelle Order
- `GET /api/v1/trading/performance` - Performance-Metriken
- `POST /api/v1/trading/backtest` - Backtest ausführen

### Strategies
- `GET /api/v1/strategies/list` - Strategien auflisten
- `GET /api/v1/strategies/<name>` - Strategie-Details
- `POST /api/v1/strategies/<name>/validate` - Konfiguration validieren
- `POST /api/v1/strategies/<name>/signal` - Signal generieren
- `POST /api/v1/strategies/<name>/backtest` - Strategie-Backtest
- `GET /api/v1/strategies/active` - Aktive Strategien
- `POST /api/v1/strategies/optimize` - Parameter-Optimierung

### Monitoring
- `GET /api/v1/monitoring/health` - System-Health
- `GET /api/v1/monitoring/metrics` - System-Metriken
- `GET /api/v1/monitoring/logs` - System-Logs
- `GET /api/v1/monitoring/errors` - Fehler-Logs
- `GET /api/v1/monitoring/alerts` - Aktive Alerts
- `GET /api/v1/monitoring/safety-status` - Safety-Manager-Status

## WebSocket-Verbindungen

### Verbindung herstellen
```javascript
const socket = io('http://localhost:5000', {
  auth: {
    token: 'your-jwt-token'
  }
});
```

### Events abonnieren
```javascript
socket.emit('subscribe', {
  channels: ['trading_updates', 'market_data', 'alerts', 'performance']
});
```

### Event-Typen
- `trading_updates` - Trading-Bot-Updates
- `market_data` - Marktdaten
- `alerts` - System-Alerts
- `performance` - Performance-Updates
- `order_update` - Order-Status-Updates
- `position_update` - Positions-Updates

## Authentifizierung

### JWT-Token erhalten
```python
import requests

response = requests.post('http://localhost:5000/auth/login', json={
    'username': 'admin',
    'password': 'admin123'
})

token = response.json()['access_token']
```

### Token verwenden
```python
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:5000/api/v1/trading/status', headers=headers)
```

## Beispiel-Code

### Python-Client
```python
import requests
import json

class TradingBotClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.token = None
    
    def login(self, username, password):
        response = requests.post(f'{self.base_url}/auth/login', json={
            'username': username,
            'password': password
        })
        self.token = response.json()['access_token']
        return response.json()
    
    def get_headers(self):
        return {'Authorization': f'Bearer {self.token}'}
    
    def get_trading_status(self):
        response = requests.get(
            f'{self.base_url}/api/v1/trading/status',
            headers=self.get_headers()
        )
        return response.json()
    
    def start_trading(self, mode='paper', strategy='momentum'):
        response = requests.post(
            f'{self.base_url}/api/v1/trading/start',
            json={'mode': mode, 'strategy': strategy},
            headers=self.get_headers()
        )
        return response.json()

# Verwendung
client = TradingBotClient()
client.login('admin', 'admin123')
status = client.get_trading_status()
print(json.dumps(status, indent=2))
```

### JavaScript-Client
```javascript
class TradingBotClient {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
    this.token = null;
  }
  
  async login(username, password) {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await response.json();
    this.token = data.access_token;
    return data;
  }
  
  getHeaders() {
    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    };
  }
  
  async getTradingStatus() {
    const response = await fetch(`${this.baseUrl}/api/v1/trading/status`, {
      headers: this.getHeaders()
    });
    return await response.json();
  }
  
  async startTrading(mode = 'paper', strategy = 'momentum') {
    const response = await fetch(`${this.baseUrl}/api/v1/trading/start`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ mode, strategy })
    });
    return await response.json();
  }
}

// Verwendung
const client = new TradingBotClient();
await client.login('admin', 'admin123');
const status = await client.getTradingStatus();
console.log(status);
```

## Sicherheit

### Benutzer-Management
```python
# Neuen Benutzer hinzufügen (über SecretManager)
from utils.secret_manager import SecretManager
import bcrypt

secret_manager = SecretManager('trading_bot_api')
users = secret_manager.get_secret('allowed_users')

# Passwort hashen
password_hash = bcrypt.hashpw('new_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Benutzer hinzufügen
users['new_user'] = {
    'password_hash': password_hash,
    'roles': ['trader'],
    'active': True
}

secret_manager.store_secret('allowed_users', str(users))
```

### Rollen-System
- **admin**: Vollzugriff auf alle Funktionen
- **trader**: Zugriff auf Trading-Funktionen
- **viewer**: Nur Lesezugriff (implementierbar)

## Monitoring

### Health-Check
```bash
curl http://localhost:5000/api/v1/monitoring/health
```

### Metriken abrufen
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/v1/monitoring/metrics
```

## Fehlerbehandlung

Die API verwendet das eingebaute SecureErrorHandler-System:
- Sichere Fehlerprotokollierung
- Eindeutige Fehler-IDs
- Datenbereinigung in Fehlermeldungen
- Strukturierte Fehler-Responses

## Deployment

### Docker (empfohlen)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "api/app.py"]
```

### Systemd-Service
```ini
[Unit]
Description=Trading Bot API
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/trading-bot
ExecStart=/opt/trading-bot/venv/bin/python api/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Entwicklung

### Tests ausführen
```bash
pytest tests/
```

### Linting
```bash
flake8 api/
black api/
```

### Lokale Entwicklung
```bash
export FLASK_DEBUG=True
export FLASK_ENV=development
python api/app.py
```

## Troubleshooting

### Häufige Probleme

1. **JWT-Token ungültig**
   - Token erneuern über `/auth/refresh`
   - Neu anmelden

2. **CORS-Fehler**
   - CORS_ORIGINS Umgebungsvariable prüfen
   - Frontend-URL in allowedOrigins hinzufügen

3. **WebSocket-Verbindung fehlgeschlagen**
   - JWT-Token in auth-Parameter übergeben
   - Netzwerk-Konfiguration prüfen

4. **Permissions-Fehler**
   - Benutzer-Rollen prüfen
   - Admin-Rechte für bestimmte Endpoints erforderlich

### Debug-Modus
```bash
export FLASK_DEBUG=True
python api/app.py
```

## Beitragen

1. Fork des Repositories
2. Feature-Branch erstellen
3. Änderungen committen
4. Tests ausführen
5. Pull Request stellen

## Lizenz

MIT License - siehe LICENSE-Datei für Details.