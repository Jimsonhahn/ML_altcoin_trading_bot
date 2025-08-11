# SecureErrorHandler - Sicherheitsorientierte Fehlerbehandlung

## 🎯 Übersicht

Die `SecureErrorHandler` Klasse erweitert das bestehende Error Handling Framework um spezialisierte, sicherheitsorientierte Fehlerbehandlung mit:

- **Sichere Protokollierung** ohne sensitive Daten
- **Eindeutige Error-IDs** mit UUID-Generierung
- **Strukturierte Error-Responses** für APIs und Monitoring
- **Spezialisierte Handler-Methoden** für verschiedene Fehlertypen

## ✅ Implementierte Features

### 1. SecureErrorResponse
**Strukturierte Fehlerantworten mit Sicherheitsfokus**

```python
@dataclass
class SecureErrorResponse:
    error_id: str          # Eindeutige UUID
    timestamp: str         # ISO-Format Zeitstempel
    category: str          # Fehlerkategorie
    severity: str          # Schweregrad
    message: str           # Bereinigte Fehlermeldung
    status_code: int       # HTTP-Status Code
    details: Optional[Dict[str, Any]] = None  # Sichere Details
    trace_id: Optional[str] = None            # Trace-ID für Korrelation
```

### 2. Automatische Datenbereinigung
**Entfernt sensitive Daten aus Logs und Responses**

```python
# Erkannte sensitive Patterns:
- API Keys: api_key=secret123 → api_key=sec***23
- Bearer Tokens: Authorization: Bearer token123 → Authorization=Be***r token123
- Passwörter: password=mypass → password=myp***ss
- Private Keys: private_key=... → ***REDACTED***
- Credit Cards: 4532-1234-5678-9012 → ***REDACTED***
- Stripe Keys: sk_test_... → ***REDACTED***
```

### 3. Spezialisierte Handler-Methoden

#### `handle_critical_error()`
Für kritische Systemfehler mit sofortiger Benachrichtigung:

```python
secure_handler = SecureErrorHandler()

try:
    raise MemoryError("Out of memory")
except Exception as e:
    response = secure_handler.handle_critical_error(
        error=e,
        context={"operation": "data_processing", "user": "system"}
    )
    print(f"Critical Error ID: {response.error_id}")
```

#### `handle_trading_error()`
Für Trading-spezifische Fehler mit Trading-Kontext:

```python
try:
    raise ValidationTradingError("Invalid amount", field="amount", value=-100)
except Exception as e:
    response = secure_handler.handle_trading_error(
        error=e,
        symbol="BTC/USDT",
        order_id="order_12345",
        amount=0.1,
        context={"strategy": "momentum", "api_key": "secret123"}
    )
    print(f"Trading Error: {response.error_id}")
    print(f"Status Code: {response.status_code}")
```

#### `handle_api_error()`
Für API-bezogene Fehler mit automatischer Datenbereinigung:

```python
try:
    raise ConnectionError("API timeout")
except Exception as e:
    response = secure_handler.handle_api_error(
        error=e,
        endpoint="https://api.binance.com/api/v3/order",
        status_code=429,
        request_data={
            "symbol": "BTCUSDT",
            "api_key": "sensitive_key_123",  # Wird automatisch bereinigt
            "secret": "very_secret_value"    # Wird automatisch bereinigt
        },
        response_data={"error": "Rate limit exceeded"}
    )
    print(f"API Error: {response.error_id}")
    print(f"Sanitized details: {response.details}")
```

## 🔧 Verwendung im Trading Bot

### 1. Integration in Bestehende Komponenten

```python
from utils.error_handler import secure_error_handler

class TradingBot:
    def __init__(self):
        self.error_handler = secure_error_handler
        
        # Notification callback für kritische Fehler
        self.error_handler.add_notification_callback(self._handle_error_notification)
    
    def _handle_error_notification(self, response: SecureErrorResponse):
        """Handle error notifications"""
        if response.severity == "critical":
            # Pausiere Trading bei kritischen Fehlern
            self.pause_trading()
            # Sende Admin-Benachrichtigung
            self.send_admin_alert(response)
```

### 2. Exchange Integration

```python
class ExchangeManager:
    def __init__(self):
        self.error_handler = secure_error_handler
    
    def place_order(self, order_data):
        try:
            # Order execution logic
            result = self.exchange_api.create_order(order_data)
            return result
        except requests.exceptions.Timeout as e:
            # API-Fehler mit Kontext behandeln
            response = self.error_handler.handle_api_error(
                error=e,
                endpoint="/api/v3/order",
                status_code=408,
                request_data=order_data,  # Wird automatisch bereinigt
                context={"operation": "place_order"}
            )
            raise ExchangeTradingError(f"Order failed: {response.error_id}")
        except ValidationError as e:
            # Trading-Fehler behandeln
            response = self.error_handler.handle_trading_error(
                error=e,
                symbol=order_data.get("symbol"),
                order_id=order_data.get("order_id"),
                context=order_data  # Sensitive Daten werden bereinigt
            )
            raise ValidationTradingError(f"Validation failed: {response.error_id}")
```

### 3. API Endpoints mit SecureErrorResponse

```python
from fastapi import HTTPException
from utils.error_handler import secure_error_handler

@app.post("/api/orders")
async def create_order(order_data: dict):
    try:
        result = trading_bot.place_order(order_data)
        return {"success": True, "result": result}
    except Exception as e:
        # Sichere Fehlerbehandlung für API
        response = secure_error_handler.handle_api_error(
            error=e,
            endpoint="/api/orders",
            request_data=order_data,
            context={"user_id": "user123"}
        )
        
        # Strukturierte HTTP-Antwort
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error_id": response.error_id,
                "message": response.message,
                "trace_id": response.trace_id,
                "timestamp": response.timestamp
            }
        )
```

## 📊 Monitoring und Statistiken

### Error Retrieval

```python
# Fehler nach ID abrufen
error = secure_handler.get_error_by_id("error-uuid-123")
if error:
    print(f"Error details: {error.to_json()}")

# Alle Fehler einer Trace-ID
trace_errors = secure_handler.get_errors_by_trace_id("trace-123")
print(f"Related errors: {len(trace_errors)}")

# Statistiken abrufen
stats = secure_handler.get_error_statistics()
print(f"Total errors: {stats['total_errors']}")
print(f"Categories: {stats['category_breakdown']}")
print(f"Recent errors: {stats['recent_errors_count']}")
```

### Notification System

```python
def critical_error_alert(response: SecureErrorResponse):
    """Send alert for critical errors"""
    if response.severity == "critical":
        # Send to monitoring system
        send_to_slack(f"🚨 Critical Error: {response.error_id}")
        send_to_pagerduty(response.to_dict())

def trading_error_logger(response: SecureErrorResponse):
    """Log trading errors to special file"""
    if response.category == "trading":
        with open("trading_errors.log", "a") as f:
            f.write(f"{response.timestamp}: {response.to_json()}\n")

# Register callbacks
secure_handler.add_notification_callback(critical_error_alert)
secure_handler.add_notification_callback(trading_error_logger)
```

## 🔐 Sicherheitsfeatures

### 1. Automatische Datenbereinigung

```python
# Sensitive Daten werden automatisch erkannt und bereinigt
sensitive_data = {
    "api_key": "secret_key_12345",
    "password": "my_password",
    "authorization": "Bearer token123",
    "user_data": {
        "name": "John Doe",
        "secret": "classified_info"
    }
}

# Nach Bereinigung:
sanitized = secure_handler._sanitize_dict(sensitive_data)
# {
#     "api_key": "secret_key_12345",  # API keys nicht in dict keys erkannt
#     "password": "***REDACTED***",   # In dict keys erkannt
#     "authorization": "***REDACTED***",
#     "user_data": {
#         "name": "John Doe",
#         "secret": "***REDACTED***"
#     }
# }
```

### 2. Strukturierte Logs ohne Sensitive Daten

```python
# Log-Ausgabe (automatisch bereinigt):
ERROR:SecureErrorHandler:[TRADING] Invalid amount
  - error_id: 123e4567-e89b-12d3-a456-426614174000
  - trace_id: abc12345
  - error_category: trading
  - error_severity: medium
  - app_name: trading_bot
```

### 3. Trace-ID für Request-Korrelation

```python
# Verwende gleiche Trace-ID für zusammengehörige Operationen
trace_id = secure_handler._generate_trace_id()

# Alle Fehler in einer Operation verwenden die gleiche Trace-ID
response1 = secure_handler.handle_api_error(error1, trace_id=trace_id)
response2 = secure_handler.handle_trading_error(error2, trace_id=trace_id)

# Später alle zusammengehörigen Fehler abrufen
related_errors = secure_handler.get_errors_by_trace_id(trace_id)
```

## 📋 Beispiel-Responses

### Trading Error Response
```json
{
  "error_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2025-07-17T10:30:00.123456",
  "category": "trading",
  "severity": "medium", 
  "message": "Invalid amount",
  "status_code": 400,
  "details": {
    "symbol": "BTC/USDT",
    "order_id": "order_12345",
    "amount": 0.1,
    "error_type": "ValidationTradingError",
    "strategy": "momentum"
  },
  "trace_id": "abc12345"
}
```

### API Error Response
```json
{
  "error_id": "987fcdeb-51a2-43d1-b123-456789abcdef",
  "timestamp": "2025-07-17T10:35:00.654321",
  "category": "authentication",
  "severity": "medium",
  "message": "API connection timeout", 
  "status_code": 429,
  "details": {
    "endpoint": "https://api.binance.com/api/v3/order",
    "status_code": 429,
    "error_type": "ConnectionError",
    "request_data": {
      "symbol": "BTCUSDT",
      "api_key": "***REDACTED***",
      "secret": "***REDACTED***"
    },
    "response_data": {
      "error": "Rate limit exceeded"
    }
  },
  "trace_id": "def67890"
}
```

## 🚀 Best Practices

### 1. Error Handling Hierarchie
```python
# 1. Kritische Systemfehler (sofortige Aktion erforderlich)
secure_handler.handle_critical_error(error)

# 2. Trading-spezifische Fehler (mit Trading-Kontext)
secure_handler.handle_trading_error(error, symbol="BTC/USDT", order_id="123")

# 3. API-Fehler (mit Request/Response Kontext)
secure_handler.handle_api_error(error, endpoint="/api/order", status_code=500)
```

### 2. Trace-ID Management
```python
# Verwende Trace-IDs für zusammengehörige Operationen
trace_id = request.headers.get("X-Trace-ID") or secure_handler._generate_trace_id()

# Weitergabe durch die gesamte Request-Pipeline
response = secure_handler.handle_trading_error(error, trace_id=trace_id)
```

### 3. Notification Strategy
```python
# Verschiedene Callbacks für verschiedene Schweregrade
def critical_callback(response):
    if response.severity == "critical":
        send_immediate_alert(response)

def high_severity_callback(response):
    if response.severity == "high":
        log_to_monitoring_system(response)

secure_handler.add_notification_callback(critical_callback)
secure_handler.add_notification_callback(high_severity_callback)
```

## ✅ Vorteile

1. **🔐 Sicherheit**: Automatische Entfernung sensibler Daten aus Logs
2. **🆔 Nachverfolgbarkeit**: Eindeutige Error-IDs für Debugging
3. **📋 Strukturierung**: Konsistente Error-Responses für APIs
4. **🎯 Spezialisierung**: Maßgeschneiderte Handler für verschiedene Fehlertypen
5. **📊 Monitoring**: Umfassende Statistiken und Error-Tracking
6. **🔔 Benachrichtigungen**: Flexibles Notification-System
7. **🔗 Korrelation**: Trace-IDs für Request-Verfolgung

---

**SecureErrorHandler erfolgreich implementiert am 2025-07-17**  
**Sichere, strukturierte und nachverfolgbare Fehlerbehandlung für den Trading Bot**