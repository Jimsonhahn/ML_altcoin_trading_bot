"""
Utility Components
==================

Hilfsfunktionen und -klassen für den Trading Bot:
- Logger: Strukturiertes Logging
- Notifier: Multi-Channel Benachrichtigungen  
- Validators: Input-Validierung
- SecretManager: Sichere Schlüsselverwaltung
- ErrorHandler: Fehlerbehandlung
"""

# Logging und Monitoring
try:
    from .logger import get_logger
    # setup_logging Funktion wird möglicherweise anders benannt
    try:
        from .logger import setup_logging
    except ImportError:
        from .logger import setup_logger as setup_logging
except ImportError:
    # Fallback Logger
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

# Benachrichtigungen
from .notifier import NotificationManager

# Validierung und Sicherheit
from .validators import validate_config, validate_trade_params
from .secret_manager import SecretManager
from .error_handler import ErrorHandler

# Exceptions
from .exceptions import TradingBotError, ValidationError, ExchangeError

__version__ = "1.0.0"
__all__ = [
    'get_logger',
    'setup_logging',
    'NotificationManager',
    'validate_config',
    'validate_trade_params', 
    'SecretManager',
    'ErrorHandler',
    'TradingBotError',
    'ValidationError',
    'ExchangeError'
]