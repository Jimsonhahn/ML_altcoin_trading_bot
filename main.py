#!/usr/bin/env python3
"""
Altcoin Trading Bot - Unified Main Entry Point
Konsolidierte Version mit sauberer Architektur und Import-Struktur
"""

import argparse
import asyncio
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Logging früh konfigurieren
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'trading_bot.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file), mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Core imports mit strukturierter Fehlerbehandlung
try:
    from config.environment import get_config, TradingMode
    from config.settings import Settings
    from data_sources.data_manager import DataManager
    from core.trading_bot import TradingBot
    from core.strategy_router import StrategyRouter
    from core.market_analyzer import MarketAnalyzer
    from core.safety_manager import SafetyManager
    from core.di_container import DIContainer
except ImportError as e:
    logger.error(f"Kritischer Import fehlgeschlagen: {e}")
    sys.exit(1)

# Optionale Komponenten
ML_AVAILABLE = False
NOTIFIER_AVAILABLE = False

try:
    from ml_components.enhanced_ml_components import create_enhanced_ml_components
    ML_AVAILABLE = True
    logger.info("ML-Komponenten verfügbar")
except ImportError:
    logger.warning("ML-Komponenten nicht verfügbar - Trading ohne ML-Unterstützung")

try:
    from utils.notifier import Notifier
    NOTIFIER_AVAILABLE = True
    logger.info("Notifier verfügbar")
except ImportError:
    logger.warning("Notifier nicht verfügbar - Keine Benachrichtigungen möglich")


class TradingBotApplication:
    """Hauptanwendungsklasse für den Trading Bot"""
    
    def __init__(self):
        self.config = None
        self.settings = None
        self.di_container = None
        self.trading_bot = None
        self.is_running = False
        
    def parse_arguments(self) -> argparse.Namespace:
        """Kommandozeilen-Argumente parsen"""
        parser = argparse.ArgumentParser(
            description="Altcoin Trading Bot mit ML-Unterstützung"
        )
        
        # Kern-Argumente
        parser.add_argument(
            '--mode', 
            choices=['live', 'paper', 'backtest', 'optimize'], 
            default='paper',
            help='Trading-Modus (Standard: paper)'
        )
        
        parser.add_argument(
            '--strategy', 
            type=str, 
            default='momentum',
            help='Trading-Strategie (Standard: momentum)'
        )
        
        parser.add_argument(
            '--config-profile', 
            type=str, 
            default='default',
            help='Konfigurations-Profil (Standard: default)'
        )
        
        parser.add_argument(
            '--auto-strategy', 
            action='store_true',
            help='Automatische Strategieauswahl mit ML aktivieren'
        )
        
        parser.add_argument(
            '--disable-ml', 
            action='store_true',
            help='ML-Features komplett deaktivieren'
        )
        
        # Trading-Parameter
        parser.add_argument(
            '--symbols', 
            type=str, 
            nargs='+',
            help='Trading-Symbole (z.B. BTC/USDT ETH/USDT)'
        )
        
        parser.add_argument(
            '--capital', 
            type=float,
            help='Anfangskapital'
        )
        
        # Entwicklung und Testing
        parser.add_argument(
            '--dry-run', 
            action='store_true',
            help='Ohne Trade-Ausführung laufen'
        )
        
        parser.add_argument(
            '--verbose', 
            action='store_true',
            help='Ausführliches Logging aktivieren'
        )
        
        return parser.parse_args()
    
    def setup_logging(self, verbose: bool = False):
        """Logging-Konfiguration"""
        level = logging.DEBUG if verbose else logging.INFO
        
        # Logs-Verzeichnis erstellen
        Path('logs').mkdir(exist_ok=True)
        
        # Root-Logger konfigurieren
        logging.getLogger().setLevel(level)
        
        # Externe Logger dämpfen
        for logger_name in ['urllib3', 'requests', 'websocket', 'ccxt']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def initialize_components(self, args: argparse.Namespace):
        """Alle Komponenten initialisieren"""
        logger.info("Initialisiere Komponenten...")
        
        # Environment-Config laden
        self.config = get_config()
        
        # Settings mit Profil laden
        self.settings = Settings(args.config_profile)
        
        # Command-Line-Überschreibungen anwenden
        self._apply_command_line_overrides(args)
        
        # Dependency Injection Container erstellen
        self.di_container = DIContainer(self.settings, self.config)
        
        # ML-Komponenten registrieren (wenn verfügbar und nicht deaktiviert)
        if ML_AVAILABLE and not args.disable_ml:
            self._register_ml_components()
        
        # Notifier registrieren (wenn verfügbar)
        if NOTIFIER_AVAILABLE:
            self._register_notifier()
        
        # Core-Komponenten registrieren
        self._register_core_components()
        
        # Trading Bot erstellen
        self._create_trading_bot(args)
        
        logger.info("Alle Komponenten erfolgreich initialisiert")
    
    def _apply_command_line_overrides(self, args: argparse.Namespace):
        """Command-Line-Parameter auf Settings anwenden"""
        if args.symbols:
            self.settings.set('symbols', args.symbols)
        if args.capital:
            self.settings.set('trading.initial_capital', args.capital)
        if args.mode:
            self.settings.set('trading.mode', args.mode)
    
    def _register_ml_components(self):
        """ML-Komponenten im DI-Container registrieren"""
        try:
            ml_components = create_enhanced_ml_components(self.settings)
            self.di_container.register('ml_components', ml_components)
            logger.info("ML-Komponenten registriert")
        except Exception as e:
            logger.error(f"Fehler beim Registrieren der ML-Komponenten: {e}")
    
    def _register_notifier(self):
        """Notifier im DI-Container registrieren"""
        try:
            notifier = Notifier(self.settings)
            self.di_container.register('notifier', notifier)
            logger.info("Notifier registriert")
        except Exception as e:
            logger.error(f"Fehler beim Registrieren des Notifiers: {e}")
    
    def _register_core_components(self):
        """Core-Komponenten im DI-Container registrieren"""
        # Data Manager
        data_manager = DataManager(self.settings)
        self.di_container.register('data_manager', data_manager)
        
        # Market Analyzer
        market_analyzer = MarketAnalyzer(self.settings)
        self.di_container.register('market_analyzer', market_analyzer)
        
        # Safety Manager
        safety_manager = SafetyManager(self.settings)
        self.di_container.register('safety_manager', safety_manager)
        
        # Strategy Router (wenn Auto-Strategy aktiviert)
        if self.settings.get('strategy_router.enabled', False):
            strategy_router = StrategyRouter(self.settings)
            self.di_container.register('strategy_router', strategy_router)
    
    def _create_trading_bot(self, args: argparse.Namespace):
        """Trading Bot mit allen Abhängigkeiten erstellen"""
        mode = args.mode
        strategy_name = args.strategy
        
        # Bei Auto-Strategy entsprechend anpassen
        if args.auto_strategy:
            strategy_name = "auto_routed"
            self.settings.set('strategy_router.enabled', True)
        
        # Trading Bot erstellen
        self.trading_bot = TradingBot(
            mode=mode,
            strategy_name=strategy_name,
            settings=self.settings,
            data_manager=self.di_container.get('data_manager'),
            ml_components=self.di_container.get('ml_components'),
            strategy_router=self.di_container.get('strategy_router'),
            safety_manager=self.di_container.get('safety_manager')
        )
        
        logger.info(f"Trading Bot erstellt - Modus: {mode}, Strategie: {strategy_name}")
    
    def setup_signal_handlers(self):
        """Signal-Handler für sauberes Herunterfahren einrichten"""
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} empfangen, fahre herunter...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Hauptausführungsschleife"""
        try:
            self.is_running = True
            await self.trading_bot.start()
            
            # Im Backtest-Modus direkt beenden
            if self.trading_bot.mode == 'backtest':
                logger.info("Backtest abgeschlossen")
                return
            
            # Ansonsten weiterlaufen
            logger.info("Trading Bot läuft. Drücke Ctrl+C zum Beenden.")
            while self.is_running and self.trading_bot.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Beendigung durch Benutzer angefordert")
        except Exception as e:
            logger.error(f"Fehler in der Hauptschleife: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Anwendung sauber herunterfahren"""
        logger.info("Fahre Trading Bot herunter...")
        self.is_running = False
        
        if self.trading_bot:
            self.trading_bot.stop()
        
        if self.di_container:
            self.di_container.cleanup()
        
        logger.info("Trading Bot heruntergefahren")
    
    def print_startup_info(self, args: argparse.Namespace):
        """Startup-Informationen ausgeben"""
        logger.info("=" * 80)
        logger.info("🚀 Altcoin Trading Bot gestartet")
        logger.info("=" * 80)
        logger.info(f"Umgebung: {self.config.environment.value}")
        logger.info(f"Trading-Modus: {args.mode}")
        logger.info(f"Strategie: {args.strategy}")
        logger.info(f"Konfigurations-Profil: {args.config_profile}")
        logger.info(f"ML aktiviert: {ML_AVAILABLE and not args.disable_ml}")
        logger.info(f"Auto-Strategie: {args.auto_strategy}")
        logger.info(f"Kapital: ${self.settings.get('trading.initial_capital'):,.2f}")
        logger.info(f"Symbole: {self.settings.get('symbols', [])}")
        logger.info("=" * 80)


def main():
    """Haupteinstiegspunkt"""
    app = TradingBotApplication()
    
    try:
        # Argumente parsen
        args = app.parse_arguments()
        
        # Logging einrichten
        app.setup_logging(args.verbose)
        
        # Komponenten initialisieren
        app.initialize_components(args)
        
        # Startup-Info ausgeben
        app.print_startup_info(args)
        
        # Signal-Handler einrichten
        app.setup_signal_handlers()
        
        # Bot ausführen
        asyncio.run(app.run())
        
    except Exception as e:
        logger.error(f"Fataler Fehler: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Trading Bot beendet")


if __name__ == "__main__":
    main()