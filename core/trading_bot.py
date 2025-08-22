"""
Trading Bot - Konsolidierte Version mit sauberer Architektur
"""

import asyncio
import logging
import importlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import pandas as pd

# Core imports
from config.settings import Settings
from config.environment import EnvironmentConfig
from data_sources.data_manager import DataManager
from core.interfaces import (
    IStrategy, IMarketAnalyzer, IRiskManager, 
    IPositionManager, IOrderManager, ISafetyManager
)
from utils.exceptions import (
    TradingBotException, StrategyError, ConfigurationError, 
    MLError, DataError, RiskManagementError
)

# Import core managers
from core.position import PositionManager
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from analysis.performance_tracker import PerformanceTracker
from core.paper_trading_engine import PaperTradingEngine

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Hauptklasse für den Trading Bot mit modularer Architektur
    """
    
    def __init__(
        self, 
        mode: str, 
        strategy_name: str, 
        settings: Settings,
        data_manager: DataManager,
        ml_components: Optional[Any] = None,
        strategy_router: Optional[Any] = None,
        safety_manager: Optional[ISafetyManager] = None,
        paper_trading: bool = False
    ):
        """
        Initialisiert den Trading Bot
        
        Args:
            mode: Trading-Modus (live, paper, backtest, optimize)
            strategy_name: Name der zu verwendenden Strategie
            settings: Konfigurationseinstellungen
            data_manager: Manager für Marktdaten
            ml_components: Optionale ML-Komponenten
            strategy_router: Optionaler Strategy Router für dynamische Strategieauswahl
            safety_manager: Optionaler Safety Manager für Risikokontrolle
        """
        
        # Validierung
        self._validate_configuration(mode, strategy_name, settings)
        
        # Core-Konfiguration
        self.mode = mode
        # Auto-enable paper trading if mode is 'paper'
        self.paper_trading = paper_trading or (mode == 'paper')
        self.strategy_name = strategy_name
        self.settings = settings
        self.data_manager = data_manager
        self.is_running = False
        
        # Paper Trading Engine initialisieren wenn aktiviert
        self.paper_engine = None
        if self.paper_trading:
            initial_balance = self.settings.get('paper_trading.initial_balance', 10000.0)
            self.paper_engine = PaperTradingEngine(
                initial_balance=initial_balance,
                exchange_client=self.data_manager.exchange if hasattr(self.data_manager, 'exchange') else None
            )
            logger.info(f"📝 Paper Trading Mode aktiviert mit ${initial_balance} virtueller Balance")
        
        # Symbol-Konfigurationen
        self.symbol_configs = self._build_symbol_configs()
        if not self.symbol_configs:
            raise ConfigurationError("Keine gültigen Symbole konfiguriert")
        
        logger.info(f"Konfigurierte Symbole: {list(self.symbol_configs.keys())}")
        
        # Core-Manager initialisieren
        self.position_manager = PositionManager(settings)
        self.order_manager = OrderManager(settings, self.position_manager)
        self.risk_manager = RiskManager(settings, self.position_manager)
        self.performance_tracker = PerformanceTracker(settings)
        
        # Optionale Komponenten
        self.ml_components = ml_components
        self.ml_enhanced = ml_components is not None
        self.strategy_router = strategy_router
        self.safety_manager = safety_manager
        
        # Strategie-Management
        self.strategies: Dict[str, IStrategy] = {}
        self.active_strategy: Optional[IStrategy] = None
        self._initialize_strategies()
        
        # Background-Tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Timing-Kontrolle
        self.last_check_time: Dict[str, datetime] = {}
        self.check_intervals = {
            'market_regime': timedelta(minutes=30),
            'performance': timedelta(minutes=15),
            'safety': timedelta(minutes=5)
        }
        
        logger.info(
            f"Trading Bot initialisiert - Modus: {mode}, "
            f"Strategie: {strategy_name}, ML: {self.ml_enhanced}, "
            f"Paper Trading: {self.paper_trading}"
        )
    
    def _validate_configuration(self, mode: str, strategy_name: str, settings: Settings):
        """Validiert die Bot-Konfiguration"""
        valid_modes = ['live', 'paper', 'backtest', 'optimize']
        if mode not in valid_modes:
            raise ConfigurationError(f"Ungültiger Modus '{mode}'. Erlaubt: {valid_modes}")
        
        if not isinstance(settings, Settings):
            raise ConfigurationError("Settings muss eine Settings-Instanz sein")
        
        # Pflicht-Einstellungen prüfen
        required_settings = [
            'trading.initial_capital',
            'exchange.name',
            'timeframes.analysis'
        ]
        
        for setting in required_settings:
            if settings.get(setting) is None:
                raise ConfigurationError(f"Pflicht-Einstellung fehlt: {setting}")
    
    def _build_symbol_configs(self) -> Dict[str, Dict[str, Any]]:
        """Erstellt Symbol-Konfigurationen aus Settings"""
        symbol_configs = {}
        
        # Symbole aus Settings laden
        symbols = self.settings.get('symbols', [])
        if not symbols:
            logger.warning("Keine Symbole konfiguriert, verwende Standard BTC/USDT")
            symbols = ['BTC/USDT']
        
        # Konfiguration für jedes Symbol erstellen
        for symbol in symbols:
            try:
                symbol_configs[symbol] = {
                    'timeframe': self.settings.get('timeframes.analysis', '1h'),
                    'min_candles': self.settings.get('data.min_candles', 200),
                    'enabled': True,
                    'max_position_size': self.settings.get(
                        f'symbols.{symbol}.max_position_size',
                        self.settings.get('trading.max_position_size', 0.1)
                    )
                }
            except Exception as e:
                logger.error(f"Fehler beim Konfigurieren von Symbol {symbol}: {e}")
                continue
        
        return symbol_configs
    
    def _initialize_strategies(self):
        """Initialisiert Trading-Strategien"""
        try:
            if self.strategy_name == "auto_routed":
                # Alle verfügbaren Strategien für Routing initialisieren
                self._load_all_strategies()
            else:
                # Einzelne Strategie laden
                strategy = self._load_strategy(self.strategy_name)
                if strategy:
                    self.strategies[self.strategy_name] = strategy
                    self.active_strategy = strategy
                else:
                    raise StrategyError(f"Strategie konnte nicht geladen werden: {self.strategy_name}")
            
            logger.info(f"{len(self.strategies)} Strategien initialisiert")
            
        except Exception as e:
            logger.error(f"Fehler bei Strategie-Initialisierung: {e}")
            raise StrategyError(f"Strategie-Initialisierung fehlgeschlagen: {e}")
    
    def _load_strategy(self, strategy_name: str) -> Optional[IStrategy]:
        """Lädt eine einzelne Strategie"""
        try:
            # Strategie-Konfiguration laden
            strategy_config = self.settings.get(f'strategy_configs.{strategy_name}', {})
            
            # ML-Enhancement hinzufügen wenn verfügbar
            if self.ml_enhanced:
                strategy_config['ml_components'] = self.ml_components
            
            # Strategie-Modul importieren
            module_path = f'strategies.{strategy_name}'
            module = importlib.import_module(module_path)
            
            # Strategie-Klasse finden
            strategy_class = self._find_strategy_class(module, strategy_name)
            if not strategy_class:
                raise AttributeError(f"Keine Strategie-Klasse in {module_path} gefunden")
            
            # Strategie instanziieren
            strategy = strategy_class(strategy_config)
            logger.info(f"Strategie geladen: {strategy_name}")
            return strategy
            
        except ImportError as e:
            logger.error(f"Strategie-Modul nicht gefunden: {strategy_name} - {e}")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Laden der Strategie {strategy_name}: {e}")
            return None
    
    def _find_strategy_class(self, module, strategy_name: str) -> Optional[Type[IStrategy]]:
        """Findet die Strategie-Klasse in einem Modul"""
        # Import here to avoid circular dependency
        from abc import ABC
        
        # Versuche verschiedene Namenskonventionen
        possible_names = [
            ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy',
            strategy_name.replace('_', '').capitalize() + 'Strategy',
            'Strategy'
        ]
        
        for name in possible_names:
            if hasattr(module, name):
                cls = getattr(module, name)
                # Check if it's a class that inherits from ABC (our Strategy base)
                if isinstance(cls, type) and issubclass(cls, ABC) and hasattr(cls, 'calculate_signal'):
                    return cls
        
        # Fallback: Erste Strategy-ähnliche Klasse finden
        for name, obj in vars(module).items():
            if (isinstance(obj, type) and 
                issubclass(obj, ABC) and 
                hasattr(obj, 'calculate_signal') and
                name != 'Strategy'):  # Exclude base Strategy class
                return obj
        
        return None
    
    def _load_all_strategies(self):
        """Lädt alle verfügbaren Strategien für Strategy Routing"""
        try:
            # Strategie-Registry importieren
            from strategies import STRATEGIES
            
            for strategy_name, strategy_class in STRATEGIES.items():
                try:
                    strategy_config = self.settings.get(f'strategy_configs.{strategy_name}', {})
                    
                    if self.ml_enhanced:
                        strategy_config['ml_components'] = self.ml_components
                    
                    strategy = strategy_class(strategy_config)
                    self.strategies[strategy_name] = strategy
                    
                except Exception as e:
                    logger.error(f"Fehler beim Initialisieren von Strategie {strategy_name}: {e}")
                    continue
            
            logger.info(f"{len(self.strategies)} Strategien für Routing geladen")
            
        except ImportError:
            logger.error("Strategie-Registry konnte nicht importiert werden")
            raise StrategyError("Keine Strategien für Auto-Routing verfügbar")
    
    async def start(self) -> None:
        """Startet den Trading Bot"""
        try:
            logger.info("Starte Trading Bot...")
            self.is_running = True
            
            # ML-Komponenten starten
            if self.ml_components and hasattr(self.ml_components, 'start'):
                try:
                    await self.ml_components.start()
                    logger.info("ML-Komponenten gestartet")
                except Exception as e:
                    logger.error(f"Fehler beim Starten der ML-Komponenten: {e}")
            
            # Background-Tasks starten
            self._start_background_tasks()
            
            # Haupt-Trading-Loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"Fehler beim Starten des Trading Bots: {e}")
            raise TradingBotException(f"Bot-Start fehlgeschlagen: {e}")
    
    def stop(self) -> None:
        """Stoppt den Trading Bot"""
        try:
            logger.info("Stoppe Trading Bot...")
            self.is_running = False
            
            # Background-Tasks beenden
            for task in self._background_tasks:
                task.cancel()
            
            # ML-Komponenten stoppen
            if self.ml_components and hasattr(self.ml_components, 'stop'):
                try:
                    self.ml_components.stop()
                    logger.info("ML-Komponenten gestoppt")
                except Exception as e:
                    logger.error(f"Fehler beim Stoppen der ML-Komponenten: {e}")
            
            logger.info("Trading Bot gestoppt")
            
        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Trading Bots: {e}")
    
    def _start_background_tasks(self):
        """Startet Background-Monitoring-Tasks"""
        # Safety-Monitoring
        if self.safety_manager:
            task = asyncio.create_task(self._safety_monitoring_loop())
            self._background_tasks.append(task)
        
        # Market-Regime-Monitoring für Auto-Routing
        if self.strategy_router and self.ml_components:
            task = asyncio.create_task(self._market_regime_monitoring_loop())
            self._background_tasks.append(task)
        
        # Performance-Monitoring
        task = asyncio.create_task(self._performance_monitoring_loop())
        self._background_tasks.append(task)
    
    async def _safety_monitoring_loop(self):
        """Background-Loop für Safety-Monitoring"""
        while self.is_running:
            try:
                if self._should_check('safety'):
                    await self.safety_manager.check_safety()
                    self.last_check_time['safety'] = datetime.now()
                
                await asyncio.sleep(60)  # Prüfe jede Minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Safety-Monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _market_regime_monitoring_loop(self):
        """Background-Loop für Market-Regime-Monitoring"""
        while self.is_running:
            try:
                if self._should_check('market_regime'):
                    await self._update_market_regime()
                    self.last_check_time['market_regime'] = datetime.now()
                
                await asyncio.sleep(300)  # Prüfe alle 5 Minuten
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Market-Regime-Monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _performance_monitoring_loop(self):
        """Background-Loop für Performance-Monitoring"""
        while self.is_running:
            try:
                if self._should_check('performance'):
                    self.performance_tracker.update_metrics()
                    self.last_check_time['performance'] = datetime.now()
                
                await asyncio.sleep(900)  # Prüfe alle 15 Minuten
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Performance-Monitoring: {e}")
                await asyncio.sleep(900)
    
    def _should_check(self, check_type: str) -> bool:
        """Prüft ob eine bestimmte Überprüfung durchgeführt werden soll"""
        last_check = self.last_check_time.get(check_type, datetime.min)
        interval = self.check_intervals.get(check_type, timedelta(minutes=5))
        return datetime.now() - last_check > interval
    
    async def _update_market_regime(self):
        """Aktualisiert das Market Regime für Strategy Routing"""
        if not self.ml_components or not hasattr(self.ml_components, 'market_regime_detector'):
            return
        
        try:
            # Marktdaten für Regime-Erkennung abrufen
            core_symbols = self.settings.get('ml.regime_core_symbols', ['BTC/USDT'])
            timeframe = self.settings.get('timeframes.analysis', '1h')
            
            all_data = []
            for symbol in core_symbols:
                data = await self.data_manager.fetch_data(symbol, timeframe, limit=100)
                if data is not None and not data.empty:
                    all_data.append(data)
            
            if all_data:
                # Regime erkennen
                combined_data = pd.concat(all_data, ignore_index=True)
                regime = self.ml_components.market_regime_detector.predict_regime(combined_data)
                
                # Strategy Router aktualisieren
                if self.strategy_router:
                    self.strategy_router.update_market_regime(regime)
                    logger.info(f"Market Regime aktualisiert: {regime}")
            
        except Exception as e:
            logger.error(f"Fehler bei Market-Regime-Update: {e}")
    
    async def _main_trading_loop(self):
        """Haupt-Trading-Ausführungsschleife"""
        logger.info("Starte Haupt-Trading-Loop...")
        
        while self.is_running:
            try:
                # Prüfe ob Trading pausiert ist
                if self.safety_manager and self.safety_manager.is_trading_paused():
                    logger.info("Trading durch Safety Manager pausiert")
                    await asyncio.sleep(60)
                    continue
                
                # Verarbeite jedes Symbol
                for symbol in self.symbol_configs:
                    if not self.is_running:
                        break
                    
                    try:
                        await self._process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Fehler beim Verarbeiten von Symbol {symbol}: {e}")
                        continue
                
                # Warte vor nächster Iteration
                check_interval = self.settings.get('timeframes.check_interval', 300)
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in der Haupt-Trading-Loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Haupt-Trading-Loop beendet")
    
    async def _process_symbol(self, symbol: str):
        """Verarbeitet Trading-Logik für ein spezifisches Symbol"""
        try:
            # Symbol-Konfiguration abrufen
            symbol_config = self.symbol_configs[symbol]
            if not symbol_config.get('enabled', True):
                return
            
            timeframe = symbol_config['timeframe']
            min_candles = symbol_config['min_candles']
            
            # Marktdaten abrufen
            data = await self.data_manager.fetch_data(symbol, timeframe, limit=min_candles)
            if data is None or data.empty:
                logger.warning(f"Keine Daten für {symbol} verfügbar")
                return
            
            # Aktuellen Preis ermitteln
            current_price = float(data['close'].iloc[-1])
            
            # Trading-Signale generieren
            signals = await self._generate_signals(symbol, data, current_price)
            
            # Signale verarbeiten
            for signal_type, signal_data in signals:
                if signal_type in ['BUY', 'SELL']:
                    await self._execute_signal(symbol, signal_type, signal_data, current_price)
            
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten von Symbol {symbol}: {e}")
            raise DataError(f"Symbol-Verarbeitung fehlgeschlagen für {symbol}: {e}")
    
    async def _generate_signals(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        current_price: float
    ) -> List[tuple]:
        """Generiert Trading-Signale mit aktiven Strategien"""
        signals = []
        
        try:
            # Aktive Strategie(n) bestimmen
            if self.strategy_router:
                # Multi-Strategie-Routing
                active_strategies = self.strategy_router.get_active_strategies(symbol)
                
                for strategy_name in active_strategies:
                    strategy = self.strategies.get(strategy_name)
                    if not strategy:
                        continue
                    
                    signal = await self._get_strategy_signal(strategy, symbol, data, current_price)
                    if signal[0] != 'HOLD':
                        signals.append(signal)
            
            else:
                # Einzel-Strategie-Modus
                if self.active_strategy:
                    signal = await self._get_strategy_signal(
                        self.active_strategy, symbol, data, current_price
                    )
                    if signal[0] != 'HOLD':
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Fehler bei Signal-Generierung für {symbol}: {e}")
            return []
    
    async def _get_strategy_signal(
        self, 
        strategy: IStrategy, 
        symbol: str, 
        data: pd.DataFrame, 
        current_price: float
    ) -> tuple:
        """Holt Signal von einer Strategie"""
        try:
            # ML-Enhanced Signal wenn verfügbar
            if hasattr(strategy, 'calculate_ml_enhanced_signal'):
                return await strategy.calculate_ml_enhanced_signal(symbol, data, current_price)
            else:
                # Standard-Signal
                return await strategy.calculate_signal(symbol, data, current_price)
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Signals von {strategy.name}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    async def _execute_signal(
        self, 
        symbol: str, 
        signal_type: str, 
        signal_data: Dict[str, Any], 
        current_price: float
    ):
        """Führt Trading-Signal mit Risiko-Management aus"""
        try:
            # Risiko-Management-Prüfungen
            if not self.risk_manager.can_open_position(symbol, signal_type, current_price):
                logger.info(f"Risiko-Manager blockiert {signal_type}-Signal für {symbol}")
                return
            
            # Positionsgröße berechnen
            confidence = signal_data.get('confidence', 0.5)
            position_size = self.risk_manager.calculate_position_size(
                symbol, signal_type, current_price, confidence
            )
            
            if position_size <= 0:
                logger.warning(f"Ungültige Positionsgröße für {symbol} berechnet")
                return
            
            # Order erstellen
            order = {
                'symbol': symbol,
                'side': signal_type.lower(),
                'amount': position_size,
                'price': current_price,
                'type': 'market',
                'strategy': signal_data.get('strategy', self.strategy_name),
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            # Order ausführen
            if self.mode == 'live':
                result = await self.order_manager.place_order(order)
            else:
                # Paper Trading oder Backtest
                result = await self.order_manager.simulate_order(order)
            
            if result.get('success', False):
                logger.info(
                    f"{signal_type}-Order ausgeführt für {symbol}: "
                    f"{position_size} @ {current_price}"
                )
            else:
                logger.error(f"Order-Ausführung fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
            
        except Exception as e:
            logger.error(f"Fehler bei Signal-Ausführung: {e}")
            raise TradingBotException(f"Signal-Ausführung fehlgeschlagen: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt umfassenden Bot-Status zurück"""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'strategy': self.strategy_name,
            'ml_enhanced': self.ml_enhanced,
            'strategies_count': len(self.strategies),
            'symbols': list(self.symbol_configs.keys()),
            'positions': self.position_manager.get_open_positions(),
            'performance': self.performance_tracker.get_summary(),
            'ml_status': self.ml_components.get_status() if self.ml_components else None,
            'safety_status': self.safety_manager.get_status() if self.safety_manager else None
        }
    
    def __enter__(self):
        """Context Manager Eintritt"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Austritt"""
        self.stop()
    
    def get_balance(self) -> Dict[str, Any]:
        """Get current balance"""
        try:
            if self.paper_trading and self.paper_engine:
                # Get balance from paper trading engine
                return {
                    'total': {
                        'USDT': self.paper_engine.virtual_balance
                    },
                    'free': {
                        'USDT': self.paper_engine.virtual_balance
                    }
                }
            elif hasattr(self.data_manager, 'exchange') and self.data_manager.exchange:
                # Get balance from exchange
                balance_usdt = self.data_manager.exchange.get_balance('USDT')
                return {
                    'total': {
                        'USDT': balance_usdt
                    },
                    'free': {
                        'USDT': balance_usdt
                    }
                }
            else:
                return {'total': {}, 'free': {}}
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'total': {}, 'free': {}}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if self.paper_trading and self.paper_engine:
                # Get positions from paper trading engine
                positions = []
                for pos_id, pos in self.paper_engine.positions.items():
                    if pos.status == 'open':
                        positions.append({
                            'symbol': pos.symbol,
                            'side': pos.side,
                            'amount': pos.size,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price,
                            'pnl': pos.unrealized_pnl,
                            'pnl_percent': pos.unrealized_pnl_percent,
                            'status': pos.status
                        })
                return positions
            elif hasattr(self, 'position_manager') and self.position_manager:
                # Get positions from position manager
                return self.position_manager.get_positions()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []