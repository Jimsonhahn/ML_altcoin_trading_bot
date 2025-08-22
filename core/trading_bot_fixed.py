"""
Enhanced Trading Bot - Fixed version with proper ML integration and error handling
"""

import asyncio
import logging
import importlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

# Core imports
from config.settings import Settings
from data_sources.data_manager import DataManager
from core.position import PositionManager
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from analysis.performance_tracker import PerformanceTracker
from core.safety_manager import SafetyManager
from strategies.strategy_base import Strategy
from utils.exceptions import (
    TradingBotException, StrategyError, ConfigurationError, 
    MLError, DataError, RiskManagementError
)

# Optional imports with fallbacks
try:
    from core.strategy_router import StrategyRouter
    HAS_STRATEGY_ROUTER = True
except ImportError:
    HAS_STRATEGY_ROUTER = False
    StrategyRouter = None

try:
    from ml_components import MLComponents
    HAS_ML_COMPONENTS = True
except ImportError:
    HAS_ML_COMPONENTS = False
    MLComponents = None

try:
    from utils.websocket_manager import WebSocketManager
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    WebSocketManager = None

try:
    from utils.notifier import Notifier
    HAS_NOTIFIER = True
except ImportError:
    HAS_NOTIFIER = False
    Notifier = None


logger = logging.getLogger(__name__)


class TradingBot:
    """
    Enhanced Trading Bot with ML integration and robust error handling
    """
    
    def __init__(self, mode: str, strategy_name: str, settings: Settings,
                 data_manager: DataManager,
                 ml_components: Optional[MLComponents] = None,
                 strategy_router: Optional[StrategyRouter] = None,
                 safety_manager: Optional[SafetyManager] = None):
        
        # Validate inputs
        self._validate_bot_configuration(mode, strategy_name, settings)
        
        # Core configuration
        self.mode = mode
        self.strategy_name = strategy_name
        self.settings = settings
        self.data_manager = data_manager
        self.is_running = False
        
        # Symbol configurations
        self.symbol_configs = self._build_symbol_configs()
        if not self.symbol_configs:
            raise ConfigurationError("No valid symbols configured")
        
        logger.info(f"Configured symbols: {list(self.symbol_configs.keys())}")
        
        # Initialize core managers
        self.position_manager = PositionManager(settings)
        self.order_manager = OrderManager(settings, self.position_manager)
        self.risk_manager = RiskManager(settings, self.position_manager)
        self.performance_tracker = PerformanceTracker(settings)
        
        # Initialize safety manager
        self.safety_manager = safety_manager
        if self.safety_manager:
            self.safety_manager.set_trading_bot(self)
        
        # Initialize ML components with enhanced error handling
        self.ml_components = ml_components
        self.ml_enhanced = ml_components is not None
        
        # Initialize strategy router
        self.strategy_router = strategy_router
        
        # Initialize strategies
        self.strategies = {}
        self.current_active_strategy = None
        self._initialize_strategies()
        
        # Optional components
        self.websocket_manager = None
        self.notifier = None
        self._initialize_optional_components()
        
        # Timing and control
        self.last_market_regime_check_time = datetime.min
        self.regime_check_interval = self.settings.get('ml.regime_check_interval', 1800)
        
        # Tasks
        self.safety_check_task = None
        self.market_regime_task = None
        
        logger.info(f"TradingBot initialized in {self.mode} mode with strategy: {self.strategy_name}")
        if self.ml_enhanced:
            logger.info("ML enhancement enabled")
        if self.strategy_router:
            logger.info("Strategy routing enabled")
    
    def _validate_bot_configuration(self, mode: str, strategy_name: str, settings: Settings):
        """Validate bot configuration parameters"""
        valid_modes = ['live', 'paper', 'backtest', 'optimize']
        if mode not in valid_modes:
            raise ConfigurationError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        
        if not isinstance(settings, Settings):
            raise ConfigurationError("Settings must be a Settings instance")
        
        # Validate required settings
        required_settings = [
            'trading.initial_capital',
            'exchange.name',
            'timeframes.analysis'
        ]
        
        for setting in required_settings:
            if settings.get(setting) is None:
                raise ConfigurationError(f"Required setting missing: {setting}")
    
    def _build_symbol_configs(self) -> Dict[str, Dict]:
        """Build symbol configurations from settings"""
        symbol_configs = {}
        
        # Get symbols from settings
        symbols = self.settings.get('symbols', [])
        if not symbols:
            logger.warning("No symbols configured, using default BTC/USDT")
            symbols = ['BTC/USDT']
        
        # Build config for each symbol
        for symbol in symbols:
            try:
                symbol_configs[symbol] = {
                    'timeframe': self.settings.get('timeframes.analysis', '1h'),
                    'min_candles': self.settings.get('data.min_candles', 200),
                    'enabled': True
                }
            except Exception as e:
                logger.error(f"Failed to configure symbol {symbol}: {e}")
                continue
        
        return symbol_configs
    
    def _initialize_strategies(self):
        """Initialize trading strategies with ML enhancement"""
        try:
            if self.strategy_name == "auto_routed":
                # Initialize all available strategies for routing
                self._initialize_all_strategies()
            else:
                # Initialize single strategy
                strategy = self._initialize_single_strategy(self.strategy_name)
                if strategy:
                    self.strategies[self.strategy_name] = strategy
                    self.current_active_strategy = strategy
                else:
                    raise StrategyError(f"Failed to initialize strategy: {self.strategy_name}")
            
            logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            raise StrategyError(f"Failed to initialize strategies: {e}")
    
    def _initialize_single_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """Initialize a single strategy with ML enhancement"""
        try:
            # Get strategy configuration
            strategy_config = self.settings.get(f'strategy_configs.{strategy_name}', {})
            
            # Add ML enhancement configuration
            if self.ml_enhanced:
                ml_config = self.settings.get('ml.strategy_enhancement', {})
                strategy_config.update(ml_config)
            
            # Import strategy module
            module = importlib.import_module(f'strategies.{strategy_name}')
            
            # Get strategy class
            class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
            strategy_class = getattr(module, class_name)
            
            # Initialize strategy with ML components
            strategy = strategy_class(strategy_config, ml_components=self.ml_components)
            
            logger.info(f"Initialized strategy: {strategy_name}")
            return strategy
            
        except ImportError as e:
            logger.error(f"Strategy module not found: {strategy_name} - {e}")
            return None
        except AttributeError as e:
            logger.error(f"Strategy class not found: {class_name} - {e}")
            return None
        except Exception as e:
            logger.error(f"Strategy initialization error: {strategy_name} - {e}")
            return None
    
    def _initialize_all_strategies(self):
        """Initialize all available strategies for strategy routing"""
        # Import strategy registry
        try:
            from strategies import STRATEGIES
        except ImportError:
            logger.error("Failed to import strategy registry")
            return
        
        # Initialize each strategy
        for strategy_name, strategy_class in STRATEGIES.items():
            try:
                strategy_config = self.settings.get(f'strategy_configs.{strategy_name}', {})
                
                # Add ML enhancement configuration
                if self.ml_enhanced:
                    ml_config = self.settings.get('ml.strategy_enhancement', {})
                    strategy_config.update(ml_config)
                
                # Initialize with ML components
                strategy = strategy_class(strategy_config, ml_components=self.ml_components)
                self.strategies[strategy_name] = strategy
                
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
                continue
        
        logger.info(f"Initialized {len(self.strategies)} strategies for routing")
    
    def _initialize_optional_components(self):
        """Initialize optional components with error handling"""
        # WebSocket Manager
        if HAS_WEBSOCKET and self.settings.get('websocket.enabled', False):
            try:
                self.websocket_manager = WebSocketManager(self.settings)
                logger.info("WebSocket Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WebSocket Manager: {e}")
        
        # Notifier
        if HAS_NOTIFIER and self.settings.get('notifications.enabled', False):
            try:
                self.notifier = Notifier(self.settings)
                logger.info("Notifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Notifier: {e}")
    
    async def start(self) -> None:
        """Start the trading bot with comprehensive error handling"""
        try:
            logger.info("Starting Altcoin Trading Bot...")
            self.is_running = True
            
            # Start ML components if available
            if self.ml_components and hasattr(self.ml_components, 'start'):
                try:
                    self.ml_components.start()
                    logger.info("ML components started")
                except Exception as e:
                    logger.error(f"Failed to start ML components: {e}")
            
            # Start optional components
            if self.websocket_manager:
                try:
                    await self.websocket_manager.start()
                    logger.info("WebSocket Manager started")
                except Exception as e:
                    logger.warning(f"Failed to start WebSocket Manager: {e}")
            
            # Initialize market regime for auto-routed strategies
            if self.strategy_name == "auto_routed" and self.ml_components:
                await self._initialize_market_regime()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            raise TradingBotException(f"Failed to start trading bot: {e}")
    
    def stop(self) -> None:
        """Stop the trading bot gracefully"""
        try:
            logger.info("Stopping trading bot...")
            self.is_running = False
            
            # Stop ML components
            if self.ml_components and hasattr(self.ml_components, 'stop'):
                try:
                    self.ml_components.stop()
                    logger.info("ML components stopped")
                except Exception as e:
                    logger.error(f"Error stopping ML components: {e}")
            
            # Cancel background tasks
            if self.safety_check_task:
                self.safety_check_task.cancel()
            if self.market_regime_task:
                self.market_regime_task.cancel()
            
            # Stop websocket manager
            if self.websocket_manager:
                try:
                    asyncio.create_task(self.websocket_manager.stop())
                except Exception as e:
                    logger.error(f"Error stopping WebSocket Manager: {e}")
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Safety monitoring
        if self.safety_manager:
            self.safety_check_task = asyncio.create_task(self._safety_check_loop())
        
        # Market regime monitoring for auto-routing
        if self.strategy_name == "auto_routed" and self.ml_components:
            self.market_regime_task = asyncio.create_task(self._market_regime_check_loop())
    
    async def _safety_check_loop(self):
        """Background safety monitoring loop"""
        while self.is_running:
            try:
                if self.safety_manager:
                    await self.safety_manager.check_safety()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Safety check error: {e}")
                await asyncio.sleep(60)
    
    async def _market_regime_check_loop(self):
        """Background market regime monitoring loop"""
        while self.is_running:
            try:
                await self._check_and_update_market_regime()
                await asyncio.sleep(self.regime_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market regime check error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _initialize_market_regime(self):
        """Initialize market regime for strategy routing"""
        if not self.ml_components or not hasattr(self.ml_components, 'market_regime_detector'):
            logger.warning("Market regime detector not available")
            return
        
        try:
            # Get core symbols for regime detection
            core_symbols = self.settings.get('ml.regime_core_symbols', ['BTC/USDT'])
            timeframe = self.settings.get('timeframes.analysis', '1h')
            
            # Fetch recent data for regime detection
            all_data = {}
            for symbol in core_symbols:
                try:
                    data = await self.data_manager.fetch_data(symbol, timeframe, limit=200)
                    if data is not None and not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            if all_data:
                # Detect initial market regime
                regime = self.ml_components.market_regime_detector.predict_regime(
                    pd.concat(list(all_data.values()), ignore_index=True)
                )
                
                # Update strategy router
                if self.strategy_router:
                    self.strategy_router.update_market_regime(regime)
                
                logger.info(f"Initial market regime detected: {regime}")
            else:
                logger.warning("No data available for market regime detection")
                
        except Exception as e:
            logger.error(f"Failed to initialize market regime: {e}")
    
    async def _check_and_update_market_regime(self):
        """Check and update market regime periodically"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_market_regime_check_time).total_seconds() < self.regime_check_interval:
                return
            
            if not self.ml_components or not hasattr(self.ml_components, 'market_regime_detector'):
                return
            
            # Get fresh data for regime detection
            core_symbols = self.settings.get('ml.regime_core_symbols', ['BTC/USDT'])
            timeframe = self.settings.get('timeframes.analysis', '1h')
            
            all_data = {}
            for symbol in core_symbols:
                try:
                    data = await self.data_manager.fetch_data(symbol, timeframe, limit=100)
                    if data is not None and not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch regime data for {symbol}: {e}")
            
            if all_data:
                # Predict current regime
                combined_data = pd.concat(list(all_data.values()), ignore_index=True)
                new_regime = self.ml_components.market_regime_detector.predict_regime(combined_data)
                
                # Update strategy router if regime changed
                if self.strategy_router:
                    current_regime = getattr(self.strategy_router, 'current_regime', None)
                    if new_regime != current_regime:
                        logger.info(f"Market regime changed: {current_regime} -> {new_regime}")
                        self.strategy_router.update_market_regime(new_regime)
                        
                        # Send notification if available
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"Market regime changed to {new_regime}",
                                "regime_change"
                            )
            
            self.last_market_regime_check_time = current_time
            
        except Exception as e:
            logger.error(f"Error checking market regime: {e}")
    
    async def _main_trading_loop(self):
        """Main trading execution loop"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check if trading is paused by safety manager
                if self.safety_manager and self.safety_manager.is_trading_paused():
                    logger.info("Trading paused by safety manager")
                    await asyncio.sleep(60)
                    continue
                
                # Execute trading logic for each symbol
                for symbol in self.symbol_configs:
                    if not self.is_running:
                        break
                    
                    try:
                        await self._process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {e}")
                        continue
                
                # Update performance metrics
                self.performance_tracker.update_metrics()
                
                # Wait before next iteration
                check_interval = self.settings.get('timeframes.check_interval', 300)
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        logger.info("Main trading loop stopped")
    
    async def _process_symbol(self, symbol: str):
        """Process trading logic for a specific symbol"""
        try:
            # Get symbol configuration
            symbol_config = self.symbol_configs[symbol]
            timeframe = symbol_config['timeframe']
            min_candles = symbol_config['min_candles']
            
            # Fetch market data
            data = await self.data_manager.fetch_data(symbol, timeframe, limit=min_candles)
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Get current price
            current_price = float(data['close'].iloc[-1])
            
            # Generate trading signals
            signals = await self._generate_signals(symbol, data, current_price)
            
            # Process signals
            for signal, signal_data in signals:
                if signal in ['BUY', 'SELL']:
                    await self._execute_signal(symbol, signal, signal_data, current_price)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            raise DataError(f"Failed to process {symbol}: {e}")
    
    async def _generate_signals(self, symbol: str, data: pd.DataFrame, 
                               current_price: float) -> List[tuple]:
        """Generate trading signals using active strategies"""
        signals = []
        
        try:
            if self.strategy_router:
                # Multi-strategy routing
                active_strategies = self.strategy_router.get_active_strategies()
                
                for strategy_name in active_strategies:
                    strategy = self.strategies.get(strategy_name)
                    if not strategy:
                        logger.warning(f"Strategy {strategy_name} not found")
                        continue
                    
                    # Generate signal with ML enhancement
                    signal, signal_data = self._get_enhanced_signal(
                        strategy, symbol, data, current_price
                    )
                    
                    if signal != 'HOLD':
                        signals.append((signal, signal_data))
            
            else:
                # Single strategy mode
                if self.current_active_strategy:
                    signal, signal_data = self._get_enhanced_signal(
                        self.current_active_strategy, symbol, data, current_price
                    )
                    
                    if signal != 'HOLD':
                        signals.append((signal, signal_data))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _get_enhanced_signal(self, strategy: Strategy, symbol: str, 
                           data: pd.DataFrame, current_price: float) -> tuple:
        """Get ML-enhanced signal from strategy"""
        try:
            # Use ML-enhanced signal if available
            if hasattr(strategy, 'calculate_ml_enhanced_signal'):
                return strategy.calculate_ml_enhanced_signal(symbol, data, current_price)
            else:
                # Fallback to regular signal
                return strategy.calculate_signal(symbol, data, current_price)
                
        except Exception as e:
            logger.error(f"Error getting signal from {strategy.name}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    async def _execute_signal(self, symbol: str, signal: str, signal_data: Dict, current_price: float):
        """Execute trading signal with risk management"""
        try:
            # Risk management checks
            if not self.risk_manager.can_open_position(symbol, signal, current_price):
                logger.info(f"Risk manager blocked {signal} signal for {symbol}")
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, signal, current_price, signal_data.get('confidence', 0.5)
            )
            
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}")
                return
            
            # Create order
            order = {
                'symbol': symbol,
                'side': signal.lower(),
                'amount': position_size,
                'price': current_price,
                'type': 'market',
                'strategy': signal_data.get('strategy', 'unknown'),
                'confidence': signal_data.get('confidence', 0.5),
                'timestamp': datetime.now()
            }
            
            # Execute order
            if self.mode == 'live':
                result = await self.order_manager.place_order(order)
            else:
                # Paper trading or backtest
                result = await self.order_manager.simulate_order(order)
            
            if result.get('success', False):
                logger.info(f"Executed {signal} order for {symbol}: {position_size} @ {current_price}")
                
                # Send notification
                if self.notifier:
                    await self.notifier.send_notification(
                        f"Executed {signal} {symbol}: {position_size} @ {current_price}",
                        "trade_executed"
                    )
            else:
                logger.error(f"Failed to execute order: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            raise TradingBotException(f"Signal execution failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'strategy': self.strategy_name,
            'ml_enhanced': self.ml_enhanced,
            'strategies_count': len(self.strategies),
            'symbols': list(self.symbol_configs.keys()),
            'positions': len(self.position_manager.get_open_positions()),
            'performance': self.performance_tracker.get_summary(),
            'ml_status': self.ml_components.get_ml_status() if self.ml_components else None,
            'safety_status': self.safety_manager.get_status() if self.safety_manager else None
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()