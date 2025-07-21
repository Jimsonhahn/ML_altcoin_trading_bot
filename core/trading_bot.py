
# Dependency Injection Support
from core.interfaces import ITradingBot, global_event_bus
from core.di_container import container
# core/trading_bot.py
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type, List
from pydantic import validate_arguments, Field
import pandas as pd

from config.settings import Settings
from core.exchange import ExchangeManager
from core.order_manager import OrderManager
from core.position import PositionManager
from core.risk_manager import RiskManager
# Removed direct import - using DI
from data_sources.data_manager import DataManager
from ml_components import MLComponents  # Assuming MLComponents is in ml_components/__init__.py
from core.strategy_router import StrategyRouter  # Assuming StrategyRouter is in core
from strategies import STRATEGIES, Strategy  # Assuming STRATEGIES is dict and Strategy is base class
from utils.logger import setup_logger
from Analysis.performance_tracker import PerformanceTracker

# Import validation framework
from utils.validators import (
    validate_trading_symbol, validate_amount, validate_config,
    TradingMode, ValidationError
)
from utils.error_handler import (
    handle_errors, ErrorCategory, ValidationTradingError,
    secure_error_handler, SecureErrorResponse, SecureErrorHandler
)
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)


class TradingBot(ITradingBot):
    
    @property
    def safety_manager(self):
        """Lazy loaded safety manager via DI"""
        if not hasattr(self, '_safety_manager'):
            from core.di_container import get_safety_manager
            self._safety_manager = get_safety_manager()
        return self._safety_manager

    def __init__(self, mode: str, strategy_name: str, settings: Settings,
                 data_manager: DataManager,
                 ml_components: Optional[MLComponents] = None,
                 strategy_router: Optional[StrategyRouter] = None,
                 safety_manager: Optional[SafetyManager] = None):

        # Initialize secure error handler
        self.error_handler = secure_error_handler
        
        # Validate configuration before initialization
        self._validate_bot_configuration(mode, strategy_name, settings)
        
        self.mode = mode
        self.settings = settings
        self.data_manager = data_manager
        self.running = False
        self.trade_thread: Optional[threading.Thread] = None
        self.check_interval = self.settings.get('timeframes.check_interval', 300)  # Default 5 mins

        self.exchange = ExchangeManager('binance', mode)
        self.order_manager = OrderManager(self.exchange, settings)
        self.position_manager = PositionManager(settings)
        self.risk_manager = RiskManager(settings, self.position_manager)
        self.performance_tracker = PerformanceTracker(settings)  # Initialize PerformanceTracker

        self.safety_manager: Optional[SafetyManager] = safety_manager
        if self.safety_manager:
            self.safety_manager.set_trading_bot(self)  # Ensure SafetyManager has bot reference

        self.ml_components = ml_components
        self.strategy_router = strategy_router

        self.current_active_strategy: Optional[Strategy] = None
        self.strategy_name = strategy_name  # Keep track of requested strategy, can be 'auto_routed'

        # Initialize strategy based on mode or router
        self._initialize_strategy()

        self.last_market_regime_check_time: datetime = datetime.min
        self.regime_check_interval = self.settings.get('ml.regime_check_interval', 1800)  # Default 30 mins

        logger.info(f"TradingBot initialized in {self.mode} mode with strategy: {self.strategy_name}")
        if self.ml_components and self.strategy_router:
            logger.info("ML-powered Strategy Routing is enabled.")
        elif self.strategy_name == "auto_routed":
            logger.warning(
                "Automatic strategy routing requested, but ML components or Strategy Router are not fully initialized.")

    def _validate_bot_configuration(self, mode: str, strategy_name: str, settings: Settings):
        """
        Validates bot configuration at initialization
        """
        try:
            # Validate mode
            valid_modes = ['live', 'paper', 'backtest']
            if mode.lower() not in valid_modes:
                raise ValidationTradingError(
                    f"Invalid trading mode '{mode}'. Must be one of: {valid_modes}",
                    field="mode",
                    value=mode
                )
            
            # Validate strategy name
            if not strategy_name or not isinstance(strategy_name, str):
                raise ValidationTradingError(
                    "Strategy name must be a non-empty string",
                    field="strategy_name",
                    value=strategy_name
                )
            
            # Validate strategy exists (unless auto_routed)
            if strategy_name != "auto_routed" and strategy_name not in STRATEGIES:
                available_strategies = list(STRATEGIES.keys()) + ["auto_routed"]
                raise ValidationTradingError(
                    f"Unknown strategy '{strategy_name}'. Available strategies: {available_strategies}",
                    field="strategy_name",
                    value=strategy_name
                )
            
            # Convert settings to dict for validation
            config_dict = self._extract_config_from_settings(settings)
            
            # Validate core configuration parameters
            validated_config = validate_config(config_dict)
            
            logger.info("✅ Bot configuration validation passed")
            
        except (ValidationError, PydanticValidationError) as e:
            error_response = self.error_handler.handle_trading_error(
                error=e,
                context={
                    "operation": "bot_configuration_validation",
                    "mode": mode,
                    "strategy_name": strategy_name
                }
            )
            logger.error(f"Bot configuration validation failed - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Bot configuration validation failed: {str(e)}", field="configuration")
        except Exception as e:
            error_response = self.error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "bot_configuration_validation",
                    "mode": mode,
                    "strategy_name": strategy_name
                }
            )
            logger.error(f"Critical error during configuration validation - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Unexpected error during configuration validation: {str(e)}", field="configuration")

    def _extract_config_from_settings(self, settings: Settings) -> Dict[str, Any]:
        """
        Extract relevant configuration parameters from Settings object for validation
        """
        return {
            "trading_mode": TradingMode.PAPER if self.mode == "paper" else TradingMode.LIVE,
            "max_position_size": settings.get('trading.max_position_size', 1000.0),
            "max_positions": settings.get('trading.max_positions', 5),
            "max_drawdown": settings.get('risk.max_drawdown', 0.20),
            "stop_loss_percentage": settings.get('risk.stop_loss_percentage', 0.02),
            "take_profit_percentage": settings.get('risk.take_profit_percentage', 0.05),
            "risk_per_trade": settings.get('risk.risk_per_trade', 0.02),
            "exchange_name": settings.get('exchange.name', 'binance'),
            "api_rate_limit": settings.get('exchange.rate_limit', 1200),
        }

    def _initialize_strategy(self):
        """
        Initializes the trading strategy based on self.strategy_name.
        If "auto_routed", defers to strategy_router to set the initial strategy.
        """
        if self.strategy_name == "auto_routed":
            if self.strategy_router and self.ml_components:
                # Attempt to detect initial regime and set strategy
                logger.info("Attempting initial market regime detection for auto-routing.")
                # Fetch recent data to detect initial regime
                # This needs actual data fetching for core symbols
                initial_market_data = {}
                for symbol in self.settings.get('ml.regime_core_symbols', ["BTC/USDT"]):
                    df = self.data_manager.get_historical_data(symbol, self.settings.get('timeframes.analysis', '1h'),
                                                               (datetime.now() - timedelta(
                                                                   days=self.settings.get('ml.min_data_points_for_ml',
                                                                                          200) / 24)).strftime(
                                                                   '%Y-%m-%d'),  # Approx data for period
                                                               datetime.now().strftime('%Y-%m-%d'))
                    if not df.empty:
                        initial_market_data[symbol] = df

                if initial_market_data:
                    regime_info = self.ml_components.market_regime_detector.predict_regime(initial_market_data)
                    if regime_info["status"] == "success":
                        initial_regime_label = regime_info["label"]
                        # Call router to set initial strategy based on detected regime
                        self.strategy_router.update_market_regime(initial_regime_label,
                                                                  self.position_manager.get_total_capital())
                        # The router will activate strategies. We need to get references to them.
                        active_strategies = self.strategy_router.get_active_strategies()
                        if active_strategies:
                            # For simplicity, if multiple are active, take the first one or manage a list.
                            # For now, let's assume the bot will manage these through the router.
                            # The bot itself might not directly hold a single current_active_strategy,
                            # but rather delegate to the router.
                            # For this implementation, let's ensure the bot's logic works with the router
                            # without needing a single `current_active_strategy` if multiple are managed.
                            logger.info(
                                f"Strategy Router set initial active strategies: {list(active_strategies.keys())}")
                            self.current_active_strategy = list(active_strategies.values())[
                                0] if active_strategies else None  # For simple direct use if only one
                        else:
                            logger.warning("Strategy Router could not set any initial strategies.")
                            # Fallback to default strategy if auto-routing failed initially
                            self._set_fallback_strategy(self.settings.get('trading.default_strategy', 'momentum'))
                    else:
                        logger.warning(
                            f"Initial market regime detection failed: {regime_info['reason']}. Setting fallback strategy.")
                        self._set_fallback_strategy(self.settings.get('trading.default_strategy', 'momentum'))
                else:
                    logger.warning("No initial market data for regime detection. Setting fallback strategy.")
                    self._set_fallback_strategy(self.settings.get('trading.default_strategy', 'momentum'))

            else:
                logger.warning(
                    "Strategy Router or ML Components not available for 'auto_routed' strategy. Setting fallback strategy.")
                self._set_fallback_strategy(self.settings.get('trading.default_strategy', 'momentum'))
        else:
            # Initialize a single, fixed strategy
            self._set_fixed_strategy(self.strategy_name)

    def _set_fixed_strategy(self, strategy_name: str):
        """Sets a fixed strategy."""
        if strategy_name in STRATEGIES:
            strategy_class = STRATEGIES[strategy_name]
            strategy_params = self.settings.get(f'strategy_configs.{strategy_name}', {})
            self.current_active_strategy = strategy_class(strategy_params)
            logger.info(f"Fixed strategy '{strategy_name}' initialized.")
        else:
            logger.error(f"Strategy '{strategy_name}' not found. Bot cannot start without a valid strategy.")
            self.running = False  # Prevent bot from starting

    def _set_fallback_strategy(self, strategy_name: str):
        """Sets a fallback strategy if auto-routing fails."""
        logger.info(f"Setting fallback strategy: {strategy_name}")
        self._set_fixed_strategy(strategy_name)

    def start(self):
        if self.running:
            logger.info("Bot is already running.")
            return

        if not self.current_active_strategy and self.strategy_name != "auto_routed":
            logger.error("No strategy initialized. Bot cannot start.")
            return

        # Start real-time risk monitoring
        initial_capital = self.settings.get('trading.initial_capital', 10000)
        try:
            self.risk_manager.start_realtime_monitoring(initial_capital)
            logger.info("Real-time risk monitoring started")
        except Exception as e:
            logger.error(f"Failed to start real-time risk monitoring: {e}")

        self.running = True
        self.trade_thread = threading.Thread(target=self._run_trading_loop, daemon=True)
        self.trade_thread.start()
        logger.info("Trading bot started.")

    def stop(self):
        if not self.running:
            logger.info("Bot is not running.")
            return
        self.running = False
        
        # Stop real-time risk monitoring
        try:
            self.risk_manager.stop_realtime_monitoring()
            logger.info("Real-time risk monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping real-time risk monitoring: {e}")
        
        if self.trade_thread:
            self.trade_thread.join(timeout=5)  # Give time for thread to finish
            if self.trade_thread.is_alive():
                logger.warning("Trade thread did not terminate gracefully.")
        logger.info("Trading bot stopped.")

    def _run_trading_loop(self):
        logger.info("Trading loop started.")
        while self.running:
            try:
                # Check killswitch first
                if self.safety_manager and self.safety_manager.is_killswitch_active():
                    logger.warning("Killswitch is active. Trading operations are paused.")
                    time.sleep(self.check_interval)  # Wait before re-checking
                    continue  # Skip trading logic

                # Periodically check and update market regime and strategies
                self._check_and_update_market_regime()

                # Fetch latest market data for active strategies
                # If using strategy router, get symbols from currently active strategies
                symbols_to_fetch: List[str] = []
                if self.strategy_router:
                    for s in self.strategy_router.get_active_strategies().values():
                        if s.trading_pair not in symbols_to_fetch:  # Assuming strategy has a trading_pair attribute
                            symbols_to_fetch.append(s.trading_pair)
                elif self.current_active_strategy:
                    symbols_to_fetch.append(
                        self.current_active_strategy.trading_pair)  # Assuming fixed strategy has one

                if not symbols_to_fetch:
                    logger.warning("No active strategies or symbols to fetch data for. Waiting...")
                    time.sleep(self.check_interval)
                    continue

                for symbol in symbols_to_fetch:
                    # Fetch latest candle data (e.g., 1-hour candle)
                    ohlcv = self.exchange.fetch_ohlcv(symbol, self.settings.get('timeframes.analysis', '1h'),
                                                      limit=self.settings.get('data.min_candles', 200))
                    if ohlcv:
                        df = self.data_manager.convert_ohlcv_to_dataframe(ohlcv)
                        latest_candle = df.iloc[-1]
                        
                        # Update real-time price data
                        current_price = latest_candle['close']
                        try:
                            self.risk_manager.update_realtime_price(symbol, current_price)
                        except Exception as e:
                            logger.debug(f"Error updating real-time price for {symbol}: {e}")

                        # Execute strategy logic
                        if self.strategy_router:
                            active_strategies = self.strategy_router.get_active_strategies()
                            for strategy_name, strategy_instance in active_strategies.items():
                                if strategy_instance.trading_pair == symbol:  # Only run for relevant strategy
                                    signal = strategy_instance.generate_signal(df, latest_candle)
                                    if signal:
                                        self._execute_signal(symbol, signal, strategy_instance)
                        elif self.current_active_strategy and self.current_active_strategy.trading_pair == symbol:
                            signal = self.current_active_strategy.generate_signal(df, latest_candle)
                            if signal:
                                self._execute_signal(symbol, signal, self.current_active_strategy)
                    else:
                        logger.warning(f"Could not fetch OHLCV data for {symbol}. Skipping this cycle.")

                # Update portfolio and performance metrics
                self.position_manager.update_portfolio_value(self.exchange.get_current_prices())
                self.performance_tracker.track_performance(self.position_manager.get_total_capital())

                time.sleep(self.check_interval)  # Wait for the next check cycle
            except Exception as e:
                # Use secure error handler for trading loop errors
                error_response = self.error_handler.handle_critical_error(
                    error=e,
                    context={
                        "operation": "trading_loop",
                        "bot_mode": self.mode,
                        "strategy": getattr(self.current_active_strategy, '__class__.__name__', 'unknown') if self.current_active_strategy else 'none',
                        "symbols_count": len(symbols_to_fetch) if 'symbols_to_fetch' in locals() else 0
                    }
                )
                logger.error(f"Critical error in trading loop - ID: {error_response.error_id}")
                # Consider adding an error count or a circuit breaker here
                time.sleep(self.check_interval * 2)  # Wait longer after an error

        logger.info("Trading loop finished.")

    def _check_and_update_market_regime(self):
        """
        Checks market regime periodically and updates strategies via StrategyRouter.
        """
        if not self.ml_components or not self.strategy_router:
            return

        if datetime.now() - self.last_market_regime_check_time >= timedelta(seconds=self.regime_check_interval):
            logger.info("Performing periodic market regime check...")

            # Fetch fresh data for core symbols for regime detection
            current_market_data: Dict[str, pd.DataFrame] = {}
            core_symbols = self.settings.get('ml.regime_core_symbols', ["BTC/USDT"])
            timeframe = self.settings.get('timeframes.analysis', '1h')
            # Fetch enough data for the feature extraction window
            limit = self.settings.get('data.min_candles', 200)

            for symbol in core_symbols:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    if ohlcv:
                        current_market_data[symbol] = self.data_manager.convert_ohlcv_to_dataframe(ohlcv)
                except Exception as e:
                    error_response = self.error_handler.handle_api_error(
                        error=e,
                        context={
                            "operation": "regime_data_fetch",
                            "symbol": symbol,
                            "timeframe": timeframe
                        }
                    )
                    logger.warning(f"Failed to fetch data for {symbol} for regime detection - ID: {error_response.error_id}")

            if current_market_data:
                regime_info = self.ml_components.market_regime_detector.predict_regime(current_market_data)
                if regime_info["status"] == "success":
                    new_regime_label = regime_info["label"]
                    # Update strategy router with the new regime
                    self.strategy_router.update_market_regime(new_regime_label,
                                                              self.position_manager.get_total_capital())
                else:
                    logger.warning(f"Market regime detection failed: {regime_info['reason']}")
            else:
                logger.warning("No market data available for regime detection.")

            self.last_market_regime_check_time = datetime.now()

    @handle_errors(category=ErrorCategory.TRADING, max_retries=2, retry_delay=1.0)
    def _execute_signal(self, symbol: str, signal: Dict[str, Any], strategy: Strategy):
        """Executes a trade signal with comprehensive validation."""
        
        # Validate signal before execution
        validated_signal = self._validate_trading_signal(symbol, signal)
        
        if self.safety_manager and self.safety_manager.is_killswitch_active():
            logger.warning(f"Killswitch active. Not executing signal for {symbol}.")
            return

        trade_type = validated_signal['trade_type']
        amount = validated_signal['amount']
        price = validated_signal.get('price')  # Optional, for limit orders

        # Basic risk management check before placing order
        if not self.risk_manager.can_enter_position(symbol, amount, trade_type):
            logger.warning(f"Risk manager prevented trade for {symbol} ({trade_type} {amount}).")
            return

        order = None
        if trade_type == 'buy':
            order = self.order_manager.create_market_buy_order(symbol, amount)  # Or limit order if price provided
        elif trade_type == 'sell':
            order = self.order_manager.create_market_sell_order(symbol, amount)  # Or limit order if price provided

        if order:
            logger.info(f"Executed {trade_type.upper()} order for {amount} {symbol}. Order ID: {order['id']}")
            # Update position manager and performance tracker
            self.position_manager.update_position_from_order(order)
            self.performance_tracker.record_trade(order)
            
            # Update real-time risk calculator with new position
            try:
                fill_price = order.get('price', order.get('average', price))
                quantity = amount if trade_type == 'buy' else -amount  # Negative for sells
                side = 'long' if trade_type == 'buy' else 'short'
                
                self.risk_manager.update_realtime_position(symbol, quantity, fill_price, side)
                logger.debug(f"Real-time position updated: {symbol} {quantity} @ {fill_price}")
            except Exception as e:
                logger.error(f"Error updating real-time position for {symbol}: {e}")
        else:
            logger.error(f"Failed to execute {trade_type} order for {symbol}.")

    def _validate_trading_signal(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a trading signal before execution
        """
        try:
            # Validate symbol
            symbol_validator = validate_trading_symbol(symbol)
            
            # Validate required signal fields
            if not isinstance(signal, dict):
                raise ValidationTradingError(
                    "Trading signal must be a dictionary",
                    field="signal",
                    value=signal
                )
            
            required_fields = ['trade_type', 'amount']
            missing_fields = [field for field in required_fields if field not in signal]
            if missing_fields:
                raise ValidationTradingError(
                    f"Trading signal missing required fields: {missing_fields}",
                    field="signal",
                    value=signal
                )
            
            # Validate trade type
            valid_trade_types = ['buy', 'sell']
            trade_type = signal['trade_type'].lower()
            if trade_type not in valid_trade_types:
                raise ValidationTradingError(
                    f"Invalid trade type '{signal['trade_type']}'. Must be one of: {valid_trade_types}",
                    field="trade_type",
                    value=signal['trade_type']
                )
            
            # Validate amount
            amount = signal['amount']
            if not isinstance(amount, (int, float)) or amount <= 0:
                raise ValidationTradingError(
                    f"Invalid amount '{amount}'. Must be a positive number",
                    field="amount",
                    value=amount
                )
            
            # Validate amount using our amount validator
            quote_currency = symbol_validator.quote_currency
            amount_validator = validate_amount(amount, quote_currency)
            
            # Validate price if provided
            price = signal.get('price')
            if price is not None:
                if not isinstance(price, (int, float)) or price <= 0:
                    raise ValidationTradingError(
                        f"Invalid price '{price}'. Must be a positive number",
                        field="price",
                        value=price
                    )
                
                # Validate price using amount validator with quote currency
                validate_amount(price, quote_currency)
            
            # Create validated signal
            validated_signal = {
                'trade_type': trade_type,
                'amount': amount_validator.amount,
                'symbol': symbol_validator.symbol
            }
            
            if price is not None:
                validated_signal['price'] = price
            
            logger.debug(f"✅ Trading signal validation passed for {symbol}")
            return validated_signal
            
        except (ValidationError, PydanticValidationError) as e:
            error_response = self.error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                context={
                    "operation": "signal_validation",
                    "signal": signal
                }
            )
            logger.error(f"Trading signal validation failed for {symbol} - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Trading signal validation failed for {symbol}: {str(e)}", field="signal", value=signal)
        except Exception as e:
            error_response = self.error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "signal_validation",
                    "symbol": symbol,
                    "signal": signal
                }
            )
            logger.error(f"Critical error during signal validation for {symbol} - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Unexpected error during signal validation for {symbol}: {str(e)}", field="signal", value=signal)

    def _validate_backtest_parameters(self, symbol: str, timeframe: str, start_date_str: str, end_date_str: str):
        """
        Validates backtest parameters
        """
        try:
            # Validate symbol
            symbol_validator = validate_trading_symbol(symbol)
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            if timeframe not in valid_timeframes:
                raise ValidationTradingError(
                    f"Invalid timeframe '{timeframe}'. Must be one of: {valid_timeframes}",
                    field="timeframe",
                    value=timeframe
                )
            
            # Validate date format and logic
            try:
                start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            except ValueError as e:
                raise ValidationTradingError(
                    f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}",
                    field="date_format",
                    value={"start_date": start_date_str, "end_date": end_date_str}
                )
            
            # Validate date range
            if start_dt >= end_dt:
                raise ValidationTradingError(
                    f"Start date ({start_date_str}) must be before end date ({end_date_str})",
                    field="date_range",
                    value={"start_date": start_date_str, "end_date": end_date_str}
                )
            
            # Validate date range is not too far in the future
            if start_dt > datetime.now():
                raise ValidationTradingError(
                    f"Start date ({start_date_str}) cannot be in the future",
                    field="start_date",
                    value=start_date_str
                )
            
            # Validate date range is reasonable (not too long)
            max_backtest_days = 365 * 2  # 2 years max
            if (end_dt - start_dt).days > max_backtest_days:
                raise ValidationTradingError(
                    f"Backtest period too long ({(end_dt - start_dt).days} days). Maximum allowed: {max_backtest_days} days",
                    field="date_range",
                    value={"start_date": start_date_str, "end_date": end_date_str}
                )
            
            logger.debug(f"✅ Backtest parameters validation passed for {symbol}")
            
        except (ValidationError, PydanticValidationError) as e:
            error_response = self.error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                context={
                    "operation": "backtest_parameters_validation",
                    "timeframe": timeframe,
                    "start_date": start_date_str,
                    "end_date": end_date_str
                }
            )
            logger.error(f"Backtest parameters validation failed - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Backtest parameters validation failed: {str(e)}", field="backtest_parameters")
        except Exception as e:
            error_response = self.error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "backtest_parameters_validation",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date_str,
                    "end_date": end_date_str
                }
            )
            logger.error(f"Critical error during backtest parameters validation - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Unexpected error during backtest parameters validation: {str(e)}", field="backtest_parameters")

    @handle_errors(category=ErrorCategory.DATA, max_retries=1, retry_delay=2.0)
    def run_backtest(self, symbol: str, timeframe: str, start_date_str: str, end_date_str: str):
        """
        Runs a backtest for a given symbol, timeframe, and date range.
        This simplified version fetches all data first then iterates.
        For ML-enhanced backtesting, you'd use core/ml_enhanced_backtesting.py
        """
        
        # Validate backtest parameters
        self._validate_backtest_parameters(symbol, timeframe, start_date_str, end_date_str)
        
        logger.info(f"Starting backtest for {symbol} on {timeframe} from {start_date_str} to {end_date_str}")
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Fetch all historical data for the backtest period
        ohlcv_data = self.data_manager.get_historical_data(symbol, timeframe, start_date_str, end_date_str)
        if ohlcv_data.empty:
            logger.error(f"No historical data available for {symbol} from {start_date_str} to {end_date_str}.")
            return

        logger.info(f"Loaded {len(ohlcv_data)} candles for backtest.")

        # Reset performance tracker for backtest
        self.performance_tracker = PerformanceTracker(self.settings, is_backtest=True)
        initial_capital = self.settings.get('trading.initial_capital', 10000)
        self.performance_tracker.track_performance(initial_capital, ohlcv_data.index[0].isoformat())

        # Initialize strategy for backtesting, potentially with the router
        # If strategy_router is enabled, it should manage strategy selection during backtest
        if self.strategy_router and self.strategy_name == "auto_routed":
            logger.info("Backtesting with Strategy Router enabled.")
            # This requires a more sophisticated backtest loop that simulates time and calls the router
            # I will provide a basic loop here, but ideally you'd use `ml_enhanced_backtesting.py`
            # For simplicity, we will simulate periodic regime checks during the backtest
            last_simulated_regime_check = datetime.min

            # The backtest should operate candle by candle
            for i in range(self.settings.get('data.min_candles', 200), len(ohlcv_data)):
                current_df = ohlcv_data.iloc[:i].copy()
                current_candle = ohlcv_data.iloc[i - 1]  # Previous candle is the current data point

                # Simulate time for regime check
                current_time = current_candle.name  # Assuming index is datetime
                if current_time - last_simulated_regime_check >= timedelta(seconds=self.regime_check_interval):
                    logger.debug(f"Simulating market regime check at {current_time}")
                    # Prepare mock live data for regime detection (last N candles for core symbols)
                    simulated_market_data: Dict[str, pd.DataFrame] = {}
                    core_symbols = self.settings.get('ml.regime_core_symbols', ["BTC/USDT"])
                    limit = self.settings.get('data.min_candles', 200)  # Use same limit as live

                    for core_sym in core_symbols:
                        # For backtest, we need to fetch this from the full ohlcv_data
                        # This assumes core_symbols also exist in the backtested symbol's data
                        # In a real setup, you'd need multi-symbol OHLCV for backtesting
                        if core_sym == symbol:  # Use the current symbol's data slice
                            simulated_market_data[core_sym] = ohlcv_data.iloc[max(0, i - limit):i].copy()
                        else:
                            # Placeholder: in a real multi-symbol backtest, fetch data for other core_symbols
                            # from a pre-loaded multi-symbol dataset. For this example, we skip if not main symbol
                            pass

                    if simulated_market_data.get(symbol):  # Ensure the main symbol's data is available for regime check
                        regime_info = self.ml_components.market_regime_detector.predict_regime(simulated_market_data)
                        if regime_info["status"] == "success":
                            new_regime_label = regime_info["label"]
                            self.strategy_router.update_market_regime(new_regime_label,
                                                                      self.position_manager.get_total_capital())
                        else:
                            logger.warning(
                                f"Backtest regime detection failed at {current_time}: {regime_info['reason']}")

                    last_simulated_regime_check = current_time

                # Get current active strategies from router
                active_strategies = self.strategy_router.get_active_strategies()
                if not active_strategies:
                    logger.debug(f"No active strategies from router at {current_time}. Skipping trade logic.")
                    continue

                for strategy_name, strategy_instance in active_strategies.items():
                    # Only execute if the strategy is meant for the current symbol being backtested
                    # This implies strategies are single-symbol for this backtest logic
                    if hasattr(strategy_instance, 'trading_pair') and strategy_instance.trading_pair == symbol:
                        signal = strategy_instance.generate_signal(current_df, current_candle)
                        if signal:
                            # Simulate trade execution for backtest
                            simulated_order = self._simulate_backtest_trade(symbol, signal, current_candle['close'])
                            if simulated_order:
                                self.position_manager.update_position_from_order(simulated_order)
                                self.performance_tracker.record_trade(simulated_order)
        else:  # Fixed strategy backtest
            if not self.current_active_strategy:
                logger.error("No strategy available for backtest.")
                return

            for i in range(self.settings.get('data.min_candles', 200), len(ohlcv_data)):
                current_df = ohlcv_data.iloc[:i].copy()
                current_candle = ohlcv_data.iloc[i - 1]  # Previous candle is the current data point

                signal = self.current_active_strategy.generate_signal(current_df, current_candle)
                if signal:
                    simulated_order = self._simulate_backtest_trade(symbol, signal, current_candle['close'])
                    if simulated_order:
                        self.position_manager.update_position_from_order(simulated_order)
                        self.performance_tracker.record_trade(simulated_order)

        # Final performance calculation
        final_capital = self.position_manager.get_total_capital(current_prices={symbol: ohlcv_data.iloc[-1]['close']})
        self.performance_tracker.track_performance(final_capital, ohlcv_data.index[-1].isoformat())

        summary = self.performance_tracker.get_performance_summary()
        logger.info("Backtest completed. Performance Summary:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        self.performance_tracker.save_results(symbol, timeframe, start_date_str, end_date_str)

    def _simulate_backtest_trade(self, symbol: str, signal: Dict[str, Any], current_price: float) -> Optional[
        Dict[str, Any]]:
        """Simulates a trade for backtesting with validation."""
        
        # Validate inputs
        self._validate_simulate_trade_inputs(symbol, signal, current_price)
        
        trade_type = signal['trade_type']
        amount = signal['amount']

        # Simple simulation: assume market order fills at current_price
        filled_price = current_price
        cost_or_revenue = amount * filled_price

        # Apply a small simulated fee
        fee_rate = self.settings.get('exchange.maker_fee', 0.001)  # Use maker fee for simulation
        fee = cost_or_revenue * fee_rate

        order_id = f"simulated_{int(time.time() * 1000)}"
        timestamp = datetime.now().timestamp() * 1000  # Milliseconds

        order_info = {
            'id': order_id,
            'symbol': symbol,
            'type': 'market',
            'side': trade_type,
            'amount': amount,
            'price': filled_price,
            'cost': cost_or_revenue,
            'fee': {'cost': fee, 'currency': symbol.split('/')[1]},  # Assuming quote currency for fees
            'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
            'timestamp': int(timestamp),
            'status': 'closed'  # Always closed for simulated market order
        }
        logger.info(f"Simulated {trade_type.upper()} order for {amount} {symbol} at {filled_price:.4f}. Fee: {fee:.4f}")
        return order_info

    def _validate_simulate_trade_inputs(self, symbol: str, signal: Dict[str, Any], current_price: float):
        """
        Validates inputs for simulated trade
        """
        try:
            # Validate symbol
            validate_trading_symbol(symbol)
            
            # Validate signal (reuse existing validation)
            if not isinstance(signal, dict):
                raise ValidationTradingError(
                    "Signal must be a dictionary",
                    field="signal",
                    value=signal
                )
            
            # Validate current price
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                raise ValidationTradingError(
                    f"Invalid current price '{current_price}'. Must be a positive number",
                    field="current_price",
                    value=current_price
                )
            
            # Validate price is reasonable (not too extreme)
            if current_price > 1000000:  # $1M max per unit
                raise ValidationTradingError(
                    f"Current price too high: {current_price}. Maximum allowed: 1,000,000",
                    field="current_price",
                    value=current_price
                )
            
            logger.debug(f"✅ Simulate trade inputs validation passed for {symbol}")
            
        except (ValidationError, PydanticValidationError) as e:
            error_response = self.error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                context={
                    "operation": "simulate_trade_inputs_validation",
                    "signal": signal,
                    "current_price": current_price
                }
            )
            logger.error(f"Simulate trade inputs validation failed - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Simulate trade inputs validation failed: {str(e)}", field="simulate_trade_inputs")
        except Exception as e:
            error_response = self.error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "simulate_trade_inputs_validation",
                    "symbol": symbol,
                    "signal": signal,
                    "current_price": current_price
                }
            )
            logger.error(f"Critical error during simulate trade inputs validation - ID: {error_response.error_id}")
            raise ValidationTradingError(f"Unexpected error during simulate trade inputs validation: {str(e)}", field="simulate_trade_inputs")

    def print_status(self):
        """Logs current bot status (replaced print statements with logging)."""
        logger.info("\n--- Trading Bot Status ---")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Running: {self.running}")
        logger.info(
            f"Active Strategy: {self.current_active_strategy.__class__.__name__ if self.current_active_strategy else 'N/A'}")
        
        if self.strategy_router:
            try:
                current_regime = self.strategy_router.get_current_regime()
                logger.info(f"Current Market Regime: {current_regime}")
                
                active_router_strategies = self.strategy_router.get_active_strategies()
                if active_router_strategies:
                    logger.info("Router Active Strategies:")
                    for name, strat_instance in active_router_strategies.items():
                        trading_pair = getattr(strat_instance, 'trading_pair', 'unknown')
                        logger.info(f"  - {name} ({trading_pair})")
                else:
                    logger.info("Router: No strategies currently active.")
            except Exception as e:
                error_response = self.error_handler.handle_critical_error(
                    error=e,
                    context={"operation": "status_strategy_router_info"}
                )
                logger.error(f"Error getting strategy router info - ID: {error_response.error_id}")

        if self.safety_manager:
            try:
                killswitch_active = self.safety_manager.is_killswitch_active()
                current_drawdown = getattr(self.safety_manager, 'current_drawdown_percent', 0.0)
                logger.info(f"Killswitch Active: {killswitch_active}")
                logger.info(f"Current Drawdown: {current_drawdown:.2%}")
            except Exception as e:
                error_response = self.error_handler.handle_critical_error(
                    error=e,
                    context={"operation": "status_safety_manager_info"}
                )
                logger.error(f"Error getting safety manager info - ID: {error_response.error_id}")

        try:
            current_prices = self.exchange.get_current_prices()
            current_capital = self.position_manager.get_total_capital(current_prices)
            logger.info(f"Total Capital: {current_capital:.2f} USDT")
        except Exception as e:
            error_response = self.error_handler.handle_api_error(
                error=e,
                context={"operation": "status_capital_calculation"}
            )
            logger.error(f"Error calculating total capital - ID: {error_response.error_id}")
            logger.info("Total Capital: Unable to calculate")
        
        logger.info("--- Positions ---")
        try:
            positions = self.position_manager.get_all_positions()
            if positions:
                for symbol, pos in positions.items():
                    logger.info(f"  {symbol}: Amount={pos['amount']:.4f}, EntryPrice={pos['entry_price']:.4f}, "
                                f"CurrentPrice={pos['current_price']:.4f}, UnrealizedPNL={pos['unrealized_pnl']:.2f}")
            else:
                logger.info("  No open positions.")
        except Exception as e:
            error_response = self.error_handler.handle_critical_error(
                error=e,
                context={"operation": "status_positions_info"}
            )
            logger.error(f"Error getting positions info - ID: {error_response.error_id}")
            logger.info("  Unable to retrieve position information")
        
        logger.info("--------------------------")