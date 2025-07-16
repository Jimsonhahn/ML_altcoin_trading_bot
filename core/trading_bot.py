import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from config.settings import Settings
from core.exchange import ExchangeManager
from core.position import Position, PositionManager
from strategies.strategy_base import Strategy
from strategies import STRATEGIES
from utils.logger import setup_logger
from data_sources import DataManager

# New imports
from ml_components import initialize_ml, get_ml_components, MLComponents
from core.strategy_router import StrategyRouter
from core.safety_manager import SafetyManager


class TradingBot:
    """
    Main class for the Trading Bot.
    Coordinates all trading activities and is the central interface for bot execution.
    """

    def __init__(self, mode: str = "paper", strategy_name: str = "default",
                 settings: Optional[Settings] = None, data_manager: Optional[DataManager] = None):
        """
        Initializes the Trading Bot.
        """
        self.settings = settings or Settings()

        log_level_str = self.settings.get('logging.level', 'INFO')
        self.logger = setup_logger(name='trading_bot', level=log_level_str)

        self.mode = mode
        self.running = False
        self.check_interval = self.settings.get('timeframes.check_interval', 300)

        self.trading_pairs = self.settings.get('trading_pairs', ['BTC/USDT', 'ETH/USDT'])
        if isinstance(self.trading_pairs, str):
            self.trading_pairs = [self.trading_pairs]

        self.data_manager = data_manager or DataManager(self.settings)
        self.exchange = ExchangeManager(self.settings, mode)
        self.position_manager = PositionManager()

        self.data_cache = {}
        self.data_cache_lock = threading.Lock()
        self.max_workers = self.settings.get('system.max_workers', 4)

        self.on_trade_callbacks = []
        self.on_error_callbacks = []
        self.on_status_update_callbacks = []

        self.start_time = None
        self.start_balance = 0
        self.last_status = {}
        self.status_update_interval = 60

        self.api_error_count = 0
        self.last_api_error_time = None
        self.max_api_errors = self.settings.get('system.max_api_errors', 10)
        self.api_error_window = self.settings.get('system.api_error_window', 300)

        self.trading_thread = None
        self.monitor_thread = None

        self.ml_components: Optional[MLComponents] = None
        self.strategy_router: Optional[StrategyRouter] = None

        self.strategy_name = strategy_name
        if self.settings.get('strategy_router.enabled', False) or self.settings.get('auto_strategy', False):
            self.strategy_router = StrategyRouter(self.settings)
            self.strategy = None
        else:
            self.strategy = self._initialize_strategy(strategy_name)

        self.safety_manager: Optional[SafetyManager] = None
        self._peak_equity: float = 0.0

        self.logger.info(f"TradingBot initialized - Mode: {mode}, Strategy: {strategy_name}")
        self.logger.info(f"Trading pairs: {self.trading_pairs}")

    def set_safety_manager(self, manager: SafetyManager):
        """Sets the safety manager for the bot."""
        self.safety_manager = manager
        self.logger.info("Safety Manager set.")

    def _initialize_strategy(self, strategy_name: str) -> Strategy:
        """
        Initializes the trading strategy.
        """
        try:
            strategy_name_lower = strategy_name.lower()
            if strategy_name_lower == "default":
                strategy_name_lower = "momentum"

            strategy_class = STRATEGIES.get(strategy_name_lower)
            if strategy_class:
                self.logger.info(f"Successfully loaded strategy: {strategy_name} ({strategy_class.__name__})")
                return strategy_class(self.settings)
            else:
                available = list(STRATEGIES.keys())
                self.logger.warning(f"Strategy '{strategy_name}' not found. Available: {available}")
                return self._load_fallback_strategy()
        except Exception as e:
            self.logger.error(f"Error loading strategy {strategy_name}: {e}")
            return self._load_fallback_strategy()

    def _load_fallback_strategy(self):
        from strategies.momentum import MomentumStrategy
        strategy = MomentumStrategy(self.settings)
        self.logger.info("Loaded fallback momentum strategy")
        return strategy

    def connect(self) -> bool:
        """Establishes connection to the exchange."""
        self.logger.info("Connecting to exchange...")
        try:
            self.exchange.connect()
            self.logger.info("Successfully connected to exchange.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            self._notify_error("connection_error", str(e))
            return False

    def add_trade_callback(self, callback: Callable[[Position], None]) -> None:
        """Adds a callback function for trade events."""
        self.on_trade_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """Adds a callback function for error events."""
        self.on_error_callbacks.append(callback)

    def add_status_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Adds a callback function for status updates."""
        self.on_status_update_callbacks.append(callback)

    def _notify_error(self, error_type: str, error_message: str) -> None:
        """
        Notifies all registered callbacks about an error.
        Enhanced to integrate with SafetyManager.
        """
        if error_type.startswith("api_") or error_type == "connection_error":
            current_time = time.time()
            if (self.last_api_error_time is None or
                    current_time - self.last_api_error_time > self.api_error_window):
                self.api_error_count = 1
            else:
                self.api_error_count += 1
            self.last_api_error_time = current_time

            if self.api_error_count >= self.max_api_errors:
                self.logger.critical(
                    f"Too many API errors ({self.api_error_count}) within {self.api_error_window} seconds. "
                    f"Shutting down for safety."
                )
                self.stop()
                error_message = f"Bot stopped due to excessive API errors: {error_message}"
                error_type = "critical_api_failure"

        if self.safety_manager:
            self.logger.debug(f"Notifying SafetyManager about error: {error_type}")

        for callback in self.on_error_callbacks:
            try:
                callback(error_type, error_message)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")

    def _notify_trade(self, position: Position) -> None:
        """Notifies all registered callbacks about a trade event."""
        for callback in self.on_trade_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")

    def _notify_status_update(self, status: Dict[str, Any]) -> None:
        """Notifies all registered callbacks about a status update."""
        for callback in self.on_status_update_callbacks:
            try:
                callback(status)
            except Exception as e:
                self.logger.error(f"Error in status update callback: {e}")

    def _is_significant_status_change(self, old_status: Dict[str, Any], new_status: Dict[str, Any]) -> bool:
        """Checks if there's a significant change in status to warrant a notification."""
        if not old_status:
            return True

        if new_status.get('current_balance', 0) != old_status.get('current_balance', 0):
            return True
        if len(new_status.get('open_positions', [])) != len(old_status.get('open_positions', [])):
            return True
        if new_status.get('strategy_name') != old_status.get('strategy_name'):
            return True
        if new_status.get('killswitch_active') != old_status.get('killswitch_active'):
            return True
        if new_status.get('ml_status', {}).get('regime') != old_status.get('ml_status', {}).get('regime'):
            return True
        return False

    def _update_data_cache(self, symbol: str) -> pd.DataFrame:
        """
        Updates the data cache for a symbol.
        Leverages DataManager.
        """
        timeframe = self.settings.get('timeframes.analysis', '1h')
        min_candles = self.settings.get('data.min_candles', 50)

        try:
            required_candles = min_candles + 50
            if self.strategy and hasattr(self.strategy, 'lookback_period'):
                required_candles = max(required_candles, self.strategy.lookback_period + 50)
            if self.ml_components and self.ml_components.market_regime_detector:
                required_candles = max(required_candles,
                                       self.ml_components.market_regime_detector.min_data_points_required)

            df = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=required_candles
            )

            if df is None or df.empty:
                self.logger.warning(f"No data available for {symbol}")
                with self.data_cache_lock:
                    return self.data_cache.get(symbol, pd.DataFrame())

            if len(df) < min_candles:
                self.logger.warning(
                    f"Insufficient data for {symbol}, got {len(df)} candles, "
                    f"need at least {min_candles}"
                )

            with self.data_cache_lock:
                self.data_cache[symbol] = df

            return df
        except Exception as e:
            error_msg = f"Error updating data cache for {symbol}: {e}"
            self.logger.error(error_msg)
            self._notify_error("data_error", error_msg)
            with self.data_cache_lock:
                return self.data_cache.get(symbol, pd.DataFrame())

    def _update_all_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """Updates data for all trading pairs in parallel."""
        self.logger.info("Updating market data for all pairs...")
        updated_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {executor.submit(self._update_data_cache, symbol): symbol for symbol in
                                self.trading_pairs}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        updated_data[symbol] = df
                except Exception as exc:
                    self.logger.error(f'{symbol} data generation produced an exception: {exc}')
                    self._notify_error("data_fetch_error", f"Failed to update data for {symbol}: {exc}")
        self.logger.info(f"Data update complete for {len(updated_data)}/{len(self.trading_pairs)} pairs.")
        return updated_data

    def _check_pair(self, symbol: str):
        """
        Check single trading pair for signals.
        Enhanced to incorporate ML-based signals and regime adaptation.
        """
        if self.safety_manager and self.safety_manager.is_active():
            self.logger.info(f"Bot is in killswitch mode. Skipping trade processing for {symbol}.")
            return

        try:
            with self.data_cache_lock:
                df = self.data_cache.get(symbol)

            if df is None or df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return

            current_price = float(df['close'].iloc[-1])
            current_position = self.position_manager.get_position_by_symbol(symbol)

            market_regime_info = {}
            if self.ml_components:
                market_regime_info = self.ml_components.get_current_regime_info()

            if self.strategy_router:
                active_strategy = self.strategy_router.get_active_strategy()
                if active_strategy:
                    self.strategy = active_strategy
                    self.strategy_name = self.strategy_router.get_current_strategy_name()
                else:
                    self.logger.warning("Strategy router has no active strategy set. Using fallback.")
                    self.strategy = self._load_fallback_strategy()

            signal, signal_data = self.strategy.calculate_signal(symbol, df, current_price, current_position)

            if self.ml_components and market_regime_info and market_regime_info.get('status') == 'available':
                ml_signal, ml_signal_data = self._generate_ml_enhanced_signal(
                    signal, signal_data, symbol, df, market_regime_info.get('regime'), current_position
                )
                signal = ml_signal
                signal_data = ml_signal_data
                signal_data['ml_enhanced'] = True

            if hasattr(signal, 'value'):
                signal_str = signal.value
            else:
                signal_str = str(signal)

            if signal_str != 'HOLD' or signal_data.get('confidence', 0) > self.settings.get('risk.min_confidence', 0.6):
                self.logger.info(f"{symbol}: {signal_str} (confidence: {signal_data.get('confidence', 0):.2f})")

            min_confidence = self.settings.get('risk.min_confidence', 0.6)
            if signal_str in ['BUY', 'SELL'] and signal_data.get('confidence', 0) >= min_confidence:
                self._process_signal(symbol, current_price, signal_data, current_position)

        except Exception as e:
            self.logger.error(f"Error checking {symbol}: {e}")
            if self.settings.get('debug', False):
                import traceback
                traceback.print_exc()

    def _process_signal(self, symbol: str, current_price: float, signal_data: Dict[str, Any],
                        current_position: Optional[Position]) -> None:
        """
        Processes a trading signal.
        Updated to set `ml_enhanced` flag on position.
        """
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0.0)
        strategy_type = signal_data.get('strategy', 'unknown')

        if strategy_type == 'grid_trading':
            self._process_grid_signal(symbol, current_price, signal_data, current_position)
            return

        position_size = self.settings.get('risk.position_size', 0.05)
        stop_loss_pct = self.settings.get('risk.stop_loss', 0.03)
        take_profit_pct = self.settings.get('risk.take_profit', 0.06)
        max_positions = self.settings.get('risk.max_open_positions', 5)
        min_confidence = self.settings.get('risk.min_confidence', 0.6)

        if self.settings.get('risk.dynamic_position_sizing', False):
            if 'volatility' in signal_data:
                volatility = signal_data['volatility']
                position_size = position_size * (1.0 - min(volatility * 2.0, 0.8))

        stop_loss_pct = signal_data.get('stop_loss_pct', stop_loss_pct)
        take_profit_pct = signal_data.get('take_profit_pct', take_profit_pct)

        if signal == 'BUY' and not current_position:
            if confidence < min_confidence:
                self.logger.info(f"Buy signal for {symbol} ignored due to low confidence: {confidence:.2f}")
                return
            if len(self.position_manager.get_all_positions()) >= max_positions:
                self.logger.info(f"Buy signal for {symbol} ignored due to max positions limit")
                return

            try:
                balance_data = self.exchange.fetch_balance()
                usdt_balance = balance_data.get('USDT', {}).get('free', 0)
                if usdt_balance <= 0:
                    self.logger.warning(f"Insufficient balance for {symbol}")
                    return

                trade_value = usdt_balance * position_size
                trade_amount = trade_value / current_price

                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='buy',
                    amount=trade_amount
                )

                position = Position(
                    symbol=symbol,
                    entry_price=current_price,
                    amount=trade_amount,
                    side='buy',
                    order_id=order.get('id'),
                    entry_time=datetime.now()
                )
                position.ml_enhanced = signal_data.get('ml_enhanced', False)
                position.set_stop_loss(percentage=stop_loss_pct)
                position.set_take_profit(percentage=take_profit_pct)
                if signal_data.get('use_trailing_stop', False):
                    trailing_stop_pct = signal_data.get('trailing_stop_pct', stop_loss_pct)
                    activation_pct = signal_data.get('trailing_activation_pct', 0.02)
                    position.set_trailing_stop(trailing_stop_pct, activation_pct)

                self.position_manager.add_position(position)
                self._notify_trade(position)
                self.logger.info(
                    f"Opened position for {symbol}: {trade_amount} @ {current_price}. "
                    f"Stop-loss: {position.stop_loss}, Take-profit: {position.take_profit}. ML-enhanced: {position.ml_enhanced}"
                )

            except Exception as e:
                error_msg = f"Failed to place buy order for {symbol}: {e}"
                self.logger.error(error_msg)
                self._notify_error("order_error", error_msg)

        elif signal == 'SELL' and current_position and current_position.side == 'buy':
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side='sell',
                    amount=current_position.amount
                )

                closed_position = self.position_manager.close_position(
                    current_position.id,
                    current_price,
                    signal_data.get('reason', "sell_signal")
                )

                if closed_position:
                    closed_position.ml_enhanced = signal_data.get('ml_enhanced', False)
                    self._notify_trade(closed_position)
                    self.logger.info(
                        f"Closed position for {symbol} at {current_price}. "
                        f"P/L: {closed_position.profit_loss_percent:.2f}%. ML-enhanced: {closed_position.ml_enhanced}"
                    )

            except Exception as e:
                error_msg = f"Failed to place sell order for {symbol}: {e}"
                self.logger.error(error_msg)
                self._notify_error("order_error", error_msg)

        elif current_position:
            current_prices = {symbol: current_price}

            closed_positions = self.position_manager.update_positions(current_prices)

            for position in closed_positions:
                try:
                    order = self.exchange.create_order(
                        symbol=position.symbol,
                        order_type='market',
                        side='sell',
                        amount=position.amount
                    )
                    position.ml_enhanced = signal_data.get('ml_enhanced', False)
                    self._notify_trade(position)
                    self.logger.info(
                        f"Position closed automatically for {position.symbol} at {position.exit_price} "
                        f"({position.exit_reason}). P/L: {position.profit_loss_percent:.2f}%. ML-enhanced: {position.ml_enhanced}"
                    )
                except Exception as e:
                    error_msg = f"Failed to place sell order for automatically closed position {position.symbol}: {e}"
                    self.logger.error(error_msg)
                    self._notify_error("order_error", error_msg)

    def _process_grid_signal(self, symbol: str, current_price: float, signal_data: Dict[str, Any],
                             current_position: Optional[Position]) -> None:
        """Processes a grid trading signal."""
        grid_action = signal_data.get('grid_action')
        buy_price = signal_data.get('buy_price')
        sell_price = signal_data.get('sell_price')
        amount = signal_data.get('amount')
        grid_level = signal_data.get('grid_level')

        self.logger.debug(f"Grid signal for {symbol}: {grid_action} at level {grid_level}")

        if grid_action == 'buy_grid_level':
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side='buy',
                    amount=amount,
                    price=buy_price
                )
                self.logger.info(
                    f"Grid: Placed BUY order for {amount} {symbol} at {buy_price} (Level {grid_level}). Order ID: {order.get('id')}")
            except Exception as e:
                self.logger.error(f"Grid: Failed to place BUY order for {symbol} at {buy_price}: {e}")
                self._notify_error("grid_order_error", str(e))
        elif grid_action == 'sell_grid_level':
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side='sell',
                    amount=amount,
                    price=sell_price
                )
                self.logger.info(
                    f"Grid: Placed SELL order for {amount} {symbol} at {sell_price} (Level {grid_level}). Order ID: {order.get('id')}")
            except Exception as e:
                self.logger.error(f"Grid: Failed to place SELL order for {symbol} at {sell_price}: {e}")
                self._notify_error("grid_order_error", str(e))

    def _generate_ml_enhanced_signal(self, standard_signal: str, standard_signal_data: Dict[str, Any],
                                     symbol: str, symbol_data: pd.DataFrame,
                                     current_regime: Optional[int] = None,
                                     current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generates an ML-enhanced trading signal. This logic will be largely similar to 
        MLEnhancedBacktester's _generate_ml_enhanced_signal, but adapted for real-time.
        """
        if not self.ml_components:
            return standard_signal, standard_signal_data

        ml_signal_data = standard_signal_data.copy()
        ml_signal = standard_signal

        if current_regime is not None and self.ml_components.market_regime_detector.model_trained:
            regime_info = self.ml_components.market_regime_detector.get_regime_label(current_regime)

            if 'bullish' in regime_info.lower() or 'aufwärtstrend' in regime_info.lower():
                if standard_signal == "BUY":
                    ml_signal_data['confidence'] = min(1.0, standard_signal_data.get('confidence', 0.5) * 1.2)
                elif standard_signal == "SELL" and current_position:
                    current_price = symbol_data.iloc[-1]['close']
                    if current_position.stop_loss and current_price < current_position.stop_loss:
                        ml_signal = "SELL"
                    else:
                        ml_signal = "HOLD"
                        ml_signal_data['reason'] = 'bull_market_hold'
            elif 'bearish' in regime_info.lower() or 'abwärtstrend' in regime_info.lower():
                if standard_signal == "SELL":
                    ml_signal_data['confidence'] = min(1.0, standard_signal_data.get('confidence', 0.5) * 1.2)
                elif standard_signal == "BUY":
                    if standard_signal_data.get('confidence', 0) < 0.8:
                        ml_signal = "HOLD"
                        ml_signal_data['reason'] = 'bear_market_caution'
            elif 'sideways' in regime_info.lower() or 'niedrige-volatilität' in regime_info.lower():
                if self.strategy_router and self.strategy_router.get_current_strategy_name() == 'grid_trading':
                    ml_signal = standard_signal
                else:
                    ml_signal = "HOLD"
                    ml_signal_data['reason'] = 'sideways_market_hold'

        ml_signal_data['ml_enhanced'] = True
        return ml_signal, ml_signal_data

    def run(self) -> None:
        """
        Starts the Trading Bot in live or paper trading mode.
        Enhanced to include market regime detection and dynamic strategy routing.
        """
        if not self.connect():
            self.logger.error("Failed to connect to exchange. Bot stopped.")
            return

        self.running = True
        self.start_time = datetime.now()

        try:
            initial_balance_data = self.exchange.fetch_balance()
            self.start_balance = initial_balance_data.get('USDT', {}).get('total', 0)
            self._peak_equity = self.start_balance
        except Exception as e:
            self.logger.warning(f"Could not get initial balance: {e}")
            self.start_balance = 10000
            self._peak_equity = 10000

        self.logger.info(f"Starting trading bot with {len(self.trading_pairs)} pairs")
        self.logger.info(f"Initial balance: {self.start_balance}")

        if self.settings.get('ml.enabled', False):
            self.ml_components = initialize_ml(self.settings, self.data_manager.cache_dir,
                                               self.settings.get('ml.models_dir', 'data/ml_models'),
                                               self.settings.get('ml.output_dir', 'data/ml_analysis'))
            if not self.ml_components.load_models():
                self.logger.warning("ML models not loaded. Attempting to train new models.")
                self.ml_components.train_ml_models(self.trading_pairs)
                self.ml_components.save_models()

        if self.settings.get('strategy_router.enabled', False) and self.ml_components:
            self.strategy_router = StrategyRouter(self.settings)

        try:
            last_regime_check_time = datetime.min

            while self.running:
                if self.safety_manager:
                    current_balance_data = self.exchange.fetch_balance()
                    current_usdt_balance = current_balance_data.get('USDT', {}).get('total', 0)

                    unrealized_pnl = 0
                    for pos in self.position_manager.get_all_positions():
                        if pos.symbol in self.data_cache and not self.data_cache[pos.symbol].empty:
                            current_price = self.data_cache[pos.symbol]['close'].iloc[-1]
                            # Assuming long position for simplicity, adjust for short if applicable
                            unrealized_pnl += (current_price - pos.entry_price) * pos.amount
                        else:
                            self.logger.warning(f"No current data for {pos.symbol} to calculate unrealized PnL.")

                    current_total_equity = current_usdt_balance + unrealized_pnl

                    self._peak_equity = max(self._peak_equity, current_total_equity)

                    self.safety_manager.check_and_apply_killswitch(current_total_equity, self._peak_equity)

                    if self.safety_manager.is_active():
                        self.logger.info("Bot is in killswitch mode, pausing trading operations.")
                        time.sleep(self.check_interval)
                        continue

                self._update_all_data_parallel()

                if self.ml_components and self.strategy_router and (
                        datetime.now() - last_regime_check_time).total_seconds() > self.settings.get(
                        'ml.regime_check_interval', 3600):
                    self.logger.info("Updating ML components and re-routing strategy...")
                    update_status = self.ml_components.update_all_components(data_manager=self.data_manager,
                                                                             symbols=self.trading_pairs)
                    if update_status.get("regime_updated"):
                        market_regime_info = self.ml_components.get_current_regime_info()
                        self.strategy_router.route_strategy(market_regime_info, self.data_cache)
                        self.strategy_router.adjust_capital_allocation(market_regime_info)
                        self.strategy = self.strategy_router.get_active_strategy()
                        self.strategy_name = self.strategy_router.get_current_strategy_name()
                    last_regime_check_time = datetime.now()

                if self.settings.get('system.parallel_signal_processing', True) and len(self.trading_pairs) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        list(executor.map(self._check_pair, self.trading_pairs))
                else:
                    for symbol in self.trading_pairs:
                        if not self.running:
                            break
                        self._check_pair(symbol)
                        time.sleep(1)

                current_prices = {s: self.data_cache[s].iloc[-1]['close'] for s in self.data_cache if
                                  not self.data_cache[s].empty}
                self.position_manager.update_positions(current_prices)

                self.logger.debug(f"Sleeping for {self.check_interval} seconds")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            error_msg = f"Error in main loop: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self._notify_error("system_error", error_msg)
        finally:
            self.stop()

    def run_in_thread(self) -> threading.Thread:
        """Runs the bot in a separate thread."""
        self.logger.info("Starting bot in a new thread...")
        self.trading_thread = threading.Thread(target=self.run)
        self.trading_thread.start()
        self.monitor_thread = threading.Thread(target=self._status_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Bot thread started.")
        return self.trading_thread

    def _status_monitor(self) -> None:
        """Monitors bot status and sends updates periodically."""
        last_update_time = time.time() - self.status_update_interval
        self.last_status = self.get_status()

        while self.running:
            current_time = time.time()
            if current_time - last_update_time >= self.status_update_interval:
                new_status = self.get_status()
                if self._is_significant_status_change(self.last_status, new_status):
                    self.logger.info("\n📊 Bot Status Update:")
                    self.logger.info(json.dumps(new_status, indent=2, default=str))
                    self._notify_status_update(new_status)
                    self.last_status = new_status
                last_update_time = current_time
            time.sleep(1)

    def stop(self) -> None:
        """
        Stops the Trading Bot.
        Enhanced to close all positions and save ML models.
        """
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping bot...")

        try:
            current_prices = {s: self.data_cache[s].iloc[-1]['close'] for s in self.data_cache if
                              not self.data_cache[s].empty}
            closed_positions = self.position_manager.close_all_positions(current_prices, reason="bot_shutdown")
            for pos in closed_positions:
                self.logger.info(f"Closed on shutdown: {pos.symbol} P/L: {pos.profit_loss_percent:.2f}%")
        except Exception as e:
            self.logger.error(f"Error closing positions on shutdown: {e}")

        try:
            final_balance_data = self.exchange.fetch_balance()
            final_balance = final_balance_data.get('USDT', {}).get('total', self.start_balance)
        except Exception as e:
            self.logger.error(f"Error getting final balance: {e}")
            final_balance = self.start_balance

        if self.ml_components:
            self.ml_components.save_models()

        final_status = self.get_status()
        final_status['final_result']['ml_status'] = "Models saved" if self.ml_components else "ML not enabled"
        if self.safety_manager:
            final_status['final_result']['killswitch_active_on_stop'] = self.safety_manager.is_active()

        self._notify_status_update(final_status)

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of the bot.
        Enhanced to include ML and SafetyManager status.
        """
        open_positions_data = [pos.to_dict() for pos in self.position_manager.get_all_positions()]
        position_stats = self.position_manager.get_position_stats()

        status = {
            "running": self.running,
            "mode": self.mode,
            "strategy_name": self.strategy_name,
            "start_time": self.start_time.isoformat() if self.start_time else "N/A",
            "current_time": datetime.now().isoformat(),
            "initial_balance": self.start_balance,
            "current_balance": self.exchange.fetch_balance().get('USDT', {}).get('total', 0),
            "open_positions": open_positions_data,
            "total_pnl_percent": position_stats.get('total_pnl_percent', 0.0),
            "total_trades": position_stats.get('total_trades', 0),
            "win_rate": position_stats.get('win_rate', 0.0),
            "max_drawdown": position_stats.get('max_drawdown', 0.0),
            "api_error_count": self.api_error_count,
            "final_result": {}
        }

        if self.ml_components:
            regime_info = self.ml_components.get_current_regime_info()
            status['ml_status'] = {
                'regime': regime_info.get('label', 'unknown'),
                'is_model_trained': self.ml_components.market_regime_detector.model_trained,
                'last_update_success': not ('error' in regime_info)
            }
            if self.strategy_router:
                status['strategy_router_active_strategy'] = self.strategy_router.get_current_strategy_name()
                status['current_capital_allocation'] = self.settings.get('autopilot.capital_allocation', {})

        if self.safety_manager:
            status['killswitch_active'] = self.safety_manager.is_active()
            status['max_drawdown_limit'] = self.safety_manager.max_drawdown_percent
            status[
                'last_killswitch_activation'] = self.safety_manager.last_killswitch_activation_time.isoformat() if self.safety_manager and self.safety_manager.last_killswitch_activation_time else "N/A"

        return status

    def save_state(self, filepath: str) -> bool:
        """Saves the current state of the bot to a JSON file."""
        state_data = {
            "mode": self.mode,
            "strategy_name": self.strategy_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "start_balance": self.start_balance,
            "open_positions": [pos.to_dict() for pos in self.position_manager.get_all_positions()],
            "settings_config": self.settings.config,
            "api_error_count": self.api_error_count,
            "last_api_error_time": self.last_api_error_time.isoformat() if self.last_api_error_time else None,
            "_peak_equity": self._peak_equity,
            "killswitch_active": self.safety_manager.is_active() if self.safety_manager else False,
            "last_killswitch_activation_time": self.safety_manager.last_killswitch_activation_time.isoformat() if self.safety_manager and self.safety_manager.last_killswitch_activation_time else None
        }
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=4)
            self.logger.info(f"Bot state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save bot state to {filepath}: {e}")
            return False

    def generate_trading_report(self, days: int = 30, output_format: str = 'html') -> str:
        """Generates a comprehensive trading report."""
        from Analysis.performance_tracker import PerformanceTracker

        self.logger.info(f"Generating {output_format} trading report for last {days} days...")

        dummy_trades = [
            {"entry_time": datetime.now() - timedelta(days=5), "exit_time": datetime.now() - timedelta(days=4),
             "profit_loss_percent": 2.5, "symbol": "BTC/USDT"},
            {"entry_time": datetime.now() - timedelta(days=10), "exit_time": datetime.now() - timedelta(days=8),
             "profit_loss_percent": -1.2, "symbol": "ETH/USDT"},
        ]

        tracker = PerformanceTracker(self.settings)
        report_path = tracker.generate_report(dummy_trades, output_format=output_format)

        self.logger.info(f"Report generated at: {report_path}")
        return report_path

    def run_backtest(self) -> Dict[str, Any]:
        """
        Executes a backtest using the selected strategy and settings.
        This method is primarily called from main.py if mode is 'backtest'.
        """
        self.logger.info("Running backtest...")

        if self.settings.get('ml.enabled', False) and self.strategy_name.lower() == 'ml':
            from core.ml_enhanced_backtesting import MLEnhancedBacktester
            backtester = MLEnhancedBacktester(self.settings, self.strategy)
        else:
            from core.enhanced_backtesting import EnhancedBacktester
            backtester = EnhancedBacktester(self.settings, self.strategy)

        results = backtester.run(
            symbols=self.settings.get('trading_pairs'),
            source=self.settings.get('data.source', 'binance'),
            timeframe=self.settings.get('timeframes.analysis', '1h'),
            use_cache=self.settings.get('data.use_cache', True),
            start_date=self.settings.get('backtest.start_date'),
            end_date=self.settings.get('backtest.end_date')
        )
        self.logger.info("Backtest completed.")
        return results