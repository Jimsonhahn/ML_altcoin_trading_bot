"""
Test Suite for Core Trading Bot Components
==========================================

Tests for:
- TradingBot initialization and core functionality
- Critical trading functions
- Risk manager calculations
- Order management
- Position management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import components to test
from core.trading_bot import TradingBot
from core.exchange import ExchangeManager, ExchangeFactory
from core.risk_manager import RiskManager
from core.order_manager import OrderManager
from core.position import PositionManager
from config.settings import Settings
from utils.error_handler import ValidationTradingError


class TestTradingBot:
    """Test TradingBot initialization and core functionality"""
    
    def test_trading_bot_initialization(self, mock_settings, mock_data_manager,
                                      mock_exchange, mock_risk_manager,
                                      mock_order_manager, mock_position_manager,
                                      mock_performance_tracker):
        """Test TradingBot initialization with valid parameters"""
        with patch('core.trading_bot.Exchange', return_value=mock_exchange), \
             patch('core.trading_bot.OrderManager', return_value=mock_order_manager), \
             patch('core.trading_bot.PositionManager', return_value=mock_position_manager), \
             patch('core.trading_bot.RiskManager', return_value=mock_risk_manager), \
             patch('core.trading_bot.PerformanceTracker', return_value=mock_performance_tracker), \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            assert bot.mode == 'paper'
            assert bot.strategy_name == 'momentum'
            assert bot.settings == mock_settings
            assert bot.data_manager == mock_data_manager
            assert not bot.running
            assert bot.trade_thread is None
    
    def test_trading_bot_invalid_mode(self, mock_settings, mock_data_manager):
        """Test TradingBot initialization with invalid mode"""
        with pytest.raises(ValidationTradingError):
            TradingBot(
                mode='invalid_mode',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
    
    def test_trading_bot_invalid_strategy(self, mock_settings, mock_data_manager):
        """Test TradingBot initialization with invalid strategy"""
        with patch('strategies.STRATEGIES', {'momentum': Mock()}):
            with pytest.raises(ValidationTradingError):
                TradingBot(
                    mode='paper',
                    strategy_name='invalid_strategy',
                    settings=mock_settings,
                    data_manager=mock_data_manager
                )
    
    def test_trading_bot_start_stop(self, mock_settings, mock_data_manager,
                                   mock_exchange, mock_strategy):
        """Test starting and stopping the trading bot"""
        with patch('core.trading_bot.Exchange', return_value=mock_exchange), \
             patch('core.trading_bot.OrderManager'), \
             patch('core.trading_bot.PositionManager'), \
             patch('core.trading_bot.RiskManager'), \
             patch('core.trading_bot.PerformanceTracker'), \
             patch('strategies.STRATEGIES', {'momentum': Mock(return_value=mock_strategy)}):
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Test start
            bot.start()
            assert bot.running
            assert bot.trade_thread is not None
            
            # Test stop
            bot.stop()
            assert not bot.running
    
    def test_signal_validation(self, mock_settings, mock_data_manager,
                              trading_signal):
        """Test trading signal validation"""
        with patch('core.trading_bot.Exchange'), \
             patch('core.trading_bot.OrderManager'), \
             patch('core.trading_bot.PositionManager'), \
             patch('core.trading_bot.RiskManager'), \
             patch('core.trading_bot.PerformanceTracker'), \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Test valid signal
            validated = bot._validate_trading_signal('BTC/USDT', trading_signal)
            assert validated['trade_type'] == 'buy'
            assert validated['amount'] == 0.001
            assert validated['symbol'] == 'BTC/USDT'
            
            # Test invalid signal
            invalid_signal = {'trade_type': 'invalid', 'amount': -1}
            with pytest.raises(ValidationTradingError):
                bot._validate_trading_signal('BTC/USDT', invalid_signal)
    
    def test_backtest_parameter_validation(self, mock_settings, mock_data_manager):
        """Test backtest parameter validation"""
        with patch('core.trading_bot.Exchange'), \
             patch('core.trading_bot.OrderManager'), \
             patch('core.trading_bot.PositionManager'), \
             patch('core.trading_bot.RiskManager'), \
             patch('core.trading_bot.PerformanceTracker'), \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Test valid parameters
            bot._validate_backtest_parameters(
                'BTC/USDT', '1h', '2024-01-01', '2024-01-10'
            )
            
            # Test invalid date range
            with pytest.raises(ValidationTradingError):
                bot._validate_backtest_parameters(
                    'BTC/USDT', '1h', '2024-01-10', '2024-01-01'
                )
            
            # Test invalid timeframe
            with pytest.raises(ValidationTradingError):
                bot._validate_backtest_parameters(
                    'BTC/USDT', 'invalid', '2024-01-01', '2024-01-10'
                )
    
    def test_mock_trade_simulation(self, mock_settings, mock_data_manager,
                                  trading_signal):
        """Test simulated trade execution for backtesting"""
        with patch('core.trading_bot.Exchange'), \
             patch('core.trading_bot.OrderManager'), \
             patch('core.trading_bot.PositionManager'), \
             patch('core.trading_bot.RiskManager'), \
             patch('core.trading_bot.PerformanceTracker'), \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Test simulated trade
            order = bot._simulate_backtest_trade('BTC/USDT', trading_signal, 50000.0)
            
            assert order is not None
            assert order['symbol'] == 'BTC/USDT'
            assert order['side'] == 'buy'
            assert order['amount'] == 0.001
            assert order['status'] == 'closed'
            assert 'id' in order
            assert 'fee' in order


class TestExchangeManager:
    """Test ExchangeManager functionality"""
    
    def test_exchange_initialization(self):
        """Test ExchangeManager initialization"""
        exchange = ExchangeManager('binance', 'paper')
        
        assert exchange.exchange_name == 'binance'
        assert exchange.mode == 'paper'
        assert not exchange.connected
        assert exchange.exchange is None
    
    def test_paper_trading_connection(self):
        """Test paper trading connection"""
        with patch('ccxt.binance') as mock_binance:
            mock_binance.return_value = Mock()
            
            exchange = ExchangeManager('binance', 'paper')
            result = exchange.connect()
            
            assert result is True
            assert exchange.connected is True
    
    def test_mock_ticker_data(self):
        """Test mock ticker data generation"""
        exchange = ExchangeManager('binance', 'paper')
        ticker = exchange._get_mock_ticker('BTC/USDT')
        
        assert ticker['symbol'] == 'BTC/USDT'
        assert ticker['last'] > 0
        assert ticker['bid'] < ticker['ask']
        assert ticker['volume'] > 0
        assert 'timestamp' in ticker
    
    def test_mock_ohlcv_data(self):
        """Test mock OHLCV data generation"""
        exchange = ExchangeManager('binance', 'paper')
        df = exchange._get_mock_ohlcv('BTC/USDT', '1h', 100)
        
        assert len(df) == 100
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_paper_order_creation(self):
        """Test order creation in paper trading mode"""
        exchange = ExchangeManager('binance', 'paper')
        exchange.connected = True
        
        with patch.object(exchange, 'fetch_ticker', return_value={'last': 50000}):
            order = exchange.create_order('BTC/USDT', 'market', 'buy', 0.001)
            
            assert order['symbol'] == 'BTC/USDT'
            assert order['type'] == 'market'
            assert order['side'] == 'buy'
            assert order['amount'] == 0.001
            assert order['status'] == 'closed'
            assert 'id' in order
    
    def test_exchange_factory(self):
        """Test ExchangeFactory functionality"""
        with patch('core.exchange.ExchangeManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            mock_instance.connect.return_value = True
            
            exchange = ExchangeFactory.create_exchange('binance', 'paper')
            
            mock_manager.assert_called_once_with('binance', 'paper')
            mock_instance.connect.assert_called_once()


class TestRiskManager:
    """Test RiskManager calculations and functionality"""
    
    @pytest.fixture
    def risk_manager(self, mock_settings, mock_position_manager):
        """Create RiskManager instance for testing"""
        return RiskManager(mock_settings, mock_position_manager)
    
    def test_risk_manager_initialization(self, risk_manager, mock_settings):
        """Test RiskManager initialization"""
        assert risk_manager.settings == mock_settings
        assert risk_manager.max_drawdown == 0.20
        assert risk_manager.max_position_size == 1000
        assert risk_manager.risk_per_trade == 0.02
    
    def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        with patch.object(risk_manager.position_manager, 'get_total_capital', return_value=10000):
            size = risk_manager.calculate_position_size('BTC/USDT', 50000, 'buy')
            
            # Should be based on risk_per_trade (2% of 10000 = 200 USD)
            # At 50000 price: 200/50000 = 0.004 BTC
            assert size > 0
            assert size <= 0.004  # Should not exceed calculated size
    
    def test_stop_loss_calculation(self, risk_manager):
        """Test stop loss calculation"""
        entry_price = 50000
        stop_loss = risk_manager.calculate_stop_loss(entry_price, 'buy')
        
        # For buy order, stop loss should be below entry price
        assert stop_loss < entry_price
        # Should be 2% below (default stop_loss_percentage)
        expected = entry_price * (1 - 0.02)
        assert abs(stop_loss - expected) < 1  # Allow small rounding differences
    
    def test_take_profit_calculation(self, risk_manager):
        """Test take profit calculation"""
        entry_price = 50000
        take_profit = risk_manager.calculate_take_profit(entry_price, 'buy')
        
        # For buy order, take profit should be above entry price
        assert take_profit > entry_price
        # Should be 5% above (default take_profit_percentage)
        expected = entry_price * (1 + 0.05)
        assert abs(take_profit - expected) < 1
    
    def test_risk_limit_validation(self, risk_manager):
        """Test risk limit validation"""
        with patch.object(risk_manager.position_manager, 'get_total_capital', return_value=10000), \
             patch.object(risk_manager.position_manager, 'get_position_count', return_value=3):
            
            # Should allow position within limits
            assert risk_manager.can_enter_position('BTC/USDT', 0.001, 'buy') is True
            
            # Test with too many positions
            with patch.object(risk_manager.position_manager, 'get_position_count', return_value=6):
                assert risk_manager.can_enter_position('BTC/USDT', 0.001, 'buy') is False
    
    def test_drawdown_calculation(self, risk_manager):
        """Test drawdown calculation"""
        with patch.object(risk_manager.position_manager, 'get_peak_capital', return_value=12000), \
             patch.object(risk_manager.position_manager, 'get_total_capital', return_value=10000):
            
            drawdown = risk_manager.calculate_current_drawdown()
            expected = (12000 - 10000) / 12000  # 16.67%
            assert abs(drawdown - expected) < 0.001
    
    def test_risk_parameters_validation(self, risk_manager):
        """Test validation of risk parameters"""
        # Test valid parameters
        assert risk_manager.validate_risk_parameters(
            amount=0.001,
            price=50000,
            stop_loss=49000,
            take_profit=52500
        ) is True
        
        # Test invalid stop loss (above entry for buy)
        assert risk_manager.validate_risk_parameters(
            amount=0.001,
            price=50000,
            stop_loss=51000,  # Above entry price for buy
            take_profit=52500,
            side='buy'
        ) is False


class TestOrderManager:
    """Test OrderManager functionality"""
    
    @pytest.fixture
    def order_manager(self, mock_exchange, mock_settings):
        """Create OrderManager instance for testing"""
        return OrderManager(mock_exchange, mock_settings)
    
    def test_order_manager_initialization(self, order_manager, mock_exchange, mock_settings):
        """Test OrderManager initialization"""
        assert order_manager.exchange == mock_exchange
        assert order_manager.settings == mock_settings
        assert isinstance(order_manager.active_orders, dict)
    
    def test_market_buy_order(self, order_manager):
        """Test market buy order creation"""
        order = order_manager.create_market_buy_order('BTC/USDT', 0.001)
        
        assert order is not None
        assert order['symbol'] == 'BTC/USDT'
        assert order['side'] == 'buy'
        assert order['type'] == 'market'
        assert order['amount'] == 0.001
    
    def test_market_sell_order(self, order_manager):
        """Test market sell order creation"""
        order = order_manager.create_market_sell_order('BTC/USDT', 0.001)
        
        assert order is not None
        assert order['symbol'] == 'BTC/USDT'
        assert order['side'] == 'sell'
        assert order['type'] == 'market'
        assert order['amount'] == 0.001
    
    def test_limit_order_creation(self, order_manager):
        """Test limit order creation"""
        order = order_manager.create_limit_order('BTC/USDT', 'buy', 0.001, 49000)
        
        assert order is not None
        assert order['symbol'] == 'BTC/USDT'
        assert order['side'] == 'buy'
        assert order['type'] == 'limit'
        assert order['amount'] == 0.001
        assert order['price'] == 49000
    
    def test_order_cancellation(self, order_manager):
        """Test order cancellation"""
        # First create an order
        order = order_manager.create_limit_order('BTC/USDT', 'buy', 0.001, 49000)
        order_id = order['id']
        
        # Then cancel it
        result = order_manager.cancel_order(order_id, 'BTC/USDT')
        assert result is True
    
    def test_order_status_tracking(self, order_manager):
        """Test order status tracking"""
        order = order_manager.create_limit_order('BTC/USDT', 'buy', 0.001, 49000)
        order_id = order['id']
        
        # Check order status
        status = order_manager.get_order_status(order_id, 'BTC/USDT')
        assert status is not None
        assert 'status' in status
    
    def test_active_orders_management(self, order_manager):
        """Test active orders management"""
        # Create multiple orders
        order1 = order_manager.create_limit_order('BTC/USDT', 'buy', 0.001, 49000)
        order2 = order_manager.create_limit_order('ETH/USDT', 'sell', 0.1, 3100)
        
        # Check active orders
        active = order_manager.get_active_orders()
        assert len(active) >= 2
        
        # Cancel an order and check it's removed from active
        order_manager.cancel_order(order1['id'], 'BTC/USDT')
        active_after = order_manager.get_active_orders()
        assert order1['id'] not in [o['id'] for o in active_after]


class TestPositionManager:
    """Test PositionManager functionality"""
    
    @pytest.fixture
    def position_manager(self, mock_settings):
        """Create PositionManager instance for testing"""
        return PositionManager(mock_settings)
    
    def test_position_manager_initialization(self, position_manager):
        """Test PositionManager initialization"""
        assert isinstance(position_manager.positions, dict)
        assert position_manager.initial_capital == 10000
    
    def test_position_opening(self, position_manager):
        """Test opening a new position"""
        order = {
            'id': 'test_order_1',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000,
            'cost': 50,
            'fee': {'cost': 0.05, 'currency': 'USDT'}
        }
        
        position_manager.update_position_from_order(order)
        
        positions = position_manager.get_all_positions()
        assert 'BTC/USDT' in positions
        
        btc_position = positions['BTC/USDT']
        assert btc_position['amount'] == 0.001
        assert btc_position['entry_price'] == 50000
    
    def test_position_closing(self, position_manager):
        """Test closing a position"""
        # First open a position
        buy_order = {
            'id': 'test_buy',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000,
            'cost': 50,
            'fee': {'cost': 0.05, 'currency': 'USDT'}
        }
        position_manager.update_position_from_order(buy_order)
        
        # Then close it
        sell_order = {
            'id': 'test_sell',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'amount': 0.001,
            'price': 51000,
            'cost': 51,
            'fee': {'cost': 0.051, 'currency': 'USDT'}
        }
        position_manager.update_position_from_order(sell_order)
        
        positions = position_manager.get_all_positions()
        # Position should be closed (amount = 0) or removed
        if 'BTC/USDT' in positions:
            assert positions['BTC/USDT']['amount'] == 0
    
    def test_portfolio_value_calculation(self, position_manager):
        """Test portfolio value calculation"""
        # Open some positions
        orders = [
            {
                'id': 'order_1',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.001,
                'price': 50000,
                'cost': 50,
                'fee': {'cost': 0.05, 'currency': 'USDT'}
            },
            {
                'id': 'order_2',
                'symbol': 'ETH/USDT',
                'side': 'buy',
                'amount': 0.1,
                'price': 3000,
                'cost': 300,
                'fee': {'cost': 0.3, 'currency': 'USDT'}
            }
        ]
        
        for order in orders:
            position_manager.update_position_from_order(order)
        
        # Calculate portfolio value with current prices
        current_prices = {
            'BTC/USDT': 52000,  # 4% gain
            'ETH/USDT': 3100    # 3.33% gain
        }
        
        position_manager.update_portfolio_value(current_prices)
        total_value = position_manager.get_total_capital(current_prices)
        
        # Should be initial capital minus costs plus current position values
        expected_btc_value = 0.001 * 52000  # 52
        expected_eth_value = 0.1 * 3100     # 310
        expected_cash = 10000 - 50 - 300 - 0.05 - 0.3  # Initial - costs - fees
        expected_total = expected_cash + expected_btc_value + expected_eth_value
        
        assert abs(total_value - expected_total) < 1  # Allow small rounding differences
    
    def test_pnl_calculation(self, position_manager):
        """Test PnL calculation"""
        order = {
            'id': 'test_order',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'price': 50000,
            'cost': 50,
            'fee': {'cost': 0.05, 'currency': 'USDT'}
        }
        
        position_manager.update_position_from_order(order)
        
        # Test unrealized PnL
        current_prices = {'BTC/USDT': 55000}  # 10% gain
        position_manager.update_portfolio_value(current_prices)
        
        positions = position_manager.get_all_positions()
        btc_position = positions['BTC/USDT']
        
        # Unrealized PnL should be (55000 - 50000) * 0.001 = 5 USDT
        assert abs(btc_position['unrealized_pnl'] - 5.0) < 0.01
    
    def test_position_count(self, position_manager):
        """Test position counting"""
        assert position_manager.get_position_count() == 0
        
        # Add some positions
        orders = [
            {
                'id': 'order_1',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.001,
                'price': 50000,
                'cost': 50,
                'fee': {'cost': 0.05, 'currency': 'USDT'}
            },
            {
                'id': 'order_2',
                'symbol': 'ETH/USDT',
                'side': 'buy',
                'amount': 0.1,
                'price': 3000,
                'cost': 300,
                'fee': {'cost': 0.3, 'currency': 'USDT'}
            }
        ]
        
        for order in orders:
            position_manager.update_position_from_order(order)
        
        assert position_manager.get_position_count() == 2


class TestSettings:
    """Test Settings configuration management"""
    
    def test_settings_initialization(self):
        """Test Settings object initialization"""
        settings = Settings('default')
        
        assert settings.get('exchange.name') == 'binance'
        assert settings.get('trading.initial_capital') == 10000
        assert settings.get('risk.max_drawdown') == 0.20
    
    def test_settings_get_with_default(self):
        """Test Settings.get() with default values"""
        settings = Settings('default')
        
        # Existing key
        assert settings.get('exchange.name') == 'binance'
        
        # Non-existing key with default
        assert settings.get('non.existing.key', 'default_value') == 'default_value'
        
        # Non-existing key without default
        assert settings.get('non.existing.key') is None
    
    def test_settings_set(self):
        """Test Settings.set() functionality"""
        settings = Settings('default')
        
        settings.set('test.key', 'test_value')
        assert settings.get('test.key') == 'test_value'
        
        # Test nested key setting
        settings.set('nested.deep.key', 42)
        assert settings.get('nested.deep.key') == 42
    
    def test_settings_deep_update(self):
        """Test deep dictionary update functionality"""
        settings = Settings('default')
        
        # Test that nested dictionaries are properly merged
        original_exchange = settings.get('exchange')
        settings.set('exchange.new_param', 'new_value')
        
        # Original exchange config should still be there
        assert settings.get('exchange.name') == original_exchange['name']
        # New parameter should be added
        assert settings.get('exchange.new_param') == 'new_value'


class TestIntegrationScenarios:
    """Integration tests for complete trading scenarios"""
    
    def test_complete_trading_cycle(self, mock_settings, mock_data_manager,
                                   sample_ohlcv_data, trading_signal):
        """Test a complete trading cycle from signal to execution"""
        with patch('core.trading_bot.Exchange') as mock_exchange_class, \
             patch('core.trading_bot.OrderManager') as mock_order_class, \
             patch('core.trading_bot.PositionManager') as mock_position_class, \
             patch('core.trading_bot.RiskManager') as mock_risk_class, \
             patch('core.trading_bot.PerformanceTracker') as mock_perf_class, \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            # Setup mocks
            mock_exchange = Mock()
            mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_data.reset_index().values.tolist()
            mock_exchange.get_current_prices.return_value = {'BTC/USDT': 50000}
            mock_exchange_class.return_value = mock_exchange
            
            mock_order_manager = Mock()
            mock_order_manager.create_market_buy_order.return_value = {
                'id': 'test_order_123',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.001,
                'price': 50000,
                'status': 'closed'
            }
            mock_order_class.return_value = mock_order_manager
            
            mock_position_manager = Mock()
            mock_position_manager.get_total_capital.return_value = 10000
            mock_position_class.return_value = mock_position_manager
            
            mock_risk_manager = Mock()
            mock_risk_manager.can_enter_position.return_value = True
            mock_risk_class.return_value = mock_risk_manager
            
            mock_perf_tracker = Mock()
            mock_perf_class.return_value = mock_perf_tracker
            
            # Create trading bot
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Simulate signal execution
            bot._execute_signal('BTC/USDT', trading_signal, Mock())
            
            # Verify order was created
            mock_order_manager.create_market_buy_order.assert_called_once()
            
            # Verify position was updated
            mock_position_manager.update_position_from_order.assert_called_once()
            
            # Verify performance was tracked
            mock_perf_tracker.record_trade.assert_called_once()
    
    def test_risk_management_prevents_trade(self, mock_settings, mock_data_manager,
                                          trading_signal):
        """Test that risk management prevents unsafe trades"""
        with patch('core.trading_bot.Exchange'), \
             patch('core.trading_bot.OrderManager') as mock_order_class, \
             patch('core.trading_bot.PositionManager'), \
             patch('core.trading_bot.RiskManager') as mock_risk_class, \
             patch('core.trading_bot.PerformanceTracker'), \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            # Setup risk manager to reject trades
            mock_risk_manager = Mock()
            mock_risk_manager.can_enter_position.return_value = False
            mock_risk_class.return_value = mock_risk_manager
            
            mock_order_manager = Mock()
            mock_order_class.return_value = mock_order_manager
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Attempt to execute signal
            bot._execute_signal('BTC/USDT', trading_signal, Mock())
            
            # Verify no order was created
            mock_order_manager.create_market_buy_order.assert_not_called()
            mock_order_manager.create_market_sell_order.assert_not_called()
    
    @pytest.mark.slow
    def test_backtest_execution(self, mock_settings, mock_data_manager,
                               sample_ohlcv_data):
        """Test backtest execution with sample data"""
        with patch('core.trading_bot.Exchange'), \
             patch('core.trading_bot.OrderManager'), \
             patch('core.trading_bot.PositionManager') as mock_position_class, \
             patch('core.trading_bot.RiskManager'), \
             patch('core.trading_bot.PerformanceTracker') as mock_perf_class, \
             patch('strategies.STRATEGIES', {'momentum': Mock()}):
            
            # Setup data manager to return sample data
            mock_data_manager.get_historical_data.return_value = sample_ohlcv_data
            
            mock_position_manager = Mock()
            mock_position_manager.get_total_capital.return_value = 10000
            mock_position_class.return_value = mock_position_manager
            
            mock_perf_tracker = Mock()
            mock_perf_tracker.get_performance_summary.return_value = {
                'total_return': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.03
            }
            mock_perf_class.return_value = mock_perf_tracker
            
            bot = TradingBot(
                mode='paper',
                strategy_name='momentum',
                settings=mock_settings,
                data_manager=mock_data_manager
            )
            
            # Run backtest
            bot.run_backtest('BTC/USDT', '1h', '2024-01-01', '2024-01-10')
            
            # Verify backtest completed
            mock_perf_tracker.get_performance_summary.assert_called_once()
            mock_perf_tracker.save_results.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])