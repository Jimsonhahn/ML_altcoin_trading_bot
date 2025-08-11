"""
Pytest Configuration and Fixtures
==================================

Provides common fixtures for testing the trading bot components.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from core.trading_bot import TradingBot
from core.exchange import ExchangeManager
from data_sources.data_manager import DataManager
from utils.secret_manager import SecretManager
from utils.error_handler import SecureErrorHandler


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_settings():
    """Mock Settings object with test configuration"""
    settings = Mock(spec=Settings)
    settings.get.side_effect = lambda key, default=None: {
        'exchange.name': 'binance',
        'exchange.testnet': True,
        'exchange.rate_limit': 1200,
        'trading.initial_capital': 10000,
        'trading.max_positions': 5,
        'trading.position_sizing': 'fixed',
        'trading.risk_per_trade': 0.02,
        'trading.default_strategy': 'momentum',
        'trading.max_position_size': 1000.0,
        'timeframes.analysis': '1h',
        'timeframes.check_interval': 300,
        'timeframes.secondary': '4h',
        'data.source': 'exchange',
        'data.min_candles': 200,
        'data.cache_dir': 'data/market_data',
        'ml.enabled': True,
        'ml.regime_core_symbols': ['BTC/USDT', 'ETH/USDT'],
        'ml.regime_check_interval': 1800,
        'ml.min_data_points_for_ml': 200,
        'risk.max_drawdown': 0.20,
        'risk.stop_loss_percentage': 0.02,
        'risk.take_profit_percentage': 0.05,
        'risk.risk_per_trade': 0.02,
        'risk.max_position_size': 1000,
        'risk.max_positions': 5,
        'strategy_configs.momentum.trading_pair': 'BTC/USDT',
        'strategy_configs.momentum.rsi_oversold': 30,
        'strategy_configs.momentum.rsi_overbought': 70,
        'strategy_configs.momentum.sma_short_period': 5,
        'strategy_configs.momentum.sma_long_period': 20,
        'logging.level': 'INFO',
        'logging.format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    }.get(key, default)
    return settings


@pytest.fixture
def mock_exchange():
    """Mock ExchangeManager for testing"""
    exchange = Mock(spec=ExchangeManager)
    exchange.connected = True
    exchange.exchange_name = 'binance'
    exchange.mode = 'paper'
    
    # Mock methods
    exchange.connect.return_value = True
    exchange.disconnect.return_value = None
    exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 50000.0,
        'bid': 49995.0,
        'ask': 50005.0,
        'high': 51000.0,
        'low': 49000.0,
        'volume': 1000000,
        'timestamp': int(datetime.now().timestamp() * 1000)
    }
    
    # Mock OHLCV data
    def mock_fetch_ohlcv(symbol, timeframe='1h', limit=100):
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='h')
        base_price = 50000.0
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.01, limit)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.001, 0.01)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            close = np.random.uniform(low, high)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(100000, 1000000)
            
            data.append([
                int(date.timestamp() * 1000),  # timestamp
                open_price,  # open
                high,        # high
                low,         # low
                close,       # close
                volume       # volume
            ])
        
        return data
    
    exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv
    
    exchange.fetch_balance.return_value = {
        'USDT': {'free': 10000, 'used': 0, 'total': 10000},
        'BTC': {'free': 0, 'used': 0, 'total': 0},
        'ETH': {'free': 0, 'used': 0, 'total': 0}
    }
    
    exchange.create_order.return_value = {
        'id': 'test_order_123',
        'symbol': 'BTC/USDT',
        'type': 'market',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0,
        'status': 'closed',
        'timestamp': int(datetime.now().timestamp() * 1000)
    }
    
    exchange.cancel_order.return_value = True
    exchange.fetch_order.return_value = {
        'id': 'test_order_123',
        'symbol': 'BTC/USDT',
        'status': 'closed',
        'filled': 1.0,
        'remaining': 0.0
    }
    
    exchange.get_current_prices.return_value = {
        'BTC/USDT': 50000.0,
        'ETH/USDT': 3000.0
    }
    
    return exchange


@pytest.fixture
def mock_data_manager():
    """Mock DataManager for testing"""
    data_manager = Mock(spec=DataManager)
    
    def mock_get_historical_data(symbol, timeframe, start_date, end_date):
        # Generate mock historical data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='h')
        
        base_price = 50000.0 if 'BTC' in symbol else 3000.0
        returns = np.random.normal(0, 0.01, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, len(dates)),
            'high': prices * np.random.uniform(1.0, 1.02, len(dates)),
            'low': prices * np.random.uniform(0.98, 1.0, len(dates)),
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, len(dates))
        }, index=dates)
        
        return df
    
    data_manager.get_historical_data.side_effect = mock_get_historical_data
    
    def mock_convert_ohlcv_to_dataframe(ohlcv_data):
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    data_manager.convert_ohlcv_to_dataframe.side_effect = mock_convert_ohlcv_to_dataframe
    
    return data_manager


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='h')
    base_price = 50000.0
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.01, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.99, 1.01, len(dates)),
        'high': prices * np.random.uniform(1.0, 1.02, len(dates)),
        'low': prices * np.random.uniform(0.98, 1.0, len(dates)),
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, len(dates))
    }, index=dates)
    
    return df


@pytest.fixture
def mock_strategy():
    """Mock Strategy for testing"""
    strategy = Mock()
    strategy.name = 'test_strategy'
    strategy.trading_pair = 'BTC/USDT'
    
    def mock_generate_signal(data, current_candle):
        # Random signal generation for testing
        signals = ['buy', 'sell', 'hold']
        signal = np.random.choice(signals, p=[0.3, 0.3, 0.4])
        
        if signal == 'hold':
            return None
        
        return {
            'trade_type': signal,
            'amount': 0.001,
            'confidence': np.random.uniform(0.6, 0.9),
            'reason': f'test_{signal}_signal'
        }
    
    strategy.generate_signal.side_effect = mock_generate_signal
    strategy.calculate_signal.return_value = ('HOLD', {'confidence': 0.5})
    
    return strategy


@pytest.fixture
def mock_ml_components():
    """Mock ML Components for testing"""
    ml_components = Mock()
    
    # Mock market regime detector
    ml_components.market_regime_detector = Mock()
    ml_components.market_regime_detector.predict_regime.return_value = {
        'status': 'success',
        'label': 'bull',
        'confidence': 0.8,
        'regime_probabilities': {
            'bull': 0.8,
            'bear': 0.1,
            'sideways': 0.1
        }
    }
    
    return ml_components


@pytest.fixture
def mock_strategy_router():
    """Mock Strategy Router for testing"""
    router = Mock()
    router.get_current_regime.return_value = 'bull'
    router.get_active_strategies.return_value = {}
    router.update_market_regime.return_value = None
    
    return router


@pytest.fixture
def mock_safety_manager():
    """Mock Safety Manager for testing"""
    safety_manager = Mock()
    safety_manager.is_killswitch_active.return_value = False
    safety_manager.current_drawdown_percent = 0.05
    safety_manager.set_trading_bot.return_value = None
    
    return safety_manager


@pytest.fixture
def mock_risk_manager():
    """Mock Risk Manager for testing"""
    risk_manager = Mock()
    risk_manager.can_enter_position.return_value = True
    risk_manager.calculate_position_size.return_value = 0.001
    risk_manager.check_risk_limits.return_value = True
    
    return risk_manager


@pytest.fixture
def mock_order_manager():
    """Mock Order Manager for testing"""
    order_manager = Mock()
    order_manager.create_market_buy_order.return_value = {
        'id': 'test_buy_order',
        'symbol': 'BTC/USDT',
        'type': 'market',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0,
        'status': 'closed'
    }
    order_manager.create_market_sell_order.return_value = {
        'id': 'test_sell_order',
        'symbol': 'BTC/USDT',
        'type': 'market',
        'side': 'sell',
        'amount': 0.001,
        'price': 50000.0,
        'status': 'closed'
    }
    
    return order_manager


@pytest.fixture
def mock_position_manager():
    """Mock Position Manager for testing"""
    position_manager = Mock()
    position_manager.get_total_capital.return_value = 10000.0
    position_manager.get_all_positions.return_value = {}
    position_manager.update_position_from_order.return_value = None
    position_manager.update_portfolio_value.return_value = None
    
    return position_manager


@pytest.fixture
def mock_performance_tracker():
    """Mock Performance Tracker for testing"""
    tracker = Mock()
    tracker.track_performance.return_value = None
    tracker.record_trade.return_value = None
    tracker.get_performance_summary.return_value = {
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.03,
        'win_rate': 0.65,
        'total_trades': 100
    }
    tracker.save_results.return_value = None
    
    return tracker


@pytest.fixture
def mock_secret_manager(temp_dir):
    """Mock SecretManager for testing"""
    with patch('utils.secret_manager.SecretManager') as mock_sm:
        instance = mock_sm.return_value
        instance.store_secret.return_value = True
        instance.get_secret.return_value = 'test_secret_value'
        instance.delete_secret.return_value = True
        instance.list_secrets.return_value = ['test_service_api_key', 'test_service_api_secret']
        yield instance


@pytest.fixture
def mock_secure_error_handler():
    """Mock SecureErrorHandler for testing"""
    handler = Mock(spec=SecureErrorHandler)
    
    def mock_handle_error(error, **kwargs):
        return Mock(
            error_id='test-error-123',
            timestamp='2024-01-01T12:00:00',
            category='test',
            severity='medium',
            message=str(error)
        )
    
    handler.handle_critical_error.side_effect = mock_handle_error
    handler.handle_trading_error.side_effect = mock_handle_error
    handler.handle_api_error.side_effect = mock_handle_error
    
    return handler


@pytest.fixture
def api_credentials():
    """Test API credentials"""
    return {
        'api_key': 'test_api_key_12345',
        'api_secret': 'test_api_secret_67890'
    }


@pytest.fixture
def trading_signal():
    """Sample trading signal for testing"""
    return {
        'trade_type': 'buy',
        'amount': 0.001,
        'symbol': 'BTC/USDT',
        'confidence': 0.8,
        'reason': 'test_signal'
    }


@pytest.fixture
def market_data():
    """Sample market data for testing"""
    return {
        'BTC/USDT': {
            'price': 50000.0,
            'volume': 1000000.0,
            '24h_change': 0.02,
            'bid': 49995.0,
            'ask': 50005.0
        },
        'ETH/USDT': {
            'price': 3000.0,
            'volume': 500000.0,
            '24h_change': 0.01,
            'bid': 2999.0,
            'ask': 3001.0
        }
    }


@pytest.fixture(autouse=True)
def reset_loggers():
    """Reset loggers before each test to avoid conflicts"""
    import logging
    # Clear all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset to basic config
    logging.basicConfig(level=logging.WARNING, handlers=[])
    yield
    
    # Cleanup after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture
def env_vars():
    """Setup test environment variables"""
    test_env = {
        'BINANCE_API_KEY': 'test_binance_key',
        'BINANCE_API_SECRET': 'test_binance_secret',
        'TELEGRAM_BOT_TOKEN': 'test_telegram_token',
        'TELEGRAM_CHAT_ID': 'test_chat_id'
    }
    
    # Set env vars
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Clean up
    for key in test_env.keys():
        os.environ.pop(key, None)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )


# Custom assertions
def assert_valid_trading_signal(signal):
    """Assert that a trading signal is valid"""
    assert isinstance(signal, dict)
    assert 'trade_type' in signal
    assert signal['trade_type'] in ['buy', 'sell']
    assert 'amount' in signal
    assert signal['amount'] > 0
    assert 'symbol' in signal or 'trading_pair' in signal


def assert_valid_error_response(response):
    """Assert that an error response is valid"""
    assert hasattr(response, 'error_id')
    assert hasattr(response, 'timestamp')
    assert hasattr(response, 'category')
    assert hasattr(response, 'severity')
    assert hasattr(response, 'message')


# Test data generators
def generate_price_data(symbol='BTC/USDT', days=30, base_price=50000):
    """Generate realistic price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
    
    # Generate returns with some trend and volatility
    returns = np.random.normal(0.0001, 0.01, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.999, 1.001, len(dates)),
        'high': prices * np.random.uniform(1.0, 1.01, len(dates)),
        'low': prices * np.random.uniform(0.99, 1.0, len(dates)),
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, len(dates))
    }, index=dates)
    
    return df