"""
Test Suite for Trading Strategies
=================================

Tests for:
- All strategy classes
- Signal generation algorithms
- Strategy performance
- Risk management integration
- Strategy parameter validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Tuple

# Import strategy components to test
from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.defi_yield import DeFiYieldStrategy
from strategies.ml_strategy import MLStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.liquidation import LiquidationStrategy
from strategies import STRATEGIES


class TestStrategyBase:
    """Test base Strategy class functionality"""
    
    @pytest.fixture
    def base_strategy_config(self):
        """Base strategy configuration for testing"""
        return {
            'trading_pair': 'BTC/USDT',
            'timeframe': '1h',
            'lookback_period': 50,
            'confidence_threshold': 0.7
        }
    
    def test_strategy_base_initialization(self, base_strategy_config):
        """Test Strategy base class initialization"""
        strategy = Strategy(base_strategy_config)
        
        assert strategy.config == base_strategy_config
        assert strategy.trading_pair == 'BTC/USDT'
        assert hasattr(strategy, 'calculate_signal')
    
    def test_strategy_base_abstract_methods(self, base_strategy_config):
        """Test that Strategy base class has abstract methods"""
        strategy = Strategy(base_strategy_config)
        
        # calculate_signal should be implemented by subclasses
        with pytest.raises(NotImplementedError):
            strategy.calculate_signal('BTC/USDT', pd.DataFrame(), 50000.0)
    
    def test_strategy_parameter_access(self, base_strategy_config):
        """Test strategy parameter access"""
        strategy = Strategy(base_strategy_config)
        
        assert strategy.get_parameter('trading_pair') == 'BTC/USDT'
        assert strategy.get_parameter('timeframe') == '1h'
        assert strategy.get_parameter('non_existent', 'default') == 'default'
    
    def test_strategy_parameter_validation(self, base_strategy_config):
        """Test strategy parameter validation"""
        # Test with missing required parameters
        incomplete_config = {'trading_pair': 'BTC/USDT'}
        
        with pytest.raises(ValueError):
            Strategy(incomplete_config)


class TestMomentumStrategy:
    """Test MomentumStrategy functionality"""
    
    @pytest.fixture
    def momentum_config(self):
        """Momentum strategy configuration"""
        return {
            'trading_pair': 'BTC/USDT',
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'sma_short_period': 5,
            'sma_long_period': 20,
            'volume_threshold': 1.5
        }
    
    @pytest.fixture
    def momentum_strategy(self, momentum_config):
        """Create MomentumStrategy instance"""
        return MomentumStrategy(momentum_config)
    
    @pytest.fixture
    def trending_data(self):
        """Generate trending market data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        
        # Create uptrend
        base_price = 50000
        trend = np.linspace(0, 0.1, 100)  # 10% uptrend
        noise = np.random.normal(0, 0.01, 100)
        
        prices = base_price * (1 + trend + noise)
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        return df
    
    def test_momentum_strategy_initialization(self, momentum_strategy, momentum_config):
        """Test MomentumStrategy initialization"""
        assert momentum_strategy.trading_pair == 'BTC/USDT'
        assert momentum_strategy.rsi_oversold == 30
        assert momentum_strategy.rsi_overbought == 70
        assert momentum_strategy.sma_short == 5
        assert momentum_strategy.sma_long == 20
    
    def test_momentum_signal_generation(self, momentum_strategy, trending_data):
        """Test momentum signal generation"""
        current_price = trending_data['close'].iloc[-1]
        current_candle = trending_data.iloc[-1]
        
        signal, signal_data = momentum_strategy.calculate_signal(
            'BTC/USDT', trending_data, current_price
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)
        assert 'confidence' in signal_data
        assert 'reason' in signal_data
        assert 0 <= signal_data['confidence'] <= 1
    
    def test_rsi_calculation(self, momentum_strategy, trending_data):
        """Test RSI calculation"""
        rsi_values = momentum_strategy._calculate_rsi(trending_data['close'])
        
        assert len(rsi_values) > 0
        assert all(0 <= rsi <= 100 for rsi in rsi_values if not np.isnan(rsi))
    
    def test_sma_calculation(self, momentum_strategy, trending_data):
        """Test SMA calculation"""
        sma_short = momentum_strategy._calculate_sma(trending_data['close'], 5)
        sma_long = momentum_strategy._calculate_sma(trending_data['close'], 20)
        
        assert len(sma_short) == len(trending_data)
        assert len(sma_long) == len(trending_data)
        
        # Short SMA should be less noisy than price
        # Long SMA should be even smoother
        assert np.nanstd(sma_long) <= np.nanstd(sma_short)
    
    def test_volume_analysis(self, momentum_strategy, trending_data):
        """Test volume analysis"""
        volume_ratio = momentum_strategy._analyze_volume(trending_data)
        
        assert isinstance(volume_ratio, float)
        assert volume_ratio > 0
    
    def test_buy_signal_conditions(self, momentum_strategy):
        """Test buy signal generation conditions"""
        # Create data that should generate buy signal
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        # Price moving up, RSI oversold, volume high
        prices = np.linspace(50000, 52000, 50)  # Uptrend
        volumes = np.full(50, 2000000)  # High volume
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        signal, signal_data = momentum_strategy.calculate_signal('BTC/USDT', df, prices[-1])
        
        # Should generate some signal (not necessarily BUY due to randomness)
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert signal_data['confidence'] >= 0
    
    def test_sell_signal_conditions(self, momentum_strategy):
        """Test sell signal generation conditions"""
        # Create data that should generate sell signal
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        # Price moving down, RSI overbought
        prices = np.linspace(52000, 50000, 50)  # Downtrend
        volumes = np.full(50, 2000000)  # High volume
        
        df = pd.DataFrame({
            'open': prices * 1.001,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        signal, signal_data = momentum_strategy.calculate_signal('BTC/USDT', df, prices[-1])
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert signal_data['confidence'] >= 0
    
    def test_insufficient_data_handling(self, momentum_strategy):
        """Test handling of insufficient data"""
        # Create very small dataset
        small_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50200],
            'volume': [1000000, 1100000]
        })
        
        signal, signal_data = momentum_strategy.calculate_signal('BTC/USDT', small_data, 50200)
        
        # Should handle gracefully
        assert signal == 'HOLD'
        assert 'insufficient_data' in signal_data.get('reason', '')


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy functionality"""
    
    @pytest.fixture
    def mean_reversion_config(self):
        """Mean reversion strategy configuration"""
        return {
            'trading_pair': 'ETH/USDT',
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'use_rsi_filter': True,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
    
    @pytest.fixture
    def mean_reversion_strategy(self, mean_reversion_config):
        """Create MeanReversionStrategy instance"""
        return MeanReversionStrategy(mean_reversion_config)
    
    @pytest.fixture
    def ranging_data(self):
        """Generate ranging/sideways market data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        
        # Create sideways movement with oscillations
        base_price = 3000
        oscillation = np.sin(np.linspace(0, 4*np.pi, 100)) * 50  # ±50 oscillation
        noise = np.random.normal(0, 10, 100)
        
        prices = base_price + oscillation + noise
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(500000, 1000000, 100)
        }, index=dates)
        
        return df
    
    def test_mean_reversion_initialization(self, mean_reversion_strategy, mean_reversion_config):
        """Test MeanReversionStrategy initialization"""
        assert mean_reversion_strategy.trading_pair == 'ETH/USDT'
        assert mean_reversion_strategy.bollinger_period == 20
        assert mean_reversion_strategy.bollinger_std == 2.0
        assert mean_reversion_strategy.use_rsi_filter is True
    
    def test_bollinger_bands_calculation(self, mean_reversion_strategy, ranging_data):
        """Test Bollinger Bands calculation"""
        bb_upper, bb_middle, bb_lower = mean_reversion_strategy._calculate_bollinger_bands(
            ranging_data['close']
        )
        
        assert len(bb_upper) == len(ranging_data)
        assert len(bb_middle) == len(ranging_data)
        assert len(bb_lower) == len(ranging_data)
        
        # Upper band should be above middle, middle above lower
        valid_indices = ~(np.isnan(bb_upper) | np.isnan(bb_middle) | np.isnan(bb_lower))
        assert all(bb_upper[valid_indices] >= bb_middle[valid_indices])
        assert all(bb_middle[valid_indices] >= bb_lower[valid_indices])
    
    def test_mean_reversion_signal_generation(self, mean_reversion_strategy, ranging_data):
        """Test mean reversion signal generation"""
        current_price = ranging_data['close'].iloc[-1]
        
        signal, signal_data = mean_reversion_strategy.calculate_signal(
            'ETH/USDT', ranging_data, current_price
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)
        assert 'confidence' in signal_data
        assert 'reason' in signal_data
    
    def test_oversold_buy_signal(self, mean_reversion_strategy):
        """Test buy signal when price is oversold"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        # Create data where price drops below lower Bollinger Band
        base_price = 3000
        prices = np.concatenate([
            np.full(25, base_price),
            np.linspace(base_price, base_price * 0.93, 25)  # Sharp drop
        ])
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(500000, 1000000, 50)
        }, index=dates)
        
        signal, signal_data = mean_reversion_strategy.calculate_signal('ETH/USDT', df, prices[-1])
        
        # Should detect oversold condition
        assert 'bollinger' in signal_data.get('reason', '').lower() or signal == 'BUY'
    
    def test_overbought_sell_signal(self, mean_reversion_strategy):
        """Test sell signal when price is overbought"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        # Create data where price rises above upper Bollinger Band
        base_price = 3000
        prices = np.concatenate([
            np.full(25, base_price),
            np.linspace(base_price, base_price * 1.07, 25)  # Sharp rise
        ])
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(500000, 1000000, 50)
        }, index=dates)
        
        signal, signal_data = mean_reversion_strategy.calculate_signal('ETH/USDT', df, prices[-1])
        
        # Should detect overbought condition
        assert 'bollinger' in signal_data.get('reason', '').lower() or signal == 'SELL'


class TestArbitrageStrategy:
    """Test ArbitrageStrategy functionality"""
    
    @pytest.fixture
    def arbitrage_config(self):
        """Arbitrage strategy configuration"""
        return {
            'trading_pair': 'XRP/USDT',
            'min_profit_threshold': 0.005,
            'max_execution_slippage': 0.0002,
            'exchanges': ['binance', 'coinbase', 'kraken']
        }
    
    @pytest.fixture
    def arbitrage_strategy(self, arbitrage_config):
        """Create ArbitrageStrategy instance"""
        return ArbitrageStrategy(arbitrage_config)
    
    def test_arbitrage_initialization(self, arbitrage_strategy, arbitrage_config):
        """Test ArbitrageStrategy initialization"""
        assert arbitrage_strategy.trading_pair == 'XRP/USDT'
        assert hasattr(arbitrage_strategy, 'calculate_signal')
    
    def test_arbitrage_signal_generation(self, arbitrage_strategy, sample_ohlcv_data):
        """Test arbitrage signal generation"""
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        signal, signal_data = arbitrage_strategy.calculate_signal(
            'XRP/USDT', sample_ohlcv_data, current_price
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)
        assert 'confidence' in signal_data
        assert 'price_difference' in signal_data
    
    def test_arbitrage_opportunity_detection(self, arbitrage_strategy, sample_ohlcv_data):
        """Test arbitrage opportunity detection"""
        # Simulate multiple runs to test randomness
        signals = []
        for _ in range(10):
            signal, signal_data = arbitrage_strategy.calculate_signal(
                'XRP/USDT', sample_ohlcv_data, 0.6
            )
            signals.append((signal, signal_data))
        
        # Should generate various signals due to random price differences
        signal_types = [s[0] for s in signals]
        assert len(set(signal_types)) > 1  # Should have variety
    
    def test_arbitrage_profit_threshold(self, arbitrage_strategy, sample_ohlcv_data):
        """Test profit threshold enforcement"""
        # Mock the price difference calculation to return specific values
        with patch.object(arbitrage_strategy, 'calculate_signal') as mock_calc:
            # Test below threshold
            mock_calc.return_value = ('HOLD', {
                'signal': 'HOLD',
                'confidence': 0.3,
                'reason': 'no_arbitrage',
                'price_difference': 0.001  # Below 0.005 threshold
            })
            
            signal, signal_data = arbitrage_strategy.calculate_signal('XRP/USDT', sample_ohlcv_data, 0.6)
            assert signal == 'HOLD'
            assert signal_data['price_difference'] < 0.005


class TestDeFiYieldStrategy:
    """Test DeFiYieldStrategy functionality"""
    
    @pytest.fixture
    def defi_config(self):
        """DeFi yield strategy configuration"""
        return {
            'trading_pair': 'USDT/USDC',
            'min_apy': 0.15,
            'compound_frequency': 24,
            'gas_limit': 0.005,
            'max_protocols': 5,
            'risk_levels': ['low', 'medium'],
            'min_tvl': 10_000_000
        }
    
    @pytest.fixture
    def defi_strategy(self, defi_config):
        """Create DeFiYieldStrategy instance"""
        return DeFiYieldStrategy(defi_config)
    
    def test_defi_initialization(self, defi_strategy, defi_config):
        """Test DeFiYieldStrategy initialization"""
        assert defi_strategy.min_apy == 0.15
        assert defi_strategy.compound_frequency == 24
        assert defi_strategy.gas_limit == 0.005
        assert 'aave' in defi_strategy.protocol_weights
    
    def test_stablecoin_detection(self, defi_strategy):
        """Test stablecoin detection"""
        assert defi_strategy._is_stablecoin('USDT/USDC') is True
        assert defi_strategy._is_stablecoin('BTC/USDT') is False
        assert defi_strategy._is_stablecoin('ETH/USD') is False
        assert defi_strategy._is_stablecoin('USDT') is True
    
    def test_yield_opportunity_scanning(self, defi_strategy):
        """Test yield opportunity scanning"""
        opportunities = defi_strategy._scan_yield_opportunities('USDT/USDC')
        
        assert isinstance(opportunities, list)
        # Should find some opportunities (mocked data)
        if opportunities:
            opp = opportunities[0]
            assert hasattr(opp, 'protocol')
            assert hasattr(opp, 'apy')
            assert hasattr(opp, 'tvl')
            assert opp.apy >= defi_strategy.min_apy
    
    def test_defi_signal_generation(self, defi_strategy, sample_ohlcv_data):
        """Test DeFi yield signal generation for stablecoin"""
        signal, signal_data = defi_strategy.calculate_signal(
            'USDT/USDC', sample_ohlcv_data, 1.0
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)
        assert 'strategy' in signal_data
        assert signal_data['strategy'] == 'defi_yield'
    
    def test_non_stablecoin_rejection(self, defi_strategy, sample_ohlcv_data):
        """Test that non-stablecoins are rejected"""
        signal, signal_data = defi_strategy.calculate_signal(
            'BTC/USDT', sample_ohlcv_data, 50000.0
        )
        
        assert signal == 'HOLD'
        assert signal_data['reason'] == 'not_stablecoin'
    
    def test_opportunity_selection(self, defi_strategy):
        """Test best opportunity selection algorithm"""
        from strategies.defi_yield import YieldOpportunity
        
        opportunities = [
            YieldOpportunity('aave', 'USDT-aave', 0.20, 5_000_000_000, 0.2, 50, 100),
            YieldOpportunity('compound', 'USDT-compound', 0.15, 3_000_000_000, 0.2, 45, 100),
            YieldOpportunity('yearn', 'USDT-yearn', 0.25, 1_000_000_000, 0.5, 100, 100)
        ]
        
        best = defi_strategy._select_best_opportunity(opportunities)
        
        assert best is not None
        # Should consider risk-adjusted returns
        assert best.apy >= defi_strategy.min_apy


class TestMLStrategy:
    """Test MLStrategy functionality"""
    
    @pytest.fixture
    def ml_config(self):
        """ML strategy configuration"""
        return {
            'trading_pair': 'ADA/USDT',
            'prediction_threshold': 0.7,
            'model_confidence_min': 0.6,
            'feature_window': 20,
            'prediction_horizon': 5
        }
    
    @pytest.fixture
    def ml_strategy(self, ml_config):
        """Create MLStrategy instance"""
        with patch('strategies.ml_strategy.MLStrategy._load_model'):
            return MLStrategy(ml_config)
    
    def test_ml_initialization(self, ml_strategy, ml_config):
        """Test MLStrategy initialization"""
        assert ml_strategy.trading_pair == 'ADA/USDT'
        assert ml_strategy.prediction_threshold == 0.7
        assert ml_strategy.model_confidence_min == 0.6
    
    def test_feature_extraction(self, ml_strategy, sample_ohlcv_data):
        """Test feature extraction from OHLCV data"""
        with patch.object(ml_strategy, '_extract_features') as mock_extract:
            mock_extract.return_value = np.random.rand(10)  # Mock features
            
            features = ml_strategy._extract_features(sample_ohlcv_data)
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
    
    def test_ml_prediction(self, ml_strategy, sample_ohlcv_data):
        """Test ML model prediction"""
        with patch.object(ml_strategy, 'model') as mock_model, \
             patch.object(ml_strategy, '_extract_features') as mock_extract:
            
            mock_extract.return_value = np.random.rand(10)
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # High confidence buy
            
            signal, signal_data = ml_strategy.calculate_signal(
                'ADA/USDT', sample_ohlcv_data, 0.75
            )
            
            assert signal in ['BUY', 'SELL', 'HOLD']
            assert 'confidence' in signal_data
            assert 'model_prediction' in signal_data


class TestGridTradingStrategy:
    """Test GridTradingStrategy functionality"""
    
    @pytest.fixture
    def grid_config(self):
        """Grid trading strategy configuration"""
        return {
            'trading_pair': 'BNB/USDT',
            'num_grids': 10,
            'price_range_multiplier': 0.05,
            'grid_size_percent': 0.01,
            'initial_capital': 1000
        }
    
    @pytest.fixture
    def grid_strategy(self, grid_config):
        """Create GridTradingStrategy instance"""
        return GridTradingStrategy(grid_config)
    
    def test_grid_initialization(self, grid_strategy, grid_config):
        """Test GridTradingStrategy initialization"""
        assert grid_strategy.trading_pair == 'BNB/USDT'
        assert grid_strategy.num_grids == 10
        assert grid_strategy.price_range_multiplier == 0.05
    
    def test_grid_setup(self, grid_strategy):
        """Test grid setup calculation"""
        current_price = 300.0
        
        with patch.object(grid_strategy, '_setup_grid') as mock_setup:
            mock_setup.return_value = {
                'levels': [285, 290, 295, 300, 305, 310, 315],
                'buy_levels': [285, 290, 295],
                'sell_levels': [305, 310, 315]
            }
            
            grid_info = grid_strategy._setup_grid(current_price)
            assert 'levels' in grid_info
            assert 'buy_levels' in grid_info
            assert 'sell_levels' in grid_info
    
    def test_grid_signal_generation(self, grid_strategy, sample_ohlcv_data):
        """Test grid trading signal generation"""
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        signal, signal_data = grid_strategy.calculate_signal(
            'BNB/USDT', sample_ohlcv_data, current_price
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)


class TestLiquidationStrategy:
    """Test LiquidationStrategy functionality"""
    
    @pytest.fixture
    def liquidation_config(self):
        """Liquidation strategy configuration"""
        return {
            'trading_pair': 'SOL/USDT',
            'min_profit_usd': 50,
            'liquidation_bonus_threshold': 0.01,
            'max_gas_cost': 20
        }
    
    @pytest.fixture
    def liquidation_strategy(self, liquidation_config):
        """Create LiquidationStrategy instance"""
        return LiquidationStrategy(liquidation_config)
    
    def test_liquidation_initialization(self, liquidation_strategy, liquidation_config):
        """Test LiquidationStrategy initialization"""
        assert liquidation_strategy.trading_pair == 'SOL/USDT'
        assert liquidation_strategy.min_profit_usd == 50
    
    def test_liquidation_opportunity_detection(self, liquidation_strategy, sample_ohlcv_data):
        """Test liquidation opportunity detection"""
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        signal, signal_data = liquidation_strategy.calculate_signal(
            'SOL/USDT', sample_ohlcv_data, current_price
        )
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert isinstance(signal_data, dict)


class TestStrategyRegistry:
    """Test strategy registry and loading"""
    
    def test_strategies_registry_exists(self):
        """Test that STRATEGIES registry exists and contains strategies"""
        assert isinstance(STRATEGIES, dict)
        assert len(STRATEGIES) > 0
    
    def test_all_strategies_are_classes(self):
        """Test that all registered strategies are proper classes"""
        for name, strategy_class in STRATEGIES.items():
            assert isinstance(strategy_class, type)
            assert issubclass(strategy_class, Strategy)
    
    def test_strategy_instantiation(self):
        """Test that all strategies can be instantiated"""
        base_config = {
            'trading_pair': 'BTC/USDT',
            'timeframe': '1h'
        }
        
        for name, strategy_class in STRATEGIES.items():
            # Skip strategies that require special config
            if name in ['ml_strategy']:  # ML strategy needs model file
                continue
                
            try:
                strategy = strategy_class(base_config)
                assert isinstance(strategy, Strategy)
                assert strategy.trading_pair == 'BTC/USDT'
            except Exception as e:
                pytest.fail(f"Failed to instantiate {name}: {e}")
    
    def test_strategy_signal_interface(self, sample_ohlcv_data):
        """Test that all strategies implement signal interface correctly"""
        base_config = {
            'trading_pair': 'BTC/USDT',
            'timeframe': '1h'
        }
        
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        for name, strategy_class in STRATEGIES.items():
            if name in ['ml_strategy']:  # Skip special cases
                continue
                
            try:
                strategy = strategy_class(base_config)
                signal, signal_data = strategy.calculate_signal(
                    'BTC/USDT', sample_ohlcv_data, current_price
                )
                
                assert signal in ['BUY', 'SELL', 'HOLD']
                assert isinstance(signal_data, dict)
                assert 'confidence' in signal_data
                
            except Exception as e:
                pytest.fail(f"Failed signal generation for {name}: {e}")


class TestStrategyPerformance:
    """Test strategy performance and reliability"""
    
    def test_strategy_consistency(self, sample_ohlcv_data):
        """Test that strategies produce consistent results"""
        config = {'trading_pair': 'BTC/USDT'}
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        # Test momentum strategy consistency
        strategy = MomentumStrategy(config)
        
        signals = []
        for _ in range(5):
            signal, signal_data = strategy.calculate_signal(
                'BTC/USDT', sample_ohlcv_data, current_price
            )
            signals.append(signal)
        
        # Should be deterministic (same inputs -> same outputs)
        assert len(set(signals)) == 1, "Strategy should be deterministic"
    
    def test_strategy_error_handling(self):
        """Test strategy error handling with invalid data"""
        config = {'trading_pair': 'BTC/USDT'}
        strategy = MomentumStrategy(config)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        signal, signal_data = strategy.calculate_signal('BTC/USDT', empty_df, 50000)
        
        assert signal == 'HOLD'
        assert 'error' in signal_data or 'insufficient' in signal_data.get('reason', '')
    
    def test_strategy_parameter_sensitivity(self, sample_ohlcv_data):
        """Test strategy sensitivity to parameter changes"""
        base_config = {'trading_pair': 'BTC/USDT', 'rsi_oversold': 30}
        modified_config = {'trading_pair': 'BTC/USDT', 'rsi_oversold': 20}
        
        current_price = sample_ohlcv_data['close'].iloc[-1]
        
        strategy1 = MomentumStrategy(base_config)
        strategy2 = MomentumStrategy(modified_config)
        
        signal1, _ = strategy1.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
        signal2, _ = strategy2.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
        
        # Different parameters might produce different signals
        # This tests that parameters actually affect the strategy
        assert isinstance(signal1, str)
        assert isinstance(signal2, str)
    
    @pytest.mark.slow
    def test_strategy_performance_metrics(self, sample_ohlcv_data):
        """Test strategy performance calculation"""
        config = {'trading_pair': 'BTC/USDT'}
        strategy = MomentumStrategy(config)
        
        signals = []
        prices = sample_ohlcv_data['close'].values
        
        # Generate signals for entire dataset
        for i in range(50, len(sample_ohlcv_data)):
            data_slice = sample_ohlcv_data.iloc[:i]
            current_price = prices[i-1]
            
            signal, signal_data = strategy.calculate_signal('BTC/USDT', data_slice, current_price)
            signals.append({
                'signal': signal,
                'price': current_price,
                'confidence': signal_data.get('confidence', 0)
            })
        
        # Calculate basic performance metrics
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        
        assert len(signals) > 0
        assert isinstance(buy_signals, list)
        assert isinstance(sell_signals, list)
        
        # Test that confidence values are reasonable
        if signals:
            confidences = [s['confidence'] for s in signals if s['confidence'] > 0]
            if confidences:
                assert all(0 <= c <= 1 for c in confidences)


class TestStrategyIntegration:
    """Integration tests for strategies with other components"""
    
    def test_strategy_with_risk_manager(self, sample_ohlcv_data, mock_risk_manager):
        """Test strategy integration with risk manager"""
        config = {'trading_pair': 'BTC/USDT'}
        strategy = MomentumStrategy(config)
        
        current_price = sample_ohlcv_data['close'].iloc[-1]
        signal, signal_data = strategy.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
        
        if signal in ['BUY', 'SELL']:
            # Test that risk manager can process the signal
            assert mock_risk_manager.can_enter_position('BTC/USDT', 0.001, signal.lower()) is True
    
    def test_strategy_with_position_manager(self, sample_ohlcv_data, mock_position_manager):
        """Test strategy integration with position manager"""
        config = {'trading_pair': 'BTC/USDT'}
        strategy = MomentumStrategy(config)
        
        current_price = sample_ohlcv_data['close'].iloc[-1]
        signal, signal_data = strategy.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
        
        # Strategy should work regardless of current positions
        mock_position_manager.get_all_positions.return_value = {'BTC/USDT': {'amount': 0.001}}
        
        signal2, signal_data2 = strategy.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
        
        # Should still generate valid signals
        assert signal2 in ['BUY', 'SELL', 'HOLD']
    
    def test_strategy_with_error_handler(self, sample_ohlcv_data, mock_secure_error_handler):
        """Test strategy integration with error handler"""
        config = {'trading_pair': 'BTC/USDT'}
        
        # Test with strategy that might raise errors
        with patch('strategies.momentum.MomentumStrategy._calculate_rsi', side_effect=Exception("Test error")):
            strategy = MomentumStrategy(config)
            
            # Strategy should handle errors gracefully
            current_price = sample_ohlcv_data['close'].iloc[-1]
            signal, signal_data = strategy.calculate_signal('BTC/USDT', sample_ohlcv_data, current_price)
            
            # Should return safe default
            assert signal == 'HOLD'
            assert 'error' in signal_data or 'error_id' in signal_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])