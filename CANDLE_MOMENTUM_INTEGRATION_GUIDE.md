# Candle Momentum Strategy Integration Guide

## Overview
This guide explains how to integrate the new Candle Momentum Strategy into your existing trading bot system.

## Files Created/Modified

### New Files:
1. `strategies/candle_momentum.py` - Main strategy implementation
2. `analysis/technical.py` - Enhanced technical analysis module
3. `config/candle_momentum_config.json` - Configuration example
4. `CANDLE_MOMENTUM_INTEGRATION_GUIDE.md` - This guide

## Integration Steps

### 1. Update main.py

Add the following imports and modifications to your `main.py`:

```python
# Add to imports section
from strategies.candle_momentum import CandleMomentumStrategy

# Add to strategy registry (if you have one)
AVAILABLE_STRATEGIES = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'ml_strategy': MLStrategy,
    'candle_momentum': CandleMomentumStrategy,  # NEW
    # ... other strategies
}

# In your strategy initialization section
def initialize_strategy(strategy_name, config):
    """Initialize strategy based on name"""
    if strategy_name == 'candle_momentum':
        return CandleMomentumStrategy(config.get('candle_momentum', {}))
    elif strategy_name == 'momentum':
        return MomentumStrategy(config.get('momentum', {}))
    # ... other strategy initializations
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
```

### 2. Update config/default.json

Add the candle momentum configuration to your main config file:

```json
{
  "strategies": {
    "candle_momentum": {
      "enabled": true,
      "lookback_period": 20,
      "sma_period": 50,
      "use_ema": false,
      "volume_filter": true,
      "volume_period": 20,
      "min_momentum_ratio": 1.2,
      "min_confidence": 0.5,
      "multi_timeframe": false,
      "higher_timeframe": "4h"
    }
  },
  "trading": {
    "default_strategy": "candle_momentum",
    "max_position_size": 0.1,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.04
  }
}
```

### 3. Update strategies/__init__.py

If you have an `__init__.py` file in your strategies directory, add:

```python
from .candle_momentum import CandleMomentumStrategy

__all__ = [
    'Strategy',  # base class
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MLStrategy',
    'CandleMomentumStrategy',  # NEW
]
```

### 4. Update requirements.txt (if needed)

Ensure these packages are in your requirements.txt:

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
```

## Usage Examples

### Basic Usage

```python
from strategies.candle_momentum import CandleMomentumStrategy
import pandas as pd

# Initialize strategy
config = {
    'lookback_period': 20,
    'sma_period': 50,
    'volume_filter': True,
    'min_momentum_ratio': 1.2
}

strategy = CandleMomentumStrategy(config)

# Generate signals
signals = strategy.generate_signals(ohlcv_data, 'BTC/USDT')
print(f"Signal: {signals['signal']}, Confidence: {signals['confidence']}")
```

### Backtesting Integration

```python
# In your backtesting engine
def run_backtest(strategy_name, symbol, start_date, end_date):
    # Load data
    data = load_historical_data(symbol, start_date, end_date)
    
    # Initialize strategy
    if strategy_name == 'candle_momentum':
        strategy = CandleMomentumStrategy(config['candle_momentum'])
    
    # Run backtest
    results = []
    for i in range(len(data)):
        window = data.iloc[max(0, i-100):i+1]  # Use sliding window
        signal = strategy.generate_signals(window, symbol)
        results.append(signal)
    
    return results
```

### Live Trading Integration

```python
# In your live trading loop
def trading_loop():
    while True:
        for symbol in trading_pairs:
            # Get latest data
            data = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Generate signal
            signal = strategy.generate_signals(df, symbol)
            
            if signal['signal'] != 'hold' and signal['confidence'] > 0.6:
                # Calculate position size
                position_size = strategy.calculate_position_size(
                    signal, account_balance, df['close'].iloc[-1]
                )
                
                # Execute trade
                if signal['signal'] == 'buy':
                    place_buy_order(symbol, position_size)
                elif signal['signal'] == 'sell':
                    place_sell_order(symbol, position_size)
        
        time.sleep(60)  # Wait 1 minute
```

## Performance Monitoring

### Getting Strategy Information

```python
# Get current strategy state
info = strategy.get_strategy_info()
print(f"Active symbols: {info['active_symbols']}")
print(f"Parameters: {info['parameters']}")
```

### Visualization Data

```python
# Get data for plotting momentum
viz_data = strategy.get_momentum_visualization_data('BTC/USDT', data)

# Plot momentum (example with matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(viz_data['timestamp'], viz_data['price'], label='Price')
plt.plot(viz_data['timestamp'], viz_data['trend_line'], label='Trend Line')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(viz_data['timestamp'], viz_data['bullish_strength'], label='Bullish Strength', color='green')
plt.plot(viz_data['timestamp'], viz_data['bearish_strength'], label='Bearish Strength', color='red')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(viz_data['timestamp'], viz_data['momentum_ratio'], label='Momentum Ratio')
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
```

## Configuration Options

### Core Parameters
- `lookback_period`: Number of candles to analyze for momentum (default: 20)
- `sma_period`: Period for trend filter SMA/EMA (default: 50)
- `use_ema`: Use EMA instead of SMA for trend filter (default: false)
- `volume_filter`: Enable volume filtering (default: true)
- `min_momentum_ratio`: Minimum momentum ratio for signals (default: 1.2)
- `min_confidence`: Minimum confidence threshold (default: 0.5)

### Advanced Parameters
- `multi_timeframe`: Enable multi-timeframe analysis (default: false)
- `higher_timeframe`: Higher timeframe for confirmation (default: "4h")
- `volume_period`: Period for volume average calculation (default: 20)

## Risk Management

The strategy includes built-in risk management:

1. **Position Sizing**: Adjusts based on momentum strength and confidence
2. **Stop Loss**: Configurable stop-loss percentage
3. **Take Profit**: Configurable take-profit targets
4. **Volume Filtering**: Only trades on above-average volume
5. **Trend Confirmation**: Requires price above/below trend line

## Troubleshooting

### Common Issues

1. **"Insufficient Data" Warning**
   - Ensure you have at least `max(lookback_period, sma_period) + 10` candles
   - Default minimum: ~60 candles

2. **No Signals Generated**
   - Check `min_confidence` and `min_momentum_ratio` settings
   - Verify volume filter isn't too restrictive
   - Check if trend filter is preventing signals

3. **Poor Performance**
   - Optimize parameters using backtesting
   - Consider different timeframes
   - Adjust risk management settings

### Logging

Enable detailed logging to debug issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The strategy will log:
# - Signal generation details
# - Momentum calculations
# - Risk management decisions
# - Error conditions
```

## Testing

Before live trading, thoroughly test the strategy:

1. **Unit Tests**: Test individual methods
2. **Backtesting**: Run historical backtests
3. **Paper Trading**: Test with live data, no real money
4. **Small Position**: Start with minimal position sizes

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify your data format matches expected OHLCV structure
3. Ensure all dependencies are installed
4. Review configuration parameters

The strategy is designed to be robust and handle edge cases, but always monitor performance and adjust parameters as needed for your specific use case.