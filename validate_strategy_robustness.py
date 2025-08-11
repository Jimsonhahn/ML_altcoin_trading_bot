#!/usr/bin/env python3
"""
Strategy Robustness Validation - Multiple Market Conditions
==========================================================

Tests the profitable strategy across different market scenarios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List
import sys
sys.path.append('.')

# Import the standalone strategy
from test_standalone_profitable import (
    StandaloneProfitableStrategy, 
    StandaloneIndicatorEngine, 
    StandaloneBacktester
)


def generate_market_scenario(scenario_type: str, days: int = 90) -> pd.DataFrame:
    """Generate different market scenarios for testing"""
    print(f"📊 Generating {scenario_type} market scenario ({days} days)...")
    
    timestamps = []
    prices = []
    volumes = []
    
    current_time = datetime(2023, 1, 1)
    current_price = 45000.0
    
    if scenario_type == "bull_market":
        # Strong uptrend with low volatility
        trend_strength = 0.004
        base_volatility = 0.02
        mean_reversion = 0.01
        
    elif scenario_type == "bear_market":
        # Downtrend with moderate volatility
        trend_strength = -0.003
        base_volatility = 0.035
        mean_reversion = 0.015
        
    elif scenario_type == "sideways_market":
        # Range-bound with mean reversion
        trend_strength = 0.0
        base_volatility = 0.025
        mean_reversion = 0.04
        
    elif scenario_type == "high_volatility":
        # Choppy market with high volatility
        trend_strength = 0.001
        base_volatility = 0.06
        mean_reversion = 0.02
        
    elif scenario_type == "crash_recovery":
        # Sharp drop followed by recovery
        trend_strength = 0.0
        base_volatility = 0.04
        mean_reversion = 0.025
    
    for i in range(days * 24):  # Hourly data
        # Base price movement
        random_shock = np.random.normal(0, base_volatility / np.sqrt(24))
        trend_component = trend_strength / 24
        mean_reversion_component = -mean_reversion * (current_price - 45000) / 45000 / 24
        
        # Special scenarios
        if scenario_type == "crash_recovery":
            if i == days * 24 // 3:  # Crash at 1/3 point
                random_shock = -0.15  # 15% crash
            elif i > days * 24 // 3:  # Recovery phase
                trend_component = 0.003 / 24
        
        # Market cycles
        daily_cycle = 0.0003 * np.sin(i * 2 * np.pi / 24)
        
        price_change = (trend_component + mean_reversion_component + 
                       random_shock + daily_cycle)
        
        current_price *= (1 + price_change)
        current_price = max(current_price, 20000)  # Floor
        
        # Volume modeling
        base_volume = 2000
        volatility_volume = abs(price_change) * 50000
        if scenario_type == "high_volatility":
            volatility_volume *= 2
        volume = base_volume + volatility_volume + np.random.exponential(500)
        
        timestamps.append(current_time)
        prices.append(current_price)
        volumes.append(volume)
        
        current_time += timedelta(hours=1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    
    buyhold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    volatility = df['close'].pct_change().std() * np.sqrt(24 * 365)
    
    print(f"   Price Range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    print(f"   Buy&Hold Return: {buyhold_return:.2%}")
    print(f"   Annualized Volatility: {volatility:.1%}")
    
    return df


def test_strategy_scenario(scenario_name: str, market_data: pd.DataFrame) -> Dict[str, Any]:
    """Test strategy on a specific market scenario"""
    print(f"\n🧪 Testing {scenario_name}...")
    
    # Initialize components
    strategy = StandaloneProfitableStrategy()
    indicator_engine = StandaloneIndicatorEngine()
    backtester = StandaloneBacktester(100000)
    
    signals_generated = 0
    high_quality_signals = 0
    trades_executed = 0
    
    for i, (timestamp, row) in enumerate(market_data.iterrows()):
        price = row['close']
        volume = row['volume']
        
        # Update indicators
        indicators = indicator_engine.update(price, volume)
        
        # Generate signal after warmup
        if i >= 100:  # Shorter warmup for 90-day tests
            signal_strength, signal_data = strategy.calculate_signal_strength(indicators, price)
            
            if signal_data.get('direction') != 'hold':
                signals_generated += 1
                
                if signal_data.get('confidence', 0) >= strategy.min_signal_strength:
                    high_quality_signals += 1
            
            # Process signal
            result = backtester.process_tick(timestamp, price, signal_data, strategy)
            
            if result.get('action') in ['position_entered', 'position_exited']:
                trades_executed += 1
    
    # Finalize
    backtester.finalize(market_data.index[-1], market_data['close'].iloc[-1])
    metrics = backtester.get_metrics()
    
    # Calculate scenario results
    days = (market_data.index[-1] - market_data.index[0]).days
    annual_multiplier = 365.25 / days if days > 0 else 1
    
    annualized_return = ((1 + metrics.get('total_return', 0)) ** annual_multiplier - 1) if metrics.get('total_return', 0) > -1 else -1
    
    result = {
        'scenario': scenario_name,
        'days': days,
        'total_return': metrics.get('total_return', 0),
        'annualized_return': annualized_return,
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'max_drawdown': metrics.get('max_drawdown', 0),
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'profit_factor': metrics.get('profit_factor', 0),
        'signals_generated': signals_generated,
        'high_quality_signals': high_quality_signals,
        'signal_selectivity': high_quality_signals / signals_generated if signals_generated > 0 else 0
    }
    
    # Print summary
    print(f"   Return: {result['total_return']:.1%} ({result['annualized_return']:.1%} annualized)")
    print(f"   Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"   Drawdown: {result['max_drawdown']:.1%}")
    print(f"   Trades: {result['total_trades']} (Win Rate: {result['win_rate']:.1%})")
    
    return result


def main():
    """Run robustness validation across multiple market scenarios"""
    print("🔬 STRATEGY ROBUSTNESS VALIDATION")
    print("=" * 80)
    print("Testing profitable strategy across different market conditions\n")
    
    # Define test scenarios
    scenarios = [
        ("Bull Market", "bull_market"),
        ("Bear Market", "bear_market"),
        ("Sideways Market", "sideways_market"),
        ("High Volatility", "high_volatility"),
        ("Crash & Recovery", "crash_recovery")
    ]
    
    results = []
    
    for scenario_name, scenario_type in scenarios:
        # Generate market data
        market_data = generate_market_scenario(scenario_type, days=90)
        
        # Test strategy
        result = test_strategy_scenario(scenario_name, market_data)
        results.append(result)
    
    # Summary Analysis
    print("\n" + "=" * 80)
    print("📊 ROBUSTNESS VALIDATION SUMMARY")
    print("=" * 80)
    
    # Performance across scenarios
    print("\n🎯 PERFORMANCE ACROSS SCENARIOS:")
    print(f"{'Scenario':<20} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Win Rate':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['scenario']:<20} "
              f"{result['annualized_return']:>10.1%} "
              f"{result['sharpe_ratio']:>10.2f} "
              f"{result['max_drawdown']:>10.1%} "
              f"{result['win_rate']:>10.1%}")
    
    # Calculate aggregate metrics
    avg_return = np.mean([r['annualized_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    worst_drawdown = max([r['max_drawdown'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    
    positive_scenarios = sum(1 for r in results if r['total_return'] > 0)
    target_achieving = sum(1 for r in results if r['annualized_return'] > 0.30 and r['sharpe_ratio'] > 2.0)
    
    print(f"\n📈 AGGREGATE METRICS:")
    print(f"   Average Annual Return: {avg_return:.1%}")
    print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Worst Drawdown: {worst_drawdown:.1%}")
    print(f"   Average Win Rate: {avg_win_rate:.1%}")
    print(f"   Profitable Scenarios: {positive_scenarios}/{len(results)}")
    print(f"   Target-Achieving Scenarios: {target_achieving}/{len(results)}")
    
    # Robustness Assessment
    print(f"\n🛡️ ROBUSTNESS ASSESSMENT:")
    
    if avg_return > 0.30 and avg_sharpe > 1.5:
        print("   ✅ HIGHLY ROBUST - Performs well across all market conditions")
        robustness_score = "EXCELLENT"
    elif avg_return > 0.15 and avg_sharpe > 1.0:
        print("   ✅ ROBUST - Good performance in most conditions")
        robustness_score = "GOOD"
    elif avg_return > 0.05:
        print("   ⚠️ MODERATE - Profitable but sensitive to market conditions")
        robustness_score = "MODERATE"
    else:
        print("   ❌ NEEDS IMPROVEMENT - Too sensitive to market conditions")
        robustness_score = "POOR"
    
    # Risk Assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    if worst_drawdown < 0.10:
        print("   ✅ LOW RISK - Maximum drawdown under 10%")
    elif worst_drawdown < 0.20:
        print("   ⚠️ MODERATE RISK - Maximum drawdown under 20%")
    else:
        print("   ❌ HIGH RISK - Maximum drawdown over 20%")
    
    # Export validation results
    validation_results = {
        'validation_timestamp': datetime.now().isoformat(),
        'scenarios_tested': len(results),
        'scenario_results': results,
        'aggregate_metrics': {
            'avg_annual_return': avg_return,
            'avg_sharpe_ratio': avg_sharpe,
            'worst_drawdown': worst_drawdown,
            'avg_win_rate': avg_win_rate,
            'profitable_scenarios': positive_scenarios,
            'target_achieving_scenarios': target_achieving
        },
        'robustness_score': robustness_score,
        'risk_level': 'LOW' if worst_drawdown < 0.10 else 'MODERATE' if worst_drawdown < 0.20 else 'HIGH'
    }
    
    filename = f"strategy_robustness_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Validation results exported: {filename}")
    
    # Final Recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    if robustness_score in ["EXCELLENT", "GOOD"] and worst_drawdown < 0.20:
        print("   ✅ STRATEGY VALIDATED FOR DEPLOYMENT")
        print("   The strategy demonstrates robust performance across various market conditions")
        print("   with acceptable risk levels. Ready for live trading implementation.")
    else:
        print("   ⚠️ STRATEGY NEEDS REFINEMENT")
        print("   Consider adjusting parameters for better stability across market conditions.")
    
    return validation_results


if __name__ == "__main__":
    main()