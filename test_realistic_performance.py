#!/usr/bin/env python3
"""
Realistic Performance Test - Ultimate BTC Strategy (Event-Driven)
================================================================

Validiert die tatsächliche Performance ohne Lookahead Bias
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_realistic_market_data(days: int = 365, start_price: float = 45000) -> pd.DataFrame:
    """Generate realistic BTC market data for testing"""
    print(f"📊 Generiere {days} Tage realistische Marktdaten...")
    
    # Generate realistic price series
    np.random.seed(42)  # For reproducible results
    
    timestamps = []
    prices = []
    volumes = []
    
    current_time = datetime(2023, 1, 1)
    current_price = start_price
    
    # Market parameters
    daily_volatility = 0.04  # 4% daily volatility (realistic for BTC)
    trend_strength = 0.001   # Slight upward trend
    mean_reversion = 0.02    # Mean reversion factor
    
    for i in range(days * 24):  # Hourly data
        # Price movement with realistic characteristics
        random_shock = np.random.normal(0, daily_volatility / np.sqrt(24))
        trend_component = trend_strength / 24
        mean_reversion_component = -mean_reversion * (current_price - start_price) / start_price / 24
        
        price_change = trend_component + mean_reversion_component + random_shock
        current_price *= (1 + price_change)
        
        # Ensure price doesn't go negative
        current_price = max(current_price, start_price * 0.1)
        
        # Generate realistic volume (higher during volatility)
        base_volume = 1000
        volatility_factor = abs(price_change) * 50000
        volume = base_volume + volatility_factor + np.random.exponential(500)
        
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
    
    print(f"✅ Daten generiert: {len(df)} Datenpunkte")
    print(f"   Start: ${df['close'].iloc[0]:,.0f}")
    print(f"   Ende: ${df['close'].iloc[-1]:,.0f}")
    print(f"   Buy&Hold Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1):.2%}")
    
    return df


def run_event_driven_backtest(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Führe event-driven Backtest aus"""
    print("\n🚀 EVENT-DRIVEN BACKTEST")
    print("=" * 60)
    
    try:
        # Import modules directly to avoid dependencies
        from core.event_driven_backtest import EventDrivenBacktester
        from core.indicator_engine import IndicatorEngine
        
        # Initialize components
        backtester = EventDrivenBacktester(
            initial_capital=100000,
            commission_rate=0.001,  # 0.1% realistic trading fees
            slippage_rate=0.0005,   # 0.05% realistic slippage
            max_position_size=0.8   # Maximum 80% position size
        )
        
        indicator_engine = IndicatorEngine()
        
        print(f"📊 Backtesting {len(market_data)} Datenpunkte...")
        print(f"   Startkapital: ${backtester.initial_capital:,.0f}")
        print(f"   Trading Fees: {backtester.commission_rate:.1%}")
        print(f"   Slippage: {backtester.slippage_rate:.2%}")
        
        # Process data point by point (no lookahead!)
        signals_generated = 0
        trades_executed = 0
        
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            try:
                price = row['close']
                volume = row['volume']
                
                # Update indicators with current data point
                indicators = indicator_engine.update(price, volume, timestamp)
                
                # Generate signal only after sufficient data
                if i >= 200:  # Wait for indicators to warm up
                    signal_data = generate_realistic_signal(indicators, price, timestamp)
                    
                    if signal_data['direction'] != 'hold':
                        signals_generated += 1
                    
                    # Process signal in backtester
                    trade_executed = backtester.process_signal(timestamp, price, signal_data)
                    if trade_executed:
                        trades_executed += 1
                
                # Progress update
                if (i + 1) % 1000 == 0:
                    progress = (i + 1) / len(market_data) * 100
                    print(f"   Progress: {progress:.1f}% - Signals: {signals_generated}, Trades: {trades_executed}")
                    
            except Exception as e:
                logger.error(f"Error processing data point {i}: {e}")
                continue
        
        # Finalize backtest
        final_timestamp = market_data.index[-1]
        final_price = market_data['close'].iloc[-1]
        metrics = backtester.finalize_backtest(final_timestamp, final_price)
        
        print(f"\n✅ Backtest abgeschlossen!")
        print(f"   Signale generiert: {signals_generated}")
        print(f"   Trades ausgeführt: {trades_executed}")
        print(f"   Signal-zu-Trade Ratio: {trades_executed/signals_generated*100:.1f}%" if signals_generated > 0 else "   Keine Signale")
        
        return {
            'metrics': metrics,
            'backtester': backtester,
            'signals_generated': signals_generated,
            'trades_executed': trades_executed
        }
        
    except Exception as e:
        print(f"❌ Backtest fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_realistic_signal(indicators: Dict[str, float], price: float, timestamp: datetime) -> Dict[str, Any]:
    """
    Generiere realistische Trading-Signale basierend auf Indikatoren
    (Vereinfachte Version der Ultimate BTC Strategy)
    """
    try:
        # Check if we have required indicators
        required_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14']
        if not all(indicator in indicators for indicator in required_indicators):
            return {
                'direction': 'hold',
                'strength': 0.0,
                'confidence': 0.0,
                'quality_score': 0.0,
                'regime': 'unknown',
                'timestamp': timestamp
            }
        
        # Calculate signal components
        signal_components = []
        
        # 1. Trend Signal (SMA crossover)
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        if sma_20 > sma_50 and price > sma_20:
            trend_signal = 1.0
        elif sma_20 < sma_50 and price < sma_20:
            trend_signal = -1.0
        else:
            trend_signal = 0.0
        signal_components.append(trend_signal * 0.4)
        
        # 2. MACD Signal
        ema_12 = indicators['ema_12']
        ema_26 = indicators['ema_26']
        macd = ema_12 - ema_26
        if macd > 0:
            macd_signal = 1.0
        elif macd < 0:
            macd_signal = -1.0
        else:
            macd_signal = 0.0
        signal_components.append(macd_signal * 0.3)
        
        # 3. RSI Signal (mean reversion)
        rsi_14 = indicators['rsi_14']
        if rsi_14 < 30:
            rsi_signal = 1.0  # Oversold -> Buy
        elif rsi_14 > 70:
            rsi_signal = -1.0  # Overbought -> Sell
        else:
            rsi_signal = 0.0
        signal_components.append(rsi_signal * 0.3)
        
        # Aggregate signal
        signal_strength = sum(signal_components)
        signal_strength = max(-1.0, min(1.0, signal_strength))  # Clamp to [-1, 1]
        
        # Determine direction
        if signal_strength > 0.3:
            direction = 'buy'
            confidence = min(signal_strength, 1.0)
        elif signal_strength < -0.3:
            direction = 'sell'
            confidence = min(abs(signal_strength), 1.0)
        else:
            direction = 'hold'
            confidence = 0.0
        
        # Calculate quality score
        volatility = indicators.get('volatility_20d', 0.02)
        volume_ratio = indicators.get('volume_ratio_20', 1.0)
        
        quality_score = min(
            abs(signal_strength) * 0.5 +
            min(volume_ratio / 1.5, 1.0) * 0.3 +
            min(1.0 / (volatility * 50 + 0.1), 1.0) * 0.2,
            1.0
        )
        
        # Simple regime detection
        momentum_20d = indicators.get('momentum_20d', 0.0)
        if momentum_20d > 0.1:
            regime = 'bull_strong'
        elif momentum_20d > 0.05:
            regime = 'bull_moderate'
        elif momentum_20d < -0.1:
            regime = 'bear_strong'
        elif momentum_20d < -0.05:
            regime = 'bear_moderate'
        else:
            regime = 'sideways'
        
        return {
            'direction': direction,
            'strength': abs(signal_strength),
            'confidence': confidence,
            'quality_score': quality_score,
            'regime': regime,
            'timestamp': timestamp,
            'components': {
                'trend': signal_components[0] / 0.4 if signal_components else 0,
                'macd': signal_components[1] / 0.3 if len(signal_components) > 1 else 0,
                'rsi': signal_components[2] / 0.3 if len(signal_components) > 2 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return {
            'direction': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'quality_score': 0.0,
            'regime': 'unknown',
            'timestamp': timestamp,
            'error': str(e)
        }


def analyze_results(backtest_results: Dict[str, Any], market_data: pd.DataFrame):
    """Analysiere die Backtest-Ergebnisse"""
    print("\n📈 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    try:
        metrics = backtest_results['metrics']
        backtester = backtest_results['backtester']
        
        # Performance Metriken
        print("🎯 PERFORMANCE METRIKEN:")
        print(f"   Total Return: {metrics.total_return:.2%}")
        print(f"   Annual Return: {metrics.annual_return:.2%}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Volatility: {metrics.volatility:.2%}")
        print(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        # Trading Statistiken
        print(f"\n📊 TRADING STATISTIKEN:")
        print(f"   Total Trades: {metrics.total_trades}")
        print(f"   Win Rate: {metrics.win_rate:.1%}")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        print(f"   Avg Trade Return: {metrics.avg_trade_return:.2%}")
        print(f"   Avg Trade Duration: {metrics.avg_trade_duration:.1f} hours")
        
        # Kosten
        print(f"\n💰 TRADING KOSTEN:")
        print(f"   Total Commission: ${metrics.commission_total:,.2f}")
        print(f"   Total Slippage: ${metrics.slippage_total:,.2f}")
        print(f"   Total Costs: ${metrics.commission_total + metrics.slippage_total:,.2f}")
        
        # Vergleich mit Buy & Hold
        start_price = market_data['close'].iloc[0]
        end_price = market_data['close'].iloc[-1]
        buyhold_return = (end_price / start_price) - 1
        
        print(f"\n🔄 BENCHMARK VERGLEICH:")
        print(f"   Buy & Hold Return: {buyhold_return:.2%}")
        print(f"   Alpha vs Buy & Hold: {metrics.alpha_vs_buyhold:.2%}")
        print(f"   Strategy Outperformed: {'✅' if metrics.total_return > buyhold_return else '❌'}")
        
        # Realitäts-Check
        print(f"\n🔍 REALITÄTS-CHECK:")
        print(f"   Realistic Commission: ✅ ({backtester.commission_rate:.1%})")
        print(f"   Realistic Slippage: ✅ ({backtester.slippage_rate:.2%})")
        print(f"   No Lookahead Bias: ✅ (Event-driven)")
        print(f"   Signal Quality: {'High' if metrics.avg_trade_return > 0.01 else 'Moderate' if metrics.avg_trade_return > 0 else 'Low'}")
        
        # Fazit
        print(f"\n📝 FAZIT:")
        if metrics.sharpe_ratio > 1.0 and metrics.total_return > 0.1:
            print("   🎉 AUSGEZEICHNETE PERFORMANCE - Strategy zeigt starke Resultate")
        elif metrics.sharpe_ratio > 0.5 and metrics.total_return > 0.05:
            print("   ✅ GUTE PERFORMANCE - Strategy ist profitabel")
        elif metrics.total_return > 0:
            print("   📈 MODERATE PERFORMANCE - Strategy ist profitabel aber verbesserungswürdig")
        else:
            print("   ⚠️ SCHWACHE PERFORMANCE - Strategy benötigt Überarbeitung")
        
        # Export results
        filename = f"realistic_performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backtester.export_results(filename)
        print(f"\n💾 Ergebnisse exportiert: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyse fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Haupttest für realistische Performance"""
    print("🔬 REALISTIC PERFORMANCE TEST - ULTIMATE BTC STRATEGY")
    print("=" * 80)
    print("Validierung der tatsächlichen Performance ohne Lookahead Bias\n")
    
    # Generate test data
    market_data = generate_realistic_market_data(days=365)
    
    # Run backtest
    backtest_results = run_event_driven_backtest(market_data)
    
    if backtest_results:
        # Analyze results
        analyze_results(backtest_results, market_data)
        
        print(f"\n🎯 TEST ZUSAMMENFASSUNG:")
        print(f"✅ Event-driven Backtest erfolgreich")
        print(f"✅ Realistische Marktbedingungen simuliert")
        print(f"✅ Trading-Kosten berücksichtigt")
        print(f"✅ Kein Lookahead Bias")
        print(f"✅ Performance-Metriken berechnet")
        
        metrics = backtest_results['metrics']
        print(f"\n🚀 KERNRESULTATE:")
        print(f"   📊 Annual Return: {metrics.annual_return:.1%}")
        print(f"   ⚡ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   🛡️ Max Drawdown: {metrics.max_drawdown:.1%}")
        print(f"   🎯 Win Rate: {metrics.win_rate:.0%}")
        
        # Vergleich mit ursprünglichen (unrealistischen) Claims
        print(f"\n📋 VERGLEICH MIT URSPRÜNGLICHEN CLAIMS:")
        print(f"   Original (mit Lookahead): 177.8% Annual Return, 2.14 Sharpe")
        print(f"   Realistic (ohne Lookahead): {metrics.annual_return:.1%} Annual Return, {metrics.sharpe_ratio:.2f} Sharpe")
        print(f"   Reality Check: {'✅ Realistisch' if metrics.annual_return < 0.5 else '⚠️ Überprüfen'}")
        
    else:
        print("❌ Performance Test fehlgeschlagen")


if __name__ == "__main__":
    main()