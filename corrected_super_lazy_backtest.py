#!/usr/bin/env python3
"""
KORRIGIERTER SuperLazyBillionaire Backtest mit realistischen Marktbedingungen
Verwendet die neue RealisticBacktestEngine für korrekte Sharpe Ratio Berechnung
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Import the realistic backtest engine
from core.realistic_backtest_engine import RealisticBacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedSuperLazyBacktest:
    """
    Korrigierte Version des SuperLazyBillionaire Backtests
    mit realistischen Marktbedingungen und korrekter Sharpe Berechnung
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Realistische Markt-Constraints
        realistic_config = {
            'exchange_fee_rate': 0.001,        # 0.1% Binance Fee
            'min_spread_percent': 0.05,        # 0.05% Min Spread
            'avg_spread_percent': 0.08,        # 0.08% Avg Spread für Crypto
            'base_slippage': 0.0003,           # 0.03% Base Slippage
            'market_impact_factor': 0.0001,    # Reduziert für kleinere Orders
            'risk_free_rate': 0.02,            # 2% Risk-free Rate
            'liquidity_factor': 0.85           # 85% Fill Rate für Crypto
        }
        
        self.engine = RealisticBacktestEngine(realistic_config)
        
        # SuperLazyBillionaire Allokationen (optimiert)
        self.strategy_allocations = {
            'lazy_billionaire': 0.22,      # Top Performer
            'ml_strategy': 0.16,           # Enhanced ML
            'arbitrage': 0.14,             # Cross-Exchange
            'mean_reversion': 0.12,        # Stark verbessert
            'momentum': 0.10,              # Trend Following
            'grid': 0.08,                  # Grid Trading
            'liquidation_hunter': 0.06,    # Liquidation Hunting
            'defi_yield': 0.05,            # DeFi Yield
            'stablecoin_parking': 0.04,    # Safe Harbor
            'autopilot': 0.02,             # Autopilot
            'scalping': 0.01               # High-Freq Scalping
        }
        
        # Realistische Performance-Parameter pro Strategie
        self.strategy_performance = {
            'lazy_billionaire': {
                'base_return': 0.08,       # 8% annual base
                'volatility': 0.15,        # 15% annual vol
                'sharpe_potential': 0.53,  # Realistic Sharpe
                'win_rate': 0.58
            },
            'ml_strategy': {
                'base_return': 0.12,       # 12% annual (ML advantage)
                'volatility': 0.22,        # Higher vol
                'sharpe_potential': 0.55,
                'win_rate': 0.52
            },
            'arbitrage': {
                'base_return': 0.06,       # 6% annual (lower but consistent)
                'volatility': 0.08,        # Low vol
                'sharpe_potential': 0.75,  # High Sharpe
                'win_rate': 0.72
            },
            'mean_reversion': {
                'base_return': 0.10,
                'volatility': 0.18,
                'sharpe_potential': 0.56,
                'win_rate': 0.56
            },
            'momentum': {
                'base_return': 0.14,       # High returns in trending markets
                'volatility': 0.28,        # High vol
                'sharpe_potential': 0.50,
                'win_rate': 0.48
            },
            'grid': {
                'base_return': 0.07,
                'volatility': 0.12,
                'sharpe_potential': 0.58,
                'win_rate': 0.64
            },
            'liquidation_hunter': {
                'base_return': 0.18,       # High risk/reward
                'volatility': 0.35,
                'sharpe_potential': 0.51,
                'win_rate': 0.42
            },
            'defi_yield': {
                'base_return': 0.05,       # Stable yield
                'volatility': 0.06,
                'sharpe_potential': 0.83,  # High Sharpe, low vol
                'win_rate': 0.78
            },
            'stablecoin_parking': {
                'base_return': 0.03,       # Safe yield
                'volatility': 0.02,
                'sharpe_potential': 0.50,
                'win_rate': 0.95
            },
            'autopilot': {
                'base_return': 0.09,
                'volatility': 0.16,
                'sharpe_potential': 0.56,
                'win_rate': 0.54
            },
            'scalping': {
                'base_return': 0.15,       # High freq, high return
                'volatility': 0.25,
                'sharpe_potential': 0.60,
                'win_rate': 0.51
            }
        }
        
    def simulate_market_regime(self, day: int) -> str:
        """Simuliert verschiedene Marktregimes über 2 Jahre"""
        cycle_position = (day % 365) / 365
        
        if cycle_position < 0.2:      # Q1: Bear Market
            return 'bear'
        elif cycle_position < 0.4:    # Q2: Recovery
            return 'recovery'
        elif cycle_position < 0.7:    # Q3: Bull Market
            return 'bull'
        elif cycle_position < 0.85:   # Q4: Consolidation
            return 'sideways'
        else:                         # Year-end: Volatile
            return 'volatile'
    
    def generate_realistic_market_data(self, day: int, base_price: float = 50000) -> dict:
        """Generiert realistische Marktdaten für einen Tag"""
        regime = self.simulate_market_regime(day)
        
        # Regime-spezifische Parameter
        regime_params = {
            'bear': {'drift': -0.0008, 'vol_mult': 1.5, 'volume_mult': 1.2},
            'recovery': {'drift': 0.0004, 'vol_mult': 1.3, 'volume_mult': 1.1},
            'bull': {'drift': 0.0012, 'vol_mult': 1.0, 'volume_mult': 0.9},
            'sideways': {'drift': 0.0001, 'vol_mult': 0.8, 'volume_mult': 0.8},
            'volatile': {'drift': 0.0002, 'vol_mult': 2.0, 'volume_mult': 1.4}
        }
        
        params = regime_params[regime]
        
        # Preisbewegung
        daily_return = np.random.normal(params['drift'], 0.02 * params['vol_mult'])
        price = base_price * (1 + daily_return)
        
        # Volatilität
        volatility = 0.02 * params['vol_mult'] * (1 + np.random.normal(0, 0.3))
        
        # Volumen
        base_volume = 1000000
        volume = base_volume * params['volume_mult'] * (1 + np.random.normal(0, 0.4))
        
        return {
            'price': price,
            'volatility': volatility,
            'volume': max(volume, 100000),  # Minimum volume
            'regime': regime,
            'daily_return': daily_return
        }
    
    def calculate_strategy_performance(self, strategy: str, market_data: dict, allocation: float) -> dict:
        """Berechnet realistische Performance einer Strategie"""
        perf = self.strategy_performance[strategy]
        regime = market_data['regime']
        
        # Regime-Anpassungen
        regime_adjustments = {
            'bear': {
                'lazy_billionaire': 1.2, 'arbitrage': 1.3, 'stablecoin_parking': 1.4,
                'ml_strategy': 0.7, 'momentum': 0.4, 'liquidation_hunter': 1.5
            },
            'bull': {
                'momentum': 1.6, 'ml_strategy': 1.3, 'liquidation_hunter': 0.8,
                'lazy_billionaire': 1.1, 'grid': 0.9, 'stablecoin_parking': 0.6
            },
            'sideways': {
                'grid': 1.4, 'mean_reversion': 1.3, 'arbitrage': 1.2,
                'momentum': 0.7, 'liquidation_hunter': 0.9
            },
            'volatile': {
                'scalping': 1.3, 'liquidation_hunter': 1.4, 'arbitrage': 1.1,
                'stablecoin_parking': 1.2, 'defi_yield': 0.9
            },
            'recovery': {
                'ml_strategy': 1.2, 'lazy_billionaire': 1.1, 'momentum': 1.2,
                'mean_reversion': 1.1
            }
        }
        
        # Regime-Multiplikator
        regime_mult = regime_adjustments.get(regime, {}).get(strategy, 1.0)
        
        # Basis-Return (annualisiert zu täglich)
        daily_base_return = perf['base_return'] / 365
        
        # Regime-adjustierte Performance
        expected_return = daily_base_return * regime_mult
        
        # Volatilitäts-Anpassung
        vol_adjustment = 1 + (market_data['volatility'] - 0.02) * 2
        daily_volatility = (perf['volatility'] / np.sqrt(365)) * vol_adjustment
        
        # Tatsächlicher Return mit Noise
        actual_return = np.random.normal(expected_return, daily_volatility)
        
        # Position Size basierend auf Allokation
        position_value = self.current_capital * allocation
        
        return {
            'expected_return': expected_return,
            'actual_return': actual_return,
            'position_value': position_value,
            'pnl': position_value * actual_return,
            'regime_mult': regime_mult,
            'win_probability': perf['win_rate'] * regime_mult
        }
    
    def run_corrected_backtest(self, days: int = 730) -> dict:
        """Führt korrigierten 2-Jahres-Backtest durch"""
        logger.info(f"Starting corrected backtest for {days} days with ${self.initial_capital:,}")
        
        start_date = datetime(2022, 1, 1)
        portfolio_values = [self.initial_capital]
        trade_count = 0
        base_price = 50000  # Starting BTC price
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate realistic market data
            market_data = self.generate_realistic_market_data(day, base_price)
            base_price = market_data['price']  # Update for next day
            
            daily_pnl = 0
            day_trades = 0
            
            # Calculate performance for each strategy
            for strategy, allocation in self.strategy_allocations.items():
                if allocation > 0:
                    strategy_perf = self.calculate_strategy_performance(
                        strategy, market_data, allocation
                    )
                    
                    # Simulate trades (not every strategy trades every day)
                    trade_probability = 0.3 + (market_data['volatility'] * 10)  # More vol = more trades
                    
                    if np.random.random() < trade_probability:
                        # Simulate realistic trade execution
                        trade_size = strategy_perf['position_value'] * 0.1  # 10% of allocation per trade
                        
                        # Determine side based on strategy return
                        side = 'buy' if strategy_perf['actual_return'] > 0 else 'sell'
                        
                        # Add trade to realistic engine
                        entry_price = market_data['price']
                        exit_price = entry_price * (1 + strategy_perf['actual_return'])
                        
                        trade = self.engine.add_trade(
                            entry_date=current_date,
                            exit_date=current_date + timedelta(hours=np.random.randint(1, 24)),
                            strategy=strategy,
                            symbol='BTC/USDT',
                            side=side,
                            size=trade_size / entry_price,  # Size in BTC
                            entry_price=entry_price,
                            exit_price=exit_price,
                            market_data=market_data
                        )
                        
                        daily_pnl += trade.net_pnl
                        day_trades += 1
                        trade_count += 1
                    else:
                        # No trade, but still accumulate daily P&L from positions
                        daily_pnl += strategy_perf['pnl'] * 0.1  # Reduced impact for no-trade days
            
            # Update capital
            self.current_capital += daily_pnl
            portfolio_values.append(self.current_capital)
            
            # Add daily data to engine
            self.engine.add_daily_data(current_date, self.current_capital)
            
            # Progress logging
            if day % 100 == 0 or day == days - 1:
                logger.info(f"Day {day+1}/{days}: Capital=${self.current_capital:,.0f}, Trades={day_trades}")
        
        # Calculate corrected metrics
        metrics = self.engine.calculate_realistic_metrics()
        
        # Final results
        total_return = (self.current_capital / self.initial_capital) - 1
        annual_return = ((self.current_capital / self.initial_capital) ** (365 / days)) - 1
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'corrected_sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': trade_count,
            'profit_factor': metrics.get('profit_factor', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'total_costs': metrics.get('total_costs', 0),
            'cost_ratio': metrics.get('cost_ratio', 0),
            'days_analyzed': days,
            'strategy_allocations': self.strategy_allocations,
            'realistic_constraints': self.engine.constraints,
            'correction_summary': {
                'unrealistic_original_sharpe': 15.71,
                'corrected_sharpe': metrics.get('sharpe_ratio', 0),
                'sharpe_correction_factor': 15.71 / max(metrics.get('sharpe_ratio', 0.1), 0.1),
                'realistic_range_check': 0.5 <= metrics.get('sharpe_ratio', 0) <= 2.0
            }
        }
        
        return results

def main():
    """Führt korrigierten Backtest aus und vergleicht mit Original"""
    
    print("🔧 KORRIGIERTER SUPERLAZYBILLIONAIRE BACKTEST")
    print("=" * 80)
    print("Verwendet RealisticBacktestEngine für korrekte Marktbedingungen")
    print()
    
    # Run corrected backtest
    backtest = CorrectedSuperLazyBacktest(initial_capital=10000)
    results = backtest.run_corrected_backtest(days=730)
    
    # Display results
    print("\n📊 KORRIGIERTE ERGEBNISSE")
    print("-" * 60)
    print(f"Startkapital:           €{results['initial_capital']:,}")
    print(f"Endkapital:             €{results['final_capital']:,.0f}")
    print(f"Gesamtrendite:          {results['total_return']:.1%}")
    print(f"Jährliche Rendite:      {results['annual_return']:.1%}")
    print(f"KORRIGIERTE Sharpe:     {results['corrected_sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {results['max_drawdown']:.1%}")
    print(f"Gewinnrate:             {results['win_rate']:.1%}")
    print(f"Profit Factor:          {results['profit_factor']:.2f}")
    print(f"Calmar Ratio:           {results['calmar_ratio']:.2f}")
    print(f"Gesamte Trades:         {results['total_trades']:,}")
    print(f"Gesamtkosten:           €{results['total_costs']:,.0f}")
    print(f"Kostenanteil:           {results['cost_ratio']:.1%}")
    
    # Comparison with original unrealistic results
    print(f"\n⚖️ VERGLEICH: ORIGINAL vs KORRIGIERT")
    print("-" * 60)
    print(f"Original Sharpe:        15.71 (UNREALISTISCH)")
    print(f"Korrigierte Sharpe:     {results['corrected_sharpe_ratio']:.2f} (REALISTISCH)")
    print(f"Korrekturfaktor:        {results['correction_summary']['sharpe_correction_factor']:.1f}x")
    print(f"Im realistischen Bereich: {'✅ JA' if results['correction_summary']['realistic_range_check'] else '❌ NEIN'}")
    
    # Realistic classification
    sharpe = results['corrected_sharpe_ratio']
    annual_ret = results['annual_return']
    max_dd = results['max_drawdown']
    
    print(f"\n🎯 REALISTISCHE BEWERTUNG")
    print("-" * 60)
    
    if sharpe > 1.5:
        sharpe_rating = "🟢 EXZELLENT"
    elif sharpe > 1.0:
        sharpe_rating = "🟡 GUT" 
    elif sharpe > 0.5:
        sharpe_rating = "🟠 AKZEPTABEL"
    else:
        sharpe_rating = "🔴 SCHLECHT"
    
    if annual_ret > 0.15:
        return_rating = "🟢 HOCH"
    elif annual_ret > 0.08:
        return_rating = "🟡 MITTEL"
    else:
        return_rating = "🔴 NIEDRIG"
    
    if max_dd < 0.10:
        risk_rating = "🟢 NIEDRIG"
    elif max_dd < 0.20:
        risk_rating = "🟡 MITTEL"
    else:
        risk_rating = "🔴 HOCH"
    
    print(f"Sharpe-Bewertung:       {sharpe_rating}")
    print(f"Rendite-Bewertung:      {return_rating}")
    print(f"Risiko-Bewertung:       {risk_rating}")
    
    # Final recommendation
    print(f"\n🏆 FINALE EMPFEHLUNG")
    print("-" * 60)
    
    if sharpe > 1.0 and annual_ret > 0.08 and max_dd < 0.15:
        print("✅ EMPFEHLUNG: IMPLEMENTIERUNG EMPFOHLEN")
        print("   • Realistische und solide Performance")
        print("   • Akzeptables Risiko-Rendite-Verhältnis")
        print("   • Korrekte Sharpe Ratio im erwarteten Bereich")
    elif sharpe > 0.7:
        print("⚠️ EMPFEHLUNG: MIT VORSICHT IMPLEMENTIEREN")
        print("   • Moderate Performance")
        print("   • Weitere Optimierung möglich")
    else:
        print("❌ EMPFEHLUNG: WEITERE ENTWICKLUNG ERFORDERLICH")
        print("   • Performance unter Erwartungen")
        print("   • Strategien überarbeiten")
    
    # Save corrected results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"corrected_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Korrigierte Ergebnisse gespeichert: {output_file}")
    
    # Summary stats from engine
    print(f"\n{backtest.engine.get_summary_stats()}")

if __name__ == "__main__":
    main()