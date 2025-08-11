#!/usr/bin/env python3
"""
Strategies Performance Comparison
=================================

Vergleicht die Performance der neuen defensiven Strategien 
mit existierenden Strategien im Projekt.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_defensive_results():
    """Lädt die Ergebnisse der defensiven Strategien"""
    try:
        # Try to find the latest defensive backtest results
        with open('simple_defensive_backtest_20250724_151134.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open('simple_defensive_backtest_20250724_145958.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Defensive strategy results nicht gefunden")
            return {}


def load_existing_results():
    """Lädt existierende Strategie-Ergebnisse"""
    existing_results = {}
    
    # Realistic backtest results
    try:
        with open('realistic_backtest_results_20250721_222434.json', 'r') as f:
            realistic_data = json.load(f)
            if 'Conservative Test' in realistic_data:
                existing_results['Conservative Strategy'] = {
                    'total_return': realistic_data['Conservative Test']['results']['metrics']['total_return'],
                    'annual_return': realistic_data['Conservative Test']['results']['metrics']['annual_return'],
                    'sharpe_ratio': realistic_data['Conservative Test']['results']['metrics']['sharpe_ratio'],
                    'max_drawdown': realistic_data['Conservative Test']['results']['metrics']['max_drawdown'],
                    'volatility': realistic_data['Conservative Test']['results']['metrics']['volatility'],
                    'total_trades': realistic_data['Conservative Test']['results']['metrics']['total_trades'],
                    'win_rate': realistic_data['Conservative Test']['results']['metrics']['win_rate']
                }
    except FileNotFoundError:
        logger.warning("Realistic backtest results nicht gefunden")
    
    # Profitable strategy results
    try:
        with open('profitable_strategy_results_20250721_214554.json', 'r') as f:
            profitable_data = json.load(f)
            if 'performance_summary' in profitable_data:
                existing_results['Profitable BTC Strategy'] = {
                    'total_return': profitable_data['performance_summary'].get('total_return', 0),
                    'annual_return': profitable_data['performance_summary'].get('annual_return', 0),
                    'sharpe_ratio': profitable_data['performance_summary'].get('sharpe_ratio', 0),
                    'max_drawdown': profitable_data['performance_summary'].get('max_drawdown', 0),
                    'volatility': profitable_data['performance_summary'].get('volatility', 0),
                    'total_trades': profitable_data['performance_summary'].get('total_trades', 0),
                    'win_rate': profitable_data['performance_summary'].get('win_rate', 0)
                }
    except FileNotFoundError:
        logger.warning("Profitable strategy results nicht gefunden")
    
    # Ultimate BTC strategy results
    try:
        with open('ultimate_btc_strategy_results_20250720_225105.json', 'r') as f:
            ultimate_data = json.load(f)
            if 'performance_metrics' in ultimate_data:
                existing_results['Ultimate BTC Strategy'] = {
                    'total_return': ultimate_data['performance_metrics'].get('total_return', 0),
                    'annual_return': ultimate_data['performance_metrics'].get('annual_return', 0),
                    'sharpe_ratio': ultimate_data['performance_metrics'].get('sharpe_ratio', 0),
                    'max_drawdown': ultimate_data['performance_metrics'].get('max_drawdown', 0),
                    'volatility': ultimate_data['performance_metrics'].get('volatility', 0),
                    'total_trades': ultimate_data['performance_metrics'].get('total_trades', 0),
                    'win_rate': ultimate_data['performance_metrics'].get('win_rate', 0)
                }
    except FileNotFoundError:
        logger.warning("Ultimate BTC strategy results nicht gefunden")
    
    return existing_results


def normalize_defensive_results(defensive_results):
    """Normalisiert die defensiven Strategie-Ergebnisse für Vergleich"""
    normalized = {}
    
    for strategy_name, results in defensive_results.items():
        normalized[results['strategy']] = {
            'total_return': results['total_return'],
            'annual_return': results['annualized_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'volatility': results['volatility'],
            'total_trades': results['total_trades'],
            'win_rate': 0.5,  # Nicht direkt verfügbar, Schätzung
            'total_fees': results['total_fees']
        }
    
    return normalized


def create_comparison_report(defensive_results, existing_results):
    """Erstellt umfassenden Vergleichsbericht"""
    
    print("="*80)
    print("🔍 TRADING STRATEGIES PERFORMANCE COMPARISON")
    print("="*80)
    
    # Alle Strategien kombinieren
    all_strategies = {}
    all_strategies.update(defensive_results)
    all_strategies.update(existing_results)
    
    if not all_strategies:
        print("❌ Keine Strategien zum Vergleichen gefunden")
        return
    
    # Performance DataFrame erstellen
    df_data = []
    for strategy_name, metrics in all_strategies.items():
        df_data.append({
            'Strategy': strategy_name,
            'Total Return': metrics.get('total_return', 0),
            'Annual Return': metrics.get('annual_return', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0),
            'Volatility': metrics.get('volatility', 0),
            'Total Trades': metrics.get('total_trades', 0),
            'Win Rate': metrics.get('win_rate', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    print("\n📊 PERFORMANCE OVERVIEW")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Total Return':<12} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Strategy']:<25} {row['Total Return']:>10.2%} "
              f"{row['Sharpe Ratio']:>7.2f} {row['Max Drawdown']:>7.2%} "
              f"{row['Total Trades']:>7.0f}")
    
    print("\n🏆 RANKINGS")
    print("-" * 50)
    
    # Rankings
    rankings = {
        'Total Return': df.nlargest(5, 'Total Return'),
        'Sharpe Ratio': df.nlargest(5, 'Sharpe Ratio'),
        'Low Drawdown': df.nsmallest(5, 'Max Drawdown'),
        'Risk-Adjusted Return': df.assign(
            risk_adj=df['Total Return'] / (df['Max Drawdown'] + 0.01)
        ).nlargest(5, 'risk_adj')
    }
    
    for metric, top_strategies in rankings.items():
        print(f"\n🥇 Best {metric}:")
        for i, (_, row) in enumerate(top_strategies.head(3).iterrows(), 1):
            if metric == 'Total Return':
                value = f"{row['Total Return']:.2%}"
            elif metric == 'Sharpe Ratio':
                value = f"{row['Sharpe Ratio']:.2f}"
            elif metric == 'Low Drawdown':
                value = f"{row['Max Drawdown']:.2%}"
            else:
                value = f"{row['Total Return'] / (row['Max Drawdown'] + 0.01):.2f}"
            
            print(f"   {i}. {row['Strategy']}: {value}")
    
    # Defensive Strategies Analysis
    print("\n🛡️  DEFENSIVE STRATEGIES ANALYSIS")
    print("-" * 50)
    
    defensive_names = ['Advanced Portfolio', 'Defensive Volatility', 'Smart Rebalancing']
    defensive_df = df[df['Strategy'].isin(defensive_names)]
    
    if not defensive_df.empty:
        print("\nDefensive Strategies Performance:")
        for _, row in defensive_df.iterrows():
            print(f"\n📈 {row['Strategy']}:")
            print(f"   • Total Return: {row['Total Return']:.2%}")
            print(f"   • Sharpe Ratio: {row['Sharpe Ratio']:.2f}")
            print(f"   • Max Drawdown: {row['Max Drawdown']:.2%}")
            print(f"   • Volatility: {row['Volatility']:.2%}")
            print(f"   • Trades Executed: {row['Total Trades']:.0f}")
        
        # Best defensive strategy
        best_defensive = defensive_df.loc[defensive_df['Total Return'].idxmax()]
        print(f"\n🏆 Best Defensive Strategy: {best_defensive['Strategy']}")
        print(f"   → Return: {best_defensive['Total Return']:.2%}")
        print(f"   → Sharpe: {best_defensive['Sharpe Ratio']:.2f}")
        print(f"   → Max DD: {best_defensive['Max Drawdown']:.2%}")
    
    # Risk-Return Analysis
    print("\n📊 RISK-RETURN ANALYSIS")
    print("-" * 50)
    
    # Kategorisierung
    categories = {
        'Conservative': [],
        'Moderate': [],
        'Aggressive': []
    }
    
    for _, row in df.iterrows():
        if row['Volatility'] < 0.15 and row['Max Drawdown'] < 0.10:
            categories['Conservative'].append(row['Strategy'])
        elif row['Volatility'] < 0.30 and row['Max Drawdown'] < 0.20:
            categories['Moderate'].append(row['Strategy'])
        else:
            categories['Aggressive'].append(row['Strategy'])
    
    for category, strategies in categories.items():
        if strategies:
            print(f"\n{category} Strategies:")
            for strategy in strategies:
                strategy_data = df[df['Strategy'] == strategy].iloc[0]
                print(f"   • {strategy}: {strategy_data['Total Return']:.2%} return, "
                      f"{strategy_data['Max Drawdown']:.2%} max DD")
    
    # Empfehlungen
    print("\n💡 STRATEGY RECOMMENDATIONS")
    print("-" * 50)
    
    # Beste für verschiedene Profile
    conservative_strategies = df[
        (df['Max Drawdown'] < 0.10) & (df['Sharpe Ratio'] > 0)
    ].nlargest(1, 'Sharpe Ratio')
    
    if not conservative_strategies.empty:
        best_conservative = conservative_strategies.iloc[0]
        print(f"\n🛡️  For Conservative Investors:")
        print(f"   Recommended: {best_conservative['Strategy']}")
        print(f"   → Stable returns with low risk")
        print(f"   → Return: {best_conservative['Total Return']:.2%}")
        print(f"   → Max Drawdown: {best_conservative['Max Drawdown']:.2%}")
    
    high_return_strategies = df[df['Total Return'] > 0.05].nlargest(1, 'Total Return')
    
    if not high_return_strategies.empty:
        best_growth = high_return_strategies.iloc[0]
        print(f"\n📈 For Growth Investors:")
        print(f"   Recommended: {best_growth['Strategy']}")
        print(f"   → Higher returns with managed risk")
        print(f"   → Return: {best_growth['Total Return']:.2%}")
        print(f"   → Sharpe Ratio: {best_growth['Sharpe Ratio']:.2f}")
    
    # Portfolio Allocation Suggestion
    if len(defensive_df) >= 2:
        print(f"\n🎯 PORTFOLIO ALLOCATION SUGGESTION")
        print("-" * 50)
        print("For a balanced approach, consider combining:")
        
        # Top 2 defensive strategies
        top_defensive = defensive_df.nlargest(2, 'Sharpe Ratio')
        total_allocation = 0
        
        for i, (_, strategy) in enumerate(top_defensive.iterrows()):
            allocation = 60 - i * 20  # 60%, 40%
            total_allocation += allocation
            print(f"   • {strategy['Strategy']}: {allocation}%")
            print(f"     → Expected return: {strategy['Total Return'] * allocation / 100:.2%}")
        
        if total_allocation < 100:
            print(f"   • Cash/Stable Assets: {100 - total_allocation}%")
    
    # Summary
    print("\n📋 EXECUTIVE SUMMARY")
    print("-" * 50)
    
    total_strategies = len(df)
    profitable_strategies = len(df[df['Total Return'] > 0])
    
    print(f"   • Total Strategies Analyzed: {total_strategies}")
    print(f"   • Profitable Strategies: {profitable_strategies} ({profitable_strategies/total_strategies:.1%})")
    
    if not defensive_df.empty:
        avg_defensive_return = defensive_df['Total Return'].mean()
        avg_defensive_sharpe = defensive_df['Sharpe Ratio'].mean()
        avg_defensive_dd = defensive_df['Max Drawdown'].mean()
        
        print(f"   • Defensive Strategies Average Return: {avg_defensive_return:.2%}")
        print(f"   • Defensive Strategies Average Sharpe: {avg_defensive_sharpe:.2f}")
        print(f"   • Defensive Strategies Average Max DD: {avg_defensive_dd:.2%}")
    
    # Best overall strategy
    if not df.empty:
        # Composite Score: Return * Sharpe / (1 + Max Drawdown)
        df['Composite Score'] = (df['Total Return'] * df['Sharpe Ratio']) / (1 + df['Max Drawdown'])
        best_overall = df.loc[df['Composite Score'].idxmax()]
        
        print(f"\n🎖️  BEST OVERALL STRATEGY: {best_overall['Strategy']}")
        print(f"   → Balanced performance across all metrics")
        print(f"   → Total Return: {best_overall['Total Return']:.2%}")
        print(f"   → Sharpe Ratio: {best_overall['Sharpe Ratio']:.2f}")
        print(f"   → Max Drawdown: {best_overall['Max Drawdown']:.2%}")
    
    print("\n" + "="*80)
    
    return df


def save_comparison_report(df, defensive_results, existing_results):
    """Speichert den Vergleichsbericht"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategies_performance_comparison_{timestamp}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_summary': {
            'total_strategies_analyzed': len(df),
            'profitable_strategies': len(df[df['Total Return'] > 0]),
            'best_strategy_by_return': df.loc[df['Total Return'].idxmax(), 'Strategy'] if not df.empty else None,
            'best_strategy_by_sharpe': df.loc[df['Sharpe Ratio'].idxmax(), 'Strategy'] if not df.empty else None,
            'lowest_drawdown_strategy': df.loc[df['Max Drawdown'].idxmin(), 'Strategy'] if not df.empty else None
        },
        'defensive_strategies_performance': defensive_results,
        'existing_strategies_performance': existing_results,
        'detailed_comparison': df.to_dict('records') if not df.empty else []
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Vergleichsbericht gespeichert in {filename}")
    return filename


def main():
    """Hauptfunktion für den Performance-Vergleich"""
    
    logger.info("Starte Performance-Vergleich der Strategien...")
    
    # Lade Ergebnisse
    defensive_results = load_defensive_results()
    existing_results = load_existing_results()
    
    if not defensive_results:
        logger.error("Keine defensiven Strategien gefunden")
        return
    
    # Normalisiere defensive Ergebnisse
    normalized_defensive = normalize_defensive_results(defensive_results)
    
    # Erstelle Vergleichsbericht
    df = create_comparison_report(normalized_defensive, existing_results)
    
    # Speichere Bericht
    if df is not None and not df.empty:
        filename = save_comparison_report(df, normalized_defensive, existing_results)
        print(f"\n💾 Detaillierter Bericht gespeichert: {filename}")
    
    logger.info("Performance-Vergleich abgeschlossen")


if __name__ == "__main__":
    main()