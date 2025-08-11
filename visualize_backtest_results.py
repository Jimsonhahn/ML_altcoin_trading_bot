#!/usr/bin/env python3
"""
Visualisierung der SuperLazyBillionaire 2-Jahres-Backtest Ergebnisse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_comprehensive_analysis():
    """Erstelle umfassende Analyse-Grafiken"""
    
    # Load latest results
    results_dir = Path("results/super_lazy_backtest")
    
    # Find latest files
    json_files = list(results_dir.glob("backtest_results_*.json"))
    if not json_files:
        print("No backtest results found!")
        return
    
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
    
    # Load data
    with open(latest_json, 'r') as f:
        results = json.load(f)
    
    timestamp = latest_json.stem.split('_')[-1]
    
    # Load portfolio history
    portfolio_file = results_dir / f"portfolio_history_{timestamp}.csv"
    trades_file = results_dir / f"trades_{timestamp}.csv"
    
    if portfolio_file.exists():
        portfolio_df = pd.read_csv(portfolio_file)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    else:
        print("Portfolio history file not found!")
        return
    
    if trades_file.exists():
        trades_df = pd.read_csv(trades_file)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
    else:
        print("Trades file not found!")
        return
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Portfolio Value Over Time
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(portfolio_df['date'], portfolio_df['total_value'], linewidth=2, color='#2E8B57')
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.title('💰 Portfolio Value Evolution (2 Years)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (€)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for key milestones
    final_value = portfolio_df['total_value'].iloc[-1]
    plt.annotate(f'Final: €{final_value:,.0f}\n(+{(final_value/10000-1)*100:.1f}%)', 
                xy=(portfolio_df['date'].iloc[-1], final_value),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    # 2. Cumulative Returns
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(portfolio_df['date'], portfolio_df['total_return'] * 100, linewidth=2, color='#FF6347')
    plt.title('📈 Cumulative Returns (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    ax3 = plt.subplot(3, 3, 3)
    plt.fill_between(portfolio_df['date'], 0, -portfolio_df['max_drawdown'] * 100, 
                     color='red', alpha=0.6, label='Drawdown')
    plt.title('📉 Maximum Drawdown Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    max_dd = portfolio_df['max_drawdown'].max() * 100
    plt.text(0.02, 0.95, f'Max DD: {max_dd:.1f}%', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 4. Strategy Performance Breakdown
    ax4 = plt.subplot(3, 3, 4)
    strategy_alloc = results['strategy_allocations']
    strategies = list(strategy_alloc.keys())
    allocations = list(strategy_alloc.values())
    
    # Create pie chart
    colors = sns.color_palette("husl", len(strategies))
    wedges, texts, autotexts = plt.pie(allocations, labels=strategies, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    plt.title('🎯 Strategy Allocation', fontsize=14, fontweight='bold')
    
    # 5. Daily P&L Distribution
    ax5 = plt.subplot(3, 3, 5)
    daily_pnl = portfolio_df['daily_pnl'].dropna()
    plt.hist(daily_pnl, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=daily_pnl.mean(), color='red', linestyle='--', 
                label=f'Mean: €{daily_pnl.mean():.1f}')
    plt.title('📊 Daily P&L Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily P&L (€)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Win Rate by Strategy
    ax6 = plt.subplot(3, 3, 6)
    if len(trades_df) > 0:
        strategy_wins = trades_df.groupby('strategy').agg({
            'pnl': ['count', lambda x: (x > 0).sum()]
        }).round(3)
        strategy_wins.columns = ['total_trades', 'winning_trades']
        strategy_wins['win_rate'] = strategy_wins['winning_trades'] / strategy_wins['total_trades']
        strategy_wins = strategy_wins.sort_values('win_rate', ascending=True)
        
        plt.barh(range(len(strategy_wins)), strategy_wins['win_rate'] * 100, color='lightcoral')
        plt.yticks(range(len(strategy_wins)), strategy_wins.index)
        plt.title('🎯 Win Rate by Strategy (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(strategy_wins['win_rate'] * 100):
            plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 7. Monthly Returns Heatmap
    ax7 = plt.subplot(3, 3, 7)
    portfolio_df['year'] = portfolio_df['date'].dt.year
    portfolio_df['month'] = portfolio_df['date'].dt.month
    
    # Calculate monthly returns
    monthly_returns = portfolio_df.groupby(['year', 'month'])['total_value'].last().pct_change()
    monthly_returns = monthly_returns.reset_index()
    monthly_returns['return_pct'] = monthly_returns.iloc[:, 2] * 100
    
    # Create pivot table for heatmap
    pivot_table = monthly_returns.pivot(index='year', columns='month', values='return_pct')
    
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax7)
    plt.title('🔥 Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # 8. Sharpe Ratio Evolution
    ax8 = plt.subplot(3, 3, 8)
    sharpe_data = portfolio_df['sharpe_ratio'].rolling(window=30).mean()  # 30-day rolling
    plt.plot(portfolio_df['date'], sharpe_data, linewidth=2, color='purple')
    plt.title('📏 Sharpe Ratio Evolution (30d Rolling)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    final_sharpe = results['sharpe_ratio']
    plt.text(0.02, 0.95, f'Final Sharpe: {final_sharpe:.2f}', transform=ax8.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 9. Key Performance Metrics Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create metrics text
    metrics_text = f"""
🏆 PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━
📅 Period: 2 Years (2022-2024)
💰 Initial Capital: €{results['initial_capital']:,}
💎 Final Capital: €{results['final_capital']:,.0f}
📈 Total Return: {results['total_return']:.1%}
📊 Annualized Return: {results['annualized_return']:.1%}
⚡ Sharpe Ratio: {results['sharpe_ratio']:.2f}
📉 Max Drawdown: {results['max_drawdown']:.1%}
🎯 Win Rate: {results['win_rate']:.1%}
🔄 Total Trades: {results['total_trades']:,}
📆 Profitable Days: {results['profitable_days']}/{results['total_days']}
💪 Daily Win Rate: {results['profitable_days']/results['total_days']:.1%}

🚀 VERDICT: {"EXCELLENT" if results['total_return'] > 0.5 else "VERY GOOD" if results['total_return'] > 0.3 else "GOOD"}
🎖️ RISK-ADJUSTED: {"LOW RISK" if results['max_drawdown'] < 0.1 else "MEDIUM RISK"}
"""
    
    plt.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('SuperLazyBillionaire Strategy - 2-Year Backtest Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save the dashboard
    output_file = results_dir / f"dashboard_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved to: {output_file}")
    
    plt.show()
    
    # Create additional analysis
    print("\n" + "="*80)
    print("📈 ZUSÄTZLICHE PERFORMANCE-ANALYSE")
    print("="*80)
    
    # Risk-Return Analysis
    annual_vol = portfolio_df['daily_pnl'].std() * np.sqrt(365) / 10000  # Annualized volatility
    print(f"\n📊 RISK ANALYSIS:")
    print(f"   Annualized Volatility: {annual_vol:.1%}")
    print(f"   Risk-Adjusted Return: {results['annualized_return']/annual_vol:.2f}")
    print(f"   Calmar Ratio: {results['annualized_return']/results['max_drawdown']:.2f}")
    
    # Best/Worst Periods
    portfolio_df['rolling_return'] = portfolio_df['total_value'].pct_change(30)  # 30-day returns
    best_month = portfolio_df.loc[portfolio_df['rolling_return'].idxmax()]
    worst_month = portfolio_df.loc[portfolio_df['rolling_return'].idxmin()]
    
    print(f"\n📅 BEST/WORST PERIODS:")
    print(f"   Best 30-day period: {best_month['rolling_return']:.1%} ending {best_month['date'].strftime('%Y-%m-%d')}")
    print(f"   Worst 30-day period: {worst_month['rolling_return']:.1%} ending {worst_month['date'].strftime('%Y-%m-%d')}")
    
    # Strategy Contribution Analysis
    if len(trades_df) > 0:
        strategy_pnl = trades_df.groupby('strategy')['pnl'].agg(['sum', 'count', 'mean'])
        strategy_pnl['total_contribution'] = strategy_pnl['sum'] / strategy_pnl['sum'].sum()
        strategy_pnl = strategy_pnl.sort_values('sum', ascending=False)
        
        print(f"\n🎯 TOP STRATEGY CONTRIBUTORS:")
        for idx, (strategy, data) in enumerate(strategy_pnl.head().iterrows()):
            print(f"   {idx+1}. {strategy:20} €{data['sum']:7.0f} ({data['total_contribution']:5.1%}) - {data['count']:4.0f} trades")

def main():
    """Hauptfunktion"""
    print("🎨 Creating SuperLazyBillionaire Backtest Dashboard...")
    create_comprehensive_analysis()
    print("✅ Analysis completed!")

if __name__ == "__main__":
    main()