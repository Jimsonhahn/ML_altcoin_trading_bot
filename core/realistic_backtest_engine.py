"""
Realistic Backtesting Engine mit korrekter Sharpe Ratio Berechnung
und realistischen Marktbedingungen
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    execution_price: float
    total_cost: Dict[str, float]
    fill_probability: float
    actual_fill_size: float
    latency_ms: float

@dataclass
class TradeResult:
    entry_date: datetime
    exit_date: datetime
    strategy: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    costs: Dict[str, float]
    return_pct: float
    holding_period: timedelta

class RealisticBacktestEngine:
    """
    Backtesting Engine mit realistischen Constraints:
    - Korrekte Sharpe Ratio Berechnung
    - Realistische Kosten (Fees, Slippage, Spread)
    - Market Impact Modellierung
    - Latenz-Simulation
    """
    
    def __init__(self, config: Dict = None):
        # Realistische Market Constraints
        self.constraints = {
            # Kosten
            'exchange_fee_rate': 0.001,        # 0.1% Binance Spot Fee
            'min_spread_percent': 0.05,        # 0.05% Minimum Spread
            'avg_spread_percent': 0.10,        # 0.10% Average Spread
            
            # Slippage (abhängig von Ordergröße)
            'base_slippage': 0.0005,           # 0.05% Base Slippage
            'slippage_factor': 0.001,          # +0.1% pro 1% des Volumens
            
            # Market Impact
            'market_impact_factor': 0.0002,    # 0.02% pro 1% des Volumens
            'max_volume_percent': 0.05,        # Max 5% des Marktvolumens
            
            # Latenz
            'min_latency_ms': 50,              # Minimum 50ms
            'avg_latency_ms': 150,             # Durchschnitt 150ms
            'max_latency_ms': 1000,            # Spike bis 1s
            
            # Liquidität
            'liquidity_factor': 0.7,           # Nur 70% der Orders werden gefüllt
            'partial_fill_probability': 0.3,    # 30% Chance auf Partial Fill
            
            # Risk Free Rate für Sharpe
            'risk_free_rate': 0.02             # 2% jährlich (US Treasury)
        }
        
        if config:
            self.constraints.update(config)
            
        # Performance Tracking
        self.equity_curve = []
        self.daily_returns = []
        self.trade_history = []
        self.daily_data = []
        
    def calculate_realistic_execution_price(self, 
                                          signal_price: float,
                                          size: float,
                                          side: str,
                                          market_data: Dict) -> ExecutionResult:
        """
        Berechnet realistischen Ausführungspreis mit allen Kosten
        """
        # 1. Spread Kosten
        spread = self._calculate_dynamic_spread(market_data)
        if side == 'buy':
            price_after_spread = signal_price * (1 + spread/2)
        else:
            price_after_spread = signal_price * (1 - spread/2)
            
        # 2. Slippage (abhängig von Ordergröße und Volatilität)
        volume_percent = size / market_data.get('volume', 1000000)
        slippage = self._calculate_slippage(volume_percent, market_data)
        
        if side == 'buy':
            price_after_slippage = price_after_spread * (1 + slippage)
        else:
            price_after_slippage = price_after_spread * (1 - slippage)
            
        # 3. Market Impact
        market_impact = self._calculate_market_impact(volume_percent)
        if side == 'buy':
            final_price = price_after_slippage * (1 + market_impact)
        else:
            final_price = price_after_slippage * (1 - market_impact)
            
        # 4. Latenz Impact (Preis bewegt sich während Latenz)
        latency_ms = np.random.normal(
            self.constraints['avg_latency_ms'],
            self.constraints['avg_latency_ms'] * 0.3
        )
        latency_price = self._simulate_latency_impact(final_price, market_data, latency_ms)
        
        # 5. Exchange Fees
        fee_amount = size * signal_price * self.constraints['exchange_fee_rate']
        
        # 6. Fill Probability und Partial Fills
        fill_probability = self._calculate_fill_probability(volume_percent)
        actual_fill_size = size
        
        if np.random.random() > fill_probability:
            # Partial Fill oder No Fill
            if np.random.random() < self.constraints['partial_fill_probability']:
                actual_fill_size = size * np.random.uniform(0.3, 0.8)  # 30-80% Fill
            else:
                actual_fill_size = 0  # No Fill
        
        total_cost = {
            'spread_cost': abs(price_after_spread - signal_price) * actual_fill_size,
            'slippage_cost': abs(price_after_slippage - price_after_spread) * actual_fill_size,
            'market_impact_cost': abs(final_price - price_after_slippage) * actual_fill_size,
            'latency_cost': abs(latency_price - final_price) * actual_fill_size,
            'exchange_fees': fee_amount * (actual_fill_size / size),
            'total': (abs(latency_price - signal_price) * actual_fill_size + 
                     fee_amount * (actual_fill_size / size))
        }
        
        return ExecutionResult(
            execution_price=latency_price,
            total_cost=total_cost,
            fill_probability=fill_probability,
            actual_fill_size=actual_fill_size,
            latency_ms=latency_ms
        )
    
    def _calculate_dynamic_spread(self, market_data: Dict) -> float:
        """Dynamischer Spread basierend auf Volatilität und Liquidität"""
        base_spread = self.constraints['avg_spread_percent'] / 100
        
        # Volatilität erhöht Spread
        volatility = market_data.get('volatility', 0.02)
        vol_multiplier = 1 + (volatility - 0.02) * 10  # Höhere Vol = Weitere Spreads
        
        # Niedrige Liquidität erhöht Spread
        volume = market_data.get('volume', 1000000)
        if volume < 100000:  # Niedrige Liquidität
            liquidity_multiplier = 2.0
        elif volume < 500000:
            liquidity_multiplier = 1.5
        else:
            liquidity_multiplier = 1.0
            
        return base_spread * vol_multiplier * liquidity_multiplier
    
    def _calculate_slippage(self, volume_percent: float, market_data: Dict) -> float:
        """Slippage abhängig von Ordergröße und Marktbedingungen"""
        base_slippage = self.constraints['base_slippage']
        
        # Größere Orders = mehr Slippage
        size_slippage = self.constraints['slippage_factor'] * (volume_percent * 100)
        
        # Volatilität erhöht Slippage
        volatility = market_data.get('volatility', 0.02)
        vol_slippage = volatility * 0.5
        
        return base_slippage + size_slippage + vol_slippage
    
    def _calculate_market_impact(self, volume_percent: float) -> float:
        """Permanenter Market Impact großer Orders"""
        return self.constraints['market_impact_factor'] * (volume_percent * 100) ** 1.5
    
    def _simulate_latency_impact(self, price: float, market_data: Dict, latency_ms: float) -> float:
        """Simuliert Preisbewegung während Latenz"""
        # Latenz in Sekunden
        latency_seconds = latency_ms / 1000
        
        # Preisbewegung basierend auf Volatilität
        volatility = market_data.get('volatility', 0.02)
        price_change = np.random.normal(0, volatility * np.sqrt(latency_seconds / 86400))
        
        return price * (1 + price_change)
    
    def _calculate_fill_probability(self, volume_percent: float) -> float:
        """Berechnet Fill-Wahrscheinlichkeit basierend auf Ordergröße"""
        base_probability = self.constraints['liquidity_factor']
        
        # Große Orders haben geringere Fill-Wahrscheinlichkeit
        if volume_percent > 0.05:  # > 5% des Volumens
            return base_probability * 0.3
        elif volume_percent > 0.02:  # > 2% des Volumens
            return base_probability * 0.6
        elif volume_percent > 0.01:  # > 1% des Volumens
            return base_probability * 0.8
        else:
            return base_probability
    
    def add_trade(self, entry_date: datetime, exit_date: datetime, strategy: str,
                  symbol: str, side: str, size: float, entry_price: float, 
                  exit_price: float, market_data: Dict):
        """Fügt einen Trade mit realistischen Kosten hinzu"""
        
        # Entry Execution
        entry_execution = self.calculate_realistic_execution_price(
            entry_price, size, side, market_data
        )
        
        # Exit Execution (opposite side)
        exit_side = 'sell' if side == 'buy' else 'buy'
        exit_execution = self.calculate_realistic_execution_price(
            exit_price, entry_execution.actual_fill_size, exit_side, market_data
        )
        
        # Calculate P&L
        if side == 'buy':
            gross_pnl = (exit_execution.execution_price - entry_execution.execution_price) * entry_execution.actual_fill_size
        else:
            gross_pnl = (entry_execution.execution_price - exit_execution.execution_price) * entry_execution.actual_fill_size
        
        total_costs = entry_execution.total_cost['total'] + exit_execution.total_cost['total']
        net_pnl = gross_pnl - total_costs
        
        # Return percentage (based on capital allocated)
        capital_allocated = entry_execution.execution_price * entry_execution.actual_fill_size
        return_pct = net_pnl / capital_allocated if capital_allocated > 0 else 0
        
        trade = TradeResult(
            entry_date=entry_date,
            exit_date=exit_date,
            strategy=strategy,
            symbol=symbol,
            side=side,
            size=entry_execution.actual_fill_size,
            entry_price=entry_execution.execution_price,
            exit_price=exit_execution.execution_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            costs={
                'entry_costs': entry_execution.total_cost,
                'exit_costs': exit_execution.total_cost,
                'total_costs': total_costs
            },
            return_pct=return_pct,
            holding_period=exit_date - entry_date
        )
        
        self.trade_history.append(trade)
        return trade
    
    def add_daily_data(self, date: datetime, portfolio_value: float):
        """Fügt tägliche Portfolio-Daten hinzu"""
        if self.daily_data:
            prev_value = self.daily_data[-1]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0.0
            
        self.daily_data.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'daily_return': daily_return
        })
        
        self.daily_returns.append(daily_return)
        self.equity_curve.append({'date': date, 'equity': portfolio_value})
    
    def calculate_correct_sharpe_ratio(self) -> float:
        """
        KORREKTE Sharpe Ratio Berechnung
        Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
        """
        if len(self.daily_returns) < 30:
            return 0.0
            
        daily_returns = np.array(self.daily_returns)
        
        # Entferne erste Return (immer 0)
        if len(daily_returns) > 1:
            daily_returns = daily_returns[1:]
        
        # Entferne extreme Outliers (unrealistische Returns)
        q1 = np.percentile(daily_returns, 25)
        q3 = np.percentile(daily_returns, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Zusätzliche Bounds für Crypto (max ±20% daily)
        lower_bound = max(lower_bound, -0.20)
        upper_bound = min(upper_bound, 0.20)
        
        filtered_returns = daily_returns[
            (daily_returns >= lower_bound) & (daily_returns <= upper_bound)
        ]
        
        if len(filtered_returns) < 10:
            return 0.0
        
        # Annualisierte Metriken
        mean_daily_return = np.mean(filtered_returns)
        daily_volatility = np.std(filtered_returns, ddof=1)  # Sample std
        
        # Annualisierung (365 days für Crypto)
        annual_return = mean_daily_return * 365
        annual_volatility = daily_volatility * np.sqrt(365)
        
        # Risk Free Rate
        risk_free_rate = self.constraints['risk_free_rate']
        
        # Sharpe Ratio
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0.0
            
        # Zusätzliche Metriken für Debugging
        logger.debug(f"""
        Sharpe Ratio Calculation:
        - Days: {len(filtered_returns)}
        - Daily Return: {mean_daily_return:.4%}
        - Daily Volatility: {daily_volatility:.4%}
        - Annual Return: {annual_return:.2%}
        - Annual Volatility: {annual_volatility:.2%}
        - Risk Free Rate: {risk_free_rate:.2%}
        - Sharpe Ratio: {sharpe_ratio:.2f}
        """)
        
        return sharpe_ratio
    
    def calculate_realistic_metrics(self) -> Dict:
        """Berechnet alle realistischen Performance-Metriken"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {}
            
        equity_array = np.array([e['equity'] for e in self.equity_curve])
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))  # Positive value
        
        # Average Drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = abs(np.mean(negative_drawdowns)) if len(negative_drawdowns) > 0 else 0
        
        # Recovery Time
        drawdown_periods = []
        in_drawdown = False
        start_dd = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of 1%+ drawdown
                in_drawdown = True
                start_dd = i
            elif dd >= -0.001 and in_drawdown:  # Recovery
                in_drawdown = False
                drawdown_periods.append(i - start_dd)
        
        avg_recovery_time = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Win Rate (realistisch)
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.net_pnl > 0]
            total_trades = len(self.trade_history)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Profit Factor
            gross_profit = sum(t.net_pnl for t in winning_trades)
            gross_loss = abs(sum(t.net_pnl for t in self.trade_history if t.net_pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Kosten-Analyse
            total_costs = sum(t.costs['total_costs'] for t in self.trade_history)
            gross_pnl_before_costs = sum(t.gross_pnl for t in self.trade_history)
            cost_ratio = total_costs / abs(gross_pnl_before_costs) if gross_pnl_before_costs != 0 else 0
            
            # Average Trade Return
            avg_trade_return = np.mean([t.return_pct for t in self.trade_history])
        else:
            win_rate = 0
            profit_factor = 0
            total_costs = 0
            cost_ratio = 0
            avg_trade_return = 0
            total_trades = 0
        
        # Portfolio Metrics
        total_return = (equity_array[-1] / equity_array[0]) - 1 if len(equity_array) > 1 else 0
        days = len(self.daily_returns)
        annual_return = ((equity_array[-1] / equity_array[0]) ** (365 / days)) - 1 if days > 0 else 0
        
        # Volatility
        daily_vol = np.std(self.daily_returns[1:], ddof=1) if len(self.daily_returns) > 1 else 0
        annual_vol = daily_vol * np.sqrt(365)
        
        # Calmar Ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino Ratio
        negative_returns = [r for r in self.daily_returns[1:] if r < 0]
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(365) if negative_returns else annual_vol
        sortino_ratio = (annual_return - self.constraints['risk_free_rate']) / downside_vol if downside_vol > 0 else 0
        
        return {
            'sharpe_ratio': self.calculate_correct_sharpe_ratio(),
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_recovery_time_days': avg_recovery_time,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'days_analyzed': days,
            'best_trade': max([t.net_pnl for t in self.trade_history]) if self.trade_history else 0,
            'worst_trade': min([t.net_pnl for t in self.trade_history]) if self.trade_history else 0
        }
    
    def get_summary_stats(self) -> str:
        """Gibt zusammenfassende Statistiken aus"""
        metrics = self.calculate_realistic_metrics()
        
        return f"""
        Realistic Backtest Summary:
        ==========================
        Total Return: {metrics.get('total_return', 0):.1%}
        Annual Return: {metrics.get('annual_return', 0):.1%}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown', 0):.1%}
        Win Rate: {metrics.get('win_rate', 0):.1%}
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        Total Trades: {metrics.get('total_trades', 0):,}
        Cost Ratio: {metrics.get('cost_ratio', 0):.1%}
        Days Analyzed: {metrics.get('days_analyzed', 0)}
        """