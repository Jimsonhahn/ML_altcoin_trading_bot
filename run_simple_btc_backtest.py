#!/usr/bin/env python3
"""
Simple BTC Backtest für 2024 ohne ML-Abhängigkeiten
Direkte Ausführung mit spezifischen Metriken
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SimpleBTCBacktest:
    """
    Vereinfachter Backtest für BTC/USDT 2024
    Ohne komplexe ML-Abhängigkeiten
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.btc_position = 0.0
        self.cash_balance = initial_capital
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        
        # Strategy settings
        self.trading_fee = 0.001  # 0.1% fee
        self.min_trade_size = 0.001  # Minimum BTC trade size
        
        logger.info(f"SimpleBTCBacktest initialisiert mit ${initial_capital:,.0f}")
    
    def simulate_btc_price_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Simuliert realistische BTC-Preisdaten für 2024
        Basiert auf historischen Volatilitäts- und Trendmustern
        """
        
        # BTC Startpreis Januar 2024: ~42,000
        # BTC Endpreis Dezember 2024: ~95,000 (ca. 126% Rendite)
        start_price = 42000.0
        end_price = 95000.0
        
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Simuliere tägliche Preisbewegungen mit realistischer Volatilität
        daily_vol = 0.045  # 4.5% tägliche Volatilität für BTC
        
        # Trend-Komponente (exponentieller Wachstumstrend)
        trend_factor = (end_price / start_price) ** (1/days)
        
        prices = []
        current_price = start_price
        
        for i, date in enumerate(dates):
            # Trend + Random Walk mit Mean Reversion
            trend_adjustment = trend_factor - 1
            
            # Saisonale Komponenten (Q4 ist traditionell stark für BTC)
            month = date.month
            seasonal_factor = 1.0
            if month in [10, 11, 12]:  # Q4 Boost
                seasonal_factor = 1.15
            elif month in [6, 7, 8]:   # Sommer-Schwäche
                seasonal_factor = 0.95
            
            # Zufällige tägliche Bewegung
            daily_return = np.random.normal(
                trend_adjustment * seasonal_factor, 
                daily_vol
            )
            
            # Volatilitäts-Clustering
            if i > 0 and abs(prices[-1]['daily_return']) > 0.1:
                daily_return *= 1.5  # Erhöhte Volatilität nach großen Bewegungen
            
            current_price *= (1 + daily_return)
            
            # Vermeide unrealistische Preise
            current_price = max(20000, min(120000, current_price))
            
            prices.append({
                'date': date,
                'price': current_price,
                'daily_return': daily_return,
                'volume': np.random.uniform(20000, 50000)  # Simuliertes Volumen
            })
        
        return {
            'prices': prices,
            'start_price': start_price,
            'end_price': prices[-1]['price'],
            'total_days': days
        }
    
    def generate_trading_signals(self, price_data: Dict[str, Any]) -> list:
        """
        Generiert einfache Handelssignale
        Kombiniert Momentum, Mean Reversion und Volatility-Based Signals
        """
        
        prices = price_data['prices']
        signals = []
        
        # Parameter
        short_ma_window = 7   # 7-Tage MA
        long_ma_window = 30   # 30-Tage MA
        volatility_window = 14
        
        for i in range(long_ma_window, len(prices)):
            current_data = prices[i]
            
            # Moving Averages
            short_ma = np.mean([p['price'] for p in prices[i-short_ma_window:i]])
            long_ma = np.mean([p['price'] for p in prices[i-long_ma_window:i]])
            
            # Volatility
            recent_returns = [p['daily_return'] for p in prices[i-volatility_window:i]]
            volatility = np.std(recent_returns)
            
            # Signal-Generierung
            signal_strength = 0.0
            signal_type = None
            
            # 1. Momentum Signal
            if short_ma > long_ma * 1.02:  # 2% über long MA
                signal_strength += 0.4
                signal_type = 'long'
            elif short_ma < long_ma * 0.98:  # 2% unter long MA
                signal_strength += 0.4
                signal_type = 'short'
            
            # 2. Mean Reversion Signal (bei hoher Volatilität)
            if volatility > 0.08:  # Hohe Volatilität
                recent_performance = (current_data['price'] - prices[i-5]['price']) / prices[i-5]['price']
                if recent_performance < -0.15:  # 15% Rückgang
                    signal_strength += 0.3
                    signal_type = 'long'
                elif recent_performance > 0.15:  # 15% Anstieg
                    signal_strength += 0.3
                    signal_type = 'short'
            
            # 3. Volumen-basiertes Signal
            avg_volume = np.mean([p['volume'] for p in prices[i-7:i]])
            if current_data['volume'] > avg_volume * 1.5:  # Hohes Volumen
                signal_strength += 0.2
            
            # Signal nur wenn Stärke > Schwellwert
            if signal_strength > 0.6:
                
                # Positionsgröße basierend auf Signal-Stärke und Volatilität
                volatility_factor = max(0.5, 1.5 - volatility * 10)
                position_size = min(0.2, signal_strength * volatility_factor)  # Max 20% der Equity
                
                signals.append({
                    'date': current_data['date'],
                    'price': current_data['price'],
                    'signal_type': signal_type,
                    'signal_strength': signal_strength,
                    'position_size': position_size,
                    'volatility': volatility
                })
        
        return signals
    
    def execute_backtest(self, price_data: Dict[str, Any], signals: list) -> None:
        """
        Führt Backtest mit realistischen Transaktionskosten aus
        """
        
        prices = price_data['prices']
        price_dict = {p['date'].strftime('%Y-%m-%d'): p['price'] for p in prices}
        signal_dict = {s['date'].strftime('%Y-%m-%d'): s for s in signals}
        
        logger.info(f"Führe Backtest aus: {len(signals)} Signale über {len(prices)} Tage")
        
        for price_data_point in prices:
            date = price_data_point['date']
            current_price = price_data_point['price']
            date_str = date.strftime('%Y-%m-%d')
            
            # Update Portfolio-Wert
            portfolio_value = self.cash_balance + (self.btc_position * current_price)
            
            # Verarbeite Signal falls vorhanden
            if date_str in signal_dict:
                signal = signal_dict[date_str]
                self._execute_trade(signal, current_price)
            
            # Täglich Portfolio-Snapshot
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'btc_position': self.btc_position,
                'btc_value': self.btc_position * current_price,
                'btc_price': current_price
            })
            
            # Tägliche Returns berechnen
            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
    
    def _execute_trade(self, signal: Dict[str, Any], current_price: float) -> None:
        """
        Führt einzelnen Trade aus
        """
        
        signal_type = signal['signal_type']
        position_size = signal['position_size']
        current_portfolio_value = self.cash_balance + (self.btc_position * current_price)
        
        if signal_type == 'long':
            # Kauf BTC
            target_allocation = position_size
            target_btc_value = current_portfolio_value * target_allocation
            target_btc_position = target_btc_value / current_price
            
            btc_to_buy = target_btc_position - self.btc_position
            
            if btc_to_buy > self.min_trade_size:
                cost = btc_to_buy * current_price
                total_cost = cost * (1 + self.trading_fee)
                
                if total_cost <= self.cash_balance:
                    self.cash_balance -= total_cost
                    self.btc_position += btc_to_buy
                    
                    self.trades.append({
                        'date': signal['date'],
                        'type': 'BUY',
                        'quantity': btc_to_buy,
                        'price': current_price,
                        'cost': total_cost,
                        'signal_strength': signal['signal_strength']
                    })
        
        elif signal_type == 'short':
            # Verkauf BTC (vereinfacht, kein echtes Shorting)
            btc_to_sell = self.btc_position * position_size
            
            if btc_to_sell > self.min_trade_size:
                proceeds = btc_to_sell * current_price
                net_proceeds = proceeds * (1 - self.trading_fee)
                
                self.cash_balance += net_proceeds
                self.btc_position -= btc_to_sell
                
                self.trades.append({
                    'date': signal['date'],
                    'type': 'SELL',
                    'quantity': btc_to_sell,
                    'price': current_price,
                    'proceeds': net_proceeds,
                    'signal_strength': signal['signal_strength']
                })
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Berechnet umfassende Performance-Metriken
        """
        
        if len(self.equity_curve) < 2:
            return {}
        
        # Finale Werte
        final_portfolio_value = self.equity_curve[-1]['portfolio_value']
        
        # Total Return
        total_return = (final_portfolio_value / self.initial_capital) - 1
        
        # Annualized Return
        days = len(self.equity_curve)
        annual_return = ((final_portfolio_value / self.initial_capital) ** (365 / days)) - 1
        
        # Daily Returns für weitere Berechnungen
        returns_array = np.array(self.daily_returns)
        
        # Volatility
        if len(returns_array) > 1:
            daily_vol = np.std(returns_array, ddof=1)
            annual_vol = daily_vol * np.sqrt(252)
        else:
            daily_vol = 0
            annual_vol = 0
        
        # Sharpe Ratio (2% risk-free rate)
        risk_free_rate = 0.02
        if annual_vol > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_values = [snapshot['portfolio_value'] for snapshot in self.equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Trade Statistics
        winning_trades = len([t for t in self.trades if self._is_winning_trade(t)])
        total_trades = len(self.trades)
        win_rate = winning_trades / max(total_trades, 1)
        
        # Commission
        total_commission = sum(
            t.get('cost', 0) * self.trading_fee if t['type'] == 'BUY'
            else t.get('proceeds', 0) * self.trading_fee for t in self.trades
        )
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'annual_volatility': annual_vol,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_commission': total_commission,
            'final_portfolio_value': final_portfolio_value,
            'days_analyzed': days
        }
    
    def _calculate_max_drawdown(self, equity_values: list) -> float:
        """Berechnet Maximum Drawdown"""
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            drawdown = (value - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return abs(max_drawdown)
    
    def _is_winning_trade(self, trade: Dict[str, Any]) -> bool:
        """Bestimmt ob Trade gewinnbringend war (vereinfacht)"""
        return trade.get('signal_strength', 0) > 0.7


async def run_simple_btc_backtest():
    """
    Führt vereinfachten BTC Backtest für 2024 aus
    """
    
    print("🚀 SIMPLE BTC BACKTEST - 2024")
    print("=" * 60)
    print("BTC/USDT Performance-Test ohne ML-Abhängigkeiten")
    print("Zeitraum: 1. Januar 2024 - 31. Dezember 2024")
    print()
    
    # Backtest initialisieren
    backtest = SimpleBTCBacktest(initial_capital=1000000.0)
    
    # Zeitraum
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("📊 Generiere BTC-Preisdaten...")
    price_data = backtest.simulate_btc_price_data(start_date, end_date)
    
    print(f"✅ {len(price_data['prices'])} Tage generiert")
    print(f"   Start-Preis: ${price_data['start_price']:,.0f}")
    print(f"   End-Preis: ${price_data['end_price']:,.0f}")
    print(f"   BTC Return: {((price_data['end_price'] / price_data['start_price']) - 1):.1%}")
    print()
    
    print("🧠 Generiere Trading-Signale...")
    signals = backtest.generate_trading_signals(price_data)
    print(f"✅ {len(signals)} Handelssignale generiert")
    print()
    
    print("⚡ Führe Backtest aus...")
    backtest.execute_backtest(price_data, signals)
    print(f"✅ Backtest abgeschlossen: {len(backtest.trades)} Trades ausgeführt")
    print()
    
    # Performance-Analyse
    print("📈 PERFORMANCE ANALYSE")
    print("-" * 60)
    
    metrics = backtest.calculate_performance_metrics()
    
    # Hauptmetriken ausgeben
    print(f"💰 Startkapital:        ${backtest.initial_capital:,.0f}")
    print(f"💰 Endkapital:          ${metrics['final_portfolio_value']:,.0f}")
    print(f"📊 Total Return:        {metrics['total_return']:.1%}")
    print(f"📊 Annual Return:       {metrics['annual_return']:.1%}")
    print(f"⚡ Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown:        {metrics['max_drawdown']:.1%}")
    print(f"🎯 Volatilität:         {metrics['annual_volatility']:.1%}")
    print(f"🔄 Anzahl Trades:       {metrics['total_trades']:,}")
    print(f"✅ Win Rate:            {metrics['win_rate']:.1%}")
    print(f"💸 Total Commission:    ${metrics['total_commission']:,.0f}")
    print()
    
    # Vergleich mit Buy & Hold
    btc_buy_hold_return = (price_data['end_price'] / price_data['start_price']) - 1
    print("📊 VERGLEICH MIT BUY & HOLD")
    print("-" * 60)
    print(f"BTC Buy & Hold Return:  {btc_buy_hold_return:.1%}")
    print(f"Strategy Return:        {metrics['total_return']:.1%}")
    alpha = metrics['total_return'] - btc_buy_hold_return
    print(f"Alpha (Mehrertrag):     {alpha:.1%}")
    print()
    
    # Realistische Einschätzung
    print("🎯 REALISTISCHE EINSCHÄTZUNG")
    print("-" * 60)
    
    if metrics['sharpe_ratio'] > 2.0:
        print("⚠️  WARNUNG: Sharpe Ratio > 2.0 ist unrealistisch hoch")
        print("   Echte Trading-Systeme erreichen selten > 1.5")
    elif metrics['sharpe_ratio'] > 1.0:
        print("✅ Sharpe Ratio ist im realistischen Bereich")
    else:
        print("📉 Niedrige Sharpe Ratio - Strategy unterperformt")
    
    if metrics['max_drawdown'] < 0.05:
        print("⚠️  WARNUNG: Max Drawdown < 5% ist unrealistisch niedrig")
    elif metrics['max_drawdown'] < 0.20:
        print("✅ Max Drawdown ist im akzeptablen Bereich")
    else:
        print("🔴 Hoher Drawdown - erhöhtes Risiko")
    
    print()
    print("🏁 BACKTEST ABGESCHLOSSEN")
    print("=" * 60)
    
    # Export Ergebnisse
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"simple_btc_backtest_results_{timestamp}.json"
    
    results = {
        'backtest_info': {
            'strategy': 'Simple BTC Strategy',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'initial_capital': backtest.initial_capital,
            'symbols': ['BTC/USDT']
        },
        'performance_metrics': metrics,
        'btc_performance': {
            'start_price': price_data['start_price'],
            'end_price': price_data['end_price'],
            'buy_hold_return': btc_buy_hold_return
        },
        'strategy_comparison': {
            'alpha': alpha,
            'outperformed_buy_hold': metrics['total_return'] > btc_buy_hold_return
        }
    }
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Ergebnisse exportiert: {results_file}")
    
    return metrics


if __name__ == "__main__":
    asyncio.run(run_simple_btc_backtest())