#!/usr/bin/env python3
"""
Simple Defensive Strategies Backtest
===================================

Vereinfachtes aber realistisches Backtesting für defensive Strategien
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDefensiveBacktest:
    """Vereinfachtes Backtest für defensive Strategien"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.trading_fee = 0.001  # 0.1%
        
        # Generiere realistische Marktdaten
        self.data = self._generate_market_data()
        
    def _generate_market_data(self) -> pd.DataFrame:
        """Generiert realistische Marktdaten für 1 Jahr"""
        np.random.seed(42)
        
        # 1 Jahr täglich
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # BTC-ähnliche Preisentwicklung
        base_price = 30000
        returns = []
        
        # Verschiedene Marktphasen simulieren
        for i in range(n_days):
            if i < 100:  # Q1: Bear Market
                daily_return = np.random.normal(-0.002, 0.04)
            elif i < 200:  # Q2: Recovery
                daily_return = np.random.normal(0.001, 0.03)
            elif i < 300:  # Q3: Bull Run
                daily_return = np.random.normal(0.003, 0.045)
            else:  # Q4: Consolidation
                daily_return = np.random.normal(0.0005, 0.025)
            
            returns.append(daily_return)
        
        # Preise berechnen
        prices = [base_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # Minimum $1000
        
        prices = prices[:-1]  # Letzten Wert entfernen
        
        return pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': returns,
            'volatility': pd.Series(returns).rolling(30).std() * np.sqrt(365)
        })
    
    def backtest_advanced_portfolio(self) -> dict:
        """
        Backtest: Advanced Portfolio Strategy
        - Diversifizierte Allokation
        - Monatliches Rebalancing
        - Risiko-adjustierte Positionierung
        """
        logger.info("Starte Advanced Portfolio Backtest...")
        
        portfolio_value = self.initial_capital
        positions = {'crypto': 0.0, 'stable': 0.0}
        trades = []
        portfolio_history = []
        
        # Monatliches Rebalancing
        for i in range(0, len(self.data), 30):
            current_data = self.data.iloc[:i+1] if i > 0 else self.data.iloc[:1]
            current_price = current_data['price'].iloc[-1]
            current_vol = current_data['volatility'].iloc[-1] if not pd.isna(current_data['volatility'].iloc[-1]) else 0.3
            
            # Regime-abhängige Allokation
            if current_vol > 0.6:  # Hohe Volatilität
                target_crypto = 0.3
            elif current_vol > 0.4:  # Mittlere Volatilität
                target_crypto = 0.6
            else:  # Niedrige Volatilität
                target_crypto = 0.8
            
            target_stable = 1.0 - target_crypto
            
            # Aktueller Portfolio-Wert
            crypto_value = positions['crypto'] * current_price
            stable_value = positions['stable']
            total_value = crypto_value + stable_value
            
            if total_value > 0:
                current_crypto_weight = crypto_value / total_value
                current_stable_weight = stable_value / total_value
                
                # Rebalancing wenn Abweichung > 10%
                if abs(current_crypto_weight - target_crypto) > 0.1:
                    # Handel ausführen
                    target_crypto_value = total_value * target_crypto
                    trade_value = target_crypto_value - crypto_value
                    
                    # Trading Fees
                    fee = abs(trade_value) * self.trading_fee
                    total_value -= fee
                    
                    # Positionen anpassen
                    if trade_value > 0:  # Crypto kaufen
                        crypto_bought = (trade_value - fee) / current_price
                        positions['crypto'] += crypto_bought
                        positions['stable'] -= trade_value
                        trades.append({
                            'date': current_data['date'].iloc[-1],
                            'action': 'buy_crypto',
                            'amount': crypto_bought,
                            'price': current_price,
                            'fee': fee
                        })
                    else:  # Crypto verkaufen
                        crypto_sold = abs(trade_value) / current_price
                        positions['crypto'] -= crypto_sold
                        positions['stable'] += abs(trade_value) - fee
                        trades.append({
                            'date': current_data['date'].iloc[-1],
                            'action': 'sell_crypto',
                            'amount': crypto_sold,
                            'price': current_price,
                            'fee': fee
                        })
            else:
                # Initial Allocation
                crypto_value = self.initial_capital * target_crypto
                stable_value = self.initial_capital * target_stable
                
                positions['crypto'] = (crypto_value - crypto_value * self.trading_fee) / current_price
                positions['stable'] = stable_value
                
                trades.append({
                    'date': current_data['date'].iloc[-1],
                    'action': 'initial_buy',
                    'amount': positions['crypto'],
                    'price': current_price,
                    'fee': crypto_value * self.trading_fee
                })
            
            # Portfolio-Wert aktualisieren
            portfolio_value = positions['crypto'] * current_price + positions['stable']
            portfolio_history.append({
                'date': current_data['date'].iloc[-1],
                'portfolio_value': portfolio_value,
                'crypto_weight': (positions['crypto'] * current_price) / portfolio_value if portfolio_value > 0 else 0,
                'volatility': current_vol
            })
        
        # Performance berechnen
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Volatilität der Portfolio-Returns
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        portfolio_vol = portfolio_returns.std() * np.sqrt(365)
        sharpe = (portfolio_returns.mean() * 365 - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Max Drawdown
        portfolio_values = portfolio_df['portfolio_value']
        peak = portfolio_values.expanding().max()
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()
        
        return {
            'strategy': 'Advanced Portfolio',
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365/len(self.data)) - 1,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'total_fees': sum(t['fee'] for t in trades),
            'portfolio_history': portfolio_history[-10:],  # Letzten 10 Punkte
            'trades': trades[-5:]  # Letzten 5 Trades
        }
    
    def backtest_defensive_volatility(self) -> dict:
        """
        Backtest: Defensive Volatility Strategy
        - Position Sizing basierend auf Volatilität
        - Defensive Positionierung bei hoher Volatilität
        """
        logger.info("Starte Defensive Volatility Backtest...")
        
        portfolio_value = self.initial_capital
        position_size = 0.0  # Crypto Position
        trades = []
        portfolio_history = []
        
        # Wöchentliches Trading
        for i in range(30, len(self.data), 7):  # Start nach 30 Tagen für Volatilitäts-Berechnung
            current_data = self.data.iloc[:i+1]
            current_price = current_data['price'].iloc[-1]
            current_vol = current_data['volatility'].iloc[-1]
            
            # Volatilitäts-basierte Position Sizing
            if current_vol > 0.8:  # Extreme Volatilität
                target_position = 0.05  # 5%
            elif current_vol > 0.5:  # Hohe Volatilität
                target_position = 0.15  # 15%
            elif current_vol > 0.3:  # Normale Volatilität
                target_position = 0.30  # 30%
            else:  # Niedrige Volatilität
                target_position = 0.50  # 50%
            
            # Signal-Generierung (vereinfacht)
            price_ma_short = current_data['price'].rolling(10).mean().iloc[-1]
            price_ma_long = current_data['price'].rolling(30).mean().iloc[-1]
            
            signal = 0  # 0=Hold, 1=Buy, -1=Sell
            if current_vol < 0.4:  # Nur bei niedriger Volatilität traden
                if current_price > price_ma_short > price_ma_long:
                    signal = 1  # Uptrend
                elif current_price < price_ma_short < price_ma_long:
                    signal = -1  # Downtrend
            
            # Position anpassen
            current_crypto_value = position_size * current_price
            current_weight = current_crypto_value / portfolio_value if portfolio_value > 0 else 0
            
            if signal == 1 and current_weight < target_position:
                # Crypto kaufen
                target_value = portfolio_value * target_position
                buy_value = target_value - current_crypto_value
                fee = buy_value * self.trading_fee
                
                crypto_bought = (buy_value - fee) / current_price
                position_size += crypto_bought
                portfolio_value -= fee
                
                trades.append({
                    'date': current_data['date'].iloc[-1],
                    'action': 'buy',
                    'amount': crypto_bought,
                    'price': current_price,
                    'fee': fee,
                    'volatility': current_vol
                })
                
            elif signal == -1 and current_weight > 0.05:
                # 50% der Position verkaufen
                sell_amount = position_size * 0.5
                sell_value = sell_amount * current_price
                fee = sell_value * self.trading_fee
                
                position_size -= sell_amount
                portfolio_value += sell_value - fee
                
                trades.append({
                    'date': current_data['date'].iloc[-1],
                    'action': 'sell',
                    'amount': sell_amount,
                    'price': current_price,
                    'fee': fee,
                    'volatility': current_vol
                })
            
            # Portfolio-Wert aktualisieren
            portfolio_value = (portfolio_value - position_size * current_price) + position_size * current_price
            
            portfolio_history.append({
                'date': current_data['date'].iloc[-1],
                'portfolio_value': portfolio_value,
                'position_weight': (position_size * current_price) / portfolio_value if portfolio_value > 0 else 0,
                'volatility': current_vol,
                'signal': signal
            })
        
        # Performance berechnen
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        portfolio_df = pd.DataFrame(portfolio_history)
        if len(portfolio_df) > 1:
            portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            portfolio_vol = portfolio_returns.std() * np.sqrt(52) if len(portfolio_returns) > 0 else 0
            sharpe = (portfolio_returns.mean() * 52 - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            portfolio_values = portfolio_df['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = drawdown.max()
        else:
            portfolio_vol = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'strategy': 'Defensive Volatility',
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365/len(self.data)) - 1,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'total_fees': sum(t['fee'] for t in trades),
            'avg_volatility': np.mean([t['volatility'] for t in trades]) if trades else 0,
            'portfolio_history': portfolio_history[-10:],
            'trades': trades[-5:]
        }
    
    def backtest_smart_rebalancing(self) -> dict:
        """
        Backtest: Smart Rebalancing Strategy
        - Kostenoptimiertes Rebalancing
        - Momentum-bewusste Allokation
        """
        logger.info("Starte Smart Rebalancing Backtest...")
        
        portfolio_value = self.initial_capital
        assets = {'BTC': 0.0, 'ETH': 0.0, 'BNB': 0.0}  # Vereinfacht auf 3 Assets
        target_weights = {'BTC': 0.5, 'ETH': 0.3, 'BNB': 0.2}
        cash = self.initial_capital  # Verfügbares Cash
        
        # Preise simulieren (vereinfacht)
        np.random.seed(123)  # Für konsistente Ergebnisse
        btc_prices = self.data['price'].values
        eth_prices = btc_prices * 0.07 + np.random.normal(0, 30, len(btc_prices))  # ETH ~0.07 BTC
        bnb_prices = btc_prices * 0.01 + np.random.normal(0, 5, len(btc_prices))   # BNB ~0.01 BTC
        
        trades = []
        portfolio_history = []
        
        # Weekly Rebalancing (häufiger für mehr Aktivität)
        for i in range(0, len(self.data), 7):
            current_prices = {
                'BTC': btc_prices[i],
                'ETH': max(eth_prices[i], 100),  # Min $100
                'BNB': max(bnb_prices[i], 50)   # Min $50
            }
            
            # Initial Allocation beim ersten Durchlauf
            if i == 0:
                for asset in assets.keys():
                    target_value = self.initial_capital * target_weights[asset]
                    fee = target_value * self.trading_fee
                    
                    asset_amount = (target_value - fee) / current_prices[asset]
                    assets[asset] = asset_amount
                    cash -= target_value
                    
                    trades.append({
                        'date': self.data['date'].iloc[i],
                        'asset': asset,
                        'action': 'initial_buy',
                        'amount': asset_amount,
                        'price': current_prices[asset],
                        'fee': fee,
                        'target_weight': target_weights[asset],
                        'momentum': 0.0
                    })
                
                portfolio_value = sum(assets[asset] * current_prices[asset] for asset in assets.keys()) + cash
                portfolio_history.append({
                    'date': self.data['date'].iloc[i],
                    'portfolio_value': portfolio_value,
                    'asset_weights': {asset: (assets[asset] * current_prices[asset]) / portfolio_value 
                                    for asset in assets.keys()},
                    'total_trades': len(trades)
                })
                continue
            
            # Momentum berechnen (7-Tage-Trend)
            if i >= 7:
                momentum = {}
                for asset in ['BTC', 'ETH', 'BNB']:
                    if asset == 'BTC':
                        prices = btc_prices[max(0, i-7):i+1]
                    elif asset == 'ETH':
                        prices = eth_prices[max(0, i-7):i+1]
                    else:
                        prices = bnb_prices[max(0, i-7):i+1]
                    
                    if len(prices) > 1:
                        momentum[asset] = (prices[-1] - prices[0]) / prices[0]
                    else:
                        momentum[asset] = 0.0
                
                # Momentum-adjustierte Gewichte (reduzierte Anpassung)
                adjusted_weights = {}
                for asset in assets.keys():
                    base_weight = target_weights[asset]
                    momentum_adj = momentum[asset] * 0.05  # 5% Anpassung (weniger aggressiv)
                    adjusted_weights[asset] = max(0.05, min(base_weight + momentum_adj, 0.7))
                
                # Gewichte normalisieren
                total_weight = sum(adjusted_weights.values())
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            else:
                adjusted_weights = target_weights.copy()
                momentum = {asset: 0.0 for asset in assets.keys()}
            
            # Aktueller Portfolio-Wert berechnen
            current_portfolio_value = sum(assets[asset] * current_prices[asset] for asset in assets.keys()) + cash
            
            # Rebalancing ausführen (gelockerte Bedingungen)
            for asset in assets.keys():
                current_value = assets[asset] * current_prices[asset]
                current_weight = current_value / current_portfolio_value if current_portfolio_value > 0 else 0
                target_weight = adjusted_weights[asset]
                
                # Rebalancing wenn Abweichung > 3% (reduziert von 8%)
                weight_deviation = abs(current_weight - target_weight)
                if weight_deviation > 0.03:
                    target_value = current_portfolio_value * target_weight
                    trade_value = target_value - current_value
                    
                    # Transaktionskosten
                    fee = abs(trade_value) * self.trading_fee
                    
                    # Gelockerte Kosten-Nutzen-Prüfung: Benefit > 1.2x Kosten (reduziert von 2x)
                    expected_benefit = weight_deviation * current_portfolio_value * 0.02  # 2% erwarteter Nutzen
                    
                    # Zusätzlich: Minimaler Trade-Betrag von $100
                    min_trade_value = 100.0
                    
                    if expected_benefit > fee * 1.2 and abs(trade_value) > min_trade_value:
                        # Trade ausführen
                        if trade_value > 0:  # Kaufen
                            if cash >= abs(trade_value):  # Genug Cash verfügbar
                                asset_bought = (trade_value - fee) / current_prices[asset]
                                assets[asset] += asset_bought
                                cash -= abs(trade_value)
                                
                                trades.append({
                                    'date': self.data['date'].iloc[i],
                                    'asset': asset,
                                    'action': 'buy',
                                    'amount': asset_bought,
                                    'price': current_prices[asset],
                                    'fee': fee,
                                    'target_weight': target_weight,
                                    'current_weight': current_weight,
                                    'momentum': momentum.get(asset, 0),
                                    'benefit_cost_ratio': expected_benefit / fee if fee > 0 else 0
                                })
                        else:  # Verkaufen
                            asset_sold = abs(trade_value) / current_prices[asset]
                            if assets[asset] >= asset_sold:  # Genug Assets verfügbar
                                assets[asset] -= asset_sold
                                cash += abs(trade_value) - fee
                                
                                trades.append({
                                    'date': self.data['date'].iloc[i],
                                    'asset': asset,
                                    'action': 'sell',
                                    'amount': asset_sold,
                                    'price': current_prices[asset],
                                    'fee': fee,
                                    'target_weight': target_weight,
                                    'current_weight': current_weight,
                                    'momentum': momentum.get(asset, 0),
                                    'benefit_cost_ratio': expected_benefit / fee if fee > 0 else 0
                                })
            
            # Portfolio-Wert neu berechnen
            portfolio_value = sum(assets[asset] * current_prices[asset] for asset in assets.keys()) + cash
            
            portfolio_history.append({
                'date': self.data['date'].iloc[i],
                'portfolio_value': portfolio_value,
                'asset_weights': {asset: (assets[asset] * current_prices[asset]) / portfolio_value 
                                if portfolio_value > 0 else 0 for asset in assets.keys()},
                'cash_weight': cash / portfolio_value if portfolio_value > 0 else 0,
                'total_trades': len(trades)
            })
        
        # Performance berechnen
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        portfolio_df = pd.DataFrame(portfolio_history)
        if len(portfolio_df) > 1:
            portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            portfolio_vol = portfolio_returns.std() * np.sqrt(26) if len(portfolio_returns) > 0 else 0
            sharpe = (portfolio_returns.mean() * 26 - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            portfolio_values = portfolio_df['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = drawdown.max()
        else:
            portfolio_vol = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'strategy': 'Smart Rebalancing',
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365/len(self.data)) - 1,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'total_fees': sum(t['fee'] for t in trades),
            'rebalances_executed': len([t for t in trades if abs(t.get('momentum', 0)) > 0.1]),
            'portfolio_history': portfolio_history[-10:],
            'trades': trades[-5:]
        }


def run_simple_backtest():
    """Führt vereinfachte Backtests durch"""
    logger.info("=== Simple Defensive Strategies Backtest ===")
    
    backtest = SimpleDefensiveBacktest(initial_capital=100000.0)
    
    results = {
        'advanced_portfolio': backtest.backtest_advanced_portfolio(),
        'defensive_volatility': backtest.backtest_defensive_volatility(),
        'smart_rebalancing': backtest.backtest_smart_rebalancing()
    }
    
    # Ergebnisse anzeigen
    print("\n=== BACKTEST ERGEBNISSE ===")
    for strategy_name, result in results.items():
        print(f"\n{result['strategy']}:")
        print(f"  Final Value: ${result['final_value']:,.2f}")
        print(f"  Total Return: {result['total_return']:.2%}")
        print(f"  Annualized Return: {result['annualized_return']:.2%}")
        print(f"  Volatility: {result['volatility']:.2%}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"  Total Trades: {result['total_trades']}")
        print(f"  Total Fees: ${result['total_fees']:.2f}")
    
    # Beste Strategie ermitteln
    best_strategy = max(results.items(), key=lambda x: x[1]['total_return'])
    print(f"\n🏆 BESTE STRATEGIE: {best_strategy[1]['strategy']}")
    print(f"   Return: {best_strategy[1]['total_return']:.2%}")
    print(f"   Sharpe: {best_strategy[1]['sharpe_ratio']:.2f}")
    
    # Ergebnisse speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simple_defensive_backtest_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Ergebnisse gespeichert in {filename}")
    return results


if __name__ == "__main__":
    results = run_simple_backtest()