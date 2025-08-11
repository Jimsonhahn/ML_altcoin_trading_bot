#!/usr/bin/env python3
"""
Defensive Strategies Realistic Backtest
======================================

Backtesting für die drei neuen defensiven Strategien:
- Advanced Portfolio Strategy
- Defensive Volatility Strategy  
- Smart Rebalancing Strategy

Mit realistischen Marktbedingungen und Transaktionskosten.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest-Ergebnisse für eine Strategie"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    avg_trade_return: float
    transaction_costs: float
    performance_metrics: Dict[str, Any]


class DefensiveBacktestEngine:
    """
    Realistic Backtest Engine für defensive Strategien
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        
        # Trading Kosten (realistisch)
        self.maker_fee = 0.001  # 0.1%
        self.taker_fee = 0.001  # 0.1%
        self.slippage = 0.0005  # 0.05%
        
        # Marktdaten (simuliert für Demo)
        self.market_data = self._generate_realistic_market_data()
        
        logger.info("Defensive Backtest Engine initialisiert")
    
    def _generate_realistic_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generiert realistische Marktdaten für Backtesting
        """
        np.random.seed(42)  # Für reproduzierbare Ergebnisse
        
        # 2 Jahre Daten, stündlich
        periods = 24 * 365 * 2  # 2 Jahre
        dates = pd.date_range(start='2022-01-01', periods=periods, freq='H')
        
        market_data = {}
        
        # Basis-Parameter für verschiedene Assets
        asset_params = {
            'BTC/USDT': {'base_price': 40000, 'volatility': 0.05, 'trend': 0.0001},
            'ETH/USDT': {'base_price': 3000, 'volatility': 0.06, 'trend': 0.00008},
            'BNB/USDT': {'base_price': 400, 'volatility': 0.07, 'trend': 0.00005},
            'ADA/USDT': {'base_price': 1.0, 'volatility': 0.08, 'trend': 0.00003},
            'SOL/USDT': {'base_price': 100, 'volatility': 0.09, 'trend': 0.00004}
        }
        
        for symbol, params in asset_params.items():
            # Geometric Brownian Motion mit Regime-Wechseln
            returns = self._generate_regime_aware_returns(periods, params['volatility'])
            
            # Preise generieren
            prices = [params['base_price']]
            for i in range(1, len(returns)):
                price = prices[-1] * (1 + returns[i] + params['trend'])
                prices.append(max(price, 0.01))  # Keine negativen Preise
            
            # OHLCV Daten erstellen
            df = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'volume': np.random.lognormal(15, 1, len(prices))
            })
            
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            market_data[symbol] = df
        
        return market_data
    
    def _generate_regime_aware_returns(self, periods: int, base_vol: float) -> np.ndarray:
        """
        Generiert Returns mit verschiedenen Marktregimen
        """
        returns = []
        
        # Regime-Wechsel alle 3-6 Monate
        regime_length = np.random.randint(2160, 4320)  # 3-6 Monate in Stunden
        current_regime = 0
        
        regimes = [
            {'vol_mult': 0.7, 'trend': 0.0002},   # Bull Market
            {'vol_mult': 1.5, 'trend': -0.0001},  # Bear Market
            {'vol_mult': 1.0, 'trend': 0.0},      # Sideways
            {'vol_mult': 2.5, 'trend': -0.0005}   # Crisis
        ]
        
        for i in range(periods):
            if i % regime_length == 0:
                current_regime = np.random.randint(0, len(regimes))
                regime_length = np.random.randint(2160, 4320)
            
            regime = regimes[current_regime]
            vol = base_vol * regime['vol_mult']
            trend = regime['trend']
            
            ret = np.random.normal(trend, vol)
            returns.append(ret)
        
        return np.array(returns)
    
    def backtest_advanced_portfolio(self) -> BacktestResult:
        """
        Backtest für Advanced Portfolio Strategy
        """
        logger.info("Starte Backtest für Advanced Portfolio Strategy...")
        
        # Portfolio-Parameter
        portfolio_value = self.initial_capital
        positions = {symbol: 0.0 for symbol in self.symbols}
        target_weights = {symbol: 0.2 for symbol in self.symbols}  # Gleichgewichtet
        
        trades = []
        portfolio_history = []
        transaction_costs = 0.0
        
        # Rebalancing alle 24 Stunden
        rebalance_frequency = 24
        
        for i in range(0, len(self.market_data['BTC/USDT']), rebalance_frequency):
            timestamp = self.market_data['BTC/USDT'].iloc[i]['timestamp']
            
            # Aktuelle Portfolio-Bewertung
            current_portfolio_value = 0.0
            current_prices = {}
            
            for symbol in self.symbols:
                if i < len(self.market_data[symbol]):
                    price = self.market_data[symbol].iloc[i]['close']
                    current_prices[symbol] = price
                    current_portfolio_value += positions[symbol] * price
            
            # Cash hinzufügen (vereinfacht)
            cash = portfolio_value - sum(positions[symbol] * current_prices[symbol] 
                                       for symbol in self.symbols if symbol in current_prices)
            current_portfolio_value += cash
            
            # Risk-Parity Rebalancing
            if current_portfolio_value > 0:
                for symbol in self.symbols:
                    if symbol in current_prices:
                        # Volatilitäts-adjustierte Gewichtung
                        vol = self._calculate_volatility(symbol, i)
                        vol_adjusted_weight = target_weights[symbol] * (0.2 / max(vol, 0.01))
                        vol_adjusted_weight = min(vol_adjusted_weight, 0.25)  # Max 25%
                        
                        target_value = current_portfolio_value * vol_adjusted_weight
                        current_value = positions[symbol] * current_prices[symbol]
                        
                        # Trade ausführen wenn Abweichung > 5%
                        if abs(target_value - current_value) > current_portfolio_value * 0.05:
                            trade_value = target_value - current_value
                            trade_size = trade_value / current_prices[symbol]
                            
                            # Transaktionskosten
                            cost = abs(trade_value) * self.taker_fee
                            transaction_costs += cost
                            current_portfolio_value -= cost
                            
                            positions[symbol] += trade_size
                            
                            trades.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'size': trade_size,
                                'price': current_prices[symbol],
                                'value': trade_value,
                                'cost': cost
                            })
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'positions': positions.copy()
            })
            
            portfolio_value = current_portfolio_value
        
        # Performance-Metriken berechnen
        returns = self._calculate_returns(portfolio_history)
        
        return BacktestResult(
            strategy_name="Advanced Portfolio",
            start_date=self.market_data['BTC/USDT'].iloc[0]['timestamp'],
            end_date=self.market_data['BTC/USDT'].iloc[-1]['timestamp'],
            initial_capital=self.initial_capital,
            final_value=portfolio_value,
            total_return=(portfolio_value - self.initial_capital) / self.initial_capital,
            annualized_return=self._calculate_annualized_return(returns),
            volatility=np.std(returns) * np.sqrt(365 * 24),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(portfolio_history),
            calmar_ratio=self._calculate_calmar_ratio(returns, portfolio_history),
            total_trades=len(trades),
            win_rate=self._calculate_win_rate(trades),
            avg_trade_return=np.mean([t.get('value', 0) for t in trades]) if trades else 0,
            transaction_costs=transaction_costs,
            performance_metrics={
                'portfolio_history': portfolio_history[-100:],  # Letzten 100 Punkte
                'trade_summary': {
                    'total_trades': len(trades),
                    'avg_trade_size': np.mean([abs(t.get('value', 0)) for t in trades]) if trades else 0
                }
            }
        )
    
    def backtest_defensive_volatility(self) -> BacktestResult:
        """
        Backtest für Defensive Volatility Strategy
        """
        logger.info("Starte Backtest für Defensive Volatility Strategy...")
        
        portfolio_value = self.initial_capital
        positions = {symbol: 0.0 for symbol in self.symbols}
        trades = []
        portfolio_history = []
        transaction_costs = 0.0
        
        # Trading alle 4 Stunden
        trading_frequency = 4
        
        for i in range(0, len(self.market_data['BTC/USDT']), trading_frequency):
            timestamp = self.market_data['BTC/USDT'].iloc[i]['timestamp']
            
            current_portfolio_value = 0.0
            current_prices = {}
            
            for symbol in self.symbols:
                if i < len(self.market_data[symbol]):
                    price = self.market_data[symbol].iloc[i]['close']
                    current_prices[symbol] = price
                    current_portfolio_value += positions[symbol] * price
            
            # Volatilitäts-basierte Position Sizing
            for symbol in self.symbols:
                if symbol in current_prices and i >= 30:  # Mindestens 30 Perioden für Vol-Berechnung
                    vol = self._calculate_volatility(symbol, i, lookback=30)
                    
                    # Volatilitäts-Regime bestimmen
                    if vol > 0.8:  # Extreme Volatilität
                        position_size = 0.01  # Sehr kleine Position
                    elif vol > 0.4:  # Hohe Volatilität
                        position_size = 0.03  # Kleine Position
                    else:  # Normale Volatilität
                        position_size = 0.08  # Standard Position
                    
                    target_value = current_portfolio_value * position_size
                    current_value = positions[symbol] * current_prices[symbol]
                    
                    # Signal basierend auf Volatilitäts-Regime
                    signal = self._get_volatility_signal(symbol, i, vol)
                    
                    if signal in ['BUY', 'SELL']:
                        if signal == 'BUY' and current_value < target_value:
                            trade_value = target_value - current_value
                        elif signal == 'SELL' and current_value > 0:
                            trade_value = -current_value * 0.5  # Halbe Position verkaufen
                        else:
                            continue
                        
                        trade_size = trade_value / current_prices[symbol]
                        
                        # Transaktionskosten
                        cost = abs(trade_value) * self.taker_fee
                        transaction_costs += cost
                        current_portfolio_value -= cost
                        
                        positions[symbol] += trade_size
                        
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'signal': signal,
                            'size': trade_size,
                            'price': current_prices[symbol],
                            'value': trade_value,
                            'cost': cost,
                            'volatility': vol
                        })
            
            # Portfolio-Wert neu berechnen
            current_portfolio_value = sum(positions[symbol] * current_prices[symbol] 
                                        for symbol in self.symbols if symbol in current_prices)
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'positions': positions.copy()
            })
            
            portfolio_value = current_portfolio_value
        
        # Performance-Metriken berechnen
        returns = self._calculate_returns(portfolio_history)
        
        return BacktestResult(
            strategy_name="Defensive Volatility",
            start_date=self.market_data['BTC/USDT'].iloc[0]['timestamp'],
            end_date=self.market_data['BTC/USDT'].iloc[-1]['timestamp'],
            initial_capital=self.initial_capital,
            final_value=portfolio_value,
            total_return=(portfolio_value - self.initial_capital) / self.initial_capital,
            annualized_return=self._calculate_annualized_return(returns),
            volatility=np.std(returns) * np.sqrt(365 * 24),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(portfolio_history),
            calmar_ratio=self._calculate_calmar_ratio(returns, portfolio_history),
            total_trades=len(trades),
            win_rate=self._calculate_win_rate(trades),
            avg_trade_return=np.mean([t.get('value', 0) for t in trades]) if trades else 0,
            transaction_costs=transaction_costs,
            performance_metrics={
                'portfolio_history': portfolio_history[-100:],
                'volatility_stats': self._calculate_volatility_stats(trades)
            }
        )
    
    def backtest_smart_rebalancing(self) -> BacktestResult:
        """
        Backtest für Smart Rebalancing Strategy
        """
        logger.info("Starte Backtest für Smart Rebalancing Strategy...")
        
        portfolio_value = self.initial_capital
        positions = {symbol: 0.0 for symbol in self.symbols}
        target_weights = {symbol: 0.2 for symbol in self.symbols}
        
        trades = []
        portfolio_history = []
        transaction_costs = 0.0
        last_rebalance = 0
        
        # Rebalancing Check alle 6 Stunden
        check_frequency = 6
        min_rebalance_interval = 24  # Minimum 24h zwischen Rebalances
        
        for i in range(0, len(self.market_data['BTC/USDT']), check_frequency):
            timestamp = self.market_data['BTC/USDT'].iloc[i]['timestamp']
            
            current_portfolio_value = 0.0
            current_prices = {}
            
            for symbol in self.symbols:
                if i < len(self.market_data[symbol]):
                    price = self.market_data[symbol].iloc[i]['close']
                    current_prices[symbol] = price
                    current_portfolio_value += positions[symbol] * price
            
            # Zeit seit letztem Rebalancing
            hours_since_rebalance = (i - last_rebalance)
            
            # Smart Rebalancing Logic
            if hours_since_rebalance >= min_rebalance_interval and current_portfolio_value > 0:
                rebalance_needed = False
                
                for symbol in self.symbols:
                    if symbol in current_prices:
                        current_weight = (positions[symbol] * current_prices[symbol]) / current_portfolio_value
                        target_weight = target_weights[symbol]
                        
                        # Momentum-adjustierte Zielgewichtung
                        momentum = self._calculate_momentum(symbol, i)
                        momentum_adjustment = momentum * 0.1  # 10% Anpassung basierend auf Momentum
                        adjusted_target = target_weight + momentum_adjustment
                        adjusted_target = max(0.05, min(adjusted_target, 0.35))  # 5-35% Grenzen
                        
                        weight_deviation = abs(current_weight - adjusted_target)
                        
                        # Rebalancing wenn Abweichung > 5%
                        if weight_deviation > 0.05:
                            rebalance_needed = True
                            
                            target_value = current_portfolio_value * adjusted_target
                            current_value = positions[symbol] * current_prices[symbol]
                            trade_value = target_value - current_value
                            
                            # Kosten-Nutzen-Analyse
                            transaction_cost = abs(trade_value) * self.taker_fee
                            expected_benefit = weight_deviation * current_portfolio_value * 0.05
                            
                            # Nur traden wenn Nutzen > 2x Kosten
                            if expected_benefit > transaction_cost * 2:
                                trade_size = trade_value / current_prices[symbol]
                                
                                transaction_costs += transaction_cost
                                current_portfolio_value -= transaction_cost
                                
                                positions[symbol] += trade_size
                                
                                trades.append({
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'size': trade_size,
                                    'price': current_prices[symbol],
                                    'value': trade_value,
                                    'cost': transaction_cost,
                                    'current_weight': current_weight,
                                    'target_weight': adjusted_target,
                                    'momentum': momentum,
                                    'benefit_cost_ratio': expected_benefit / transaction_cost
                                })
                
                if rebalance_needed:
                    last_rebalance = i
            
            # Portfolio-Wert neu berechnen
            current_portfolio_value = sum(positions[symbol] * current_prices[symbol] 
                                        for symbol in self.symbols if symbol in current_prices)
            
            portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': current_portfolio_value,
                'positions': positions.copy()
            })
            
            portfolio_value = current_portfolio_value
        
        # Performance-Metriken berechnen
        returns = self._calculate_returns(portfolio_history)
        
        return BacktestResult(
            strategy_name="Smart Rebalancing",
            start_date=self.market_data['BTC/USDT'].iloc[0]['timestamp'],
            end_date=self.market_data['BTC/USDT'].iloc[-1]['timestamp'],
            initial_capital=self.initial_capital,
            final_value=portfolio_value,
            total_return=(portfolio_value - self.initial_capital) / self.initial_capital,
            annualized_return=self._calculate_annualized_return(returns),
            volatility=np.std(returns) * np.sqrt(365 * 24),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(portfolio_history),
            calmar_ratio=self._calculate_calmar_ratio(returns, portfolio_history),
            total_trades=len(trades),
            win_rate=self._calculate_win_rate(trades),
            avg_trade_return=np.mean([t.get('value', 0) for t in trades]) if trades else 0,
            transaction_costs=transaction_costs,
            performance_metrics={
                'portfolio_history': portfolio_history[-100:],
                'rebalancing_stats': self._calculate_rebalancing_stats(trades)
            }
        )
    
    def _calculate_volatility(self, symbol: str, index: int, lookback: int = 24) -> float:
        """Berechnet Rolling Volatility"""
        if index < lookback:
            return 0.3  # Default
        
        data = self.market_data[symbol].iloc[max(0, index-lookback):index]
        returns = data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(24 * 365) if len(returns) > 0 else 0.3
    
    def _get_volatility_signal(self, symbol: str, index: int, volatility: float) -> str:
        """Generiert Signal basierend auf Volatilitäts-Regime"""
        if index < 50:
            return 'HOLD'
        
        data = self.market_data[symbol].iloc[max(0, index-50):index]
        
        # RSI für Überkauft/Überverkauft
        rsi = self._calculate_rsi(data['close'])
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Volatilitäts-basierte Signale
        if volatility > 0.6:  # Hohe Volatilität - Mean Reversion
            if current_rsi < 30:
                return 'BUY'
            elif current_rsi > 70:
                return 'SELL'
        elif volatility < 0.3:  # Niedrige Volatilität - Trend Following
            sma_short = data['close'].rolling(10).mean().iloc[-1]
            sma_long = data['close'].rolling(20).mean().iloc[-1]
            
            if sma_short > sma_long:
                return 'BUY'
            elif sma_short < sma_long:
                return 'SELL'
        
        return 'HOLD'
    
    def _calculate_momentum(self, symbol: str, index: int, lookback: int = 72) -> float:
        """Berechnet Momentum für Smart Rebalancing"""
        if index < lookback:
            return 0.0
        
        data = self.market_data[symbol].iloc[max(0, index-lookback):index]
        returns = data['close'].pct_change().dropna()
        return returns.mean() if len(returns) > 0 else 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Berechnet RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_returns(self, portfolio_history: List[Dict]) -> np.ndarray:
        """Berechnet Portfolio Returns"""
        values = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(values) / values[:-1]
        return returns
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Berechnet annualisierte Rendite"""
        if len(returns) == 0:
            return 0.0
        cumulative_return = np.prod(1 + returns) - 1
        periods_per_year = 365 * 24 / len(returns) * (len(returns))
        return (1 + cumulative_return) ** (365 * 24 / len(returns)) - 1
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Berechnet Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_return = np.mean(returns) - 0.02 / (365 * 24)  # 2% Risk-free rate
        return excess_return / np.std(returns) * np.sqrt(365 * 24)
    
    def _calculate_max_drawdown(self, portfolio_history: List[Dict]) -> float:
        """Berechnet Maximum Drawdown"""
        values = [p['portfolio_value'] for p in portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, portfolio_history: List[Dict]) -> float:
        """Berechnet Calmar Ratio"""
        ann_return = self._calculate_annualized_return(returns)
        max_dd = self._calculate_max_drawdown(portfolio_history)
        return ann_return / max_dd if max_dd > 0 else 0.0
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Berechnet Win Rate"""
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('value', 0) > 0)
        return profitable_trades / len(trades)
    
    def _calculate_volatility_stats(self, trades: List[Dict]) -> Dict[str, Any]:
        """Berechnet Volatilitäts-Statistiken"""
        if not trades:
            return {}
        
        volatilities = [t.get('volatility', 0) for t in trades if 'volatility' in t]
        
        return {
            'avg_volatility': np.mean(volatilities) if volatilities else 0,
            'max_volatility': np.max(volatilities) if volatilities else 0,
            'min_volatility': np.min(volatilities) if volatilities else 0,
            'vol_regime_distribution': {
                'low_vol_trades': sum(1 for v in volatilities if v < 0.3),
                'normal_vol_trades': sum(1 for v in volatilities if 0.3 <= v < 0.6),
                'high_vol_trades': sum(1 for v in volatilities if v >= 0.6)
            }
        }
    
    def _calculate_rebalancing_stats(self, trades: List[Dict]) -> Dict[str, Any]:
        """Berechnet Rebalancing-Statistiken"""
        if not trades:
            return {}
        
        benefit_cost_ratios = [t.get('benefit_cost_ratio', 0) for t in trades 
                              if 'benefit_cost_ratio' in t]
        
        return {
            'avg_benefit_cost_ratio': np.mean(benefit_cost_ratios) if benefit_cost_ratios else 0,
            'min_benefit_cost_ratio': np.min(benefit_cost_ratios) if benefit_cost_ratios else 0,
            'max_benefit_cost_ratio': np.max(benefit_cost_ratios) if benefit_cost_ratios else 0,
            'rebalances_per_asset': {
                symbol: sum(1 for t in trades if t.get('symbol') == symbol)
                for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            }
        }


def run_defensive_strategies_backtest():
    """
    Führt Backtests für alle defensive Strategien durch
    """
    logger.info("=== Defensive Strategies Backtest gestartet ===")
    
    engine = DefensiveBacktestEngine(initial_capital=100000.0)
    results = {}
    
    # Backtest für jede Strategie
    strategies = [
        ('advanced_portfolio', engine.backtest_advanced_portfolio),
        ('defensive_volatility', engine.backtest_defensive_volatility),
        ('smart_rebalancing', engine.backtest_smart_rebalancing)
    ]
    
    for strategy_name, backtest_func in strategies:
        logger.info(f"Starte Backtest für {strategy_name}...")
        try:
            result = backtest_func()
            results[strategy_name] = result
            
            logger.info(f"{strategy_name} Backtest abgeschlossen:")
            logger.info(f"  Total Return: {result.total_return:.2%}")
            logger.info(f"  Annualized Return: {result.annualized_return:.2%}")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"  Total Trades: {result.total_trades}")
            logger.info(f"  Transaction Costs: ${result.transaction_costs:.2f}")
            
        except Exception as e:
            logger.error(f"Fehler bei {strategy_name} Backtest: {e}")
            continue
    
    # Ergebnisse vergleichen
    comparison = compare_strategies(results)
    
    # Ergebnisse speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"defensive_strategies_backtest_{timestamp}.json"
    
    save_results(results, comparison, filename)
    
    logger.info("=== Defensive Strategies Backtest abgeschlossen ===")
    return results, comparison


def compare_strategies(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
    """
    Vergleicht die Performance der Strategien
    """
    if not results:
        return {}
    
    comparison = {
        'strategy_rankings': {},
        'performance_summary': {},
        'risk_metrics': {}
    }
    
    # Performance Ranking
    strategies_by_return = sorted(results.items(), 
                                key=lambda x: x[1].total_return, reverse=True)
    strategies_by_sharpe = sorted(results.items(), 
                                key=lambda x: x[1].sharpe_ratio, reverse=True)
    strategies_by_drawdown = sorted(results.items(), 
                                  key=lambda x: x[1].max_drawdown)
    
    comparison['strategy_rankings'] = {
        'by_total_return': [(name, res.total_return) for name, res in strategies_by_return],
        'by_sharpe_ratio': [(name, res.sharpe_ratio) for name, res in strategies_by_sharpe],
        'by_max_drawdown': [(name, res.max_drawdown) for name, res in strategies_by_drawdown]
    }
    
    # Performance Summary
    comparison['performance_summary'] = {
        name: {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'transaction_costs': result.transaction_costs,
            'win_rate': result.win_rate
        }
        for name, result in results.items()
    }
    
    # Beste Strategie overall (gewichteter Score)
    best_strategy = None
    best_score = -999
    
    for name, result in results.items():
        # Gewichteter Score: Return (40%) + Sharpe (30%) + Low Drawdown (30%)
        score = (result.total_return * 0.4 + 
                result.sharpe_ratio * 0.1 * 0.3 + 
                (1 - result.max_drawdown) * 0.3)
        
        if score > best_score:
            best_score = score
            best_strategy = name
    
    comparison['best_strategy'] = {
        'name': best_strategy,
        'score': best_score,
        'reason': 'Beste Kombination aus Rendite, Sharpe Ratio und niedrigem Drawdown'
    }
    
    return comparison


def save_results(results: Dict[str, BacktestResult], comparison: Dict[str, Any], filename: str):
    """
    Speichert Backtest-Ergebnisse in JSON-Datei
    """
    output = {
        'timestamp': datetime.now().isoformat(),
        'backtest_settings': {
            'initial_capital': 100000.0,
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'slippage': 0.0005,
            'backtest_period': '2 years',
            'frequency': 'hourly'
        },
        'results': {},
        'comparison': comparison
    }
    
    # Ergebnisse serialisierbar machen
    for name, result in results.items():
        output['results'][name] = {
            'strategy_name': result.strategy_name,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_capital': result.initial_capital,
            'final_value': result.final_value,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'avg_trade_return': result.avg_trade_return,
            'transaction_costs': result.transaction_costs,
            'performance_metrics': result.performance_metrics
        }
    
    # JSON speichern
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Ergebnisse gespeichert in {filename}")


if __name__ == "__main__":
    results, comparison = run_defensive_strategies_backtest()
    
    print("\n=== DEFENSIVE STRATEGIES BACKTEST ZUSAMMENFASSUNG ===")
    print(f"Beste Strategie: {comparison.get('best_strategy', {}).get('name', 'N/A')}")
    print(f"Grund: {comparison.get('best_strategy', {}).get('reason', 'N/A')}")
    
    print("\n=== PERFORMANCE RANKING ===")
    for i, (name, return_val) in enumerate(comparison.get('strategy_rankings', {}).get('by_total_return', []), 1):
        print(f"{i}. {name}: {return_val:.2%} Total Return")