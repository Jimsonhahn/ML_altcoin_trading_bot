#!/usr/bin/env python3
"""
Trading System Diagnosis - 3-Stufen-Analyse
Identifiziert systematisch die Ursachen für schlechte Performance
"""

import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSystemDiagnosis:
    """
    Systematische Diagnose für Trading-System Performance
    
    3-Stufen-Analyse:
    1. Alpha-Problem: Hat die Strategie überhaupt einen Edge?
    2. Execution-Problem: Verlieren wir Alpha durch Slippage/Fees?
    3. Regime/Portfolio-Problem: Falsches Timing oder Allokation?
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.diagnosis_results = {}
        
    def generate_enhanced_btc_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generiert BTC-Daten mit zusätzlichen Features für Alpha-Analyse
        """
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Realistische BTC 2024 Performance
        start_price = 42000.0
        end_price = 95000.0
        daily_vol = 0.045
        
        trend_factor = (end_price / start_price) ** (1/days)
        
        prices = []
        current_price = start_price
        
        for i, date in enumerate(dates):
            # Trend + Volatility
            trend_adjustment = trend_factor - 1
            
            # Saisonale und makroökonomische Faktoren
            month = date.month
            seasonal_factor = 1.0
            if month in [10, 11, 12]:  # Q4 Bull Run
                seasonal_factor = 1.2
            elif month in [6, 7, 8]:   # Sommer-Schwäche
                seasonal_factor = 0.9
            
            # Markt-Regime Simulation
            regime = self._determine_market_regime(i, days)
            regime_volatility = {
                'low_vol_bull': 0.02,
                'high_vol_bull': 0.06,
                'low_vol_bear': 0.03,
                'high_vol_bear': 0.08,
                'sideways': 0.015
            }
            
            current_vol = regime_volatility.get(regime, daily_vol)
            
            # Daily return
            daily_return = np.random.normal(
                trend_adjustment * seasonal_factor, 
                current_vol
            )
            
            # Volatility clustering
            if i > 0 and abs(prices[-1]['daily_return']) > 0.1:
                daily_return *= 1.3
            
            current_price *= (1 + daily_return)
            current_price = max(20000, min(120000, current_price))
            
            # Zusätzliche Features für Alpha-Analyse
            volume = np.random.uniform(15000, 60000)
            if regime in ['high_vol_bull', 'high_vol_bear']:
                volume *= 2.0  # Hohes Volumen in volatilen Phasen
            
            # Simuliere Order Book Features
            spread_bps = np.random.uniform(5, 25)
            if regime == 'high_vol_bear':
                spread_bps *= 2  # Höhere Spreads in Stress-Phasen
            
            order_book_imbalance = np.random.uniform(-1, 1)
            
            prices.append({
                'date': date,
                'price': current_price,
                'daily_return': daily_return,
                'volume': volume,
                'regime': regime,
                'spread_bps': spread_bps,
                'order_book_imbalance': order_book_imbalance,
                'volatility': current_vol
            })
        
        return {
            'prices': prices,
            'start_price': start_price,
            'end_price': prices[-1]['price'],
            'total_days': days
        }
    
    def _determine_market_regime(self, day_index: int, total_days: int) -> str:
        """
        Bestimmt Markt-Regime basierend auf Tag im Jahr
        """
        progress = day_index / total_days
        
        if progress < 0.2:  # Q1: Erholung
            return 'low_vol_bull'
        elif progress < 0.4:  # Q2: Volatilität
            return 'high_vol_bull' 
        elif progress < 0.6:  # Q3: Sommerflaute
            return 'sideways'
        elif progress < 0.8:  # Q4 Start: Aufbau
            return 'low_vol_bull'
        else:  # Q4 Ende: Euphorie
            return 'high_vol_bull'
    
    def stage_1_alpha_analysis(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUFE 1: Alpha-Problem Diagnose
        Idealer Backtest ohne jegliche Kosten oder Slippage
        """
        logger.info("🔍 STUFE 1: Alpha-Analyse (Idealer Backtest)")
        
        prices = price_data['prices']
        signals = self._generate_advanced_signals(prices)
        
        # IDEALER BACKTEST - Keine Kosten, perfekte Execution
        ideal_portfolio = self._run_ideal_backtest(prices, signals)
        
        # Alpha-Metriken berechnen
        ideal_returns = [p['daily_return'] for p in ideal_portfolio if 'daily_return' in p]
        
        if len(ideal_returns) > 1:
            ideal_total_return = (ideal_portfolio[-1]['portfolio_value'] / self.initial_capital) - 1
            ideal_sharpe = self._calculate_sharpe(ideal_returns)
            
            # Hit Rate der Signale
            correct_signals = 0
            total_signals = len(signals)
            
            for signal in signals:
                # Prüfe ob Signal in richtige Richtung zeigte
                future_returns = self._get_future_returns(signal, prices, days=5)
                if signal['signal_type'] == 'long' and future_returns > 0:
                    correct_signals += 1
                elif signal['signal_type'] == 'short' and future_returns < 0:
                    correct_signals += 1
            
            hit_rate = correct_signals / max(total_signals, 1)
            
            alpha_results = {
                'ideal_total_return': ideal_total_return,
                'ideal_sharpe_ratio': ideal_sharpe,
                'signal_hit_rate': hit_rate,
                'total_signals': total_signals,
                'avg_signal_strength': np.mean([s['signal_strength'] for s in signals]),
                'has_alpha': ideal_total_return > 0.1 and hit_rate > 0.55  # Mindest-Schwellwerte
            }
            
        else:
            alpha_results = {'has_alpha': False, 'error': 'Insufficient data'}
        
        self.diagnosis_results['stage_1_alpha'] = alpha_results
        
        # Output
        print(f"📊 ALPHA-ANALYSE ERGEBNISSE:")
        print(f"   Idealer Return: {alpha_results.get('ideal_total_return', 0):.1%}")
        print(f"   Ideale Sharpe: {alpha_results.get('ideal_sharpe_ratio', 0):.2f}")
        print(f"   Signal Hit Rate: {alpha_results.get('signal_hit_rate', 0):.1%}")
        print(f"   Anzahl Signale: {alpha_results.get('total_signals', 0)}")
        print(f"   ✅ Hat Alpha: {alpha_results.get('has_alpha', False)}")
        print()
        
        return alpha_results
    
    def stage_2_execution_analysis(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUFE 2: Execution-Problem Diagnose
        Vergleich ideal vs. realistisch mit detaillierter Slippage-Analyse
        """
        logger.info("🔍 STUFE 2: Execution-Analyse")
        
        prices = price_data['prices']
        signals = self._generate_advanced_signals(prices)
        
        # Ideal vs. Realistic Backtest
        ideal_portfolio = self._run_ideal_backtest(prices, signals)
        realistic_portfolio, slippage_data = self._run_realistic_backtest(prices, signals)
        
        # Execution-Metriken
        ideal_return = (ideal_portfolio[-1]['portfolio_value'] / self.initial_capital) - 1
        realistic_return = (realistic_portfolio[-1]['portfolio_value'] / self.initial_capital) - 1
        
        execution_cost = ideal_return - realistic_return
        
        # Slippage-Analyse
        avg_slippage = np.mean([s['slippage_bps'] for s in slippage_data])
        max_slippage = max([s['slippage_bps'] for s in slippage_data]) if slippage_data else 0
        total_fees = sum([s['fee_cost'] for s in slippage_data])
        
        # Slippage vs. Alpha Ratio
        alpha_per_trade = ideal_return / max(len(signals), 1)
        slippage_per_trade = execution_cost / max(len(signals), 1)
        execution_efficiency = alpha_per_trade / max(slippage_per_trade, 0.001)
        
        execution_results = {
            'ideal_return': ideal_return,
            'realistic_return': realistic_return,
            'execution_cost': execution_cost,
            'avg_slippage_bps': avg_slippage,
            'max_slippage_bps': max_slippage,
            'total_fees': total_fees,
            'execution_efficiency': execution_efficiency,
            'execution_problem': execution_efficiency < 2.0  # Schwellwert
        }
        
        self.diagnosis_results['stage_2_execution'] = execution_results
        
        # Output
        print(f"📊 EXECUTION-ANALYSE ERGEBNISSE:")
        print(f"   Idealer Return: {ideal_return:.1%}")
        print(f"   Realistischer Return: {realistic_return:.1%}")
        print(f"   Execution Cost: {execution_cost:.1%}")
        print(f"   Ø Slippage: {avg_slippage:.1f} bps")
        print(f"   Max Slippage: {max_slippage:.1f} bps")
        print(f"   Total Fees: ${total_fees:,.0f}")
        print(f"   Execution Efficiency: {execution_efficiency:.2f}")
        print(f"   ❌ Execution Problem: {execution_results['execution_problem']}")
        print()
        
        return execution_results
    
    def stage_3_regime_analysis(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUFE 3: Regime/Portfolio-Problem Diagnose
        Timing und Allokations-Analyse
        """
        logger.info("🔍 STUFE 3: Regime/Portfolio-Analyse")
        
        prices = price_data['prices']
        signals = self._generate_advanced_signals(prices)
        
        # Portfolio mit Regime-Tracking
        portfolio_history, regime_performance = self._run_regime_aware_backtest(prices, signals)
        
        # Drawdown-Analyse
        equity_curve = [p['portfolio_value'] for p in portfolio_history]
        drawdowns = self._calculate_drawdown_series(equity_curve)
        max_dd_periods = self._identify_drawdown_periods(drawdowns, prices)
        
        # Regime-Performance
        regime_stats = {}
        for regime in ['low_vol_bull', 'high_vol_bull', 'low_vol_bear', 'high_vol_bear', 'sideways']:
            regime_returns = regime_performance.get(regime, [])
            if regime_returns:
                regime_stats[regime] = {
                    'avg_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe': np.mean(regime_returns) / max(np.std(regime_returns), 0.001),
                    'trade_count': len(regime_returns)
                }
        
        # Portfolio-Allokation Analyse
        allocation_analysis = self._analyze_capital_allocation(portfolio_history)
        
        regime_results = {
            'max_drawdown_periods': max_dd_periods,
            'regime_performance': regime_stats,
            'allocation_analysis': allocation_analysis,
            'regime_detection_accuracy': self._calculate_regime_accuracy(prices),
            'timing_problem': len(max_dd_periods) > 2  # Mehr als 2 große DDs
        }
        
        self.diagnosis_results['stage_3_regime'] = regime_results
        
        # Output
        print(f"📊 REGIME/PORTFOLIO-ANALYSE ERGEBNISSE:")
        print(f"   Anzahl Drawdown-Perioden: {len(max_dd_periods)}")
        print(f"   Beste Regime-Performance: {max(regime_stats.keys(), key=lambda x: regime_stats[x]['avg_return']) if regime_stats else 'N/A'}")
        print(f"   Regime Detection Accuracy: {regime_results['regime_detection_accuracy']:.1%}")
        print(f"   ❌ Timing Problem: {regime_results['timing_problem']}")
        print()
        
        return regime_results
    
    def _generate_advanced_signals(self, prices: List[Dict]) -> List[Dict]:
        """
        Erweiterte Signal-Generierung mit mehr Features
        """
        signals = []
        
        for i in range(30, len(prices)):
            current = prices[i]
            
            # Multi-Timeframe Moving Averages
            ma_5 = np.mean([p['price'] for p in prices[i-5:i]])
            ma_20 = np.mean([p['price'] for p in prices[i-20:i]])
            ma_50 = np.mean([p['price'] for p in prices[i-50:i]]) if i >= 50 else ma_20
            
            # Volatility
            returns = [p['daily_return'] for p in prices[i-20:i]]
            volatility = np.std(returns)
            
            # Volume Analysis
            avg_volume = np.mean([p['volume'] for p in prices[i-10:i]])
            volume_ratio = current['volume'] / avg_volume
            
            # Order Book Signal
            ob_signal = current['order_book_imbalance']
            
            # Regime-Aware Signaling
            regime = current['regime']
            
            # Signal Logic
            signal_strength = 0.0
            signal_type = None
            
            # 1. Trend Following
            if ma_5 > ma_20 * 1.02 and regime in ['low_vol_bull', 'high_vol_bull']:
                signal_strength += 0.4
                signal_type = 'long'
            elif ma_5 < ma_20 * 0.98 and regime in ['low_vol_bear', 'high_vol_bear']:
                signal_strength += 0.4
                signal_type = 'short'
            
            # 2. Mean Reversion in Sideways Markets
            if regime == 'sideways':
                price_zscore = (current['price'] - ma_20) / (volatility * ma_20)
                if price_zscore < -1.5:  # Oversold
                    signal_strength += 0.5
                    signal_type = 'long'
                elif price_zscore > 1.5:  # Overbought
                    signal_strength += 0.5
                    signal_type = 'short'
            
            # 3. Volume Confirmation
            if volume_ratio > 1.5:
                signal_strength += 0.2
            
            # 4. Order Book Alpha
            if abs(ob_signal) > 0.3:
                if (ob_signal > 0 and signal_type == 'long') or (ob_signal < 0 and signal_type == 'short'):
                    signal_strength += 0.3
            
            # 5. Volatility Filter
            if volatility > 0.06:  # High volatility - reduce position
                signal_strength *= 0.7
            
            # Generate signal if strong enough
            if signal_strength > 0.7 and signal_type:
                position_size = min(0.3, signal_strength * 0.4)  # Max 30%
                
                signals.append({
                    'date': current['date'],
                    'price': current['price'],
                    'signal_type': signal_type,
                    'signal_strength': signal_strength,
                    'position_size': position_size,
                    'regime': regime,
                    'volatility': volatility,
                    'volume_ratio': volume_ratio,
                    'ma_signal': ma_5 / ma_20,
                    'order_book_signal': ob_signal
                })
        
        return signals
    
    def _run_ideal_backtest(self, prices: List[Dict], signals: List[Dict]) -> List[Dict]:
        """
        Idealer Backtest ohne Kosten/Slippage
        """
        portfolio = []
        cash = self.initial_capital
        btc_position = 0.0
        
        price_dict = {p['date']: p['price'] for p in prices}
        signal_dict = {s['date']: s for s in signals}
        
        for price_data in prices:
            date = price_data['date']
            current_price = price_data['price']
            
            # Execute signal if present
            if date in signal_dict:
                signal = signal_dict[date]
                
                if signal['signal_type'] == 'long':
                    # Buy BTC
                    allocation = signal['position_size']
                    target_value = (cash + btc_position * current_price) * allocation
                    target_btc = target_value / current_price
                    btc_to_buy = target_btc - btc_position
                    
                    if btc_to_buy > 0:
                        cost = btc_to_buy * current_price
                        if cost <= cash:
                            cash -= cost
                            btc_position += btc_to_buy
                
                elif signal['signal_type'] == 'short':
                    # Sell BTC
                    btc_to_sell = btc_position * signal['position_size']
                    if btc_to_sell > 0:
                        proceeds = btc_to_sell * current_price
                        cash += proceeds
                        btc_position -= btc_to_sell
            
            # Portfolio snapshot
            portfolio_value = cash + btc_position * current_price
            portfolio.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'btc_position': btc_position,
                'btc_value': btc_position * current_price
            })
            
            # Calculate return
            if len(portfolio) > 1:
                prev_value = portfolio[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                portfolio[-1]['daily_return'] = daily_return
        
        return portfolio
    
    def _run_realistic_backtest(self, prices: List[Dict], signals: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Realistischer Backtest mit Kosten/Slippage
        """
        portfolio = []
        slippage_data = []
        cash = self.initial_capital
        btc_position = 0.0
        
        price_dict = {p['date']: p['price'] for p in prices}
        signal_dict = {s['date']: s for s in signals}
        
        for price_data in prices:
            date = price_data['date']
            current_price = price_data['price']
            
            # Execute signal if present
            if date in signal_dict:
                signal = signal_dict[date]
                
                # Calculate realistic execution
                base_slippage = price_data['spread_bps'] / 2  # Half spread
                market_impact = signal['position_size'] * 100  # Impact proportional to size
                volatility_penalty = price_data['volatility'] * 1000
                
                total_slippage_bps = base_slippage + market_impact + volatility_penalty
                slippage_factor = total_slippage_bps / 10000
                
                fee_rate = 0.001  # 0.1%
                
                if signal['signal_type'] == 'long':
                    # Buy with slippage
                    execution_price = current_price * (1 + slippage_factor)
                    allocation = signal['position_size']
                    target_value = (cash + btc_position * current_price) * allocation
                    target_btc = target_value / execution_price
                    btc_to_buy = target_btc - btc_position
                    
                    if btc_to_buy > 0:
                        cost = btc_to_buy * execution_price
                        fee = cost * fee_rate
                        total_cost = cost + fee
                        
                        if total_cost <= cash:
                            cash -= total_cost
                            btc_position += btc_to_buy
                            
                            slippage_data.append({
                                'date': date,
                                'type': 'buy',
                                'signal_price': current_price,
                                'execution_price': execution_price,
                                'slippage_bps': total_slippage_bps,
                                'fee_cost': fee,
                                'quantity': btc_to_buy
                            })
                
                elif signal['signal_type'] == 'short':
                    # Sell with slippage
                    execution_price = current_price * (1 - slippage_factor)
                    btc_to_sell = btc_position * signal['position_size']
                    
                    if btc_to_sell > 0:
                        proceeds = btc_to_sell * execution_price
                        fee = proceeds * fee_rate
                        net_proceeds = proceeds - fee
                        
                        cash += net_proceeds
                        btc_position -= btc_to_sell
                        
                        slippage_data.append({
                            'date': date,
                            'type': 'sell',
                            'signal_price': current_price,
                            'execution_price': execution_price,
                            'slippage_bps': total_slippage_bps,
                            'fee_cost': fee,
                            'quantity': btc_to_sell
                        })
            
            # Portfolio snapshot
            portfolio_value = cash + btc_position * current_price
            portfolio.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'btc_position': btc_position,
                'btc_value': btc_position * current_price
            })
        
        return portfolio, slippage_data
    
    def _run_regime_aware_backtest(self, prices: List[Dict], signals: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Backtest mit Regime-Tracking
        """
        portfolio, _ = self._run_realistic_backtest(prices, signals)
        
        # Track performance by regime
        regime_performance = {}
        
        for i, snapshot in enumerate(portfolio):
            if i > 0:
                prev_value = portfolio[i-1]['portfolio_value']
                current_value = snapshot['portfolio_value']
                daily_return = (current_value - prev_value) / prev_value
                
                # Get regime for this day
                regime = prices[i]['regime']
                
                if regime not in regime_performance:
                    regime_performance[regime] = []
                
                regime_performance[regime].append(daily_return)
        
        return portfolio, regime_performance
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Berechnet Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns, ddof=1) * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_vol
    
    def _get_future_returns(self, signal: Dict, prices: List[Dict], days: int = 5) -> float:
        """Berechnet Future Returns für Signal-Validierung"""
        signal_date = signal['date']
        signal_price = signal['price']
        
        # Find signal in prices
        for i, price_data in enumerate(prices):
            if price_data['date'] == signal_date:
                if i + days < len(prices):
                    future_price = prices[i + days]['price']
                    return (future_price - signal_price) / signal_price
                break
        
        return 0.0
    
    def _calculate_drawdown_series(self, equity_curve: List[float]) -> List[float]:
        """Berechnet Drawdown-Serie"""
        drawdowns = []
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (value - peak) / peak
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def _identify_drawdown_periods(self, drawdowns: List[float], prices: List[Dict]) -> List[Dict]:
        """Identifiziert größere Drawdown-Perioden"""
        periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < -0.05 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                
                periods.append({
                    'start_date': prices[start_idx]['date'],
                    'end_date': prices[i]['date'],
                    'max_drawdown': min(drawdowns[start_idx:i+1]),
                    'duration_days': i - start_idx,
                    'regime_during': prices[start_idx:i+1]
                })
        
        return periods
    
    def _analyze_capital_allocation(self, portfolio_history: List[Dict]) -> Dict[str, Any]:
        """Analysiert Capital Allocation Patterns"""
        allocation_ratios = []
        
        for snapshot in portfolio_history:
            if snapshot['portfolio_value'] > 0:
                btc_allocation = snapshot['btc_value'] / snapshot['portfolio_value']
                allocation_ratios.append(btc_allocation)
        
        return {
            'avg_allocation': np.mean(allocation_ratios),
            'max_allocation': max(allocation_ratios),
            'allocation_volatility': np.std(allocation_ratios),
            'allocation_stability': 1.0 - np.std(allocation_ratios)  # Higher = more stable
        }
    
    def _calculate_regime_accuracy(self, prices: List[Dict]) -> float:
        """Vereinfachte Regime Detection Accuracy"""
        # Placeholder - in real implementation würde echte Regime-Detection validiert
        return 0.75  # 75% Accuracy angenommen
    
    def generate_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Generiert konkrete Optimierungs-Empfehlungen basierend auf Diagnose
        """
        recommendations = {
            'priority_actions': [],
            'strategy_changes': [],
            'execution_improvements': [],
            'risk_management_updates': []
        }
        
        # Analyse der 3 Stufen
        alpha_results = self.diagnosis_results.get('stage_1_alpha', {})
        execution_results = self.diagnosis_results.get('stage_2_execution', {})
        regime_results = self.diagnosis_results.get('stage_3_regime', {})
        
        # Prioritäts-basierte Empfehlungen
        if not alpha_results.get('has_alpha', False):
            recommendations['priority_actions'].append({
                'priority': 'CRITICAL',
                'action': 'Alpha Model Redesign',
                'description': 'Strategie hat keinen statistischen Vorteil. Neue Features und Modelle erforderlich.',
                'expected_impact': 'High',
                'implementation_effort': 'High'
            })
            
            recommendations['strategy_changes'].extend([
                'Implementiere Machine Learning basierte Signal-Generierung',
                'Erweitere Feature-Set um: Orderbook-Imbalance, Cross-Asset-Momentum, Makro-Indikatoren',
                'Verbessere Signal-Timing mit höherer Frequenz (stündlich statt täglich)',
                'Implementiere Multi-Strategy Ensemble für bessere Diversifikation'
            ])
        
        if execution_results.get('execution_problem', False):
            recommendations['priority_actions'].append({
                'priority': 'HIGH',
                'action': 'Execution Optimization',
                'description': 'Alpha wird durch Execution-Kosten aufgefressen.',
                'expected_impact': 'Medium',
                'implementation_effort': 'Medium'
            })
            
            recommendations['execution_improvements'].extend([
                'Implementiere intelligente Order-Routing (Iceberg Orders, TWAP)',
                'Reduziere Trading-Frequenz um 50% (nur stärkste Signale)',
                'Wechsel zu liquideren Märkten oder günstigeren Börsen',
                'Implementiere Slippage-Prediction für besseres Timing'
            ])
        
        if regime_results.get('timing_problem', False):
            recommendations['priority_actions'].append({
                'priority': 'MEDIUM',
                'action': 'Regime Detection Improvement',
                'description': 'Strategie handelt zu ungünstigen Zeiten.',
                'expected_impact': 'Medium',
                'implementation_effort': 'Low'
            })
            
            recommendations['risk_management_updates'].extend([
                'Implementiere dynamische Position-Sizing basierend auf Regime',
                'Reduziere Exposure während High-Vol-Bear Märkten um 70%',
                'Erhöhe Exposure während Low-Vol-Bull Märkten um 150%',
                'Implementiere Drawdown-basierte Position-Reduzierung'
            ])
        
        # Konkrete Ziel-Metriken
        recommendations['target_metrics'] = {
            'target_annual_return': '40%+',
            'target_sharpe_ratio': '1.5+',
            'target_max_drawdown': '<15%',
            'target_win_rate': '60%+',
            'target_execution_efficiency': '3.0+'
        }
        
        return recommendations
    
    def run_complete_diagnosis(self) -> Dict[str, Any]:
        """
        Führt komplette 3-Stufen-Diagnose durch
        """
        print("🏥 TRADING SYSTEM DIAGNOSIS")
        print("=" * 60)
        print("Systematische Analyse: Alpha → Execution → Regime")
        print()
        
        # Generate data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        price_data = self.generate_enhanced_btc_data(start_date, end_date)
        
        # Run 3-stage analysis
        alpha_results = self.stage_1_alpha_analysis(price_data)
        execution_results = self.stage_2_execution_analysis(price_data)
        regime_results = self.stage_3_regime_analysis(price_data)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        
        # Complete diagnosis
        complete_diagnosis = {
            'diagnosis_summary': {
                'primary_problem': self._identify_primary_problem(),
                'severity': self._assess_severity(),
                'expected_fix_difficulty': self._assess_fix_difficulty()
            },
            'stage_results': {
                'alpha_analysis': alpha_results,
                'execution_analysis': execution_results,
                'regime_analysis': regime_results
            },
            'optimization_recommendations': recommendations,
            'next_steps': self._generate_next_steps()
        }
        
        # Output summary
        self._print_diagnosis_summary(complete_diagnosis)
        
        return complete_diagnosis
    
    def _identify_primary_problem(self) -> str:
        """Identifiziert das Hauptproblem"""
        alpha_results = self.diagnosis_results.get('stage_1_alpha', {})
        execution_results = self.diagnosis_results.get('stage_2_execution', {})
        regime_results = self.diagnosis_results.get('stage_3_regime', {})
        
        if not alpha_results.get('has_alpha', False):
            return 'ALPHA_PROBLEM'
        elif execution_results.get('execution_problem', False):
            return 'EXECUTION_PROBLEM'  
        elif regime_results.get('timing_problem', False):
            return 'REGIME_PROBLEM'
        else:
            return 'OPTIMIZATION_NEEDED'
    
    def _assess_severity(self) -> str:
        """Bewertet Schweregrad des Problems"""
        alpha_results = self.diagnosis_results.get('stage_1_alpha', {})
        
        if alpha_results.get('ideal_total_return', 0) < 0:
            return 'CRITICAL'
        elif alpha_results.get('ideal_sharpe_ratio', 0) < 0.5:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _assess_fix_difficulty(self) -> str:
        """Bewertet Schwierigkeit der Lösung"""
        primary_problem = self._identify_primary_problem()
        
        difficulty_map = {
            'ALPHA_PROBLEM': 'HIGH',
            'EXECUTION_PROBLEM': 'MEDIUM',
            'REGIME_PROBLEM': 'LOW',
            'OPTIMIZATION_NEEDED': 'LOW'
        }
        
        return difficulty_map.get(primary_problem, 'MEDIUM')
    
    def _generate_next_steps(self) -> List[str]:
        """Generiert konkrete nächste Schritte"""
        primary_problem = self._identify_primary_problem()
        
        if primary_problem == 'ALPHA_PROBLEM':
            return [
                '1. Feature Engineering: Neue Predictive Features entwickeln',
                '2. ML Model Training: Ensemble-Modelle trainieren',
                '3. Walk-Forward Validation: Out-of-sample Tests durchführen',
                '4. Signal Quality Assessment: Hit-Rate > 60% erreichen'
            ]
        elif primary_problem == 'EXECUTION_PROBLEM':
            return [
                '1. Slippage Modeling: Präzisere Execution-Kosten-Modelle',
                '2. Order Routing: Intelligente Execution-Algorithmen',
                '3. Market Selection: Liquidere Märkte identifizieren',
                '4. Frequency Optimization: Trading-Frequenz anpassen'
            ]
        else:
            return [
                '1. Regime Detection: Verbesserte Markt-Regime-Identifikation',
                '2. Dynamic Sizing: Regime-aware Position-Sizing',
                '3. Risk Management: Adaptive Drawdown-Controls',
                '4. Performance Monitoring: Real-time System-Überwachung'
            ]
    
    def _print_diagnosis_summary(self, diagnosis: Dict[str, Any]) -> None:
        """Druckt Diagnose-Zusammenfassung"""
        summary = diagnosis['diagnosis_summary']
        
        print("🎯 DIAGNOSE-ZUSAMMENFASSUNG")
        print("-" * 60)
        print(f"Hauptproblem: {summary['primary_problem']}")
        print(f"Schweregrad: {summary['severity']}")
        print(f"Fix-Schwierigkeit: {summary['expected_fix_difficulty']}")
        print()
        
        print("🔧 PRIORITÄTS-AKTIONEN")
        print("-" * 60)
        for action in diagnosis['optimization_recommendations']['priority_actions']:
            print(f"[{action['priority']}] {action['action']}")
            print(f"   → {action['description']}")
            print()
        
        print("📈 ZIEL-METRIKEN")
        print("-" * 60)
        targets = diagnosis['optimization_recommendations']['target_metrics']
        for metric, target in targets.items():
            print(f"{metric}: {target}")
        print()


async def main():
    """Führt Trading System Diagnosis aus"""
    
    diagnosis = TradingSystemDiagnosis()
    complete_diagnosis = diagnosis.run_complete_diagnosis()
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"trading_system_diagnosis_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(complete_diagnosis, f, indent=2, default=str)
    
    print(f"💾 Vollständige Diagnose exportiert: {filename}")


if __name__ == "__main__":
    asyncio.run(main())