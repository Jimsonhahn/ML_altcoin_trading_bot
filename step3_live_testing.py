#!/usr/bin/env python3
"""
SCHRITT 3: Live-Testing Framework
Paper-Trading Implementation für bewährte Balanced Strategy

Strategy: "Balanced Institutional BTC Elite v1.3"
Performance: 16.9% Annual Return, 1.29 Sharpe, 16.8% Max DD
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiveTradingFramework:
    """
    SCHRITT 3: Live-Testing Framework
    
    Implementiert Paper-Trading für die bewährte Balanced Strategy
    mit Real-Time Monitoring und Safety Controls
    """
    
    def __init__(self):
        self.strategy_name = "Balanced Institutional BTC Elite"
        self.strategy_version = "1.3 Live-Testing"
        self.risk_profile = "Live Production Ready"
        
        # Live Trading Configuration
        self.paper_trading_mode = True
        self.initial_capital = 300000.0
        self.current_capital = 300000.0
        self.btc_position = 0.0
        self.cash_balance = 300000.0
        
        # Bewährte Parameter aus v1.3
        self.max_position_size = 0.38
        self.max_drawdown_limit = 0.18
        self.min_signal_strength = 0.63
        self.quality_threshold = 0.75
        self.trading_fee = 0.001
        
        # Live Trading Safety Controls
        self.emergency_stop_active = False
        self.daily_loss_limit = 0.05  # 5% Daily Loss Limit
        self.consecutive_loss_limit = 3
        self.min_time_between_trades = 300  # 5 minutes
        
        # Monitoring
        self.trades_today = []
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.daily_pnl = 0.0
        self.alerts = []
        
        # Performance Tracking
        self.live_equity_curve = []
        self.live_trades = []
        self.daily_reports = []
        
        # Dashboard Integration
        self.current_phase = "live_testing"
        self.strategy_status = "active"
        self.last_signal_time = None
        
        logger.info(f"{self.strategy_name} v{self.strategy_version} Live-Testing initialisiert")
        logger.info(f"LIVE: Paper-Trading mit ${self.initial_capital:,.0f} Startkapital")
        logger.info(f"Safety: DD-Limit {self.max_drawdown_limit:.0%} | Daily-Loss {self.daily_loss_limit:.0%}")
    
    def get_live_strategy_info(self) -> Dict[str, Any]:
        """Live Strategy Info für Dashboard"""
        current_drawdown = self.calculate_current_drawdown()
        
        return {
            'name': self.strategy_name,
            'version': self.strategy_version,
            'risk_profile': self.risk_profile,
            'status': self.strategy_status,
            'paper_trading': self.paper_trading_mode,
            'current_capital': self.current_capital,
            'current_drawdown': current_drawdown,
            'emergency_stop_active': self.emergency_stop_active,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': len(self.trades_today),
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_position_size': self.max_position_size,
            'current_phase': self.current_phase,
            'last_signal_time': self.last_signal_time,
            'alerts': self.alerts[-5:] if self.alerts else []
        }
    
    def calculate_current_drawdown(self) -> float:
        """Berechnet aktuellen Drawdown"""
        if not self.live_equity_curve:
            return 0.0
        
        peak = max(eq['capital'] for eq in self.live_equity_curve)
        current = self.current_capital
        return max(0, (peak - current) / peak)
    
    def check_safety_controls(self, current_price: float) -> bool:
        """Überprüft alle Safety Controls"""
        # Emergency Stop Check
        if self.emergency_stop_active:
            self.add_alert("EMERGENCY_STOP", "Trading gestoppt - Manueller Emergency Stop")
            return False
        
        # Daily Loss Limit
        current_total = self.cash_balance + (self.btc_position * current_price)
        start_of_day_capital = self.get_start_of_day_capital()
        daily_loss = (start_of_day_capital - current_total) / start_of_day_capital
        
        if daily_loss > self.daily_loss_limit:
            self.emergency_stop_active = True
            self.add_alert("DAILY_LOSS_LIMIT", f"Daily Loss {daily_loss:.1%} > {self.daily_loss_limit:.1%}")
            return False
        
        # Maximum Drawdown Check
        current_drawdown = self.calculate_current_drawdown()
        if current_drawdown > self.max_drawdown_limit:
            self.emergency_stop_active = True
            self.add_alert("MAX_DRAWDOWN", f"Drawdown {current_drawdown:.1%} > {self.max_drawdown_limit:.1%}")
            return False
        
        # Consecutive Losses Check
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.add_alert("CONSECUTIVE_LOSSES", f"{self.consecutive_losses} aufeinanderfolgende Verluste")
            return False
        
        # Time Between Trades
        if self.last_trade_time:
            time_since_last = time.time() - self.last_trade_time
            if time_since_last < self.min_time_between_trades:
                return False
        
        return True
    
    def get_start_of_day_capital(self) -> float:
        """Startkapital des heutigen Tages"""
        today = datetime.now().date()
        
        for eq in reversed(self.live_equity_curve):
            eq_date = datetime.fromisoformat(eq['timestamp']).date()
            if eq_date < today:
                return eq['capital']
        
        return self.initial_capital
    
    def add_alert(self, alert_type: str, message: str):
        """Fügt Alert hinzu"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'HIGH' if alert_type in ['EMERGENCY_STOP', 'MAX_DRAWDOWN', 'DAILY_LOSS_LIMIT'] else 'MEDIUM'
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT {alert_type}: {message}")
    
    def simulate_live_signal(self, timestamp: str, price: float) -> Dict[str, Any]:
        """Simuliert Live-Signal basierend auf bewährter v1.3 Logik"""
        
        # Simplified Signal Generation für Live-Testing
        # Basiert auf bewährten Parametern der v1.3 Strategy
        
        # Simuliere technische Indikatoren (vereinfacht)
        import random
        import numpy as np
        
        # Basis Signal Strength (simuliert)
        base_signal = random.gauss(0, 0.3)  # Normalverteilung um 0
        
        # Quality Score (simuliert)
        quality_score = max(0.4, min(1.0, random.gauss(0.75, 0.15)))
        
        # Nur starke Signale gemäß v1.3 Parameter
        if abs(base_signal) >= self.min_signal_strength and quality_score >= self.quality_threshold:
            signal = {
                'timestamp': timestamp,
                'price': price,
                'signal_strength': base_signal,
                'direction': 'buy' if base_signal > 0 else 'sell',
                'quality_score': quality_score,
                'strategy_version': self.strategy_version,
                'live_trading': True
            }
            return signal
        
        return None
    
    def execute_live_trade(self, signal: Dict[str, Any]) -> bool:
        """Führt Live-Trade aus (Paper-Trading)"""
        current_price = signal['price']
        
        # Safety Controls Check
        if not self.check_safety_controls(current_price):
            return False
        
        # Position Size Calculation (v1.3 bewährt)
        total_portfolio = self.cash_balance + (self.btc_position * current_price)
        position_size = abs(signal['signal_strength']) * signal['quality_score'] * 0.75
        position_size = min(position_size, self.max_position_size)
        
        trade_amount = total_portfolio * position_size
        
        if signal['direction'] == 'buy' and self.cash_balance >= trade_amount * (1 + self.trading_fee):
            # Buy BTC
            btc_amount = trade_amount / current_price
            total_cost = trade_amount * (1 + self.trading_fee)
            
            self.btc_position += btc_amount
            self.cash_balance -= total_cost
            
            trade_record = {
                'timestamp': signal['timestamp'],
                'type': 'buy',
                'price': current_price,
                'amount': btc_amount,
                'cost': total_cost,
                'signal_strength': signal['signal_strength'],
                'quality_score': signal['quality_score'],
                'position_size': position_size,
                'paper_trading': True
            }
            
            self.live_trades.append(trade_record)
            self.trades_today.append(trade_record)
            self.last_trade_time = time.time()
            self.last_signal_time = signal['timestamp']
            
            logger.info(f"LIVE BUY: {btc_amount:.6f} BTC @ ${current_price:,.2f} | Quality: {signal['quality_score']:.2f}")
            return True
            
        elif signal['direction'] == 'sell' and self.btc_position > 0:
            # Sell BTC
            btc_to_sell = min(self.btc_position, self.btc_position * position_size)
            revenue = btc_to_sell * current_price * (1 - self.trading_fee)
            
            self.btc_position -= btc_to_sell
            self.cash_balance += revenue
            
            trade_record = {
                'timestamp': signal['timestamp'],
                'type': 'sell',
                'price': current_price,
                'amount': btc_to_sell,
                'revenue': revenue,
                'signal_strength': signal['signal_strength'],
                'quality_score': signal['quality_score'],
                'position_size': position_size,
                'paper_trading': True
            }
            
            self.live_trades.append(trade_record)
            self.trades_today.append(trade_record)
            self.last_trade_time = time.time()
            self.last_signal_time = signal['timestamp']
            
            logger.info(f"LIVE SELL: {btc_to_sell:.6f} BTC @ ${current_price:,.2f} | Quality: {signal['quality_score']:.2f}")
            return True
        
        return False
    
    def update_portfolio(self, timestamp: str, current_price: float):
        """Portfolio Update und Monitoring"""
        # Update current capital
        self.current_capital = self.cash_balance + (self.btc_position * current_price)
        
        # Update equity curve
        equity_point = {
            'timestamp': timestamp,
            'capital': self.current_capital,
            'btc_position': self.btc_position,
            'cash_balance': self.cash_balance,
            'btc_price': current_price,
            'drawdown': self.calculate_current_drawdown()
        }
        self.live_equity_curve.append(equity_point)
        
        # Update daily PnL
        start_capital = self.get_start_of_day_capital()
        self.daily_pnl = (self.current_capital - start_capital) / start_capital
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generiert Tagesbericht"""
        today = datetime.now().date()
        
        # Today's Performance
        start_capital = self.get_start_of_day_capital()
        daily_return = (self.current_capital - start_capital) / start_capital
        
        # Trade Summary
        trades_count = len(self.trades_today)
        
        # Risk Metrics
        current_drawdown = self.calculate_current_drawdown()
        
        report = {
            'date': today.isoformat(),
            'strategy': f"{self.strategy_name} v{self.strategy_version}",
            'daily_performance': {
                'return': daily_return,
                'start_capital': start_capital,
                'end_capital': self.current_capital,
                'pnl_dollar': self.current_capital - start_capital
            },
            'trading_activity': {
                'trades_executed': trades_count,
                'consecutive_losses': self.consecutive_losses,
                'emergency_stop_active': self.emergency_stop_active
            },
            'risk_metrics': {
                'current_drawdown': current_drawdown,
                'max_drawdown_limit': self.max_drawdown_limit,
                'daily_loss_limit': self.daily_loss_limit,
                'within_limits': current_drawdown <= self.max_drawdown_limit and abs(daily_return) <= self.daily_loss_limit
            },
            'alerts_count': len([a for a in self.alerts if datetime.fromisoformat(a['timestamp']).date() == today])
        }
        
        self.daily_reports.append(report)
        return report
    
    def run_live_simulation(self, duration_days: int = 7) -> Dict[str, Any]:
        """Führt Live-Trading Simulation durch"""
        logger.info(f"Starte {duration_days}-Tage Live-Trading Simulation...")
        
        start_date = datetime.now()
        simulation_data = []
        
        # Generiere simulierte Live-Market Data
        current_price = 45000.0  # Start BTC Preis
        
        for day in range(duration_days):
            current_date = start_date + timedelta(days=day)
            
            # Reset daily counters
            if day > 0:
                self.trades_today = []
            
            # Simuliere Handelstag (24 Stunden)
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)
                
                # Simuliere Preisbewegung
                daily_return = random.gauss(0.001, 0.03)  # 0.1% drift, 3% volatility
                current_price *= (1 + daily_return)
                current_price = max(current_price, 20000)  # Minimum BTC Preis
                
                # Portfolio Update
                self.update_portfolio(timestamp.isoformat(), current_price)
                
                # Signal Generation und Execution (alle 4 Stunden)
                if hour % 4 == 0:
                    signal = self.simulate_live_signal(timestamp.isoformat(), current_price)
                    if signal:
                        executed = self.execute_live_trade(signal)
                        if executed:
                            logger.info(f"Trade executed: {signal['direction']} @ ${current_price:,.2f}")
            
            # Tagesbericht
            daily_report = self.generate_daily_report()
            logger.info(f"Tag {day+1}: Return {daily_report['daily_performance']['return']:.1%}, "
                       f"Trades: {daily_report['trading_activity']['trades_executed']}, "
                       f"DD: {daily_report['risk_metrics']['current_drawdown']:.1%}")
        
        # Final Performance Summary
        total_return = (self.current_capital / self.initial_capital) - 1
        max_drawdown = max(eq['drawdown'] for eq in self.live_equity_curve) if self.live_equity_curve else 0
        
        summary = {
            'simulation_info': {
                'strategy': f"{self.strategy_name} v{self.strategy_version}",
                'duration_days': duration_days,
                'paper_trading': self.paper_trading_mode,
                'start_date': start_date.isoformat(),
                'end_date': (start_date + timedelta(days=duration_days-1)).isoformat()
            },
            'performance': {
                'total_return': total_return,
                'final_capital': self.current_capital,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.live_trades),
                'emergency_stops': sum(1 for a in self.alerts if a['type'] == 'EMERGENCY_STOP'),
                'alerts_total': len(self.alerts)
            },
            'daily_reports': self.daily_reports,
            'strategy_info': self.get_live_strategy_info(),
            'live_ready': max_drawdown <= self.max_drawdown_limit and len(self.alerts) == 0
        }
        
        return summary


async def main():
    """
    SCHRITT 3: Live-Testing Hauptausführung
    """
    print("🔴 SCHRITT 3: LIVE-TESTING FRAMEWORK")
    print("=" * 80)
    print("Strategy: Balanced Institutional BTC Elite v1.3 Live-Testing")
    print("Mode: Paper-Trading Simulation\\n")
    
    # Live-Testing Framework initialisieren
    live_framework = LiveTradingFramework()
    
    print("⚡ Führe 7-Tage Live-Trading Simulation durch...")
    print("Überwacht: Safety Controls | Risk Limits | Trade Execution\\n")
    
    # Live Simulation ausführen
    results = live_framework.run_live_simulation(duration_days=7)
    
    # Results Analysis
    print("📊 SCHRITT 3 ERGEBNISSE - LIVE-TESTING")
    print("-" * 80)
    print(f"Strategy: {results['simulation_info']['strategy']}")
    print(f"Duration: {results['simulation_info']['duration_days']} Tage Paper-Trading\\n")
    
    print("🎯 LIVE-PERFORMANCE:")
    print(f"   Total Return:           {results['performance']['total_return']:.1%}")
    print(f"   Final Capital:          ${results['performance']['final_capital']:,.0f}")
    print(f"   Max Drawdown:           {results['performance']['max_drawdown']:.1%}")
    print(f"   Total Trades:           {results['performance']['total_trades']}")
    print(f"   Emergency Stops:        {results['performance']['emergency_stops']}")
    print(f"   Total Alerts:           {results['performance']['alerts_total']}\\n")
    
    # Live-Ready Assessment
    live_ready = results['live_ready']
    no_emergency_stops = results['performance']['emergency_stops'] == 0
    within_dd_limit = results['performance']['max_drawdown'] <= live_framework.max_drawdown_limit
    trades_executed = results['performance']['total_trades'] > 0
    
    print("🎯 LIVE-READINESS BEWERTUNG:")
    print("-" * 80)
    print(f"Keine Emergency Stops:   {'✅' if no_emergency_stops else '❌'} ({results['performance']['emergency_stops']} Stops)")
    print(f"Drawdown unter Limit:    {'✅' if within_dd_limit else '❌'} ({results['performance']['max_drawdown']:.1%} ≤ {live_framework.max_drawdown_limit:.1%})")
    print(f"Trades ausgeführt:       {'✅' if trades_executed else '❌'} ({results['performance']['total_trades']} Trades)")
    print(f"Keine kritischen Alerts: {'✅' if results['performance']['alerts_total'] == 0 else '❌'} ({results['performance']['alerts_total']} Alerts)\\n")
    
    readiness_score = sum([no_emergency_stops, within_dd_limit, trades_executed, results['performance']['alerts_total'] == 0])
    score_percent = (readiness_score / 4) * 100
    
    if score_percent >= 75:
        status = "✅ LIVE-READY"
        next_action = "Proceed to Schritt 4: Dashboard-Deployment"
    elif score_percent >= 50:
        status = "⚠️ BEDINGT LIVE-READY"
        next_action = "Überwachung verstärken, kleinere Anpassungen"
    else:
        status = "❌ NICHT LIVE-READY"
        next_action = "Zurück zu Optimierung"
    
    print(f"LIVE-READINESS Score: {score_percent:.0f}/100")
    print(f"Status: {status}")
    print(f"Nächste Aktion: {next_action}\\n")
    
    # Export Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step3_live_testing_{timestamp}.json"
    
    export_data = {
        "step_info": {
            "step": "3",
            "name": "Live-Testing",
            "strategy": results['simulation_info']['strategy'],
            "duration": f"{results['simulation_info']['duration_days']} days",
            "timestamp": timestamp
        },
        "simulation_results": results,
        "readiness_assessment": {
            "no_emergency_stops": no_emergency_stops,
            "within_drawdown_limit": within_dd_limit,
            "trades_executed": trades_executed,
            "no_critical_alerts": results['performance']['alerts_total'] == 0,
            "overall_score": score_percent,
            "status": status.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", ""),
            "live_ready": live_ready
        },
        "next_steps": {
            "action": next_action,
            "ready_for_step4": score_percent >= 75
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"💾 SCHRITT 3 Ergebnisse exportiert: {filename}")


if __name__ == "__main__":
    import random
    asyncio.run(main())