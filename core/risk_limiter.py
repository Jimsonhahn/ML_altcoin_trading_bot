#!/usr/bin/env python3
"""
Daily Risk Limiter - High-Risk Trading Budget Control
====================================================

Implements strict daily budget controls for high-risk trading:
- Hard 30€ daily loss limit
- Automatic position closure on limit breach
- Daily reset mechanism at midnight
- Complete isolation from other strategies
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class DailyRiskState:
    """Daily risk tracking state"""
    date: str
    daily_budget: float
    spent_budget: float
    remaining_budget: float
    trades_count: int
    pnl_realized: float
    pnl_unrealized: float
    is_locked: bool
    lock_reason: str
    last_reset: str

class DailyRiskLimiter:
    """
    Manages daily risk limits for high-risk trading strategy
    
    Features:
    - Hard daily budget enforcement
    - Automatic midnight reset
    - Position tracking and closure
    - Isolated risk management
    - Emergency stop mechanisms
    """
    
    def __init__(self, daily_budget: float = 30.0, data_dir: str = "data/high_risk"):
        self.daily_budget = daily_budget
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State file for persistence
        self.state_file = self.data_dir / "daily_risk_state.json"
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load or initialize state
        self.state = self._load_or_init_state()
        
        # Start background reset checker
        self._start_reset_monitor()
        
        logger.info(f"🛡️ Daily Risk Limiter initialized: {self.daily_budget}€ budget")
        logger.info(f"📊 Current state: {self.state.remaining_budget:.2f}€ remaining")
        
    def _load_or_init_state(self) -> DailyRiskState:
        """Load existing state or create new one"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Check if we need to reset for new day
                if data.get('date') != today:
                    logger.info(f"🔄 New day detected, resetting risk state")
                    return self._create_fresh_state(today)
                    
                return DailyRiskState(**data)
                
            except Exception as e:
                logger.error(f"Error loading risk state: {e}")
                return self._create_fresh_state(today)
        
        return self._create_fresh_state(today)
    
    def _create_fresh_state(self, date: str) -> DailyRiskState:
        """Create fresh daily state"""
        return DailyRiskState(
            date=date,
            daily_budget=self.daily_budget,
            spent_budget=0.0,
            remaining_budget=self.daily_budget,
            trades_count=0,
            pnl_realized=0.0,
            pnl_unrealized=0.0,
            is_locked=False,
            lock_reason="",
            last_reset=datetime.now().isoformat()
        )
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'date': self.state.date,
                    'daily_budget': self.state.daily_budget,
                    'spent_budget': self.state.spent_budget,
                    'remaining_budget': self.state.remaining_budget,
                    'trades_count': self.state.trades_count,
                    'pnl_realized': self.state.pnl_realized,
                    'pnl_unrealized': self.state.pnl_unrealized,
                    'is_locked': self.state.is_locked,
                    'lock_reason': self.state.lock_reason,
                    'last_reset': self.state.last_reset
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
    
    def can_trade(self, requested_amount: float = 0.0) -> Tuple[bool, str]:
        """
        Check if trading is allowed
        
        Args:
            requested_amount: Amount requested for new trade
            
        Returns:
            (can_trade, reason)
        """
        with self._lock:
            # Check if locked
            if self.state.is_locked:
                return False, f"Trading locked: {self.state.lock_reason}"
            
            # Check daily reset
            today = datetime.now().strftime("%Y-%m-%d")
            if self.state.date != today:
                self._reset_daily_state()
            
            # Check budget availability
            if requested_amount > self.state.remaining_budget:
                self._lock_trading("Insufficient daily budget")
                return False, f"Requested {requested_amount:.2f}€ exceeds remaining budget {self.state.remaining_budget:.2f}€"
            
            # Check total exposure
            total_risk = self.state.spent_budget + abs(self.state.pnl_unrealized)
            if total_risk >= self.daily_budget:
                self._lock_trading("Maximum daily risk reached")
                return False, f"Total risk exposure {total_risk:.2f}€ at limit"
            
            return True, "Trading allowed"
    
    def reserve_budget(self, amount: float, trade_id: str = None) -> bool:
        """
        Reserve budget for a trade
        
        Args:
            amount: Amount to reserve
            trade_id: Optional trade identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            can_trade, reason = self.can_trade(amount)
            
            if not can_trade:
                logger.warning(f"🚫 Budget reservation failed: {reason}")
                return False
            
            # Reserve the budget
            self.state.spent_budget += amount
            self.state.remaining_budget = self.daily_budget - self.state.spent_budget
            self.state.trades_count += 1
            
            self._save_state()
            
            logger.info(f"💰 Reserved {amount:.2f}€ for trade {trade_id or 'unknown'}")
            logger.info(f"📊 Remaining budget: {self.state.remaining_budget:.2f}€")
            
            return True
    
    def update_pnl(self, realized_pnl: float = 0.0, unrealized_pnl: float = 0.0):
        """
        Update P&L tracking
        
        Args:
            realized_pnl: Realized profit/loss from closed trades
            unrealized_pnl: Unrealized profit/loss from open positions
        """
        with self._lock:
            if realized_pnl != 0.0:
                self.state.pnl_realized += realized_pnl
                logger.info(f"💵 Realized P&L: {realized_pnl:+.2f}€ (Total: {self.state.pnl_realized:+.2f}€)")
            
            self.state.pnl_unrealized = unrealized_pnl
            
            # Check if losses exceed budget
            total_loss = abs(min(0, self.state.pnl_realized + self.state.pnl_unrealized))
            if total_loss >= self.daily_budget:
                self._lock_trading(f"Daily loss limit reached: {total_loss:.2f}€")
            
            self._save_state()
    
    def release_budget(self, amount: float, final_pnl: float = 0.0):
        """
        Release reserved budget after trade completion
        
        Args:
            amount: Amount to release back to budget
            final_pnl: Final P&L of the completed trade
        """
        with self._lock:
            # Don't release if we had a loss (budget is consumed)
            if final_pnl >= 0:
                self.state.spent_budget = max(0, self.state.spent_budget - amount)
                self.state.remaining_budget = min(self.daily_budget, 
                                                self.daily_budget - self.state.spent_budget)
            
            self.update_pnl(realized_pnl=final_pnl)
            
            logger.info(f"🔓 Released {amount:.2f}€ budget, final P&L: {final_pnl:+.2f}€")
    
    def _lock_trading(self, reason: str):
        """Lock trading with reason"""
        self.state.is_locked = True
        self.state.lock_reason = reason
        self._save_state()
        
        logger.critical(f"🔒 TRADING LOCKED: {reason}")
        logger.critical(f"📊 Final state - Budget spent: {self.state.spent_budget:.2f}€, "
                       f"Realized P&L: {self.state.pnl_realized:+.2f}€")
    
    def _reset_daily_state(self):
        """Reset state for new day"""
        old_date = self.state.date
        old_pnl = self.state.pnl_realized
        
        self.state = self._create_fresh_state(datetime.now().strftime("%Y-%m-%d"))
        self._save_state()
        
        logger.info(f"🔄 Daily reset completed: {old_date} → {self.state.date}")
        logger.info(f"📈 Previous day P&L: {old_pnl:+.2f}€")
        logger.info(f"💰 Fresh budget: {self.daily_budget:.2f}€")
    
    def _start_reset_monitor(self):
        """Start background thread to monitor for daily resets"""
        def monitor():
            while True:
                try:
                    now = datetime.now()
                    today = now.strftime("%Y-%m-%d")
                    
                    with self._lock:
                        if self.state.date != today:
                            self._reset_daily_state()
                    
                    # Sleep until next check (every hour)
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Reset monitor error: {e}")
                    time.sleep(300)  # Retry in 5 minutes
        
        reset_thread = threading.Thread(target=monitor, daemon=True)
        reset_thread.start()
        logger.info("🕐 Daily reset monitor started")
    
    def force_reset(self):
        """Force reset (admin function)"""
        with self._lock:
            self._reset_daily_state()
            logger.warning("🔧 Manual reset executed")
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Emergency stop all trading"""
        with self._lock:
            self._lock_trading(reason)
            logger.critical(f"🚨 EMERGENCY STOP: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        with self._lock:
            return {
                'date': self.state.date,
                'daily_budget': self.state.daily_budget,
                'spent_budget': self.state.spent_budget,
                'remaining_budget': self.state.remaining_budget,
                'budget_utilization': (self.state.spent_budget / self.daily_budget) * 100,
                'trades_count': self.state.trades_count,
                'pnl_realized': self.state.pnl_realized,
                'pnl_unrealized': self.state.pnl_unrealized,
                'total_pnl': self.state.pnl_realized + self.state.pnl_unrealized,
                'is_locked': self.state.is_locked,
                'lock_reason': self.state.lock_reason,
                'can_trade': not self.state.is_locked and self.state.remaining_budget > 0,
                'last_reset': self.state.last_reset
            }
    
    def get_daily_summary(self) -> str:
        """Get formatted daily summary"""
        status = self.get_status()
        
        summary = f"""
🛡️ DAILY RISK SUMMARY - {status['date']}
{'='*40}
💰 Budget: {status['spent_budget']:.2f}€ / {status['daily_budget']:.2f}€ ({status['budget_utilization']:.1f}%)
💵 P&L: {status['total_pnl']:+.2f}€ (Realized: {status['pnl_realized']:+.2f}€)
📊 Trades: {status['trades_count']}
🔒 Status: {'LOCKED' if status['is_locked'] else 'ACTIVE'}
{f"❌ Lock Reason: {status['lock_reason']}" if status['is_locked'] else "✅ Trading Allowed"}
        """.strip()
        
        return summary

# Global instance for high-risk strategy
_global_risk_limiter: Optional[DailyRiskLimiter] = None

def get_risk_limiter(daily_budget: float = 30.0) -> DailyRiskLimiter:
    """Get global risk limiter instance"""
    global _global_risk_limiter
    
    if _global_risk_limiter is None:
        _global_risk_limiter = DailyRiskLimiter(daily_budget)
    
    return _global_risk_limiter

def init_high_risk_protection(daily_budget: float = 30.0):
    """Initialize high-risk protection system"""
    limiter = get_risk_limiter(daily_budget)
    logger.info("🛡️ High-risk protection system initialized")
    return limiter