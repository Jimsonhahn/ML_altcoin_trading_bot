#!/usr/bin/env python3
"""
High-Risk Trading Isolated Logger
================================

Specialized logging system for high-risk trading strategy:
- Separate log files and directories
- Detailed trade tracking
- Performance analytics
- Risk metric monitoring
- Emergency event logging
"""

import os
import json
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
import csv

@dataclass
class TradeLog:
    """Individual trade log entry"""
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # ENTRY, EXIT, PARTIAL_EXIT
    side: str    # BUY, SELL
    quantity: float
    price: float
    value: float
    commission: float
    pnl: float
    pnl_pct: float
    budget_used: float
    remaining_budget: float
    confidence: float
    entry_signal: str
    exit_reason: str
    hold_duration: float
    metadata: Dict[str, Any]

@dataclass
class DailyPerformance:
    """Daily performance summary"""
    date: str
    trades_count: int
    budget_allocated: float
    budget_used: float
    gross_pnl: float
    net_pnl: float
    commission_total: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    signals_generated: int
    signals_acted: int
    top_performer: str
    worst_performer: str

class HighRiskLogger:
    """
    Comprehensive logging system for high-risk trading
    
    Features:
    - Isolated log directories
    - Multiple log levels and files
    - CSV trade exports
    - Performance analytics
    - Real-time monitoring
    - Emergency notifications
    """
    
    def __init__(self, base_dir: str = "logs/high_risk"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.trades_dir = self.base_dir / "trades"
        self.performance_dir = self.base_dir / "performance"
        self.signals_dir = self.base_dir / "signals"
        self.errors_dir = self.base_dir / "errors"
        
        for dir_path in [self.trades_dir, self.performance_dir, self.signals_dir, self.errors_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Trade tracking
        self.trade_logs: List[TradeLog] = []
        self.daily_performance: Dict[str, DailyPerformance] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize loggers
        self._setup_loggers()
        
        # CSV writers
        self._setup_csv_writers()
        
        self.main_logger.info("🔥 High-Risk Logger initialized")
        self.main_logger.info(f"📁 Log directory: {self.base_dir.absolute()}")
    
    def _setup_loggers(self):
        """Setup specialized loggers"""
        
        # Main strategy logger
        self.main_logger = self._create_logger(
            'high_risk_main',
            self.base_dir / 'high_risk_main.log',
            level=logging.INFO
        )
        
        # Trade execution logger
        self.trade_logger = self._create_logger(
            'high_risk_trades',
            self.trades_dir / 'trades.log',
            level=logging.INFO
        )
        
        # Signal generation logger
        self.signal_logger = self._create_logger(
            'high_risk_signals',
            self.signals_dir / 'signals.log',
            level=logging.DEBUG
        )
        
        # Performance tracking logger
        self.perf_logger = self._create_logger(
            'high_risk_performance',
            self.performance_dir / 'performance.log',
            level=logging.INFO
        )
        
        # Error and risk logger
        self.error_logger = self._create_logger(
            'high_risk_errors',
            self.errors_dir / 'errors.log',
            level=logging.WARNING
        )
        
        # Emergency logger (critical events)
        self.emergency_logger = self._create_logger(
            'high_risk_emergency',
            self.base_dir / 'EMERGENCY.log',
            level=logging.CRITICAL
        )
    
    def _create_logger(self, name: str, file_path: Path, level: int) -> logging.Logger:
        """Create individual logger with rotation"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        # Console handler for critical logs
        if level >= logging.WARNING:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(asctime)s - HIGH-RISK - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Detailed file formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_csv_writers(self):
        """Setup CSV file writers for structured data"""
        
        # Daily trades CSV
        self.trades_csv_path = self.trades_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        self.trades_csv_headers = [
            'timestamp', 'trade_id', 'symbol', 'action', 'side', 'quantity', 
            'price', 'value', 'commission', 'pnl', 'pnl_pct', 'budget_used',
            'remaining_budget', 'confidence', 'entry_signal', 'exit_reason',
            'hold_duration'
        ]
        
        # Initialize CSV if not exists
        if not self.trades_csv_path.exists():
            with open(self.trades_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.trades_csv_headers)
        
        # Daily performance CSV
        self.perf_csv_path = self.performance_dir / "daily_performance.csv"
        self.perf_csv_headers = [
            'date', 'trades_count', 'budget_used', 'gross_pnl', 'net_pnl',
            'commission_total', 'win_rate', 'profit_factor', 'sharpe_ratio',
            'max_drawdown', 'signals_generated', 'signals_acted'
        ]
        
        if not self.perf_csv_path.exists():
            with open(self.perf_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.perf_csv_headers)
    
    def log_trade_entry(self, trade_data: Dict[str, Any]):
        """Log trade entry"""
        with self._lock:
            trade_log = TradeLog(
                trade_id=trade_data.get('trade_id', ''),
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', ''),
                action='ENTRY',
                side=trade_data.get('side', ''),
                quantity=trade_data.get('quantity', 0.0),
                price=trade_data.get('price', 0.0),
                value=trade_data.get('value', 0.0),
                commission=trade_data.get('commission', 0.0),
                pnl=0.0,
                pnl_pct=0.0,
                budget_used=trade_data.get('budget_used', 0.0),
                remaining_budget=trade_data.get('remaining_budget', 0.0),
                confidence=trade_data.get('confidence', 0.0),
                entry_signal=trade_data.get('entry_signal', ''),
                exit_reason='',
                hold_duration=0.0,
                metadata=trade_data.get('metadata', {})
            )
            
            self.trade_logs.append(trade_log)
            
            # Log to file
            self.trade_logger.info(
                f"🔥 ENTRY: {trade_log.symbol} {trade_log.side} "
                f"{trade_log.quantity:.6f} @ ${trade_log.price:.6f} "
                f"(Budget: {trade_log.budget_used:.2f}€, "
                f"Signal: {trade_log.entry_signal}, "
                f"Confidence: {trade_log.confidence:.2f})"
            )
            
            # Write to CSV
            self._write_trade_csv(trade_log)
            
            # Main log
            self.main_logger.info(
                f"Trade Entry: {trade_log.trade_id} - {trade_log.symbol} "
                f"{trade_log.side} {trade_log.value:.2f}€"
            )
    
    def log_trade_exit(self, trade_data: Dict[str, Any]):
        """Log trade exit"""
        with self._lock:
            trade_log = TradeLog(
                trade_id=trade_data.get('trade_id', ''),
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', ''),
                action='EXIT',
                side='SELL' if trade_data.get('original_side') == 'BUY' else 'BUY',
                quantity=trade_data.get('quantity', 0.0),
                price=trade_data.get('price', 0.0),
                value=trade_data.get('value', 0.0),
                commission=trade_data.get('commission', 0.0),
                pnl=trade_data.get('pnl', 0.0),
                pnl_pct=trade_data.get('pnl_pct', 0.0),
                budget_used=0.0,  # Budget released on exit
                remaining_budget=trade_data.get('remaining_budget', 0.0),
                confidence=0.0,
                entry_signal=trade_data.get('entry_signal', ''),
                exit_reason=trade_data.get('exit_reason', ''),
                hold_duration=trade_data.get('hold_duration', 0.0),
                metadata=trade_data.get('metadata', {})
            )
            
            self.trade_logs.append(trade_log)
            
            # Log to file with P&L highlighting
            pnl_emoji = "💰" if trade_log.pnl > 0 else "💸" if trade_log.pnl < 0 else "⚖️"
            
            self.trade_logger.info(
                f"{pnl_emoji} EXIT: {trade_log.symbol} {trade_log.side} "
                f"{trade_log.quantity:.6f} @ ${trade_log.price:.6f} "
                f"P&L: {trade_log.pnl:+.2f}€ ({trade_log.pnl_pct:+.1%}) "
                f"Hold: {trade_log.hold_duration:.1f}h "
                f"Reason: {trade_log.exit_reason}"
            )
            
            # Write to CSV
            self._write_trade_csv(trade_log)
            
            # Main log
            self.main_logger.info(
                f"Trade Exit: {trade_log.trade_id} - P&L: {trade_log.pnl:+.2f}€ "
                f"({trade_log.pnl_pct:+.1%})"
            )
            
            # Update performance tracking
            self._update_daily_performance(trade_log)
    
    def log_signal_generated(self, signal_data: Dict[str, Any]):
        """Log signal generation"""
        self.signal_logger.info(
            f"🎯 SIGNAL: {signal_data.get('symbol', 'UNKNOWN')} "
            f"{signal_data.get('type', 'UNKNOWN')} "
            f"confidence={signal_data.get('confidence', 0.0):.2f} "
            f"source={signal_data.get('source', 'UNKNOWN')} "
            f"metadata={json.dumps(signal_data.get('metadata', {}))}"
        )
    
    def log_signal_acted(self, signal_data: Dict[str, Any], action: str):
        """Log when signal is acted upon"""
        self.signal_logger.info(
            f"⚡ ACTED: {signal_data.get('symbol', 'UNKNOWN')} "
            f"signal resulted in {action} "
            f"confidence={signal_data.get('confidence', 0.0):.2f}"
        )
    
    def log_signal_ignored(self, signal_data: Dict[str, Any], reason: str):
        """Log when signal is ignored"""
        self.signal_logger.debug(
            f"🚫 IGNORED: {signal_data.get('symbol', 'UNKNOWN')} "
            f"signal ignored due to: {reason}"
        )
    
    def log_risk_event(self, event: str, details: Dict[str, Any]):
        """Log risk management events"""
        self.error_logger.warning(
            f"⚠️ RISK EVENT: {event} - {json.dumps(details)}"
        )
        
        self.main_logger.warning(f"Risk Event: {event}")
    
    def log_emergency_stop(self, reason: str, final_state: Dict[str, Any]):
        """Log emergency stop events"""
        self.emergency_logger.critical(
            f"🚨 EMERGENCY STOP: {reason} - Final State: {json.dumps(final_state)}"
        )
        
        self.main_logger.critical(f"EMERGENCY STOP: {reason}")
        
        # Force flush all logs
        for logger in [self.main_logger, self.trade_logger, self.signal_logger, 
                      self.perf_logger, self.error_logger, self.emergency_logger]:
            for handler in logger.handlers:
                handler.flush()
    
    def log_daily_reset(self, previous_performance: Dict[str, Any]):
        """Log daily reset event"""
        self.main_logger.info(
            f"🔄 DAILY RESET - Previous day performance: "
            f"P&L: {previous_performance.get('total_pnl', 0.0):+.2f}€, "
            f"Trades: {previous_performance.get('trades_count', 0)}, "
            f"Win Rate: {previous_performance.get('win_rate', 0.0):.1%}"
        )
        
        self.perf_logger.info(f"Daily Reset: {json.dumps(previous_performance)}")
    
    def _write_trade_csv(self, trade_log: TradeLog):
        """Write trade to CSV file"""
        try:
            with open(self.trades_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_log.timestamp.isoformat(),
                    trade_log.trade_id,
                    trade_log.symbol,
                    trade_log.action,
                    trade_log.side,
                    trade_log.quantity,
                    trade_log.price,
                    trade_log.value,
                    trade_log.commission,
                    trade_log.pnl,
                    trade_log.pnl_pct,
                    trade_log.budget_used,
                    trade_log.remaining_budget,
                    trade_log.confidence,
                    trade_log.entry_signal,
                    trade_log.exit_reason,
                    trade_log.hold_duration
                ])
        except Exception as e:
            self.error_logger.error(f"Failed to write trade CSV: {e}")
    
    def _update_daily_performance(self, trade_log: TradeLog):
        """Update daily performance metrics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_performance:
            self.daily_performance[today] = DailyPerformance(
                date=today,
                trades_count=0,
                budget_allocated=30.0,  # Default budget
                budget_used=0.0,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_total=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                signals_generated=0,
                signals_acted=0,
                top_performer='',
                worst_performer=''
            )
        
        perf = self.daily_performance[today]
        
        if trade_log.action == 'EXIT':
            perf.trades_count += 1
            perf.gross_pnl += trade_log.pnl
            perf.net_pnl += (trade_log.pnl - trade_log.commission)
            perf.commission_total += trade_log.commission
            
            # Update win/loss stats
            if trade_log.pnl > 0:
                perf.max_win = max(perf.max_win, trade_log.pnl)
            else:
                perf.max_loss = min(perf.max_loss, trade_log.pnl)
            
            # Calculate win rate (simplified)
            wins = sum(1 for log in self.trade_logs 
                      if log.action == 'EXIT' and log.pnl > 0 
                      and log.timestamp.strftime('%Y-%m-%d') == today)
            total_trades = sum(1 for log in self.trade_logs 
                             if log.action == 'EXIT' 
                             and log.timestamp.strftime('%Y-%m-%d') == today)
            
            perf.win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    def get_daily_summary(self) -> str:
        """Get formatted daily summary"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_performance:
            return f"🔥 HIGH-RISK SUMMARY - {today}\nNo trades executed today."
        
        perf = self.daily_performance[today]
        
        summary = f"""
🔥 HIGH-RISK DAILY SUMMARY - {today}
{'='*45}
💰 P&L: {perf.net_pnl:+.2f}€ (Gross: {perf.gross_pnl:+.2f}€)
📊 Trades: {perf.trades_count} (Win Rate: {perf.win_rate:.1%})
💵 Commissions: {perf.commission_total:.2f}€
🎯 Best Trade: {perf.max_win:+.2f}€
📉 Worst Trade: {perf.max_loss:+.2f}€
💡 Signals Generated: {perf.signals_generated}
⚡ Signals Acted: {perf.signals_acted}
        """.strip()
        
        return summary
    
    def export_daily_trades(self, date: str = None) -> str:
        """Export daily trades to CSV and return file path"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        export_path = self.trades_dir / f"export_trades_{date}.csv"
        
        daily_trades = [
            trade for trade in self.trade_logs 
            if trade.timestamp.strftime('%Y-%m-%d') == date
        ]
        
        with open(export_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.trades_csv_headers + ['metadata'])
            
            for trade in daily_trades:
                row = [
                    trade.timestamp.isoformat(),
                    trade.trade_id,
                    trade.symbol,
                    trade.action,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.value,
                    trade.commission,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.budget_used,
                    trade.remaining_budget,
                    trade.confidence,
                    trade.entry_signal,
                    trade.exit_reason,
                    trade.hold_duration,
                    json.dumps(trade.metadata)
                ]
                writer.writerow(row)
        
        return str(export_path)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_dir in [self.trades_dir, self.performance_dir, self.signals_dir, self.errors_dir]:
            for file_path in log_dir.glob("*.log*"):
                try:
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        self.main_logger.info(f"Cleaned up old log: {file_path.name}")
                except Exception as e:
                    self.error_logger.error(f"Error cleaning up {file_path}: {e}")

# Global logger instance
_global_high_risk_logger: Optional[HighRiskLogger] = None

def get_high_risk_logger() -> HighRiskLogger:
    """Get global high-risk logger instance"""
    global _global_high_risk_logger
    
    if _global_high_risk_logger is None:
        _global_high_risk_logger = HighRiskLogger()
    
    return _global_high_risk_logger

def init_high_risk_logging() -> HighRiskLogger:
    """Initialize high-risk logging system"""
    logger = get_high_risk_logger()
    logger.main_logger.info("🔥 High-Risk Logging System Active")
    return logger