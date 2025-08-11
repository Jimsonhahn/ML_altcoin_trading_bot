"""
Database Models
===============

SQLite database models for trading bot persistence.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Simple SQLite database for trading bot data"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "trading_bot.db"
        
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Trade History Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,  -- 'buy' or 'sell'
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        total_value REAL NOT NULL,
                        strategy TEXT,
                        mode TEXT,  -- 'paper' or 'live'
                        order_id TEXT,
                        status TEXT DEFAULT 'completed',
                        fees REAL DEFAULT 0,
                        notes TEXT,
                        metadata TEXT  -- JSON string for additional data
                    )
                ''')
                
                # Performance Metrics Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        date DATE NOT NULL,
                        total_balance REAL NOT NULL,
                        total_pnl REAL DEFAULT 0,
                        daily_pnl REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        strategy TEXT,
                        mode TEXT,
                        metadata TEXT
                    )
                ''')
                
                # System Logs Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT NOT NULL,  -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                        module TEXT,
                        message TEXT NOT NULL,
                        error_id TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Positions Table (for active positions)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL,
                        pnl REAL DEFAULT 0,
                        pnl_percentage REAL DEFAULT 0,
                        stop_loss REAL,
                        take_profit REAL,
                        strategy TEXT,
                        mode TEXT,
                        status TEXT DEFAULT 'open',  -- 'open', 'closed'
                        metadata TEXT
                    )
                ''')
                
                # Settings/Configuration Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        description TEXT
                    )
                ''')
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
            finally:
                conn.close()
    
    def add_trade(self, symbol: str, side: str, quantity: float, price: float, 
                  strategy: str = None, mode: str = 'paper', **kwargs) -> int:
        """Add a trade record"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                total_value = quantity * price
                
                cursor.execute('''
                    INSERT INTO trades (symbol, side, quantity, price, total_value, strategy, mode, 
                                      order_id, fees, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, side, quantity, price, total_value, strategy, mode,
                    kwargs.get('order_id'), kwargs.get('fees', 0), 
                    kwargs.get('notes'), json.dumps(kwargs.get('metadata', {}))
                ))
                
                trade_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Trade recorded: {side} {quantity} {symbol} @ {price}")
                return trade_id
                
            finally:
                conn.close()
    
    def get_trades(self, limit: int = 100, symbol: str = None, strategy: str = None) -> List[Dict]:
        """Get trade history"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM trades"
                params = []
                
                conditions = []
                if symbol:
                    conditions.append("symbol = ?")
                    params.append(symbol)
                if strategy:
                    conditions.append("strategy = ?")
                    params.append(strategy)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                
                trades = []
                for row in cursor.fetchall():
                    trade = dict(zip(columns, row))
                    if trade['metadata']:
                        trade['metadata'] = json.loads(trade['metadata'])
                    trades.append(trade)
                
                return trades
                
            finally:
                conn.close()
    
    def add_performance_metric(self, date: str, total_balance: float, 
                               total_pnl: float = 0, daily_pnl: float = 0, **kwargs):
        """Add daily performance metrics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Check if entry for date already exists
                cursor.execute("SELECT id FROM performance_metrics WHERE date = ?", (date,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE performance_metrics 
                        SET total_balance = ?, total_pnl = ?, daily_pnl = ?, 
                            win_rate = ?, total_trades = ?, winning_trades = ?, 
                            losing_trades = ?, max_drawdown = ?, sharpe_ratio = ?,
                            strategy = ?, mode = ?, metadata = ?, timestamp = CURRENT_TIMESTAMP
                        WHERE date = ?
                    ''', (
                        total_balance, total_pnl, daily_pnl,
                        kwargs.get('win_rate', 0), kwargs.get('total_trades', 0),
                        kwargs.get('winning_trades', 0), kwargs.get('losing_trades', 0),
                        kwargs.get('max_drawdown', 0), kwargs.get('sharpe_ratio', 0),
                        kwargs.get('strategy'), kwargs.get('mode', 'paper'),
                        json.dumps(kwargs.get('metadata', {})), date
                    ))
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO performance_metrics (date, total_balance, total_pnl, daily_pnl,
                                                       win_rate, total_trades, winning_trades, 
                                                       losing_trades, max_drawdown, sharpe_ratio,
                                                       strategy, mode, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date, total_balance, total_pnl, daily_pnl,
                        kwargs.get('win_rate', 0), kwargs.get('total_trades', 0),
                        kwargs.get('winning_trades', 0), kwargs.get('losing_trades', 0),
                        kwargs.get('max_drawdown', 0), kwargs.get('sharpe_ratio', 0),
                        kwargs.get('strategy'), kwargs.get('mode', 'paper'),
                        json.dumps(kwargs.get('metadata', {}))
                    ))
                
                conn.commit()
                
            finally:
                conn.close()
    
    def get_performance_metrics(self, days: int = 30) -> List[Dict]:
        """Get performance metrics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            try:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    ORDER BY date DESC 
                    LIMIT ?
                ''', (days,))
                
                columns = [desc[0] for desc in cursor.description]
                metrics = []
                for row in cursor.fetchall():
                    metric = dict(zip(columns, row))
                    if metric['metadata']:
                        metric['metadata'] = json.loads(metric['metadata'])
                    metrics.append(metric)
                
                return metrics
                
            finally:
                conn.close()
    
    def log_system_event(self, level: str, message: str, module: str = None, 
                        error_id: str = None, **kwargs):
        """Log system events"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_logs (level, module, message, error_id, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    level, module, message, error_id, json.dumps(kwargs)
                ))
                
                conn.commit()
                
            finally:
                conn.close()
    
    def get_system_logs(self, limit: int = 100, level: str = None) -> List[Dict]:
        """Get system logs"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM system_logs"
                params = []
                
                if level:
                    query += " WHERE level = ?"
                    params.append(level)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                
                logs = []
                for row in cursor.fetchall():
                    log = dict(zip(columns, row))
                    if log['metadata']:
                        log['metadata'] = json.loads(log['metadata'])
                    logs.append(log)
                
                return logs
                
            finally:
                conn.close()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Clean old logs
                cursor.execute('''
                    DELETE FROM system_logs 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                # Clean old metrics (keep more performance data)
                cursor.execute('''
                    DELETE FROM performance_metrics 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep * 2))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                
            finally:
                conn.close()

# Global database instance
db = TradingDatabase()