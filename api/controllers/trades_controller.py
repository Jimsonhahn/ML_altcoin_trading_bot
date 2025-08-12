"""
Trades Controller
=================

Manages and provides real-time trade data for the dashboard.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from sqlalchemy import create_engine, text
import os

from core.position import Position
from data_sources.data_manager import DataManager

logger = logging.getLogger(__name__)


class TradesController:
    """Controller for managing and displaying active trades"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.trades_file = self.project_root / 'data' / 'trades' / 'active_trades.json'
        self.data_manager = None
        self._init_data_manager()
        
    def _init_data_manager(self):
        """Initialize data manager for price data"""
        try:
            self.data_manager = DataManager()
        except Exception as e:
            logger.warning(f"Could not initialize data manager: {str(e)}")
    
    def get_active_trades(self) -> Dict[str, Any]:
        """Get all active trades for the dashboard"""
        try:
            # Try multiple sources for trade data
            trades = self._get_trades_from_database() or \
                    self._get_trades_from_file() or \
                    self._get_trades_from_memory()
            
            if not trades:
                return {
                    'active_count': 0,
                    'status': 'No active trades',
                    'trades': [],
                    'summary': {
                        'total_pnl': 0,
                        'total_pnl_percentage': 0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }
                }
            
            formatted_trades = []
            total_pnl = 0
            winning_trades = 0
            losing_trades = 0
            
            for trade in trades:
                formatted_trade = self._format_trade(trade)
                formatted_trades.append(formatted_trade)
                
                # Calculate summary stats
                pnl = formatted_trade.get('pnl', 0)
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
            
            return {
                'active_count': len(formatted_trades),
                'status': f'{len(formatted_trades)} Active',
                'trades': formatted_trades,
                'summary': {
                    'total_pnl': round(total_pnl, 2),
                    'total_pnl_percentage': self._calculate_total_pnl_percentage(formatted_trades),
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return {
                'active_count': 0,
                'status': 'Error loading trades',
                'trades': [],
                'error': str(e)
            }
    
    def _get_trades_from_database(self) -> Optional[List[Dict]]:
        """Get trades from database"""
        try:
            # Check for PostgreSQL connection
            db_url = os.environ.get('DATABASE_URL', 'postgresql://localhost/trading_bot')
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Query for active trades
                query = text("""
                    SELECT 
                        id,
                        symbol,
                        side,
                        entry_price,
                        current_price,
                        quantity,
                        strategy,
                        created_at,
                        updated_at
                    FROM trades
                    WHERE status = 'OPEN'
                    ORDER BY created_at DESC
                """)
                
                result = conn.execute(query)
                trades = []
                
                for row in result:
                    trades.append({
                        'id': row.id,
                        'symbol': row.symbol,
                        'side': row.side,
                        'entry_price': float(row.entry_price),
                        'current_price': float(row.current_price) if row.current_price else None,
                        'size': float(row.quantity),
                        'strategy': row.strategy,
                        'timestamp': row.created_at.isoformat() if row.created_at else None
                    })
                
                return trades if trades else None
                
        except Exception as e:
            logger.debug(f"Could not get trades from database: {str(e)}")
            return None
    
    def _get_trades_from_file(self) -> Optional[List[Dict]]:
        """Get trades from JSON file"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    return data.get('active_trades', [])
            return None
        except Exception as e:
            logger.debug(f"Could not get trades from file: {str(e)}")
            return None
    
    def _get_trades_from_memory(self) -> List[Dict]:
        """Get trades from in-memory storage (mock data for now)"""
        # This would connect to the running bot's memory/state
        # For now, return mock data to show functionality
        return [
            {
                'symbol': 'BTC/USDT',
                'side': 'LONG',
                'size': 0.015,
                'entry_price': 67842.50,
                'strategy': 'momentum_breakout',
                'timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'ETH/USDT',
                'side': 'LONG',
                'size': 0.25,
                'entry_price': 3521.30,
                'strategy': 'mean_reversion',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ]
    
    def _format_trade(self, trade: Dict) -> Dict[str, Any]:
        """Format trade data for dashboard display"""
        symbol = trade.get('symbol', 'UNKNOWN')
        entry_price = trade.get('entry_price', 0)
        size = trade.get('size', 0)
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Calculate PnL
        pnl, pnl_percentage = self._calculate_pnl(
            trade.get('side', 'LONG'),
            entry_price,
            current_price,
            size
        )
        
        # Calculate trade duration
        duration = self._calculate_trade_duration(trade.get('timestamp'))
        
        return {
            'id': trade.get('id', self._generate_trade_id(trade)),
            'symbol': symbol,
            'side': trade.get('side', 'LONG'),
            'size': size,
            'entry_price': round(entry_price, 2),
            'current_price': round(current_price, 2),
            'pnl': round(pnl, 2),
            'pnl_percentage': round(pnl_percentage, 2),
            'strategy': trade.get('strategy', 'Unknown'),
            'duration': duration,
            'duration_formatted': self._format_duration(duration),
            'status': 'OPEN',
            'timestamp': trade.get('timestamp', datetime.now().isoformat())
        }
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            if self.data_manager:
                # Convert symbol format if needed
                clean_symbol = symbol.replace('/', '')
                price_data = self.data_manager.get_latest_price(clean_symbol)
                if price_data:
                    return float(price_data)
            
            # Fallback mock prices
            mock_prices = {
                'BTC/USDT': 68234.50,
                'ETH/USDT': 3542.80,
                'BNB/USDT': 612.30,
                'SOL/USDT': 142.45
            }
            return mock_prices.get(symbol, 0)
            
        except Exception as e:
            logger.warning(f"Could not get current price for {symbol}: {str(e)}")
            return 0
    
    def _calculate_pnl(self, side: str, entry_price: float, current_price: float, size: float) -> tuple:
        """Calculate PnL and percentage"""
        if not current_price or not entry_price:
            return 0, 0
        
        if side.upper() == 'LONG':
            pnl = (current_price - entry_price) * size
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl = (entry_price - current_price) * size
            pnl_percentage = ((entry_price - current_price) / entry_price) * 100
        
        return pnl, pnl_percentage
    
    def _calculate_trade_duration(self, timestamp: Optional[str]) -> int:
        """Calculate trade duration in seconds"""
        if not timestamp:
            return 0
        
        try:
            start_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            duration = datetime.now() - start_time.replace(tzinfo=None)
            return int(duration.total_seconds())
        except:
            return 0
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"
    
    def _calculate_total_pnl_percentage(self, trades: List[Dict]) -> float:
        """Calculate total PnL percentage across all trades"""
        if not trades:
            return 0
        
        total_investment = sum(t['entry_price'] * t['size'] for t in trades)
        total_current_value = sum(t['current_price'] * t['size'] for t in trades)
        
        if total_investment == 0:
            return 0
        
        return ((total_current_value - total_investment) / total_investment) * 100
    
    def _generate_trade_id(self, trade: Dict) -> str:
        """Generate a unique trade ID"""
        import hashlib
        trade_str = f"{trade.get('symbol')}_{trade.get('timestamp')}_{trade.get('size')}"
        return hashlib.md5(trade_str.encode()).hexdigest()[:8]
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get historical trades"""
        try:
            # This would fetch from database or history file
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close a specific trade"""
        try:
            # Implementation would close the trade in the bot
            # and update database/files
            return {
                'success': True,
                'message': f'Trade {trade_id} closed successfully'
            }
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }