"""
Portfolio Controller
====================

Manages portfolio data and wealth tracking for the dashboard.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import os
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class PortfolioController:
    """Controller for portfolio and wealth data"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.portfolio_file = self.project_root / 'data' / 'portfolio' / 'portfolio_state.json'
        self.performance_file = self.project_root / 'data' / 'performance' / 'daily_performance.json'
        self._ensure_data_directories()
        
    def _ensure_data_directories(self):
        """Ensure data directories exist"""
        self.portfolio_file.parent.mkdir(exist_ok=True, parents=True)
        self.performance_file.parent.mkdir(exist_ok=True, parents=True)
    
    def get_wealth_data(self) -> Dict[str, Any]:
        """Get comprehensive wealth data for the Wealth Accumulator panel"""
        try:
            portfolio_data = self._load_portfolio_data()
            performance_data = self._load_performance_data()
            
            # Calculate current values
            total_value = portfolio_data.get('total_balance', 25000)  # Default starting balance
            daily_pnl = performance_data.get('daily_pnl', 0)
            daily_pnl_pct = performance_data.get('daily_pnl_percentage', 0)
            
            # Calculate profit streak
            profit_streak = self._calculate_profit_streak(performance_data)
            confidence_boost = self._calculate_confidence_multiplier(profit_streak)
            
            # Progress calculations
            daily_target = 500  # $500 daily target
            weekly_target = 3500  # $3500 weekly target
            monthly_target = 15000  # $15000 monthly target
            
            weekly_pnl = self._calculate_period_pnl(7)
            monthly_pnl = self._calculate_period_pnl(30)
            
            return {
                # Main portfolio value
                'total_value': round(total_value, 2),
                'total_value_formatted': f"${total_value:,.2f}",
                
                # Today's performance
                'daily_pnl': round(daily_pnl, 2),
                'daily_pnl_formatted': f"${daily_pnl:+,.2f}",
                'daily_pnl_percentage': round(daily_pnl_pct, 2),
                
                # Profit streak
                'profit_streak_hours': profit_streak,
                'confidence_boost': confidence_boost,
                'confidence_boost_formatted': f"{confidence_boost}x",
                
                # Progress tracking
                'daily_progress': {
                    'current': round(daily_pnl, 2),
                    'target': daily_target,
                    'percentage': min(100, max(0, (daily_pnl / daily_target) * 100)) if daily_target > 0 else 0,
                    'formatted': f"${daily_pnl:,.0f} / ${daily_target:,.0f}"
                },
                'weekly_progress': {
                    'current': round(weekly_pnl, 2),
                    'target': weekly_target,
                    'percentage': min(100, max(0, (weekly_pnl / weekly_target) * 100)) if weekly_target > 0 else 0,
                    'formatted': f"${weekly_pnl:,.0f} / ${weekly_target:,.0f}"
                },
                'monthly_progress': {
                    'current': round(monthly_pnl, 2),
                    'target': monthly_target,
                    'percentage': min(100, max(0, (monthly_pnl / monthly_target) * 100)) if monthly_target > 0 else 0,
                    'formatted': f"${monthly_pnl:,.0f} / ${monthly_target:,.0f}"
                },
                
                # Additional metrics
                'win_rate': performance_data.get('win_rate', 0),
                'total_trades': performance_data.get('total_trades', 0),
                'best_performer': performance_data.get('best_strategy', 'momentum_breakout'),
                'risk_level': self._calculate_risk_level(portfolio_data),
                
                # Time-based performance
                'hourly_return': performance_data.get('hourly_return', 0),
                'daily_return': performance_data.get('daily_return', 0),
                'weekly_return': performance_data.get('weekly_return', 0),
                'monthly_return': performance_data.get('monthly_return', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting wealth data: {str(e)}")
            return self._get_default_wealth_data()
    
    def _load_portfolio_data(self) -> Dict[str, Any]:
        """Load portfolio data from file or database"""
        try:
            # Try database first
            data = self._load_from_database()
            if data:
                return data
            
            # Try file
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            
            # Return mock data for demonstration
            return {
                'total_balance': 27543.82,
                'available_balance': 12543.82,
                'in_positions': 15000.00,
                'unrealized_pnl': 543.82,
                'realized_pnl': 2543.82,
                'start_balance': 25000.00,
                'positions': {
                    'BTC/USDT': {'value': 8500.00, 'pnl': 234.50},
                    'ETH/USDT': {'value': 6500.00, 'pnl': 309.32}
                }
            }
            
        except Exception as e:
            logger.warning(f"Could not load portfolio data: {str(e)}")
            return {'total_balance': 25000}
    
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load performance metrics"""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            
            # Return mock performance data
            return {
                'daily_pnl': 543.82,
                'daily_pnl_percentage': 2.18,
                'win_rate': 68.5,
                'total_trades': 27,
                'winning_trades': 18,
                'losing_trades': 9,
                'best_strategy': 'momentum_breakout',
                'hourly_return': 0.09,
                'daily_return': 2.18,
                'weekly_return': 8.34,
                'monthly_return': 15.23,
                'profit_history': [
                    {'date': '2024-08-12', 'pnl': 543.82},
                    {'date': '2024-08-11', 'pnl': 312.45},
                    {'date': '2024-08-10', 'pnl': 421.33},
                    {'date': '2024-08-09', 'pnl': -123.45},
                    {'date': '2024-08-08', 'pnl': 234.56}
                ]
            }
            
        except Exception as e:
            logger.warning(f"Could not load performance data: {str(e)}")
            return {'daily_pnl': 0, 'daily_pnl_percentage': 0}
    
    def _load_from_database(self) -> Optional[Dict[str, Any]]:
        """Load portfolio data from database"""
        try:
            db_url = os.environ.get('DATABASE_URL', 'postgresql://localhost/trading_bot')
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Get latest portfolio snapshot
                query = text("""
                    SELECT 
                        total_value,
                        available_balance,
                        in_positions,
                        unrealized_pnl,
                        realized_pnl,
                        created_at
                    FROM portfolio_snapshots
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = conn.execute(query).fetchone()
                if result:
                    return {
                        'total_balance': float(result.total_value),
                        'available_balance': float(result.available_balance),
                        'in_positions': float(result.in_positions),
                        'unrealized_pnl': float(result.unrealized_pnl),
                        'realized_pnl': float(result.realized_pnl)
                    }
                    
        except Exception as e:
            logger.debug(f"Could not load from database: {str(e)}")
        
        return None
    
    def _calculate_profit_streak(self, performance_data: Dict) -> int:
        """Calculate consecutive profitable hours"""
        # This would analyze real trading history
        # For now, return mock value
        history = performance_data.get('profit_history', [])
        streak = 0
        
        for day in reversed(history):
            if day.get('pnl', 0) > 0:
                streak += 24  # Assume full day of profit
            else:
                break
        
        return streak
    
    def _calculate_confidence_multiplier(self, profit_streak_hours: int) -> float:
        """Calculate confidence boost based on profit streak"""
        # Every 24 hours of profit adds 0.1x boost, max 2.0x
        base_multiplier = 1.0
        boost = (profit_streak_hours / 24) * 0.1
        return min(2.0, base_multiplier + boost)
    
    def _calculate_period_pnl(self, days: int) -> float:
        """Calculate PnL for a specific period"""
        try:
            # This would sum actual PnL from database
            # For now, use mock calculation
            daily_avg = 400  # Average daily profit
            return daily_avg * days * (0.8 + 0.4 * (days / 30))  # Scaling factor
        except:
            return 0
    
    def _calculate_risk_level(self, portfolio_data: Dict) -> str:
        """Calculate current risk level based on portfolio metrics"""
        in_positions = portfolio_data.get('in_positions', 0)
        total_balance = portfolio_data.get('total_balance', 1)
        
        if total_balance == 0:
            return 'Unknown'
        
        position_ratio = in_positions / total_balance
        
        if position_ratio < 0.3:
            return 'Low'
        elif position_ratio < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _get_default_wealth_data(self) -> Dict[str, Any]:
        """Return default wealth data structure"""
        return {
            'total_value': 25000,
            'total_value_formatted': '$25,000.00',
            'daily_pnl': 0,
            'daily_pnl_formatted': '+$0.00',
            'daily_pnl_percentage': 0,
            'profit_streak_hours': 0,
            'confidence_boost': 1.0,
            'confidence_boost_formatted': '1.0x',
            'daily_progress': {
                'current': 0,
                'target': 500,
                'percentage': 0,
                'formatted': '$0 / $500'
            },
            'weekly_progress': {
                'current': 0,
                'target': 3500,
                'percentage': 0,
                'formatted': '$0 / $3,500'
            },
            'monthly_progress': {
                'current': 0,
                'target': 15000,
                'percentage': 0,
                'formatted': '$0 / $15,000'
            },
            'win_rate': 0,
            'total_trades': 0,
            'best_performer': 'None',
            'risk_level': 'Low'
        }
    
    def get_portfolio_breakdown(self) -> Dict[str, Any]:
        """Get detailed portfolio breakdown by asset"""
        try:
            portfolio_data = self._load_portfolio_data()
            positions = portfolio_data.get('positions', {})
            
            breakdown = []
            total_value = 0
            
            for symbol, position in positions.items():
                value = position.get('value', 0)
                total_value += value
                breakdown.append({
                    'symbol': symbol,
                    'value': value,
                    'pnl': position.get('pnl', 0),
                    'percentage': 0  # Will calculate after
                })
            
            # Calculate percentages
            if total_value > 0:
                for item in breakdown:
                    item['percentage'] = (item['value'] / total_value) * 100
            
            # Add cash position
            available = portfolio_data.get('available_balance', 0)
            if available > 0:
                breakdown.append({
                    'symbol': 'USDT',
                    'value': available,
                    'pnl': 0,
                    'percentage': (available / (total_value + available)) * 100
                })
            
            return {
                'breakdown': breakdown,
                'total_value': total_value + available,
                'assets_count': len(breakdown)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio breakdown: {str(e)}")
            return {'breakdown': [], 'total_value': 0, 'assets_count': 0}
    
    def update_portfolio_snapshot(self, data: Dict[str, Any]) -> bool:
        """Update portfolio snapshot (called by bot)"""
        try:
            # Save to file
            self.portfolio_file.parent.mkdir(exist_ok=True, parents=True)
            
            with open(self.portfolio_file, 'w') as f:
                data['last_updated'] = datetime.now().isoformat()
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {str(e)}")
            return False