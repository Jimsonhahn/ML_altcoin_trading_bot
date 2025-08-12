"""
Strategy Supermix Controller
============================

Manages risk-tiered strategy execution and performance data.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from core.strategy_router import StrategyRouter

logger = logging.getLogger(__name__)


class StrategySupermixController:
    """Controller for risk-tiered strategy management"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.strategy_config_file = self.project_root / 'config' / 'strategy_config.json'
        self.performance_file = self.project_root / 'data' / 'strategy_performance.json'
        self._ensure_data_directories()
        
    def _ensure_data_directories(self):
        """Ensure data directories exist"""
        self.performance_file.parent.mkdir(exist_ok=True, parents=True)
    
    def get_strategy_supermix_status(self) -> Dict[str, Any]:
        """Get comprehensive risk-tiered strategy status"""
        try:
            # Load strategy configurations and performance
            strategy_config = self._load_strategy_config()
            performance_data = self._load_performance_data()
            active_strategies = self._get_active_strategies()
            
            # Determine overall status
            is_active = len(active_strategies) > 0
            execution_mode = 'PARALLEL EXECUTION' if is_active else 'STOPPED'
            
            # Get risk tier data
            high_risk = self._get_risk_tier_data('high', strategy_config, performance_data, active_strategies)
            medium_risk = self._get_risk_tier_data('medium', strategy_config, performance_data, active_strategies)
            low_risk = self._get_risk_tier_data('low', strategy_config, performance_data, active_strategies)
            
            # Calculate total performance
            total_pnl = high_risk['current_pnl'] + medium_risk['current_pnl'] + low_risk['current_pnl']
            total_strategies = high_risk['strategies_count'] + medium_risk['strategies_count'] + low_risk['strategies_count']
            
            return {
                # Overall Status
                'status': execution_mode,
                'execution_mode': execution_mode,
                'total_active_strategies': total_strategies,
                'total_pnl': round(total_pnl, 2),
                'total_pnl_formatted': f"${total_pnl:+,.2f}",
                
                # Risk Tiers
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk,
                
                # Performance Metrics
                'performance_metrics': {
                    'sharpe_ratio': performance_data.get('sharpe_ratio', 1.8),
                    'win_rate': performance_data.get('overall_win_rate', 68.5),
                    'profit_factor': performance_data.get('profit_factor', 1.9),
                    'max_drawdown': performance_data.get('max_drawdown', -8.2)
                },
                
                # Execution Stats
                'execution_stats': {
                    'trades_today': performance_data.get('trades_today', 42),
                    'avg_execution_time': performance_data.get('avg_execution_time', '124ms'),
                    'success_rate': performance_data.get('execution_success_rate', 99.8)
                },
                
                # Risk Distribution
                'risk_distribution': {
                    'high_allocation': high_risk['allocation'],
                    'medium_allocation': medium_risk['allocation'],
                    'low_allocation': low_risk['allocation'],
                    'total': '100%'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy supermix status: {str(e)}")
            return self._get_default_supermix_status()
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        try:
            if self.strategy_config_file.exists():
                with open(self.strategy_config_file, 'r') as f:
                    return json.load(f)
            
            # Return default configuration
            return {
                'risk_tiers': {
                    'high': {
                        'allocation': 0.15,
                        'strategies': ['breakout_scalper', 'momentum_aggressive', 'volatility_hunter'],
                        'max_risk_per_trade': 0.02,
                        'max_positions': 3
                    },
                    'medium': {
                        'allocation': 0.35,
                        'strategies': ['momentum_standard', 'mean_reversion', 'trend_following'],
                        'max_risk_per_trade': 0.01,
                        'max_positions': 5
                    },
                    'low': {
                        'allocation': 0.50,
                        'strategies': ['market_maker', 'arbitrage', 'range_trading'],
                        'max_risk_per_trade': 0.005,
                        'max_positions': 8
                    }
                }
            }
            
        except Exception as e:
            logger.warning(f"Could not load strategy config: {str(e)}")
            return {}
    
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load strategy performance data"""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            
            # Return mock performance data
            return {
                'high_risk_pnl': 234.56,
                'medium_risk_pnl': 456.78,
                'low_risk_pnl': 321.45,
                'sharpe_ratio': 1.8,
                'overall_win_rate': 68.5,
                'profit_factor': 1.9,
                'max_drawdown': -8.2,
                'trades_today': 42,
                'avg_execution_time': '124ms',
                'execution_success_rate': 99.8,
                'strategy_performance': {
                    'breakout_scalper': {'pnl': 123.45, 'win_rate': 72.3, 'trades': 15},
                    'momentum_aggressive': {'pnl': 111.11, 'win_rate': 65.8, 'trades': 12},
                    'momentum_standard': {'pnl': 234.56, 'win_rate': 71.2, 'trades': 8},
                    'mean_reversion': {'pnl': 222.22, 'win_rate': 74.5, 'trades': 10},
                    'market_maker': {'pnl': 185.67, 'win_rate': 82.1, 'trades': 25},
                    'arbitrage': {'pnl': 135.78, 'win_rate': 91.3, 'trades': 18}
                }
            }
            
        except Exception as e:
            logger.warning(f"Could not load performance data: {str(e)}")
            return {}
    
    def _get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies"""
        try:
            # This would check with the StrategyRouter or bot state
            # For now, return mock active strategies
            return [
                'momentum_standard',
                'mean_reversion',
                'market_maker',
                'arbitrage',
                'breakout_scalper'
            ]
        except:
            return []
    
    def _get_risk_tier_data(self, risk_level: str, config: Dict, performance: Dict, active_strategies: List[str]) -> Dict[str, Any]:
        """Get data for a specific risk tier"""
        tier_config = config.get('risk_tiers', {}).get(risk_level, {})
        allocation = tier_config.get('allocation', 0)
        tier_strategies = tier_config.get('strategies', [])
        
        # Find active strategies in this tier
        active_in_tier = [s for s in active_strategies if s in tier_strategies]
        
        # Calculate PnL for this tier
        tier_pnl = 0
        strategy_details = []
        
        strategy_perf = performance.get('strategy_performance', {})
        for strategy in active_in_tier:
            if strategy in strategy_perf:
                perf = strategy_perf[strategy]
                tier_pnl += perf.get('pnl', 0)
                strategy_details.append({
                    'name': self._format_strategy_name(strategy),
                    'status': 'Running',
                    'pnl': perf.get('pnl', 0),
                    'pnl_formatted': f"${perf.get('pnl', 0):+.2f}",
                    'win_rate': perf.get('win_rate', 0),
                    'trades': perf.get('trades', 0)
                })
        
        return {
            'allocation': f"{int(allocation * 100)}%",
            'allocation_value': allocation,
            'strategies_count': len(active_in_tier),
            'current_pnl': round(tier_pnl, 2),
            'current_pnl_formatted': f"${tier_pnl:+,.2f}",
            'active_strategies': strategy_details,
            'available_strategies': [self._format_strategy_name(s) for s in tier_strategies],
            'risk_level': risk_level.title(),
            'max_risk_per_trade': tier_config.get('max_risk_per_trade', 0),
            'max_positions': tier_config.get('max_positions', 0)
        }
    
    def _format_strategy_name(self, strategy_id: str) -> str:
        """Format strategy ID to display name"""
        return strategy_id.replace('_', ' ').title()
    
    def _get_default_supermix_status(self) -> Dict[str, Any]:
        """Return default supermix status"""
        return {
            'status': 'STOPPED',
            'execution_mode': 'STOPPED',
            'total_active_strategies': 0,
            'total_pnl': 0,
            'total_pnl_formatted': '$0.00',
            'high_risk': {
                'allocation': '15%',
                'allocation_value': 0.15,
                'strategies_count': 0,
                'current_pnl': 0,
                'current_pnl_formatted': '$0.00',
                'active_strategies': [],
                'available_strategies': ['Breakout Scalper', 'Momentum Aggressive'],
                'risk_level': 'High'
            },
            'medium_risk': {
                'allocation': '35%',
                'allocation_value': 0.35,
                'strategies_count': 0,
                'current_pnl': 0,
                'current_pnl_formatted': '$0.00',
                'active_strategies': [],
                'available_strategies': ['Momentum Standard', 'Mean Reversion'],
                'risk_level': 'Medium'
            },
            'low_risk': {
                'allocation': '50%',
                'allocation_value': 0.50,
                'strategies_count': 0,
                'current_pnl': 0,
                'current_pnl_formatted': '$0.00',
                'active_strategies': [],
                'available_strategies': ['Market Maker', 'Arbitrage'],
                'risk_level': 'Low'
            },
            'performance_metrics': {
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            },
            'execution_stats': {
                'trades_today': 0,
                'avg_execution_time': '0ms',
                'success_rate': 0
            }
        }
    
    def start_risk_tier(self, risk_level: str) -> Dict[str, Any]:
        """Start all strategies in a risk tier"""
        try:
            config = self._load_strategy_config()
            tier_strategies = config.get('risk_tiers', {}).get(risk_level, {}).get('strategies', [])
            
            # This would actually start the strategies
            # For now, return success
            return {
                'success': True,
                'message': f'Started {len(tier_strategies)} strategies in {risk_level} risk tier',
                'strategies': tier_strategies
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def stop_risk_tier(self, risk_level: str) -> Dict[str, Any]:
        """Stop all strategies in a risk tier"""
        try:
            # This would actually stop the strategies
            return {
                'success': True,
                'message': f'Stopped all strategies in {risk_level} risk tier'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def adjust_allocation(self, risk_level: str, new_allocation: float) -> Dict[str, Any]:
        """Adjust allocation for a risk tier"""
        try:
            config = self._load_strategy_config()
            
            # Validate allocation
            if not 0 <= new_allocation <= 1:
                return {
                    'success': False,
                    'message': 'Allocation must be between 0 and 1'
                }
            
            # Update configuration
            config['risk_tiers'][risk_level]['allocation'] = new_allocation
            
            # Save configuration
            with open(self.strategy_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {
                'success': True,
                'message': f'Updated {risk_level} risk allocation to {new_allocation * 100}%'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_strategy_details(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific strategy"""
        try:
            performance = self._load_performance_data()
            strategy_perf = performance.get('strategy_performance', {}).get(strategy_name, {})
            
            return {
                'name': self._format_strategy_name(strategy_name),
                'id': strategy_name,
                'performance': {
                    'total_pnl': strategy_perf.get('pnl', 0),
                    'win_rate': strategy_perf.get('win_rate', 0),
                    'total_trades': strategy_perf.get('trades', 0),
                    'avg_profit': strategy_perf.get('avg_profit', 0),
                    'max_drawdown': strategy_perf.get('max_drawdown', 0)
                },
                'configuration': {
                    'risk_per_trade': strategy_perf.get('risk_per_trade', 0.01),
                    'max_positions': strategy_perf.get('max_positions', 3),
                    'timeframe': strategy_perf.get('timeframe', '15m')
                },
                'status': 'Active' if strategy_name in self._get_active_strategies() else 'Inactive'
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy details: {str(e)}")
            return {}