# risk/portfolio_manager.py
"""
Advanced Portfolio Manager with Diversification, Correlation Management, and Risk Parity
Manages portfolio-level risk and optimal asset allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class AssetAllocation:
    """Asset allocation result"""
    symbol: str
    target_weight: float
    current_weight: float
    rebalance_amount: float
    risk_contribution: float
    expected_return: float
    volatility: float
    
    @property
    def weight_difference(self) -> float:
        """Difference between target and current weight"""
        return self.target_weight - self.current_weight


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown: float
    concentration_risk: float
    correlation_risk: float
    diversification_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy logging/reporting"""
        return {
            'total_value': self.total_value,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'var_95': self.var_95,
            'max_drawdown': self.max_drawdown,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'diversification_ratio': self.diversification_ratio
        }


class PortfolioManager:
    """
    Advanced Portfolio Manager implementing modern portfolio theory,
    risk parity, and dynamic rebalancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.portfolio_config = config.get('portfolio', {})
        
        # Portfolio parameters
        self.max_positions = self.portfolio_config.get('max_positions', 10)
        self.min_weight = self.portfolio_config.get('min_weight', 0.05)  # 5%
        self.max_weight = self.portfolio_config.get('max_weight', 0.25)  # 25%
        self.rebalance_threshold = self.portfolio_config.get('rebalance_threshold', 0.05)  # 5%
        
        # Risk parameters
        self.max_correlation = self.portfolio_config.get('max_correlation', 0.8)
        self.correlation_lookback = self.portfolio_config.get('correlation_lookback', 60)  # days
        self.risk_free_rate = self.portfolio_config.get('risk_free_rate', 0.02)  # 2%
        
        # Rebalancing parameters
        self.rebalance_frequency = self.portfolio_config.get('rebalance_frequency', 'weekly')
        self.min_trade_size = self.portfolio_config.get('min_trade_size', 10)  # $10
        
        # Risk budget allocation
        self.use_risk_parity = self.portfolio_config.get('use_risk_parity', False)
        self.target_volatility = self.portfolio_config.get('target_volatility', 0.15)  # 15%
        
        # Internal state
        self.correlation_matrix = pd.DataFrame()
        self.returns_data = pd.DataFrame()
        self.last_rebalance = None
        self.allocation_history = []
        
        logger.info("PortfolioManager initialized with advanced risk management")
    
    def update_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix and returns data"""
        try:
            returns_dict = {}
            
            for symbol, data in market_data.items():
                if len(data) >= 2:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_dict[symbol] = returns
            
            if len(returns_dict) >= 2:
                # Align returns data
                returns_df = pd.DataFrame(returns_dict)
                returns_df = returns_df.dropna()
                
                # Keep only recent data for correlation calculation
                if len(returns_df) > self.correlation_lookback:
                    returns_df = returns_df.tail(self.correlation_lookback)
                
                self.returns_data = returns_df
                self.correlation_matrix = returns_df.corr()
                
                logger.debug(f"Updated correlation matrix for {len(returns_dict)} assets")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def calculate_optimal_allocation(self, 
                                   current_positions: Dict[str, float],
                                   expected_returns: Dict[str, float],
                                   method: str = 'risk_parity') -> Dict[str, AssetAllocation]:
        """
        Calculate optimal portfolio allocation using specified method
        
        Methods:
        - 'risk_parity': Equal risk contribution
        - 'mean_variance': Markowitz optimization
        - 'equal_weight': Equal weight allocation
        - 'minimum_variance': Minimum variance portfolio
        """
        try:
            if len(current_positions) == 0:
                return {}
            
            symbols = list(current_positions.keys())
            current_weights = self._normalize_weights(current_positions)
            
            # Calculate target weights based on method
            if method == 'risk_parity':
                target_weights = self._calculate_risk_parity_weights(symbols)
            elif method == 'mean_variance':
                target_weights = self._calculate_mean_variance_weights(symbols, expected_returns)
            elif method == 'minimum_variance':
                target_weights = self._calculate_minimum_variance_weights(symbols)
            else:  # equal_weight
                target_weights = self._calculate_equal_weights(symbols)
            
            # Create allocation objects
            allocations = {}
            total_value = sum(current_positions.values())
            
            for symbol in symbols:
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                rebalance_amount = (target_weight - current_weight) * total_value
                
                # Calculate risk contribution
                risk_contribution = self._calculate_risk_contribution(symbol, target_weights)
                
                # Get asset metrics
                expected_return = expected_returns.get(symbol, 0)
                volatility = self._get_asset_volatility(symbol)
                
                allocations[symbol] = AssetAllocation(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    rebalance_amount=rebalance_amount,
                    risk_contribution=risk_contribution,
                    expected_return=expected_return,
                    volatility=volatility
                )
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            return {}
    
    def _calculate_risk_parity_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate risk parity weights (equal risk contribution)"""
        try:
            if len(symbols) < 2 or self.correlation_matrix.empty:
                # Fallback to equal weights
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            # Get relevant correlation matrix
            available_symbols = [s for s in symbols if s in self.correlation_matrix.index]
            if len(available_symbols) < 2:
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            
            # Get volatilities
            volatilities = np.array([self._get_asset_volatility(symbol) for symbol in available_symbols])
            
            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
            
            # Risk parity optimization
            def risk_budget_objective(weights, cov_matrix):
                """Objective function for risk parity"""
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_var)
                
                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # We want equal risk contributions
                target_risk = portfolio_vol / len(weights)
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in available_symbols]
            
            # Initial guess
            x0 = np.array([1.0 / len(available_symbols)] * len(available_symbols))
            
            # Optimize
            result = minimize(
                risk_budget_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(available_symbols, result.x)}
                # Handle symbols not in correlation matrix
                for symbol in symbols:
                    if symbol not in weights_dict:
                        weights_dict[symbol] = self.min_weight
                
                return self._normalize_weights(weights_dict)
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
                
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    def _calculate_mean_variance_weights(self, 
                                       symbols: List[str], 
                                       expected_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate mean-variance optimal weights (Markowitz)"""
        try:
            available_symbols = [s for s in symbols if s in self.correlation_matrix.index and s in expected_returns]
            if len(available_symbols) < 2:
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            # Get correlation matrix and expected returns
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            returns_vector = np.array([expected_returns[symbol] for symbol in available_symbols])
            volatilities = np.array([self._get_asset_volatility(symbol) for symbol in available_symbols])
            
            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
            
            # Mean-variance optimization with target return
            def portfolio_variance(weights, cov_matrix):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Target return (conservative: average of all expected returns)
            target_return = np.mean(returns_vector)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, returns_vector) - target_return}
            ]
            bounds = [(self.min_weight, self.max_weight) for _ in available_symbols]
            
            # Initial guess
            x0 = np.array([1.0 / len(available_symbols)] * len(available_symbols))
            
            # Optimize
            result = minimize(
                portfolio_variance,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(available_symbols, result.x)}
                # Handle symbols not in correlation matrix
                for symbol in symbols:
                    if symbol not in weights_dict:
                        weights_dict[symbol] = self.min_weight
                
                return self._normalize_weights(weights_dict)
            else:
                logger.warning("Mean-variance optimization failed, using equal weights")
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
                
        except Exception as e:
            logger.error(f"Error in mean-variance calculation: {e}")
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    def _calculate_minimum_variance_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate minimum variance portfolio weights"""
        try:
            available_symbols = [s for s in symbols if s in self.correlation_matrix.index]
            if len(available_symbols) < 2:
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            volatilities = np.array([self._get_asset_volatility(symbol) for symbol in available_symbols])
            
            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
            
            # Minimum variance optimization
            def portfolio_variance(weights, cov_matrix):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in available_symbols]
            
            # Initial guess
            x0 = np.array([1.0 / len(available_symbols)] * len(available_symbols))
            
            # Optimize
            result = minimize(
                portfolio_variance,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(available_symbols, result.x)}
                # Handle symbols not in correlation matrix
                for symbol in symbols:
                    if symbol not in weights_dict:
                        weights_dict[symbol] = self.min_weight
                
                return self._normalize_weights(weights_dict)
            else:
                logger.warning("Minimum variance optimization failed, using equal weights")
                return {symbol: 1.0 / len(symbols) for symbol in symbols}
                
        except Exception as e:
            logger.error(f"Error in minimum variance calculation: {e}")
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    def _calculate_equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate equal weight allocation"""
        return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    def _get_asset_volatility(self, symbol: str) -> float:
        """Get asset volatility from returns data"""
        try:
            if symbol in self.returns_data.columns:
                returns = self.returns_data[symbol]
                volatility = returns.std() * np.sqrt(252)  # Annualized
                return max(volatility, 0.1)  # Minimum 10% volatility
            else:
                return 0.2  # Default 20% volatility
        except Exception:
            return 0.2
    
    def _calculate_risk_contribution(self, symbol: str, weights: Dict[str, float]) -> float:
        """Calculate risk contribution of an asset to portfolio"""
        try:
            if symbol not in self.correlation_matrix.index or len(weights) < 2:
                return 1.0 / len(weights)  # Equal contribution fallback
            
            # Get portfolio weights as array
            available_symbols = [s for s in weights.keys() if s in self.correlation_matrix.index]
            if len(available_symbols) < 2:
                return 1.0 / len(weights)
            
            weight_array = np.array([weights.get(s, 0) for s in available_symbols])
            volatilities = np.array([self._get_asset_volatility(s) for s in available_symbols])
            
            # Covariance matrix
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
            
            # Portfolio variance
            portfolio_var = np.dot(weight_array, np.dot(cov_matrix, weight_array))
            portfolio_vol = np.sqrt(portfolio_var)
            
            if portfolio_vol == 0:
                return 1.0 / len(weights)
            
            # Marginal risk contribution
            symbol_idx = available_symbols.index(symbol) if symbol in available_symbols else 0
            marginal_contrib = np.dot(cov_matrix[symbol_idx], weight_array) / portfolio_vol
            
            # Risk contribution
            risk_contrib = weights.get(symbol, 0) * marginal_contrib
            
            return risk_contrib
            
        except Exception as e:
            logger.error(f"Error calculating risk contribution for {symbol}: {e}")
            return 1.0 / len(weights)
    
    def should_rebalance(self, allocations: Dict[str, AssetAllocation]) -> bool:
        """Determine if portfolio should be rebalanced"""
        try:
            # Check time-based rebalancing
            if self.last_rebalance:
                time_since_rebalance = datetime.now() - self.last_rebalance
                
                if self.rebalance_frequency == 'daily' and time_since_rebalance.days >= 1:
                    return True
                elif self.rebalance_frequency == 'weekly' and time_since_rebalance.days >= 7:
                    return True
                elif self.rebalance_frequency == 'monthly' and time_since_rebalance.days >= 30:
                    return True
            
            # Check threshold-based rebalancing
            max_deviation = max(abs(alloc.weight_difference) for alloc in allocations.values())
            
            return max_deviation > self.rebalance_threshold
            
        except Exception as e:
            logger.error(f"Error checking rebalance condition: {e}")
            return False
    
    def generate_rebalance_orders(self, 
                                allocations: Dict[str, AssetAllocation],
                                current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalance orders"""
        try:
            orders = []
            
            for symbol, allocation in allocations.items():
                if abs(allocation.rebalance_amount) >= self.min_trade_size:
                    current_price = current_prices.get(symbol, 0)
                    if current_price > 0:
                        quantity = allocation.rebalance_amount / current_price
                        
                        order = {
                            'symbol': symbol,
                            'side': 'buy' if allocation.rebalance_amount > 0 else 'sell',
                            'quantity': abs(quantity),
                            'amount': abs(allocation.rebalance_amount),
                            'reason': 'rebalance',
                            'weight_target': allocation.target_weight,
                            'weight_current': allocation.current_weight,
                            'priority': abs(allocation.weight_difference)  # Higher deviation = higher priority
                        }
                        orders.append(order)
            
            # Sort by priority (highest deviation first)
            orders.sort(key=lambda x: x['priority'], reverse=True)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}")
            return []
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, float],
                                  expected_returns: Dict[str, float] = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            total_value = sum(positions.values())
            if total_value == 0:
                return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            weights = self._normalize_weights(positions)
            expected_returns = expected_returns or {}
            
            # Expected return
            portfolio_return = sum(
                weight * expected_returns.get(symbol, 0) 
                for symbol, weight in weights.items()
            )
            
            # Portfolio volatility
            portfolio_volatility = self._calculate_portfolio_volatility(weights)
            
            # Sharpe ratio
            excess_return = portfolio_return - self.risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # VaR (95%)
            var_95 = 1.645 * portfolio_volatility * np.sqrt(1/252)  # Daily VaR
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(weight**2 for weight in weights.values())
            
            # Correlation risk (average pairwise correlation)
            correlation_risk = self._calculate_correlation_risk(weights)
            
            # Diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(weights)
            
            # Max drawdown (if we have returns data)
            max_drawdown = self._calculate_max_drawdown(weights)
            
            return PortfolioMetrics(
                total_value=total_value,
                expected_return=portfolio_return,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(total_value, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility"""
        try:
            available_symbols = [s for s in weights.keys() if s in self.correlation_matrix.index]
            if len(available_symbols) < 2:
                # Single asset or no correlation data
                return np.mean([self._get_asset_volatility(symbol) for symbol in weights.keys()])
            
            weight_array = np.array([weights.get(s, 0) for s in available_symbols])
            volatilities = np.array([self._get_asset_volatility(s) for s in available_symbols])
            
            # Covariance matrix
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
            
            # Portfolio variance
            portfolio_var = np.dot(weight_array, np.dot(cov_matrix, weight_array))
            
            return np.sqrt(portfolio_var)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.2  # Default volatility
    
    def _calculate_correlation_risk(self, weights: Dict[str, float]) -> float:
        """Calculate average correlation risk"""
        try:
            available_symbols = [s for s in weights.keys() if s in self.correlation_matrix.index]
            if len(available_symbols) < 2:
                return 0.0
            
            corr_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]
            
            # Weight-averaged correlation
            total_weight = 0
            weighted_corr = 0
            
            for i, symbol1 in enumerate(available_symbols):
                for j, symbol2 in enumerate(available_symbols):
                    if i != j:
                        weight_product = weights.get(symbol1, 0) * weights.get(symbol2, 0)
                        correlation = corr_matrix.loc[symbol1, symbol2]
                        weighted_corr += weight_product * abs(correlation)
                        total_weight += weight_product
            
            return weighted_corr / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate diversification ratio"""
        try:
            # Diversification ratio = weighted average volatility / portfolio volatility
            weighted_avg_vol = sum(
                weight * self._get_asset_volatility(symbol) 
                for symbol, weight in weights.items()
            )
            
            portfolio_vol = self._calculate_portfolio_volatility(weights)
            
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0
    
    def _calculate_max_drawdown(self, weights: Dict[str, float]) -> float:
        """Calculate maximum drawdown from returns data"""
        try:
            if self.returns_data.empty:
                return 0.0
            
            # Calculate portfolio returns
            available_symbols = [s for s in weights.keys() if s in self.returns_data.columns]
            if len(available_symbols) == 0:
                return 0.0
            
            portfolio_returns = sum(
                weights.get(symbol, 0) * self.returns_data[symbol] 
                for symbol in available_symbols
            )
            
            # Calculate cumulative returns
            cumulative = (1 + portfolio_returns).cumprod()
            
            # Calculate drawdown
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            return abs(drawdown.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _normalize_weights(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Normalize position values to weights"""
        total_value = sum(positions.values())
        if total_value == 0:
            return {}
        
        return {symbol: value / total_value for symbol, value in positions.items()}
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get current correlation matrix"""
        return self.correlation_matrix.copy()
    
    def get_risk_budget_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Get risk budget allocation (risk parity)"""
        return self._calculate_risk_parity_weights(symbols)
    
    def mark_rebalanced(self):
        """Mark portfolio as rebalanced"""
        self.last_rebalance = datetime.now()
    
    def add_to_allocation_history(self, allocations: Dict[str, AssetAllocation]):
        """Add allocation to history for tracking"""
        allocation_record = {
            'timestamp': datetime.now(),
            'allocations': {symbol: alloc.target_weight for symbol, alloc in allocations.items()},
            'total_positions': len(allocations)
        }
        
        self.allocation_history.append(allocation_record)
        
        # Keep only last 100 records
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]