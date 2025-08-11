# risk/risk_monitor.py
"""
Real-time Risk Monitor with Drawdown Tracking and Alert System
Monitors portfolio risk metrics and triggers alerts/actions when thresholds are breached
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import json
from threading import Thread, Lock
import time

from .position_calculator import PositionCalculator, StrategyStats, MarketConditions
from .portfolio_manager import PortfolioManager, PortfolioMetrics

# Try to import notifier (graceful fallback if not available)
try:
    from utils.notifier import send_warning, send_error, send_critical, AlertLevel, AlertType
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    alert_type: str
    severity: str  # WARNING, ERROR, CRITICAL
    metric: str
    current_value: float
    threshold_value: float
    message: str
    action_taken: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'metric': self.metric,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'action_taken': self.action_taken
        }


@dataclass
class DrawdownMetrics:
    """Drawdown tracking metrics"""
    current_drawdown: float
    max_drawdown: float
    peak_value: float
    current_value: float
    drawdown_duration_days: int
    time_to_recovery_estimate: Optional[int] = None  # days
    underwater_curve: List[float] = None  # Historical drawdown values
    
    @property
    def is_new_peak(self) -> bool:
        """Check if current value is new peak"""
        return self.current_value >= self.peak_value
    
    @property
    def recovery_percentage(self) -> float:
        """Percentage recovered from max drawdown"""
        if self.max_drawdown == 0:
            return 1.0
        return 1.0 - (self.current_drawdown / self.max_drawdown)


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: datetime
    portfolio_value: float
    drawdown_metrics: DrawdownMetrics
    portfolio_metrics: PortfolioMetrics
    var_daily: float
    var_weekly: float
    var_monthly: float
    stress_test_results: Dict[str, float]
    active_alerts: List[RiskAlert]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'drawdown_metrics': asdict(self.drawdown_metrics),
            'portfolio_metrics': self.portfolio_metrics.to_dict(),
            'var_daily': self.var_daily,
            'var_weekly': self.var_weekly,
            'var_monthly': self.var_monthly,
            'stress_test_results': self.stress_test_results,
            'active_alerts': [alert.to_dict() for alert in self.active_alerts]
        }


class RiskMonitor:
    """
    Real-time Risk Monitoring System
    Tracks portfolio risk metrics, drawdown, and triggers alerts/actions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get('risk', {})
        self.monitoring_config = config.get('risk_monitoring', {})
        
        # Load risk profile
        self.risk_profile = self._load_risk_profile()
        
        # Monitoring parameters
        self.check_frequency = self.monitoring_config.get('check_frequency_minutes', 15)
        self.alert_thresholds = self.monitoring_config.get('alert_thresholds', {})
        
        # Auto position reduction
        self.auto_reduce_enabled = self.monitoring_config.get('auto_reduce_positions', {}).get('enabled', True)
        self.auto_reduce_threshold = self.monitoring_config.get('auto_reduce_positions', {}).get('drawdown_threshold', 0.9)
        self.reduction_factor = self.monitoring_config.get('auto_reduce_positions', {}).get('reduction_factor', 0.5)
        
        # State tracking
        self.portfolio_history = []
        self.drawdown_start = None
        self.peak_value = 0.0
        self.max_drawdown_seen = 0.0
        self.active_alerts = []
        self.last_alert_times = {}  # Prevent alert spam
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = Lock()
        
        # Components
        self.position_calculator = None
        self.portfolio_manager = None
        
        logger.info(f"RiskMonitor initialized with {self.check_frequency}min frequency")
    
    def _load_risk_profile(self) -> Dict[str, Any]:
        """Load risk profile from config"""
        try:
            profile_name = self.config.get('risk_profile', 'balanced')
            
            # Try to load from risk_profiles.json
            try:
                with open('config/risk_profiles.json', 'r') as f:
                    profiles = json.load(f)
                    profile = profiles['risk_profiles'].get(profile_name, {})
                    if profile:
                        logger.info(f"Loaded risk profile: {profile_name}")
                        return profile.get('parameters', {})
            except FileNotFoundError:
                logger.warning("Risk profiles file not found, using defaults")
            
            # Fallback to config defaults
            return {
                'max_drawdown': 0.20,
                'target_sharpe': 1.5,
                'max_risk_per_trade': 0.02,
                'var_confidence': 0.95
            }
            
        except Exception as e:
            logger.error(f"Error loading risk profile: {e}")
            return {}
    
    def set_components(self, position_calculator: PositionCalculator, portfolio_manager: PortfolioManager):
        """Set risk management components"""
        self.position_calculator = position_calculator
        self.portfolio_manager = portfolio_manager
        logger.info("Risk monitoring components set")
    
    def start_monitoring(self):
        """Start background risk monitoring"""
        if self.monitoring_active:
            logger.warning("Risk monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop background risk monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        while self.monitoring_active:
            try:
                time.sleep(self.check_frequency * 60)  # Convert minutes to seconds
                
                if not self.monitoring_active:
                    break
                
                # Run risk checks (this should be called with real data from main bot)
                logger.debug("Risk monitoring check (background)")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def update_portfolio(self, 
                        current_value: float, 
                        positions: Dict[str, float],
                        market_data: Dict[str, pd.DataFrame] = None,
                        strategy_stats: Dict[str, StrategyStats] = None) -> RiskMetrics:
        """
        Update portfolio data and perform risk checks
        """
        try:
            with self.lock:
                timestamp = datetime.now()
                
                # Update peak and calculate drawdown
                drawdown_metrics = self._calculate_drawdown_metrics(current_value, timestamp)
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(positions, market_data)
                
                # Calculate VaR
                var_metrics = self._calculate_var_metrics(positions, market_data)
                
                # Run stress tests
                stress_results = self._run_stress_tests(positions, market_data)
                
                # Check for alerts
                new_alerts = self._check_risk_alerts(
                    current_value, drawdown_metrics, portfolio_metrics, var_metrics
                )
                
                # Update active alerts
                self._update_active_alerts(new_alerts)
                
                # Create risk metrics snapshot
                risk_metrics = RiskMetrics(
                    timestamp=timestamp,
                    portfolio_value=current_value,
                    drawdown_metrics=drawdown_metrics,
                    portfolio_metrics=portfolio_metrics,
                    var_daily=var_metrics['daily'],
                    var_weekly=var_metrics['weekly'],
                    var_monthly=var_metrics['monthly'],
                    stress_test_results=stress_results,
                    active_alerts=self.active_alerts.copy()
                )
                
                # Store in history
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'value': current_value,
                    'drawdown': drawdown_metrics.current_drawdown
                })
                
                # Keep only last 1000 records
                if len(self.portfolio_history) > 1000:
                    self.portfolio_history = self.portfolio_history[-1000:]
                
                # Process alerts and actions
                self._process_alerts(risk_metrics)
                
                return risk_metrics
            
        except Exception as e:
            logger.error(f"Error updating portfolio risk metrics: {e}")
            # Return minimal risk metrics
            return self._create_fallback_metrics(current_value)
    
    def _calculate_drawdown_metrics(self, current_value: float, timestamp: datetime) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics"""
        try:
            # Update peak
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.drawdown_start = None  # Reset drawdown start
            
            # Calculate current drawdown
            current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
            
            # Update max drawdown
            if current_drawdown > self.max_drawdown_seen:
                self.max_drawdown_seen = current_drawdown
            
            # Calculate drawdown duration
            drawdown_duration = 0
            if current_drawdown > 0:
                if self.drawdown_start is None:
                    self.drawdown_start = timestamp
                drawdown_duration = (timestamp - self.drawdown_start).days
            
            # Estimate recovery time
            recovery_estimate = self._estimate_recovery_time(current_drawdown, current_value)
            
            # Build underwater curve from recent history
            underwater_curve = []
            if len(self.portfolio_history) > 0:
                for record in self.portfolio_history[-30:]:  # Last 30 records
                    underwater_curve.append(record.get('drawdown', 0))
            
            return DrawdownMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=self.max_drawdown_seen,
                peak_value=self.peak_value,
                current_value=current_value,
                drawdown_duration_days=drawdown_duration,
                time_to_recovery_estimate=recovery_estimate,
                underwater_curve=underwater_curve
            )
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return DrawdownMetrics(0, 0, current_value, current_value, 0)
    
    def _estimate_recovery_time(self, current_drawdown: float, current_value: float) -> Optional[int]:
        """Estimate time to recovery based on historical performance"""
        try:
            if current_drawdown <= 0.01:  # Less than 1% drawdown
                return 0
            
            # Calculate average daily return from recent history
            if len(self.portfolio_history) < 10:
                return None
            
            recent_values = [record['value'] for record in self.portfolio_history[-30:]]
            daily_returns = pd.Series(recent_values).pct_change().dropna()
            
            if len(daily_returns) == 0:
                return None
            
            avg_daily_return = daily_returns.mean()
            
            if avg_daily_return <= 0:
                return None  # Can't estimate if negative returns
            
            # Estimate days to recover to peak
            amount_to_recover = self.peak_value - current_value
            daily_recovery_amount = current_value * avg_daily_return
            
            if daily_recovery_amount <= 0:
                return None
            
            estimated_days = int(amount_to_recover / daily_recovery_amount)
            return min(estimated_days, 365)  # Cap at 1 year
            
        except Exception as e:
            logger.error(f"Error estimating recovery time: {e}")
            return None
    
    def _calculate_portfolio_metrics(self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame] = None) -> PortfolioMetrics:
        """Calculate portfolio metrics using portfolio manager"""
        try:
            if self.portfolio_manager and market_data:
                self.portfolio_manager.update_market_data(market_data)
                return self.portfolio_manager.calculate_portfolio_metrics(positions)
            else:
                # Fallback basic metrics
                total_value = sum(positions.values())
                return PortfolioMetrics(
                    total_value=total_value,
                    expected_return=0.0,
                    volatility=0.15,  # Default 15%
                    sharpe_ratio=0.0,
                    var_95=0.0,
                    max_drawdown=self.max_drawdown_seen,
                    concentration_risk=0.0,
                    correlation_risk=0.0,
                    diversification_ratio=1.0
                )
        
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            total_value = sum(positions.values()) if positions else 0
            return PortfolioMetrics(total_value, 0, 0.15, 0, 0, 0, 0, 0, 1.0)
    
    def _calculate_var_metrics(self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate Value at Risk for different time horizons"""
        try:
            if not self.position_calculator or not market_data:
                return {'daily': 0.0, 'weekly': 0.0, 'monthly': 0.0}
            
            # Use portfolio manager for VaR calculation if available
            if self.portfolio_manager:
                correlation_matrix = self.portfolio_manager.get_correlation_matrix()
                if not correlation_matrix.empty:
                    symbols = list(positions.keys())
                    available_symbols = [s for s in symbols if s in correlation_matrix.index]
                    
                    if len(available_symbols) >= 2:
                        correlations = correlation_matrix.loc[available_symbols, available_symbols].values
                        volatilities = np.array([
                            self.portfolio_manager._get_asset_volatility(symbol) 
                            for symbol in available_symbols
                        ])
                        
                        # Daily VaR
                        daily_var = self.position_calculator.calculate_portfolio_var(
                            {s: positions.get(s, 0) for s in available_symbols},
                            correlations,
                            volatilities / np.sqrt(252)  # Convert to daily
                        )
                        
                        return {
                            'daily': daily_var,
                            'weekly': daily_var * np.sqrt(7),
                            'monthly': daily_var * np.sqrt(30)
                        }
            
            # Fallback: simple VaR estimation
            total_value = sum(positions.values())
            estimated_daily_vol = 0.02  # 2% daily volatility
            z_score = 1.645  # 95% confidence
            
            daily_var = total_value * estimated_daily_vol * z_score
            
            return {
                'daily': daily_var,
                'weekly': daily_var * np.sqrt(7),
                'monthly': daily_var * np.sqrt(30)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'daily': 0.0, 'weekly': 0.0, 'monthly': 0.0}
    
    def _run_stress_tests(self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame] = None) -> Dict[str, float]:
        """Run portfolio stress tests"""
        try:
            total_value = sum(positions.values())
            if total_value == 0:
                return {}
            
            # Define stress scenarios
            stress_scenarios = {
                'market_crash_10': -0.10,      # 10% market crash
                'market_crash_20': -0.20,      # 20% market crash
                'crypto_winter': -0.50,        # 50% crypto crash
                'volatility_spike': -0.15,     # High volatility scenario
                'correlation_breakdown': -0.08  # Correlation goes to 1
            }
            
            results = {}
            for scenario, shock in stress_scenarios.items():
                # Simple stress test: apply shock to portfolio
                stressed_value = total_value * (1 + shock)
                loss = total_value - stressed_value
                results[scenario] = loss
            
            return results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    def _check_risk_alerts(self, 
                          current_value: float, 
                          drawdown_metrics: DrawdownMetrics,
                          portfolio_metrics: PortfolioMetrics,
                          var_metrics: Dict[str, float]) -> List[RiskAlert]:
        """Check for risk threshold breaches and generate alerts"""
        alerts = []
        timestamp = datetime.now()
        
        try:
            # Drawdown alerts
            max_drawdown_limit = self.risk_profile.get('max_drawdown', 0.20)
            
            if drawdown_metrics.current_drawdown > max_drawdown_limit * self.alert_thresholds.get('drawdown_warning', 0.8):
                severity = 'WARNING'
                if drawdown_metrics.current_drawdown > max_drawdown_limit * self.alert_thresholds.get('drawdown_critical', 0.95):
                    severity = 'CRITICAL'
                elif drawdown_metrics.current_drawdown > max_drawdown_limit * 0.9:
                    severity = 'ERROR'
                
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='DRAWDOWN_BREACH',
                    severity=severity,
                    metric='current_drawdown',
                    current_value=drawdown_metrics.current_drawdown,
                    threshold_value=max_drawdown_limit,
                    message=f"Drawdown {drawdown_metrics.current_drawdown:.1%} exceeds {severity.lower()} threshold"
                ))
            
            # VaR breach alerts
            max_var_daily = current_value * self.risk_profile.get('max_risk_per_trade', 0.02) * 5  # 5x daily risk
            if var_metrics.get('daily', 0) > max_var_daily * self.alert_thresholds.get('var_breach_warning', 1.2):
                severity = 'WARNING'
                if var_metrics['daily'] > max_var_daily * self.alert_thresholds.get('var_breach_critical', 1.5):
                    severity = 'CRITICAL'
                
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='VAR_BREACH',
                    severity=severity,
                    metric='daily_var',
                    current_value=var_metrics['daily'],
                    threshold_value=max_var_daily,
                    message=f"Daily VaR ${var_metrics['daily']:,.0f} exceeds {severity.lower()} threshold"
                ))
            
            # Concentration risk alerts
            if portfolio_metrics.concentration_risk > self.alert_thresholds.get('concentration_warning', 0.4):
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='CONCENTRATION_RISK',
                    severity='WARNING',
                    metric='concentration_risk',
                    current_value=portfolio_metrics.concentration_risk,
                    threshold_value=0.4,
                    message=f"Portfolio concentration risk {portfolio_metrics.concentration_risk:.1%} is high"
                ))
            
            # Correlation risk alerts
            if portfolio_metrics.correlation_risk > self.alert_thresholds.get('correlation_warning', 0.85):
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='CORRELATION_RISK',
                    severity='WARNING',
                    metric='correlation_risk',
                    current_value=portfolio_metrics.correlation_risk,
                    threshold_value=0.85,
                    message=f"Average portfolio correlation {portfolio_metrics.correlation_risk:.1%} is high"
                ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
            return []
    
    def _update_active_alerts(self, new_alerts: List[RiskAlert]):
        """Update active alerts list"""
        try:
            # Remove old alerts of same type
            for new_alert in new_alerts:
                self.active_alerts = [
                    alert for alert in self.active_alerts 
                    if alert.alert_type != new_alert.alert_type
                ]
            
            # Add new alerts
            self.active_alerts.extend(new_alerts)
            
            # Remove alerts older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.active_alerts = [
                alert for alert in self.active_alerts 
                if alert.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error updating active alerts: {e}")
    
    def _process_alerts(self, risk_metrics: RiskMetrics):
        """Process alerts: send notifications and take actions"""
        try:
            for alert in risk_metrics.active_alerts:
                # Check if we've already alerted recently
                alert_key = f"{alert.alert_type}_{alert.severity}"
                last_alert_time = self.last_alert_times.get(alert_key)
                
                # Rate limit: don't send same alert more than once per hour
                if last_alert_time and (datetime.now() - last_alert_time).seconds < 3600:
                    continue
                
                # Send notification
                self._send_risk_notification(alert, risk_metrics)
                
                # Take automatic actions if needed
                action_taken = self._take_automatic_action(alert, risk_metrics)
                if action_taken:
                    alert.action_taken = action_taken
                
                # Update last alert time
                self.last_alert_times[alert_key] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    def _send_risk_notification(self, alert: RiskAlert, risk_metrics: RiskMetrics):
        """Send risk alert notification"""
        try:
            if not NOTIFIER_AVAILABLE:
                logger.warning(f"Risk alert: {alert.message}")
                return
            
            # Prepare message
            message = f"{alert.message}\n\n"
            message += f"Portfolio: ${risk_metrics.portfolio_value:,.2f}\n"
            message += f"Current Drawdown: {risk_metrics.drawdown_metrics.current_drawdown:.1%}\n"
            message += f"Max Drawdown: {risk_metrics.drawdown_metrics.max_drawdown:.1%}"
            
            if risk_metrics.drawdown_metrics.time_to_recovery_estimate:
                message += f"\nEst. Recovery: {risk_metrics.drawdown_metrics.time_to_recovery_estimate} days"
            
            # Send based on severity
            if alert.severity == 'CRITICAL':
                send_critical(message)
            elif alert.severity == 'ERROR':
                send_error(message)
            else:
                send_warning(message)
            
            logger.info(f"Risk alert sent: {alert.alert_type} - {alert.severity}")
            
        except Exception as e:
            logger.error(f"Error sending risk notification: {e}")
    
    def _take_automatic_action(self, alert: RiskAlert, risk_metrics: RiskMetrics) -> Optional[str]:
        """Take automatic action based on alert"""
        try:
            if not self.auto_reduce_enabled:
                return None
            
            # Auto-reduce positions on critical drawdown
            if (alert.alert_type == 'DRAWDOWN_BREACH' and 
                alert.severity == 'CRITICAL' and
                risk_metrics.drawdown_metrics.current_drawdown > self.auto_reduce_threshold):
                
                logger.warning(f"Auto-reducing positions due to {alert.current_value:.1%} drawdown")
                
                # This would integrate with the main trading system
                # For now, just log the action
                action = f"Auto-reduced position sizes by {(1-self.reduction_factor)*100:.0f}%"
                
                if NOTIFIER_AVAILABLE:
                    send_critical(f"🚨 AUTOMATIC RISK ACTION\n\n{action}\n\nDrawdown: {alert.current_value:.1%}")
                
                return action
            
            return None
            
        except Exception as e:
            logger.error(f"Error taking automatic action: {e}")
            return None
    
    def _create_fallback_metrics(self, current_value: float) -> RiskMetrics:
        """Create minimal risk metrics on error"""
        timestamp = datetime.now()
        
        drawdown_metrics = DrawdownMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            peak_value=current_value,
            current_value=current_value,
            drawdown_duration_days=0
        )
        
        portfolio_metrics = PortfolioMetrics(
            total_value=current_value,
            expected_return=0.0,
            volatility=0.15,
            sharpe_ratio=0.0,
            var_95=0.0,
            max_drawdown=0.0,
            concentration_risk=0.0,
            correlation_risk=0.0,
            diversification_ratio=1.0
        )
        
        return RiskMetrics(
            timestamp=timestamp,
            portfolio_value=current_value,
            drawdown_metrics=drawdown_metrics,
            portfolio_metrics=portfolio_metrics,
            var_daily=0.0,
            var_weekly=0.0,
            var_monthly=0.0,
            stress_test_results={},
            active_alerts=[]
        )
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'current_peak': self.peak_value,
                'max_drawdown_seen': self.max_drawdown_seen,
                'active_alerts_count': len(self.active_alerts),
                'last_check': datetime.now().isoformat(),
                'auto_reduce_enabled': self.auto_reduce_enabled,
                'risk_profile': self.risk_profile.get('name', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {'error': str(e)}
    
    def get_drawdown_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get drawdown history for charting"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            history = [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'value': record['value'],
                    'drawdown': record.get('drawdown', 0)
                }
                for record in self.portfolio_history
                if record['timestamp'] > cutoff_date
            ]
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting drawdown history: {e}")
            return []
    
    def export_risk_report(self, filepath: str = None) -> str:
        """Export comprehensive risk report"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"risk_report_{timestamp}.json"
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'risk_profile': self.risk_profile,
                'current_status': self.get_current_risk_status(),
                'drawdown_history': self.get_drawdown_history(90),  # 90 days
                'active_alerts': [alert.to_dict() for alert in self.active_alerts],
                'portfolio_history_count': len(self.portfolio_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Risk report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return ""