
# Dependency Injection Support  
from core.interfaces import ISafetyManager, global_event_bus
# core/safety_manager.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, TYPE_CHECKING, Dict

from config.settings import Settings
from utils.notifier import NotificationManager

# Import advanced risk management components
try:
    from risk.risk_monitor import RiskMonitor
    from risk.position_calculator import PositionCalculator, StrategyStats, MarketConditions, create_market_conditions
    from risk.portfolio_manager import PortfolioManager
    ADVANCED_RISK_AVAILABLE = True
except ImportError:
    ADVANCED_RISK_AVAILABLE = False

# To avoid circular import, use TYPE_CHECKING for type hints
if TYPE_CHECKING:
    # Removed circular import - using events

logger = logging.getLogger(__name__)


class SafetyManager(ISafetyManager):
    """
    Enhanced Safety Manager with advanced risk management components
    Manages safety mechanisms like killswitch based on drawdown and integrates with advanced risk monitoring.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.killswitch_enabled = self.settings.get('risk_management.killswitch.enabled', True)
        self.max_drawdown_percent = self.settings.get('risk_management.killswitch.max_drawdown', 0.15)
        self.auto_reactivate_after_hours = self.settings.get('risk_management.killswitch.auto_reactivate_after_hours',
                                                             24)
        self.notification_on_trigger = self.settings.get('risk_management.killswitch.notification_on_trigger', True)

        self.is_killswitch_active: bool = False
        self.last_killswitch_activation_time: Optional[datetime] = None
        self.peak_equity: float = self.settings.get('trading.initial_capital',
                                                    10000)  # Initial capital as starting peak
        self.current_drawdown_percent: float = 0.0

        self.notification_manager = NotificationManager(self.settings)

        self.trading_bot: Optional['TradingBot'] = None  # Reference to the trading bot instance
        
        # Advanced risk management components
        self.risk_monitor = None
        self.position_calculator = None
        self.portfolio_manager = None
        
        if ADVANCED_RISK_AVAILABLE:
            try:
                # Initialize advanced risk components
                self.position_calculator = PositionCalculator(self.settings)
                self.portfolio_manager = PortfolioManager(self.settings)
                self.risk_monitor = RiskMonitor(self.settings)
                
                # Set up components
                self.risk_monitor.set_components(self.position_calculator, self.portfolio_manager)
                
                # Start risk monitoring
                self.risk_monitor.start_monitoring()
                
                logger.info("Advanced risk management components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced risk management: {e}")
                self.risk_monitor = None
                self.position_calculator = None
                self.portfolio_manager = None

        logger.info(
            f"SafetyManager initialized. Killswitch enabled: {self.killswitch_enabled}, Max Drawdown: {self.max_drawdown_percent:.2%}")

    def set_trading_bot(self, bot: 'TradingBot'):
        """Sets the reference to the trading bot instance."""
        self.trading_bot = bot
        logger.info("SafetyManager received TradingBot instance reference.")

    def update_equity(self, current_equity: float, positions: Dict[str, float] = None, market_data: Dict = None):
        """
        Enhanced equity update with advanced risk monitoring integration.
        Updates the current equity and checks for drawdown to trigger killswitch.
        """
        if not self.killswitch_enabled:
            # Still update advanced risk monitoring even if killswitch disabled
            if self.risk_monitor and positions is not None:
                self._update_advanced_risk_monitoring(current_equity, positions, market_data)
            return

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown_percent = 0.0
        else:
            if self.peak_equity > 0:  # Avoid division by zero
                self.current_drawdown_percent = (self.peak_equity - current_equity) / self.peak_equity
            else:
                self.current_drawdown_percent = 0.0  # Or some other handling for zero peak equity

        # Update advanced risk monitoring if available
        if self.risk_monitor and positions is not None:
            try:
                risk_metrics = self._update_advanced_risk_monitoring(current_equity, positions, market_data)
                
                # Check if advanced risk monitoring suggests killswitch activation
                if risk_metrics and self._should_activate_killswitch_advanced(risk_metrics):
                    return  # Advanced system already handled activation
                    
            except Exception as e:
                logger.error(f"Error in advanced risk monitoring: {e}")

        # Traditional killswitch logic
        if self.current_drawdown_percent >= self.max_drawdown_percent and not self.is_killswitch_active:
            self._activate_killswitch(
                f"Drawdown {self.current_drawdown_percent:.2%} exceeded {self.max_drawdown_percent:.2%} threshold from peak equity of {self.peak_equity:.2f}.")

        self._check_auto_reactivation()

    def _activate_killswitch(self, reason: str):
        """Activates the killswitch and pauses trading operations."""
        self.is_killswitch_active = True
        self.last_killswitch_activation_time = datetime.now()
        logger.critical(f"KILLSWITCH ACTIVATED! Reason: {reason}")
        if self.notification_on_trigger:
            self.notification_manager.send_alert(f"KILLSWITCH ACTIVATED! Reason: {reason}. Bot paused.",
                                                 level="CRITICAL")

        if self.trading_bot:
            # The bot should have its own internal mechanism to pause trading when killswitch is active
            # This can be done by checking `is_killswitch_active()` in its main loop.
            logger.info("Trading operations effectively paused due to killswitch.")
            # Optionally, if the bot needs explicit stopping of active strategies:
            # self.trading_bot.strategy_router._deactivate_all_strategies() 
            # Or self.trading_bot.current_active_strategy.stop_trading() if single strategy

    def _check_auto_reactivation(self):
        """Checks if the killswitch should be automatically deactivated."""
        if self.is_killswitch_active and self.auto_reactivate_after_hours > 0:
            if self.last_killswitch_activation_time and \
                    (
                            datetime.now() - self.last_killswitch_activation_time).total_seconds() / 3600 >= self.auto_reactivate_after_hours:
                self._deactivate_killswitch(f"Automatic reactivation after {self.auto_reactivate_after_hours} hours.")

    def _deactivate_killswitch(self, reason: str):
        """Deactivates the killswitch and resumes trading operations."""
        self.is_killswitch_active = False
        self.last_killswitch_activation_time = None
        self.current_drawdown_percent = 0.0  # Reset drawdown after reactivation
        self.peak_equity = self.trading_bot.position_manager.get_total_capital(
            self.trading_bot.exchange.get_current_prices()) if self.trading_bot else self.settings.get(
            'trading.initial_capital', 10000)  # Reset peak equity

        logger.warning(f"KILLSWITCH DEACTIVATED. Reason: {reason}. Trading can resume.")
        if self.notification_on_trigger:
            self.notification_manager.send_alert(f"KILLSWITCH DEACTIVATED. Reason: {reason}. Bot can resume trading.",
                                                 level="WARNING")

        if self.trading_bot:
            # If the strategy router was used, it might need to re-evaluate regime and reactivate strategies
            if self.trading_bot.strategy_router:
                logger.info("Triggering StrategyRouter to re-evaluate market regime and reactivate strategies.")
                # Force a regime check and strategy update
                # This assumes a mock current_total_capital can be passed,
                # or that strategy_router gets it from bot's position manager
                self.trading_bot.strategy_router.update_market_regime(
                    self.trading_bot.strategy_router.get_current_regime(),
                    # Re-evaluate current regime or force a new check
                    self.trading_bot.position_manager.get_total_capital(self.trading_bot.exchange.get_current_prices())
                )
            logger.info("Trading bot can resume operations.")

    def _update_advanced_risk_monitoring(self, current_equity: float, positions: Dict[str, float], market_data: Dict = None):
        """Update advanced risk monitoring components"""
        try:
            if self.risk_monitor:
                # Convert market_data format if needed
                formatted_market_data = self._format_market_data_for_risk_monitor(market_data)
                
                # Update portfolio with current state
                risk_metrics = self.risk_monitor.update_portfolio(
                    current_value=current_equity,
                    positions=positions,
                    market_data=formatted_market_data
                )
                
                return risk_metrics
        except Exception as e:
            logger.error(f"Error updating advanced risk monitoring: {e}")
            return None
    
    def _format_market_data_for_risk_monitor(self, market_data: Dict) -> Dict:
        """Format market data for risk monitor (if needed)"""
        # This might need adjustment based on how market_data is structured
        # The risk monitor expects Dict[str, pd.DataFrame]
        if market_data is None:
            return {}
        
        # If market_data is already in the right format, return as-is
        if isinstance(market_data, dict):
            return market_data
        
        return {}
    
    def _should_activate_killswitch_advanced(self, risk_metrics) -> bool:
        """Check if advanced risk metrics suggest killswitch activation"""
        try:
            if not risk_metrics:
                return False
            
            # Check if risk monitor already triggered critical alerts
            critical_alerts = [
                alert for alert in risk_metrics.active_alerts 
                if alert.severity == 'CRITICAL'
            ]
            
            if critical_alerts:
                # Check for automatic risk reduction
                auto_reduce_alerts = [
                    alert for alert in critical_alerts 
                    if alert.action_taken and 'Auto-reduced' in alert.action_taken
                ]
                
                if auto_reduce_alerts:
                    self._activate_killswitch(
                        f"Advanced risk system triggered auto-reduction: {auto_reduce_alerts[0].message}"
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking advanced killswitch conditions: {e}")
            return False
    
    def get_optimal_position_size(self, symbol: str, strategy_stats: Dict, market_data: Dict = None) -> float:
        """Get optimal position size using advanced risk management"""
        try:
            if not self.position_calculator:
                return 0.02  # Default 2% risk
            
            # Convert strategy stats to StrategyStats object
            stats = self._convert_to_strategy_stats(strategy_stats)
            
            # Create market conditions
            current_price = market_data.get('current_price', 1.0) if market_data else 1.0
            market_conditions = self._create_market_conditions(symbol, current_price, market_data)
            
            # Calculate optimal position size
            position_size = self.position_calculator.calculate_position_size(
                strategy_stats=stats,
                market_conditions=market_conditions,
                current_equity=self.peak_equity,  # Use current equity
                active_positions={}  # Would need real position data
            )
            
            return position_size.adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size: {e}")
            return 0.02  # Safe fallback
    
    def _convert_to_strategy_stats(self, stats_dict: Dict) -> 'StrategyStats':
        """Convert strategy statistics dictionary to StrategyStats object"""
        try:
            return StrategyStats(
                win_rate=stats_dict.get('win_rate', 0.5),
                avg_win=stats_dict.get('avg_win', 0.02),
                avg_loss=stats_dict.get('avg_loss', -0.01),
                total_trades=stats_dict.get('total_trades', 0),
                profit_factor=stats_dict.get('profit_factor', 1.0),
                sharpe_ratio=stats_dict.get('sharpe_ratio', 0.0),
                max_drawdown=stats_dict.get('max_drawdown', 0.0),
                volatility=stats_dict.get('volatility', 0.15)
            )
        except Exception as e:
            logger.error(f"Error converting strategy stats: {e}")
            # Return safe defaults
            return StrategyStats(0.5, 0.02, -0.01, 0, 1.0, 0.0, 0.0, 0.15)
    
    def _create_market_conditions(self, symbol: str, current_price: float, market_data: Dict = None) -> 'MarketConditions':
        """Create market conditions object"""
        try:
            if market_data and 'dataframe' in market_data:
                return create_market_conditions(
                    symbol=symbol,
                    current_price=current_price,
                    market_data=market_data['dataframe'],
                    correlations=market_data.get('correlations', {}),
                    market_regime=market_data.get('market_regime', 'unknown')
                )
            else:
                # Create with minimal data
                return create_market_conditions(
                    symbol=symbol,
                    current_price=current_price,
                    market_data=None,
                    correlations={},
                    market_regime='unknown'
                )
        except Exception as e:
            logger.error(f"Error creating market conditions: {e}")
            # Return safe defaults
            return MarketConditions(
                symbol=symbol,
                current_price=current_price,
                atr=current_price * 0.02,
                atr_ratio=0.02,
                btc_correlation=0.0,
                eth_correlation=0.0,
                volume_ratio=1.0,
                market_regime='unknown',
                volatility_percentile=0.5
            )

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the safety manager including advanced risk metrics."""
        status = {
            "killswitch_enabled": self.killswitch_enabled,
            "is_killswitch_active": self.is_killswitch_active,
            "last_activation_time": self.last_killswitch_activation_time.isoformat() if self.last_killswitch_activation_time else None,
            "current_drawdown_percent": self.current_drawdown_percent,
            "peak_equity": self.peak_equity,
            "advanced_risk_enabled": ADVANCED_RISK_AVAILABLE and self.risk_monitor is not None
        }
        
        # Add advanced risk status if available
        if self.risk_monitor:
            try:
                risk_status = self.risk_monitor.get_current_risk_status()
                status["advanced_risk"] = risk_status
            except Exception as e:
                logger.error(f"Error getting advanced risk status: {e}")
                status["advanced_risk_error"] = str(e)
        
        return status
    
    def stop_monitoring(self):
        """Stop all risk monitoring components"""
        try:
            if self.risk_monitor:
                self.risk_monitor.stop_monitoring()
                logger.info("Advanced risk monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping risk monitoring: {e}")