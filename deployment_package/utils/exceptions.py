"""
Custom Exceptions for the Altcoin Trading Bot
"""


class TradingBotException(Exception):
    """Base exception class for all trading bot related errors"""
    pass


class TradingBotError(TradingBotException):
    """Alias for TradingBotException for backward compatibility"""
    pass


class ValidationError(TradingBotException):
    """Raised when validation fails"""
    pass


class StrategyError(TradingBotException):
    """Raised when there's an error in strategy execution or configuration"""
    pass


class ConfigurationError(TradingBotException):
    """Raised when there's an error in bot configuration"""
    pass


class DataError(TradingBotException):
    """Raised when there's an error with market data"""
    pass


class ExchangeError(TradingBotException):
    """Raised when there's an error with exchange operations"""
    pass


class RiskManagementError(TradingBotException):
    """Raised when risk management limits are violated"""
    pass


class MLError(TradingBotException):
    """Raised when there's an error with ML components"""
    pass


class PositionError(TradingBotException):
    """Raised when there's an error with position management"""
    pass


class SafetyError(TradingBotException):
    """Raised when safety checks fail"""
    pass


class NotificationError(TradingBotException):
    """Raised when notification sending fails"""
    pass