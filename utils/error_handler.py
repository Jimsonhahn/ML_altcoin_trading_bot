"""
Comprehensive Error Handling Framework for Trading Bot
Provides centralized error handling, logging, and recovery mechanisms
"""

import logging
import traceback
import functools
import asyncio
import time
import uuid
import re
import json
from typing import Callable, Any, Optional, Dict, Type, Union, List, Set
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

from utils.validators import ValidationError
from pydantic import ValidationError as PydanticValidationError


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    NETWORK = "network"
    EXCHANGE = "exchange"
    DATA = "data"
    TRADING = "trading"
    STRATEGY = "strategy"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"


@dataclass
class ErrorContext:
    """Context information for errors"""
    function_name: str
    module_name: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    user_message: str
    technical_details: str
    recoverable: bool
    retry_count: int = 0
    additional_data: Optional[Dict[str, Any]] = None


class TradingBotError(Exception):
    """Base exception for trading bot errors"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, recoverable: bool = True,
                 additional_data: Optional[Dict[str, Any]] = None):
        self.message = message
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.additional_data = additional_data or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)


class ValidationTradingError(TradingBotError):
    """Validation-related trading errors"""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            additional_data={"field": field, "value": value}
        )


class NetworkTradingError(TradingBotError):
    """Network-related trading errors"""
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            recoverable=True,
            additional_data={"url": url, "status_code": status_code}
        )


class ExchangeTradingError(TradingBotError):
    """Exchange-related trading errors"""
    def __init__(self, message: str, exchange: Optional[str] = None, order_id: Optional[str] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.HIGH,
            recoverable=True,
            additional_data={"exchange": exchange, "order_id": order_id}
        )


class DataTradingError(TradingBotError):
    """Data-related trading errors"""
    def __init__(self, message: str, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            additional_data={"symbol": symbol, "timeframe": timeframe}
        )


class RateLimitTradingError(TradingBotError):
    """Rate limit errors"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            additional_data={"retry_after": retry_after}
        )


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000
        self.notification_callbacks: List[Callable[[ErrorContext], None]] = []
        
        # Error mapping for common exceptions
        self.exception_mapping = {
            ValidationError: ErrorCategory.VALIDATION,
            PydanticValidationError: ErrorCategory.VALIDATION,
            ConnectionError: ErrorCategory.NETWORK,
            TimeoutError: ErrorCategory.NETWORK,
            ValueError: ErrorCategory.VALIDATION,
            KeyError: ErrorCategory.CONFIGURATION,
            FileNotFoundError: ErrorCategory.SYSTEM,
        }
    
    def add_notification_callback(self, callback: Callable[[ErrorContext], None]):
        """Add a callback for error notifications"""
        self.notification_callbacks.append(callback)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error and create error context"""
        
        # Determine error category and severity
        category = self._categorize_error(error)
        severity = self._determine_severity(error, category)
        
        # Create error context
        error_context = ErrorContext(
            function_name=context.get('function_name', 'unknown') if context else 'unknown',
            module_name=context.get('module_name', 'unknown') if context else 'unknown',
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            user_message=self._create_user_message(error, category),
            technical_details=self._create_technical_details(error),
            recoverable=self._is_recoverable(error),
            additional_data=context
        )
        
        # Log the error
        self._log_error(error_context, error)
        
        # Store in history
        self._store_error(error_context)
        
        # Send notifications
        self._send_notifications(error_context)
        
        return error_context
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize the error based on its type"""
        if isinstance(error, TradingBotError):
            return error.category
        
        error_type = type(error)
        return self.exception_mapping.get(error_type, ErrorCategory.SYSTEM)
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(error, TradingBotError):
            return error.severity
        
        # Default severity mapping by category
        severity_mapping = {
            ErrorCategory.VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.EXCHANGE: ErrorSeverity.HIGH,
            ErrorCategory.DATA: ErrorSeverity.MEDIUM,
            ErrorCategory.TRADING: ErrorSeverity.HIGH,
            ErrorCategory.STRATEGY: ErrorSeverity.MEDIUM,
            ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
            ErrorCategory.SYSTEM: ErrorSeverity.CRITICAL,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.LOW,
        }
        
        return severity_mapping.get(category, ErrorSeverity.MEDIUM)
    
    def _create_user_message(self, error: Exception, category: ErrorCategory) -> str:
        """Create user-friendly error message"""
        if isinstance(error, TradingBotError):
            return error.message
        
        user_messages = {
            ErrorCategory.VALIDATION: "Invalid input data provided",
            ErrorCategory.NETWORK: "Network connection issue occurred",
            ErrorCategory.EXCHANGE: "Exchange communication error",
            ErrorCategory.DATA: "Market data retrieval failed",
            ErrorCategory.TRADING: "Trading operation failed",
            ErrorCategory.STRATEGY: "Strategy execution error",
            ErrorCategory.CONFIGURATION: "Configuration error detected",
            ErrorCategory.SYSTEM: "System error occurred",
            ErrorCategory.AUTHENTICATION: "Authentication failed",
            ErrorCategory.RATE_LIMIT: "Rate limit exceeded, please wait",
        }
        
        base_message = user_messages.get(category, "An error occurred")
        return f"{base_message}: {str(error)}"
    
    def _create_technical_details(self, error: Exception) -> str:
        """Create technical error details"""
        return f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable"""
        if isinstance(error, TradingBotError):
            return error.recoverable
        
        # Non-recoverable error types
        non_recoverable = (
            SystemExit, KeyboardInterrupt, MemoryError,
            ImportError, SyntaxError
        )
        
        return not isinstance(error, non_recoverable)
    
    def _log_error(self, context: ErrorContext, error: Exception):
        """Log the error with appropriate level"""
        log_levels = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        
        level = log_levels.get(context.severity, logging.ERROR)
        
        self.logger.log(
            level,
            f"[{context.category.value.upper()}] {context.user_message} "
            f"in {context.module_name}.{context.function_name}"
        )
        
        # Log technical details at debug level
        self.logger.debug(f"Technical details: {context.technical_details}")
    
    def _store_error(self, context: ErrorContext):
        """Store error in history"""
        self.error_history.append(context)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _send_notifications(self, context: ErrorContext):
        """Send error notifications"""
        for callback in self.notification_callbacks:
            try:
                callback(context)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last hour)
        recent_time = datetime.now().timestamp() - 3600
        recent_errors = [
            e for e in self.error_history 
            if e.timestamp.timestamp() > recent_time
        ]
        
        return {
            "total_errors": total_errors,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recent_errors_count": len(recent_errors),
            "last_error": self.error_history[-1].timestamp if self.error_history else None
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    category: Optional[ErrorCategory] = None,
    severity: Optional[ErrorSeverity] = None,
    recoverable: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    reraise: bool = False
):
    """Decorator for automatic error handling"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    # Create context for error handling
                    context = {
                        'function_name': func.__name__,
                        'module_name': func.__module__,
                        'args': str(args) if args else None,
                        'kwargs': str(kwargs) if kwargs else None,
                        'retry_count': retry_count
                    }
                    
                    # Handle the error
                    error_context = error_handler.handle_error(e, context)
                    
                    # Check if we should retry
                    if (retry_count < max_retries and 
                        error_context.recoverable and 
                        recoverable):
                        retry_count += 1
                        time.sleep(retry_delay * retry_count)  # Exponential backoff
                        continue
                    
                    # If we shouldn't retry or max retries reached
                    if reraise:
                        raise
                    
                    # Return None or raise custom error based on configuration
                    if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        raise
                    
                    return None
                    
        # Async version
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    context = {
                        'function_name': func.__name__,
                        'module_name': func.__module__,
                        'args': str(args) if args else None,
                        'kwargs': str(kwargs) if kwargs else None,
                        'retry_count': retry_count
                    }
                    
                    error_context = error_handler.handle_error(e, context)
                    
                    if (retry_count < max_retries and 
                        error_context.recoverable and 
                        recoverable):
                        retry_count += 1
                        await asyncio.sleep(retry_delay * retry_count)
                        continue
                    
                    if reraise:
                        raise
                    
                    if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        raise
                    
                    return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[ErrorContext]]:
    """Safely execute a function and return result and error context"""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        context = {
            'function_name': func.__name__,
            'module_name': getattr(func, '__module__', 'unknown'),
            'args': str(args) if args else None,
            'kwargs': str(kwargs) if kwargs else None
        }
        error_context = error_handler.handle_error(e, context)
        return None, error_context


# Utility functions for common error scenarios
def handle_validation_error(error: Exception, field: Optional[str] = None, value: Any = None):
    """Handle validation errors specifically"""
    raise ValidationTradingError(str(error), field=field, value=value)


def handle_network_error(error: Exception, url: Optional[str] = None, status_code: Optional[int] = None):
    """Handle network errors specifically"""
    raise NetworkTradingError(str(error), url=url, status_code=status_code)


def handle_exchange_error(error: Exception, exchange: Optional[str] = None, order_id: Optional[str] = None):
    """Handle exchange errors specifically"""
    raise ExchangeTradingError(str(error), exchange=exchange, order_id=order_id)


def handle_data_error(error: Exception, symbol: Optional[str] = None, timeframe: Optional[str] = None):
    """Handle data errors specifically"""
    raise DataTradingError(str(error), symbol=symbol, timeframe=timeframe)


@dataclass
class SecureErrorResponse:
    """Structured error response with security considerations"""
    error_id: str
    timestamp: str
    category: str
    severity: str
    message: str
    status_code: int
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class SecureErrorHandler:
    """
    Enhanced error handler with security-focused features:
    - Secure logging without sensitive data
    - Unique error IDs with UUID
    - Structured error responses
    - Specialized error handling methods
    """
    
    # Sensitive data patterns to filter from logs
    SENSITIVE_PATTERNS = [
        r'(?i)(api[_-]?key|secret|token|password|auth)["\s]*[:=]["\s]*([a-zA-Z0-9_-]+)',
        r'(?i)(authorization|bearer)["\s]*[:=]["\s]*([a-zA-Z0-9_.-]+)',
        r'(?i)(private[_-]?key)["\s]*[:=]["\s]*([a-zA-Z0-9_/+=]+)',
        r'(?i)(access[_-]?token)["\s]*[:=]["\s]*([a-zA-Z0-9_.-]+)',
        r'["\']sk_[a-zA-Z0-9]{24,}["\']',  # Stripe secret keys
        r'["\']pk_[a-zA-Z0-9]{24,}["\']',  # Stripe public keys
        r'[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}',  # Credit card numbers
    ]
    
    def __init__(self, logger: Optional[logging.Logger] = None, app_name: str = "trading_bot"):
        self.logger = logger or logging.getLogger(__name__)
        self.app_name = app_name
        self.error_history: List[SecureErrorResponse] = []
        self.max_history_size = 1000
        self.notification_callbacks: List[Callable[[SecureErrorResponse], None]] = []
        
        # Compile sensitive patterns for performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.SENSITIVE_PATTERNS]
    
    def add_notification_callback(self, callback: Callable[[SecureErrorResponse], None]):
        """Add a callback for error notifications"""
        self.notification_callbacks.append(callback)
    
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive data from error messages"""
        sanitized = message
        
        for pattern in self.compiled_patterns:
            # Replace sensitive values with masked version
            def replace_match(match):
                try:
                    if len(match.groups()) >= 2:
                        return f"{match.group(1)}={self._mask_sensitive_value(match.group(2))}"
                    else:
                        return "***REDACTED***"
                except IndexError:
                    return "***REDACTED***"
            
            sanitized = pattern.sub(replace_match, sanitized)
        
        return sanitized
    
    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive values while preserving some characters for debugging"""
        if len(value) <= 4:
            return "***"
        elif len(value) <= 8:
            return f"{value[:2]}***{value[-1:]}"
        else:
            return f"{value[:3]}***{value[-2:]}"
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        return str(uuid.uuid4())
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID for request correlation"""
        return str(uuid.uuid4())[:8]
    
    def _create_secure_response(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> SecureErrorResponse:
        """Create a secure error response"""
        
        error_id = self._generate_error_id()
        timestamp = datetime.now().isoformat()
        
        # Sanitize error message
        raw_message = str(error)
        sanitized_message = self._sanitize_message(raw_message)
        
        # Create secure details without sensitive information
        secure_details = {}
        if details:
            for key, value in details.items():
                if isinstance(value, str):
                    secure_details[key] = self._sanitize_message(value)
                elif key.lower() in ['password', 'secret', 'token', 'key', 'auth']:
                    secure_details[key] = "***REDACTED***"
                else:
                    secure_details[key] = value
        
        response = SecureErrorResponse(
            error_id=error_id,
            timestamp=timestamp,
            category=category.value,
            severity=severity.value,
            message=sanitized_message,
            status_code=status_code,
            details=secure_details,
            trace_id=trace_id or self._generate_trace_id()
        )
        
        return response
    
    def _log_secure_error(self, response: SecureErrorResponse, error: Exception):
        """Log error securely without sensitive information"""
        
        log_level_mapping = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        
        level = log_level_mapping.get(ErrorSeverity(response.severity), logging.ERROR)
        
        # Log with structured data (avoid 'message' key conflict)
        log_extra = {
            "error_id": response.error_id,
            "trace_id": response.trace_id,
            "error_category": response.category,
            "error_severity": response.severity,
            "error_timestamp": response.timestamp,
            "app_name": self.app_name
        }
        
        self.logger.log(level, f"[{response.category.upper()}] {response.message}", extra=log_extra)
        
        # Log stack trace at debug level (sanitized)
        if level >= logging.ERROR:
            stack_trace = traceback.format_exc()
            sanitized_trace = self._sanitize_message(stack_trace)
            self.logger.debug(f"Stack trace for {response.error_id}: {sanitized_trace}")
    
    def _store_error_response(self, response: SecureErrorResponse):
        """Store error response in history"""
        self.error_history.append(response)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _send_notifications(self, response: SecureErrorResponse):
        """Send error notifications"""
        for callback in self.notification_callbacks:
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
    
    def handle_critical_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> SecureErrorResponse:
        """
        Handle critical system errors that require immediate attention
        """
        
        # Determine error category
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            category = ErrorCategory.SYSTEM
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            category = ErrorCategory.CONFIGURATION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK
        else:
            category = ErrorCategory.SYSTEM
        
        response = self._create_secure_response(
            error=error,
            category=category,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            details=context,
            trace_id=trace_id
        )
        
        # Log critical error
        self._log_secure_error(response, error)
        
        # Store in history
        self._store_error_response(response)
        
        # Send immediate notifications for critical errors
        self._send_notifications(response)
        
        # Additional logging for critical errors
        self.logger.critical(
            f"CRITICAL ERROR DETECTED - ID: {response.error_id} | "
            f"Immediate investigation required"
        )
        
        return response
    
    def handle_trading_error(
        self,
        error: Exception,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        amount: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> SecureErrorResponse:
        """
        Handle trading-specific errors with trading context
        """
        
        # Determine severity based on error type
        if isinstance(error, ValidationTradingError):
            severity = ErrorSeverity.MEDIUM
            status_code = 400
        elif isinstance(error, ExchangeTradingError):
            severity = ErrorSeverity.HIGH
            status_code = 502
        elif isinstance(error, RateLimitTradingError):
            severity = ErrorSeverity.LOW
            status_code = 429
        else:
            severity = ErrorSeverity.HIGH
            status_code = 500
        
        # Create trading-specific details
        trading_details = {
            "symbol": symbol,
            "order_id": order_id,
            "amount": amount,
            "error_type": type(error).__name__
        }
        
        # Merge with additional context
        if context:
            trading_details.update(context)
        
        response = self._create_secure_response(
            error=error,
            category=ErrorCategory.TRADING,
            severity=severity,
            status_code=status_code,
            details=trading_details,
            trace_id=trace_id
        )
        
        # Log trading error
        self._log_secure_error(response, error)
        
        # Store in history
        self._store_error_response(response)
        
        # Send notifications for high severity trading errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_notifications(response)
        
        return response
    
    def handle_api_error(
        self,
        error: Exception,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> SecureErrorResponse:
        """
        Handle API-related errors with API context
        """
        
        # Determine category and severity
        if status_code:
            if 400 <= status_code < 500:
                severity = ErrorSeverity.MEDIUM
                category = ErrorCategory.VALIDATION if status_code == 400 else ErrorCategory.AUTHENTICATION
            elif status_code == 429:
                severity = ErrorSeverity.LOW
                category = ErrorCategory.RATE_LIMIT
            elif 500 <= status_code < 600:
                severity = ErrorSeverity.HIGH
                category = ErrorCategory.NETWORK
            else:
                severity = ErrorSeverity.MEDIUM
                category = ErrorCategory.NETWORK
        else:
            severity = ErrorSeverity.HIGH
            category = ErrorCategory.NETWORK
        
        # Create API-specific details (sanitized)
        api_details = {
            "endpoint": endpoint,
            "status_code": status_code,
            "error_type": type(error).__name__
        }
        
        # Sanitize response data
        if response_data:
            api_details["response_data"] = self._sanitize_dict(response_data)
        
        # Sanitize request data (especially important for API requests)
        if request_data:
            api_details["request_data"] = self._sanitize_dict(request_data)
        
        # Merge with additional context
        if context:
            api_details.update(context)
        
        response = self._create_secure_response(
            error=error,
            category=category,
            severity=severity,
            status_code=status_code or 500,
            details=api_details,
            trace_id=trace_id
        )
        
        # Log API error
        self._log_secure_error(response, error)
        
        # Store in history
        self._store_error_response(response)
        
        # Send notifications for high severity API errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_notifications(response)
        
        return response
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        sanitized = {}
        
        for key, value in data.items():
            # Check if key indicates sensitive data
            if key.lower() in ['password', 'secret', 'token', 'key', 'auth', 'authorization']:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_message(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_message(item) if isinstance(item, str)
                    else self._sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get secure error statistics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        
        # Count by category
        category_counts = {}
        for response in self.error_history:
            category = response.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for response in self.error_history:
            severity = response.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last hour)
        recent_time = datetime.now().timestamp() - 3600
        recent_errors = [
            r for r in self.error_history
            if datetime.fromisoformat(r.timestamp).timestamp() > recent_time
        ]
        
        return {
            "total_errors": total_errors,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recent_errors_count": len(recent_errors),
            "last_error_id": self.error_history[-1].error_id if self.error_history else None,
            "last_error_timestamp": self.error_history[-1].timestamp if self.error_history else None
        }
    
    def get_error_by_id(self, error_id: str) -> Optional[SecureErrorResponse]:
        """Retrieve specific error by ID"""
        for response in self.error_history:
            if response.error_id == error_id:
                return response
        return None
    
    def get_errors_by_trace_id(self, trace_id: str) -> List[SecureErrorResponse]:
        """Retrieve all errors with specific trace ID"""
        return [
            response for response in self.error_history
            if response.trace_id == trace_id
        ]
    
    def clear_error_history(self):
        """Clear error history (for maintenance)"""
        self.error_history.clear()
        self.logger.info("Error history cleared")


# Global secure error handler instance
secure_error_handler = SecureErrorHandler()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test basic error handling
    print("🧪 Testing Error Handling Framework")
    print("=" * 50)
    
    @handle_errors(category=ErrorCategory.TRADING, max_retries=2)
    def test_function_with_error():
        """Test function that raises an error"""
        raise ValueError("Test error for demonstration")
    
    @handle_errors(category=ErrorCategory.VALIDATION, reraise=True)
    def test_validation_function(value: float):
        """Test validation function"""
        if value < 0:
            raise ValidationTradingError("Value must be positive", field="value", value=value)
        return value * 2
    
    # Test error handling
    print("1. Testing function with retries...")
    result = test_function_with_error()
    print(f"Result: {result}")
    
    print("\n2. Testing validation error...")
    try:
        result = test_validation_function(-10)
    except ValidationTradingError as e:
        print(f"Caught validation error: {e}")
    
    print("\n3. Testing safe execution...")
    result, error_context = safe_execute(lambda: 10 / 0)
    print(f"Result: {result}, Error: {error_context.user_message if error_context else None}")
    
    print("\n4. Error statistics:")
    stats = error_handler.get_error_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Error handling framework testing completed!")
    
    # Test SecureErrorHandler
    print("\n" + "=" * 50)
    print("🔒 Testing SecureErrorHandler")
    print("=" * 50)
    
    secure_handler = SecureErrorHandler()
    
    # Test critical error handling
    print("\n1. Testing critical error handling...")
    try:
        raise SystemExit("System shutdown requested")
    except Exception as e:
        response = secure_handler.handle_critical_error(e, context={"user": "admin"})
        print(f"✅ Critical error handled - ID: {response.error_id}")
        print(f"   Severity: {response.severity}, Category: {response.category}")
    
    # Test trading error handling
    print("\n2. Testing trading error handling...")
    try:
        raise ValidationTradingError("Invalid amount", field="amount", value=-100)
    except Exception as e:
        response = secure_handler.handle_trading_error(
            e, 
            symbol="BTC/USDT", 
            order_id="12345", 
            amount=0.1,
            context={"strategy": "momentum"}
        )
        print(f"✅ Trading error handled - ID: {response.error_id}")
        print(f"   Status Code: {response.status_code}, Details: {response.details}")
    
    # Test API error handling with sensitive data
    print("\n3. Testing API error handling with sensitive data...")
    try:
        raise ConnectionError("API connection failed")
    except Exception as e:
        response = secure_handler.handle_api_error(
            e,
            endpoint="https://api.binance.com/api/v3/order",
            status_code=500,
            request_data={
                "symbol": "BTCUSDT",
                "api_key": "secret_key_12345",
                "secret": "very_secret_value"
            },
            response_data={"error": "Internal server error"}
        )
        print(f"✅ API error handled - ID: {response.error_id}")
        print(f"   Sanitized details: {response.details}")
    
    # Test secure response features
    print("\n4. Testing secure response features...")
    if secure_handler.error_history:
        latest_error = secure_handler.error_history[-1]
        print(f"✅ Latest error JSON: {latest_error.to_json()}")
        
        # Test error retrieval
        retrieved = secure_handler.get_error_by_id(latest_error.error_id)
        if retrieved:
            print(f"✅ Error retrieval by ID successful")
    
    # Test statistics
    print("\n5. Testing secure error statistics...")
    stats = secure_handler.get_error_statistics()
    print(f"✅ Secure statistics: {stats}")
    
    print("\n🔒 SecureErrorHandler testing completed!")