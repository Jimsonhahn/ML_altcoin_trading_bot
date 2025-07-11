# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger Setup - Comprehensive Logging System
==========================================

Provides structured logging with:
- Multiple output handlers (console, file)
- Log rotation
- Structured formatting
- Performance logging
- Trade logging
- Error tracking
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Optional, Dict, Any
import traceback

# Create logs directory if it doesn't exist
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        formatted = super().format(record)

        # Reset level name
        record.levelname = levelname

        return formatted


class TradeLogger:
    """Specialized logger for trade execution"""

    def __init__(self, log_file: str = 'trades.log'):
        self.log_file = os.path.join(LOG_DIR, log_file)
        self.logger = logging.getLogger('trades')
        self.logger.setLevel(logging.INFO)

        # File handler with rotation
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )

        # JSON formatter for structured logs
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'trade',
            **trade_data
        }
        self.logger.info(json.dumps(log_entry))

    def log_order(self, order_data: Dict[str, Any]):
        """Log order placement"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'order',
            **order_data
        }
        self.logger.info(json.dumps(log_entry))

    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'signal',
            **signal_data
        }
        self.logger.info(json.dumps(log_entry))


class PerformanceLogger:
    """Specialized logger for performance metrics"""

    def __init__(self, log_file: str = 'performance.log'):
        self.log_file = os.path.join(LOG_DIR, log_file)
        self.logger = logging.getLogger('performance')
        self.logger.setLevel(logging.INFO)

        # Daily rotation
        handler = TimedRotatingFileHandler(
            self.log_file,
            when='midnight',
            interval=1,
            backupCount=30
        )

        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'metrics',
            **metrics
        }
        self.logger.info(json.dumps(log_entry))

    def log_portfolio(self, portfolio_data: Dict[str, Any]):
        """Log portfolio state"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'portfolio',
            **portfolio_data
        }
        self.logger.info(json.dumps(log_entry))


def setup_logger(name: str = None, level: str = 'INFO',
                 log_file: Optional[str] = None,
                 console: bool = True,
                 file_logging: bool = True) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name (if None, returns root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file name
        console: Enable console logging
        file_logging: Enable file logging

    Returns:
        Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name) if name else logging.getLogger()

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        # Use colored formatter for console
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if file_logging:
        if not log_file:
            # Default log file based on logger name or main.log
            log_file = f"{name}.log" if name else "main.log"

        file_path = os.path.join(LOG_DIR, log_file)

        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(numeric_level)

        # Detailed format for file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def setup_error_logger(log_file: str = 'errors.log') -> logging.Logger:
    """Setup specialized error logger"""
    logger = logging.getLogger('errors')
    logger.setLevel(logging.ERROR)

    # File handler
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=10
    )

    # Detailed error format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d\n'
        '%(message)s\n'
        'Exception: %(exc_info)s\n'
        '-' * 80,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_exception(logger: logging.Logger, exception: Exception,
                  context: Optional[Dict[str, Any]] = None):
    """Log exception with full traceback and context"""
    error_data = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }

    logger.error(f"Exception occurred: {exception}", extra=error_data, exc_info=True)


def setup_all_loggers(config: Optional[Dict[str, Any]] = None):
    """Setup all loggers based on configuration"""
    if not config:
        config = {
            'level': 'INFO',
            'console': True,
            'file_logging': True
        }

    # Setup main logger
    main_logger = setup_logger(
        name=None,
        level=config.get('level', 'INFO'),
        console=config.get('console', True),
        file_logging=config.get('file_logging', True)
    )

    # Setup specialized loggers
    trade_logger = TradeLogger()
    performance_logger = PerformanceLogger()
    error_logger = setup_error_logger()

    # Setup module-specific loggers
    modules = [
        'trading_bot',
        'strategies',
        'core.exchange',
        'core.risk_manager',
        'core.order_manager',
        'core.data_collector',
        'analysis.performance_tracker',
        'data_sources'
    ]

    for module in modules:
        setup_logger(
            name=module,
            level=config.get('level', 'INFO'),
            log_file=f"{module.replace('.', '_')}.log",
            console=False,  # Only file logging for modules
            file_logging=True
        )

    main_logger.info("Logging system initialized")

    return {
        'main': main_logger,
        'trades': trade_logger,
        'performance': performance_logger,
        'errors': error_logger
    }


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)


def cleanup_old_logs(days: int = 30):
    """Remove log files older than specified days"""
    import glob
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=days)

    for log_file in glob.glob(os.path.join(LOG_DIR, '*.log*')):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            if file_time < cutoff_date:
                os.remove(log_file)
                print(f"Removed old log file: {log_file}")
        except Exception as e:
            print(f"Error removing log file {log_file}: {e}")


# Initialize default loggers when module is imported
if __name__ != "__main__":
    # Setup basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Example usage
if __name__ == "__main__":
    # Setup all loggers
    loggers = setup_all_loggers()

    # Test main logger
    logger = loggers['main']
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test trade logger
    trade_logger = loggers['trades']
    trade_logger.log_trade({
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000,
        'amount': 0.001,
        'strategy': 'momentum'
    })

    # Test performance logger
    perf_logger = loggers['performance']
    perf_logger.log_metrics({
        'total_value': 10500,
        'daily_pnl': 50,
        'win_rate': 0.65
    })

    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        log_exception(loggers['errors'], e, {'operation': 'division'})
