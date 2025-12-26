"""
SWING_BOT Standardized Logging Setup

Provides consistent logging configuration across all modules.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    module_name: str,
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup standardized logging for a module.

    Args:
        module_name: Name of the module (used in log filename)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        log_dir = Path('outputs/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f'{module_name}_{datetime.now().strftime("%Y%m%d")}.log'

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for the given module name.
    Uses default settings optimized for production.
    """
    return setup_logging(module_name)

def log_function_call(logger: logging.Logger):
    """
    Decorator to log function entry and exit.

    Usage:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def log_performance(logger: logging.Logger):
    """
    Decorator to log function performance.

    Usage:
        @log_performance(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(".2f")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(".2f")
                raise
        return wrapper
    return decorator

# Global logger for utilities
utils_logger = setup_logging('utils', log_level='INFO')

def log_system_info(logger: logging.Logger):
    """Log system information for debugging."""
    try:
        import platform
        import psutil

        logger.info("System Information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Python: {sys.version}")
        logger.info(f"  CPU Cores: {psutil.cpu_count()}")
        logger.info(".1f")
        logger.info(f"  Working Directory: {os.getcwd()}")

        # Environment variables (sensitive ones masked)
        env_vars = {}
        for key in ['ENVIRONMENT', 'PYTHONPATH', 'VIRTUAL_ENV']:
            value = os.getenv(key)
            if value:
                if 'SECRET' in key.upper() or 'KEY' in key.upper() or 'TOKEN' in key.upper():
                    env_vars[key] = '***MASKED***'
                else:
                    env_vars[key] = value

        if env_vars:
            logger.info("  Environment Variables:")
            for key, value in env_vars.items():
                logger.info(f"    {key}: {value}")

    except ImportError:
        logger.warning("psutil not available for system info logging")
    except Exception as e:
        logger.warning(f"Could not log system info: {str(e)}")

# Export common logging levels for convenience
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL