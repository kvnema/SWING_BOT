"""
SWING_BOT Utilities
==================

Common utilities for data handling, timezones, logging, and file operations.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.timezone('UTC')

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger('swingbot')
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_directories(*dirs: str) -> None:
    """Ensure directories exist, create if they don't."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_ist_now() -> datetime:
    """Get current datetime in IST timezone."""
    return datetime.now(IST)

def convert_to_ist(dt: datetime) -> datetime:
    """Convert datetime to IST timezone."""
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    return dt.astimezone(IST)

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    return dt.strftime(format_str)

def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d") -> datetime:
    """Parse datetime from string."""
    dt = datetime.strptime(date_str, format_str)
    return IST.localize(dt) if dt.tzinfo is None else dt

def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of trading days between two dates (excluding weekends)."""
    # Simple approximation - doesn't account for holidays
    total_days = (end_date - start_date).days
    weekends = total_days // 7 * 2
    extra_days = total_days % 7

    # Adjust for weekend days in extra days
    if start_date.weekday() <= end_date.weekday():
        weekends += 2 if extra_days >= 6 else 1 if extra_days >= 5 else 0
    else:
        weekends += 1

    return total_days - weekends + 1  # +1 to include both start and end dates

def safe_divide(a: Union[float, pd.Series, np.ndarray],
                b: Union[float, pd.Series, np.ndarray],
                default: float = 0.0) -> Union[float, pd.Series, np.ndarray]:
    """Safe division that handles division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, default)
        elif pd.api.types.is_series(a) or pd.api.types.is_series(b):
            result = result.fillna(default).replace([np.inf, -np.inf], default)
        else:
            result = default if not np.isfinite(result) else result
    return result

def round_to_decimals(value: Union[float, pd.Series, np.ndarray],
                     decimals: int = 2) -> Union[float, pd.Series, np.ndarray]:
    """Round values to specified decimal places."""
    return np.round(value, decimals)

def calculate_percentage_change(old_value: Union[float, pd.Series],
                               new_value: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Calculate percentage change between two values."""
    return safe_divide((new_value - old_value), old_value) * 100

def get_column_letter(col_num: int) -> str:
    """Convert column number to Excel column letter (1-based indexing)."""
    result = ""
    while col_num > 0:
        col_num -= 1
        result = chr(col_num % 26 + ord('A')) + result
        col_num //= 26
    return result

def validate_dataframe(df: pd.DataFrame,
                      required_columns: List[str],
                      logger: Optional[logging.Logger] = None) -> bool:
    """Validate DataFrame has required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        if logger:
            logger.error(f"Missing required columns: {missing_columns}")
        return False

    if logger:
        logger.info(f"DataFrame validation passed. Shape: {df.shape}")

    return True

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Get metadata about a file."""
    path = Path(file_path)

    if not path.exists():
        return {"exists": False}

    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime, IST),
        "created_time": datetime.fromtimestamp(stat.st_ctime, IST),
        "is_file": path.is_file(),
        "extension": path.suffix
    }

def calculate_age_days(file_path: str) -> Optional[float]:
    """Calculate age of file in days."""
    metadata = get_file_metadata(file_path)

    if not metadata.get("exists"):
        return None

    modified_time = metadata["modified_time"]
    now = get_ist_now()
    age = (now - modified_time).total_seconds() / (24 * 3600)

    return age

def read_parquet_safe(file_path: str,
                     logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Safely read parquet file with error handling."""
    try:
        df = pd.read_parquet(file_path)
        if logger:
            logger.info(f"Successfully read parquet file: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        if logger:
            logger.error(f"Failed to read parquet file {file_path}: {str(e)}")
        return None

def write_parquet_safe(df: pd.DataFrame,
                      file_path: str,
                      logger: Optional[logging.Logger] = None) -> bool:
    """Safely write parquet file with error handling."""
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(file_path, index=False)
        if logger:
            logger.info(f"Successfully wrote parquet file: {file_path}, shape: {df.shape}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to write parquet file {file_path}: {str(e)}")
        return False

def read_csv_safe(file_path: str,
                 logger: Optional[logging.Logger] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
    """Safely read CSV file with error handling."""
    try:
        df = pd.read_csv(file_path, **kwargs)
        if logger:
            logger.info(f"Successfully read CSV file: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        if logger:
            logger.error(f"Failed to read CSV file {file_path}: {str(e)}")
        return None

def write_csv_safe(df: pd.DataFrame,
                  file_path: str,
                  logger: Optional[logging.Logger] = None,
                  **kwargs) -> bool:
    """Safely write CSV file with error handling."""
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file_path, index=False, **kwargs)
        if logger:
            logger.info(f"Successfully wrote CSV file: {file_path}, shape: {df.shape}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to write CSV file {file_path}: {str(e)}")
        return False

def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default."""
    return os.getenv(var_name, default)

def validate_env_vars(required_vars: List[str]) -> List[str]:
    """Validate that required environment variables are set."""
    missing = []
    for var in required_vars:
        if not get_env_var(var):
            missing.append(var)
    return missing

# Global logger instance
logger = setup_logging(
    level=get_env_var("LOG_LEVEL", "INFO"),
    log_file=get_env_var("LOG_FILE")
)
