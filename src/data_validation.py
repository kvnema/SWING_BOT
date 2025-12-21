import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pytz
import logging
from dataclasses import dataclass


class ValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


@dataclass
class DataMetadata:
    """Metadata structure for data validation."""
    file_path: str
    latest_date: pd.Timestamp
    earliest_date: pd.Timestamp
    symbols_present: set
    rows_count: int
    trading_days_count: int
    is_sorted_by_symbol_date: bool


def check_file_exists(path: str) -> None:
    """
    Check if file exists, raise ValidationError if missing.
    """
    if not Path(path).exists():
        raise ValidationError(f"File not found: {Path(path).absolute()}")


def load_metadata(csv_or_parquet_path: str) -> DataMetadata:
    """
    Load metadata from CSV or Parquet file.
    """
    path = Path(csv_or_parquet_path)
    check_file_exists(str(path))

    try:
        if path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if df.empty:
            raise ValidationError(f"Empty file: {path.absolute()}")

        # Ensure Date column exists and is datetime
        if 'Date' not in df.columns:
            raise ValidationError(f"Missing 'Date' column in {path.absolute()}")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        if df.empty:
            raise ValidationError(f"No valid dates in {path.absolute()}")

        # Ensure Symbol column exists
        if 'Symbol' not in df.columns:
            raise ValidationError(f"Missing 'Symbol' column in {path.absolute()}")

        # Compute metadata
        latest_date = df['Date'].max()
        earliest_date = df['Date'].min()
        symbols_present = set(df['Symbol'].dropna().unique())
        rows_count = len(df)

        # Trading days count: unique dates
        trading_days_count = df['Date'].nunique()

        # Check if sorted by Symbol then Date
        is_sorted_by_symbol_date = (
            df.sort_values(['Symbol', 'Date']).equals(df) or
            df.sort_values(['Date', 'Symbol']).equals(df)
        )

        return DataMetadata(
            file_path=str(path.absolute()),
            latest_date=latest_date,
            earliest_date=earliest_date,
            symbols_present=symbols_present,
            rows_count=rows_count,
            trading_days_count=trading_days_count,
            is_sorted_by_symbol_date=is_sorted_by_symbol_date
        )

    except Exception as e:
        raise ValidationError(f"Failed to load metadata from {path.absolute()}: {e}")


def validate_recency(meta: DataMetadata, today_ist: datetime, max_age_days: int = 1) -> None:
    """
    Validate that latest_date is within max_age_days of today_ist.
    """
    age_days = (today_ist.date() - meta.latest_date.date()).days
    if age_days > max_age_days:
        raise ValidationError(
            f"Stale data: latest_date={meta.latest_date.date()} is {age_days} days old "
            f"(max_age_days={max_age_days}, today_ist={today_ist.date()})"
        )


def validate_window(meta: DataMetadata, required_days: int = 500) -> None:
    """
    Validate that trading_days_count >= required_days.
    """
    if meta.trading_days_count < required_days:
        raise ValidationError(
            f"Insufficient coverage: trading_days_count={meta.trading_days_count} "
            f"(< required_days={required_days}). "
            f"earliest_date={meta.earliest_date.date()}, latest_date={meta.latest_date.date()}"
        )


def validate_symbols(meta: DataMetadata, expected_count: int = 50) -> None:
    """
    Validate that symbols_present count == expected_count.
    """
    actual_count = len(meta.symbols_present)
    if actual_count != expected_count:
        raise ValidationError(
            f"Symbol count mismatch: found {actual_count} symbols "
            f"(expected {expected_count}). Symbols: {sorted(meta.symbols_present)}"
        )


def validate_sorted(meta: DataMetadata) -> None:
    """
    Validate that data is sorted by symbol and date.
    """
    if not meta.is_sorted_by_symbol_date:
        raise ValidationError(
            f"Data not sorted by symbol and date in {meta.file_path}"
        )


def validate_cross_file_dates(main_meta: DataMetadata, derived_files_meta: List[DataMetadata]) -> None:
    """
    Validate that derived files have the same latest_date as main file.
    """
    main_date = main_meta.latest_date.date()
    for derived_meta in derived_files_meta:
        derived_date = derived_meta.latest_date.date()
        if derived_date != main_date:
            raise ValidationError(
                f"Inconsistent outputs: {Path(derived_meta.file_path).name} "
                f"latest={derived_date} vs indicators latest={main_date}"
            )


def compute_trading_calendar_gap(meta: DataMetadata) -> Dict:
    """
    Compute gaps in trading calendar (optional).
    """
    dates = pd.date_range(meta.earliest_date, meta.latest_date, freq='D')
    # Simple gap detection: count missing weekdays
    weekdays = dates[dates.weekday < 5]  # Mon-Fri
    expected_trading_days = len(weekdays)
    actual_trading_days = meta.trading_days_count

    return {
        'expected_trading_days': expected_trading_days,
        'actual_trading_days': actual_trading_days,
        'gap_days': expected_trading_days - actual_trading_days
    }


def summarize(meta: DataMetadata) -> str:
    """
    Return a single-line summary string.
    """
    return (
        f"{Path(meta.file_path).name}: "
        f"earliest={meta.earliest_date.date()}, "
        f"latest={meta.latest_date.date()}, "
        f"days={meta.trading_days_count}, "
        f"symbols={len(meta.symbols_present)}, "
        f"rows={meta.rows_count}"
    )


def load_excel_metadata(excel_path: str) -> DataMetadata:
    """
    Load metadata from Excel file (for final plan).
    """
    path = Path(excel_path)
    check_file_exists(str(path))

    try:
        # Read the plan sheet
        df = pd.read_excel(path, sheet_name='Plan', engine='openpyxl')

        if df.empty:
            raise ValidationError(f"Empty Excel plan sheet in {path.absolute()}")

        # Look for Date column or Generated_At_IST
        date_col = None
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'Generated_At_IST' in df.columns:
            date_col = 'Generated_At_IST'

        if date_col is None:
            raise ValidationError(f"No date column found in Excel {path.absolute()}")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        if df.empty:
            raise ValidationError(f"No valid dates in Excel {path.absolute()}")

        # For Excel, we don't have Symbol column, so use a dummy
        latest_date = df[date_col].max()
        earliest_date = df[date_col].min()
        symbols_present = set()  # Not applicable
        rows_count = len(df)
        trading_days_count = df[date_col].nunique()
        is_sorted_by_symbol_date = True  # Assume sorted

        return DataMetadata(
            file_path=str(path.absolute()),
            latest_date=latest_date,
            earliest_date=earliest_date,
            symbols_present=symbols_present,
            rows_count=rows_count,
            trading_days_count=trading_days_count,
            is_sorted_by_symbol_date=is_sorted_by_symbol_date
        )

    except Exception as e:
        raise ValidationError(f"Failed to load Excel metadata from {path.absolute()}: {e}")


def get_today_ist() -> datetime:
    """
    Get current datetime in IST timezone.
    """
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)