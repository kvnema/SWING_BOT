import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import pytz

from src.data_validation import (
    check_file_exists, load_metadata, validate_recency, validate_window,
    validate_symbols, validate_sorted, validate_cross_file_dates,
    load_excel_metadata, get_today_ist, ValidationError, summarize,
    DataMetadata
)


class TestDataValidation:
    """Test data validation functions."""

    def test_check_file_exists_success(self):
        """Test file existence check with existing file."""
        with tempfile.NamedTemporaryFile() as tmp:
            check_file_exists(tmp.name)  # Should not raise

    def test_check_file_exists_failure(self):
        """Test file existence check with missing file."""
        with pytest.raises(ValidationError, match="File not found"):
            check_file_exists("nonexistent_file.txt")

    def test_load_metadata_csv(self):
        """Test loading metadata from CSV."""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        symbols = ['TCS', 'INFY', 'RELIANCE'] * 33 + ['TCS']  # 100 rows
        data = {
            'Date': dates,
            'Symbol': symbols,
            'Close': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            meta = load_metadata(tmp_path)
            assert meta.rows_count == 100
            assert len(meta.symbols_present) == 3
            assert meta.trading_days_count == 100  # All dates are unique
            assert meta.latest_date == dates.max()
            assert meta.earliest_date == dates.min()
        finally:
            Path(tmp_path).unlink()

    def test_validate_recency_success(self):
        """Test recency validation with fresh data."""
        today = get_today_ist()
        latest_date = today - timedelta(days=0)  # Today
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=latest_date,
            earliest_date=latest_date - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )
        validate_recency(meta, today, max_age_days=1)  # Should not raise

    def test_validate_recency_failure(self):
        """Test recency validation with stale data."""
        today = get_today_ist()
        latest_date = today - timedelta(days=3)  # 3 days old
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=latest_date,
            earliest_date=latest_date - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )
        with pytest.raises(ValidationError, match="Stale data.*3 days old"):
            validate_recency(meta, today, max_age_days=1)

    def test_validate_window_success(self):
        """Test window validation with sufficient days."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=600),
            symbols_present={'TCS'},
            rows_count=500,
            trading_days_count=500,
            is_sorted_by_symbol_date=True
        )
        validate_window(meta, required_days=500)  # Should not raise

    def test_validate_window_failure(self):
        """Test window validation with insufficient days."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=400),
            symbols_present={'TCS'},
            rows_count=400,
            trading_days_count=400,
            is_sorted_by_symbol_date=True
        )
        with pytest.raises(ValidationError, match="Insufficient coverage.*400.*< required_days=500"):
            validate_window(meta, required_days=500)

    def test_validate_symbols_success(self):
        """Test symbols validation with correct count."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={f'SYMBOL{i}' for i in range(50)},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )
        validate_symbols(meta, expected_count=50)  # Should not raise

    def test_validate_symbols_failure(self):
        """Test symbols validation with wrong count."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={f'SYMBOL{i}' for i in range(49)},  # 49 symbols
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )
        with pytest.raises(ValidationError, match="Symbol count mismatch.*found 49.*expected 50"):
            validate_symbols(meta, expected_count=50)

    def test_validate_sorted_success(self):
        """Test sorted validation with sorted data."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )
        validate_sorted(meta)  # Should not raise

    def test_validate_sorted_failure(self):
        """Test sorted validation with unsorted data."""
        meta = DataMetadata(
            file_path="test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=False
        )
        with pytest.raises(ValidationError, match="Data not sorted"):
            validate_sorted(meta)

    def test_validate_cross_file_dates_success(self):
        """Test cross-file date validation with matching dates."""
        main_meta = DataMetadata(
            file_path="indicators.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )

        derived_meta = DataMetadata(
            file_path="screener.csv",
            latest_date=pd.Timestamp('2024-01-01'),  # Same date
            earliest_date=pd.Timestamp('2024-01-01'),
            symbols_present={'TCS'},
            rows_count=1,
            trading_days_count=1,
            is_sorted_by_symbol_date=True
        )

        validate_cross_file_dates(main_meta, [derived_meta])  # Should not raise

    def test_validate_cross_file_dates_failure(self):
        """Test cross-file date validation with mismatched dates."""
        main_meta = DataMetadata(
            file_path="indicators.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2024-01-01') - timedelta(days=100),
            symbols_present={'TCS'},
            rows_count=100,
            trading_days_count=100,
            is_sorted_by_symbol_date=True
        )

        derived_meta = DataMetadata(
            file_path="screener.csv",
            latest_date=pd.Timestamp('2023-12-31'),  # Different date
            earliest_date=pd.Timestamp('2023-12-31'),
            symbols_present={'TCS'},
            rows_count=1,
            trading_days_count=1,
            is_sorted_by_symbol_date=True
        )

        with pytest.raises(ValidationError, match="Inconsistent outputs.*screener.csv.*2023-12-31.*vs.*2024-01-01"):
            validate_cross_file_dates(main_meta, [derived_meta])

    def test_summarize(self):
        """Test metadata summarization."""
        meta = DataMetadata(
            file_path="/path/to/test.csv",
            latest_date=pd.Timestamp('2024-01-01'),
            earliest_date=pd.Timestamp('2023-09-15'),
            symbols_present={'TCS', 'INFY'},
            rows_count=1000,
            trading_days_count=500,
            is_sorted_by_symbol_date=True
        )

        summary = summarize(meta)
        expected = "test.csv: earliest=2023-09-15, latest=2024-01-01, days=500, symbols=2, rows=1000"
        assert summary == expected

    def test_get_today_ist(self):
        """Test getting current IST datetime."""
        today_ist = get_today_ist()
        assert isinstance(today_ist, datetime)
        # Should be IST timezone
        assert today_ist.tzinfo is not None
        assert today_ist.tzinfo.zone == 'Asia/Kolkata'