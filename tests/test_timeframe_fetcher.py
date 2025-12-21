import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.timeframe_fetcher import resample_ohlc, compute_indicators_for_tf, latest_window


class TestTimeframeFetcher:

    @pytest.fixture
    def sample_ohlc_data(self):
        """Sample 1-minute OHLC data for testing."""
        dates = pd.date_range('2023-01-01 09:15:00', periods=100, freq='1min')
        np.random.seed(42)
        data = {
            'Symbol': ['RELIANCE'] * 100,
            'Date': dates,
            'Open': 100 + np.random.randn(100) * 2,
            'High': 102 + np.random.randn(100) * 2,
            'Low': 98 + np.random.randn(100) * 2,
            'Close': 100 + np.random.randn(100) * 2,
            'Volume': np.random.randint(1000, 10000, 100)
        }
        df = pd.DataFrame(data)
        # Ensure High >= max(Open, Close), Low <= min(Open, Close)
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        return df

    def test_resample_to_15m(self, sample_ohlc_data):
        """Test resampling 1m to 15m."""
        resampled = resample_ohlc(sample_ohlc_data, '15m')
        assert not resampled.empty
        assert len(resampled) < len(sample_ohlc_data)  # Should be fewer bars
        assert 'Open' in resampled.columns
        assert 'High' in resampled.columns
        assert 'Low' in resampled.columns
        assert 'Close' in resampled.columns
        assert 'Volume' in resampled.columns
        # Check OHLC rules
        assert (resampled['High'] >= resampled['Open']).all()
        assert (resampled['High'] >= resampled['Close']).all()
        assert (resampled['Low'] <= resampled['Open']).all()
        assert (resampled['Low'] <= resampled['Close']).all()

    def test_resample_to_1h(self, sample_ohlc_data):
        """Test resampling 1m to 1h."""
        resampled = resample_ohlc(sample_ohlc_data, '1h')
        assert not resampled.empty
        assert len(resampled) < len(sample_ohlc_data)

    def test_resample_to_4h(self, sample_ohlc_data):
        """Test resampling 1m to 4h."""
        resampled = resample_ohlc(sample_ohlc_data, '4h')
        assert not resampled.empty
        assert len(resampled) < len(sample_ohlc_data)

    def test_resample_to_1w(self, sample_ohlc_data):
        """Test resampling 1m to 1w."""
        # Need more data for weekly
        long_dates = pd.date_range('2023-01-01 09:15:00', periods=1000, freq='1min')
        long_data = sample_ohlc_data.copy()
        long_data['Date'] = long_dates[:len(long_data)]
        resampled = resample_ohlc(long_data, '1w')
        assert not resampled.empty

    def test_resample_to_1mo(self, sample_ohlc_data):
        """Test resampling 1m to 1mo."""
        long_dates = pd.date_range('2023-01-01 09:15:00', periods=2000, freq='1min')
        long_data = sample_ohlc_data.copy()
        long_data['Date'] = long_dates[:len(long_data)]
        resampled = resample_ohlc(long_data, '1mo')
        assert not resampled.empty

    def test_compute_indicators_for_tf(self, sample_ohlc_data):
        """Test indicator computation."""
        df_ind = compute_indicators_for_tf(sample_ohlc_data, '1m')
        assert 'EMA20' in df_ind.columns
        assert 'EMA50' in df_ind.columns
        assert 'EMA200' in df_ind.columns
        assert 'RSI14' in df_ind.columns
        assert 'ATR14' in df_ind.columns
        assert 'BB_Upper' in df_ind.columns
        assert 'BB_Lower' in df_ind.columns
        assert 'DonchianH20' in df_ind.columns
        assert 'RVOL20' in df_ind.columns
        assert 'KC_Upper' in df_ind.columns
        assert 'KC_Lower' in df_ind.columns

    def test_latest_window(self, sample_ohlc_data):
        """Test latest window slicing."""
        window = latest_window(sample_ohlc_data, 50)
        assert len(window) == 50
        assert window['Date'].is_monotonic_increasing

        # Test when data is smaller than window
        small_window = latest_window(sample_ohlc_data.head(20), 50)
        assert len(small_window) == 20