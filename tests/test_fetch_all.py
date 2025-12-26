"""
Tests for AllFetch functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_fetch import fetch_all_timeframes


class TestFetchAll:
    
    @patch('src.data_fetch.requests.get')
    @patch('src.data_fetch.os.getenv')
    def test_fetch_all_timeframes_success(self, mock_getenv, mock_get):
        """Test successful AllFetch for multiple timeframes."""
        
        # Mock environment
        mock_getenv.return_value = "test_token"
        
        # Mock API responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'candles': [
                    ['2024-01-15T09:15:00+05:30', 100.0, 105.0, 95.0, 102.0, 100000, 0],
                    ['2024-01-15T09:30:00+05:30', 102.0, 107.0, 97.0, 104.0, 110000, 0]
                ]
            }
        }
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "multi_tf"
            
            symbols = ['RELIANCE.NS', 'TCS.NS']
            timeframes = ['1m', '1d']
            start_date = '2024-01-01'
            end_date = '2024-01-15'
            
            results = fetch_all_timeframes(symbols, timeframes, start_date, end_date, str(out_dir))
            
            # Check results
            assert len(results) == 2
            assert '1m' in results
            assert '1d' in results
            
            # Check files were created
            assert (out_dir / 'nifty50_1m.csv').exists()
            assert (out_dir / 'nifty50_1d.csv').exists()
            
            # Check file contents
            df_1m = pd.read_csv(out_dir / 'nifty50_1m.csv')
            assert len(df_1m) > 0
            assert 'Symbol' in df_1m.columns
            assert 'timestamp' in df_1m.columns
            assert set(df_1m['Symbol'].unique()) == set(symbols)
    
    @patch('src.data_fetch.requests.get')
    @patch('src.data_fetch.os.getenv')
    def test_fetch_all_timeframes_api_error(self, mock_getenv, mock_get):
        """Test AllFetch with API errors."""
        
        mock_getenv.return_value = "test_token"
        
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 429  # Rate limit
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "multi_tf"
            
            symbols = ['RELIANCE.NS']
            timeframes = ['1d']
            
            results = fetch_all_timeframes(symbols, timeframes, '2024-01-01', '2024-01-15', str(out_dir))
            
            # Should still create empty files or handle gracefully
            assert isinstance(results, dict)
    
    @patch('src.data_fetch.os.getenv')
    def test_fetch_all_timeframes_no_token(self, mock_getenv):
        """Test AllFetch without access token."""
        
        mock_getenv.return_value = None
        
        with pytest.raises(ValueError) as exc_info:
            fetch_all_timeframes(['TEST.NS'], ['1d'], '2024-01-01', '2024-01-15', 'output')
        
        assert "UPSTOX_ACCESS_TOKEN not set" in str(exc_info.value)
    
    @patch('src.data_fetch.requests.get')
    @patch('src.data_fetch.os.getenv')
    def test_fetch_all_timeframes_empty_response(self, mock_getenv, mock_get):
        """Test AllFetch with empty API response."""
        
        mock_getenv.return_value = "test_token"
        
        # Mock empty response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': {'candles': []}}
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "multi_tf"
            
            symbols = ['RELIANCE.NS']
            timeframes = ['1d']
            
            results = fetch_all_timeframes(symbols, timeframes, '2024-01-01', '2024-01-15', str(out_dir))
            
            # Should handle empty data gracefully
            assert '1d' in results
    
    def test_timeframe_mapping(self):
        """Test timeframe to Upstox interval mapping."""
        
        from src.data_fetch import fetch_all_timeframes
        
        # Test the mapping logic indirectly through parameters
        with patch('src.data_fetch.requests.get') as mock_get, \
             patch('src.data_fetch.os.getenv') as mock_getenv:
            
            mock_getenv.return_value = "test_token"
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': {'candles': []}}
            mock_get.return_value = mock_response
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Test different timeframes
                timeframes = ['1m', '15m', '1h', '4h', '1d', '1w', '1mo']
                results = fetch_all_timeframes(['TEST.NS'], timeframes, '2024-01-01', '2024-01-15', tmpdir)
                
                # Should handle all timeframes
                assert len(results) == len(timeframes)
                for tf in timeframes:
                    assert tf in results
    
    @patch('src.data_fetch.requests.get')
    @patch('src.data_fetch.os.getenv')
    def test_fetch_all_timeframes_resampling(self, mock_getenv, mock_get):
        """Test that 4h, 1w, 1mo timeframes are properly resampled."""
        
        mock_getenv.return_value = "test_token"
        
        # Mock daily data that will be resampled
        candles = []
        base_date = pd.Timestamp('2024-01-01')
        for i in range(30):  # 30 days of data
            date = base_date + pd.Timedelta(days=i)
            candles.append([
                date.isoformat(), 100.0, 105.0, 95.0, 102.0, 100000, 0
            ])
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': {'candles': candles}}
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "multi_tf"
            
            symbols = ['RELIANCE.NS']
            
            # Test weekly resampling
            results = fetch_all_timeframes(symbols, ['1w'], '2024-01-01', '2024-01-30', str(out_dir))
            
            assert '1w' in results
            
            df_weekly = pd.read_csv(out_dir / 'nifty50_1w.csv')
            assert len(df_weekly) > 0
            assert 'Symbol' in df_weekly.columns
            
            # Weekly data should have fewer records than daily
            assert len(df_weekly) < 30
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        
        with patch('src.data_fetch.requests.get') as mock_get, \
             patch('src.data_fetch.os.getenv') as mock_getenv:
            
            mock_getenv.return_value = "test_token"
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': {'candles': []}}
            mock_get.return_value = mock_response
            
            with tempfile.TemporaryDirectory() as tmpdir:
                out_dir = Path(tmpdir) / "new_multi_tf_dir"
                assert not out_dir.exists()
                
                fetch_all_timeframes(['TEST.NS'], ['1d'], '2024-01-01', '2024-01-15', str(out_dir))
                
                assert out_dir.exists()
                assert (out_dir / 'nifty50_1d.csv').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])