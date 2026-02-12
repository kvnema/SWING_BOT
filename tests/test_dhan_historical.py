import os
import sys
sys.path.insert(0, os.getcwd())

from unittest.mock import patch, MagicMock
from src.data_fetch import fetch_single_instrument


def test_fetch_single_instrument_dhan_historical(monkeypatch):
    # Mock DhanClient.get_historical
    fake_resp = {'data': {'candles': [['2026-02-10', 100, 110, 95, 105, 1000]]}}

    class FakeClient:
        def get_historical(self, symbol, start, end, interval='1d'):
            return fake_resp

    with patch('src.dhan_api.DhanClient', return_value=FakeClient()):
        symbol, df, status = fetch_single_instrument('INFY.NS', days=1, headers={'x': 'y'}, broker='dhan')
        assert status == 'SUCCESS'
        assert df is not None
        assert df.iloc[0]['Open'] == 100 or 'Open' in df.columns
