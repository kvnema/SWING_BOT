import pandas as pd
from unittest.mock import MagicMock

from src.ltp_reconcile import fetch_live_quotes


def test_fetch_live_quotes_uses_ws_snapshot(monkeypatch):
    # Build fake WS client
    fake_ws = MagicMock()
    fake_ws.is_connected = True
    # Provide snapshot for one instrument
    fake_ws.get_snapshot.return_value = {
        'NSE_EQ:INFY': {
            'symbol': 'INFY',
            'last_price': 1500.5,
            'timestamp': '2026-02-03T09:30:00Z',
            'ohlc': {'open': 1490, 'high': 1510, 'low': 1485, 'close': 1500}
        }
    }

    # Monkeypatch get_upstox_ws to return our fake client
    monkeypatch.setattr('src.upstox_ws.get_upstox_ws', lambda token: fake_ws)

    df = fetch_live_quotes(['NSE_EQ:INFY'], 'dummy_token', broker='upstox')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.iloc[0]['instrument_token'] == 'NSE_EQ:INFY'
    assert df.iloc[0]['last_price'] == 1500.5


def test_fetch_live_quotes_falls_back_to_rest(monkeypatch):
    # Force WS to be disabled
    monkeypatch.setattr('src.upstox_ws.get_upstox_ws', lambda token: None)

    # Monkeypatch requests.get used in _fetch_upstox_quotes
    import requests
    class FakeResp:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': {
                'NSE_EQ:INFY': {'symbol': 'INFY', 'last_price': 1300.0, 'timestamp': '2026-02-03T09:30:00Z', 'ohlc': {}}
            }}

    monkeypatch.setattr('requests.get', lambda *a, **k: FakeResp())

    df = fetch_live_quotes(['NSE_EQ:INFY'], 'dummy_token', broker='upstox')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.iloc[0]['last_price'] == 1300.0
