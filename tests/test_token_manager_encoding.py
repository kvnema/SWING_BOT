import os
import sys
sys.path.insert(0, os.getcwd())
from unittest.mock import patch, MagicMock
from src.token_manager import UpstoxTokenManager


def test_refresh_handles_udapi100069_and_no_encoding_error(tmp_path, caplog):
    env_file = tmp_path / '.env_test_encoding'
    env_file.write_text('UPSTOX_API_KEY=abc\nUPSTOX_API_SECRET=def\nUPSTOX_ACCESS_TOKEN=expired\n')

    manager = UpstoxTokenManager(env_file=str(env_file))

    # Mock requests.post to simulate UDAPI100069 error response
    resp = MagicMock()
    resp.status_code = 400
    resp.text = '{"status":"error","errors":[{"errorCode":"UDAPI100069","message":"Check your \'client_id\' and \'client_secret\'"}]}'

    with patch('requests.post', return_value=resp):
        # Provide a dummy authorization code so the exchange runs
        with patch('builtins.input', return_value='dummy_code'):
            # capture only error logs
            import logging
            caplog.set_level(logging.ERROR)
            result = manager.refresh_token_via_oauth()
            assert result is None
            # Check logs contain helpful UDAPI100069 message
            assert any('UDAPI100069' in rec.getMessage() for rec in caplog.records)
