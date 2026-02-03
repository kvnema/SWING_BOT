import os
import tempfile
import sys
# Ensure project root is on sys.path for imports
sys.path.insert(0, os.getcwd())
from src.token_manager import UpstoxTokenManager
from unittest.mock import patch, MagicMock


def test_refresh_with_refresh_token(tmp_path, monkeypatch):
    # Create a temporary .env file
    env_file = tmp_path / '.env_test'
    env_file.write_text('UPSTOX_API_KEY=abc\nUPSTOX_API_SECRET=def\nUPSTOX_REFRESH_TOKEN=old_refresh\nUPSTOX_ACCESS_TOKEN=old_token\n')

    manager = UpstoxTokenManager(env_file=str(env_file))

    # Mock requests.post to simulate refresh token response
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {'access_token': 'new_access', 'refresh_token': 'new_refresh'}

    with patch('requests.post', return_value=resp) as mock_post:
        new_token = manager.refresh_token_using_refresh_token('old_refresh')
        assert new_token == 'new_access'
        # Check .env updated
        content = env_file.read_text()
        # tokens may be written with/without quotes; assert presence
        assert 'new_access' in content
        assert 'new_refresh' in content


def test_check_and_refresh_token_falls_back_to_oauth(monkeypatch, tmp_path):
    env_file = tmp_path / '.env_test2'
    env_file.write_text('UPSTOX_API_KEY=abc\nUPSTOX_API_SECRET=def\nUPSTOX_ACCESS_TOKEN=expired\n')

    manager = UpstoxTokenManager(env_file=str(env_file))

    # Simulate refresh grant failure and user-led OAuth (we'll mock refresh flow to return None)
    with patch.object(manager, 'refresh_token_using_refresh_token', return_value=None):
        # Mock refresh_token_via_oauth to return a token
        with patch.object(manager, 'refresh_token_via_oauth', return_value='oauth_token'):
            with patch.object(manager, 'test_token_validity', return_value=True):
                # Ensure get_token_expiration returns a datetime so save_token_status doesn't fail
                from datetime import datetime
                with patch.object(manager, 'get_token_expiration', return_value=datetime.now()):
                    ok = manager.check_and_refresh_token(force_refresh=True)
                    assert ok
                    content = env_file.read_text()
                    assert 'oauth_token' in content or manager.access_token == 'oauth_token'
