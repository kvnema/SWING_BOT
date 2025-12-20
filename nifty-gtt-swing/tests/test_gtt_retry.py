import pytest
import requests
from types import SimpleNamespace

from src.upstox_gtt import place_gtt_with_retries


class DummySession:
    def __init__(self, responses):
        self._responses = responses

    def post(self, url, json, headers=None):
        resp = self._responses.pop(0)
        return SimpleNamespace(status_code=resp['status_code'], content=resp.get('content', b'{}'), json=lambda: resp.get('body', {}))


def test_retry_on_429_then_success():
    responses = [
        {'status_code': 429, 'body': {'error': 'rate limit'}},
        {'status_code': 200, 'body': {'gtt_id': 'GTT123'}}
    ]
    sess = DummySession(responses)
    payload = {'instrument_token': 'X', 'quantity': 1}
    res = place_gtt_with_retries('DUMMY', payload, dry_run=False, retries=3, backoff=0.01, log_path='outputs/logs/test_gtt_retry.log', session=sess)
    assert res['status_code'] == 200
    assert res['body'].get('gtt_id') == 'GTT123'
