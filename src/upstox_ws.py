"""Upstox WebSocket client (lightweight wrapper)

Provides a background WebSocket that maintains a thread-safe cache of latest
quotes and exposes a simple get_snapshot(instrument_tokens) API for callers.

This implementation is resilient if the broker WebSocket is unavailable - it
fails gracefully and returns None so callers can fallback to REST polling.
"""
import os
import threading
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websocket
except Exception:
    websocket = None

_WS_SINGLETON = None
_WS_LOCK = threading.Lock()


class UpstoxWebSocket:
    def __init__(self, access_token: str, ws_url: Optional[str] = None):
        self.access_token = access_token
        self.ws_url = ws_url or os.getenv('UPSTOX_WS_URL', 'wss://stream.upstox.com/stream')
        self.is_connected = False
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread = None

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Expecting messages in a dict mapping instrument_key -> quote
            # For safety, handle different message shapes
            if isinstance(data, dict) and 'data' in data:
                payload = data['data']
            else:
                payload = data

            with self._lock:
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        self._cache[k] = v
        except Exception as e:
            logger.debug(f"WebSocket on_message parse error: {e}")

    def _on_open(self, ws):
        logger.info("Upstox WS connected")
        self.is_connected = True

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("Upstox WS disconnected")
        self.is_connected = False

    def _on_error(self, ws, error):
        logger.error(f"Upstox WS error: {error}")

    def start(self):
        if websocket is None:
            logger.warning("websocket-client not installed; WS disabled")
            return

        if self._ws is not None:
            return

        headers = [f"Authorization: Bearer {self.access_token}"]

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_message=self._on_message,
            on_open=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
        )

        def run():
            try:
                self._ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run_forever error: {e}")
            finally:
                self.is_connected = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self.is_connected = False

    def subscribe(self, instrument_tokens: List[str]):
        # Send subscription message if ws is available
        if not self._ws or not self.is_connected:
            return

        msg = json.dumps({"action": "subscribe", "instruments": instrument_tokens})
        try:
            self._ws.send(msg)
        except Exception as e:
            logger.debug(f"Failed to send subscribe: {e}")

    def get_snapshot(self, instrument_tokens: List[str]) -> Dict[str, Dict]:
        """Return the latest quotes for the given instrument tokens from cache."""
        with self._lock:
            out = {t: self._cache.get(t) for t in instrument_tokens if t in self._cache}
        return out


def get_upstox_ws(access_token: str) -> Optional[UpstoxWebSocket]:
    """Return a singleton UpstoxWebSocket instance if WS is enabled.

    The presence of the environment variable UPSTOX_WS_ENABLED=true enables
    WebSocket attempts; otherwise returns None so callers know to fall back to REST.
    """
    enabled = os.getenv('UPSTOX_WS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
    if not enabled:
        return None

    global _WS_SINGLETON
    with _WS_LOCK:
        if _WS_SINGLETON is None:
            _WS_SINGLETON = UpstoxWebSocket(access_token)
            try:
                _WS_SINGLETON.start()
            except Exception as e:
                logger.warning(f"Failed to start Upstox WS: {e}")
                return None
        return _WS_SINGLETON
