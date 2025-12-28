#!/usr/bin/env python3
"""
SWING_BOT ICICI Direct API Client
Handles authentication, orders, and market data for ICICI Direct.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/icici_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ICICIDirectAPI:
    """ICICI Direct API client with OAuth 2.0 authentication."""

    BASE_URL = "https://api.icicidirect.com/api/v2"

    def __init__(self, env_file: str = '.env'):
        self.env_file = Path(env_file)
        self.token_file = Path('data/icici_token_status.json')
        self.load_credentials()

        # OAuth endpoints
        self.auth_url = f"{self.BASE_URL}/oauth/authorize"
        self.token_url = f"{self.BASE_URL}/oauth/token"

        # API endpoints
        self.orders_url = f"{self.BASE_URL}/orders"
        self.portfolio_url = f"{self.BASE_URL}/portfolio"
        self.market_data_url = f"{self.BASE_URL}/marketdata"

    def load_credentials(self):
        """Load API credentials from environment."""
        load_dotenv(self.env_file)
        self.api_key = os.getenv('ICICI_API_KEY')
        self.api_secret = os.getenv('ICICI_API_SECRET')
        self.access_token = os.getenv('ICICI_ACCESS_TOKEN')
        self.refresh_token = os.getenv('ICICI_REFRESH_TOKEN')

        if not all([self.api_key, self.api_secret]):
            raise ValueError("Missing ICICI_API_KEY or ICICI_API_SECRET in .env")

    def get_authorization_url(self) -> str:
        """Generate OAuth authorization URL."""
        params = {
            'client_id': self.api_key,
            'response_type': 'code',
            'redirect_uri': 'http://127.0.0.1',
            'scope': 'orders portfolio marketdata',
            'state': 'icici_auth'
        }
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{query_string}"

    def exchange_code_for_tokens(self, auth_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""
        data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': 'http://127.0.0.1',
            'client_id': self.api_key,
            'client_secret': self.api_secret
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        response = requests.post(self.token_url, data=data, headers=headers, timeout=30)

        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            self.save_tokens(token_data)
            logger.info("Successfully obtained ICICI Direct tokens")
            return token_data
        else:
            logger.error(f"Token exchange failed: {response.text}")
            raise Exception(f"Token exchange failed: {response.status_code}")

    def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.api_key,
            'client_secret': self.api_secret
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        try:
            response = requests.post(self.token_url, data=data, headers=headers, timeout=30)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')  # May be updated
                self.save_tokens(token_data)
                logger.info("Successfully refreshed ICICI Direct access token")
                return True
            else:
                logger.error(f"Token refresh failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def save_tokens(self, token_data: Dict[str, Any]):
        """Save tokens to .env file."""
        try:
            # Update .env
            set_key(self.env_file, 'ICICI_ACCESS_TOKEN', token_data.get('access_token', ''))
            set_key(self.env_file, 'ICICI_REFRESH_TOKEN', token_data.get('refresh_token', ''))

            # Save token status
            status = {
                'access_token': token_data.get('access_token'),
                'refresh_token': token_data.get('refresh_token'),
                'expires_at': token_data.get('expires_at'),
                'updated_at': datetime.now().isoformat()
            }

            with open(self.token_file, 'w') as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def is_token_expired(self, token: str) -> bool:
        """Check if access token is expired."""
        try:
            # Decode JWT payload (ICICI uses JWT tokens)
            payload = token.split('.')[1]
            decoded = json.loads(base64.urlsafe_b64decode(payload + '==').decode())
            exp = decoded.get('exp')
            if exp:
                return datetime.fromtimestamp(exp) < datetime.now()
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return True
        return False

    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def place_gtt_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a GTT order."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.orders_url}/gtt"
        response = requests.post(url, json=order_data, headers=self.get_headers(), timeout=30)

        if response.status_code == 201:
            return response.json()
        else:
            logger.error(f"GTT order placement failed: {response.text}")
            raise Exception(f"GTT order failed: {response.status_code} - {response.text}")

    def get_gtt_orders(self) -> List[Dict[str, Any]]:
        """Get all GTT orders."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.orders_url}/gtt"
        response = requests.get(url, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get GTT orders: {response.text}")
            return []

    def modify_gtt_order(self, order_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing GTT order."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.orders_url}/gtt/{order_id}"
        response = requests.put(url, json=order_data, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"GTT order modification failed: {response.text}")
            raise Exception(f"GTT modify failed: {response.status_code} - {response.text}")

    def cancel_gtt_order(self, order_id: str) -> bool:
        """Cancel a GTT order."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.orders_url}/gtt/{order_id}"
        response = requests.delete(url, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            return True
        else:
            logger.error(f"GTT order cancellation failed: {response.text}")
            return False

    def get_live_quote(self, symbol: str) -> Dict[str, Any]:
        """Get live market quote for a symbol."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.market_data_url}/quote"
        params = {'symbol': symbol}
        response = requests.get(url, params=params, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Live quote fetch failed: {response.text}")
            raise Exception(f"Quote fetch failed: {response.status_code}")

    def get_multiple_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get live quotes for multiple symbols."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.market_data_url}/quotes"
        data = {'symbols': symbols}
        response = requests.post(url, json=data, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Multiple quotes fetch failed: {response.text}")
            return []

    def get_historical_data(self, instrument_key: str, start_date: str, end_date: str, interval: str = '1day') -> List[Dict[str, Any]]:
        """Get historical market data for a symbol."""
        if self.is_token_expired(self.access_token):
            if not self.refresh_access_token():
                raise Exception("Token refresh failed")

        url = f"{self.market_data_url}/historical"
        params = {
            'symbol': instrument_key,
            'from_date': start_date,
            'to_date': end_date,
            'interval': interval
        }
        response = requests.get(url, params=params, headers=self.get_headers(), timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            logger.error(f"Historical data fetch failed: {response.text}")
            return []

    def get_live_quotes(self, instrument_tokens: List[str]) -> List[Dict[str, Any]]:
        """Get live quotes for multiple instrument tokens."""
        # Convert instrument tokens to symbols if needed
        # For now, assume instrument_tokens are actually symbols
        return self.get_multiple_quotes(instrument_tokens)