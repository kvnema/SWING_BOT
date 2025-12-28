#!/usr/bin/env python3
"""
SWING_BOT ICICI Direct Breeze API Client
ICICI Direct API integration for GTT orders, market data, and portfolio management.
"""

import requests
import os
import time
import logging
import json
import hashlib
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

# ICICI Direct Breeze API Configuration
BASE_URL = 'https://api.icicidirect.com/breezeapi/api/v1'
HISTORICAL_V2_URL = 'https://breezeapi.icicidirect.com/api/v2/historicalcharts'

class ICICIAPIClient:
    """ICICI Direct Breeze API Client for trading operations."""

    def __init__(self, api_key: str, api_secret: str, session_token: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.session = requests.Session()

        # Setup logging
        self.logger = logging.getLogger('icici_api')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _generate_checksum(self, data: str, timestamp: str) -> str:
        """Generate SHA256 checksum for API requests."""
        message = f"{timestamp}{data}{self.api_secret}"
        return hashlib.sha256(message.encode()).hexdigest()

    def _get_headers(self, data: str = "") -> Dict[str, str]:
        """Generate required headers for API requests."""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        checksum = self._generate_checksum(data, timestamp)

        return {
            'Content-Type': 'application/json',
            'X-Checksum': checksum,
            'X-Timestamp': timestamp,
            'X-AppKey': self.api_key,
            'X-SessionToken': self.session_token
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                     params: Optional[Dict] = None, retries: int = 3) -> Dict:
        """Make HTTP request with retry logic and error handling."""
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(retries):
            try:
                json_data = json.dumps(data) if data else ""
                headers = self._get_headers(json_data)

                self.logger.info(f"Making {method} request to {endpoint}")

                if method.upper() == 'GET':
                    response = self.session.get(url, headers=headers, params=params, timeout=30)
                elif method.upper() == 'POST':
                    response = self.session.post(url, headers=headers, data=json_data, timeout=30)
                elif method.upper() == 'PUT':
                    response = self.session.put(url, headers=headers, data=json_data, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, headers=headers, data=json_data, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                self.logger.info(f"Response status: {response.status_code}")

                if response.status_code == 200:
                    return {'status_code': response.status_code, 'body': response.json()}
                elif response.status_code == 401:
                    # Session expired
                    self.logger.error("Session token expired")
                    return {'status_code': response.status_code, 'body': {'error': 'Session expired'}}
                elif response.status_code >= 500:
                    # Server error, retry
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {'status_code': response.status_code, 'body': {'error': 'Server error'}}
                else:
                    # Client error
                    return {'status_code': response.status_code, 'body': response.json() if response.content else {}}

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'status_code': 0, 'body': {'error': str(e)}}

        return {'status_code': 0, 'body': {'error': 'Max retries exceeded'}}

    # Customer Details
    def get_customer_details(self) -> Dict:
        """Get customer account details."""
        return self._make_request('GET', '/customerdetails')

    # Portfolio
    def get_portfolio_positions(self) -> Dict:
        """Get current portfolio positions."""
        return self._make_request('GET', '/portfoliopositions')

    def get_portfolio_holdings(self, exchange_code: str, from_date: str, to_date: str,
                              stock_code: str = "", portfolio_type: str = "") -> Dict:
        """Get portfolio holdings."""
        params = {
            'exchange_code': exchange_code,
            'from_date': from_date,
            'to_date': to_date,
            'stock_code': stock_code,
            'portfolio_type': portfolio_type
        }
        return self._make_request('GET', '/portfolioholdings', params=params)

    def get_funds(self) -> Dict:
        """Get account funds information."""
        return self._make_request('GET', '/funds')

    # Market Data
    def get_quotes(self, stock_code: str, exchange_code: str, expiry_date: str = "",
                  product_type: str = "cash", right: str = "", strike_price: str = "") -> Dict:
        """Get live market quotes."""
        params = {
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'expiry_date': expiry_date,
            'product_type': product_type,
            'right': right,
            'strike_price': strike_price
        }
        return self._make_request('GET', '/quotes', params=params)

    def get_historical_data(self, interval: str, from_date: str, to_date: str,
                           stock_code: str, exchange_code: str, product_type: str = "cash",
                           expiry_date: str = "", right: str = "", strike_price: str = "") -> Dict:
        """Get historical market data."""
        params = {
            'interval': interval,
            'from_date': from_date,
            'to_date': to_date,
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'product_type': product_type,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price
        }
        return self._make_request('GET', '/historicalcharts', params=params)

    def get_historical_data_v2(self, interval: str, from_date: str, to_date: str,
                              stock_code: str, exchange_code: str, product_type: str = "cash",
                              expiry_date: str = "", right: str = "", strike_price: str = "") -> Dict:
        """Get historical market data v2 (enhanced)."""
        params = {
            'interval': interval,
            'from_date': from_date,
            'to_date': to_date,
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'product_type': product_type,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price
        }

        # Use different URL for v2
        try:
            headers = self._get_headers()
            response = self.session.get(HISTORICAL_V2_URL, headers=headers, params=params, timeout=30)
            return {'status_code': response.status_code, 'body': response.json() if response.content else {}}
        except Exception as e:
            self.logger.error(f"Historical data v2 request failed: {e}")
            return {'status_code': 0, 'body': {'error': str(e)}}

    # Orders
    def place_order(self, stock_code: str, exchange_code: str, product: str, action: str,
                   order_type: str, quantity: str, price: str, validity: str = "day",
                   stoploss: str = "", disclosed_quantity: str = "", expiry_date: str = "",
                   right: str = "", strike_price: str = "", user_remark: str = "") -> Dict:
        """Place a regular order."""
        data = {
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'product': product,
            'action': action,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'validity': validity,
            'stoploss': stoploss,
            'disclosed_quantity': disclosed_quantity,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price,
            'user_remark': user_remark
        }
        return self._make_request('POST', '/order', data)

    def get_order_detail(self, exchange_code: str, order_id: str) -> Dict:
        """Get details of a specific order."""
        params = {
            'exchange_code': exchange_code,
            'order_id': order_id
        }
        return self._make_request('GET', '/order', params=params)

    def get_order_list(self, exchange_code: str, from_date: str, to_date: str) -> Dict:
        """Get list of orders within date range."""
        params = {
            'exchange_code': exchange_code,
            'from_date': from_date,
            'to_date': to_date
        }
        return self._make_request('GET', '/order', params=params)

    def cancel_order(self, order_id: str, exchange_code: str) -> Dict:
        """Cancel an order."""
        data = {
            'order_id': order_id,
            'exchange_code': exchange_code
        }
        return self._make_request('DELETE', '/order', data)

    def modify_order(self, order_id: str, exchange_code: str, order_type: str = "",
                    stoploss: str = "", quantity: str = "", price: str = "",
                    validity: str = "", disclosed_quantity: str = "",
                    expiry_date: str = "", right: str = "", strike_price: str = "") -> Dict:
        """Modify an existing order."""
        data = {
            'order_id': order_id,
            'exchange_code': exchange_code,
            'order_type': order_type,
            'stoploss': stoploss,
            'quantity': quantity,
            'price': price,
            'validity': validity,
            'disclosed_quantity': disclosed_quantity,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price
        }
        return self._make_request('PUT', '/order', data)

    # GTT Orders
    def place_gtt_order(self, stock_code: str, exchange_code: str, product: str, action: str,
                       quantity: str, expiry_date: str, right: str, strike_price: str,
                       gtt_type: str, fresh_order_action: str, fresh_order_price: str,
                       fresh_order_type: str, index_or_stock: str, trade_date: str,
                       order_details: List[Dict]) -> Dict:
        """Place a GTT order (single leg or three leg OCO)."""
        data = {
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'product': product,
            'action': action,
            'quantity': quantity,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price,
            'gtt_type': gtt_type,
            'fresh_order_action': fresh_order_action,
            'fresh_order_price': fresh_order_price,
            'fresh_order_type': fresh_order_type,
            'index_or_stock': index_or_stock,
            'trade_date': trade_date,
            'order_details': order_details
        }
        return self._make_request('POST', '/gttorder', data)

    def get_gtt_order_book(self, exchange_code: str, from_date: str, to_date: str) -> Dict:
        """Get GTT order book."""
        params = {
            'exchange_code': exchange_code,
            'from_date': from_date,
            'to_date': to_date
        }
        return self._make_request('GET', '/gttorder', params=params)

    def cancel_gtt_order(self, gtt_order_id: str, exchange_code: str) -> Dict:
        """Cancel a GTT order."""
        data = {
            'gtt_order_id': gtt_order_id,
            'exchange_code': exchange_code
        }
        return self._make_request('DELETE', '/gttorder', data)

    def modify_gtt_order(self, gtt_order_id: str, exchange_code: str, gtt_type: str,
                        order_details: List[Dict]) -> Dict:
        """Modify a GTT order."""
        data = {
            'gtt_order_id': gtt_order_id,
            'exchange_code': exchange_code,
            'gtt_type': gtt_type,
            'order_details': order_details
        }
        return self._make_request('PUT', '/gttorder', data)

    # Square Off
    def square_off(self, source_flag: str, stock_code: str, exchange_code: str, quantity: str,
                  price: str, action: str, order_type: str, validity: str, stoploss_price: str = "",
                  disclosed_quantity: str = "", product_type: str = "", expiry_date: str = "",
                  right: str = "", strike_price: str = "") -> Dict:
        """Square off positions."""
        data = {
            'source_flag': source_flag,
            'stock_code': stock_code,
            'exchange_code': exchange_code,
            'quantity': quantity,
            'price': price,
            'action': action,
            'order_type': order_type,
            'validity': validity,
            'stoploss_price': stoploss_price,
            'disclosed_quantity': disclosed_quantity,
            'product_type': product_type,
            'expiry_date': expiry_date,
            'right': right,
            'strike_price': strike_price
        }
        return self._make_request('POST', '/squareoff', data)

    # Margin Calculator
    def margin_calculator(self, list_of_positions: List[Dict], exchange_code: str) -> Dict:
        """Calculate margin requirements."""
        data = {
            'list_of_positions': list_of_positions,
            'exchange_code': exchange_code
        }
        return self._make_request('POST', '/margincalculator', data)

    # Utility methods for GTT
    def place_gtt_single_leg(self, stock_code: str, exchange_code: str, product: str,
                           quantity: str, expiry_date: str, right: str, strike_price: str,
                           action: str, limit_price: str, trigger_price: str,
                           index_or_stock: str = "index") -> Dict:
        """Place a single leg GTT order."""
        order_details = [{
            'action': action,
            'limit_price': limit_price,
            'trigger_price': trigger_price
        }]

        return self.place_gtt_order(
            stock_code=stock_code,
            exchange_code=exchange_code,
            product=product,
            action=action,
            quantity=quantity,
            expiry_date=expiry_date,
            right=right,
            strike_price=strike_price,
            gtt_type="single",
            fresh_order_action=action,
            fresh_order_price=limit_price,
            fresh_order_type="limit",
            index_or_stock=index_or_stock,
            trade_date=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            order_details=order_details
        )

    def place_gtt_three_leg_oco(self, stock_code: str, exchange_code: str, product: str,
                               quantity: str, expiry_date: str, right: str, strike_price: str,
                               entry_price: str, target_price: str, stoploss_price: str,
                               index_or_stock: str = "index") -> Dict:
        """Place a three leg OCO GTT order (entry + target + stoploss)."""
        order_details = [
            {
                'gtt_leg_type': 'target',
                'action': 'sell' if right.lower() == 'call' else 'buy',
                'limit_price': target_price,
                'trigger_price': target_price
            },
            {
                'gtt_leg_type': 'stoploss',
                'action': 'sell' if right.lower() == 'call' else 'buy',
                'limit_price': stoploss_price,
                'trigger_price': stoploss_price
            }
        ]

        return self.place_gtt_order(
            stock_code=stock_code,
            exchange_code=exchange_code,
            product=product,
            action='buy' if right.lower() == 'call' else 'sell',
            quantity=quantity,
            expiry_date=expiry_date,
            right=right,
            strike_price=strike_price,
            gtt_type="cover_oco",
            fresh_order_action='buy' if right.lower() == 'call' else 'sell',
            fresh_order_price=entry_price,
            fresh_order_type="limit",
            index_or_stock=index_or_stock,
            trade_date=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            order_details=order_details
        )