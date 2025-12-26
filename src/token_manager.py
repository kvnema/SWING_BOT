#!/usr/bin/env python3
"""
SWING_BOT Token Manager
Automated Upstox API token refresh and management system.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import base64
from dotenv import load_dotenv, set_key

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/token_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UpstoxTokenManager:
    """Manages Upstox API token lifecycle and automatic refresh."""

    def __init__(self, env_file: str = '.env'):
        self.env_file = Path(env_file)
        self.token_file = Path('data/token_status.json')
        self.load_credentials()

    def load_credentials(self):
        """Load API credentials from environment."""
        load_dotenv(self.env_file)
        self.api_key = os.getenv('UPSTOX_API_KEY')
        self.api_secret = os.getenv('UPSTOX_API_SECRET')
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')

        if not all([self.api_key, self.api_secret]):
            raise ValueError("Missing UPSTOX_API_KEY or UPSTOX_API_SECRET in .env")

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token to extract expiration info."""
        try:
            header, payload, signature = token.split('.')
            payload += '=' * (4 - len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
        except Exception as e:
            logger.error(f"Failed to decode token: {e}")
            return {}

    def get_token_expiration(self, token: str) -> Optional[datetime]:
        """Get token expiration datetime."""
        decoded = self.decode_token(token)
        exp_timestamp = decoded.get('exp')
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp)
        return None

    def is_token_expired(self, token: str, buffer_minutes: int = 30) -> bool:
        """Check if token is expired or will expire soon."""
        expiration = self.get_token_expiration(token)
        if not expiration:
            return True

        buffer_time = datetime.now() + timedelta(minutes=buffer_minutes)
        return expiration <= buffer_time

    def test_token_validity(self, token: str) -> bool:
        """Test if token is valid by making a simple API call."""
        headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
        try:
            response = requests.get('https://api.upstox.com/v2/user/profile', headers=headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Token validity test failed: {e}")
            return False

    def refresh_token_via_oauth(self) -> Optional[str]:
        """
        Refresh token using OAuth flow.
        Note: This requires manual intervention for authorization code.
        """
        logger.info("Token refresh requires manual authorization code")

        # Generate authorization URL
        auth_url = (
            "https://api.upstox.com/v2/login/authorization/dialog?"
            f"client_id={self.api_key}&"
            "redirect_uri=http://127.0.0.1&"
            "response_type=code&"
            "scope=openid%20profile%20email&"
            "state=token_refresh"
        )

        print("=" * 60)
        print("🔄 UPSTOX TOKEN REFRESH REQUIRED")
        print("=" * 60)
        print("1. Open this URL in your browser:")
        print(f"   {auth_url}")
        print()
        print("2. Log in to your Upstox account")
        print("3. Grant permission for the app")
        print("4. Copy the authorization code from the redirect URL")
        print()

        auth_code = input("Enter the authorization code: ").strip()

        if not auth_code:
            logger.error("No authorization code provided")
            return None

        # Exchange code for token
        token_url = "https://api.upstox.com/v2/login/authorization/token"
        data = {
            'code': auth_code,
            'client_id': self.api_key,
            'client_secret': self.api_secret,
            'redirect_uri': 'http://127.0.0.1',
            'grant_type': 'authorization_code'
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}

        try:
            response = requests.post(token_url, data=data, headers=headers, timeout=30)
            if response.status_code == 200:
                token_data = response.json()
                new_token = token_data.get('access_token')
                if new_token:
                    logger.info("Successfully obtained new access token")
                    return new_token
                else:
                    logger.error("No access_token in response")
            else:
                logger.error(f"Token exchange failed: {response.text}")
        except Exception as e:
            logger.error(f"Token exchange error: {e}")

        return None

    def update_token_in_env(self, new_token: str):
        """Update the access token in the .env file."""
        try:
            set_key(self.env_file, 'UPSTOX_ACCESS_TOKEN', new_token)
            self.access_token = new_token
            logger.info("Updated access token in .env file")
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
            raise

    def save_token_status(self):
        """Save current token status for monitoring."""
        status = {
            'last_check': datetime.now().isoformat(),
            'token_valid': self.test_token_validity(self.access_token),
            'expiration': self.get_token_expiration(self.access_token).isoformat() if self.access_token else None,
            'is_expired': self.is_token_expired(self.access_token) if self.access_token else True
        }

        try:
            self.token_file.parent.mkdir(exist_ok=True)
            with open(self.token_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save token status: {e}")

    def load_token_status(self) -> Dict[str, Any]:
        """Load saved token status."""
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load token status: {e}")
        return {}

    def check_and_refresh_token(self, force_refresh: bool = False) -> bool:
        """
        Check token status and refresh if needed.

        Args:
            force_refresh: Force token refresh even if valid

        Returns:
            bool: True if token is valid/refreshed successfully
        """
        logger.info("Checking token status...")

        # Check if token exists
        if not self.access_token:
            logger.warning("No access token found")
            return False

        # Check if token is expired or will expire soon
        if force_refresh or self.is_token_expired(self.access_token):
            logger.info("Token expired or expiring soon, refreshing...")

            # Attempt to refresh token
            new_token = self.refresh_token_via_oauth()

            if new_token:
                # Update environment
                self.update_token_in_env(new_token)

                # Test new token
                if self.test_token_validity(new_token):
                    logger.info("✅ Token refresh successful")
                    self.save_token_status()
                    return True
                else:
                    logger.error("❌ New token is invalid")
                    return False
            else:
                logger.error("❌ Token refresh failed")
                return False
        else:
            # Token is still valid
            logger.info("✅ Token is still valid")
            self.save_token_status()
            return True

    def get_token_info(self) -> Dict[str, Any]:
        """Get comprehensive token information."""
        info = {
            'has_token': bool(self.access_token),
            'token_length': len(self.access_token) if self.access_token else 0,
            'is_expired': self.is_token_expired(self.access_token) if self.access_token else True,
            'expiration': None,
            'hours_until_expiry': None,
            'is_valid': False
        }

        if self.access_token:
            expiration = self.get_token_expiration(self.access_token)
            if expiration:
                info['expiration'] = expiration.isoformat()
                now = datetime.now()
                if expiration > now:
                    info['hours_until_expiry'] = (expiration - now).total_seconds() / 3600
                else:
                    info['hours_until_expiry'] = -((now - expiration).total_seconds() / 3600)

            info['is_valid'] = self.test_token_validity(self.access_token)

        return info

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Upstox Token Manager')
    parser.add_argument('--check', action='store_true', help='Check token status')
    parser.add_argument('--refresh', action='store_true', help='Force token refresh')
    parser.add_argument('--info', action='store_true', help='Show token information')
    parser.add_argument('--auto', action='store_true', help='Auto-check and refresh if needed')

    args = parser.parse_args()

    try:
        manager = UpstoxTokenManager()

        if args.info:
            info = manager.get_token_info()
            print("🔍 Token Information:")
            print(f"   Has Token: {info['has_token']}")
            print(f"   Token Length: {info['token_length']}")
            print(f"   Is Valid: {info['is_valid']}")
            print(f"   Is Expired: {info['is_expired']}")
            if info['expiration']:
                print(f"   Expiration: {info['expiration']}")
            if info['hours_until_expiry'] is not None:
                print(f"   Hours until expiry: {info['hours_until_expiry']:.1f}")
        elif args.check:
            valid = manager.test_token_validity(manager.access_token)
            print(f"✅ Token Valid: {valid}" if valid else f"❌ Token Invalid")
        elif args.refresh:
            success = manager.check_and_refresh_token(force_refresh=True)
            print(f"✅ Refresh Successful: {success}" if success else f"❌ Refresh Failed")
        elif args.auto:
            success = manager.check_and_refresh_token()
            status = "✅ Token OK" if success else "❌ Token Refresh Needed"
            print(status)
        else:
            # Default: show info
            info = manager.get_token_info()
            print("🔍 Token Status:")
            print(f"   Valid: {info['is_valid']}")
            print(f"   Expired: {info['is_expired']}")
            if info['hours_until_expiry'] is not None:
                if info['hours_until_expiry'] > 0:
                    print(f"   Hours until expiry: {info['hours_until_expiry']:.1f}")
                else:
                    print(f"   Hours expired: {abs(info['hours_until_expiry']):.1f}")
            if info['is_expired']:
                print("   ⚠️  ACTION REQUIRED: Run with --refresh to update token")

    except Exception as e:
        logger.error(f"Token manager error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()