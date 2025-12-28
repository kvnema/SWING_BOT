#!/usr/bin/env python3
"""
SWING_BOT ICICI Direct Session Manager
Manages ICICI Direct API session tokens and automatic re-authentication.
"""

import os
import json
import time
import logging
import requests
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import urllib.parse
from dotenv import load_dotenv, set_key

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/icici_session_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ICICISessionManager:
    """Manages ICICI Direct API session lifecycle and automatic refresh."""

    def __init__(self, env_file: str = '.env'):
        self.env_file = Path(env_file)
        self.session_file = Path('data/icici_session_status.json')
        self.load_credentials()

    def load_credentials(self):
        """Load API credentials from environment."""
        load_dotenv(self.env_file)
        self.api_key = os.getenv('ICICI_API_KEY')
        self.api_secret = os.getenv('ICICI_API_SECRET')
        self.session_token = os.getenv('ICICI_SESSION_TOKEN')

        if not all([self.api_key, self.api_secret]):
            raise ValueError("Missing ICICI_API_KEY or ICICI_API_SECRET in .env")

    def decode_session_token(self, session_token: str) -> Dict[str, Any]:
        """Decode base64 session token to extract information."""
        try:
            # Remove padding if present and add required padding
            session_token = session_token.replace('=', '')
            missing_padding = len(session_token) % 4
            if missing_padding:
                session_token += '=' * (4 - missing_padding)

            decoded = base64.urlsafe_b64decode(session_token)
            decoded_str = decoded.decode('utf-8')

            # Try to parse as JSON, if not return as string
            try:
                return json.loads(decoded_str)
            except json.JSONDecodeError:
                return {'raw': decoded_str}

        except Exception as e:
            logger.error(f"Failed to decode session token: {e}")
            return {}

    def test_session_validity(self, session_token: str) -> bool:
        """Test if session token is valid by making a simple API call."""
        try:
            from .icici_gtt import ICICIAPIClient
            client = ICICIAPIClient(self.api_key, self.api_secret, session_token)
            response = client.get_customer_details()

            if response['status_code'] == 200:
                return True
            elif response['status_code'] == 401:
                logger.info("Session token expired")
                return False
            else:
                logger.warning(f"Unexpected response: {response}")
                return False

        except Exception as e:
            logger.error(f"Session validity test failed: {e}")
            return False

    def generate_login_url(self) -> str:
        """Generate the login URL for ICICI Direct."""
        encoded_api_key = urllib.parse.quote_plus(self.api_key)
        return f"https://api.icicidirect.com/apiuser/login?api_key={encoded_api_key}"

    def authenticate_via_browser(self) -> Optional[str]:
        """
        Authenticate via browser login flow.
        Returns session token if successful.
        """
        logger.info("ICICI Direct session authentication required")

        print("=" * 60)
        print("🔐 ICICI DIRECT SESSION AUTHENTICATION REQUIRED")
        print("=" * 60)
        print("1. Open this URL in your browser:")
        print(f"   {self.generate_login_url()}")
        print()
        print("2. Log in to your ICICI Direct account")
        print("3. Grant permission for the app")
        print("4. Copy the API Session value from the browser URL after login")
        print("   (It will look like: ...api_session=ABC123...)")
        print()

        api_session = input("Enter the API Session value: ").strip()

        if not api_session:
            logger.error("No API session provided")
            return None

        # Exchange API session for session token
        return self.exchange_api_session_for_token(api_session)

    def exchange_api_session_for_token(self, api_session: str) -> Optional[str]:
        """Exchange API session for session token using CustomerDetails API."""
        try:
            # Make request to get customer details with API session
            url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({
                'SessionToken': api_session,
                'AppKey': self.api_key
            })

            response = requests.post(url, headers=headers, data=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('Status') == 200 and 'Success' in result:
                    session_token = result['Success'].get('session_token')
                    if session_token:
                        logger.info("Successfully obtained session token")
                        return session_token
                    else:
                        logger.error("No session_token in response")
                else:
                    logger.error(f"Authentication failed: {result}")
            else:
                logger.error(f"HTTP error during authentication: {response.status_code}")

        except Exception as e:
            logger.error(f"Authentication error: {e}")

        return None

    def update_session_token_in_env(self, new_token: str):
        """Update the session token in the .env file."""
        try:
            set_key(self.env_file, 'ICICI_SESSION_TOKEN', new_token)
            self.session_token = new_token
            logger.info("Updated session token in .env file")
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
            raise

    def save_session_status(self):
        """Save current session status for monitoring."""
        status = {
            'last_check': datetime.now().isoformat(),
            'session_valid': self.test_session_validity(self.session_token) if self.session_token else False,
            'session_token_length': len(self.session_token) if self.session_token else 0,
            'decoded_info': self.decode_session_token(self.session_token) if self.session_token else {}
        }

        try:
            self.session_file.parent.mkdir(exist_ok=True)
            with open(self.session_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session status: {e}")

    def load_session_status(self) -> Dict[str, Any]:
        """Load saved session status."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load session status: {e}")
        return {}

    def check_and_refresh_session(self, force_refresh: bool = False) -> bool:
        """
        Check session status and refresh if needed.

        Args:
            force_refresh: Force session refresh even if valid

        Returns:
            bool: True if session is valid/refreshed successfully
        """
        logger.info("Checking ICICI session status...")

        # Check if session token exists
        if not self.session_token:
            logger.warning("No session token found")
            return False

        # Check if session is still valid
        if not force_refresh and self.test_session_validity(self.session_token):
            logger.info("✅ Session is still valid")
            self.save_session_status()
            return True

        # Session expired or force refresh requested
        logger.info("Session expired or refresh requested, re-authenticating...")

        # Attempt to get new session token
        new_token = self.authenticate_via_browser()

        if new_token:
            # Update environment
            self.update_session_token_in_env(new_token)

            # Test new session
            if self.test_session_validity(new_token):
                logger.info("✅ Session refresh successful")
                self.save_session_status()
                return True
            else:
                logger.error("❌ New session token is invalid")
                return False
        else:
            logger.error("❌ Session refresh failed")
            return False

    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information."""
        info = {
            'has_session_token': bool(self.session_token),
            'session_token_length': len(self.session_token) if self.session_token else 0,
            'is_valid': False,
            'decoded_info': {}
        }

        if self.session_token:
            info['is_valid'] = self.test_session_validity(self.session_token)
            info['decoded_info'] = self.decode_session_token(self.session_token)

        return info

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='ICICI Direct Session Manager')
    parser.add_argument('--check', action='store_true', help='Check session status')
    parser.add_argument('--refresh', action='store_true', help='Force session refresh')
    parser.add_argument('--info', action='store_true', help='Show session information')
    parser.add_argument('--auto', action='store_true', help='Auto-check and refresh if needed')

    args = parser.parse_args()

    try:
        manager = ICICISessionManager()

        if args.info:
            info = manager.get_session_info()
            print("🔍 Session Information:")
            print(f"   Has Session Token: {info['has_session_token']}")
            print(f"   Token Length: {info['session_token_length']}")
            print(f"   Is Valid: {info['is_valid']}")
            if info['decoded_info']:
                print(f"   Decoded Info: {info['decoded_info']}")
        elif args.check:
            valid = manager.test_session_validity(manager.session_token)
            print(f"✅ Session Valid: {valid}" if valid else f"❌ Session Invalid")
        elif args.refresh:
            success = manager.check_and_refresh_session(force_refresh=True)
            print(f"✅ Refresh Successful: {success}" if success else f"❌ Refresh Failed")
        elif args.auto:
            success = manager.check_and_refresh_session()
            status = "✅ Session OK" if success else "❌ Session Refresh Needed"
            print(status)
        else:
            # Default: show info
            info = manager.get_session_info()
            print("🔍 Session Status:")
            print(f"   Valid: {info['is_valid']}")
            if not info['is_valid']:
                print("   ⚠️  ACTION REQUIRED: Run with --refresh to update session")

    except Exception as e:
        logger.error(f"Session manager error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()