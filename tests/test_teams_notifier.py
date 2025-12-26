"""
Tests for Teams notification functionality.
"""

import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.teams_notifier import post_plan_summary, post_error_notification


class TestTeamsNotifier:
    
    @patch('src.teams_notifier.requests.post')
    def test_post_plan_summary_success(self, mock_post):
        """Test successful Teams notification posting."""
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Sample data
        top_rows = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS'],
            'ENTRY_trigger_price': [2500.0, 3200.0],
            'STOPLOSS_trigger_price': [2400.0, 3100.0],
            'TARGET_trigger_price': [2600.0, 3300.0],
            'DecisionConfidence': [4.5, 4.2],
            'Audit_Flag': ['PASS', 'FAIL']
        })
        
        webhook_url = "https://test.webhook.url"
        result = post_plan_summary(webhook_url, "2024-01-15", 8, 2, "test.xlsx", top_rows)
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        assert 'type' in payload
        assert 'attachments' in payload
        assert len(payload['attachments']) == 1
        
        card = payload['attachments'][0]['content']
        assert card['type'] == 'AdaptiveCard'
        assert 'body' in card
        assert 'actions' in card
        
        # Check facts
        facts = None
        for element in card['body']:
            if element.get('type') == 'FactSet':
                facts = element.get('facts', [])
                break
        
        assert facts is not None
        assert len(facts) >= 3  # At least pass, fail, total counts
    
    @patch('src.teams_notifier.requests.post')
    def test_post_plan_summary_failure(self, mock_post):
        """Test Teams notification failure."""
        
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        top_rows = pd.DataFrame({
            'Symbol': ['RELIANCE.NS'],
            'ENTRY_trigger_price': [2500.0],
            'STOPLOSS_trigger_price': [2400.0],
            'TARGET_trigger_price': [2600.0],
            'DecisionConfidence': [4.5],
            'Audit_Flag': ['PASS']
        })
        
        result = post_plan_summary("https://test.url", "2024-01-15", 1, 0, "test.xlsx", top_rows)
        
        assert result is False
    
    @patch('src.teams_notifier.requests.post')
    def test_post_plan_summary_empty_data(self, mock_post):
        """Test Teams notification with empty data."""
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        empty_df = pd.DataFrame()
        
        result = post_plan_summary("https://test.url", "2024-01-15", 0, 0, "test.xlsx", empty_df)
        
        assert result is True
    
    @patch('src.teams_notifier.requests.post')
    def test_post_error_notification_success(self, mock_post):
        """Test successful error notification."""
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = post_error_notification("https://test.url", "Data fetch failed", "fetch")
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify error card structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        card = payload['attachments'][0]['content']
        assert card['type'] == 'AdaptiveCard'
        assert '🚨 SWING_BOT Error Alert' in card['body'][0]['text']
    
    @patch('src.teams_notifier.requests.post')
    def test_post_error_notification_failure(self, mock_post):
        """Test error notification failure."""
        
        mock_post.side_effect = requests.exceptions.RequestException("Webhook error")
        
        result = post_error_notification("https://test.url", "Test error", "test")
        
        assert result is False
    
    def test_adaptive_card_structure(self):
        """Test Adaptive Card payload structure."""
        
        with patch('src.teams_notifier.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            top_rows = pd.DataFrame({
                'Symbol': ['TEST.NS'],
                'ENTRY_trigger_price': [100.0],
                'STOPLOSS_trigger_price': [95.0],
                'TARGET_trigger_price': [110.0],
                'DecisionConfidence': [4.0],
                'Audit_Flag': ['PASS']
            })
            
            post_plan_summary("https://test.url", "2024-01-15", 1, 0, "test.xlsx", top_rows)
            
            # Verify the call was made
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            
            # Check required Adaptive Card fields
            assert payload['type'] == 'message'
            assert 'attachments' in payload
            
            card = payload['attachments'][0]['content']
            assert card['$schema'] == 'http://adaptivecards.io/schemas/adaptive-card.json'
            assert card['type'] == 'AdaptiveCard'
            assert card['version'] == '1.4'
            assert isinstance(card['body'], list)
            assert len(card['body']) > 0
            
            # Check for title
            title_found = False
            for element in card['body']:
                if element.get('text', '').startswith('🎯 SWING_BOT EOD Summary'):
                    title_found = True
                    break
            assert title_found, "Title not found in card body"
    
    def test_webhook_url_validation(self):
        """Test that webhook URL is properly passed."""
        
        with patch('src.teams_notifier.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            test_url = "https://custom.webhook.url/teams"
            top_rows = pd.DataFrame({
                'Symbol': ['TEST.NS'],
                'ENTRY_trigger_price': [100.0],
                'STOPLOSS_trigger_price': [95.0],
                'TARGET_trigger_price': [110.0],
                'DecisionConfidence': [4.0],
                'Audit_Flag': ['PASS']
            })
            
            post_plan_summary(test_url, "2024-01-15", 1, 0, "test.xlsx", top_rows)
            
            # Verify URL was used
            call_args = mock_post.call_args
            assert call_args[0][0] == test_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])