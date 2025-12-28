"""
Teams notification module for SWING_BOT.
Posts Adaptive Card notifications to Microsoft Teams via webhooks.
"""

import requests
import json
from typing import Optional
import pandas as pd


def post_plan_summary(webhook_url: str, date: str, pass_count: int, fail_count: int, 
                     excel_path: str, top_rows: pd.DataFrame) -> bool:
    """
    Post GTT plan summary to Teams using Adaptive Cards.
    
    Args:
        webhook_url: Teams webhook URL
        date: Date string for the plan
        pass_count: Number of audit passes
        fail_count: Number of audit fails
        excel_path: Path to the final Excel file
        top_rows: Top 5 rows from audited plan
        
    Returns:
        True if successful, False otherwise
    """
    
    # Build the Adaptive Card payload
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "🎯 SWING_BOT EOD Summary",
                "weight": "Bolder",
                "size": "Large",
                "color": "Accent"
            },
            {
                "type": "TextBlock", 
                "text": f"📅 Date: {date}",
                "weight": "Default",
                "size": "Medium"
            },
            {
                "type": "FactSet",
                "facts": [
                    {
                        "title": "✅ Audit Passes:",
                        "value": str(pass_count)
                    },
                    {
                        "title": "❌ Audit Fails:",
                        "value": str(fail_count)
                    },
                    {
                        "title": "📊 Total Positions:",
                        "value": str(pass_count + fail_count)
                    }
                ]
            },
            {
                "type": "TextBlock",
                "text": "📈 Top 5 Positions:",
                "weight": "Bolder",
                "size": "Medium",
                "spacing": "Medium"
            }
        ]
    }
    
    # Add top positions table
    if not top_rows.empty:
        table_rows = []
        for _, row in top_rows.iterrows():
            audit_icon = "✅" if row['Audit_Flag'] == 'PASS' else "❌"
            table_rows.append({
                "type": "FactSet",
                "facts": [
                    {"title": "Symbol:", "value": str(row['Symbol'])},
                    {"title": "Entry:", "value": f"₹{row['ENTRY_trigger_price']:.2f}"},
                    {"title": "Stop:", "value": f"₹{row['STOPLOSS_trigger_price']:.2f}"},
                    {"title": "Target:", "value": f"₹{row['TARGET_trigger_price']:.2f}"},
                    {"title": "Confidence:", "value": f"{row['DecisionConfidence']:.1f}%"},
                    {"title": "Audit:", "value": f"{audit_icon} {row['Audit_Flag']}"}
                ]
            })
        
        card["body"].extend(table_rows)
    
    # Add action button for Excel download
    card["actions"] = [
        {
            "type": "Action.OpenUrl",
            "title": "📁 Download Excel Report",
            "url": f"file://{excel_path}"  # This will work if Teams can access the file path
        }
    ]
    
    # Post to Teams
    payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Teams notification posted successfully")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post to Teams: {str(e)}")
        return False


def post_error_notification(webhook_url: str, error_message: str, stage: str) -> bool:
    """
    Post error notification to Teams.
    
    Args:
        webhook_url: Teams webhook URL
        error_message: Error message
        stage: Pipeline stage where error occurred
        
    Returns:
        True if successful, False otherwise
    """
    
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "🚨 SWING_BOT Error Alert",
                "weight": "Bolder",
                "size": "Large",
                "color": "Attention"
            },
            {
                "type": "TextBlock",
                "text": f"❌ Stage: {stage}",
                "weight": "Default",
                "size": "Medium"
            },
            {
                "type": "TextBlock",
                "text": f"Error: {error_message}",
                "wrap": True
            }
        ]
    }
    
    payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post error to Teams: {str(e)}")
        return False


def post_live_results_summary(plan_df: pd.DataFrame, placed_orders: list) -> bool:
    """
    Post live orchestration results summary to Teams.

    Args:
        plan_df: DataFrame with reconciled plan
        placed_orders: List of placed order records

    Returns:
        True if successful, False otherwise
    """
    import os

    webhook_url = os.environ.get('TEAMS_WEBHOOK_URL')
    if not webhook_url:
        print("⚠️  TEAMS_WEBHOOK_URL not set, skipping Teams notification")
        return False

    # Calculate summary stats
    total_candidates = len(plan_df)
    pass_count = (plan_df['Audit_Flag'] == 'PASS').sum()
    fail_count = (plan_df['Audit_Flag'] == 'FAIL').sum()
    placed_count = len(placed_orders)

    # Build top placed orders text
    top_placed_text = ""
    if placed_orders:
        top_placed = placed_orders[:5]  # Top 5
        for order in top_placed:
            symbol = order.get('Symbol', 'UNKNOWN')
            entry = order.get('ENTRY_trigger_price', 0)
            stop = order.get('STOPLOSS_trigger_price', 0)
            target = order.get('TARGET_trigger_price', 0)
            conf = order.get('DecisionConfidence', 0)
            order_id = order.get('order_id', 'UNKNOWN')
            top_placed_text += f"• {symbol}: ₹{entry:.2f} → ₹{target:.2f} (Stop: ₹{stop:.2f}) Conf:{conf:.2f} ID:{order_id}\\n"

    if not top_placed_text:
        top_placed_text = "No orders placed"

    # Build the Adaptive Card payload
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "🚀 SWING_BOT Live EOD Results",
                "weight": "Bolder",
                "size": "Large",
                "color": "Good"
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Candidates:", "value": str(total_candidates)},
                    {"title": "PASS:", "value": str(pass_count)},
                    {"title": "FAIL:", "value": str(fail_count)},
                    {"title": "GTT Placed:", "value": str(placed_count)}
                ]
            },
            {
                "type": "TextBlock",
                "text": f"**Placed Orders (Top 5):**\\n{top_placed_text}",
                "wrap": True
            }
        ]
    }

    payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post live results to Teams: {str(e)}")
        return False


def post_teams_message(webhook_url: str, message: str, title: str = "SWING_BOT Notification") -> bool:
    """
    Post a simple message to Microsoft Teams.

    Args:
        webhook_url: Teams webhook URL
        message: Message content (can include markdown)
        title: Message title

    Returns:
        bool: True if successful
    """
    try:
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "text": message,
                "markdown": True
            }]
        }

        response = requests.post(
            webhook_url,
            json=card,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post message to Teams: {str(e)}")
        return False