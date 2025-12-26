"""
SWING_BOT Teams Dashboard

Generates HTML reports and Adaptive Card payloads for Teams notifications.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pytz

from ..utils import get_ist_now, format_datetime

def build_daily_html(
    plan_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    screener_df: pd.DataFrame,
    out_html: str,
    reconciled_df: Optional[pd.DataFrame] = None,
    tz: str = "Asia/Kolkata"
) -> None:
    """
    Build daily HTML dashboard report.

    Args:
        plan_df: GTT plan DataFrame
        audit_df: Audited plan DataFrame
        screener_df: Screener results DataFrame
        out_html: Output HTML file path
        reconciled_df: Optional reconciled plan DataFrame for LTP metrics
        tz: Timezone for timestamps
    """

    # Ensure output directory exists
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    # Get current timestamp
    now = get_ist_now()
    latest_date = now.date()

    # Calculate metrics
    pass_count = (audit_df['Audit_Flag'] == 'PASS').sum() if 'Audit_Flag' in audit_df.columns else 0
    fail_count = (audit_df['Audit_Flag'] == 'FAIL').sum() if 'Audit_Flag' in audit_df.columns else 0
    total_positions = len(audit_df)

    # LTP Reconciliation metrics
    reconciled_count = 0
    avg_ltp_delta_pct = 0.0
    if reconciled_df is not None and not reconciled_df.empty:
        reconciled_count = len(reconciled_df)
        if 'ltp_delta_pct' in reconciled_df.columns:
            avg_ltp_delta_pct = reconciled_df['ltp_delta_pct'].mean()
            # Convert to percentage for display
            avg_ltp_delta_pct *= 100

    # Data freshness (assume latest data from screener)
    data_freshness_days = 0
    if 'Date' in screener_df.columns:
        try:
            latest_data_date = pd.to_datetime(screener_df['Date']).max().date()
            data_freshness_days = (latest_date - latest_data_date).days
        except:
            data_freshness_days = 0

    # Coverage days (assume from data validation)
    coverage_days = 500  # Default assumption
    symbols_count = len(screener_df['Symbol'].unique()) if 'Symbol' in screener_df.columns else 0

    # Runtime (mock for now - would be passed from orchestration)
    runtime_seconds = 45.2

    # Top positions
    top_positions = audit_df.head(10) if not audit_df.empty else pd.DataFrame()

    # Audit issues summary
    issues_summary = {}
    if 'Issues' in audit_df.columns and 'Fix_Suggestion' in audit_df.columns:
        for _, row in audit_df.iterrows():
            issue = str(row.get('Issues', ''))
            fix = str(row.get('Fix_Suggestion', ''))
            if issue:
                if issue not in issues_summary:
                    issues_summary[issue] = {'count': 0, 'fix': fix}
                issues_summary[issue]['count'] += 1

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWING_BOT Daily Dashboard - {latest_date}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .kpi-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .kpi-label {{
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .status-pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .issues-list {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        .issue-item {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SWING_BOT</h1>
            <div class="subtitle">Daily Trading Dashboard - {latest_date}</div>
            <div style="margin-top: 20px; font-size: 1.2em;">
                Generated at {format_datetime(now)}
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">{pass_count}/{total_positions}</div>
                <div class="kpi-label">Audit Pass Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{data_freshness_days}</div>
                <div class="kpi-label">Data Freshness (days)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{symbols_count}</div>
                <div class="kpi-label">Symbols Tracked</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{coverage_days}</div>
                <div class="kpi-label">Coverage Days</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{runtime_seconds:.1f}s</div>
                <div class="kpi-label">Runtime</div>
            </div>
            {f'''
            <div class="kpi-card">
                <div class="kpi-value">{reconciled_count}</div>
                <div class="kpi-label">Positions Reconciled</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{avg_ltp_delta_pct:+.2f}%</div>
                <div class="kpi-label">Avg LTP Delta</div>
            </div>
            ''' if reconciled_df is not None else ''}
        </div>

        <div class="section">
            <h2>Top Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Entry Price</th>
                        <th>Stop Loss</th>
                        <th>Target</th>
                        <th>Confidence</th>
                        <th>Audit Status</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add top positions rows
    if not top_positions.empty:
        for _, row in top_positions.iterrows():
            audit_flag = row.get('Audit_Flag', 'UNKNOWN')
            status_class = 'status-pass' if audit_flag == 'PASS' else 'status-fail'

            html_content += f"""
                    <tr>
                        <td>{row.get('Symbol', 'N/A')}</td>
                        <td>₹{row.get('ENTRY_trigger_price', 0):.2f}</td>
                        <td>₹{row.get('STOPLOSS_trigger_price', 0):.2f}</td>
                        <td>₹{row.get('TARGET_trigger_price', 0):.2f}</td>
                        <td>{row.get('DecisionConfidence', 0):.3f}</td>
                        <td class="{status_class}">{audit_flag}</td>
                    </tr>
"""
    else:
        html_content += """
                    <tr>
                        <td colspan="6" style="text-align: center;">No positions available</td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Audit Issues Summary</h2>
"""

    if issues_summary:
        for issue, data in issues_summary.items():
            html_content += f"""
            <div class="issues-list">
                <div class="issue-item">
                    <strong>Issue:</strong> {issue}<br>
                    <strong>Count:</strong> {data['count']}<br>
                    <strong>Suggested Fix:</strong> {data['fix']}
                </div>
            </div>
"""
    else:
        html_content += """
            <div class="issues-list">
                <div class="issue-item">
                    ✅ No audit issues found
                </div>
            </div>
"""

    html_content += f"""
        </div>

        <div class="footer">
            <p>SWING_BOT Automated Trading System | Generated on {format_datetime(now)}</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Dashboard HTML generated: {out_html}")


def build_adaptive_card_summary(
    latest_date: str,
    pass_count: int,
    fail_count: int,
    top_rows_df: pd.DataFrame,
    links: Dict[str, str]
) -> Dict[str, Any]:
    """
    Build Adaptive Card JSON for success summary.

    Args:
        latest_date: Latest trading date
        pass_count: Number of passing positions
        fail_count: Number of failing positions
        top_rows_df: Top positions DataFrame
        links: Dictionary of file links

    Returns:
        Adaptive Card JSON payload
    """

    # Build facts
    facts = [
        {"title": "Date:", "value": latest_date},
        {"title": "Pass Count:", "value": str(pass_count)},
        {"title": "Fail Count:", "value": str(fail_count)},
        {"title": "Total Positions:", "value": str(pass_count + fail_count)}
    ]

    # Build top positions text
    top_positions_text = "Top Positions:\\n"
    if not top_rows_df.empty:
        for i, (_, row) in enumerate(top_rows_df.head(5).iterrows(), 1):
            symbol = row.get('Symbol', 'N/A')
            confidence = row.get('DecisionConfidence', 0)
            audit_flag = row.get('Audit_Flag', 'UNKNOWN')
            top_positions_text += f"{i}. {symbol} (Conf: {confidence:.2f}, Audit: {audit_flag})\\n"
    else:
        top_positions_text += "No positions available"

    # Build actions
    actions = []
    for name, url in links.items():
        if url.startswith('file://') or url.startswith('http'):
            actions.append({
                "type": "Action.OpenUrl",
                "title": f"View {name}",
                "url": url
            })

    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "📊 SWING_BOT Daily Summary",
                "weight": "Bolder",
                "size": "Large",
                "color": "Good"
            },
            {
                "type": "FactSet",
                "facts": facts
            },
            {
                "type": "TextBlock",
                "text": top_positions_text,
                "wrap": True
            }
        ]
    }

    if actions:
        card["actions"] = actions

    return card


def build_failure_card(
    stage: str,
    error_msg: str,
    hints: List[str],
    links: Dict[str, str]
) -> Dict[str, Any]:
    """
    Build Adaptive Card JSON for failure notification.

    Args:
        stage: Pipeline stage where failure occurred
        error_msg: Error message
        hints: List of troubleshooting hints
        links: Dictionary of file links

    Returns:
        Adaptive Card JSON payload
    """

    # Build hints text
    hints_text = "Troubleshooting Hints:\\n" + "\\n".join(f"• {hint}" for hint in hints)

    # Build actions
    actions = []
    for name, url in links.items():
        if url.startswith('file://') or url.startswith('http'):
            actions.append({
                "type": "Action.OpenUrl",
                "title": f"View {name}",
                "url": url
            })

    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "🚨 SWING_BOT Pipeline Failure",
                "weight": "Bolder",
                "size": "Large",
                "color": "Attention"
            },
            {
                "type": "TextBlock",
                "text": f"**Stage:** {stage}",
                "weight": "Bolder"
            },
            {
                "type": "TextBlock",
                "text": f"**Error:** {error_msg}",
                "wrap": True,
                "color": "Attention"
            },
            {
                "type": "TextBlock",
                "text": hints_text,
                "wrap": True
            }
        ]
    }

    if actions:
        card["actions"] = actions

    return card