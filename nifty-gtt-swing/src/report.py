import pandas as pd
import requests
import os


def save_screener(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def save_backtest_outputs(kpi: dict, trades: pd.DataFrame, equity_curve: pd.DataFrame, outdir: str):
    import os
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame([kpi]).to_csv(f'{outdir}/kpi.csv', index=False)
    trades.to_csv(f'{outdir}/trades.csv', index=False)
    equity_curve.to_csv(f'{outdir}/equity_curve.csv', index=False)


def send_teams_notification(webhook_url: str, title: str, text: str, facts: list = None):
    """Send Adaptive Card to Teams webhook."""
    card = {
        "type": "AdaptiveCard",
        "version": "1.2",
        "body": [
            {"type": "TextBlock", "text": title, "weight": "Bolder", "size": "Medium"},
            {"type": "TextBlock", "text": text, "wrap": True}
        ]
    }
    if facts:
        card["body"].append({
            "type": "FactSet",
            "facts": [{"title": k, "value": v} for k, v in facts]
        })
    payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
    requests.post(webhook_url, json=payload)
