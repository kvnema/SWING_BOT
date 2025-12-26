import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta


def print_live_results_summary(plan_df: pd.DataFrame, placed_orders_df: pd.DataFrame = None) -> None:
    """Print human-readable summary of live orchestration results.

    Args:
        plan_df: DataFrame with audited plan (Symbol, Audit_Flag, DecisionConfidence, etc.)
        placed_orders_df: DataFrame with placed orders (Symbol, order_id, etc.)
    """
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    print(f"\n{'='*60}")
    print(f"SWING_BOT Live EOD — {ist_now.strftime('%Y-%m-%d')} (IST)")
    print(f"{'='*60}")

    # Basic counts
    total_candidates = len(plan_df)
    pass_count = (plan_df['Audit_Flag'] == 'PASS').sum()
    fail_count = (plan_df['Audit_Flag'] == 'FAIL').sum()

    placed_count = len(placed_orders_df) if placed_orders_df is not None else 0
    modified_count = 0  # TODO: track modifications separately if needed
    skipped_count = pass_count - placed_count if pass_count > 0 else 0

    print(f"Candidates: {total_candidates} | PASS: {pass_count} | FAIL: {fail_count}")
    print(f"Placed GTT: {placed_count} | Modified existing: {modified_count} | Skipped: {skipped_count}")

    # Placed orders details
    if placed_orders_df is not None and not placed_orders_df.empty:
        print("\nPlaced (Top 10):")
        top_placed = placed_orders_df.head(10)

        for _, row in top_placed.iterrows():
            symbol = row.get('Symbol', 'UNKNOWN')
            entry = row.get('ENTRY_trigger_price', 0)
            stop = row.get('STOPLOSS_trigger_price', 0)
            target = row.get('TARGET_trigger_price', 0)
            conf = row.get('DecisionConfidence', 0)
            order_id = row.get('order_id', 'UNKNOWN')

            print(f"  {symbol:<10} Entry=₹{entry:>8.2f} Stop=₹{stop:>8.2f} Target=₹{target:>8.2f}  Conf={conf:.2f}  OrderID={order_id}")

    # Skipped reasons summary
    if skipped_count > 0:
        print("\nSkipped reasons:")
        skipped_reasons = []

        # Confidence too low
        low_conf = plan_df[(plan_df['Audit_Flag'] == 'PASS') & (plan_df['DecisionConfidence'] < 0.70)]
        if not low_conf.empty:
            skipped_reasons.append(f"Confidence<0.70 ({len(low_conf)})")

        # Audit failures
        if fail_count > 0:
            skipped_reasons.append(f"Audit FAIL ({fail_count})")

        # Other reasons (instrument mismatch, etc.)
        other_skip = skipped_count - len(low_conf)
        if other_skip > 0:
            skipped_reasons.append(f"Other ({other_skip})")

        print("  " + ", ".join(skipped_reasons))

    # Artifacts
    print("\nArtifacts:")
    print("  outputs/gtt/GTT_Delivery_Final.xlsx")
    print("  outputs/gtt/gtt_plan_latest.csv")
    print("  outputs/dashboard.html")

    print(f"{'='*60}\n")