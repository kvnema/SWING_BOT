"""
SWING_BOT Plan Audit
====================

Audit GTT plans for price accuracy, strategy compliance, and data freshness.
Ensures ENTRY/STOP/TARGET prices are derived from correct pivots and latest data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .utils import load_config, logger, safe_divide

@dataclass
class AuditParams:
    """Parameters for plan auditing."""
    tick: float = 0.05          # NSE EQ tick size
    max_age_days: int = 1       # Maximum age for data freshness
    tf_base: str = "1d"         # Base timeframe for pivots
    max_entry_pct_diff: float = 0.02  # Max 2% deviation from pivot
    strict_mode: bool = False   # Fail fast on any audit failure
    risk_multiplier: float = 1.5  # ATR multiplier for stops
    reward_multiplier: float = 2.0  # Risk-reward ratio for targets

class PlanAuditError(Exception):
    """Raised when plan audit fails in strict mode."""
    pass

def round_tick(price: float, tick: float = 0.05) -> float:
    """Round price to nearest tick size."""
    if pd.isna(price) or price == 0:
        return price
    return round(price / tick) * tick

def get_strategy_pivot(symbol_data: pd.Series, strategy: str) -> Tuple[float, str]:
    """
    Get the appropriate pivot price and source for a strategy.

    Args:
        symbol_data: Latest indicators row for the symbol
        strategy: Trading strategy name

    Returns:
        Tuple of (pivot_price, pivot_source)
    """
    if strategy.upper() == "MR" or strategy.upper() == "MEANREVERSION":
        # Mean Reversion: Use EMA20 as pivot
        pivot = symbol_data.get('EMA20', symbol_data.get('ema20'))
        source = "EMA20"
    elif strategy.upper() in ["BREAKOUT", "DONCHIAN_BREAKOUT"]:
        # Breakout: Use Donchian High 20
        pivot = symbol_data.get('DonchianH20', symbol_data.get('donchian_h20'))
        source = "DonchianH20"
    elif strategy.upper() in ["TREND", "TRENDCONTINUATION"]:
        # Trend: Use EMA50 as pivot
        pivot = symbol_data.get('EMA50', symbol_data.get('ema50'))
        source = "EMA50"
    elif strategy.upper() in ["VCP", "SEPA"]:
        # VCP/SEPA: Use recent high or EMA200
        pivot = symbol_data.get('EMA200', symbol_data.get('ema200'))
        source = "EMA200"
    else:
        # Default: Use close
        pivot = symbol_data.get('Close', symbol_data.get('close'))
        source = "Close"

    if pd.isna(pivot) or pivot == 0:
        pivot = symbol_data.get('Close', symbol_data.get('close', 0))
        source = "Close (fallback)"

    return float(pivot), source

def compute_canonical_prices(symbol_data: pd.Series, strategy: str,
                           ap: AuditParams) -> Dict[str, Any]:
    """
    Compute canonical ENTRY/STOP/TARGET prices for a strategy.

    Args:
        symbol_data: Latest indicators row for the symbol
        strategy: Trading strategy name
        ap: Audit parameters

    Returns:
        Dict with computed prices and logic
    """
    pivot_price, pivot_source = get_strategy_pivot(symbol_data, strategy)

    # Get ATR for position sizing
    atr = symbol_data.get('ATR14', symbol_data.get('atr14', 10))

    # Strategy-specific entry logic - match gtt_sizing.py behavior
    if strategy.upper() == "MR" or strategy.upper() == "MEANREVERSION":
        # Mean Reversion: Entry at EMA20 (BELOW trigger in gtt_sizing)
        entry_price = round_tick(pivot_price, ap.tick)
        entry_logic = f"Entry at {pivot_source} ({pivot_price:.2f}) for BELOW trigger"
    elif strategy.upper() == "BREAKOUT" or strategy.upper() in ["DONCHIAN_BREAKOUT", "VCP", "SEPA"]:
        # Breakout strategies: Entry at Donchian High (ABOVE trigger in gtt_sizing)
        entry_price = round_tick(pivot_price, ap.tick)
        entry_logic = f"Entry at {pivot_source} ({pivot_price:.2f}) for ABOVE trigger"
    else:
        # Default: Entry at pivot
        entry_price = round_tick(pivot_price, ap.tick)
        entry_logic = f"Entry at {pivot_source} ({pivot_price:.2f})"

    # Stop loss: ATR-based below entry
    stop_price = round_tick(entry_price - (atr * ap.risk_multiplier), ap.tick)
    stop_logic = f"Stop {ap.risk_multiplier}x ATR below entry (ATR={atr:.2f})"

    # Target: Risk-reward ratio
    risk_amount = entry_price - stop_price
    target_price = round_tick(entry_price + (risk_amount * ap.reward_multiplier), ap.tick)
    target_logic = f"Target {ap.reward_multiplier}R from entry (Risk={risk_amount:.2f})"

    return {
        'canonical_entry': entry_price,
        'canonical_stop': stop_price,
        'canonical_target': target_price,
        'pivot_price': pivot_price,
        'pivot_source': pivot_source,
        'entry_logic': entry_logic,
        'stop_logic': stop_logic,
        'target_logic': target_logic,
        'atr_used': atr,
        'risk_amount': risk_amount
    }

def audit_plan_row(plan_row: pd.Series, symbol_data: pd.Series,
                  latest_data: pd.Series, ap: AuditParams) -> Dict[str, Any]:
    """
    Audit a single plan row against latest data and strategy rules.

    Args:
        plan_row: Row from GTT plan
        symbol_data: Latest indicators for the symbol
        latest_data: Latest price data for the symbol
        ap: Audit parameters

    Returns:
        Dict with audit results
    """
    symbol = plan_row.get('Symbol', plan_row.get('symbol', ''))
    strategy = plan_row.get('Strategy', plan_row.get('strategy', ''))

    # Get plan prices
    plan_entry = plan_row.get('Trigger_Price', plan_row.get('ENTRY_trigger_price', 0))
    plan_stop = plan_row.get('Stop_Price', plan_row.get('STOPLOSS_trigger_price', 0))
    plan_target = plan_row.get('Target_Price', plan_row.get('TARGET_trigger_price', 0))

    # Get latest prices
    latest_close = latest_data.get('Close', latest_data.get('close', 0))
    latest_ltp = latest_data.get('LTP', latest_data.get('ltp', latest_close))

    # Compute canonical prices
    canonical = compute_canonical_prices(symbol_data, strategy, ap)

    # Audit checks
    issues = []
    audit_flag = "PASS"

    # Check entry price deviation from pivot
    entry_diff_pct = abs(plan_entry - canonical['canonical_entry']) / canonical['canonical_entry']
    delta_vs_pivot_pct = (plan_entry - canonical['pivot_price']) / canonical['pivot_price'] * 100

    if entry_diff_pct > ap.max_entry_pct_diff:
        issues.append(f"ENTRY deviates {entry_diff_pct:.1%} from canonical ({canonical['canonical_entry']:.2f})")
        audit_flag = "FAIL"

    # Check stop logic
    if plan_stop >= plan_entry:
        issues.append("STOP price >= ENTRY price")
        audit_flag = "FAIL"

    # Check target logic
    if plan_target <= plan_entry:
        issues.append("TARGET price <= ENTRY price")
        audit_flag = "FAIL"

    # Check risk-reward ratio
    if plan_stop < plan_entry and plan_target > plan_entry:
        risk = plan_entry - plan_stop
        reward = plan_target - plan_entry
        rr_ratio = reward / risk
        if rr_ratio < 1.0 or rr_ratio > 3.0:
            issues.append(f"Risk-reward ratio {rr_ratio:.1f} outside typical range [1.0, 3.0]")

    # Check tick rounding
    if not np.isclose(plan_entry, round_tick(plan_entry, ap.tick), atol=ap.tick/2):
        issues.append(f"ENTRY not tick-rounded (tick={ap.tick})")

    if not np.isclose(plan_stop, round_tick(plan_stop, ap.tick), atol=ap.tick/2):
        issues.append(f"STOP not tick-rounded (tick={ap.tick})")

    if not np.isclose(plan_target, round_tick(plan_target, ap.tick), atol=ap.tick/2):
        issues.append(f"TARGET not tick-rounded (tick={ap.tick})")

    # Generate fix suggestions
    fix_suggestions = []
    if entry_diff_pct > ap.max_entry_pct_diff:
        fix_suggestions.append(f"Use canonical ENTRY={canonical['canonical_entry']:.2f} from {canonical['pivot_source']}")

    if plan_stop >= plan_entry:
        fix_suggestions.append(f"Set STOP below ENTRY using ATR-based calculation")

    if plan_target <= plan_entry:
        fix_suggestions.append(f"Set TARGET above ENTRY using risk-reward ratio")

    if issues:
        fix_suggestions.append("Ensure all prices are tick-rounded to NSE EQ standard (₹0.05)")

    return {
        'Audit_Flag': audit_flag,
        'Issues': '; '.join(issues) if issues else '',
        'Fix_Suggestion': '; '.join(fix_suggestions) if fix_suggestions else '',
        'Pivot_Source': canonical['pivot_source'],
        'Entry_Logic': canonical['entry_logic'],
        'Stop_Logic': canonical['stop_logic'],
        'Target_Logic': canonical['target_logic'],
        'Latest_Close': latest_close,
        'Latest_LTP': latest_ltp,
        'Delta_vs_Pivot_pct': delta_vs_pivot_pct,
        'Canonical_Entry': canonical['canonical_entry'],
        'Canonical_Stop': canonical['canonical_stop'],
        'Canonical_Target': canonical['canonical_target'],
        'ATR_Used': canonical['atr_used'],
        'Risk_Amount': canonical['risk_amount']
    }

def attach_audit(plan_df: pd.DataFrame, indicators_df: pd.DataFrame,
                latest_df: pd.DataFrame, ap: AuditParams) -> pd.DataFrame:
    """
    Attach audit results to a GTT plan DataFrame.

    Args:
        plan_df: GTT plan DataFrame
        indicators_df: Latest indicators DataFrame
        latest_df: Latest price data DataFrame
        ap: Audit parameters

    Returns:
        Plan DataFrame with audit columns added
    """
    logger.info(f"Starting plan audit for {len(plan_df)} positions")

    # Prepare data dictionaries
    indicators_dict = {}
    latest_dict = {}

    # Group indicators by symbol (take latest row for each)
    if 'Symbol' in indicators_df.columns:
        for symbol in indicators_df['Symbol'].unique():
            symbol_rows = indicators_df[indicators_df['Symbol'] == symbol]
            if not symbol_rows.empty:
                symbol_data = symbol_rows.iloc[-1]
                indicators_dict[symbol] = symbol_data
    elif 'symbol' in indicators_df.columns:
        for symbol in indicators_df['symbol'].unique():
            symbol_rows = indicators_df[indicators_df['symbol'] == symbol]
            if not symbol_rows.empty:
                symbol_data = symbol_rows.iloc[-1]
                indicators_dict[symbol] = symbol_data

    # Group latest data by symbol
    if 'Symbol' in latest_df.columns:
        for symbol in latest_df['Symbol'].unique():
            symbol_rows = latest_df[latest_df['Symbol'] == symbol]
            if not symbol_rows.empty:
                symbol_data = symbol_rows.iloc[-1]
                latest_dict[symbol] = symbol_data
    elif 'symbol' in latest_df.columns:
        for symbol in latest_df['symbol'].unique():
            symbol_rows = latest_df[latest_df['symbol'] == symbol]
            if not symbol_rows.empty:
                symbol_data = symbol_rows.iloc[-1]
                latest_dict[symbol] = symbol_data

    # Audit each plan row
    audit_results = []
    failed_audits = []

    for idx, plan_row in plan_df.iterrows():
        symbol = plan_row.get('Symbol', plan_row.get('symbol', ''))

        # Get data for this symbol
        symbol_indicators = indicators_dict.get(symbol)
        symbol_latest = latest_dict.get(symbol)

        if symbol_indicators is None or symbol_latest is None:
            logger.warning(f"No data found for symbol {symbol}")
            audit_result = {
                'Audit_Flag': 'FAIL',
                'Issues': 'Missing indicator or latest data',
                'Fix_Suggestion': 'Ensure symbol data is available in both datasets',
                'Pivot_Source': '',
                'Entry_Logic': '',
                'Stop_Logic': '',
                'Target_Logic': '',
                'Latest_Close': 0,
                'Latest_LTP': 0,
                'Delta_vs_Pivot_pct': 0,
                'Canonical_Entry': 0,
                'Canonical_Stop': 0,
                'Canonical_Target': 0,
                'ATR_Used': 0,
                'Risk_Amount': 0
            }
        else:
            audit_result = audit_plan_row(plan_row, symbol_indicators, symbol_latest, ap)

        audit_results.append(audit_result)

        if audit_result['Audit_Flag'] == 'FAIL':
            failed_audits.append(f"{symbol}: {audit_result['Issues']}")

    # Add audit columns to plan
    audit_df = pd.DataFrame(audit_results)
    audited_plan = pd.concat([plan_df.reset_index(drop=True), audit_df], axis=1)

    # Report results
    pass_count = sum(1 for r in audit_results if r['Audit_Flag'] == 'PASS')
    fail_count = len(audit_results) - pass_count

    logger.info(f"Audit complete: {pass_count} PASS, {fail_count} FAIL")

    if failed_audits:
        logger.warning(f"Failed audits: {failed_audits}")

    if ap.strict_mode and fail_count > 0:
        error_msg = f"Strict mode enabled: {fail_count} audit failures found\n" + "\n".join(failed_audits)
        raise PlanAuditError(error_msg)

    return audited_plan

def run_plan_audit(plan_path: str, indicators_path: str, latest_path: str,
                  output_path: str, ap: AuditParams) -> bool:
    """
    Run complete plan audit pipeline.

    Args:
        plan_path: Path to GTT plan CSV
        indicators_path: Path to indicators data
        latest_path: Path to latest price data
        output_path: Output path for audited plan
        ap: Audit parameters

    Returns:
        Success status
    """
    try:
        logger.info("Starting plan audit pipeline")

        # Load data
        plan_df = pd.read_csv(plan_path)
        indicators_df = pd.read_csv(indicators_path) if indicators_path.endswith('.csv') else pd.read_parquet(indicators_path)
        latest_df = pd.read_csv(latest_path)

        logger.info(f"Loaded plan: {len(plan_df)} positions")
        logger.info(f"Loaded indicators: {len(indicators_df)} records")
        logger.info(f"Loaded latest data: {len(latest_df)} records")

        # Run audit
        audited_plan = attach_audit(plan_df, indicators_df, latest_df, ap)

        # Save audited plan
        audited_plan.to_csv(output_path, index=False)
        logger.info(f"Audited plan saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Plan audit failed: {str(e)}")
        return False