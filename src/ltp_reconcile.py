"""
SWING_BOT LTP Reconciliation Module

Fetches live quotes and reconciles GTT plan prices to ensure consistency with current market LTP.
"""

import os
import requests
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LTPParams:
    """Parameters for LTP reconciliation."""
    tick: float = 0.05                   # NSE EQ tick size
    max_entry_ltppct: float = 0.02       # 2% max diff allowed vs LTP
    adjust_mode: str = "soft"            # "soft"=auto-adjust, "strict"=fail-fast
    timeframe_base: str = "1d"           # ensure we reconcile against same TF as plan


def fetch_live_quotes(instrument_tokens: List[str], access_token: str, broker: str = 'upstox') -> pd.DataFrame:
    """
    Fetch live LTP for each instrument_token using the specified broker's API.

    Args:
        instrument_tokens: List of instrument tokens (NSE_EQ)
        access_token: API access token
        broker: Broker to use ('upstox' or 'icici')

    Returns:
        DataFrame: instrument_token, symbol, last_price (LTP), timestamp, ohlc
    """
    if not access_token:
        raise ValueError(f"{broker.upper()}_ACCESS_TOKEN not set")

    if not instrument_tokens:
        return pd.DataFrame(columns=['instrument_token', 'symbol', 'last_price', 'timestamp', 'ohlc'])

    if broker.lower() == 'upstox':
        return _fetch_upstox_quotes(instrument_tokens, access_token)
    elif broker.lower() == 'icici':
        return _fetch_icici_quotes(instrument_tokens, access_token)
    else:
        raise ValueError(f"Unsupported broker: {broker}")


def _fetch_upstox_quotes(instrument_tokens: List[str], access_token: str) -> pd.DataFrame:
    """Fetch live quotes from Upstox API."""
    # Upstox Full Market Quotes v2 endpoint
    base_url = "https://api.upstox.com/v2/market-quote/quotes"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    # Split into batches of 10 (API limit, matching data_fetch.py)
    batch_size = 10
    all_quotes = []

    for i in range(0, len(instrument_tokens), batch_size):
        batch_tokens = instrument_tokens[i:i + batch_size]
        instrument_keys = ",".join(batch_tokens)

        params = {
            'instrument_key': instrument_keys
        }

        try:
            print(f"DEBUG: Fetching quotes for {len(batch_tokens)} instruments: {batch_tokens[:2]}")
            logger.info("Fetching quotes for {} instruments...".format(len(batch_tokens)))
            logger.info("Instrument keys: {}".format(instrument_keys))
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            print(f"DEBUG: API response status: {response.status_code}")
            logger.info("API response status: {}".format(response.status_code))
            response.raise_for_status()

            data = response.json()
            print(f"DEBUG: API response data keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
            logger.info("API response data keys: {}".format(list(data.keys()) if isinstance(data, dict) else 'not dict'))

            if 'data' in data:
                print(f"DEBUG: Found data with {len(data['data'])} items")
                for instrument_key, quote_data in data['data'].items():
                    print(f"DEBUG: Processing {instrument_key}: {quote_data.get('last_price', 'no price')}")
                    quote = {
                        'instrument_token': instrument_key,
                        'symbol': quote_data.get('symbol', ''),
                        'last_price': quote_data.get('last_price', 0.0),
                        'timestamp': quote_data.get('timestamp', ''),
                        'ohlc': quote_data.get('ohlc', {})
                    }
                    all_quotes.append(quote)
            else:
                print(f"DEBUG: No 'data' key in response or data is empty")

            # Rate limiting
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"DEBUG: API request failed: {str(e)}")
            logger.error("Failed to fetch quotes for batch: {}".format(str(e)))
            # Continue with other batches

    df = pd.DataFrame(all_quotes)
    logger.info("Fetched {} live quotes".format(len(df)))
    return df


def _fetch_icici_quotes(instrument_tokens: List[str], access_token: str) -> pd.DataFrame:
    """Fetch live quotes from ICICI API."""
    from .icici_api import ICICIDirectAPI
    
    api = ICICIDirectAPI()
    all_quotes = []
    
    # ICICI API expects symbols, not instrument tokens
    # We need to convert instrument tokens to symbols
    from .data_fetch import load_instrument_keys
    instrument_keys = load_instrument_keys()
    token_to_symbol = {v: k for k, v in instrument_keys.items()}
    symbols = [token_to_symbol.get(token, token) for token in instrument_tokens]
    
    try:
        quotes_data = api.get_live_quotes(symbols)
        
        for i, quote in enumerate(quotes_data):
            # Normalize ICICI response to match Upstox format
            # Assume quotes_data is a list of quote dictionaries
            normalized_quote = {
                'instrument_token': instrument_tokens[i] if i < len(instrument_tokens) else '',
                'symbol': quote.get('symbol', symbols[i] if i < len(symbols) else ''),
                'last_price': quote.get('last_price', quote.get('ltp', 0.0)),
                'timestamp': quote.get('timestamp', ''),
                'ohlc': quote.get('ohlc', {})
            }
            all_quotes.append(normalized_quote)
            
    except Exception as e:
        logger.error(f"Failed to fetch ICICI quotes: {str(e)}")
    
    df = pd.DataFrame(all_quotes)
    logger.info("Fetched {} ICICI live quotes".format(len(df)))
    return df


def round_to_tick(price: float, tick: float = 0.05) -> float:
    """Round price to nearest tick size."""
    return round(price / tick) * tick


def compute_stop_target_from_entry(entry: float, strategy: str, atr_value: float = None) -> Tuple[float, float]:
    """
    Compute stop loss and target based on entry price and strategy.

    Args:
        entry: Entry price
        strategy: Strategy name (MR, Donchian, SEPA, VCP, Squeeze, AVWAP)
        atr_value: ATR value if available

    Returns:
        Tuple of (stop_loss, target)
    """
    if atr_value and atr_value > 0:
        # Use ATR-based stops for better risk management
        if strategy.upper() in ['MR', 'AVWAP']:
            # Mean reversion: tighter stops
            stop_distance = atr_value * 1.0
            target_distance = atr_value * 2.0
        else:
            # Breakout strategies: wider stops
            stop_distance = atr_value * 1.5
            target_distance = atr_value * 3.0
    else:
        # Fallback: percentage-based
        if strategy.upper() in ['MR', 'AVWAP']:
            stop_pct = 0.02  # 2%
            target_pct = 0.04  # 4%
        else:
            stop_pct = 0.03  # 3%
            target_pct = 0.06  # 6%

        stop_distance = entry * stop_pct
        target_distance = entry * target_pct

    stop_loss = round_to_tick(entry - stop_distance)
    target = round_to_tick(entry + target_distance)

    return stop_loss, target


def reconcile_entry_stop_target(plan_df: pd.DataFrame, quotes_df: pd.DataFrame, ap: LTPParams, token_to_symbol: dict = None) -> pd.DataFrame:
    """
    Reconcile plan prices with live LTP.

    Args:
        plan_df: Original GTT plan DataFrame
        quotes_df: Live quotes DataFrame
        ap: LTP reconciliation parameters

    Returns:
        Updated plan_df with reconciliation results
    """
    if plan_df.empty:
        return plan_df

    # If no live quotes available, return original plan with reconciliation flags set to SKIP
    if quotes_df.empty:
        logger.warning("No live quotes available, skipping LTP reconciliation")
        plan_df = plan_df.copy()
        plan_df['LTP_Reconciled'] = False
        plan_df['LTP_Delta_Pct'] = 0.0
        plan_df['Reconciliation_Audit'] = 'SKIP_NO_QUOTES'
        plan_df['Reconciliation_Issues'] = 'Live quotes not available'
        plan_df['Reconciliation_Fix'] = 'Ensure API token is valid for live quote fetching'
        return plan_df

    # Merge plan_df with quotes_df on Symbol
    merged_df = plan_df.merge(quotes_df, left_on='Symbol', right_on='symbol', how='left')

    # Initialize new columns
    reconciled_rows = []

    for idx, row in merged_df.iterrows():
        original_row = row.copy()

        # Extract prices
        entry_price = row.get('ENTRY_trigger_price', 0.0)
        stop_price = row.get('STOPLOSS_trigger_price', 0.0)
        target_price = row.get('TARGET_trigger_price', 0.0)
        ltp = row.get('last_price', 0.0)
        strategy = row.get('Strategy', 'UNKNOWN')
        atr_value = row.get('ATR', None)  # If available from indicators

        # Initialize reconciliation fields
        reconciled_entry = entry_price
        reconciled_stop = stop_price
        reconciled_target = target_price
        ltp_delta_pct = 0.0
        audit_flag = 'PASS'
        issues = ''
        fix_suggestion = ''

        # Check if we have LTP data
        if pd.isna(ltp) or ltp == 0.0:
            audit_flag = 'FAIL'
            issues = 'Quote fetch failed'
            fix_suggestion = 'Retry quote fetch or use last known LTP'
        else:
            # Compute LTP delta
            if entry_price > 0:
                ltp_delta_pct = abs(entry_price - ltp) / ltp

                if ltp_delta_pct <= ap.max_entry_ltppct:
                    # Within tolerance - keep original prices
                    reconciled_entry = entry_price
                    reconciled_stop = stop_price
                    reconciled_target = target_price
                else:
                    # Outside tolerance - apply reconciliation
                    if ap.adjust_mode == 'strict':
                        audit_flag = 'FAIL'
                        issues = 'Entry price {:.2%} from LTP (>{:.2%})'.format(ltp_delta_pct, ap.max_entry_ltppct)
                        fix_suggestion = 'Recompute from latest LTP/pivot'
                    else:  # soft mode
                        # Strategy-aware adjustment
                        if strategy.upper() in ['MR', 'AVWAP']:
                            # Mean reversion: snap to LTP or nearest pivot
                            reconciled_entry = round_to_tick(ltp)
                        else:
                            # Breakout: use LTP if above pivot, otherwise keep pivot
                            pivot_high = row.get('DonchianH20', entry_price)
                            if ltp >= pivot_high:
                                reconciled_entry = round_to_tick(ltp)
                            else:
                                reconciled_entry = round_to_tick(pivot_high)

                        # Recompute stop and target
                        reconciled_stop, reconciled_target = compute_stop_target_from_entry(
                            reconciled_entry, strategy, atr_value
                        )

                        issues = 'Adjusted entry from {:.2f} to {:.2f} ({:.2%} from LTP)'.format(entry_price, reconciled_entry, ltp_delta_pct)
                        fix_suggestion = 'Auto-adjusted within tolerance band'

        # Update row with reconciliation results
        original_row['LTP'] = ltp if not pd.isna(ltp) else 0.0
        original_row['ltp_delta_pct'] = ltp_delta_pct
        original_row['Reconciled_Entry'] = reconciled_entry
        original_row['Reconciled_Stop'] = reconciled_stop
        original_row['Reconciled_Target'] = reconciled_target
        original_row['Audit_Flag'] = audit_flag
        original_row['Issues'] = issues
        original_row['Fix_Suggestion'] = fix_suggestion

        reconciled_rows.append(original_row)

    return pd.DataFrame(reconciled_rows)


def write_reconciled(plan_df: pd.DataFrame, out_csv: str, out_report: str) -> None:
    """
    Write reconciled plan and summary report.

    Args:
        plan_df: Reconciled plan DataFrame
        out_csv: Output CSV path for reconciled plan
        out_report: Output path for reconciliation report
    """
    # Write reconciled plan
    plan_df.to_csv(out_csv, index=False)
    logger.info(f"Reconciled plan written to {out_csv}")

    # Generate summary report
    total_rows = len(plan_df)
    
    # Check if reconciliation was performed (columns exist) or skipped
    if 'LTP_Reconciled' in plan_df.columns and not plan_df['LTP_Reconciled'].any():
        # Reconciliation was skipped
        rows_adjusted = 0
        rows_pass = len(plan_df[plan_df.get('Reconciliation_Audit', '') == 'PASS'])
        rows_fail = len(plan_df[plan_df.get('Reconciliation_Audit', '') == 'FAIL'])
        avg_ltp_delta_pct = 0.0
        reconciliation_status = "SKIPPED - No live quotes available"
    else:
        # Reconciliation was performed
        rows_adjusted = len(plan_df[plan_df.get('ltp_delta_pct', 0) > 0.001])  # Adjusted if delta > 0.1%
        rows_pass = len(plan_df[plan_df.get('Audit_Flag', '') == 'PASS'])
        rows_fail = len(plan_df[plan_df.get('Audit_Flag', '') == 'FAIL'])
        avg_ltp_delta_pct = plan_df.get('ltp_delta_pct', pd.Series([0.0])).mean()
        reconciliation_status = "COMPLETED"

    # Generate summary report
    if 'LTP_Reconciled' in plan_df.columns and not plan_df['LTP_Reconciled'].any():
        # Reconciliation was skipped - generate minimal report
        report_content = f"""SWING_BOT LTP Reconciliation Report
{'='*50}

Status: {reconciliation_status}

Summary:
- Total positions: {total_rows}
- Reconciliation skipped due to unavailable live quotes
- Plan remains unchanged from audit

Note: Live quotes are required for LTP reconciliation. Ensure API token is valid for production use.
"""
    else:
        # Reconciliation was performed - generate full report
        report_content = f"""SWING_BOT LTP Reconciliation Report
{'='*50}

Status: {reconciliation_status}

Summary:
- Total positions: {total_rows}
- Rows adjusted: {rows_adjusted}
- Rows passing: {rows_pass}
- Rows failing: {rows_fail}
- Average LTP delta: {avg_ltp_delta_pct:.2%}

Failures by issue:
{plan_df[plan_df.get('Audit_Flag', '') == 'FAIL']['Issues'].value_counts().to_string()}

Top adjustments:
{plan_df.nlargest(5, 'ltp_delta_pct')[['Symbol', 'ltp_delta_pct', 'Issues']].to_string(index=False)}
"""

    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, 'w') as f:
        f.write(report_content)

    logger.info(f"Reconciliation report written to {out_report}")


def reconcile_plan(plan_csv: str, out_csv: str, out_report: str,
                  adjust_mode: str = "soft", max_entry_ltppct: float = 0.02, broker: str = 'upstox') -> pd.DataFrame:
    """
    Main function to reconcile a GTT plan with live LTP.

    Args:
        plan_csv: Input plan CSV path
        out_csv: Output reconciled plan CSV path
        out_report: Output reconciliation report path
        adjust_mode: "soft" or "strict"
        max_entry_ltppct: Maximum allowed LTP delta percentage
        broker: Broker to use for live quotes ('upstox' or 'icici')

    Returns:
        Reconciled plan DataFrame
    """
    logger.info("Starting LTP reconciliation for {}".format(plan_csv))

    # Load plan
    plan_df = pd.read_csv(plan_csv)
    logger.info("Loaded {} positions from {}".format(len(plan_df), plan_csv))

    # Extract instrument tokens
    instrument_tokens = plan_df['InstrumentToken'].dropna().unique().tolist()
    print(f"DEBUG: Extracted {len(instrument_tokens)} instrument tokens: {instrument_tokens[:3]}")

    # Get access token based on broker
    if broker.lower() == 'upstox':
        access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    elif broker.lower() == 'icici':
        access_token = os.environ.get('ICICI_ACCESS_TOKEN')
        if not access_token:
            logger.warning("ICICI_ACCESS_TOKEN not set, falling back to Upstox for LTP reconciliation")
            broker = 'upstox'  # Fallback
            access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    else:
        raise ValueError(f"Unsupported broker: {broker}")

    if not access_token:
        raise ValueError(f"Access token not available for {broker.upper()}")

    # Fetch live quotes directly using the broker-specific function
    quotes_df = fetch_live_quotes(instrument_tokens, access_token, broker)
    
    # Create reverse mapping from instrument token to symbol
    from .data_fetch import load_instrument_keys
    instrument_keys = load_instrument_keys()
    token_to_symbol = {v: k for k, v in instrument_keys.items()}

    # Setup reconciliation parameters
    params = LTPParams(
        adjust_mode=adjust_mode,
        max_entry_ltppct=max_entry_ltppct
    )

    # Reconcile prices
    reconciled_df = reconcile_entry_stop_target(plan_df, quotes_df, params, token_to_symbol)

    # Write outputs
    write_reconciled(reconciled_df, out_csv, out_report)

    logger.info("LTP reconciliation complete. {} positions processed.".format(len(reconciled_df)))

    return reconciled_df