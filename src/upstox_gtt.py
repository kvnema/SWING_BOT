import requests
import os
import time
import logging
from typing import Dict, Any, Optional
from .gtt_sizing import build_upstox_payload

BASE = 'https://api.upstox.com/v3'


def _headers(access_token: str) -> Dict[str, str]:
    return {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}


def _ensure_log_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def place_gtt_order(access_token: str, payload: Dict[str, Any], session: Optional[requests.Session] = None) -> Dict:
    url = BASE + '/order/gtt/place'
    s = session or requests
    resp = s.post(url, json=payload, headers=_headers(access_token))
    return {'status_code': resp.status_code, 'body': resp.json() if resp.content else {}}


def place_gtt_with_retries(access_token: str, payload: Dict[str, Any], dry_run: bool = False, retries: int = 3, backoff: float = 1.0, log_path: str = 'outputs/logs/gtt_place.log', session: Optional[requests.Session] = None) -> Dict:
    """Place GTT with retry/backoff and logging. Returns dict with status and body.

    dry_run: if True, returns payload without calling API.
    retries: number of attempts on transient errors (429, 5xx).
    backoff: base seconds to multiply by attempt index.
    log_path: path to write per-attempt logs.
    """
    if dry_run:
        return {'status_code': 0, 'body': payload}

    _ensure_log_dir(log_path)
    logger = logging.getLogger('nifty_gtt')
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    s = session or requests
    url = BASE + '/order/gtt/place'
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = s.post(url, json=payload, headers=_headers(access_token))
            status = resp.status_code
            body = resp.json() if resp.content else {}
            logger.info('Attempt %d status=%s symbol=%s', attempt, status, payload.get('instrument_token'))
            if status in (200, 201, 202):
                return {'status_code': status, 'body': body}
            # transient
            if status == 429 or status >= 500:
                sleep_for = backoff * attempt
                logger.warning('Transient status %s, sleeping %.1fs and retrying', status, sleep_for)
                time.sleep(sleep_for)
                continue
            # non-retriable
            logger.error('Non-retriable status %s body=%s', status, body)
            return {'status_code': status, 'body': body}
        except Exception as e:
            last_exc = e
            logger.exception('Exception on attempt %d: %s', attempt, str(e))
            time.sleep(backoff * attempt)
            continue

    # exhausted
    logger.error('Exhausted retries; last_exc=%s', str(last_exc))
    return {'status_code': 0, 'body': {'error': str(last_exc)}}


def _check_edis_for_symbol(symbol: str, cfg: Dict[str, Any]) -> bool:
    """Placeholder EDIS check: returns True if env var UPSTOX_EDIS_GRANTED=1 or cfg allows it.

    Real implementations should call Upstox API or check account settings.
    """
    if cfg and cfg.get('gtt', {}).get('allow_sell_without_edis', False):
        return True
    val = os.environ.get('UPSTOX_EDIS_GRANTED', '0')
    return val in ('1', 'true', 'True')


def place_gtt_bulk(access_token: str, plan_rows: list, cfg: Dict[str, Any], dry_run: bool = True, rate_limit_sleep: float = 0.5, per_symbol_retries: int = 3, backoff: float = 1.0, log_path: str = 'outputs/logs/gtt_place_bulk.log', session: Optional[requests.Session] = None) -> list:
    """Place multiple GTTs sequentially with rate limiting and per-symbol retry policy.

    plan_rows: list of dict-like rows (Symbol, InstrumentToken, Qty, ENTRY_trigger_type, ENTRY_trigger_price, STOPLOSS_trigger_price, TARGET_trigger_price, Strategy)
    Returns list of responses per row.
    """
    from .gtt_sizing import build_upstox_payload, validate_upstox_payload
    responses = []
    for r in plan_rows:
        # build payload
        payload = build_upstox_payload(r, cfg)
        ok, errs = validate_upstox_payload(payload)
        if not ok:
            responses.append({'Symbol': r.get('Symbol'), 'status_code': 0, 'error': errs})
            continue
        # EDIS check for SELL legs (if any) - our plan uses BUY GTTs mainly
        strat = r.get('Strategy', '')
        # if SELL appears in intended transaction type (not common here) check EDIS
        if r.get('transaction_type', 'BUY') == 'SELL':
            if not _check_edis_for_symbol(r.get('Symbol'), cfg):
                responses.append({'Symbol': r.get('Symbol'), 'status_code': 0, 'error': 'EDIS not granted for SELL'})
                continue

        # place with retries per symbol
        resp = place_gtt_with_retries(access_token, payload, dry_run=dry_run, retries=per_symbol_retries, backoff=backoff, log_path=log_path, session=session)
        responses.append({'Symbol': r.get('Symbol'), 'response': resp})
        # rate limit pacing
        time.sleep(rate_limit_sleep)
    return responses


def place_gtt_order_payload(access_token: str, payload: Dict[str, Any], dry_run: bool = False) -> Dict:
    """Backward-compatible: place payload (no retries)."""
    if dry_run:
        return {'status_code': 0, 'body': payload}
    return place_gtt_order(access_token, payload)


def place_gtt_from_plan(access_token: str, plan_row: Dict[str, Any], cfg: Dict[str, Any], dry_run: bool = True, retries: int = 3, backoff: float = 1.0, log_path: str = 'outputs/logs/gtt_place.log', session: Optional[requests.Session] = None) -> Dict:
    """Build payload from plan row and place GTT (supports retries and dry-run)."""
    row = dict(plan_row)
    payload = build_upstox_payload(row, cfg)
    return place_gtt_with_retries(access_token, payload, dry_run=dry_run, retries=retries, backoff=backoff, log_path=log_path, session=session)


def get_gtt_order(access_token: str, gtt_id: str) -> Dict:
    url = BASE + f'/order/gtt/{gtt_id}'
    resp = requests.get(url, headers=_headers(access_token))
    return {'status_code': resp.status_code, 'body': resp.json() if resp.content else {}}


def modify_gtt_order(access_token: str, payload: Dict[str, Any]) -> Dict:
    url = BASE + '/order/gtt/modify'
    resp = requests.post(url, json=payload, headers=_headers(access_token))
    return {'status_code': resp.status_code, 'body': resp.json() if resp.content else {}}


def place_gtt_order_multi(instrument_token: str, quantity: int, product: str, rules: list, transaction_type: str, access_token: str, tsl_gap: Optional[float] = None, dry_run: bool = False, retries: int = 3, backoff: float = 1.0, log_path: str = 'outputs/logs/gtt_place_multi.log') -> Dict:
    """Place multi-leg GTT order with ENTRY, STOPLOSS, TARGET rules.

    Args:
        instrument_token: Upstox instrument token
        quantity: Order quantity (1 for testing)
        product: 'D' for delivery
        rules: List of rule dicts [{'strategy': 'ENTRY', 'trigger_type': 'ABOVE', 'trigger_price': float}, ...]
        transaction_type: 'BUY' or 'SELL'
        access_token: Upstox access token
        tsl_gap: Optional trailing stop loss gap for STOPLOSS rule
        dry_run: If True, return payload without API call
        retries: Number of retry attempts
        backoff: Base backoff time
        log_path: Log file path

    Returns:
        Dict with status_code and body/order_id
    """
    payload = {
        "type": "MULTIPLE",
        "quantity": quantity,
        "product": product,
        "transaction_type": transaction_type,
        "instrument_token": instrument_token,
        "rules": rules
    }

    # Add TSL to STOPLOSS rule if specified
    if tsl_gap is not None:
        for rule in payload["rules"]:
            if rule.get("strategy") == "STOPLOSS":
                rule["trailing_gap"] = tsl_gap
                break

    if dry_run:
        return {'status_code': 0, 'body': payload, 'order_id': 'DRY_RUN'}

    _ensure_log_dir(log_path)
    logger = logging.getLogger('gtt_multi')
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    url = BASE + '/order/gtt/place'
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=_headers(access_token))
            status = resp.status_code
            body = resp.json() if resp.content else {}

            logger.info('Multi-leg GTT attempt %d status=%s symbol_token=%s', attempt, status, instrument_token)

            if status in (200, 201, 202):
                order_id = body.get('order_id', 'UNKNOWN')
                return {'status_code': status, 'body': body, 'order_id': order_id}

            # Handle duplicate GTT - try modify instead
            if status == 400 and 'duplicate' in str(body).lower():
                logger.info('Duplicate GTT detected, attempting modify')
                modify_resp = requests.post(BASE + '/order/gtt/modify', json=payload, headers=_headers(access_token))
                if modify_resp.status_code in (200, 201, 202):
                    modify_body = modify_resp.json() if modify_resp.content else {}
                    order_id = modify_body.get('order_id', 'MODIFIED')
                    return {'status_code': modify_resp.status_code, 'body': modify_body, 'order_id': order_id}
                else:
                    logger.warning('Modify also failed: %s', modify_resp.text)

            # Transient errors
            if status == 429 or status >= 500:
                sleep_for = backoff * attempt
                logger.warning('Transient status %s, sleeping %.1fs and retrying', status, sleep_for)
                time.sleep(sleep_for)
                continue

            # Non-retriable
            logger.error('Non-retriable status %s body=%s', status, body)
            return {'status_code': status, 'body': body, 'order_id': None}

        except Exception as e:
            last_exc = e
            logger.exception('Exception on attempt %d: %s', attempt, str(e))
            time.sleep(backoff * attempt)
            continue

    # Exhausted retries
    logger.error('Exhausted retries for multi-leg GTT; last_exc=%s', str(last_exc))
    return {'status_code': 0, 'body': {'error': str(last_exc)}, 'order_id': None}
