import pandas as pd
from math import floor
import numpy as np
from .success_model import lookup_confidence
from .plan_audit import attach_audit, AuditParams


def round_tick(price: float, tick=0.05) -> float:
    return round(price / tick) * tick


def build_explanation(row: pd.Series) -> str:
    """
    Returns a concise explanation of selection, derivation, indicators, and strategy.
    """
    strategy = row.get('Strategy', 'Unknown')
    rsi = row.get('RSI14', 0)
    rsi_status = row.get('RSI14_Status', '')
    golden_bull = row.get('GoldenBull_Flag', 0)
    golden_bear = row.get('GoldenBear_Flag', 0)
    golden_bull_date = row.get('GoldenBull_Date', '')
    golden_bear_date = row.get('GoldenBear_Date', '')
    entry_type = row.get('ENTRY_trigger_type', '')
    entry_trigger = row.get('ENTRY_trigger_price', 0)
    stop_trigger = row.get('STOPLOSS_trigger_price', 0)
    target_trigger = row.get('TARGET_trigger_price', 0)
    atr = row.get('ATR14', 0)
    donchian_h20 = row.get('DonchianH20', 0)
    rvol20 = row.get('RVOL20', 0)
    close = row.get('Close', 0)
    ema20 = row.get('EMA20', 0)

    base = ""
    if strategy == 'MR':
        ema_dist = abs(close - ema20) / close * 100 if close != 0 else 0
        base = f"Selected via MR: uptrend (Close>EMA20>EMA50>EMA200), RSI14={rsi:.2f} → {rsi_status}, price near EMA20 ({ema_dist:.1f}%). "
    elif strategy == 'Donchian_Breakout':
        base = f"Selected via Donchian breakout: Close>{donchian_h20:.2f} with RVOL={rvol20:.1f}≥1.5. "
    elif strategy == 'SEPA':
        base = f"Selected via SEPA: Trend_OK (Close>EMA20>EMA50>EMA200), tight base (BandWidth low), breakout above pivot. "
    elif strategy == 'VCP':
        base = f"Selected via VCP: shrinking volatility (falling BandWidth), higher lows, breakout with RVOL spike. "
    else:  # CompositeScore fallback
        base = f"Selected via CompositeScore fallback (no active signals): ranked by score using RS_ROC20, RVOL20, Trend_OK, breakout flags. "

    gtt_part = f"Entry={entry_trigger:.2f} ({entry_type}), Stop={stop_trigger:.2f} (ATR×1.5), Target={target_trigger:.2f} (2R). "
    indicators = f"RSI(14) ≥50 (bullish bias), MACD cross↑ & above zero with rising histogram; breakout at DonchianH20 with RVOL. Entry/Stop/Target per strategy; audit PASS."
    if golden_bear == 1 and golden_bear_date:
        indicators += f" (on {golden_bear_date})"

    full = base + gtt_part + indicators
    return full[:280]  # truncate if too long


def context_from_row(row: pd.Series) -> dict:
    """Extract context bucket keys from a plan row."""
    # RSI14_Status
    rsi = row.get('RSI14', 50)
    if rsi <= 30:
        rsi_status = 'Oversold'
    elif rsi >= 70:
        rsi_status = 'Overbought'
    else:
        rsi_status = 'Neutral'

    # MACD_Regime
    macd_line = row.get('MACD_Line', 0)
    macd_regime = 'AboveZero' if macd_line > 0 else 'BelowZero'

    # MACD_Cross_Status
    macd_cross = row.get('MACD_CrossUp', False)
    macd_cross_status = 'CrossUp' if macd_cross else 'NoCross'

    # RVOL20_bucket
    rvol = row.get('RVOL20', 1.0)
    if rvol < 1.0:
        rvol_bucket = '<1.0'
    elif rvol <= 1.5:
        rvol_bucket = '1.0–1.5'
    else:
        rvol_bucket = '>1.5'

    # ATRpct_bucket
    atr = row.get('ATR14', 0)
    close = row.get('Close', row.get('ENTRY_trigger_price', 0))
    if close > 0:
        atr_pct = (atr / close) * 100
    else:
        atr_pct = 1.5  # default

    if atr_pct < 1.0:
        atr_bucket = '<1%'
    elif atr_pct <= 2.0:
        atr_bucket = '1–2%'
    else:
        atr_bucket = '>2%'

    return {
        'RSI14_Status': rsi_status,
        'MACD_Regime': macd_regime,
        'MACD_Cross_Status': macd_cross_status,
        'GoldenBull_Flag': int(row.get('GoldenBull_Flag', 0)),
        'GoldenBear_Flag': int(row.get('GoldenBear_Flag', 0)),
        'Trend_OK': int(row.get('Trend_OK', 0)),  # Assume available or default 0
        'RVOL20_bucket': rvol_bucket,
        'ATRpct_bucket': atr_bucket
    }


def compute_decision_confidence(row: pd.Series, model: pd.DataFrame, indicators_df: pd.DataFrame = None) -> dict:
    """
    Compute calibrated confidence with CI using hierarchical backoff.
    """
    strategy = row.get('Strategy', 'Unknown')
    
    # Get context from indicators data if available, otherwise from row
    if indicators_df is not None and not indicators_df.empty:
        symbol = row.get('Symbol', row.get('symbol', ''))
        if symbol:
            # Find the latest indicators for this symbol
            symbol_data = indicators_df[indicators_df['Symbol'] == symbol]
            if not symbol_data.empty:
                latest_row = symbol_data.sort_values('Date').iloc[-1]
                context = context_from_row(latest_row)
            else:
                context = context_from_row(row)
        else:
            context = context_from_row(row)
    else:
        context = context_from_row(row)
    
    # Add symbol and sector to context (placeholders)
    context['Symbol'] = row.get('Symbol', '')
    context['Sector'] = row.get('Sector', 'Unknown')
    
    # Use the new hierarchical lookup
    result = lookup_confidence(model, context, strategy)
    
    # Handle CompositeScore ensemble
    if strategy == 'CompositeScore':
        # Get available strategies in model
        available_strategies = model['Strategy'].unique()
        if len(available_strategies) > 0:
            # Weight by OOS trades and reliability
            weights = {}
            total_weight = 0
            
            for strat in available_strategies:
                strat_data = model[model['Strategy'] == strat]
                if not strat_data.empty:
                    # Weight by trades * reliability
                    weight = strat_data['Trades_OOS'].sum() * strat_data['Reliability'].mean()
                    weights[strat] = weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_conf = 0
                ensemble_ci_low = 0
                ensemble_ci_high = 0
                ensemble_winrate = 0
                ensemble_exp = 0
                total_trades = 0
                
                for strat, weight in weights.items():
                    strat_data = model[model['Strategy'] == strat]
                    w_norm = weight / total_weight
                    
                    strat_conf = strat_data['CalibratedWinRate'].mean()
                    strat_ci_low = strat_data['CI_low'].mean()
                    strat_ci_high = strat_data['CI_high'].mean()
                    strat_winrate = strat_data['OOS_WinRate_raw'].mean()
                    strat_exp = strat_data['OOS_ExpectancyR'].mean()
                    strat_trades = strat_data['Trades_OOS'].sum()
                    
                    ensemble_conf += w_norm * strat_conf
                    ensemble_ci_low += w_norm * strat_ci_low
                    ensemble_ci_high += w_norm * strat_ci_high
                    ensemble_winrate += w_norm * strat_winrate
                    ensemble_exp += w_norm * strat_exp
                    total_trades += strat_trades
                
                ensemble_conf = np.clip(ensemble_conf, 0.05, 0.95)
                ensemble_ci_low = np.clip(ensemble_ci_low, 0.01, ensemble_conf - 0.01)
                ensemble_ci_high = np.clip(ensemble_ci_high, ensemble_conf + 0.01, 0.99)
                
                return {
                    'DecisionConfidence': ensemble_conf,
                    'CI_low': ensemble_ci_low,
                    'CI_high': ensemble_ci_high,
                    'OOS_WinRate': ensemble_winrate,
                    'OOS_ExpectancyR': ensemble_exp,
                    'Trades_OOS': int(total_trades),
                    'CoverageNote': result.get('CoverageNote', 'Ensemble'),
                    'Confidence_Reason': f"Composite ensemble of {len(weights)} strategies (weighted by OOS trades×reliability); calibrated={ensemble_conf:.2%}, CI=[{ensemble_ci_low:.2%}, {ensemble_ci_high:.2%}]; total OOS trades={total_trades}."
                }
    
    # Regular strategy: use lookup result
    confidence = result.get('DecisionConfidence', 0.5)
    reliability = result.get('Reliability', 1.0)
    wfo_efficiency = result.get('WFO_Efficiency', 1.0)
    
    # Apply reliability and WFO efficiency
    confidence = confidence * reliability * wfo_efficiency
    
    ci_low = result.get('CI_low', max(0.01, confidence - 0.1))
    ci_high = result.get('CI_high', min(0.99, confidence + 0.1))
    
    # Clip confidence to [0.05, 0.95]
    confidence = np.clip(confidence, 0.05, 0.95)
    
    # Build detailed reason string
    reason_parts = [
        f"Strategy={strategy}",
        f"bucket={context['RSI14_Status']}/MACD={context['MACD_Regime']}+{context['MACD_Cross_Status']}/GBull={context['GoldenBull_Flag']}/GBear={context['GoldenBear_Flag']}/RVOL={context['RVOL20_bucket']}/ATR={context['ATRpct_bucket']}",
        f"OOS trades={result.get('Trades_OOS', 0)}",
        f"win={result.get('OOS_WinRate', 0.5):.2%}",
        f"expR={result.get('OOS_ExpectancyR', 0):.2f}",
        f"CI=[{ci_low:.2%}, {ci_high:.2%}]",
        f"calibrated={confidence:.2%}",
        f"pooling={result.get('CoverageNote', 'Unknown')}"
    ]
    
    return {
        'DecisionConfidence': confidence,
        'CI_low': ci_low,
        'CI_high': ci_high,
        'OOS_WinRate': result.get('OOS_WinRate', 0.5),
        'OOS_ExpectancyR': result.get('OOS_ExpectancyR', 0.0),
        'Trades_OOS': result.get('Trades_OOS', 0),
        'CoverageNote': result.get('CoverageNote', 'Unknown'),
        'Confidence_Reason': "; ".join(reason_parts)
    }


def build_gtt_plan(latest_df: pd.DataFrame, strategy_name: str, cfg: dict, instrument_map: dict, success_model: pd.DataFrame = None, indicators_df: pd.DataFrame = None, audit_params: AuditParams = None) -> pd.DataFrame:
    """Build GTT plan for candidates in latest_df (must include ATR14, DonchianH20, EMA20 etc.)"""
    rows = []
    risk_cfg = cfg.get('risk', {})
    stop_mult = risk_cfg.get('stop_multiple_atr', 1.5)
    equity = risk_cfg.get('equity', 100000)
    risk_pct = risk_cfg.get('risk_per_trade_pct', 1.0) / 100.0

    for _, r in latest_df.iterrows():
        sym = r['Symbol'] if 'Symbol' in r else r.get('Stock', None)
        if pd.isna(sym):
            continue
        
        # Determine entry type and trigger price based on strategy
        if strategy_name == 'AVWAP':
            # AVWAP strategy: entry based on AVWAP60 reclaim
            avwap60 = float(r.get('AVWAP60', r.get('Close', 0)))
            close = float(r.get('Close', 0))
            if close >= avwap60:
                # Already above AVWAP60, use current price as entry
                entry_trigger = close
                entry_type = 'ABOVE'
            else:
                # Below AVWAP60, wait for reclaim
                entry_trigger = avwap60
                entry_type = 'ABOVE'
        elif strategy_name in ('Donchian_Breakout', 'VCP_Flag', 'SEPA_Flag'):
            entry_type = 'ABOVE'
            entry_trigger = float(r.get('DonchianH20', r.get('Close', 0)))
        else:
            # Default for MR and other strategies
            entry_type = 'BELOW'
            entry_trigger = float(r.get('EMA20', r.get('Close', 0)))
        
        atr = float(r.get('ATR14', 0.0))
        stop_trigger = entry_trigger - stop_mult * atr
        R = ( (entry_trigger + 2*(entry_trigger-stop_trigger)) - entry_trigger ) / (entry_trigger - stop_trigger) if (entry_trigger - stop_trigger)!=0 else 0
        # qty sizing
        risk_amount = equity * risk_pct
        qty = floor(risk_amount / max(1e-6, (entry_trigger - stop_trigger)))
        instrument = instrument_map.get(sym, '')
        # Determine entry logic explanation
        if strategy_name == 'AVWAP':
            avwap60 = float(r.get('AVWAP60', 0))
            close = float(r.get('Close', 0))
            if close >= avwap60:
                entry_logic = f"CMP Entry: Price ₹{close:.2f} already above AVWAP60 ₹{avwap60:.2f}"
            else:
                entry_logic = f"AVWAP Reclaim: Wait for price above ₹{avwap60:.2f} (current ₹{close:.2f})"
        elif strategy_name in ('Donchian_Breakout', 'VCP_Flag', 'SEPA_Flag'):
            entry_logic = f"Breakout Entry: Above {r.get('DonchianH20', 0):.2f} with momentum"
        else:
            entry_logic = f"Pullback Entry: Below EMA20 {r.get('EMA20', 0):.2f}"
        
        rows.append({
            'Date': pd.Timestamp.now().date(),
            'Symbol': sym,
            'Strategy': strategy_name,
            'InstrumentToken': instrument,
            'Qty': qty,
            'ENTRY_trigger_type': entry_type,
            'ENTRY_trigger_price': round(entry_trigger, 2),
            'STOPLOSS_trigger_price': round(stop_trigger, 2),
            'TARGET_trigger_price': round(entry_trigger + 2*(entry_trigger - stop_trigger), 2),
            'R': round(R, 2),
            'ATR14': round(atr, 3),
            'RSI14': round(r.get('RSI14', 0), 2),
            'RSI_Above50': r.get('RSI_Above50', False),
            'RSI_Overbought': r.get('RSI_Overbought', False),
            'MACD_Line': round(r.get('MACD_Line', 0), 4),
            'MACD_Signal': round(r.get('MACD_Signal', 0), 4),
            'MACD_Hist': round(r.get('MACD_Hist', 0), 4),
            'MACD_CrossUp': r.get('MACD_CrossUp', False),
            'MACD_AboveZero': r.get('MACD_AboveZero', False),
            'RSI_MACD_Confirm_D': r.get('RSI_MACD_Confirm_D', False),
            'RSI_MACD_Confirmations_OK': r.get('RSI_MACD_Confirm_OK', False),
            'RSI_MACD_Notes': 'RSI≥50; MACD↑ & >0; Hist↑' if r.get('RSI_MACD_Confirm_OK', False) else 'RSI/MACD fail',
            'RSI14_Status': r.get('RSI14_Status', ''),
            'GoldenBull_Flag': r.get('GoldenBull_Flag', 0),
            'GoldenBear_Flag': r.get('GoldenBear_Flag', 0),
            'GoldenBull_Date': r.get('GoldenBull_Date', ''),
            'GoldenBear_Date': r.get('GoldenBear_Date', ''),
            'Trend_OK': r.get('Trend_OK', 0),
            'RS_Leader_Flag': r.get('RS_Leader_Flag', 0),
            'AVWAP60': round(r.get('AVWAP60', 0), 2),
            'EMA20': round(r.get('EMA20', 0), 2),
            'EMA50': round(r.get('EMA50', 0), 2),
            'Close': round(r.get('Close', 0), 2),
            'Entry_Logic': entry_logic,
            'Stop_Logic': f"EMA50-based stop: ₹{round(r.get('EMA50', entry_trigger - stop_mult * atr), 2)}",
            'Target_Logic': f"2R target from entry",
            'Notes': ''
        })
    df = pd.DataFrame(rows)
    df['Explanation'] = df.apply(build_explanation, axis=1)

    # Attach confidence metrics if model provided
    if success_model is not None and not success_model.empty:
        confidence_data = df.apply(lambda row: compute_decision_confidence(row, success_model, indicators_df), axis=1, result_type='expand')
        df = pd.concat([df, confidence_data], axis=1)
    else:
        # Default values when no model
        df['DecisionConfidence'] = 0.5
        df['CI_low'] = 0.4
        df['CI_high'] = 0.6
        df['OOS_WinRate'] = 0.5
        df['OOS_ExpectancyR'] = 0.0
        df['Trades_OOS'] = 0
        df['CoverageNote'] = 'NoModel'
        df['Confidence_Reason'] = 'No OOS data available'

    # Ensemble voting: require at least 2 of 4 modules (SEPA/VCP, Donchian, BBKC_Squeeze, TS_Momentum)
    def ensemble_vote_from_latest(lat_row: pd.Series) -> (int, list):
        sepa_vcp = bool(lat_row.get('SEPA_Flag', 0)) or bool(lat_row.get('VCP_Flag', 0))
        donchian = bool(lat_row.get('Donchian_Breakout', 0))
        squeeze = bool(lat_row.get('BBKC_Squeeze_Flag', 0)) or bool(lat_row.get('SqueezeBreakout_Flag', 0))
        tsm = bool(lat_row.get('TS_Momentum_Flag', 0))
        modules = [ ('SEPA_VCP', sepa_vcp), ('Donchian', donchian), ('Squeeze', squeeze), ('TSM', tsm) ]
        count = sum(1 for _m, v in modules if v)
        return count, [k for k,v in modules if v]

    # Map back to latest_df (we expect latest_df to contain indicator flags)
    votes = []
    for _, lr in latest_df.iterrows():
        c, passed = ensemble_vote_from_latest(lr)
        votes.append({'Ensemble_Count': c, 'Ensemble_PassedModules': ','.join(passed)})

    if votes:
        votes_df = pd.DataFrame(votes)
        df = pd.concat([df.reset_index(drop=True), votes_df.reset_index(drop=True)], axis=1)
    else:
        df['Ensemble_Count'] = 0
        df['Ensemble_PassedModules'] = ''

    # Decision gates and final PlaceGTT flag
    decision_threshold = cfg.get('decision', {}).get('min_confidence', 0.7)
    whitelist = cfg.get('decision', {}).get('whitelist', [])
    df['PlaceGTT'] = False
    df['NoTradeReason'] = ''

    for idx, row in df.iterrows():
        reasons = []
        # Ensemble vote gate
        if int(row.get('Ensemble_Count', 0)) < 2:
            reasons.append('EnsembleVote<2')

        # Regime & RS strength
        if row.get('Trend_OK', 0) != 1:
            reasons.append('RegimeFail')
        if row.get('RS_Leader_Flag', 0) != 1:
            reasons.append('RS_Not_Leader')

        # RSI/MACD confirmation
        if not row.get('RSI_MACD_Confirm_OK', False):
            reasons.append('RSI_MACD_Confirm_Fail')

        # Decision confidence
        conf = float(row.get('DecisionConfidence', 0.5)) if 'DecisionConfidence' in row.index else 0.5
        if conf < decision_threshold:
            reasons.append(f'LowConfidence({conf:.2f})')

        # Whitelist check
        sym = row.get('Symbol', '')
        if whitelist and sym not in whitelist:
            reasons.append('NotInWhitelist')

        # Save intermediate reasons; audit checks applied after attach_audit
        df.at[idx, 'NoTradeReason'] = ';'.join(reasons) if reasons else ''

    # Attach audit results if indicators and audit params provided
    if indicators_df is not None and audit_params is not None:
        df = attach_audit(df, indicators_df, latest_df, audit_params)

    # Finalize PlaceGTT after audit
    for idx, row in df.iterrows():
        current_reasons = []
        if row.get('NoTradeReason'):
            current_reasons.extend([r for r in row.get('NoTradeReason').split(';') if r])

        # Audit flag enforcement
        audit_flag = row.get('Audit_Flag', 'FAIL') if 'Audit_Flag' in row.index else 'FAIL'
        if audit_flag != 'PASS':
            current_reasons.append(f'Audit_{audit_flag}')

        # Freshness check
        max_age = audit_params.max_age_days if audit_params is not None else 1
        if 'Date' in row.index:
            try:
                age = (pd.Timestamp.now().date() - pd.to_datetime(row['Date']).date()).days
                if age > max_age:
                    current_reasons.append(f'DataStale({age}d)')
            except Exception:
                pass

        # Final decision
        if not current_reasons:
            df.at[idx, 'PlaceGTT'] = True
            df.at[idx, 'NoTradeReason'] = ''
        else:
            df.at[idx, 'PlaceGTT'] = False
            df.at[idx, 'NoTradeReason'] = ';'.join(current_reasons)

    return df


def build_upstox_payload(row: dict, cfg: dict) -> dict:
    """Build Upstox GTT payload for a single plan row.

    row: a dict/Series with keys: Symbol, InstrumentToken, Qty, ENTRY_trigger_type, ENTRY_trigger_price,
         STOPLOSS_trigger_price, TARGET_trigger_price
    cfg: configuration dict (expects gtt.default_product and trailing flags)
    """
    product = cfg.get('gtt', {}).get('default_product', 'D')
    trailing = cfg.get('gtt', {}).get('trailing_sl_enable', False)
    trailing_gap = cfg.get('gtt', {}).get('trailing_gap', None)

    rules = []
    # ENTRY
    rules.append({
        'strategy': 'ENTRY',
        'trigger_type': row.get('ENTRY_trigger_type', 'ABOVE'),
        'trigger_price': float(row.get('ENTRY_trigger_price', 0.0))
    })
    # TARGET
    rules.append({
        'strategy': 'TARGET',
        'trigger_type': 'IMMEDIATE',
        'trigger_price': float(row.get('TARGET_trigger_price', 0.0))
    })
    # STOPLOSS
    stop_rule = {
        'strategy': 'STOPLOSS',
        'trigger_type': 'IMMEDIATE',
        'trigger_price': float(row.get('STOPLOSS_trigger_price', 0.0))
    }
    if trailing and trailing_gap is not None:
        stop_rule['trailing_gap'] = float(trailing_gap)
    rules.append(stop_rule)

    payload = {
        'type': 'MULTIPLE',
        'quantity': int(row.get('Qty', 0)),
        'product': product,
        'instrument_token': row.get('InstrumentToken', ''),
        'transaction_type': 'BUY',
        'rules': rules
    }
    return payload


def validate_upstox_payload(payload: dict) -> (bool, list):
    """Validate minimal required fields for an Upstox GTT payload.

    Returns (ok, errors)
    """
    errs = []
    if not payload.get('instrument_token'):
        errs.append('missing instrument_token')
    if not payload.get('quantity') or int(payload.get('quantity', 0)) <= 0:
        errs.append('invalid quantity')
    if not payload.get('product'):
        errs.append('missing product')
    rules = payload.get('rules', [])
    if not isinstance(rules, list) or len(rules) < 2:
        errs.append('rules must be a list with at least ENTRY and STOP/TARGET')
    else:
        for r in rules:
            if 'trigger_type' not in r or 'trigger_price' not in r:
                errs.append('each rule requires trigger_type and trigger_price')
                break
    return (len(errs) == 0, errs)
