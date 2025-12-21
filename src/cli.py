import argparse
from pathlib import Path
import json
import os
import pandas as pd
import logging
from datetime import datetime

from .data_io import load_dataset, validate_dataset, split_by_symbol
from .signals import compute_signals
from .scoring import compute_composite_score
from .select_strategy import select_best_strategy
from .gtt_sizing import build_gtt_plan
from .upstox_gtt import place_gtt_from_plan, place_gtt_with_retries
import csv, json, os
from .utils import load_config
from .data_fetch import fetch_nifty50_data
from .success_model import build_hierarchical_model
from .live_screener import run_live_screener
from .data_validation import (
    check_file_exists, load_metadata, validate_recency, validate_window,
    validate_symbols, validate_sorted, validate_cross_file_dates,
    load_excel_metadata, get_today_ist, ValidationError, summarize
)
from .wfo import walk_forward_optimization
from .multi_tf_excel import build_multi_tf_excel


def setup_validation_logging():
    """Setup logging for validation operations."""
    log_dir = Path('outputs/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file = log_dir / f'validation_{timestamp}.log'

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_indicators_file(indicators_path: str, skip_validation: bool = False) -> None:
    """
    Lightweight validation guard for indicators file.
    """
    if skip_validation:
        print("WARNING: Skipping data validation (--skip-validation used)")
        return

    try:
        today_ist = get_today_ist()
        main_meta = load_metadata(indicators_path)
        validate_recency(main_meta, today_ist, max_age_days=1)
        validate_window(main_meta, required_days=500)
        print(f"Data validation passed: {summarize(main_meta)}")
    except ValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        print("Re-run fetch-data; ensure API token valid; verify job schedule at EOD.")
        exit(2)


def cmd_screener(args):
    validate_indicators_file(args.path, getattr(args, 'skip_validation', False))
    df = load_dataset(args.path)
    ok, missing = validate_dataset(df)
    if not ok:
        print('Missing columns:', missing)
        return
    df = compute_signals(df)
    df['CompositeScore'] = compute_composite_score(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latest = df.sort_values('Date').groupby('Symbol').tail(1)
    latest.to_csv(out_path, index=False)
    print('Screener saved to', out_path)


def cmd_fetch_data(args):
    try:
        fetch_nifty50_data(days=args.days, out_path=args.out)
        print('Data fetch completed successfully.')
    except Exception as e:
        print(f'Error during data fetch: {e}')


def cmd_wfo(args):
    df = load_dataset(args.path)
    ok, missing = validate_dataset(df)
    if not ok:
        print('Missing columns:', missing)
        return
    cfg = load_config(args.config) if args.config else {}
    wfo_cfg = cfg.get('wfo', {})
    result = walk_forward_optimization(df, args.strategy, f'{args.strategy}_Flag', cfg,
                                       cycles=wfo_cfg.get('cycles', 8),
                                       mode=wfo_cfg.get('mode', 'rolling'),
                                       is_window=wfo_cfg.get('is_window', 252),
                                       oos_window=wfo_cfg.get('oos_window', 63))
    print(json.dumps(result, indent=2))


def cmd_backtest(args):
    validate_indicators_file(args.path, getattr(args, 'skip_validation', False))
    df = load_dataset(args.path)
    ok, missing = validate_dataset(df)
    if not ok:
        print('Missing columns:', missing)
        return
    df = compute_signals(df)
    # simple flat df for backtests
    strategies = {
        'SEPA': 'SEPA_Flag',
        'VCP': 'VCP_Flag',
        'Donchian': 'Donchian_Breakout',
        'MR': 'MR_Flag',
        'Squeeze': 'SqueezeBreakout_Flag',
        'AVWAP': 'AVWAP_Reclaim_Flag'
    }
    out = args.out
    Path(out).mkdir(parents=True, exist_ok=True)
    sel = select_best_strategy(df, strategies, {'risk': {}, 'backtest': {}}, out)
    print('Selection:', sel)


def cmd_select(args):
    cmd_backtest(args)


def cmd_gtt_plan(args):
    validate_indicators_file(args.path, getattr(args, 'skip_validation', False))
    df = load_dataset(args.path)
    ok, missing = validate_dataset(df)
    if not ok:
        print('Missing columns:', missing)
        return
    df = compute_signals(df)
    df['CompositeScore'] = compute_composite_score(df)
    latest = df.sort_values('Date').groupby('Symbol').tail(1)

    # Historic date filter
    if args.historic_date:
        historic_date = pd.to_datetime(args.historic_date)
        latest = df[df['Date'] == historic_date].groupby('Symbol').tail(1) if historic_date in df['Date'].values else latest

    # Fallback strategies
    fallback_strategies = args.fallback_strategies or ['Donchian_Breakout']
    candidates = None
    for strat in [args.strategy] + fallback_strategies:
        flag_col = f'{strat}_Flag'
        if flag_col in latest.columns:
            cand = latest[latest[flag_col] == 1].nlargest(args.top, 'CompositeScore').head(args.top)
            if not cand.empty:
                candidates = cand
                selected_strategy = strat
                break
    if candidates is None:
        candidates = latest.nlargest(args.top, 'CompositeScore')
        selected_strategy = 'CompositeScore'

    # Portfolio construction
    cfg = load_config(args.config) if args.config else {}
    if cfg.get('portfolio', {}).get('enable', False):
        candidates = construct_portfolio(candidates, cfg)

    instrument_map = {}  # placeholder
    # Build hierarchical success model for confidence metrics
    success_model = build_hierarchical_model(bt_root="outputs/backtests", today=pd.Timestamp.today())
    plan = build_gtt_plan(candidates, selected_strategy, {'risk': {}}, instrument_map, success_model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False)
    print('GTT plan saved to', out_path)


def cmd_gtt_place(args):
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print('Plan file not found:', plan_path)
        return
    import pandas as pd
    plan = pd.read_csv(plan_path)
    access_token = args.access_token or os.environ.get('UPSTOX_ACCESS_TOKEN')
    cfg = load_config(args.config) if args.config else {}
    # load instrument map if provided
    instrument_map = {}
    if args.instrument_map:
        import pandas as _pd
        im = _pd.read_csv(args.instrument_map)
        if 'Symbol' in im.columns and 'InstrumentToken' in im.columns:
            instrument_map = dict(zip(im['Symbol'], im['InstrumentToken']))

    responses = []
    # prepare plan rows list of dicts
    plan_rows = []
    for _, row in plan.iterrows():
        r = row.to_dict()
        if not r.get('InstrumentToken'):
            token = instrument_map.get(r.get('Symbol'))
            if token:
                r['InstrumentToken'] = token
        # ensure required keys exist for gtt sizing
        plan_rows.append(r)

    from .upstox_gtt import place_gtt_bulk
    responses = place_gtt_bulk(access_token, plan_rows, cfg, dry_run=args.dry_run, rate_limit_sleep=args.rate_limit, per_symbol_retries=args.retries, backoff=args.backoff, log_path=args.log_path)
    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)
    with open(outp / 'place_responses.json', 'w') as f:
        json.dump(responses, f, default=str, indent=2)
    print('Place responses saved to', outp / 'place_responses.json')


def cmd_gtt_get(args):
    access_token = args.access_token or os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not access_token:
        print('Access token required')
        return
    from .upstox_gtt import get_gtt_order
    response = get_gtt_order(access_token, args.id)
    print(json.dumps(response, indent=2))


def cmd_gtt_modify(args):
    access_token = args.access_token or os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not access_token:
        print('Access token required')
        return
    with open(args.file, 'r') as f:
        payload = json.load(f)
    from .upstox_gtt import modify_gtt_order
    response = modify_gtt_order(access_token, payload)
    print(json.dumps(response, indent=2))


def cmd_rsi_golden(args):
    df = load_dataset(args.path)
    ok, missing = validate_dataset(df)
    if not ok:
        print('Missing columns:', missing)
        return
    df = compute_signals(df)
    latest = df.sort_values('Date').groupby('Symbol').tail(1)
    summary = latest[['Symbol', 'Date', 'RSI14', 'RSI14_Status', 'EMA50', 'EMA200', 'GoldenBull_Flag', 'GoldenBear_Flag', 'GoldenBull_Date', 'GoldenBear_Date']].copy()
    summary['Date'] = summary['Date'].dt.strftime('%Y-%m-%d')
    summary = summary.sort_values('Symbol')
    summary.to_csv(args.out, index=False)
    print(f"RSI-Golden summary saved to {args.out}")


def cmd_final_excel(args):
    # Validate plan file recency
    if not getattr(args, 'skip_validation', False):
        try:
            today_ist = get_today_ist()
            plan_meta = load_metadata(args.plan)
            validate_recency(plan_meta, today_ist, max_age_days=1)
            print(f"Plan validation passed: {summarize(plan_meta)}")
        except ValidationError as e:
            print(f"PLAN VALIDATION FAILED: {e}")
            print("Ensure plan is generated from recent data.")
            exit(2)

    from .final_excel import run_final_excel
    success = run_final_excel(
        gtt_plan_path=args.plan,
        output_path=args.out,
        backtest_dir=getattr(args, 'backtest_dir', 'outputs/backtest')
    )

    if success:
        print(f"Final Excel written → {args.out}")
    else:
        print("❌ Final Excel generation failed!")
        exit(1)


def cmd_confidence_report(args):
    """Generate confidence report from GTT plan CSV."""
    df = pd.read_csv(args.path)
    if df.empty:
        print("No data in plan CSV")
        return
    
    # Select only confidence columns
    confidence_cols = ['Symbol', 'Strategy', 'DecisionConfidence', 'OOS_WinRate', 'OOS_ExpectancyR', 'Trades_OOS', 'Confidence_Reason']
    available_cols = [col for col in confidence_cols if col in df.columns]
    
    if not available_cols:
        print("No confidence columns found in plan")
        return
    
    confidence_df = df[available_cols]
    confidence_df.to_csv(args.out, index=False)
    print(f"Confidence report saved to {args.out}")


def cmd_validate_latest(args):
    """Validate recency and consistency of latest data and derived outputs."""
    logger = setup_validation_logging()

    try:
        today_ist = get_today_ist()

        # Check file existence
        check_file_exists(args.data)
        if args.screener:
            check_file_exists(args.screener)
        if args.plan:
            check_file_exists(args.plan)
        if args.excel:
            check_file_exists(args.excel)

        # Load main metadata
        main_meta = load_metadata(args.data)
        logger.info(f"Main indicators: {summarize(main_meta)}")

        # Validate main file
        validate_recency(main_meta, today_ist, max_age_days=1)
        validate_window(main_meta, required_days=500)
        validate_symbols(main_meta, expected_count=50)
        validate_sorted(main_meta)

        # Load derived files metadata
        derived_meta = []
        if args.screener:
            screener_meta = load_metadata(args.screener)
            derived_meta.append(screener_meta)
            logger.info(f"Screener: {summarize(screener_meta)}")

        if args.plan:
            plan_meta = load_metadata(args.plan)
            derived_meta.append(plan_meta)
            logger.info(f"Plan: {summarize(plan_meta)}")

        if args.excel:
            excel_meta = load_excel_metadata(args.excel)
            derived_meta.append(excel_meta)
            logger.info(f"Excel: {summarize(excel_meta)}")

        # Cross-file validation
        if derived_meta:
            validate_cross_file_dates(main_meta, derived_meta)

        # Success
        print(f"VALIDATION PASSED: latest={main_meta.latest_date.date()}, "
              f"rows={main_meta.rows_count}, symbols={len(main_meta.symbols_present)}, "
              f"window>={main_meta.trading_days_count} days")
        logger.info("Validation passed")

    except ValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        logger.error(f"Validation failed: {e}")
        exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        exit(2)


def cmd_fetch_and_validate(args):
    """Fetch data and immediately validate it."""
    logger = setup_validation_logging()

    try:
        # Fetch data
        print("Fetching data...")
        fetch_nifty50_data(days=args.days, out_path=args.out)
        print("Data fetch completed.")

        # Validate immediately
        today_ist = get_today_ist()
        main_meta = load_metadata(args.out)
        logger.info(f"Fetched data: {summarize(main_meta)}")

        validate_recency(main_meta, today_ist, max_age_days=args.max_age_days)
        validate_window(main_meta, required_days=args.required_days)
        validate_symbols(main_meta, expected_count=args.required_symbols)
        validate_sorted(main_meta)

        print(f"FETCH + VALIDATION OK: latest={main_meta.latest_date.date()}, "
              f"rows={main_meta.rows_count}, symbols={len(main_meta.symbols_present)}, "
              f"window>={main_meta.trading_days_count} days")
        logger.info("Fetch and validation passed")

    except ValidationError as e:
        print(f"FETCH + VALIDATION FAILED: {e}")
        print("Re-run fetch-data; ensure API token valid; verify job schedule at EOD.")
        logger.error(f"Fetch and validation failed: {e}")
        exit(2)
    except Exception as e:
        print(f"Unexpected error during fetch/validate: {e}")
        logger.error(f"Unexpected error: {e}")
        exit(2)


def cmd_multi_tf_excel(args):
    """Generate multi-timeframe Excel workbook."""
    # Validate input data first
    today_ist = get_today_ist()
    main_meta = load_metadata(args.path)
    validate_recency(main_meta, today_ist, max_age_days=1)
    validate_window(main_meta, required_days=500)
    validate_symbols(main_meta, expected_count=50)

    # Parse TFs
    tf_list = [tf.strip() for tf in args.tfs.split(',')]

    # Determine date range (last 500 days or so)
    end = main_meta.latest_date
    start = end - pd.Timedelta(days=500)

    # Get symbols
    symbols = list(main_meta.symbols_present)

    # Build workbook
    build_multi_tf_excel(symbols, tf_list, start, end, args.out)


def cmd_live_screener(args):
    """Run the live NIFTY 50 stock screener."""
    run_live_screener()


def main():
    parser = argparse.ArgumentParser('nifty-gtt-swing')
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('wfo', help='Walk-Forward Optimization')
    p.add_argument('--path', required=True)
    p.add_argument('--strategy', required=True)
    p.add_argument('--config', default=None)
    p.set_defaults(func=cmd_wfo)

    p = sub.add_parser('screener')
    p.add_argument('--path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--skip-validation', action='store_true', help='Skip data validation checks')
    p.set_defaults(func=cmd_screener)

    p = sub.add_parser('fetch_data')
    p.add_argument('--days', type=int, default=365)
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser('backtest')
    p.add_argument('--path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--skip-validation', action='store_true', help='Skip data validation checks')
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser('select')
    p.add_argument('--path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--skip-validation', action='store_true', help='Skip data validation checks')
    p.set_defaults(func=cmd_select)

    p = sub.add_parser('gtt-plan')
    p.add_argument('--path', required=True)
    p.add_argument('--strategy', default='Donchian')
    p.add_argument('--top', type=int, default=25)
    p.add_argument('--min-score', type=float, default=60.0)
    p.add_argument('--out', required=True)
    p.add_argument('--fallback-strategies', nargs='*', default=None, help='List of fallback strategies')
    p.add_argument('--historic-date', default=None, help='YYYY-MM-DD for historic plan')
    p.add_argument('--config', default=None)
    p.add_argument('--skip-validation', action='store_true', help='Skip data validation checks')
    p.set_defaults(func=cmd_gtt_plan)

    p = sub.add_parser('gtt-place')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dry-run', type=lambda s: s.lower() in ('true','1','yes'), default=True)
    p.add_argument('--access-token', default=None)
    p.add_argument('--instrument-map', default=None, help='CSV with Symbol,InstrumentToken')
    p.add_argument('--config', default=None, help='Path to config.yaml')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--backoff', type=float, default=1.0)
    p.add_argument('--rate-limit', type=float, default=0.5, help='seconds to sleep between API calls')
    p.add_argument('--log-path', default='outputs/logs/gtt_place.log')
    p.set_defaults(func=cmd_gtt_place)

    p = sub.add_parser('gtt-get')
    p.add_argument('--id', required=True, help='GTT order ID')
    p.add_argument('--access-token', default=None)
    p.set_defaults(func=cmd_gtt_get)

    p = sub.add_parser('gtt-modify')
    p.add_argument('--file', required=True, help='JSON file with modify payload')
    p.add_argument('--access-token', default=None)
    p.set_defaults(func=cmd_gtt_modify)

    p = sub.add_parser('final-excel')
    p.add_argument('--plan', required=True, help='Path to GTT plan CSV')
    p.add_argument('--out', required=True, help='Output Excel file path')
    p.add_argument('--backtest-dir', default='outputs/backtest', help='Backtest results directory')
    p.add_argument('--skip-validation', action='store_true', help='Skip data validation checks')
    p.set_defaults(func=cmd_final_excel)

    p = sub.add_parser('rsi-golden')
    p.add_argument('--path', required=True, help='Path to indicators file')
    p.add_argument('--out', required=True, help='Output CSV file path')
    p.set_defaults(func=cmd_rsi_golden)

    p = sub.add_parser('confidence-report')
    p.add_argument('--path', required=True, help='Path to GTT plan CSV')
    p.add_argument('--out', required=True, help='Output CSV file path')
    p.set_defaults(func=cmd_confidence_report)

    p = sub.add_parser('validate-latest', help='Validate recency and consistency of data files')
    p.add_argument('--data', required=True, help='Path to main indicators file')
    p.add_argument('--screener', help='Path to screener CSV')
    p.add_argument('--plan', help='Path to GTT plan CSV')
    p.add_argument('--excel', help='Path to final Excel file')
    p.set_defaults(func=cmd_validate_latest)

    p = sub.add_parser('fetch-and-validate', help='Fetch data and validate immediately')
    p.add_argument('--out', required=True, help='Output path for fetched data')
    p.add_argument('--days', type=int, default=500, help='Days to fetch')
    p.add_argument('--max-age-days', type=int, default=1, help='Max age in days')
    p.add_argument('--required-days', type=int, default=500, help='Required trading days')
    p.add_argument('--required-symbols', type=int, default=50, help='Required number of symbols')
    p.set_defaults(func=cmd_fetch_and_validate)

    p = sub.add_parser('multi-tf-excel', help='Generate multi-timeframe Excel workbook')
    p.add_argument('--path', required=True, help='Path to indicators file')
    p.add_argument('--tfs', required=True, help='Comma-separated list of timeframes')
    p.add_argument('--out', required=True, help='Output Excel path')
    p.set_defaults(func=cmd_multi_tf_excel)

    p = sub.add_parser('live-screener', help='Run live NIFTY 50 stock screener')
    p.set_defaults(func=cmd_live_screener)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
