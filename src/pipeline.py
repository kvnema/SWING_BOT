"""
Main Pipeline Orchestrator
Coordinates all components: data fetching, indicators, strategies, regime filters,
portfolio construction, and GTT payload generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .data_fetch import fetch_single_instrument, fetch_nifty50_data
from .data_io import load_dataset
from .indicators import compute_all_indicators
from .patterns import detect_weekly_patterns
from .anchors import get_event_anchors, compute_event_anchored_avwap
from .regime import get_composite_regime
from .strategies.donchian import signal as donchian_signal
from .strategies.squeeze import signal as squeeze_signal
from .strategies.avwap import signal as avwap_signal
from .strategies.mr import signal as mr_signal
from .strategies.sepa_vcp import signal as sepa_vcp_signal
from .rr_gate import RRGate
from .portfolio import PortfolioConstructor
from .upstox_gtt import place_gtt_order_payload


class SWINGPipeline:
    """
    Main orchestrator for the SWING_BOT multi-strategy system
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.rr_gate = RRGate(
            min_rr_ratio=config.get('rr_gate', {}).get('min_ratio', 1.5),
            max_rr_ratio=config.get('rr_gate', {}).get('max_ratio', 4.0),
            max_risk_pct=config.get('rr_gate', {}).get('max_risk_pct', 0.02)
        )

        self.portfolio_constructor = PortfolioConstructor(
            max_weight=config.get('portfolio', {}).get('max_weight', 0.05),
            max_sector_weight=config.get('portfolio', {}).get('max_sector_weight', 0.20),
            min_sharpe=config.get('portfolio', {}).get('min_sharpe', 0.5)
        )

    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch stock and index data

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Tuple of (stocks_df, index_df)
        """
        self.logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")

        # Calculate days
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = (end - start).days

        # Fetch stock data using existing function
        try:
            # Use fetch_nifty50_data but modify to return data instead of saving
            # For now, load from existing data files
            stocks_df = load_dataset('data/nifty50_data.csv')
            if stocks_df.empty:
                self.logger.warning("No stock data available, running fetch_nifty50_data")
                fetch_nifty50_data(days=days)
                stocks_df = load_dataset('data/nifty50_data.csv')

        except Exception as e:
            self.logger.warning(f"Failed to fetch stock data: {e}")
            stocks_df = pd.DataFrame()

        # Fetch index data (NIFTY 50)
        try:
            index_df = load_dataset('data/nifty50_data_today.csv')
            if index_df.empty:
                # Try to get NIFTY data from stocks data
                nifty_data = stocks_df[stocks_df['Symbol'].str.contains('NIFTY', case=False)]
                if not nifty_data.empty:
                    index_df = nifty_data.copy()
                else:
                    index_df = pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Failed to fetch index data: {e}")
            index_df = pd.DataFrame()

        return stocks_df, index_df

    def compute_indicators(self, stocks_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators

        Args:
            stocks_df: Stock data
            index_df: Index data

        Returns:
            DataFrame with indicators
        """
        self.logger.info("Computing technical indicators")

        # Group by symbol and compute indicators
        indicator_frames = []

        for symbol in stocks_df['Symbol'].unique():
            symbol_data = stocks_df[stocks_df['Symbol'] == symbol].copy()

            if symbol_data.empty:
                continue

            try:
                # Compute indicators with benchmark data
                symbol_indicators = compute_all_indicators(symbol_data, index_df)
                indicator_frames.append(symbol_indicators)

            except Exception as e:
                self.logger.warning(f"Failed to compute indicators for {symbol}: {e}")

        if indicator_frames:
            return pd.concat(indicator_frames, ignore_index=True)
        else:
            return pd.DataFrame()

    def generate_strategy_signals(self, indicators_df: pd.DataFrame,
                                benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate signals from all strategies

        Args:
            indicators_df: DataFrame with indicators
            benchmark_df: Benchmark data for RS calculations

        Returns:
            DataFrame with all strategy signals
        """
        self.logger.info("Generating strategy signals")

        signals_df = indicators_df.copy()

        # Generate signals for each strategy
        try:
            # Donchian
            donchian_signals = indicators_df.groupby('Symbol').apply(
                lambda x: donchian_signal(x.assign(Symbol=x.name), benchmark_df)
            ).reset_index(drop=True)
            signals_df = signals_df.merge(donchian_signals, on=['Symbol', 'Date'], how='left')

        except Exception as e:
            self.logger.warning(f"Failed to generate Donchian signals: {e}")

        try:
            # Squeeze
            squeeze_signals = indicators_df.groupby('Symbol').apply(
                lambda x: squeeze_signal(x.assign(Symbol=x.name), benchmark_df)
            ).reset_index(drop=True)
            signals_df = signals_df.merge(squeeze_signals, on=['Symbol', 'Date'], how='left')

        except Exception as e:
            self.logger.warning(f"Failed to generate Squeeze signals: {e}")

        try:
            # AVWAP
            avwap_signals = indicators_df.groupby('Symbol').apply(
                lambda x: avwap_signal(x.assign(Symbol=x.name), benchmark_df)
            ).reset_index(drop=True)
            signals_df = signals_df.merge(avwap_signals, on=['Symbol', 'Date'], how='left')

        except Exception as e:
            self.logger.warning(f"Failed to generate AVWAP signals: {e}")

        try:
            # Mean Reversion
            mr_signals = indicators_df.groupby('Symbol').apply(
                lambda x: mr_signal(x.assign(Symbol=x.name), benchmark_df)
            ).reset_index(drop=True)
            signals_df = signals_df.merge(mr_signals, on=['Symbol', 'Date'], how='left')

        except Exception as e:
            self.logger.warning(f"Failed to generate MR signals: {e}")

        try:
            # SEPA/VCP
            sepa_vcp_signals = indicators_df.groupby('Symbol').apply(
                lambda x: sepa_vcp_signal(x.assign(Symbol=x.name), benchmark_df)
            ).reset_index(drop=True)
            signals_df = signals_df.merge(sepa_vcp_signals, on=['Symbol', 'Date'], how='left')

        except Exception as e:
            self.logger.warning(f"Failed to generate SEPA/VCP signals: {e}")

        return signals_df

    def apply_regime_filter(self, signals_df: pd.DataFrame, index_df: pd.DataFrame,
                          stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime and breadth filters

        Args:
            signals_df: Strategy signals
            index_df: Index data
            stocks_df: Stock data for breadth calculation

        Returns:
            Filtered signals
        """
        self.logger.info("Applying regime and breadth filters")

        try:
            # Compute regime indicators
            regime_df = get_composite_regime(index_df, stocks_df)

            # Apply regime filter to signals
            from .regime import apply_regime_filter
            filtered_signals = apply_regime_filter(signals_df, regime_df)

            return filtered_signals

        except Exception as e:
            self.logger.warning(f"Failed to apply regime filter: {e}")
            return signals_df

    def apply_rr_gate(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk-reward gate to signals

        Args:
            signals_df: Strategy signals

        Returns:
            RR-filtered signals
        """
        self.logger.info("Applying risk-reward gate")

        try:
            # Get current signals
            signal_cols = [col for col in signals_df.columns if col.endswith('_Flag') or col.endswith('_Signal')]

            rr_filtered_frames = []

            for symbol in signals_df['Symbol'].unique():
                symbol_data = signals_df[signals_df['Symbol'] == symbol].copy()

                if symbol_data.empty:
                    continue

                # Apply RR gate to each strategy
                for strategy_col in signal_cols:
                    if strategy_col in symbol_data.columns:
                        strategy_name = strategy_col.replace('_Flag', '').replace('_Signal', '')

                        # Get signals for this strategy
                        strategy_signals = symbol_data[symbol_data[strategy_col] == 1].copy()

                        if not strategy_signals.empty:
                            # Apply RR gate
                            rr_signals = self.rr_gate.apply_rr_gate(
                                strategy_signals, strategy_name
                            )

                            # Update the original data
                            symbol_data.loc[
                                symbol_data[strategy_col] == 1,
                                f'{strategy_col}_rr_gate'
                            ] = rr_signals['rr_gate_final']

                rr_filtered_frames.append(symbol_data)

            if rr_filtered_frames:
                return pd.concat(rr_filtered_frames, ignore_index=True)
            else:
                return signals_df

        except Exception as e:
            self.logger.warning(f"Failed to apply RR gate: {e}")
            return signals_df

    def construct_portfolio(self, signals_df: pd.DataFrame, price_df: pd.DataFrame,
                          benchmark_returns: pd.Series) -> Dict:
        """
        Construct portfolio from signals

        Args:
            signals_df: Filtered signals
            price_df: Price data
            benchmark_returns: Benchmark returns

        Returns:
            Portfolio construction result
        """
        self.logger.info("Constructing portfolio")

        try:
            portfolio_result = self.portfolio_constructor.construct_portfolio(
                signals_df, price_df, benchmark_returns
            )

            return portfolio_result

        except Exception as e:
            self.logger.warning(f"Failed to construct portfolio: {e}")
            return {}

    def generate_gtt_payloads(self, portfolio_result: Dict, signals_df: pd.DataFrame) -> List[Dict]:
        """
        Generate GTT payloads for portfolio positions

        Args:
            portfolio_result: Portfolio construction result
            signals_df: Signal data with entry details

        Returns:
            List of GTT payloads
        """
        self.logger.info("Generating GTT payloads")

        payloads = []

        try:
            weights = portfolio_result.get('weights', pd.Series())

            for symbol, weight in weights.items():
                # Get latest signal data for this symbol
                symbol_signals = signals_df[
                    (signals_df['Symbol'] == symbol) &
                    (signals_df['Date'] == signals_df['Date'].max())
                ]

                if symbol_signals.empty:
                    continue

                latest_signal = symbol_signals.iloc[0]

                # Get entry details (placeholder logic)
                entry_price = latest_signal.get('Close', 0)
                stop_loss = entry_price * 0.95  # 5% stop loss
                target_price = entry_price * 1.10  # 10% target
                quantity = max(1, int(weight * 100000 / entry_price))  # Placeholder calculation

                # Create basic GTT payload structure
                payload = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'quantity': quantity,
                    'weight': weight
                }

                payloads.append(payload)

        except Exception as e:
            self.logger.warning(f"Failed to generate GTT payloads: {e}")

        return payloads

    def run_pipeline(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Run the complete pipeline

        Args:
            symbols: List of symbols to process
            start_date: Start date
            end_date: End date

        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting SWING pipeline execution")

        results = {
            'execution_time': datetime.now(),
            'symbols_processed': len(symbols),
            'status': 'started'
        }

        try:
            # 1. Fetch data
            stocks_df, index_df = self.fetch_data(symbols, start_date, end_date)
            results['data_fetch_status'] = 'success' if not stocks_df.empty else 'failed'

            if stocks_df.empty:
                results['status'] = 'failed'
                results['error'] = 'No data fetched'
                return results

            # 2. Compute indicators
            indicators_df = self.compute_indicators(stocks_df, index_df)
            results['indicators_status'] = 'success' if not indicators_df.empty else 'failed'

            # 3. Generate strategy signals
            signals_df = self.generate_strategy_signals(indicators_df, index_df)
            results['signals_status'] = 'success'

            # 4. Apply regime filter
            regime_filtered_df = self.apply_regime_filter(signals_df, index_df, stocks_df)
            results['regime_filter_status'] = 'success'

            # 5. Apply RR gate
            rr_filtered_df = self.apply_rr_gate(regime_filtered_df)
            results['rr_gate_status'] = 'success'

            # 6. Construct portfolio
            benchmark_returns = index_df['Close'].pct_change().fillna(0)
            portfolio_result = self.construct_portfolio(rr_filtered_df, stocks_df, benchmark_returns)
            results['portfolio_status'] = 'success' if portfolio_result else 'failed'

            # 7. Generate GTT payloads
            gtt_payloads = self.generate_gtt_payloads(portfolio_result, rr_filtered_df)
            results['gtt_payloads'] = gtt_payloads
            results['gtt_count'] = len(gtt_payloads)

            results['status'] = 'completed'
            results['portfolio_result'] = portfolio_result

            self.logger.info(f"Pipeline completed successfully. Generated {len(gtt_payloads)} GTT payloads")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results


def run_swing_pipeline(config: Dict, symbols: List[str], start_date: str, end_date: str) -> Dict:
    """
    Convenience function to run the SWING pipeline

    Args:
        config: Pipeline configuration
        symbols: Symbols to process
        start_date: Start date
        end_date: End date

    Returns:
        Pipeline results
    """
    pipeline = SWINGPipeline(config)
    return pipeline.run_pipeline(symbols, start_date, end_date)