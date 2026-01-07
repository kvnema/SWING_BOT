"""
Enhanced Risk Management for SWING_BOT
Implements volatility-adjusted sizing, circuit breakers, and advanced diversification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class VolatilityAdjustedSizer:
    """
    Position sizing that adjusts for market volatility
    """

    def __init__(self, base_risk_pct: float = 1.0, vol_lookback: int = 20,
                 vol_target: float = 0.25, min_size_pct: float = 0.5,
                 max_size_pct: float = 2.0):
        """
        Initialize volatility-adjusted sizer

        Args:
            base_risk_pct: Base risk percentage per trade
            vol_lookback: Lookback period for volatility calculation
            vol_target: Target volatility level (e.g., 0.25 = 25%)
            min_size_pct: Minimum position size as % of base
            max_size_pct: Maximum position size as % of base
        """
        self.base_risk_pct = base_risk_pct
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.min_size_pct = min_size_pct
        self.max_size_pct = max_size_pct

    def compute_market_volatility(self, df: pd.DataFrame,
                                 vol_proxy: str = 'NIFTY50') -> pd.Series:
        """
        Compute market volatility using Nifty returns or VIX proxy

        Args:
            df: DataFrame with market data
            vol_proxy: Column name for volatility proxy

        Returns:
            Series with volatility estimates
        """
        if vol_proxy in df.columns:
            # Use provided volatility proxy (e.g., VIX)
            vol = df[vol_proxy].rolling(self.vol_lookback).mean()
        else:
            # Estimate volatility from returns
            returns = df['Close'].pct_change()
            vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)  # Annualized

        # Handle NaN values
        vol = vol.fillna(vol.median())
        return vol

    def adjust_position_size(self, base_quantity: int, current_vol: float,
                           avg_vol: float) -> int:
        """
        Adjust position size based on current vs target volatility

        Args:
            base_quantity: Base position size
            current_vol: Current market volatility
            avg_vol: Average historical volatility

        Returns:
            Adjusted position size
        """
        if avg_vol <= 0 or current_vol <= 0:
            return base_quantity

        # Volatility adjustment factor
        vol_ratio = current_vol / avg_vol
        adjustment_factor = self.vol_target / vol_ratio

        # Clamp adjustment factor
        adjustment_factor = np.clip(adjustment_factor,
                                  self.min_size_pct / 100,
                                  self.max_size_pct / 100)

        # Apply adjustment
        adjusted_quantity = int(base_quantity * adjustment_factor)

        # Ensure minimum lot size (assuming 1 for now)
        return max(1, adjusted_quantity)

    def get_adjusted_size(self, df: pd.DataFrame, base_quantity: int,
                         vol_proxy: str = 'NIFTY50') -> pd.Series:
        """
        Get volatility-adjusted position sizes for entire dataset

        Args:
            df: DataFrame with market data
            base_quantity: Base position size
            vol_proxy: Volatility proxy column

        Returns:
            Series with adjusted position sizes
        """
        vol_series = self.compute_market_volatility(df, vol_proxy)
        avg_vol = vol_series.median()

        adjusted_sizes = []
        for vol in vol_series:
            adj_size = self.adjust_position_size(base_quantity, vol, avg_vol)
            adjusted_sizes.append(adj_size)

        return pd.Series(adjusted_sizes, index=df.index)


class CircuitBreaker:
    """
    Circuit breaker mechanism to halt trading during adverse conditions
    """

    def __init__(self, daily_dd_threshold: float = 0.05,
                 monthly_dd_threshold: float = 0.15,
                 pause_days: int = 3,
                 volatility_threshold: float = 0.35):
        """
        Initialize circuit breaker

        Args:
            daily_dd_threshold: Daily drawdown threshold to trigger pause
            monthly_dd_threshold: Monthly drawdown threshold to trigger pause
            pause_days: Number of days to pause trading after trigger
            volatility_threshold: Volatility threshold to trigger pause
        """
        self.daily_dd_threshold = daily_dd_threshold
        self.monthly_dd_threshold = monthly_dd_threshold
        self.pause_days = pause_days
        self.volatility_threshold = volatility_threshold
        self.pause_until = None

    def check_circuit_breaker(self, equity_curve: pd.DataFrame,
                            current_date: pd.Timestamp,
                            market_vol: float) -> bool:
        """
        Check if circuit breaker should be triggered

        Args:
            equity_curve: DataFrame with equity curve data
            current_date: Current date
            market_vol: Current market volatility

        Returns:
            True if trading should be halted
        """
        # Check if currently in pause period
        if self.pause_until and current_date <= self.pause_until:
            return True

        # Check volatility threshold
        if market_vol > self.volatility_threshold:
            self.pause_until = current_date + timedelta(days=self.pause_days)
            return True

        # Check drawdown thresholds
        if len(equity_curve) < 2:
            return False

        # Daily drawdown check
        recent_equity = equity_curve.tail(2)
        if len(recent_equity) >= 2:
            daily_return = (recent_equity.iloc[-1]['Equity'] / recent_equity.iloc[-2]['Equity'] - 1)
            if daily_return < -self.daily_dd_threshold:
                self.pause_until = current_date + timedelta(days=self.pause_days)
                return True

        # Monthly drawdown check (approx 20 trading days)
        if len(equity_curve) >= 20:
            monthly_equity = equity_curve.tail(20)
            peak = monthly_equity['Equity'].max()
            current = monthly_equity.iloc[-1]['Equity']
            monthly_dd = (current - peak) / peak

            if monthly_dd < -self.monthly_dd_threshold:
                self.pause_until = current_date + timedelta(days=self.pause_days)
                return True

        return False

    def reset_pause(self):
        """Reset pause period"""
        self.pause_until = None


class EnhancedDiversification:
    """
    Advanced diversification with correlation and sector constraints
    """

    def __init__(self, max_sector_weight: float = 0.20,
                 max_correlation: float = 0.70,
                 min_sector_diversity: int = 3):
        """
        Initialize enhanced diversification

        Args:
            max_sector_weight: Maximum weight per sector
            max_correlation: Maximum correlation between positions
            min_sector_diversity: Minimum number of sectors required
        """
        self.max_sector_weight = max_sector_weight
        self.max_correlation = max_correlation
        self.min_sector_diversity = min_sector_diversity

    def compute_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix for positions

        Args:
            returns_df: DataFrame with position returns

        Returns:
            Correlation matrix
        """
        return returns_df.corr().fillna(0)

    def check_correlation_constraint(self, symbol: str,
                                   existing_positions: List[str],
                                   corr_matrix: pd.DataFrame) -> bool:
        """
        Check if new position violates correlation constraint

        Args:
            symbol: New symbol to check
            existing_positions: List of existing position symbols
            corr_matrix: Correlation matrix

        Returns:
            True if position passes correlation check
        """
        if not existing_positions:
            return True

        max_corr = 0
        for existing in existing_positions:
            if symbol in corr_matrix.index and existing in corr_matrix.columns:
                corr = abs(corr_matrix.loc[symbol, existing])
                max_corr = max(max_corr, corr)

        return max_corr <= self.max_correlation

    def check_sector_constraints(self, symbol: str,
                               sector_map: Dict[str, str],
                               current_positions: Dict[str, float],
                               sector_weights: Dict[str, float]) -> bool:
        """
        Check if new position violates sector constraints

        Args:
            symbol: New symbol to check
            sector_map: Mapping of symbols to sectors
            current_positions: Current position weights
            sector_weights: Current sector weights

        Returns:
            True if position passes sector checks
        """
        if symbol not in sector_map:
            return True  # Allow if sector unknown

        sector = sector_map[symbol]
        current_sector_weight = sector_weights.get(sector, 0)

        # Check if adding this position would exceed sector limit
        # Assume new position weight is reasonable (< 5%)
        estimated_new_weight = 0.05  # Conservative estimate

        return (current_sector_weight + estimated_new_weight) <= self.max_sector_weight

    def enforce_diversification(self, signals_df: pd.DataFrame,
                              sector_map: Optional[Dict[str, str]] = None,
                              corr_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply diversification constraints to signals

        Args:
            signals_df: DataFrame with trading signals
            sector_map: Symbol to sector mapping
            corr_matrix: Pre-computed correlation matrix

        Returns:
            Filtered signals DataFrame
        """
        filtered_df = signals_df.copy()

        # Get signal columns
        signal_cols = [col for col in signals_df.columns if col.endswith('_Flag')]

        for idx, row in signals_df.iterrows():
            active_signals = [col for col in signal_cols if row[col] == 1]

            if not active_signals:
                continue

            # For each active signal, check diversification constraints
            for signal_col in active_signals:
                symbol = row.get('Symbol', 'UNKNOWN')

                # Correlation check
                if corr_matrix is not None:
                    existing_symbols = []  # Would need to track from portfolio state
                    if not self.check_correlation_constraint(symbol, existing_symbols, corr_matrix):
                        filtered_df.loc[idx, signal_col] = 0
                        continue

                # Sector check
                if sector_map:
                    current_positions = {}  # Would need portfolio state
                    sector_weights = {}  # Would need to compute
                    if not self.check_sector_constraints(symbol, sector_map,
                                                       current_positions, sector_weights):
                        filtered_df.loc[idx, signal_col] = 0

        return filtered_df


class AdaptiveRiskManager:
    """
    Adaptive risk management combining all risk components
    """

    def __init__(self, config: Dict):
        """
        Initialize adaptive risk manager

        Args:
            config: Risk management configuration
        """
        self.config = config

        # Initialize components
        self.sizer = VolatilityAdjustedSizer(
            base_risk_pct=config.get('base_risk_pct', 1.0),
            vol_lookback=config.get('vol_lookback', 20),
            vol_target=config.get('vol_target', 0.25)
        )

        self.circuit_breaker = CircuitBreaker(
            daily_dd_threshold=config.get('daily_dd_threshold', 0.05),
            monthly_dd_threshold=config.get('monthly_dd_threshold', 0.15),
            pause_days=config.get('pause_days', 3)
        )

        self.diversification = EnhancedDiversification(
            max_sector_weight=config.get('max_sector_weight', 0.20),
            max_correlation=config.get('max_correlation', 0.70)
        )

    def get_position_size(self, df: pd.DataFrame, base_quantity: int,
                         vol_proxy: str = 'NIFTY50') -> int:
        """
        Get position size with volatility adjustment

        Args:
            df: Current market data
            base_quantity: Base position size
            vol_proxy: Volatility proxy column

        Returns:
            Adjusted position size
        """
        return self.sizer.get_adjusted_size(df, base_quantity, vol_proxy).iloc[-1]

    def should_halt_trading(self, equity_curve: pd.DataFrame,
                          current_date: pd.Timestamp,
                          market_vol: float) -> bool:
        """
        Check if trading should be halted

        Args:
            equity_curve: Portfolio equity curve
            current_date: Current date
            market_vol: Current market volatility

        Returns:
            True if trading should be halted
        """
        return self.circuit_breaker.check_circuit_breaker(equity_curve, current_date, market_vol)

    def apply_diversification_filters(self, signals_df: pd.DataFrame,
                                    sector_map: Optional[Dict[str, str]] = None,
                                    corr_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply diversification constraints

        Args:
            signals_df: Trading signals
            sector_map: Sector mapping
            corr_matrix: Correlation matrix

        Returns:
            Filtered signals
        """
        return self.diversification.enforce_diversification(signals_df, sector_map, corr_matrix)