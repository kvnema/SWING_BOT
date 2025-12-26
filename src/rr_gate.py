"""
Risk-Reward Gate
Standardized entry, stop-loss, target, and risk-reward calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union


class RRGate:
    """
    Standardized risk-reward calculations for all strategies
    """

    def __init__(self, min_rr_ratio: float = 1.5, max_rr_ratio: float = 4.0,
                 max_risk_pct: float = 0.02, atr_sl_multiplier: float = 1.5):
        """
        Initialize RR Gate parameters

        Args:
            min_rr_ratio: Minimum risk-reward ratio required
            max_rr_ratio: Maximum risk-reward ratio allowed
            max_risk_pct: Maximum risk per trade as % of capital
            atr_sl_multiplier: ATR multiplier for stop loss
        """
        self.min_rr_ratio = min_rr_ratio
        self.max_rr_ratio = max_rr_ratio
        self.max_risk_pct = max_risk_pct
        self.atr_sl_multiplier = atr_sl_multiplier

    def calculate_entry_levels(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """
        Calculate standardized entry levels for different strategy types

        Args:
            df: DataFrame with OHLC and strategy signals
            strategy_type: Type of strategy (breakout, mean_reversion, etc.)

        Returns:
            DataFrame with entry levels
        """
        result_df = df.copy()

        if strategy_type in ['breakout', 'momentum']:
            # Breakout entries: above resistance/high
            result_df['entry_price'] = result_df['High']
            result_df['entry_type'] = 'breakout'

        elif strategy_type == 'mean_reversion':
            # MR entries: pullback to support/low
            result_df['entry_price'] = result_df['Low']
            result_df['entry_type'] = 'pullback'

        elif strategy_type in ['squeeze', 'volatility']:
            # Squeeze entries: expansion breakout
            result_df['entry_price'] = result_df['Close']
            result_df['entry_type'] = 'expansion'

        elif strategy_type == 'pattern':
            # Pattern entries: confirmation close
            result_df['entry_price'] = result_df['Close']
            result_df['entry_type'] = 'pattern'

        else:
            # Default to close
            result_df['entry_price'] = result_df['Close']
            result_df['entry_type'] = 'default'

        return result_df

    def calculate_stop_loss(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """
        Calculate standardized stop loss levels

        Args:
            df: DataFrame with OHLC and ATR
            strategy_type: Strategy type for SL logic

        Returns:
            DataFrame with stop loss levels
        """
        result_df = df.copy()

        # ATR-based stop loss
        if 'ATR' in df.columns:
            atr_sl = df['ATR'] * self.atr_sl_multiplier
        else:
            # Fallback: percentage-based SL
            atr_sl = df['Close'] * 0.02  # 2% fallback

        if strategy_type in ['breakout', 'momentum']:
            # SL below recent swing low or ATR below entry
            result_df['stop_loss'] = np.minimum(
                df['entry_price'] - atr_sl,
                df.get('swing_low', df['Low'])
            )

        elif strategy_type == 'mean_reversion':
            # SL above recent swing high or ATR above entry
            result_df['stop_loss'] = np.maximum(
                df['entry_price'] + atr_sl,
                df.get('swing_high', df['High'])
            )

        elif strategy_type in ['squeeze', 'volatility']:
            # Wider stops for volatility strategies
            result_df['stop_loss'] = df['entry_price'] - (atr_sl * 1.5)

        else:
            # Default ATR-based SL
            result_df['stop_loss'] = df['entry_price'] - atr_sl

        # Ensure SL is reasonable (not too wide)
        max_sl_pct = 0.05  # 5% max stop loss
        max_sl_price = df['entry_price'] * (1 - max_sl_pct)
        result_df['stop_loss'] = np.maximum(result_df['stop_loss'], max_sl_price)

        return result_df

    def calculate_targets(self, df: pd.DataFrame, strategy_type: str,
                         rr_ratio: float = 2.0) -> pd.DataFrame:
        """
        Calculate profit targets based on risk-reward ratio

        Args:
            df: DataFrame with entry and stop loss
            strategy_type: Strategy type
            rr_ratio: Desired risk-reward ratio

        Returns:
            DataFrame with profit targets
        """
        result_df = df.copy()

        # Calculate risk amount
        risk_amount = result_df['entry_price'] - result_df['stop_loss']

        # Target based on RR ratio
        result_df['target_price'] = result_df['entry_price'] + (risk_amount * rr_ratio)

        # Strategy-specific target adjustments
        if strategy_type == 'breakout':
            # Higher targets for breakouts
            result_df['target_price'] = result_df['entry_price'] + (risk_amount * 2.5)

        elif strategy_type == 'mean_reversion':
            # Conservative targets for MR
            result_df['target_price'] = result_df['entry_price'] + (risk_amount * 1.5)

        elif strategy_type in ['squeeze', 'volatility']:
            # Very high targets for squeeze breakouts
            result_df['target_price'] = result_df['entry_price'] + (risk_amount * 3.0)

        # Partial targets at 1:1 and 2:1 RR
        result_df['target_1r'] = result_df['entry_price'] + risk_amount
        result_df['target_2r'] = result_df['entry_price'] + (risk_amount * 2)

        return result_df

    def calculate_position_size(self, df: pd.DataFrame, capital: float,
                              max_risk_pct: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate position size based on risk management

        Args:
            df: DataFrame with entry and stop loss
            capital: Available capital
            max_risk_pct: Max risk per trade (overrides instance default)

        Returns:
            DataFrame with position sizing
        """
        result_df = df.copy()
        risk_pct = max_risk_pct or self.max_risk_pct

        # Risk amount per share
        risk_per_share = result_df['entry_price'] - result_df['stop_loss']

        # Position size based on risk
        risk_amount = capital * risk_pct
        result_df['shares'] = (risk_amount / risk_per_share).astype(int)

        # Ensure minimum position size
        result_df['shares'] = np.maximum(result_df['shares'], 1)

        # Position value
        result_df['position_value'] = result_df['shares'] * result_df['entry_price']

        # Actual risk %
        result_df['actual_risk_pct'] = (risk_per_share * result_df['shares']) / capital

        return result_df

    def calculate_rr_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-reward metrics for signals

        Args:
            df: DataFrame with entry, SL, and targets

        Returns:
            DataFrame with RR metrics
        """
        result_df = df.copy()

        # Risk amount
        result_df['risk_amount'] = result_df['entry_price'] - result_df['stop_loss']

        # Reward amounts
        result_df['reward_1r'] = result_df['target_1r'] - result_df['entry_price']
        result_df['reward_2r'] = result_df['target_2r'] - result_df['entry_price']
        result_df['reward_target'] = result_df['target_price'] - result_df['entry_price']

        # RR ratios
        result_df['rr_ratio_1r'] = result_df['reward_1r'] / result_df['risk_amount']
        result_df['rr_ratio_2r'] = result_df['reward_2r'] / result_df['risk_amount']
        result_df['rr_ratio_target'] = result_df['reward_target'] / result_df['risk_amount']

        # RR gate: must meet minimum ratio
        result_df['rr_gate_pass'] = result_df['rr_ratio_target'] >= self.min_rr_ratio

        # RR gate: must not exceed maximum ratio (too aggressive)
        result_df['rr_gate_pass'] = result_df['rr_gate_pass'] & (result_df['rr_ratio_target'] <= self.max_rr_ratio)

        return result_df

    def apply_rr_gate(self, signals_df: pd.DataFrame, strategy_type: str,
                     capital: float = 100000) -> pd.DataFrame:
        """
        Apply complete RR gate to trading signals

        Args:
            signals_df: DataFrame with trading signals
            strategy_type: Strategy type
            capital: Available capital

        Returns:
            DataFrame with complete RR analysis
        """
        df = signals_df.copy()

        # Calculate entry levels
        df = self.calculate_entry_levels(df, strategy_type)

        # Calculate stop loss
        df = self.calculate_stop_loss(df, strategy_type)

        # Calculate targets
        df = self.calculate_targets(df, strategy_type)

        # Calculate position size
        df = self.calculate_position_size(df, capital)

        # Calculate RR metrics
        df = self.calculate_rr_metrics(df)

        # Final gate decision
        df['rr_gate_final'] = (
            df['rr_gate_pass'] &
            (df['actual_risk_pct'] <= self.max_risk_pct) &
            (df['shares'] >= 1)
        )

        return df

    def get_rr_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for RR analysis

        Args:
            df: DataFrame with RR calculations

        Returns:
            Dict with RR summary stats
        """
        if df.empty:
            return {}

        return {
            'total_signals': len(df),
            'rr_gate_pass': df['rr_gate_final'].sum(),
            'avg_rr_ratio': df['rr_ratio_target'].mean(),
            'min_rr_ratio': df['rr_ratio_target'].min(),
            'max_rr_ratio': df['rr_ratio_target'].max(),
            'avg_risk_pct': df['actual_risk_pct'].mean(),
            'avg_position_value': df['position_value'].mean(),
            'total_exposure': df[df['rr_gate_final']]['position_value'].sum()
        }


def validate_rr_setup(entry: float, stop: float, target: float,
                     min_rr: float = 1.5) -> Tuple[bool, float]:
    """
    Validate a single RR setup

    Args:
        entry: Entry price
        stop: Stop loss price
        target: Target price
        min_rr: Minimum RR ratio required

    Returns:
        Tuple of (is_valid, rr_ratio)
    """
    if entry <= 0 or stop <= 0 or target <= 0:
        return False, 0.0

    # For long positions
    if entry > stop:
        risk = entry - stop
        reward = target - entry
        rr_ratio = reward / risk if risk > 0 else 0
        is_valid = rr_ratio >= min_rr
    else:
        # Short position
        risk = stop - entry
        reward = entry - target
        rr_ratio = reward / risk if risk > 0 else 0
        is_valid = rr_ratio >= min_rr

    return is_valid, rr_ratio