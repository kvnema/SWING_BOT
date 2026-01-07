"""
Enhanced Profit-Taking and Exit Strategies for SWING_BOT
Implements tiered exits, trailing stops, and regime-specific strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ExitType(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"


class TieredProfitTaker:
    """
    Tiered profit-taking with partial exits at different profit levels
    """

    def __init__(self, tiers: List[Tuple[float, float]] = None,
                 trailing_stop_pct: float = 0.10):
        """
        Initialize tiered profit taker

        Args:
            tiers: List of (profit_level, exit_percentage) tuples
                  e.g., [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]
                  Means: exit 50% at 1:1 R:R, 30% at 2:1, 20% at 3:1
            trailing_stop_pct: Trailing stop percentage from peak
        """
        if tiers is None:
            tiers = [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]  # Default tiers

        self.tiers = sorted(tiers, key=lambda x: x[0])  # Sort by profit level
        self.trailing_stop_pct = trailing_stop_pct

    def calculate_exit_levels(self, entry_price: float, stop_price: float) -> Dict:
        """
        Calculate profit-taking levels based on risk-reward

        Args:
            entry_price: Entry price
            stop_price: Stop loss price

        Returns:
            Dict with exit levels and percentages
        """
        risk = abs(entry_price - stop_price)
        if risk == 0:
            return {}

        exit_levels = {}
        for rr_ratio, exit_pct in self.tiers:
            profit_target = entry_price + (risk * rr_ratio)
            exit_levels[profit_target] = exit_pct

        # Add trailing stop level
        trailing_stop = entry_price + (risk * self.tiers[-1][0] * (1 - self.trailing_stop_pct))

        return {
            'profit_targets': exit_levels,
            'trailing_stop': trailing_stop,
            'final_stop': stop_price
        }

    def get_exit_signal(self, current_price: float, entry_price: float,
                       peak_price: float, exit_levels: Dict) -> Tuple[ExitType, float]:
        """
        Determine if position should be exited

        Args:
            current_price: Current market price
            entry_price: Original entry price
            peak_price: Highest price since entry
            exit_levels: Pre-calculated exit levels

        Returns:
            Tuple of (exit_type, exit_percentage)
        """
        # Check stop loss
        if current_price <= exit_levels.get('final_stop', 0):
            return ExitType.STOP_LOSS, 1.0

        # Check profit targets
        profit_targets = exit_levels.get('profit_targets', {})
        for target_price, exit_pct in profit_targets.items():
            if peak_price >= target_price and current_price <= target_price:
                return ExitType.TAKE_PROFIT, exit_pct

        # Check trailing stop
        trailing_stop = exit_levels.get('trailing_stop', peak_price * (1 - self.trailing_stop_pct))
        if current_price <= trailing_stop:
            return ExitType.TRAILING_STOP, 1.0

        return None, 0.0


class ParabolicSARTrailingStop:
    """
    Parabolic SAR for dynamic trailing stops
    """

    def __init__(self, acceleration_factor: float = 0.02,
                 max_acceleration: float = 0.20):
        """
        Initialize Parabolic SAR

        Args:
            acceleration_factor: Starting acceleration factor
            max_acceleration: Maximum acceleration factor
        """
        self.af = acceleration_factor
        self.max_af = max_acceleration
        self.sar = None
        self.ep = None  # Extreme point
        self.trend = None  # 1 for uptrend, -1 for downtrend

    def calculate_sar(self, high: float, low: float, prev_sar: float = None) -> float:
        """
        Calculate next SAR value

        Args:
            high: Current high price
            low: Current low price
            prev_sar: Previous SAR value

        Returns:
            Next SAR value
        """
        if prev_sar is None:
            # Initialize SAR
            self.sar = low
            self.ep = high
            self.trend = 1  # Assume uptrend initially
            return self.sar

        # Determine trend
        if self.trend == 1:
            # Uptrend
            if low <= prev_sar:
                # Trend reversal
                self.trend = -1
                self.sar = self.ep
                self.ep = low
                self.af = self.af  # Reset acceleration
            else:
                # Continue uptrend
                if high > self.ep:
                    self.ep = high
                    self.af = min(self.af + self.af, self.max_af)
                self.sar = prev_sar + self.af * (self.ep - prev_sar)
        else:
            # Downtrend
            if high >= prev_sar:
                # Trend reversal
                self.trend = 1
                self.sar = self.ep
                self.ep = high
                self.af = self.af  # Reset acceleration
            else:
                # Continue downtrend
                if low < self.ep:
                    self.ep = low
                    self.af = min(self.af + self.af, self.max_af)
                self.sar = prev_sar + self.af * (self.ep - prev_sar)

        return self.sar

    def get_trailing_stop(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trailing stops for entire dataset

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with trailing stop levels
        """
        sar_values = []

        for idx, row in df.iterrows():
            high = row['High']
            low = row['Low']

            if len(sar_values) == 0:
                sar = self.calculate_sar(high, low)
            else:
                sar = self.calculate_sar(high, low, sar_values[-1])

            sar_values.append(sar)

        return pd.Series(sar_values, index=df.index)


class RegimeSpecificExits:
    """
    Exit strategies that adapt to market regime
    """

    def __init__(self):
        self.profit_taker = TieredProfitTaker()
        self.sar_trailer = ParabolicSARTrailingStop()

    def get_regime_exit_strategy(self, regime: str) -> Dict:
        """
        Get exit strategy parameters based on market regime

        Args:
            regime: Market regime ('strong_bull', 'weak_bull', 'sideways', 'bear')

        Returns:
            Dict with exit strategy parameters
        """
        strategies = {
            'strong_bull': {
                'tiers': [(1.0, 0.3), (2.0, 0.3), (4.0, 0.4)],  # Let winners run
                'use_sar': True,
                'time_exit_days': 20,
                'trailing_stop_pct': 0.15
            },
            'weak_bull': {
                'tiers': [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)],  # Moderate profit-taking
                'use_sar': True,
                'time_exit_days': 15,
                'trailing_stop_pct': 0.12
            },
            'sideways': {
                'tiers': [(1.0, 0.7), (1.5, 0.3)],  # Quick profits in ranging markets
                'use_sar': False,
                'time_exit_days': 10,
                'trailing_stop_pct': 0.08
            },
            'bear': {
                'tiers': [(0.5, 0.8), (1.0, 0.2)],  # Conservative in downtrends
                'use_sar': False,
                'time_exit_days': 5,
                'trailing_stop_pct': 0.05
            }
        }

        return strategies.get(regime, strategies['sideways'])

    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime

        Args:
            df: DataFrame with market data

        Returns:
            Regime classification
        """
        if len(df) < 20:
            return 'sideways'

        # Get recent data
        recent = df.tail(20)

        # Trend strength (ADX)
        adx = recent['ADX14'].mean() if 'ADX14' in recent.columns else 25

        # Momentum
        momentum = recent['TS_Momentum'].mean() if 'TS_Momentum' in recent.columns else 0

        # Volatility
        returns = recent['Close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.25

        # Regime classification
        if adx > 30 and momentum > 0.05 and vol < 0.30:
            return 'strong_bull'
        elif adx > 20 and momentum > 0:
            return 'weak_bull'
        elif adx < 20 and vol < 0.25:
            return 'sideways'
        else:
            return 'bear'

    def apply_regime_exits(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime-specific exit strategies to signals

        Args:
            df: Market data DataFrame
            signals_df: Signals DataFrame

        Returns:
            Enhanced signals with regime-specific exits
        """
        enhanced_df = signals_df.copy()

        # Detect regime
        regime = self.detect_regime(df)
        strategy_params = self.get_regime_exit_strategy(regime)

        # Update profit taker with regime-specific parameters
        self.profit_taker = TieredProfitTaker(
            tiers=strategy_params['tiers'],
            trailing_stop_pct=strategy_params['trailing_stop_pct']
        )

        # Add regime information
        enhanced_df['detected_regime'] = regime
        enhanced_df['regime_time_exit_days'] = strategy_params['time_exit_days']
        enhanced_df['regime_use_sar'] = strategy_params['use_sar']

        return enhanced_df


class CompoundingEngine:
    """
    Automated profit reinvestment for compounding growth
    """

    def __init__(self, reinvest_pct: float = 0.50,
                 min_reinvest_amount: float = 10000,
                 max_reinvest_pct: float = 0.80):
        """
        Initialize compounding engine

        Args:
            reinvest_pct: Percentage of profits to reinvest
            min_reinvest_amount: Minimum amount to trigger reinvestment
            max_reinvest_pct: Maximum reinvestment percentage
        """
        self.reinvest_pct = reinvest_pct
        self.min_reinvest_amount = min_reinvest_amount
        self.max_reinvest_pct = max_reinvest_pct

    def calculate_reinvestment(self, monthly_pnl: float,
                             current_equity: float) -> Tuple[float, float]:
        """
        Calculate reinvestment amount and new position sizing

        Args:
            monthly_pnl: Monthly profit/loss
            current_equity: Current equity level

        Returns:
            Tuple of (reinvest_amount, new_risk_pct)
        """
        if monthly_pnl <= 0:
            return 0.0, 1.0  # No reinvestment on losses

        # Calculate reinvestment amount
        reinvest_amount = monthly_pnl * self.reinvest_pct

        # Check minimum threshold
        if reinvest_amount < self.min_reinvest_amount:
            return 0.0, 1.0

        # Calculate new risk percentage (compounded growth)
        equity_growth_factor = (current_equity + reinvest_amount) / current_equity
        new_risk_pct = min(1.0 * equity_growth_factor, self.max_reinvest_pct)

        return reinvest_amount, new_risk_pct

    def track_compounding(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Track compounding metrics over time

        Args:
            equity_curve: Portfolio equity curve

        Returns:
            DataFrame with compounding metrics
        """
        df = equity_curve.copy()

        # Calculate monthly returns
        df['monthly_return'] = df['Equity'].pct_change(20)  # Approx monthly

        # Calculate reinvestment amounts
        df['reinvest_amount'] = 0.0
        df['new_risk_pct'] = 1.0

        for i in range(20, len(df)):
            monthly_pnl = df.loc[i, 'Equity'] - df.loc[i-20, 'Equity']
            current_equity = df.loc[i, 'Equity']

            reinvest_amt, new_risk = self.calculate_reinvestment(monthly_pnl, current_equity)
            df.loc[i, 'reinvest_amount'] = reinvest_amt
            df.loc[i, 'new_risk_pct'] = new_risk

        return df


class EnhancedExitManager:
    """
    Comprehensive exit management combining all exit strategies
    """

    def __init__(self, config: Dict):
        """
        Initialize enhanced exit manager

        Args:
            config: Exit strategy configuration
        """
        self.config = config

        self.profit_taker = TieredProfitTaker(
            tiers=config.get('profit_tiers', [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]),
            trailing_stop_pct=config.get('trailing_stop_pct', 0.10)
        )

        self.regime_exits = RegimeSpecificExits()

        self.compounding = CompoundingEngine(
            reinvest_pct=config.get('reinvest_pct', 0.50),
            min_reinvest_amount=config.get('min_reinvest_amount', 10000)
        )

    def apply_enhanced_exits(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enhanced exit strategies

        Args:
            df: Market data
            signals_df: Trading signals

        Returns:
            Enhanced signals with exit strategies
        """
        # Apply regime-specific exits
        enhanced_signals = self.regime_exits.apply_regime_exits(df, signals_df)

        # Add profit-taking levels (would be calculated per trade)
        # This is a placeholder - actual implementation would track per-position

        return enhanced_signals

    def get_compounding_adjustments(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Get compounding adjustments for position sizing

        Args:
            equity_curve: Portfolio equity curve

        Returns:
            DataFrame with compounding metrics
        """
        return self.compounding.track_compounding(equity_curve)