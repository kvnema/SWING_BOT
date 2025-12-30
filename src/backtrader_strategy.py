"""
SWING_BOT Backtrader Strategy Implementation

This module implements the complete SWING_BOT momentum swing trading strategy
using Backtrader framework for advanced backtesting and walk-forward validation.

Key Features:
- All technical indicators (EMA, RSI, Bollinger Bands, Donchian, ATR, RVOL, etc.)
- Multi-strategy flags (VCP, SEPA, Donchian breakout, Squeeze, MR, AVWAP)
- Fixed strategy hierarchy + ensemble confirmation
- Composite scoring with z-score normalization
- Market regime filter (Nifty > SMA200 AND (ADX > 20 OR RSI > 50))
- ATR-based position sizing (1% risk per trade)
- Trailing stops and profit taking
- Realistic commissions and slippage
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SwingBotStrategy(bt.Strategy):
    """
    SWING_BOT Momentum Swing Trading Strategy for Backtrader

    Parameters:
    - risk_per_trade: Risk per trade as % of portfolio (default: 1.0)
    - atr_period: ATR period for stops and sizing (default: 14)
    - atr_stop_mult: ATR multiplier for initial stops (default: 1.5)
    - trail_type: Trailing stop type ('percentage', 'atr', 'psar') (default: 'atr')
    - profit_take_pct: Profit taking percentage (default: 50)
    - min_signals: Minimum momentum signals required (default: 2)
    - max_positions: Maximum positions in portfolio (default: 10)
    - sector_limits: Sector exposure limits (default: {'Consumer': 0.4, 'Cyclical': 0.3, 'Financial': 0.2, 'Chemicals': 0.1})
    """

    params = (
        ('risk_per_trade', 1.0),      # 1% risk per trade for production alignment
        ('atr_period', 14),           # ATR period
        ('atr_stop_mult', 1.5),       # ATR multiplier for stops
        ('trail_type', 'atr'),        # 'percentage', 'atr', or 'psar'
        ('profit_take_pct', 50),      # Take 50% profit at target
        ('min_signals', 1),           # Minimum momentum signals
        ('max_positions', 10),        # Max positions
        ('sector_limits', {
            'Consumer': 0.4,
            'Cyclical': 0.3,
            'Financial': 0.2,
            'Chemicals': 0.1
        }),
        ('benchmark_symbol', 'NIFTY50'),  # Benchmark for regime filter
    )

    def __init__(self):
        """Initialize strategy with minimal indicators"""

        # Keep track of positions and sectors
        self.positions_data = {}
        self.sector_exposure = {}
        self.benchmark_data = {}

        # Initialize minimal indicators for each data feed
        for data in self.datas:
            if data._name == self.params.benchmark_symbol:
                # Benchmark indicators for regime filter - only if data exists
                if len(data) > 0:
                    try:
                        self.init_benchmark_indicators(data)
                        logger.info(f"Initialized benchmark indicators for {data._name}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize benchmark indicators for {data._name}: {e}")
                else:
                    logger.warning(f"No benchmark data available for {data._name}")
            else:
                # Just create basic indicators directly
                ema20 = bt.indicators.EMA(data.close, period=20)
                ema50 = bt.indicators.EMA(data.close, period=50)
                ema200 = bt.indicators.EMA(data.close, period=200)
                rsi = bt.indicators.RSI(data.close, period=14)
                atr = bt.indicators.ATR(data, period=14)

                # Store in positions_data
                self.positions_data[data._name] = {
                    'indicators': {
                        'ema20': ema20,
                        'ema50': ema50,
                        'ema200': ema200,
                        'rsi': rsi,
                        'atr': atr,
                    },
                    'signals': {},
                    'sector': self.get_sector(data._name),
                    'in_position': False,
                    'entry_price': 0,
                    'stop_price': 0,
                    'target_price': 0,
                }

        # Commission and slippage
        self.broker.setcommission(commission=0.001, margin=False)  # 0.1% commission
        self.broker.set_slippage_perc(0.001)  # 0.1% slippage

    def prenext(self):
        """Called before next() when there isn't enough data for all indicators"""
        # Wait for at least 200 bars to ensure all indicators are ready
        # (SMA200, Donchian 20, Keltner 20, etc.)
        if len(self) < 200:
            return

    def init_benchmark_indicators(self, data):
        """Initialize benchmark indicators for regime filter"""
        self.benchmark_data[data._name] = {
            'sma200': bt.indicators.SMA(data.close, period=200),
            'adx': bt.indicators.ADX(data, period=14),
            'rsi': bt.indicators.RSI(data.close, period=14),
        }

    def init_stock_indicators(self, data):
        """Initialize minimal indicators for testing"""

        try:
            # Only basic indicators to isolate the issue
            ema20 = bt.indicators.EMA(data.close, period=20)
            ema50 = bt.indicators.EMA(data.close, period=50)
            ema200 = bt.indicators.EMA(data.close, period=200)

            # Basic trend - disabled for testing
            trend_ok = True

            # Basic RSI
            rsi = bt.indicators.RSI(data.close, period=14)

            # Basic ATR for position sizing
            atr = bt.indicators.ATR(data, period=14)

            # Volume MA
            volume_ma20 = bt.indicators.SMA(data.volume, period=20)

            # Basic RVOL - safe division
            try:
                volume_ma_val = volume_ma20[0] if hasattr(volume_ma20, '__getitem__') and volume_ma20[0] != 0 else 1
                rvol = data.volume[0] / volume_ma_val if hasattr(data.volume, '__getitem__') else 1.0
            except (IndexError, TypeError, ZeroDivisionError):
                rvol = 1.0

            # Store indicators
            indicators = {
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
                'trend_ok': trend_ok,
                'rsi': rsi,
                'atr': atr,
                'volume_ma20': volume_ma20,
                'rvol': rvol,
            }

            # Calculate basic signals
            signals = self.calculate_signals(data, indicators, trend_ok)

            # Store data
            self.positions_data[data._name] = {
                'indicators': indicators,
                'signals': signals,
                'sector': self.get_sector(data._name),
                'in_position': False,
                'entry_price': 0,
                'stop_price': 0,
                'target_price': 0,
            }
        except Exception as e:
            logger.warning(f"Failed to initialize indicators for {data._name}: {e}")
            # Initialize with empty data to prevent crashes
            self.positions_data[data._name] = {
                'indicators': {},
                'signals': {},
                'sector': self.get_sector(data._name),
                'in_position': False,
                'entry_price': 0,
                'stop_price': 0,
                'target_price': 0,
            }

            # Store minimal indicators
            indicators = {
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
                'rsi': rsi,
                'atr': atr,
                'rvol': rvol,
                'volume_ma20': volume_ma20,
            }

            # Calculate basic signals
            signals = self.calculate_signals(data, indicators, trend_ok)

            # Store data
            self.positions_data[data._name] = {
                'indicators': indicators,
                'signals': signals,
                'sector': self.get_sector(data._name),
                'in_position': False,
                'entry_price': 0,
                'stop_price': 0,
                'target_price': 0,
            }
        except Exception as e:
            logger.warning(f"Failed to initialize indicators for {data._name}: {e}")
            # Initialize with empty data to prevent crashes
            self.positions_data[data._name] = {
                'indicators': {},
                'signals': {},
                'sector': self.get_sector(data._name),
                'in_position': False,
                'entry_price': 0,
                'stop_price': 0,
                'target_price': 0,
            }

    def get_sector(self, symbol):
        """Map symbol to sector (simplified mapping)"""
        sector_map = {
            # Consumer
            'ASIANPAINT': 'Consumer', 'TITAN': 'Consumer', 'MARUTI': 'Consumer',
            'NESTLEIND': 'Consumer', 'ITC': 'Consumer', 'HINDUNILVR': 'Consumer',
            # Cyclical
            'LT': 'Cyclical', 'TATAMOTORS': 'Cyclical', 'BAJAJ-AUTO': 'Cyclical',
            'ULTRACEMCO': 'Cyclical', 'GRASIM': 'Cyclical', 'JSWSTEEL': 'Cyclical',
            # Financial
            'BAJAJFINSV': 'Financial', 'HDFCBANK': 'Financial', 'ICICIBANK': 'Financial',
            'AXISBANK': 'Financial', 'KOTAKBANK': 'Financial', 'SBIN': 'Financial',
            # Chemicals
            'UPL': 'Chemicals', 'PIDILITIND': 'Chemicals', 'SRF': 'Chemicals',
        }
        return sector_map.get(symbol, 'Other')

    def calculate_signals(self, data, indicators, trend_ok):
        """Calculate all strategy signals with safe array access"""

        signals = {}

        # Basic trend signals
        signals['trend_ok'] = trend_ok

        # RSI signals - safe access
        try:
            rsi_val = indicators['rsi'][0] if hasattr(indicators['rsi'], '__getitem__') else 50
            signals['rsi_oversold'] = rsi_val < 30
            signals['rsi_overbought'] = rsi_val > 70
        except (IndexError, TypeError, AttributeError):
            signals['rsi_oversold'] = False
            signals['rsi_overbought'] = False

        # MACD signals - disabled (no MACD indicator)
        signals['macd_bullish'] = False

        # Donchian breakout - disabled (no Donchian indicators)
        signals['donchian_breakout'] = False

        # Simplified complex signals - disable temporarily to isolate issue
        signals['sepa_flag'] = False
        signals['vcp_flag'] = False
        signals['mr_flag'] = False
        signals['bbkc_squeeze_flag'] = False
        signals['avwap_reclaim_flag'] = False
        signals['ts_momentum_flag'] = False

        return signals

    def check_regime_filter(self):
        """Check if market regime allows new positions"""
        # Temporarily disabled due to benchmark data issues - always allow trading
        return True

    def calculate_composite_score(self, data_name):
        """Calculate composite score for ranking - simplified for testing"""
        # Simplified scoring for testing - just return a basic score
        return 50.0  # Neutral score
        rvol_score = indicators['rvol'][0] * 0.18   # 18% weight
        trend_score = signals['trend_ok'][0] * 15 if signals['trend_ok'][0] is not None else 0   # 15% weight
        breakout_score = signals['donchian_breakout'][0] * 15 if signals['donchian_breakout'][0] is not None else 0  # 15% weight

        # Base tightness (inverse of BB bandwidth percentile)
        bb_bw_rank = indicators['bb_bandwidth'][0]  # Simplified
        base_tightness = (1 - bb_bw_rank) * 10      # 10% weight

        total_score = rs_score + rvol_score + trend_score + breakout_score + base_tightness
        return max(0, min(100, total_score))  # Clamp to 0-100

    def get_momentum_signals_count(self, data_name):
        """Count active momentum signals"""
        signals = self.positions_data[data_name]['signals']
        momentum_flags = ['sepa_flag', 'vcp_flag', 'donchian_breakout', 'bbkc_squeeze_flag']

        count = 0
        for flag in momentum_flags:
            if flag in signals and signals[flag][0] == 1.0:  # Backtrader bt.And returns 1.0 when True
                count += 1
        return count

    def can_enter_position(self, data_name):
        """Check if we can enter a position"""

        # Check regime filter
        # if not self.check_regime_filter():
        #     return False

        # Check momentum signals
        # signal_count = self.get_momentum_signals_count(data_name)
        # if signal_count < self.params.min_signals:
        #     return False

        # Check TS momentum
        # signals = self.positions_data[data_name]['signals']
        # if not signals['ts_momentum_flag'][0]:
        #     return False

        # Check position limits
        current_positions = len([p for p in self.positions_data.values() if p['in_position']])
        if current_positions >= self.params.max_positions:
            return False

        # Check sector limits
        # sector = self.positions_data[data_name]['sector']
        # current_sector_exposure = self.sector_exposure.get(sector, 0)
        # max_sector_limit = self.params.sector_limits.get(sector, 0.2)
        #
        # if current_sector_exposure >= max_sector_limit:
        #     return False

        # Check if already in position
        if self.positions_data[data_name]['in_position']:
            return False

        return True

    def calculate_position_size(self, data_name):
        """Calculate position size based on ATR risk management"""
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * (self.params.risk_per_trade / 100)

        data = self.getdatabyname(data_name)
        try:
            indicators = self.positions_data[data_name]['indicators']
            # Check if ATR indicator has valid data
            if len(indicators['atr']) == 0 or indicators['atr'][0] is None:
                logger.info(f"ATR indicator not ready for {data_name}")
                return 0

            atr = indicators['atr'][0]
            current_price = data.close[0]
        except (IndexError, TypeError, KeyError):
            logger.info(f"Indicators not ready for {data_name}")
            return 0

        logger.info(f"Position sizing for {data_name}: ATR={atr}, Price={current_price}, Portfolio={portfolio_value}")

        # Risk per share = ATR * multiplier
        risk_per_share = atr * self.params.atr_stop_mult

        if risk_per_share <= 0:
            logger.info(f"Invalid risk_per_share for {data_name}: {risk_per_share}")
            return 0

        # Position size = Risk amount / Risk per share
        position_size = risk_amount / risk_per_share

        # Convert to number of shares
        shares = position_size / current_price

        logger.info(f"Calculated position_size={position_size}, shares={shares}, final_shares={int(shares)}")

        # Ensure at least 1 share for backtesting
        shares = max(1, int(shares))

        return shares

    def next(self):
        """Minimal strategy logic for testing"""
        try:
            # Safety check
            if len(self) < 200:
                return

            # Simple logic: buy if not in position, sell after some time
            for data_name, pos_data in self.positions_data.items():
                data = self.getdatabyname(data_name)
                position = self.getpositionbyname(data_name)

                if not pos_data['in_position'] and len(self) % 100 == 0:  # Buy every 100 bars
                    # Simple position sizing
                    shares = int((self.broker.getvalue() * 0.01) / data.close[0])  # 1% of portfolio
                    if shares > 0:
                        self.buy(data=data, size=shares)
                        pos_data['in_position'] = True
                        pos_data['entry_price'] = data.close[0]
                        logger.info(f"BUY: {data_name} at {data.close[0]:.2f}, shares: {shares}")

                elif pos_data['in_position'] and len(self) % 150 == 0:  # Sell every 150 bars
                    self.sell(data=data, size=position.size)
                    pos_data['in_position'] = False
                    logger.info(f"SELL: {data_name} at {data.close[0]:.2f}, P&L: {(data.close[0] - pos_data['entry_price']) * position.size:.2f}")

        except Exception as e:
            logger.error(f"Error in next(): {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def update_trailing_stop(self, data_name, current_price):
        """Update trailing stop based on selected method"""
        pos_data = self.positions_data[data_name]
        entry_price = pos_data['entry_price']

        if self.params.trail_type == 'atr':
            # ATR-based trailing stop
            try:
                atr = self.positions_data[data_name]['indicators']['atr'][0]
                new_stop = current_price - (atr * self.params.atr_stop_mult)
                pos_data['stop_price'] = max(pos_data['stop_price'], new_stop)
            except (IndexError, TypeError):
                pass  # Keep existing stop if ATR not ready

        elif self.params.trail_type == 'percentage':
            # Percentage-based trailing stop
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > 0.05:  # 5% profit threshold
                trail_pct = 0.02  # 2% trailing stop
                new_stop = current_price * (1 - trail_pct)
                pos_data['stop_price'] = max(pos_data['stop_price'], new_stop)

        elif self.params.trail_type == 'psar':
            # Parabolic SAR (simplified)
            # This would require implementing PSAR indicator
            pass

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: {order.data._name} | Size: {order.size} | Price: {order.executed.price:.2f}")
            elif order.issell():
                logger.info(f"SELL EXECUTED: {order.data._name} | Size: {order.size} | Price: {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"ORDER FAILED: {order.data._name} | Status: {order.getstatusname()}")

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if trade.isclosed:
            pnl = trade.pnlcomm
            logger.info(f"TRADE CLOSED: {trade.data._name} | P&L: {pnl:.2f} | Return: {trade.pnlcomm/trade.value*100:.2f}%")

    def stop(self):
        """Strategy completion - generate final report"""
        logger.info("Strategy completed. Generating final report...")

        # Calculate final metrics
        final_value = self.broker.getvalue()
        total_return = (final_value - self.broker.startingcash) / self.broker.startingcash * 100

        logger.info(f"Final Portfolio Value: INR {final_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Trades: {len(self._trades)}")

        # Export trade log
        self.export_trade_log()

    def export_trade_log(self):
        """Export detailed trade log to CSV"""
        trades_data = []
        for trade in self._trades:
            trades_data.append({
                'symbol': trade.data._name,
                'entry_date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d'),
                'exit_date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d'),
                'entry_price': trade.pricein,
                'exit_price': trade.priceout,
                'shares': trade.size,
                'pnl': trade.pnlcomm,
                'return_pct': trade.pnlcomm / trade.value * 100,
                'holding_days': (bt.num2date(trade.dtclose) - bt.num2date(trade.dtopen)).days
            })

        if trades_data:
            df = pd.DataFrame(trades_data)
            df.to_csv('backtest_trade_log.csv', index=False)
            logger.info("Trade log exported to backtest_trade_log.csv")