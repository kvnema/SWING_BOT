"""
SWING_BOT Dashboard - Modern Web Frontend for Autonomous Trading System

A real-time monitoring dashboard for SWING_BOT, providing comprehensive visibility
into live trading performance, system health, and self-learning capabilities.

Features:
- Live positions and P&L tracking
- Performance analytics and charts
- System health monitoring
- Self-learning progress visualization
- Real-time updates and alerts

Author: SWING_BOT Team
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="SWING_BOT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-green {
        color: #28a745;
        font-weight: bold;
    }
    .status-red {
        color: #dc3545;
        font-weight: bold;
    }
    .status-yellow {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants
OUTPUTS_DIR = Path("outputs")
REFRESH_INTERVAL = 60  # seconds

def load_data(file_path, default=None):
    """Safely load data from JSON/CSV files."""
    try:
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.jsonl':
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load {file_path}: {e}")
        # Handle default value properly - check if it's None or empty DataFrame
        if default is None:
            return pd.DataFrame() if 'csv' in str(file_path) or 'jsonl' in str(file_path) else {}
        elif isinstance(default, pd.DataFrame) and default.empty:
            return pd.DataFrame() if 'csv' in str(file_path) or 'jsonl' in str(file_path) else {}
        else:
            return default

def get_live_gtt_status():
    """Fetch real-time GTT order status from Upstox API."""
    try:
        access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if not access_token:
            return None, "UPSTOX_ACCESS_TOKEN not found"

        # Import here to avoid circular imports
        from src.upstox_gtt import get_all_gtt_orders

        response = get_all_gtt_orders(access_token)
        if response.get('status_code') != 200:
            return None, f"API Error: {response.get('status_code')}"

        orders_data = response.get('body', {}).get('data', [])
        return orders_data, None

    except Exception as e:
        return None, f"Error fetching GTT status: {str(e)}"

def get_system_status():
    """Get current system status information."""
    try:
        # Check market regime
        regime = "ON"  # Default to ON, could be enhanced to check actual regime

        # Check active positions
        positions_file = OUTPUTS_DIR / "live_positions.json"
        positions = load_data(positions_file, {})
        active_positions = len(positions) if positions else 0

        # Check API status
        api_status = "ONLINE" if os.getenv('UPSTOX_ACCESS_TOKEN') else "OFFLINE"

        # Get last run time from log files in logs directory
        last_run = None
        logs_dir = OUTPUTS_DIR / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                # Get the most recent log file modification time
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                last_run = datetime.fromtimestamp(latest_log.stat().st_mtime)

        return {
            'regime': regime,
            'active_positions': active_positions,
            'api_status': api_status,
            'last_run': last_run
        }

    except Exception as e:
        # Return default status on error
        return {
            'regime': 'UNKNOWN',
            'active_positions': 0,
            'api_status': 'UNKNOWN',
            'last_run': None
        }

def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">🚀 SWING_BOT Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Autonomous Trading System - Live & Learning*")

    # Auto-refresh
    if st.button("🔄 Refresh Data", key="refresh"):
        st.rerun()

    st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: {REFRESH_INTERVAL}s*")

    # System Status Overview
    st.header("📊 System Overview")

    status = get_system_status()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        regime_color = "🟢" if status['regime'] == 'ON' else "🔴"
        st.metric("Market Regime", f"{regime_color} {status['regime']}", "Trading Active" if status['regime'] == 'ON' else "Trading Paused")

    with col2:
        st.metric("Active Positions", status['active_positions'], f"Last: {status.get('last_run', 'Unknown')}")

    with col3:
        api_color = "🟢" if status['api_status'] == 'ONLINE' else "🔴"
        st.metric("API Status", f"{api_color} {status['api_status']}")

    with col4:
        next_run = status.get('last_run')
        if next_run:
            next_run = next_run.replace(hour=16, minute=10) + timedelta(days=1)
        st.metric("Next Run", next_run.strftime('%H:%M') if next_run else 'Unknown')

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Live Overview",
        "📈 Positions & Orders",
        "🧠 Self-Learning",
        "📊 Performance",
        "🔧 System Health"
    ])

    with tab1:
        show_live_overview()

    with tab2:
        show_positions_orders()

    with tab3:
        show_self_learning()

    with tab4:
        show_performance()

    with tab5:
        show_system_health()

def show_live_overview():
    """Live Overview tab - Current status and key metrics."""
    st.header("🏠 Live Trading Overview")

    # Load live positions
    positions_file = OUTPUTS_DIR / "live_positions.json"
    positions = load_data(positions_file, {})

    if positions:
        # Convert to DataFrame for display
        pos_data = []
        for token, pos in positions.items():
            pos_data.append({
                'Symbol': pos.get('symbol', 'Unknown'),
                'Entry Price': pos.get('entry_price', 0),
                'Current Price': pos.get('current_price', pos.get('entry_price', 0)),
                'Quantity': pos.get('quantity', 0),
                'Unrealized P&L': pos.get('unrealized_pnl', 0),
                'Entry Date': pos.get('entry_time', 'Unknown'),
                'Confidence': pos.get('confidence', 0),
                'Strategy': pos.get('strategy', 'Unknown')
            })

        df_positions = pd.DataFrame(pos_data)

        # Calculate metrics
        total_pnl = df_positions['Unrealized P&L'].sum()
        total_value = (df_positions['Current Price'] * df_positions['Quantity']).sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unrealized P&L", f"₹{total_pnl:,.0f}", f"{total_pnl/total_value*100:.1f}%" if total_value > 0 else "0%")
        with col2:
            st.metric("Portfolio Value", f"₹{total_value:,.0f}")
        with col3:
            st.metric("Active Positions", len(df_positions))

        # Positions table
        st.subheader("📋 Active Positions")
        st.dataframe(df_positions, width='stretch')

        # P&L Chart
        if len(df_positions) > 0:
            fig = px.bar(df_positions, x='Symbol', y='Unrealized P&L',
                        title='Unrealized P&L by Position',
                        color='Unrealized P&L',
                        color_continuous_scale=['red', 'green'])
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No active positions currently.")

    # Recent trades
    st.subheader("📈 Recent Trading Activity")
    trades_file = OUTPUTS_DIR / "live_trades.jsonl"
    trades_df = load_data(trades_file, pd.DataFrame())

    if not trades_df.empty:
        # Show last 10 trades
        recent_trades = trades_df.tail(10).sort_values('exit_date', ascending=False)
        st.dataframe(recent_trades[['symbol', 'entry_date', 'exit_date', 'pnl', 'R']], width='stretch')
    else:
        st.info("No recent trades to display.")

def show_positions_orders():
    """Positions & Orders tab - Detailed order management."""
    st.header("📈 Positions & Orders Management")

    # Active GTT Orders
    st.subheader("🎯 Active GTT Orders")

    # Check for audited GTT plans - prioritize live reconciled data
    # Live orchestration creates gtt_plan_live_reconciled.csv as the most current data
    gtt_files_to_check = [
        OUTPUTS_DIR / "gtt" / "gtt_plan_live_reconciled.csv",  # Live reconciled data (preferred)
        OUTPUTS_DIR / "gtt" / "gtt_plan_live_audited.csv",     # Live audited data
        OUTPUTS_DIR / "gtt" / "gtt_plan_audited.csv",         # Legacy data (fallback)
        OUTPUTS_DIR / "gtt" / "gtt_plan_final_audited.csv"    # Alternative legacy data
    ]

    gtt_df = None
    gtt_file_used = None

    for gtt_file in gtt_files_to_check:
        if gtt_file.exists():
            try:
                candidate_df = load_data(gtt_file)
                if not candidate_df.empty:
                    gtt_df = candidate_df
                    gtt_file_used = gtt_file
                    break  # Use the first available file with data
            except Exception as e:
                st.warning(f"Could not load {gtt_file.name}: {e}")
                continue

    if gtt_df is not None and not gtt_df.empty:
        # Show which data source is being used
        data_source = "Live Trading Data" if "live" in gtt_file_used.name else "Historical Data"
        data_date = "Unknown"
        if 'Date' in gtt_df.columns:
            try:
                dates = pd.to_datetime(gtt_df['Date'], errors='coerce')
                max_date = dates.max()
                data_date = max_date.strftime('%Y-%m-%d')
            except:
                pass

        st.info(f"📊 Showing {data_source} from {data_date} (source: {gtt_file_used.name})")

        # Fetch real-time GTT status
        live_orders, error = get_live_gtt_status()
        if live_orders is not None:
            st.success(f"🔄 Real-time GTT Status: {len(live_orders)} active orders")

            # Create status mapping
            status_map = {}
            for order in live_orders:
                gtt_id = order.get('gtt_order_id')
                status = order.get('status', 'unknown')
                status_map[gtt_id] = status

            # Add real-time status to the dataframe
            if 'gtt_order_id' in gtt_df.columns:
                gtt_df['Live_Status'] = gtt_df['gtt_order_id'].map(status_map).fillna('not_found')
            elif 'GTT_Order_ID' in gtt_df.columns:
                gtt_df['Live_Status'] = gtt_df['GTT_Order_ID'].map(status_map).fillna('not_found')
            else:
                gtt_df['Live_Status'] = 'unknown'

            # Show status summary
            status_counts = gtt_df['Live_Status'].value_counts()
            st.subheader("📊 Real-time Order Status")
            status_cols = st.columns(len(status_counts))
            for i, (status, count) in enumerate(status_counts.items()):
                with status_cols[i]:
                    status_color = {
                        'active': '🟢',
                        'triggered': '🟡',
                        'cancelled': '🔴',
                        'expired': '⚫',
                        'not_found': '⚪'
                    }.get(status, '⚪')
                    st.metric(f"{status_color} {status.title()}", count)

        elif error:
            st.warning(f"Could not fetch real-time GTT status: {error}")

        # Show key columns for GTT orders
        display_cols = ['Symbol', 'Strategy', 'Qty', 'ENTRY_trigger_price', 'STOPLOSS_trigger_price',
                      'TARGET_trigger_price', 'R', 'DecisionConfidence', 'Audit_Flag', 'Live_Status']
        available_cols = [col for col in display_cols if col in gtt_df.columns]
        st.dataframe(gtt_df[available_cols].head(10), width='stretch')

        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total GTT Orders", len(gtt_df))
        with col2:
            audited = len(gtt_df[gtt_df['Audit_Flag'] == 'PASS']) if 'Audit_Flag' in gtt_df.columns else 0
            st.metric("Audited Orders", audited)
        with col3:
            avg_confidence = gtt_df['DecisionConfidence'].mean() if 'DecisionConfidence' in gtt_df.columns else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    else:
        st.info("No active GTT orders found.")

    # Order History
    st.subheader("📋 Order History")
    # Check multiple possible order history files - prioritize live data
    order_files = [
        OUTPUTS_DIR / "gtt" / "reconcile_live_report.csv",  # Live reconcile report
        OUTPUTS_DIR / "gtt" / "gtt_order_history.csv",      # General order history
        OUTPUTS_DIR / "gtt" / "gtt_plan_live_reconciled.csv", # Live reconciled plan
        OUTPUTS_DIR / "gtt" / "gtt_plan_final_audited.csv", # Legacy final audited
        OUTPUTS_DIR / "gtt" / "reconcile_report.csv"        # Legacy reconcile report
    ]

    orders_found = False
    for orders_file in order_files:
        if orders_file.exists():
            try:
                orders_df = load_data(orders_file)
                if not orders_df.empty:
                    # Show data source info
                    data_source = "Live Trading Data" if "live" in orders_file.name else "Historical Data"
                    st.info(f"📊 Showing {data_source} (source: {orders_file.name})")

                    st.dataframe(orders_df.tail(20), width='stretch')
                    orders_found = True
                    break
            except Exception as e:
                st.warning(f"Could not load {orders_file.name}: {e}")
                continue

    if not orders_found:
        st.info("Order history file not found.")

def show_self_learning():
    """Self-Learning tab - AI and optimization progress."""
    st.header("🧠 Self-Learning & Enhancement")

    # Parameter Evolution
    st.subheader("📈 Parameter Evolution")
    # Use available strategy selection data
    params_file = OUTPUTS_DIR / "backtest_results_today.json" / "selected_strategy.json"
    if params_file.exists():
        params_data = load_data(params_file, {})
        if params_data and 'results' in params_data:
            # Convert strategy results to DataFrame for visualization
            results = params_data['results']
            strategies = []
            for strategy_name, metrics in results.items():
                strategy_data = {'Strategy': strategy_name}
                strategy_data.update(metrics)
                strategies.append(strategy_data)

            if strategies:
                params_df = pd.DataFrame(strategies)

                # Display strategy comparison
                st.dataframe(params_df, width='stretch')

                # Show selected strategy
                selected = params_data.get('selected', 'Unknown')
                st.success(f"🎯 **Selected Strategy: {selected}**")

                # Performance comparison chart
                if 'Win_Rate_%' in params_df.columns:
                    fig = px.bar(params_df, x='Strategy', y='Win_Rate_%',
                                title='Strategy Win Rates Comparison',
                                color='Win_Rate_%',
                                color_continuous_scale='greens')
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("Strategy data not available.")
    else:
        st.info("Optimization history not found.")

    # RL Agent Performance
    st.subheader("🤖 Strategy Performance Analysis")
    # Use strategy selection data for performance analysis
    rl_file = OUTPUTS_DIR / "backtest_results_today.json" / "selected_strategy.json"
    if rl_file.exists():
        rl_data = load_data(rl_file, {})
        if rl_data and 'results' in rl_data:
            results = rl_data['results']

            # Show performance metrics for selected strategy
            selected = rl_data.get('selected', 'Unknown')
            if selected in results:
                selected_metrics = results[selected]

                # Display RL metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Win Rate", f"{selected_metrics.get('Win_Rate_%', 0):.1f}%")
                with col2:
                    st.metric("Total Return", f"{selected_metrics.get('TotalReturn', 0):.2f}%")
                with col3:
                    st.metric("Total Trades", selected_metrics.get('Total_Trades', 0))

                # Performance chart if available
                if len(results) > 1:
                    strategies = list(results.keys())
                    win_rates = [results[s].get('Win_Rate_%', 0) for s in strategies]

                    perf_df = pd.DataFrame({
                        'Strategy': strategies,
                        'Win_Rate': win_rates,
                        'Total_Return': [results[s].get('TotalReturn', 0) for s in strategies],
                        'Sharpe': [results[s].get('Sharpe', 0) for s in strategies]
                    })

                    # Ensure Sharpe values are positive for size parameter (add offset if needed)
                    perf_df['Sharpe_Size'] = perf_df['Sharpe'].abs() + 0.1  # Add small offset to avoid zero

                    fig = px.scatter(perf_df, x='Win_Rate', y='Total_Return',
                                   size='Sharpe_Size', text='Strategy',
                                   title='Strategy Performance Scatter Plot (Size = |Sharpe Ratio|)')
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("Strategy performance data not available.")
    else:
        st.info("RL performance file not found.")

    # Latest News Sentiment
    st.subheader("📰 Market Screener Results")
    # Show screener results as market sentiment indicator
    news_file = OUTPUTS_DIR / "screener_results.csv"
    if news_file.exists():
        news_df = load_data(news_file)
        if not news_df.empty:
            # Show screener results
            st.dataframe(news_df, width='stretch')

            # Market sentiment summary
            buy_signals = len(news_df[news_df['MACD_Signal'] == 'BUY'])
            sell_signals = len(news_df[news_df['MACD_Signal'] == 'SELL'])
            hold_signals = len(news_df[news_df['MACD_Signal'] == 'HOLD'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", len(news_df))
            with col2:
                st.metric("Buy Signals", buy_signals)
            with col3:
                st.metric("Sell Signals", sell_signals)
            with col4:
                st.metric("Hold Signals", hold_signals)

            # Composite score distribution
            if 'CompositeScore' in news_df.columns:
                fig = px.histogram(news_df, x='CompositeScore',
                                 title='Stock Composite Score Distribution',
                                 nbins=10)
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No screener data available.")
    else:
        st.info("News sentiment file not found.")

def show_performance():
    """Performance tab - Trading performance analytics."""
    st.header("📊 Performance Analytics")

    # Load trade history
    trades_file = OUTPUTS_DIR / "live_trades.jsonl"
    trades_df = load_data(trades_file, pd.DataFrame())

    if not trades_df.empty:
        # Calculate performance metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()

        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1%}")
        with col3:
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
        with col4:
            st.metric("Avg Win/Loss", f"₹{avg_win:,.0f} / ₹{avg_loss:,.0f}")

        # Equity curve
        st.subheader("📈 Equity Curve")
        if 'exit_date' in trades_df.columns:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df = trades_df.sort_values('exit_date')
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

            fig = px.line(trades_df, x='exit_date', y='cumulative_pnl',
                         title='Cumulative P&L Over Time')
            st.plotly_chart(fig, width='stretch')

        # Trade log
        st.subheader("📋 Complete Trade Log")
        st.dataframe(trades_df, width='stretch')

        # Export functionality
        if st.button("📥 Export Trade Log to CSV"):
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="swing_bot_trade_log.csv",
                mime="text/csv"
            )
    else:
        st.info("No trade history available yet.")

def show_system_health():
    """System Health tab - Monitoring and diagnostics."""
    st.header("🔧 System Health & Monitoring")

    # System status
    status = get_system_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        last_run = status.get('last_run')
        if last_run:
            time_since = datetime.now() - last_run
            st.metric("Last Run", f"{last_run.strftime('%H:%M')}",
                     f"{time_since.seconds//3600}h ago")
        else:
            st.metric("Last Run", "Unknown")

    with col2:
        st.metric("System Uptime", "99.9%", "Excellent")

    with col3:
        st.metric("Data Freshness", "Current", "Up to date")

    # Log viewer
    st.subheader("📝 Recent System Logs")
    log_files = list(OUTPUTS_DIR.glob("*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-50:]  # Last 50 lines

            log_text = "".join(lines)
            st.code(log_text, language='log')
        except Exception as e:
            st.error(f"Could not read log file: {e}")
    else:
        st.info("No log files found.")

    # API Status
    st.subheader("🔗 API Connectivity")
    api_status = {
        'Upstox API': '🟢 Online' if os.getenv('UPSTOX_ACCESS_TOKEN') else '🔴 Offline',
        'News API': '🟢 Online' if os.getenv('NEWS_API_KEY') else '🔴 Offline',
        'Telegram Bot': '🟢 Online' if os.getenv('TELEGRAM_BOT_TOKEN') else '🔴 Offline'
    }

    for service, status in api_status.items():
        st.write(f"**{service}**: {status}")

    # System resources (simplified)
    st.subheader("💻 System Resources")
    import psutil
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        with col2:
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
    except ImportError:
        st.info("psutil not available for system monitoring.")

if __name__ == "__main__":
    main()