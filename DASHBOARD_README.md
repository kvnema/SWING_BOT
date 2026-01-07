# SWING_BOT Dashboard

A modern web-based dashboard for monitoring and managing the SWING_BOT autonomous trading system.

## Features

### 🏠 Live Overview
- Real-time positions and P&L tracking
- Portfolio value and performance metrics
- Active position visualization
- Recent trading activity

### 📈 Positions & Orders
- Active GTT (Good Till Triggered) orders
- Order history and management
- Position sizing and risk management

### 🧠 Self-Learning
- Parameter evolution tracking
- RL (Reinforcement Learning) agent performance
- News sentiment analysis
- Optimization history

### 📊 Performance Analytics
- Trading performance metrics (win rate, P&L)
- Equity curve visualization
- Complete trade log
- Export functionality

### 🔧 System Health
- System status and uptime monitoring
- API connectivity status
- System resource usage
- Recent system logs

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your SWING_BOT outputs are available in the `outputs/` directory

## Usage

### Method 1: Using the launcher script
```bash
python run_dashboard.py
```

### Method 2: Direct Streamlit run
```bash
streamlit run dashboard.py
```

The dashboard will be available at: http://localhost:8501

## Data Sources

The dashboard reads data from the following files in the `outputs/` directory:

- `live_positions.json` - Current active positions
- `live_trades.jsonl` - Trade history
- `gtt/latest_gtt_orders.json` - Active GTT orders
- `optimization_history.json` - Parameter optimization history
- `rl_performance.json` - RL agent performance data
- `news_sentiment.json` - News sentiment analysis
- Various log files for system health monitoring

## Configuration

The dashboard automatically detects:
- Market regime status
- API connectivity (Upstox, News API, Telegram)
- System resources (CPU, memory usage)
- Last run timestamps

## Auto-Refresh

The dashboard includes:
- Manual refresh button
- Auto-refresh indicator (60-second intervals recommended)
- Real-time data updates from SWING_BOT outputs

## Troubleshooting

### No Data Displayed
- Ensure SWING_BOT has been run and generated output files
- Check that the `outputs/` directory exists and contains data files
- Verify file permissions

### Dashboard Won't Start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Ensure port 8501 is available
- Try running with: `streamlit run dashboard.py --server.port 8502`

### Performance Issues
- The dashboard reads files on each refresh - ensure fast storage
- For large datasets, consider pagination or sampling
- Monitor system resources in the System Health tab

## Development

### Adding New Metrics
1. Add data loading logic in the respective tab function
2. Create visualization using Plotly
3. Add metrics display using Streamlit columns

### Custom Styling
- Modify the CSS in the `st.markdown()` section at the top of `dashboard.py`
- Use Streamlit's theming options for consistent styling

### Extending Tabs
- Add new tab functions following the existing pattern
- Update the main `st.tabs()` call to include the new tab
- Ensure data loading handles missing files gracefully

## Security Notes

- This dashboard is intended for local development and monitoring
- Do not expose the dashboard to public networks without proper authentication
- API keys and sensitive data are not displayed in the dashboard
- Log files may contain sensitive information - review before sharing

## Support

For issues with the dashboard:
1. Check the System Health tab for error logs
2. Verify data file formats match expected schemas
3. Ensure all dependencies are correctly installed
4. Check SWING_BOT output generation is working properly

---

*Built with Streamlit and Plotly for the SWING_BOT autonomous trading system.*