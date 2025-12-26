# SWING_BOT Dashboard

The SWING_BOT Dashboard provides a comprehensive visual summary of daily trading operations, combining data quality metrics, audit results, and position details in an easy-to-share HTML format.

## Overview

The dashboard is automatically generated during EOD orchestration when the `--dashboard` flag is used. It creates a self-contained HTML file that can be:

- Opened directly in any web browser
- Shared via email or file sharing
- Hosted on internal web servers
- Attached to Teams messages

## Dashboard Sections

### Header Section
- **Title**: SWING_BOT Daily Dashboard
- **Date**: Latest trading date processed
- **Generation Timestamp**: When the dashboard was created
- **Runtime**: Total orchestration execution time

### KPI Cards
Four key metric cards display critical system health indicators:

1. **Audit Pass Rate**: Percentage of positions passing audit validation
2. **Data Freshness**: Days since latest market data
3. **Symbols Tracked**: Number of symbols in the dataset
4. **Coverage Days**: Historical trading days available

### Top Positions Table
Displays the top 10 trading positions with:
- Symbol name
- Entry price (₹)
- Stop loss price (₹)
- Target price (₹)
- Decision confidence score
- Audit status (PASS/FAIL)

### Audit Issues Summary
Groups and summarizes audit failures by issue type:
- Issue description
- Count of occurrences
- Suggested fix actions

## File Location

```
outputs/dashboard/index.html
```

## Opening the Dashboard

### Method 1: Direct File Open
```bash
# On Windows
start outputs/dashboard/index.html

# On Linux/Mac
xdg-open outputs/dashboard/index.html
open outputs/dashboard/index.html
```

### Method 2: Web Server
```bash
# Python simple server
cd outputs/dashboard
python -m http.server 8000

# Then open: http://localhost:8000/index.html
```

### Method 3: File Share
- Copy `outputs/dashboard/index.html` to network share
- Email as attachment
- Upload to SharePoint/OneDrive

## Sharing in Teams

The dashboard can be linked in Teams notifications:

```python
file_links = {
    'Dashboard': f'file://{Path(dashboard_html).absolute()}'
}
```

## Customization

### Styling
The dashboard uses inline CSS for portability. To customize:
1. Edit `src/dashboards/teams_dashboard.py`
2. Modify the CSS string in `build_daily_html()`
3. Regenerate dashboard

### Content
To add new sections:
1. Update the HTML template in `build_daily_html()`
2. Pass additional data parameters
3. Modify the data processing logic

## Troubleshooting

### Dashboard Not Generated
- Ensure `--dashboard` flag is used in `orchestrate-eod`
- Check that all required CSV files exist
- Verify write permissions to `outputs/dashboard/`

### Display Issues
- Use modern web browser (Chrome, Firefox, Edge)
- Check console for JavaScript errors
- Ensure CSS is not blocked

### Sharing Issues
- Use absolute file paths for Teams links
- Convert to HTTPS URLs for external access
- Check file permissions for shared access

## Integration

The dashboard integrates with:
- **Teams Notifications**: Links included in Adaptive Cards
- **Email Fallbacks**: Can be attached to email notifications
- **Monitoring Systems**: Metrics displayed in dashboard KPIs

## Automation

### Daily Generation
```bash
python -m src.cli orchestrate-eod --dashboard [other-flags]
```

### Scheduled Cleanup
Old dashboards are retained for 30 days by default. Configure cleanup:

```bash
# Remove dashboards older than 30 days
find outputs/dashboard -name "*.html" -mtime +30 -delete
```

## Performance

- **Generation Time**: < 5 seconds
- **File Size**: ~50-200KB depending on position count
- **Load Time**: Instant (self-contained, no external dependencies)
- **Compatibility**: Works on all modern browsers and mobile devices</content>
<parameter name="filePath">c:\Users\K01340\SWING_BOT_GIT\SWING_BOT\docs\DASHBOARD.md