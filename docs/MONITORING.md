# SWING_BOT Monitoring & Metrics

SWING_BOT includes comprehensive monitoring capabilities using Prometheus metrics and Grafana dashboards for real-time observability of trading operations.

## Architecture

The monitoring system consists of:

1. **Metrics Exporter**: Collects and exposes system metrics
2. **Prometheus**: Time-series database for metric storage
3. **Grafana**: Visualization dashboard for metrics
4. **OpenTelemetry (OTLP)**: Alternative metrics export protocol

## Metrics Exporter

### Starting the Exporter

```bash
# Prometheus mode (default)
python -m src.cli metrics-exporter --port 9108 --mode prometheus

# OpenTelemetry OTLP mode
python -m src.cli metrics-exporter --port 4317 --mode otlp
```

### Available Metrics

#### Gauges
- `swingbot_data_freshness_days`: Days since latest market data
- `swingbot_coverage_days`: Historical trading days in dataset
- `swingbot_symbols_count`: Number of symbols being tracked
- `swingbot_audit_pass_count`: Positions passing audit
- `swingbot_audit_fail_count`: Positions failing audit

#### Histograms
- `swingbot_orchestrate_runtime_seconds`: EOD orchestration duration
  - Buckets: 10s, 30s, 60s, 120s, 300s, 600s
- `swingbot_fetch_runtime_seconds`: Data fetching duration
  - Buckets: 5s, 15s, 30s, 60s, 120s

#### Counters
- `swingbot_runs_total`: Total orchestration runs
- `swingbot_notifications_sent_total{channel}`: Notifications by channel
  - Labels: `channel=teams|email`

## Prometheus Setup

### Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'swingbot'
    static_configs:
      - targets: ['localhost:9108']
    scrape_interval: 30s
    metrics_path: /metrics
```

### Service File (Linux)

Create `/etc/systemd/system/prometheus-swingbot.service`:

```ini
[Unit]
Description=Prometheus SwingBot Metrics
After=network.target

[Service]
Type=simple
User=swingbot
ExecStart=/path/to/python -m src.cli metrics-exporter --port 9108
Restart=always

[Install]
WantedBy=multi-user.target
```

## Grafana Dashboard

### Import Dashboard

1. **Download JSON**: Copy `grafana/SWING_BOT_Dashboard.json`
2. **Grafana UI**: Dashboards → Import
3. **Upload JSON**: Paste the dashboard JSON
4. **Configure Data Source**: Select Prometheus data source
5. **Save Dashboard**

### Dashboard Panels

#### 1. Data Quality Metrics (Stat Panel)
- **Data Freshness**: Current days since latest data
- **Coverage Days**: Historical data depth
- **Symbols Count**: Active symbols tracked
- **Thresholds**: Red > 2 days for freshness

#### 2. Audit Results (Bar Gauge)
- **Pass Count**: Green bars for successful audits
- **Fail Count**: Red bars for failed audits
- **Dynamic Scaling**: Auto-adjusts to data range

#### 3. Orchestration Runtime (Histogram)
- **Distribution**: Shows runtime distribution across runs
- **Percentiles**: P50, P95, P99 markers
- **Alert Threshold**: Red > 60 seconds

#### 4. Data Fetch Runtime (Histogram)
- **Fetch Performance**: Data retrieval duration distribution
- **Network Monitoring**: Identifies connectivity issues

#### 5. Run Frequency (Stat Panel)
- **Runs per Hour**: Calculated from counter metrics
- **Expected**: ~0.04 runs/hour (Mon-Fri 15:30 IST)
- **Anomaly Detection**: Alerts on missed runs

#### 6. Notifications Sent (Table)
- **Channel Breakdown**: Teams vs Email notifications
- **Success Rate**: Compare sent vs attempted
- **Delivery Monitoring**: Track notification reliability

#### 7. System Health (Status History)
- **Uptime**: Service availability over time
- **Connectivity**: Metrics collection status

## OpenTelemetry (OTLP) Setup

### Configuration

Set environment variables:

```bash
export OTEL_SERVICE_NAME=swingbot
export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true
```

### Collector Configuration

`otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  prometheus:
    endpoint: "0.0.0.0:9108"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
```

## Alerting

### Prometheus Alert Rules

Create `alert_rules.yml`:

```yaml
groups:
  - name: swingbot
    rules:
      - alert: SwingBotDataStale
        expr: swingbot_data_freshness_days > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "SWING_BOT data is stale"
          description: "Market data is {{ $value }} days old"

      - alert: SwingBotAuditFailures
        expr: swingbot_audit_fail_count > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High audit failure rate"
          description: "{{ $value }} positions failed audit"

      - alert: SwingBotRuntimeHigh
        expr: histogram_quantile(0.95, rate(swingbot_orchestrate_runtime_seconds_bucket[5m])) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "SWING_BOT runtime is high"
          description: "95th percentile runtime is {{ $value }}s"
```

### Grafana Alerts

Configure alerts directly in Grafana dashboard panels:

1. **Panel Options** → **Alert**
2. **Set Conditions**: Query metrics with thresholds
3. **Configure Notifications**: Email, Slack, Teams integration

## Integration with Orchestration

### Automatic Metrics Collection

```bash
# Enable metrics during orchestration
python -m src.cli orchestrate-eod --metrics [other-flags]
```

### Background Metrics Server

```bash
# Start metrics server separately
python -m src.cli metrics-exporter --port 9108 &

# Run orchestration
python -m src.cli orchestrate-eod --metrics
```

## Troubleshooting

### Metrics Not Appearing

1. **Check Exporter**:
   ```bash
   curl http://localhost:9108/metrics
   ```

2. **Verify Prometheus**:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

3. **Check Grafana**: Ensure correct data source configuration

### High Cardinality

- Limit label combinations in counters
- Use histograms for timing distributions
- Aggregate metrics at collection time

### Performance Impact

- Metrics collection is lightweight (< 1% CPU)
- Use appropriate scrape intervals (30s recommended)
- Monitor exporter memory usage

## Security Considerations

### Network Security
- Bind metrics server to localhost only
- Use reverse proxy for external access
- Implement authentication if exposed

### Data Privacy
- Metrics contain no sensitive trading data
- Audit results are aggregated counts only
- No PII or financial details exposed

## Scaling

### Multiple Instances
- Use instance labels for multi-environment monitoring
- Aggregate metrics across instances
- Separate dashboards per environment

### High Availability
- Run multiple metric exporters
- Use Prometheus federation for aggregation
- Implement backup alerting channels</content>
<parameter name="filePath">c:\Users\K01340\SWING_BOT_GIT\SWING_BOT\docs\MONITORING.md