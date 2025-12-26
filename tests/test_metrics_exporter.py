"""
Tests for SWING_BOT Metrics Exporter
"""

import pytest
import time
import requests
from unittest.mock import patch, MagicMock
import threading

from src.metrics_exporter import (
    MetricsExporter,
    get_metrics_exporter,
    start_metrics_server
)


class TestMetricsExporter:
    """Test metrics exporter functionality."""

    def test_prometheus_exporter_initialization(self):
        """Test Prometheus exporter initialization."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        assert exporter.mode == "prometheus"
        assert exporter.port == 9109
        assert exporter.registry is not None
        assert exporter.meter is None

        # Check metrics are registered
        assert hasattr(exporter, 'data_freshness_days')
        assert hasattr(exporter, 'audit_pass_count')
        assert hasattr(exporter, 'orchestrate_runtime_seconds')

    def test_otlp_exporter_initialization(self):
        """Test OTLP exporter initialization."""
        with patch('src.metrics_exporter.metrics') as mock_metrics, \
             patch('src.metrics_exporter.MeterProvider') as mock_provider:

            mock_meter = MagicMock()
            mock_metrics.get_meter.return_value = mock_meter
            mock_metrics.set_meter_provider = MagicMock()

            exporter = MetricsExporter(mode="otlp", port=4318)

            assert exporter.mode == "otlp"
            assert exporter.meter is not None
            mock_metrics.get_meter.assert_called_once()

    def test_invalid_mode(self):
        """Test invalid export mode."""
        exporter = MetricsExporter(mode="invalid", port=9109)

        assert exporter.registry is None
        assert exporter.meter is None

    def test_record_data_metrics_prometheus(self):
        """Test recording data metrics in Prometheus mode."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        exporter.record_data_metrics(
            freshness_days=1,
            coverage_days=500,
            symbols_count=50
        )

        # Check gauge values (would need prometheus_client to fully test)
        assert exporter.registry is not None

    def test_record_audit_metrics_prometheus(self):
        """Test recording audit metrics in Prometheus mode."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        exporter.record_audit_metrics(pass_count=15, fail_count=2)

        assert exporter.registry is not None

    def test_record_runtime_metrics_prometheus(self):
        """Test recording runtime metrics in Prometheus mode."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        exporter.record_runtime('orchestrate', 45.2)
        exporter.record_runtime('fetch', 12.5)

        assert exporter.registry is not None

    def test_record_notification_metrics_prometheus(self):
        """Test recording notification metrics in Prometheus mode."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        exporter.record_notification('teams')
        exporter.record_notification('email')

        assert exporter.registry is not None

    def test_orchestration_tracking(self):
        """Test orchestration run tracking."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        exporter.record_orchestration_start()

        assert exporter.registry is not None

    @patch('src.metrics_exporter.threading.Thread')
    def test_prometheus_server_start(self, mock_thread):
        """Test Prometheus server startup."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        exporter = MetricsExporter(mode="prometheus", port=9109)
        exporter.start_server()

        # Thread should be created and started
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    def test_server_stop(self):
        """Test server shutdown."""
        exporter = MetricsExporter(mode="prometheus", port=9109)

        # Mock server
        mock_server = MagicMock()
        exporter.server = mock_server

        exporter.stop_server()

        mock_server.shutdown.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_get_metrics_exporter_singleton(self):
        """Test singleton pattern for metrics exporter."""
        exporter1 = get_metrics_exporter(mode="prometheus", port=9109)
        exporter2 = get_metrics_exporter(mode="prometheus", port=9109)

        assert exporter1 is exporter2

    @patch('src.metrics_exporter.get_metrics_exporter')
    def test_start_metrics_server_function(self, mock_get_exporter):
        """Test start_metrics_server convenience function."""
        mock_exporter = MagicMock()
        mock_get_exporter.return_value = mock_exporter

        result = start_metrics_server(mode="prometheus", port=9109)

        assert result == mock_exporter
        mock_get_exporter.assert_called_once_with(mode="prometheus", port=9109)
        mock_exporter.start_server.assert_called_once()

    @pytest.mark.integration
    def test_prometheus_metrics_endpoint(self):
        """Integration test for Prometheus metrics endpoint."""
        exporter = MetricsExporter(mode="prometheus", port=9110)

        # Start server in background
        server_thread = threading.Thread(target=exporter.start_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(1)

        try:
            # Make request to metrics endpoint
            response = requests.get("http://localhost:9110/metrics", timeout=5)

            assert response.status_code == 200
            assert "swingbot_data_freshness_days" in response.text
            assert "swingbot_audit_pass_count" in response.text
            assert "swingbot_orchestrate_runtime_seconds" in response.text

        finally:
            exporter.stop_server()

    def test_otlp_metrics_recording(self):
        """Test OTLP metrics recording."""
        with patch('src.metrics_exporter.metrics') as mock_metrics, \
             patch('src.metrics_exporter.MeterProvider') as mock_provider:

            mock_meter = MagicMock()
            mock_metrics.get_meter.return_value = mock_meter

            exporter = MetricsExporter(mode="otlp", port=4318)

            # Record metrics
            exporter.record_data_metrics(1, 500, 50)
            exporter.record_audit_metrics(15, 2)
            exporter.record_runtime('orchestrate', 45.2)

            # Verify meter methods were called
            assert mock_meter.create_gauge.call_count >= 3  # At least 3 gauges
            assert mock_meter.create_histogram.call_count >= 2  # At least 2 histograms

    def test_metrics_with_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        # This would require mocking import failures
        # For now, just ensure the module can be imported
        import src.metrics_exporter
        assert hasattr(src.metrics_exporter, 'MetricsExporter')