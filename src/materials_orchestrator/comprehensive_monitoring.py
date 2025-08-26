"""Comprehensive monitoring system with real-time metrics and alerting."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: Dict[str, str]


@dataclass
class Alert:
    """Alert definition and state."""

    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    message: str
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class ComprehensiveMonitoringSystem:
    """Advanced monitoring system with real-time metrics and alerting."""

    def __init__(self, retention_days: int = 7):
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.alerts = {}
        self.alert_handlers = []
        self.retention_days = retention_days
        self.monitoring_thread = None
        self.running = False
        self._lock = Lock()

        # System metrics
        self.system_metrics = {
            "experiments_total": 0,
            "experiments_successful": 0,
            "experiments_failed": 0,
            "avg_experiment_duration": 0,
            "active_experiments": 0,
            "queue_size": 0,
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
        }

        self._setup_default_alerts()
        logger.info(
            f"Monitoring system initialized with {retention_days} days retention"
        )

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric_point = MetricPoint(
                timestamp=datetime.now(), value=value, labels=labels or {}
            )
            self.metrics[name].append(metric_point)

            # Update system metrics
            if name in self.system_metrics:
                self.system_metrics[name] = value

    def increment_counter(
        self, name: str, value: float = 1, labels: Dict[str, str] = None
    ):
        """Increment a counter metric."""
        current_value = self.get_latest_metric(name) or 0
        self.record_metric(name, current_value + value, labels)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer metric (duration in seconds)."""
        self.record_metric(f"{name}_duration_seconds", duration, labels)
        self.increment_counter(f"{name}_total", 1, labels)

    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self._lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
        return None

    def get_metric_history(self, name: str, hours: int = 24) -> List[MetricPoint]:
        """Get metric history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            if name in self.metrics:
                return [
                    point
                    for point in self.metrics[name]
                    if point.timestamp > cutoff_time
                ]
        return []

    def get_metric_statistics(self, name: str, hours: int = 24) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(name, hours)

        if not history:
            return {}

        values = [point.value for point in history]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "change": values[-1] - values[0] if len(values) > 1 else 0,
        }

    def create_alert(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        message: str,
    ):
        """Create a new alert."""
        alert = Alert(
            name=name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            message=message,
        )
        self.alerts[name] = alert
        logger.info(f"Created alert: {name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def start_monitoring(self):
        """Start the monitoring background thread."""
        if self.running:
            return

        self.running = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Monitoring thread started")

    def stop_monitoring(self):
        """Stop the monitoring background thread."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring thread stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_alerts()
                self._cleanup_old_metrics()
                self._collect_system_metrics()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _check_alerts(self):
        """Check all alert conditions."""
        for alert in self.alerts.values():
            try:
                should_trigger = self._evaluate_alert_condition(alert)

                if should_trigger and not alert.active:
                    # Trigger alert
                    alert.active = True
                    alert.triggered_at = datetime.now()
                    self._trigger_alert(alert)

                elif not should_trigger and alert.active:
                    # Resolve alert
                    alert.active = False
                    alert.resolved_at = datetime.now()
                    self._resolve_alert(alert)

            except Exception as e:
                logger.error(f"Error evaluating alert {alert.name}: {e}")

    def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if an alert condition is met."""
        # Parse condition (simplified - could be more sophisticated)
        if "metric:" in alert.condition:
            metric_name = alert.condition.split("metric:")[1].split()[0]
            operator = (
                alert.condition.split()[1] if len(alert.condition.split()) > 1 else ">"
            )

            current_value = self.get_latest_metric(metric_name)
            if current_value is None:
                return False

            if operator == ">":
                return current_value > alert.threshold
            elif operator == "<":
                return current_value < alert.threshold
            elif operator == "=":
                return abs(current_value - alert.threshold) < 0.001
            elif operator == ">=":
                return current_value >= alert.threshold
            elif operator == "<=":
                return current_value <= alert.threshold

        return False

    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.message}")

        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        logger.info(f"ALERT RESOLVED: {alert.name}")

        # Could add resolved alert handlers here

    def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)

        with self._lock:
            for name, metric_deque in self.metrics.items():
                # Remove old points
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage (simplified)
            import os

            load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
            self.record_metric("system_cpu_load", load_avg)

            # Memory usage (basic estimation)
            import sys

            memory_mb = sys.getsizeof(self.metrics) / 1024 / 1024
            self.record_metric("system_memory_mb", memory_mb)

            # Disk usage
            disk_usage = 0  # Would implement actual disk usage check
            self.record_metric("system_disk_usage_percent", disk_usage)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _setup_default_alerts(self):
        """Set up default monitoring alerts."""
        # High failure rate alert
        self.create_alert(
            "high_failure_rate",
            "metric:experiment_failure_rate >",
            0.5,  # 50% failure rate
            AlertSeverity.ERROR,
            "Experiment failure rate is high (>50%)",
        )

        # Queue size alert
        self.create_alert(
            "large_queue",
            "metric:experiment_queue_size >",
            100,
            AlertSeverity.WARNING,
            "Experiment queue is getting large (>100)",
        )

        # System resource alerts
        self.create_alert(
            "high_cpu_usage",
            "metric:system_cpu_load >",
            4.0,
            AlertSeverity.WARNING,
            "System CPU load is high (>4.0)",
        )

        self.create_alert(
            "high_memory_usage",
            "metric:system_memory_mb >",
            1000,  # 1GB
            AlertSeverity.WARNING,
            "System memory usage is high (>1GB)",
        )

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self._get_system_status(),
            "recent_metrics": self._get_recent_metrics(),
            "active_alerts": self._get_active_alerts(),
            "performance_summary": self._get_performance_summary(),
            "experiment_statistics": self._get_experiment_statistics(),
        }

        return dashboard

    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        active_alerts = [alert for alert in self.alerts.values() if alert.active]

        if any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
            status = "critical"
        elif any(alert.severity == AlertSeverity.ERROR for alert in active_alerts):
            status = "error"
        elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "active_alerts_count": len(active_alerts),
            "uptime_hours": (
                datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)
            ).total_seconds()
            / 3600,
            "monitoring_active": self.running,
        }

    def _get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent metric values."""
        recent_metrics = {}

        key_metrics = [
            "experiments_total",
            "experiments_successful",
            "experiments_failed",
            "avg_experiment_duration",
            "active_experiments",
            "queue_size",
            "system_cpu_load",
            "system_memory_mb",
        ]

        for metric in key_metrics:
            stats = self.get_metric_statistics(metric, 1)  # Last hour
            if stats:
                recent_metrics[metric] = stats

        return recent_metrics

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        active_alerts = []

        for alert in self.alerts.values():
            if alert.active:
                active_alerts.append(
                    {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "triggered_at": (
                            alert.triggered_at.isoformat()
                            if alert.triggered_at
                            else None
                        ),
                        "duration_minutes": (
                            (datetime.now() - alert.triggered_at).total_seconds() / 60
                            if alert.triggered_at
                            else 0
                        ),
                    }
                )

        return active_alerts

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        experiment_stats = self.get_metric_statistics("experiments_total", 24)
        success_stats = self.get_metric_statistics("experiments_successful", 24)
        duration_stats = self.get_metric_statistics("avg_experiment_duration", 24)

        success_rate = 0
        if experiment_stats and experiment_stats["latest"] > 0:
            success_rate = (success_stats["latest"] / experiment_stats["latest"]) * 100

        return {
            "experiments_per_hour": (
                experiment_stats.get("change", 0) / 24 if experiment_stats else 0
            ),
            "success_rate_percent": success_rate,
            "avg_duration_minutes": (
                duration_stats.get("avg", 0) / 60 if duration_stats else 0
            ),
            "throughput_trend": (
                "up"
                if experiment_stats and experiment_stats.get("change", 0) > 0
                else "stable"
            ),
        }

    def _get_experiment_statistics(self) -> Dict[str, Any]:
        """Get detailed experiment statistics."""
        return {
            "total_experiments": self.system_metrics.get("experiments_total", 0),
            "successful_experiments": self.system_metrics.get(
                "experiments_successful", 0
            ),
            "failed_experiments": self.system_metrics.get("experiments_failed", 0),
            "active_experiments": self.system_metrics.get("active_experiments", 0),
            "queue_size": self.system_metrics.get("queue_size", 0),
            "avg_duration_seconds": self.system_metrics.get(
                "avg_experiment_duration", 0
            ),
        }


class AlertManager:
    """Manages alert notifications and escalations."""

    def __init__(self):
        self.notification_handlers = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.ERROR: [],
            AlertSeverity.CRITICAL: [],
        }

        logger.info("Alert manager initialized")

    def add_notification_handler(
        self, severity: AlertSeverity, handler: Callable[[Alert], None]
    ):
        """Add notification handler for specific severity."""
        self.notification_handlers[severity].append(handler)

    def handle_alert(self, alert: Alert):
        """Handle alert notification."""
        handlers = self.notification_handlers.get(alert.severity, [])

        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")

    def send_email_notification(self, alert: Alert, recipients: List[str]):
        """Send email notification (placeholder)."""
        logger.info(f"EMAIL ALERT: {alert.name} to {recipients}")

    def send_slack_notification(self, alert: Alert, channel: str):
        """Send Slack notification (placeholder)."""
        logger.info(f"SLACK ALERT: {alert.name} to {channel}")


def create_comprehensive_monitoring() -> (
    Tuple[ComprehensiveMonitoringSystem, AlertManager]
):
    """Factory function to create monitoring system."""
    monitoring = ComprehensiveMonitoringSystem()
    alert_manager = AlertManager()

    # Connect alert manager to monitoring system
    monitoring.add_alert_handler(alert_manager.handle_alert)

    # Start monitoring
    monitoring.start_monitoring()

    logger.info("Comprehensive monitoring system created and started")
    return monitoring, alert_manager
