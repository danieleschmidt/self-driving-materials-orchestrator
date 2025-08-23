"""Advanced performance monitoring and optimization system."""

import asyncio
import functools
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "gt"  # gt, lt, eq
    enabled: bool = True


class PerformanceTracker:
    """High-performance metrics tracking system."""

    def __init__(self, max_metrics_per_series: int = 10000):
        """Initialize performance tracker.
        
        Args:
            max_metrics_per_series: Maximum metrics to keep per series
        """
        self.max_metrics_per_series = max_metrics_per_series
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_metrics_per_series)
        )
        self._lock = threading.RLock()
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._alerts: List[Dict[str, Any]] = []

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping
            metadata: Optional additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            self._metrics[name].append(metric)
            self._check_thresholds(metric)

    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds thresholds.
        
        Args:
            metric: Metric to check
        """
        threshold = self._thresholds.get(metric.name)
        if not threshold or not threshold.enabled:
            return

        exceeded_level = None

        if threshold.comparison_operator == "gt":
            if metric.value > threshold.critical_threshold:
                exceeded_level = "critical"
            elif metric.value > threshold.warning_threshold:
                exceeded_level = "warning"
        elif threshold.comparison_operator == "lt":
            if metric.value < threshold.critical_threshold:
                exceeded_level = "critical"
            elif metric.value < threshold.warning_threshold:
                exceeded_level = "warning"

        if exceeded_level:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "level": exceeded_level,
                "metric_name": metric.name,
                "metric_value": metric.value,
                "threshold": threshold.critical_threshold if exceeded_level == "critical" else threshold.warning_threshold,
                "tags": metric.tags,
            }

            self._alerts.append(alert)

            logger.warning(
                f"Performance threshold {exceeded_level}: {metric.name}={metric.value} "
                f"(threshold: {alert['threshold']})"
            )

    def set_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison_operator: str = "gt"
    ) -> None:
        """Set performance threshold for a metric.
        
        Args:
            metric_name: Name of metric
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            comparison_operator: Comparison operator (gt, lt, eq)
        """
        self._thresholds[metric_name] = PerformanceThreshold(
            metric_name=metric_name,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            comparison_operator=comparison_operator
        )

    def get_metrics(
        self,
        metric_name: str,
        since: Optional[datetime] = None,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> List[PerformanceMetric]:
        """Get metrics for analysis.
        
        Args:
            metric_name: Metric name to retrieve
            since: Only return metrics after this time
            tags_filter: Filter by tags
            
        Returns:
            List of matching metrics
        """
        with self._lock:
            metrics = list(self._metrics.get(metric_name, []))

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        if tags_filter:
            def matches_tags(metric_tags: Dict[str, str]) -> bool:
                return all(
                    metric_tags.get(key) == value
                    for key, value in tags_filter.items()
                )
            metrics = [m for m in metrics if matches_tags(m.tags)]

        return metrics

    def get_statistics(
        self,
        metric_name: str,
        window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get statistical summary of metrics.
        
        Args:
            metric_name: Metric name
            window: Time window for analysis
            
        Returns:
            Statistical summary
        """
        since = datetime.now() - window if window else None
        metrics = self.get_metrics(metric_name, since=since)

        if not metrics:
            return {}

        values = [m.value for m in metrics]
        values.sort()

        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": values[len(values) // 2],
        }

        # Calculate percentiles
        for percentile in [50, 90, 95, 99]:
            index = int(len(values) * percentile / 100)
            if index < len(values):
                stats[f"p{percentile}"] = values[index]

        return stats

    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance alerts.
        
        Args:
            since: Only return alerts after this time
            
        Returns:
            List of alerts
        """
        alerts = self._alerts.copy()

        if since:
            alerts = [
                alert for alert in alerts
                if datetime.fromisoformat(alert["timestamp"]) >= since
            ]

        return alerts


# Global performance tracker
_global_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker.
    
    Returns:
        Performance tracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()

        # Set default thresholds
        _global_tracker.set_threshold("response_time_ms", 1000, 5000, "gt")
        _global_tracker.set_threshold("memory_usage_mb", 1024, 2048, "gt")
        _global_tracker.set_threshold("error_rate", 0.05, 0.1, "gt")
        _global_tracker.set_threshold("cpu_usage", 0.8, 0.95, "gt")

    return _global_tracker


def performance_monitor(
    metric_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    track_memory: bool = False,
    track_cpu: bool = False
):
    """Decorator for automatic performance monitoring.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
        tags: Tags to attach to metrics
        track_memory: Whether to track memory usage
        track_cpu: Whether to track CPU usage
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            name = metric_name or f"{func.__module__}.{func.__name__}"

            # Start monitoring
            start_time = time.perf_counter()
            start_memory = None
            start_cpu = None

            if track_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass

            if track_cpu:
                try:
                    import psutil
                    start_cpu = psutil.cpu_percent()
                except ImportError:
                    pass

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success metrics
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                tracker.record_metric(
                    f"{name}_execution_time_ms",
                    execution_time,
                    tags={**(tags or {}), "status": "success"}
                )

                if track_memory and start_memory is not None:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = end_memory - start_memory
                        tracker.record_metric(
                            f"{name}_memory_delta_mb",
                            memory_delta,
                            tags=tags
                        )
                    except ImportError:
                        pass

                if track_cpu and start_cpu is not None:
                    try:
                        import psutil
                        end_cpu = psutil.cpu_percent()
                        tracker.record_metric(
                            f"{name}_cpu_usage",
                            end_cpu,
                            tags=tags
                        )
                    except ImportError:
                        pass

                return result

            except Exception as e:
                # Record error metrics
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                tracker.record_metric(
                    f"{name}_execution_time_ms",
                    execution_time,
                    tags={**(tags or {}), "status": "error", "error_type": type(e).__name__}
                )

                tracker.record_metric(
                    f"{name}_error_count",
                    1,
                    tags={**(tags or {}), "error_type": type(e).__name__}
                )

                raise

        return wrapper

    return decorator


class AsyncPerformanceMonitor:
    """Asynchronous performance monitoring system."""

    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        """Initialize async performance monitor.
        
        Args:
            tracker: Performance tracker to use
        """
        self.tracker = tracker or get_performance_tracker()
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        self._running = True

        # Start system monitoring
        self._monitoring_tasks["system"] = asyncio.create_task(
            self._monitor_system_metrics()
        )

        logger.info("Started async performance monitoring")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._running = False

        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)

        logger.info("Stopped async performance monitoring")

    async def _monitor_system_metrics(self) -> None:
        """Monitor system metrics continuously."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available - skipping system monitoring")
            return

        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.tracker.record_metric(
                    "system_cpu_usage",
                    cpu_percent / 100.0,
                    tags={"component": "system"}
                )

                # Memory usage
                memory = psutil.virtual_memory()
                self.tracker.record_metric(
                    "system_memory_usage",
                    memory.percent / 100.0,
                    tags={"component": "system"}
                )

                self.tracker.record_metric(
                    "system_memory_available_mb",
                    memory.available / 1024 / 1024,
                    tags={"component": "system"}
                )

                # Disk usage
                disk = psutil.disk_usage('/')
                self.tracker.record_metric(
                    "system_disk_usage",
                    disk.percent / 100.0,
                    tags={"component": "system"}
                )

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error


def monitor_async_function(
    metric_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
    """Decorator for async function performance monitoring.
    
    Args:
        metric_name: Custom metric name
        tags: Tags to attach to metrics
    
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            name = metric_name or f"{func.__module__}.{func.__name__}"

            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)

                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                tracker.record_metric(
                    f"{name}_async_execution_time_ms",
                    execution_time,
                    tags={**(tags or {}), "status": "success"}
                )

                return result

            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                tracker.record_metric(
                    f"{name}_async_execution_time_ms",
                    execution_time,
                    tags={**(tags or {}), "status": "error", "error_type": type(e).__name__}
                )

                raise

        return wrapper

    return decorator


# Context manager for performance tracking
class PerformanceContext:
    """Context manager for tracking operation performance."""

    def __init__(
        self,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
        tracker: Optional[PerformanceTracker] = None
    ):
        """Initialize performance context.
        
        Args:
            operation_name: Name of operation
            tags: Tags to attach
            tracker: Performance tracker to use
        """
        self.operation_name = operation_name
        self.tags = tags or {}
        self.tracker = tracker or get_performance_tracker()
        self.start_time = None

    def __enter__(self):
        """Start tracking."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and record metrics."""
        if self.start_time is not None:
            execution_time = (time.perf_counter() - self.start_time) * 1000  # ms

            status = "error" if exc_type else "success"
            tags = {**self.tags, "status": status}

            if exc_type:
                tags["error_type"] = exc_type.__name__

            self.tracker.record_metric(
                f"{self.operation_name}_execution_time_ms",
                execution_time,
                tags=tags
            )


def create_performance_dashboard_data() -> Dict[str, Any]:
    """Create data structure for performance dashboard.
    
    Returns:
        Dashboard data
    """
    tracker = get_performance_tracker()

    # Get recent metrics (last hour)
    since = datetime.now() - timedelta(hours=1)

    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "alerts": tracker.get_alerts(since=since),
        "system_health": "healthy"
    }

    # Key metrics to display
    key_metrics = [
        "system_cpu_usage",
        "system_memory_usage",
        "system_disk_usage",
        "response_time_ms",
        "error_rate"
    ]

    for metric_name in key_metrics:
        stats = tracker.get_statistics(metric_name, window=timedelta(hours=1))
        if stats:
            dashboard_data["metrics"][metric_name] = stats

    # Determine system health
    critical_alerts = [a for a in dashboard_data["alerts"] if a["level"] == "critical"]
    warning_alerts = [a for a in dashboard_data["alerts"] if a["level"] == "warning"]

    if critical_alerts:
        dashboard_data["system_health"] = "critical"
    elif warning_alerts:
        dashboard_data["system_health"] = "warning"

    return dashboard_data
