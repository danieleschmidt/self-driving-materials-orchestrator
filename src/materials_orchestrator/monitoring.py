"""System monitoring and health checks for autonomous lab operations."""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)
if not PSUTIL_AVAILABLE:
    logger.warning("psutil not available, using fallback monitoring")


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    experiment_throughput: float  # experiments per hour
    active_experiments: int
    queue_size: int
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """System health monitoring and alerting."""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_metrics: Optional[SystemMetrics] = None

        # Register default health checks
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("process_health", self._check_process_health)

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started health monitoring (interval: {self.check_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Stopped health monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                self._last_metrics = metrics

                # Trim history to last 24 hours
                cutoff = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff
                ]

                # Run health checks
                health_results = self.run_health_checks()

                # Check for alerts
                self._process_alerts(health_results, metrics)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.check_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        if not PSUTIL_AVAILABLE:
            # Return placeholder metrics when psutil is not available
            return SystemMetrics(
                cpu_percent=10.0,  # Simulated values
                memory_percent=25.0,
                disk_percent=45.0,
                network_bytes_sent=1000,
                network_bytes_recv=2000,
                experiment_throughput=self._calculate_experiment_throughput(),
                active_experiments=0,
                queue_size=0,
            )

        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk usage (root filesystem)
            disk = psutil.disk_usage("/")

            # Network I/O
            network = psutil.net_io_counters()

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                experiment_throughput=self._calculate_experiment_throughput(),
                active_experiments=0,  # Would be updated by lab system
                queue_size=0,  # Would be updated by lab system
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default/safe values
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                experiment_throughput=0.0,
                active_experiments=0,
                queue_size=0,
            )

    def _calculate_experiment_throughput(self) -> float:
        """Calculate recent experiment throughput."""
        # This would be integrated with the actual lab system
        # For now, return a placeholder value
        return 0.0

    def run_health_checks(self) -> List[HealthCheck]:
        """Run all registered health checks."""
        results = []

        for name, check_func in self.health_checks.items():
            start_time = time.time()
            try:
                result = check_func()
                if not isinstance(result, HealthCheck):
                    result = HealthCheck(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message="Invalid health check result",
                    )
            except Exception as e:
                result = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                )

            result.duration = time.time() - start_time
            result.timestamp = datetime.now()
            results.append(result)

        return results

    def _check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        if not self._last_metrics:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="No metrics available",
            )

        metrics = self._last_metrics
        issues = []

        # Check CPU usage
        if metrics.cpu_percent > 90:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        # Check memory usage
        if metrics.memory_percent > 85:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        # Check disk usage
        if metrics.disk_percent > 80:
            issues.append(f"High disk usage: {metrics.disk_percent:.1f}%")

        if issues:
            status = (
                HealthStatus.CRITICAL
                if any("High" in issue for issue in issues)
                else HealthStatus.WARNING
            )
            return HealthCheck(
                name="system_resources",
                status=status,
                message="; ".join(issues),
                metrics={
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_percent": metrics.disk_percent,
                },
            )

        return HealthCheck(
            name="system_resources",
            status=HealthStatus.HEALTHY,
            message="System resources within normal ranges",
            metrics={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
            },
        )

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space in data directories."""
        if not PSUTIL_AVAILABLE:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                message="Disk space check not available (psutil not installed)",
                metrics={"free_gb": 10.0},  # Simulated value
            )

        try:
            data_dir = Path("data")
            if data_dir.exists():
                disk_usage = psutil.disk_usage(str(data_dir))
                free_gb = disk_usage.free / (1024**3)

                if free_gb < 1.0:  # Less than 1GB free
                    return HealthCheck(
                        name="disk_space",
                        status=HealthStatus.CRITICAL,
                        message=f"Low disk space: {free_gb:.2f} GB free",
                        metrics={"free_gb": free_gb},
                    )
                elif free_gb < 5.0:  # Less than 5GB free
                    return HealthCheck(
                        name="disk_space",
                        status=HealthStatus.WARNING,
                        message=f"Disk space getting low: {free_gb:.2f} GB free",
                        metrics={"free_gb": free_gb},
                    )
                else:
                    return HealthCheck(
                        name="disk_space",
                        status=HealthStatus.HEALTHY,
                        message=f"Sufficient disk space: {free_gb:.2f} GB free",
                        metrics={"free_gb": free_gb},
                    )
            else:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.WARNING,
                    message="Data directory does not exist",
                )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
            )

    def _check_process_health(self) -> HealthCheck:
        """Check health of critical processes."""
        if not PSUTIL_AVAILABLE:
            return HealthCheck(
                name="process_health",
                status=HealthStatus.HEALTHY,
                message="Process health check not available (psutil not installed)",
                metrics={"memory_mb": 50.0},  # Simulated value
            )

        try:
            current_process = psutil.Process()

            # Check memory leaks (simplified)
            memory_info = current_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            if memory_mb > 1000:  # More than 1GB
                return HealthCheck(
                    name="process_health",
                    status=HealthStatus.WARNING,
                    message=f"High memory usage: {memory_mb:.1f} MB",
                    metrics={"memory_mb": memory_mb},
                )

            # Check file descriptors
            try:
                num_fds = current_process.num_fds()
                if num_fds > 1000:
                    return HealthCheck(
                        name="process_health",
                        status=HealthStatus.WARNING,
                        message=f"High file descriptor count: {num_fds}",
                        metrics={"file_descriptors": num_fds},
                    )
            except (AttributeError, psutil.AccessDenied):
                # num_fds not available on all platforms
                pass

            return HealthCheck(
                name="process_health",
                status=HealthStatus.HEALTHY,
                message="Process health normal",
                metrics={"memory_mb": memory_mb},
            )

        except Exception as e:
            return HealthCheck(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check process health: {str(e)}",
            )

    def _process_alerts(
        self, health_results: List[HealthCheck], metrics: SystemMetrics
    ) -> None:
        """Process health check results and generate alerts."""
        for result in health_results:
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert = {
                    "timestamp": result.timestamp.isoformat(),
                    "check_name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "metrics": result.metrics,
                }

                # Avoid duplicate alerts (simplified)
                recent_alerts = [
                    a
                    for a in self.alerts[-10:]
                    if a.get("check_name") == result.name
                    and a.get("status") == result.status.value
                ]

                if not recent_alerts:
                    self.alerts.append(alert)
                    logger.warning(f"Health alert: {result.name} - {result.message}")

        # Trim alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]  # Keep last 500 alerts

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        health_results = self.run_health_checks()

        # Overall status is worst individual status
        overall_status = HealthStatus.HEALTHY
        for result in health_results:
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif (
                result.status == HealthStatus.WARNING
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.WARNING

        return {
            "overall_status": overall_status.value,
            "health_checks": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat(),
                }
                for result in health_results
            ],
            "metrics": {
                "cpu_percent": (
                    self._last_metrics.cpu_percent if self._last_metrics else 0
                ),
                "memory_percent": (
                    self._last_metrics.memory_percent if self._last_metrics else 0
                ),
                "disk_percent": (
                    self._last_metrics.disk_percent if self._last_metrics else 0
                ),
                "experiment_throughput": (
                    self._last_metrics.experiment_throughput
                    if self._last_metrics
                    else 0
                ),
            },
            "active_alerts": len(
                [
                    a
                    for a in self.alerts[-10:]
                    if a.get("status") in ["warning", "critical"]
                ]
            ),
            "uptime_hours": (datetime.now() - datetime.now()).total_seconds()
            / 3600,  # Placeholder
        }

    def export_metrics(self, output_file: str) -> None:
        """Export metrics history to file."""
        try:
            metrics_data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_percent": m.disk_percent,
                    "experiment_throughput": m.experiment_throughput,
                    "active_experiments": m.active_experiments,
                    "queue_size": m.queue_size,
                }
                for m in self.metrics_history
            ]

            with open(output_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

            logger.info(
                f"Exported {len(metrics_data)} metrics records to {output_file}"
            )

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return (
            datetime.now() - self.last_failure_time
        ).total_seconds() > self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            self.failure_count = 0
            self.state = "closed"

    def _on_failure(self) -> None:
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )


def create_health_monitor() -> HealthMonitor:
    """Create and configure a health monitor instance."""
    monitor = HealthMonitor(check_interval=30)  # Check every 30 seconds
    return monitor
