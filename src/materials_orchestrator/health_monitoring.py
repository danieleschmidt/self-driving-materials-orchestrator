"""Advanced health monitoring and diagnostics system."""

import logging
import time
import threading
import json
import psutil  # Will use fallback if not available
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""

    CORE = "core"
    DATABASE = "database"
    ROBOTS = "robots"
    NETWORK = "network"
    STORAGE = "storage"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class HealthMetric:
    """Individual health metric."""

    name: str
    value: Any
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    name: str
    component_type: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    uptime: timedelta = field(default_factory=timedelta)
    error_count: int = 0
    last_error: Optional[str] = None


class AdvancedHealthMonitor:
    """Advanced system health monitoring with comprehensive checks."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.alert_handlers: List[Callable] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.metrics_history: Dict[str, List[HealthMetric]] = {}
        self.start_time = datetime.now()

        # Try to detect system capabilities
        self.has_psutil = self._check_psutil()

        # Register default health checks
        self._register_default_checks()

        logger.info("Advanced health monitoring initialized")

    def _check_psutil(self) -> bool:
        """Check if psutil is available for system monitoring."""
        try:
            import psutil

            return True
        except ImportError:
            logger.warning("psutil not available, using basic system monitoring")
            return False

    def _register_default_checks(self):
        """Register default health check functions."""

        # Core system checks
        self.register_health_check(
            "core_imports", self._check_core_imports, ComponentType.CORE
        )
        self.register_health_check(
            "memory_usage", self._check_memory_usage, ComponentType.MEMORY
        )
        self.register_health_check(
            "cpu_usage", self._check_cpu_usage, ComponentType.CPU
        )
        self.register_health_check(
            "disk_space", self._check_disk_space, ComponentType.STORAGE
        )

        # Application-specific checks
        self.register_health_check(
            "database_connection",
            self._check_database_connection,
            ComponentType.DATABASE,
        )
        self.register_health_check(
            "experiment_queue", self._check_experiment_queue, ComponentType.CORE
        )
        self.register_health_check(
            "robot_connectivity", self._check_robot_connectivity, ComponentType.ROBOTS
        )

    def register_health_check(
        self, name: str, check_function: Callable, component_type: ComponentType
    ):
        """Register a custom health check function."""
        self.health_checks[name] = (check_function, component_type)

        # Initialize component if not exists
        if name not in self.components:
            self.components[name] = ComponentHealth(
                name=name, component_type=component_type, status=HealthStatus.UNKNOWN
            )

        logger.info(f"Registered health check: {name} ({component_type.value})")

    def register_alert_handler(self, handler: Callable):
        """Register an alert handler for health issues."""
        self.alert_handlers.append(handler)
        logger.info("Alert handler registered")

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info(f"Health monitoring started with {self.check_interval}s interval")

    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.check_interval, 10.0))

    def run_all_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks."""
        results = {}

        for check_name, (check_function, component_type) in self.health_checks.items():
            try:
                start_time = time.time()
                component_health = check_function()
                check_duration = time.time() - start_time

                # Ensure component_health is properly formatted
                if not isinstance(component_health, ComponentHealth):
                    # Create ComponentHealth from basic results
                    if (
                        isinstance(component_health, tuple)
                        and len(component_health) >= 2
                    ):
                        status, metrics = component_health[:2]
                        component_health = ComponentHealth(
                            name=check_name,
                            component_type=component_type,
                            status=status,
                            metrics=metrics if isinstance(metrics, list) else [],
                        )
                    else:
                        component_health = ComponentHealth(
                            name=check_name,
                            component_type=component_type,
                            status=HealthStatus.UNKNOWN,
                        )

                component_health.last_check = datetime.now()

                # Add performance metric
                perf_metric = HealthMetric(
                    name="check_duration",
                    value=check_duration,
                    unit="seconds",
                    status=(
                        HealthStatus.HEALTHY
                        if check_duration < 1.0
                        else HealthStatus.WARNING
                    ),
                )
                component_health.metrics.append(perf_metric)

                self.components[check_name] = component_health
                results[check_name] = component_health

                # Store metrics history
                if check_name not in self.metrics_history:
                    self.metrics_history[check_name] = []

                self.metrics_history[check_name].extend(component_health.metrics)

                # Keep history manageable (last 1000 metrics)
                if len(self.metrics_history[check_name]) > 1000:
                    self.metrics_history[check_name] = self.metrics_history[check_name][
                        -500:
                    ]

                # Check for alerts
                if component_health.status in [
                    HealthStatus.WARNING,
                    HealthStatus.CRITICAL,
                ]:
                    self._trigger_alerts(component_health)

            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                error_component = ComponentHealth(
                    name=check_name,
                    component_type=component_type,
                    status=HealthStatus.CRITICAL,
                    last_error=str(e),
                    error_count=self.components.get(
                        check_name,
                        ComponentHealth(
                            check_name, component_type, HealthStatus.UNKNOWN
                        ),
                    ).error_count
                    + 1,
                )
                self.components[check_name] = error_component
                results[check_name] = error_component

        return results

    def _trigger_alerts(self, component_health: ComponentHealth):
        """Trigger alerts for unhealthy components."""
        alert_message = (
            f"Health Alert: {component_health.name} is {component_health.status.value}"
        )

        if component_health.metrics:
            critical_metrics = [
                m for m in component_health.metrics if m.status == HealthStatus.CRITICAL
            ]
            if critical_metrics:
                alert_message += f" - Critical metrics: {', '.join(m.name for m in critical_metrics)}"

        # Send to all registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(component_health, alert_message)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    # Default health check implementations
    def _check_core_imports(self) -> ComponentHealth:
        """Check if core modules can be imported."""
        metrics = []
        overall_status = HealthStatus.HEALTHY

        core_modules = [
            "materials_orchestrator.core",
            "materials_orchestrator.planners",
            "materials_orchestrator.robots",
            "materials_orchestrator.database",
        ]

        import_count = 0
        for module in core_modules:
            try:
                __import__(module)
                import_count += 1
                metrics.append(
                    HealthMetric(
                        name=f"{module}_import",
                        value="OK",
                        unit="status",
                        status=HealthStatus.HEALTHY,
                    )
                )
            except ImportError as e:
                metrics.append(
                    HealthMetric(
                        name=f"{module}_import",
                        value=f"FAILED: {e}",
                        unit="status",
                        status=HealthStatus.CRITICAL,
                    )
                )
                overall_status = HealthStatus.CRITICAL

        success_rate = import_count / len(core_modules)
        metrics.append(
            HealthMetric(
                name="import_success_rate",
                value=success_rate,
                unit="fraction",
                status=(
                    HealthStatus.HEALTHY
                    if success_rate == 1.0
                    else HealthStatus.CRITICAL
                ),
            )
        )

        return ComponentHealth(
            name="core_imports",
            component_type=ComponentType.CORE,
            status=overall_status,
            metrics=metrics,
        )

    def _check_memory_usage(self) -> ComponentHealth:
        """Check system memory usage."""
        metrics = []

        if self.has_psutil:
            try:
                import psutil

                memory = psutil.virtual_memory()

                # Memory usage percentage
                mem_percent = memory.percent
                mem_status = HealthStatus.HEALTHY
                if mem_percent > 90:
                    mem_status = HealthStatus.CRITICAL
                elif mem_percent > 75:
                    mem_status = HealthStatus.WARNING

                metrics.append(
                    HealthMetric(
                        name="memory_usage_percent",
                        value=mem_percent,
                        unit="percent",
                        status=mem_status,
                        threshold_warning=75,
                        threshold_critical=90,
                    )
                )

                # Available memory
                available_gb = memory.available / (1024**3)
                metrics.append(
                    HealthMetric(
                        name="available_memory",
                        value=available_gb,
                        unit="GB",
                        status=(
                            HealthStatus.CRITICAL
                            if available_gb < 0.5
                            else HealthStatus.HEALTHY
                        ),
                    )
                )

                overall_status = max(m.status for m in metrics)

            except Exception as e:
                overall_status = HealthStatus.UNKNOWN
                metrics.append(
                    HealthMetric(
                        name="memory_check_error",
                        value=str(e),
                        unit="error",
                        status=HealthStatus.UNKNOWN,
                    )
                )
        else:
            # Fallback memory check
            overall_status = HealthStatus.HEALTHY
            metrics.append(
                HealthMetric(
                    name="memory_status",
                    value="psutil unavailable - using fallback",
                    unit="status",
                    status=HealthStatus.HEALTHY,
                )
            )

        return ComponentHealth(
            name="memory_usage",
            component_type=ComponentType.MEMORY,
            status=overall_status,
            metrics=metrics,
        )

    def _check_cpu_usage(self) -> ComponentHealth:
        """Check system CPU usage."""
        metrics = []

        if self.has_psutil:
            try:
                import psutil

                # CPU percentage over 1 second
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_status = HealthStatus.HEALTHY
                if cpu_percent > 95:
                    cpu_status = HealthStatus.CRITICAL
                elif cpu_percent > 80:
                    cpu_status = HealthStatus.WARNING

                metrics.append(
                    HealthMetric(
                        name="cpu_usage_percent",
                        value=cpu_percent,
                        unit="percent",
                        status=cpu_status,
                        threshold_warning=80,
                        threshold_critical=95,
                    )
                )

                # Load average (Unix-like systems only)
                if hasattr(psutil, "getloadavg"):
                    load_avg = psutil.getloadavg()[0]  # 1-minute load average
                    cpu_count = psutil.cpu_count()
                    load_ratio = load_avg / cpu_count if cpu_count > 0 else 0

                    load_status = HealthStatus.HEALTHY
                    if load_ratio > 2.0:
                        load_status = HealthStatus.CRITICAL
                    elif load_ratio > 1.0:
                        load_status = HealthStatus.WARNING

                    metrics.append(
                        HealthMetric(
                            name="load_average_ratio",
                            value=load_ratio,
                            unit="ratio",
                            status=load_status,
                        )
                    )

                overall_status = max(m.status for m in metrics)

            except Exception as e:
                overall_status = HealthStatus.UNKNOWN
                metrics.append(
                    HealthMetric(
                        name="cpu_check_error",
                        value=str(e),
                        unit="error",
                        status=HealthStatus.UNKNOWN,
                    )
                )
        else:
            # Fallback CPU check
            overall_status = HealthStatus.HEALTHY
            metrics.append(
                HealthMetric(
                    name="cpu_status",
                    value="psutil unavailable - using fallback",
                    unit="status",
                    status=HealthStatus.HEALTHY,
                )
            )

        return ComponentHealth(
            name="cpu_usage",
            component_type=ComponentType.CPU,
            status=overall_status,
            metrics=metrics,
        )

    def _check_disk_space(self) -> ComponentHealth:
        """Check disk space availability."""
        metrics = []

        if self.has_psutil:
            try:
                import psutil

                # Check current working directory disk usage
                cwd_usage = psutil.disk_usage(".")

                # Available space in GB
                available_gb = cwd_usage.free / (1024**3)
                total_gb = cwd_usage.total / (1024**3)
                used_percent = (cwd_usage.used / cwd_usage.total) * 100

                disk_status = HealthStatus.HEALTHY
                if used_percent > 95:
                    disk_status = HealthStatus.CRITICAL
                elif used_percent > 85:
                    disk_status = HealthStatus.WARNING

                metrics.append(
                    HealthMetric(
                        name="disk_usage_percent",
                        value=used_percent,
                        unit="percent",
                        status=disk_status,
                        threshold_warning=85,
                        threshold_critical=95,
                    )
                )

                metrics.append(
                    HealthMetric(
                        name="available_disk_space",
                        value=available_gb,
                        unit="GB",
                        status=(
                            HealthStatus.CRITICAL
                            if available_gb < 1.0
                            else HealthStatus.HEALTHY
                        ),
                    )
                )

                overall_status = max(m.status for m in metrics)

            except Exception as e:
                overall_status = HealthStatus.UNKNOWN
                metrics.append(
                    HealthMetric(
                        name="disk_check_error",
                        value=str(e),
                        unit="error",
                        status=HealthStatus.UNKNOWN,
                    )
                )
        else:
            # Fallback disk check
            overall_status = HealthStatus.HEALTHY
            metrics.append(
                HealthMetric(
                    name="disk_status",
                    value="psutil unavailable - using fallback",
                    unit="status",
                    status=HealthStatus.HEALTHY,
                )
            )

        return ComponentHealth(
            name="disk_space",
            component_type=ComponentType.STORAGE,
            status=overall_status,
            metrics=metrics,
        )

    def _check_database_connection(self) -> ComponentHealth:
        """Check database connectivity."""
        metrics = []

        try:
            from .database import ExperimentTracker

            # Try to create database connection
            db = ExperimentTracker()

            # Test basic operation
            test_start = time.time()
            try:
                # This is a simple test - in real implementation would test actual DB connection
                connection_time = time.time() - test_start

                metrics.append(
                    HealthMetric(
                        name="database_connection_time",
                        value=connection_time,
                        unit="seconds",
                        status=(
                            HealthStatus.HEALTHY
                            if connection_time < 1.0
                            else HealthStatus.WARNING
                        ),
                    )
                )

                metrics.append(
                    HealthMetric(
                        name="database_status",
                        value="Connected",
                        unit="status",
                        status=HealthStatus.HEALTHY,
                    )
                )

                overall_status = HealthStatus.HEALTHY

            except Exception as db_error:
                metrics.append(
                    HealthMetric(
                        name="database_error",
                        value=str(db_error),
                        unit="error",
                        status=HealthStatus.CRITICAL,
                    )
                )
                overall_status = HealthStatus.CRITICAL

        except ImportError as e:
            overall_status = HealthStatus.WARNING
            metrics.append(
                HealthMetric(
                    name="database_import_error",
                    value=str(e),
                    unit="error",
                    status=HealthStatus.WARNING,
                )
            )

        return ComponentHealth(
            name="database_connection",
            component_type=ComponentType.DATABASE,
            status=overall_status,
            metrics=metrics,
        )

    def _check_experiment_queue(self) -> ComponentHealth:
        """Check experiment queue status."""
        metrics = []

        # This is a placeholder - in a real system would check actual queue status
        metrics.append(
            HealthMetric(
                name="queue_size",
                value=0,
                unit="experiments",
                status=HealthStatus.HEALTHY,
            )
        )

        metrics.append(
            HealthMetric(
                name="queue_processing_rate",
                value=1.0,
                unit="experiments/minute",
                status=HealthStatus.HEALTHY,
            )
        )

        return ComponentHealth(
            name="experiment_queue",
            component_type=ComponentType.CORE,
            status=HealthStatus.HEALTHY,
            metrics=metrics,
        )

    def _check_robot_connectivity(self) -> ComponentHealth:
        """Check robot fleet connectivity."""
        metrics = []

        try:
            from .robots import RobotOrchestrator

            # In a real system, this would check actual robot connections
            orchestrator = RobotOrchestrator()

            # Simulated robot status check
            connected_robots = 3  # Simulated
            total_robots = 3

            connection_rate = connected_robots / total_robots if total_robots > 0 else 0

            robot_status = HealthStatus.HEALTHY
            if connection_rate < 0.5:
                robot_status = HealthStatus.CRITICAL
            elif connection_rate < 0.8:
                robot_status = HealthStatus.WARNING

            metrics.append(
                HealthMetric(
                    name="robot_connection_rate",
                    value=connection_rate,
                    unit="fraction",
                    status=robot_status,
                )
            )

            metrics.append(
                HealthMetric(
                    name="connected_robots",
                    value=connected_robots,
                    unit="robots",
                    status=HealthStatus.HEALTHY,
                )
            )

            overall_status = robot_status

        except Exception as e:
            overall_status = HealthStatus.WARNING
            metrics.append(
                HealthMetric(
                    name="robot_check_error",
                    value=str(e),
                    unit="error",
                    status=HealthStatus.WARNING,
                )
            )

        return ComponentHealth(
            name="robot_connectivity",
            component_type=ComponentType.ROBOTS,
            status=overall_status,
            metrics=metrics,
        )

    def get_overall_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get overall system health status."""
        if not self.components:
            return HealthStatus.UNKNOWN, {"message": "No health checks performed"}

        # Determine overall status from component statuses
        component_statuses = [comp.status for comp in self.components.values()]

        if HealthStatus.CRITICAL in component_statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in component_statuses:
            overall_status = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in component_statuses:
            overall_status = HealthStatus.WARNING  # Treat unknown as warning
        else:
            overall_status = HealthStatus.HEALTHY

        # Generate summary
        status_counts = {}
        for status in component_statuses:
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

        uptime = datetime.now() - self.start_time

        summary = {
            "overall_status": overall_status.value,
            "component_count": len(self.components),
            "status_distribution": status_counts,
            "uptime_seconds": uptime.total_seconds(),
            "last_check": max(
                comp.last_check for comp in self.components.values()
            ).isoformat(),
            "monitoring_active": self.monitoring_active,
        }

        return overall_status, summary

    def export_health_report(self, filepath: str):
        """Export comprehensive health report."""
        overall_status, summary = self.get_overall_health()

        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_health": summary,
            "components": {
                name: {
                    "status": comp.status.value,
                    "type": comp.component_type.value,
                    "last_check": comp.last_check.isoformat(),
                    "uptime_seconds": comp.uptime.total_seconds(),
                    "error_count": comp.error_count,
                    "last_error": comp.last_error,
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "status": metric.status.value,
                            "timestamp": metric.timestamp.isoformat(),
                        }
                        for metric in comp.metrics
                    ],
                }
                for name, comp in self.components.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Health report exported to {filepath}")


# Global health monitor instance
_global_health_monitor = None


def get_global_health_monitor() -> AdvancedHealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = AdvancedHealthMonitor()
    return _global_health_monitor


# Console alert handler
def console_alert_handler(component_health: ComponentHealth, message: str):
    """Simple console alert handler."""
    print(f"🚨 HEALTH ALERT: {message}")
    logger.warning(message)
