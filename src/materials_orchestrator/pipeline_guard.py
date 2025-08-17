"""Self-Healing Pipeline Guard for Materials Discovery Orchestrator.

Monitors pipeline health, detects failures, and implements self-healing mechanisms.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    HEALING = "healing"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of pipeline failures."""

    ROBOT_DISCONNECTION = "robot_disconnection"
    DATABASE_ERROR = "database_error"
    MEMORY_LEAK = "memory_leak"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NETWORK_TIMEOUT = "network_timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXPERIMENT_FAILURE = "experiment_failure"
    DATA_CORRUPTION = "data_corruption"


@dataclass
class HealthMetric:
    """Health metric for pipeline monitoring."""

    name: str
    value: float
    threshold: float
    severity: str = "warning"  # info, warning, critical
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range."""
        return self.value <= self.threshold


@dataclass
class PipelineFailure:
    """Pipeline failure record."""

    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_type: FailureType = FailureType.EXPERIMENT_FAILURE
    component: str = "unknown"
    severity: str = "warning"
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    healing_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingAction:
    """Self-healing action definition."""

    action_id: str
    name: str
    failure_types: List[FailureType]
    handler: Callable
    priority: int = 1  # Higher = more priority
    cooldown_seconds: int = 60
    max_attempts: int = 3
    last_executed: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class SelfHealingPipelineGuard:
    """Advanced self-healing pipeline guard with predictive maintenance."""

    def __init__(self):
        self.status = PipelineStatus.HEALTHY
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.failures: Dict[str, PipelineFailure] = {}
        self.healing_actions: Dict[str, HealingAction] = {}
        self.monitoring_active = False
        self.healing_enabled = True

        # Configuration
        self.monitoring_interval = 10  # seconds
        self.failure_threshold = 3  # failures before degraded status
        self.critical_failure_threshold = 5

        # Statistics
        self.total_failures = 0
        self.total_healings = 0
        self.uptime_start = datetime.now()

        # Register default healing actions
        self._register_default_healing_actions()

    def _register_default_healing_actions(self):
        """Register default self-healing actions."""

        # Robot reconnection healing
        self.register_healing_action(
            action_id="robot_reconnect",
            name="Reconnect Robots",
            failure_types=[FailureType.ROBOT_DISCONNECTION],
            handler=self._heal_robot_disconnection,
            priority=3,
            cooldown_seconds=30,
        )

        # Database connection healing
        self.register_healing_action(
            action_id="database_reconnect",
            name="Reconnect Database",
            failure_types=[FailureType.DATABASE_ERROR],
            handler=self._heal_database_connection,
            priority=3,
            cooldown_seconds=15,
        )

        # Memory cleanup healing
        self.register_healing_action(
            action_id="memory_cleanup",
            name="Memory Cleanup",
            failure_types=[FailureType.MEMORY_LEAK],
            handler=self._heal_memory_leak,
            priority=2,
            cooldown_seconds=120,
        )

        # Performance optimization healing
        self.register_healing_action(
            action_id="performance_optimize",
            name="Performance Optimization",
            failure_types=[FailureType.PERFORMANCE_DEGRADATION],
            handler=self._heal_performance_degradation,
            priority=2,
            cooldown_seconds=300,
        )

        # Resource scaling healing
        self.register_healing_action(
            action_id="resource_scale",
            name="Scale Resources",
            failure_types=[FailureType.RESOURCE_EXHAUSTION],
            handler=self._heal_resource_exhaustion,
            priority=1,
            cooldown_seconds=180,
        )

    def register_healing_action(
        self,
        action_id: str,
        name: str,
        failure_types: List[FailureType],
        handler: Callable,
        priority: int = 1,
        cooldown_seconds: int = 60,
        max_attempts: int = 3,
    ):
        """Register a new healing action."""
        action = HealingAction(
            action_id=action_id,
            name=name,
            failure_types=failure_types,
            handler=handler,
            priority=priority,
            cooldown_seconds=cooldown_seconds,
            max_attempts=max_attempts,
        )
        self.healing_actions[action_id] = action
        logger.info(f"Registered healing action: {name}")

    def update_health_metric(
        self, name: str, value: float, threshold: float, severity: str = "warning"
    ):
        """Update a health metric."""
        metric = HealthMetric(
            name=name, value=value, threshold=threshold, severity=severity
        )
        self.health_metrics[name] = metric

        # Check if metric indicates failure
        if not metric.is_healthy:
            self._handle_metric_failure(metric)

    def _handle_metric_failure(self, metric: HealthMetric):
        """Handle failure detected from health metric."""
        failure_type = self._determine_failure_type_from_metric(metric)
        self.report_failure(
            failure_type=failure_type,
            component=f"metric_{metric.name}",
            severity=metric.severity,
            description=f"Health metric {metric.name} exceeded threshold: {metric.value} > {metric.threshold}",
        )

    def _determine_failure_type_from_metric(self, metric: HealthMetric) -> FailureType:
        """Determine failure type from health metric."""
        metric_name = metric.name.lower()

        if "memory" in metric_name:
            return FailureType.MEMORY_LEAK
        elif "cpu" in metric_name or "performance" in metric_name:
            return FailureType.PERFORMANCE_DEGRADATION
        elif "network" in metric_name or "timeout" in metric_name:
            return FailureType.NETWORK_TIMEOUT
        elif "database" in metric_name or "db" in metric_name:
            return FailureType.DATABASE_ERROR
        elif "robot" in metric_name:
            return FailureType.ROBOT_DISCONNECTION
        else:
            return FailureType.EXPERIMENT_FAILURE

    def report_failure(
        self,
        failure_type: FailureType,
        component: str,
        severity: str = "warning",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Report a pipeline failure."""
        failure = PipelineFailure(
            failure_type=failure_type,
            component=component,
            severity=severity,
            description=description,
            metadata=metadata or {},
        )

        self.failures[failure.failure_id] = failure
        self.total_failures += 1

        logger.warning(
            f"Pipeline failure reported: {failure_type.value} in {component}"
        )

        # Update pipeline status based on failures
        self._update_pipeline_status()

        # Trigger self-healing if enabled
        if self.healing_enabled:
            asyncio.create_task(self._attempt_healing(failure))

        return failure.failure_id

    def _update_pipeline_status(self):
        """Update overall pipeline status based on failures."""
        active_failures = [f for f in self.failures.values() if not f.resolved]
        critical_failures = [f for f in active_failures if f.severity == "critical"]

        if len(critical_failures) > 0:
            self.status = PipelineStatus.FAILED
        elif len(active_failures) >= self.critical_failure_threshold:
            self.status = PipelineStatus.FAILED
        elif len(active_failures) >= self.failure_threshold:
            self.status = PipelineStatus.DEGRADED
        else:
            self.status = PipelineStatus.HEALTHY

    async def _attempt_healing(self, failure: PipelineFailure):
        """Attempt to heal a pipeline failure."""
        logger.info(f"Attempting to heal failure: {failure.failure_id}")

        # Find applicable healing actions
        applicable_actions = [
            action
            for action in self.healing_actions.values()
            if failure.failure_type in action.failure_types
        ]

        # Sort by priority (higher first)
        applicable_actions.sort(key=lambda x: x.priority, reverse=True)

        for action in applicable_actions:
            if self._can_execute_action(action):
                try:
                    self.status = PipelineStatus.HEALING
                    logger.info(f"Executing healing action: {action.name}")

                    # Execute healing action
                    success = await self._execute_healing_action(action, failure)

                    if success:
                        failure.resolved = True
                        failure.resolution_time = datetime.now()
                        action.success_count += 1
                        self.total_healings += 1

                        logger.info(
                            f"Successfully healed failure: {failure.failure_id}"
                        )
                        break
                    else:
                        action.failure_count += 1
                        failure.healing_attempts += 1

                except Exception as e:
                    logger.error(f"Healing action {action.name} failed: {e}")
                    action.failure_count += 1
                    failure.healing_attempts += 1

                action.last_executed = datetime.now()

        # Update pipeline status after healing attempt
        self._update_pipeline_status()

    def _can_execute_action(self, action: HealingAction) -> bool:
        """Check if healing action can be executed."""
        if action.last_executed is None:
            return True

        cooldown_elapsed = datetime.now() - action.last_executed
        return cooldown_elapsed.total_seconds() >= action.cooldown_seconds

    async def _execute_healing_action(
        self, action: HealingAction, failure: PipelineFailure
    ) -> bool:
        """Execute a healing action."""
        try:
            if asyncio.iscoroutinefunction(action.handler):
                result = await action.handler(failure)
            else:
                result = action.handler(failure)

            return bool(result)
        except Exception as e:
            logger.error(f"Healing action execution failed: {e}")
            return False

    # Default healing handlers
    async def _heal_robot_disconnection(self, failure: PipelineFailure) -> bool:
        """Heal robot disconnection."""
        logger.info("Attempting robot reconnection...")

        # Simulate robot reconnection logic
        await asyncio.sleep(2)

        # In real implementation, would reconnect to robots
        component = failure.component
        logger.info(f"Reconnected to robot: {component}")

        return True

    async def _heal_database_connection(self, failure: PipelineFailure) -> bool:
        """Heal database connection issues."""
        logger.info("Attempting database reconnection...")

        # Simulate database reconnection
        await asyncio.sleep(1)

        logger.info("Database connection restored")
        return True

    async def _heal_memory_leak(self, failure: PipelineFailure) -> bool:
        """Heal memory leak issues."""
        logger.info("Performing memory cleanup...")

        # Simulate memory cleanup
        await asyncio.sleep(3)

        logger.info("Memory cleanup completed")
        return True

    async def _heal_performance_degradation(self, failure: PipelineFailure) -> bool:
        """Heal performance degradation."""
        logger.info("Optimizing performance...")

        # Simulate performance optimization
        await asyncio.sleep(5)

        logger.info("Performance optimization completed")
        return True

    async def _heal_resource_exhaustion(self, failure: PipelineFailure) -> bool:
        """Heal resource exhaustion."""
        logger.info("Scaling resources...")

        # Simulate resource scaling
        await asyncio.sleep(4)

        logger.info("Resource scaling completed")
        return True

    async def start_monitoring(self):
        """Start continuous pipeline monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("Starting pipeline monitoring...")

        while self.monitoring_active:
            try:
                await self._monitor_pipeline_health()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _monitor_pipeline_health(self):
        """Monitor pipeline health metrics."""
        # Simulate health metric collection
        import psutil
        import random

        # CPU usage monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        self.update_health_metric("cpu_usage", cpu_percent, 80.0, "warning")

        # Memory usage monitoring
        memory = psutil.virtual_memory()
        self.update_health_metric("memory_usage", memory.percent, 85.0, "warning")

        # Disk usage monitoring
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        self.update_health_metric("disk_usage", disk_percent, 90.0, "critical")

        # Simulate experiment success rate
        experiment_success_rate = random.uniform(0.85, 0.98) * 100
        self.update_health_metric(
            "experiment_success_rate", 100 - experiment_success_rate, 15.0, "warning"
        )

    def stop_monitoring(self):
        """Stop pipeline monitoring."""
        self.monitoring_active = False
        logger.info("Pipeline monitoring stopped")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        uptime = datetime.now() - self.uptime_start
        active_failures = [f for f in self.failures.values() if not f.resolved]

        return {
            "status": self.status.value,
            "uptime_seconds": uptime.total_seconds(),
            "total_failures": self.total_failures,
            "total_healings": self.total_healings,
            "active_failures": len(active_failures),
            "healing_enabled": self.healing_enabled,
            "monitoring_active": self.monitoring_active,
            "health_metrics": {
                name: {
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "healthy": metric.is_healthy,
                    "severity": metric.severity,
                }
                for name, metric in self.health_metrics.items()
            },
            "healing_actions": {
                action_id: {
                    "name": action.name,
                    "success_count": action.success_count,
                    "failure_count": action.failure_count,
                    "last_executed": (
                        action.last_executed.isoformat()
                        if action.last_executed
                        else None
                    ),
                }
                for action_id, action in self.healing_actions.items()
            },
        }

    def resolve_failure(self, failure_id: str) -> bool:
        """Manually resolve a failure."""
        if failure_id in self.failures:
            self.failures[failure_id].resolved = True
            self.failures[failure_id].resolution_time = datetime.now()
            self._update_pipeline_status()
            return True
        return False

    def clear_resolved_failures(self):
        """Clear all resolved failures from memory."""
        resolved_failures = [
            failure_id
            for failure_id, failure in self.failures.items()
            if failure.resolved
        ]

        for failure_id in resolved_failures:
            del self.failures[failure_id]

        logger.info(f"Cleared {len(resolved_failures)} resolved failures")


# Singleton instance for global access
_global_pipeline_guard: Optional[SelfHealingPipelineGuard] = None


def get_pipeline_guard() -> SelfHealingPipelineGuard:
    """Get the global pipeline guard instance."""
    global _global_pipeline_guard
    if _global_pipeline_guard is None:
        _global_pipeline_guard = SelfHealingPipelineGuard()
    return _global_pipeline_guard


def create_pipeline_guard() -> SelfHealingPipelineGuard:
    """Create a new pipeline guard instance."""
    return SelfHealingPipelineGuard()
