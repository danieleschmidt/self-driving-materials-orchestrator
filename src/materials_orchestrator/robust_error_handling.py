"""Robust Error Handling System for Self-Healing Pipeline Guard.

Provides comprehensive error handling, recovery strategies, and resilience patterns
for materials discovery pipelines with advanced logging and alerting.
"""

import asyncio
import logging
import time
import traceback
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    DATABASE = "database"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    EXTERNAL = "external"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"


@dataclass
class ErrorContext:
    """Error context information."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SOFTWARE
    stack_trace: str = ""
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_impact: str = "none"  # none, low, medium, high
    business_impact: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_attempts": self.recovery_attempts,
            "user_impact": self.user_impact,
            "business_impact": self.business_impact,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryAction:
    """Recovery action definition."""

    action_id: str
    name: str
    strategy: RecoveryStrategy
    handler: Callable
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    timeout_seconds: float = 30.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        self._lock = threading.Lock()

    def __call__(self, func):
        """Decorator to apply circuit breaker to function."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)

        return wrapper

    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter

    def __call__(self, func):
        """Decorator to apply retry policy to function."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def execute(self, func, *args, **kwargs):
        """Execute function with retry policy."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_attempts - 1:
                    break

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.initial_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay


class BulkheadIsolation:
    """Bulkhead isolation pattern for resource protection."""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = 0
        self.total_operations = 0
        self.rejected_operations = 0

        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        """Acquire resource with bulkhead protection."""
        acquired = False
        try:
            acquired = await asyncio.wait_for(self.semaphore.acquire(), timeout=1.0)

            async with self._lock:
                self.active_operations += 1
                self.total_operations += 1

            yield

        except asyncio.TimeoutError:
            async with self._lock:
                self.rejected_operations += 1
            raise Exception("Resource pool exhausted")

        finally:
            if acquired:
                async with self._lock:
                    self.active_operations -= 1
                self.semaphore.release()


class RobustErrorHandler:
    """Comprehensive error handling system with recovery capabilities."""

    def __init__(self):
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}

        # Error statistics
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_counts: Dict[str, int] = defaultdict(int)
        self.component_health: Dict[str, float] = defaultdict(lambda: 1.0)

        # Configuration
        self.enable_auto_recovery = True
        self.enable_circuit_breakers = True
        self.enable_bulkhead_isolation = True
        self.max_recovery_attempts = 3

        # Alert thresholds
        self.error_rate_threshold = 0.1  # 10% error rate
        self.critical_error_threshold = 5  # 5 critical errors

        self._register_default_recovery_actions()

    def _register_default_recovery_actions(self):
        """Register default recovery actions."""

        # Database connection recovery
        self.register_recovery_action(
            "database_reconnect",
            "Database Reconnection",
            RecoveryStrategy.RETRY,
            self._recover_database_connection,
            max_attempts=3,
            delay_seconds=2.0,
        )

        # Memory cleanup recovery
        self.register_recovery_action(
            "memory_cleanup",
            "Memory Cleanup",
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            self._recover_memory_pressure,
            max_attempts=1,
            delay_seconds=5.0,
        )

        # Network timeout recovery
        self.register_recovery_action(
            "network_retry",
            "Network Retry",
            RecoveryStrategy.RETRY,
            self._recover_network_timeout,
            max_attempts=5,
            delay_seconds=1.0,
            backoff_factor=1.5,
        )

        # Resource exhaustion recovery
        self.register_recovery_action(
            "resource_scaling",
            "Resource Scaling",
            RecoveryStrategy.BULKHEAD,
            self._recover_resource_exhaustion,
            max_attempts=2,
            delay_seconds=10.0,
        )

    def register_recovery_action(
        self,
        action_id: str,
        name: str,
        strategy: RecoveryStrategy,
        handler: Callable,
        max_attempts: int = 3,
        delay_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        timeout_seconds: float = 30.0,
        conditions: Dict[str, Any] = None,
    ):
        """Register a recovery action."""
        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            strategy_map = {
                "retry": RecoveryStrategy.RETRY,
                "fallback": RecoveryStrategy.FALLBACK,
                "circuit_breaker": RecoveryStrategy.CIRCUIT_BREAKER,
                "graceful_degradation": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "fail_fast": RecoveryStrategy.FAIL_FAST,
                "bulkhead": RecoveryStrategy.BULKHEAD,
                "timeout": RecoveryStrategy.TIMEOUT,
            }
            strategy = strategy_map.get(strategy, RecoveryStrategy.RETRY)

        action = RecoveryAction(
            action_id=action_id,
            name=name,
            strategy=strategy,
            handler=handler,
            max_attempts=max_attempts,
            delay_seconds=delay_seconds,
            backoff_factor=backoff_factor,
            timeout_seconds=timeout_seconds,
            conditions=conditions or {},
        )

        self.recovery_actions[action_id] = action
        logger.info(f"Registered recovery action: {name}")

    async def handle_error(
        self,
        error: Exception,
        component: str = "",
        operation: str = "",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SOFTWARE,
        metadata: Dict[str, Any] = None,
    ) -> ErrorContext:
        """Handle error with comprehensive recovery strategies."""

        # Create error context
        error_context = ErrorContext(
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            metadata=metadata or {},
        )

        # Log error
        self._log_error(error_context)

        # Record error for statistics
        self._record_error(error_context)

        # Store in history
        self.error_history.append(error_context)

        # Attempt recovery if enabled
        if self.enable_auto_recovery:
            await self._attempt_recovery(error_context)

        # Check for alert conditions
        await self._check_alert_conditions(error_context)

        return error_context

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_message = (
            f"Error in {error_context.component}.{error_context.operation}: "
            f"{error_context.error_message}"
        )

        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _record_error(self, error_context: ErrorContext):
        """Record error for statistics and health tracking."""
        error_key = f"{error_context.component}.{error_context.error_type}"
        self.error_counts[error_key] += 1

        # Update component health
        component_health = self.component_health[error_context.component]
        health_impact = self._calculate_health_impact(error_context.severity)
        self.component_health[error_context.component] = max(
            0.0, component_health - health_impact
        )

    def _calculate_health_impact(self, severity: ErrorSeverity) -> float:
        """Calculate health impact based on error severity."""
        impact_map = {
            ErrorSeverity.LOW: 0.01,
            ErrorSeverity.MEDIUM: 0.05,
            ErrorSeverity.HIGH: 0.15,
            ErrorSeverity.CRITICAL: 0.30,
        }
        return impact_map.get(severity, 0.05)

    async def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt to recover from error using registered actions."""
        applicable_actions = self._find_applicable_recovery_actions(error_context)

        for action in applicable_actions:
            try:
                logger.info(f"Attempting recovery: {action.name}")

                success = await self._execute_recovery_action(action, error_context)

                if success:
                    error_context.recovery_attempts += 1
                    self.recovery_counts[action.action_id] += 1
                    action.success_count += 1

                    logger.info(f"Recovery successful: {action.name}")
                    break
                else:
                    action.failure_count += 1
                    logger.warning(f"Recovery failed: {action.name}")

            except Exception as e:
                action.failure_count += 1
                logger.error(f"Recovery action error: {action.name} - {e}")

    def _find_applicable_recovery_actions(
        self, error_context: ErrorContext
    ) -> List[RecoveryAction]:
        """Find recovery actions applicable to the error."""
        applicable_actions = []

        for action in self.recovery_actions.values():
            if self._matches_conditions(error_context, action.conditions):
                applicable_actions.append(action)

        # Sort by success rate
        applicable_actions.sort(
            key=lambda a: a.success_count / max(a.success_count + a.failure_count, 1),
            reverse=True,
        )

        return applicable_actions

    def _matches_conditions(
        self, error_context: ErrorContext, conditions: Dict[str, Any]
    ) -> bool:
        """Check if error context matches recovery action conditions."""
        if not conditions:
            return True

        for key, value in conditions.items():
            if key == "component" and error_context.component != value:
                return False
            elif key == "error_type" and error_context.error_type != value:
                return False
            elif key == "category" and error_context.category != value:
                return False
            elif key == "severity" and error_context.severity != value:
                return False

        return True

    async def _execute_recovery_action(
        self, action: RecoveryAction, error_context: ErrorContext
    ) -> bool:
        """Execute recovery action with timeout and retry logic."""
        try:
            if action.strategy == RecoveryStrategy.RETRY:
                return await self._execute_with_retry(action, error_context)
            elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_with_circuit_breaker(action, error_context)
            elif action.strategy == RecoveryStrategy.BULKHEAD:
                return await self._execute_with_bulkhead(action, error_context)
            else:
                # Direct execution
                if asyncio.iscoroutinefunction(action.handler):
                    result = await asyncio.wait_for(
                        action.handler(error_context), timeout=action.timeout_seconds
                    )
                else:
                    result = action.handler(error_context)

                return bool(result)

        except asyncio.TimeoutError:
            logger.error(f"Recovery action timeout: {action.name}")
            return False
        except Exception as e:
            logger.error(f"Recovery action failed: {action.name} - {e}")
            return False

    async def _execute_with_retry(
        self, action: RecoveryAction, error_context: ErrorContext
    ) -> bool:
        """Execute recovery action with retry policy."""
        policy = RetryPolicy(
            max_attempts=action.max_attempts,
            initial_delay=action.delay_seconds,
            backoff_factor=action.backoff_factor,
        )

        try:
            result = await policy.execute(action.handler, error_context)
            return bool(result)
        except Exception:
            return False

    async def _execute_with_circuit_breaker(
        self, action: RecoveryAction, error_context: ErrorContext
    ) -> bool:
        """Execute recovery action with circuit breaker protection."""
        circuit_breaker = self.circuit_breakers.get(action.action_id)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker()
            self.circuit_breakers[action.action_id] = circuit_breaker

        try:
            result = await circuit_breaker.call(action.handler, error_context)
            return bool(result)
        except Exception:
            return False

    async def _execute_with_bulkhead(
        self, action: RecoveryAction, error_context: ErrorContext
    ) -> bool:
        """Execute recovery action with bulkhead isolation."""
        bulkhead = self.bulkheads.get(action.action_id)
        if not bulkhead:
            bulkhead = BulkheadIsolation(max_concurrent=5)
            self.bulkheads[action.action_id] = bulkhead

        try:
            async with bulkhead.acquire():
                if asyncio.iscoroutinefunction(action.handler):
                    result = await action.handler(error_context)
                else:
                    result = action.handler(error_context)
                return bool(result)
        except Exception:
            return False

    async def _check_alert_conditions(self, error_context: ErrorContext):
        """Check if error should trigger alerts."""
        component = error_context.component

        # Check critical error threshold
        if error_context.severity == ErrorSeverity.CRITICAL:
            critical_errors = sum(
                1
                for ctx in self.error_history
                if (
                    ctx.component == component
                    and ctx.severity == ErrorSeverity.CRITICAL
                    and (datetime.now() - ctx.timestamp).total_seconds() < 300
                )  # Last 5 minutes
            )

            if critical_errors >= self.critical_error_threshold:
                await self._trigger_alert(
                    f"Critical error threshold exceeded for {component}", error_context
                )

        # Check error rate threshold
        recent_errors = [
            ctx
            for ctx in self.error_history
            if (
                ctx.component == component
                and (datetime.now() - ctx.timestamp).total_seconds() < 600
            )  # Last 10 minutes
        ]

        if len(recent_errors) > 10:  # Minimum sample size
            error_rate = len(recent_errors) / 600  # Errors per second
            if error_rate > self.error_rate_threshold:
                await self._trigger_alert(
                    f"High error rate detected for {component}: {error_rate:.3f}/s",
                    error_context,
                )

    async def _trigger_alert(self, message: str, error_context: ErrorContext):
        """Trigger alert for critical conditions."""
        logger.critical(f"ALERT: {message}")

        # In production, would integrate with alerting systems
        # (PagerDuty, Slack, email, etc.)
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "error_context": error_context.to_dict(),
        }

        # Simulate alert notification
        print(f"🚨 ALERT TRIGGERED: {message}")

    # Default recovery handlers
    async def _recover_database_connection(self, error_context: ErrorContext) -> bool:
        """Recover database connection."""
        logger.info("Attempting database reconnection...")
        await asyncio.sleep(1)  # Simulate reconnection time
        return True

    async def _recover_memory_pressure(self, error_context: ErrorContext) -> bool:
        """Recover from memory pressure."""
        logger.info("Performing memory cleanup...")
        await asyncio.sleep(3)  # Simulate cleanup time
        return True

    async def _recover_network_timeout(self, error_context: ErrorContext) -> bool:
        """Recover from network timeout."""
        logger.info("Retrying network operation...")
        await asyncio.sleep(0.5)  # Simulate retry delay
        return True

    async def _recover_resource_exhaustion(self, error_context: ErrorContext) -> bool:
        """Recover from resource exhaustion."""
        logger.info("Scaling resources...")
        await asyncio.sleep(5)  # Simulate scaling time
        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        recent_errors = [
            ctx
            for ctx in self.error_history
            if (datetime.now() - ctx.timestamp).total_seconds() < 3600  # Last hour
        ]

        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "error_counts_by_type": dict(self.error_counts),
            "recovery_counts": dict(self.recovery_counts),
            "component_health": dict(self.component_health),
            "recovery_actions": {
                action_id: {
                    "name": action.name,
                    "strategy": action.strategy.value,
                    "success_count": action.success_count,
                    "failure_count": action.failure_count,
                    "success_rate": action.success_count
                    / max(action.success_count + action.failure_count, 1),
                }
                for action_id, action in self.recovery_actions.items()
            },
            "circuit_breaker_states": {
                breaker_id: breaker.state
                for breaker_id, breaker in self.circuit_breakers.items()
            },
        }


# Decorators for robust error handling
def with_error_handling(
    component: str = "",
    operation: str = "",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SOFTWARE,
):
    """Decorator to add robust error handling to functions."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(
                    error=e,
                    component=component or func.__module__,
                    operation=operation or func.__name__,
                    severity=severity,
                    category=category,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                asyncio.create_task(
                    error_handler.handle_error(
                        error=e,
                        component=component or func.__module__,
                        operation=operation or func.__name__,
                        severity=severity,
                        category=category,
                    )
                )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Global error handler instance
_global_error_handler: Optional[RobustErrorHandler] = None


def get_global_error_handler() -> RobustErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler
