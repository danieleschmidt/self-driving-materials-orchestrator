"""Advanced error recovery and resilience system."""

import json
import logging
import random
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    stack_trace: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    success_count_in_half_open: int = 0


class ResilientExecutor:
    """Executor with comprehensive error handling and recovery."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_history: List[ErrorRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self._error_count_lock = threading.Lock()

        # Circuit breaker configuration
        self.circuit_breaker_config = {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,  # seconds
            "success_threshold": 3,  # successes needed to close circuit
        }

        logger.info("ResilientExecutor initialized with advanced error recovery")

    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register a custom recovery handler for specific error types."""
        self.recovery_handlers[error_type] = handler
        logger.info(f"Registered recovery handler for {error_type}")

    def register_fallback_strategy(self, operation: str, fallback: Callable):
        """Register a fallback strategy for operations."""
        self.fallback_strategies[operation] = fallback
        logger.info(f"Registered fallback strategy for {operation}")

    def execute_with_recovery(
        self,
        operation: Callable,
        *args,
        operation_name: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        """Execute operation with comprehensive error recovery."""

        context = context or {}
        last_exception = None

        # Check circuit breaker
        if self._is_circuit_open(operation_name):
            logger.warning(f"Circuit breaker open for {operation_name}, using fallback")
            return self._execute_fallback(operation_name, *args, **kwargs)

        for attempt in range(self.max_retries + 1):
            try:
                # Execute operation
                result = operation(*args, **kwargs)

                # Record success and potentially close circuit breaker
                self._record_success(operation_name)
                return result, True

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                # Record error
                error_record = self._record_error(
                    error_type=error_type,
                    message=str(e),
                    severity=self._classify_error_severity(e),
                    context={
                        **context,
                        "operation": operation_name,
                        "attempt": attempt + 1,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200],
                    },
                )

                # Try recovery
                recovery_result = self._attempt_recovery(
                    error_record, operation, args, kwargs
                )
                if recovery_result[1]:  # Recovery successful
                    return recovery_result[0], True

                # Check if we should continue retrying
                if attempt < self.max_retries:
                    if self._should_retry(e, attempt):
                        delay = self._calculate_retry_delay(attempt)
                        logger.info(
                            f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 2}/{self.max_retries + 1})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.info(
                            f"Not retrying {operation_name} due to error type: {error_type}"
                        )
                        break
                else:
                    logger.error(f"Max retries exceeded for {operation_name}")

        # All retries failed, record failure and try fallback
        self._record_failure(operation_name)
        fallback_result = self._execute_fallback(operation_name, *args, **kwargs)

        if fallback_result[1]:
            logger.info(f"Fallback successful for {operation_name}")
            return fallback_result
        else:
            logger.error(f"Operation {operation_name} failed completely")
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(
                    f"Operation {operation_name} failed without specific exception"
                )

    def _record_error(
        self,
        error_type: str,
        message: str,
        severity: ErrorSeverity,
        context: Dict[str, Any],
    ) -> ErrorRecord:
        """Record an error occurrence."""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc(),
        )

        with self._error_count_lock:
            self.error_history.append(error_record)

            # Keep error history manageable
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]

        # Log error with appropriate level
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{error_type}: {message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"{error_type}: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{error_type}: {message}")
        else:
            logger.info(f"{error_type}: {message}")

        return error_record

    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and message."""
        error_type = type(error).__name__
        message = str(error).lower()

        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if error_type in ["FileNotFoundError", "PermissionError", "ConnectionError"]:
            return ErrorSeverity.HIGH

        if "database" in message or "connection" in message or "network" in message:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.MEDIUM

        if "timeout" in message or "validation" in message:
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _attempt_recovery(
        self, error_record: ErrorRecord, operation: Callable, args: tuple, kwargs: dict
    ) -> Tuple[Any, bool]:
        """Attempt to recover from an error."""
        error_type = error_record.error_type

        # Try registered recovery handler
        if error_type in self.recovery_handlers:
            try:
                logger.info(f"Attempting recovery for {error_type}")
                result = self.recovery_handlers[error_type](
                    error_record, operation, args, kwargs
                )
                error_record.recovery_attempted = True
                error_record.recovery_successful = True
                error_record.recovery_strategy = RecoveryStrategy.RETRY
                return result, True
            except Exception as e:
                logger.error(f"Recovery handler failed for {error_type}: {e}")

        # Generic recovery strategies based on error type
        if error_type == "ConnectionError":
            return self._recover_connection_error(error_record, operation, args, kwargs)
        elif error_type == "TimeoutError":
            return self._recover_timeout_error(error_record, operation, args, kwargs)
        elif error_type == "ValidationError":
            return self._recover_validation_error(error_record, operation, args, kwargs)

        error_record.recovery_attempted = True
        error_record.recovery_successful = False
        return None, False

    def _recover_connection_error(
        self, error_record: ErrorRecord, operation: Callable, args: tuple, kwargs: dict
    ) -> Tuple[Any, bool]:
        """Recover from connection errors."""
        logger.info("Attempting connection error recovery")

        # Wait a bit longer for connection recovery
        time.sleep(2.0)

        try:
            # Attempt to reinitialize connection if possible
            if hasattr(operation, "__self__") and hasattr(
                operation.__self__, "reconnect"
            ):
                operation.__self__.reconnect()

            result = operation(*args, **kwargs)
            error_record.recovery_successful = True
            error_record.recovery_strategy = RecoveryStrategy.RETRY
            return result, True
        except Exception as e:
            logger.error(f"Connection recovery failed: {e}")
            return None, False

    def _recover_timeout_error(
        self, error_record: ErrorRecord, operation: Callable, args: tuple, kwargs: dict
    ) -> Tuple[Any, bool]:
        """Recover from timeout errors."""
        logger.info("Attempting timeout error recovery")

        # Extend timeout if possible
        extended_kwargs = kwargs.copy()
        if "timeout" in extended_kwargs:
            extended_kwargs["timeout"] *= 2  # Double the timeout

        try:
            result = operation(*args, **extended_kwargs)
            error_record.recovery_successful = True
            error_record.recovery_strategy = RecoveryStrategy.RETRY
            return result, True
        except Exception as e:
            logger.error(f"Timeout recovery failed: {e}")
            return None, False

    def _recover_validation_error(
        self, error_record: ErrorRecord, operation: Callable, args: tuple, kwargs: dict
    ) -> Tuple[Any, bool]:
        """Recover from validation errors."""
        logger.info("Attempting validation error recovery")

        # Try to sanitize parameters
        if "parameters" in error_record.context:
            try:
                from .security import InputValidator

                sanitized_params = InputValidator.validate_experiment_parameters(
                    error_record.context["parameters"]
                )

                # Rebuild args/kwargs with sanitized parameters
                if args and isinstance(args[0], dict):
                    new_args = (sanitized_params,) + args[1:]
                    result = operation(*new_args, **kwargs)
                    error_record.recovery_successful = True
                    error_record.recovery_strategy = RecoveryStrategy.RETRY
                    return result, True
            except Exception as e:
                logger.error(f"Parameter sanitization failed: {e}")

        return None, False

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry based on error type and attempt count."""
        error_type = type(error).__name__

        # Never retry these errors
        no_retry_errors = [
            "KeyboardInterrupt",
            "SystemExit",
            "MemoryError",
            "ImportError",
        ]
        if error_type in no_retry_errors:
            return False

        # Don't retry validation errors after first attempt
        if error_type in ["ValidationError", "ValueError"] and attempt > 0:
            return False

        # Retry everything else up to max attempts
        return True

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry using exponential backoff with jitter."""
        delay = self.base_delay * (2**attempt)

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5) * delay

        # Cap maximum delay at 30 seconds
        return min(delay + jitter, 30.0)

    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        if operation not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[operation]

        if breaker.state == "CLOSED":
            return False
        elif breaker.state == "OPEN":
            # Check if recovery timeout has passed
            if (
                breaker.last_failure_time
                and datetime.now() - breaker.last_failure_time
                > timedelta(seconds=self.circuit_breaker_config["recovery_timeout"])
            ):
                # Move to half-open state
                breaker.state = "HALF_OPEN"
                breaker.success_count_in_half_open = 0
                logger.info(f"Circuit breaker for {operation} moved to HALF_OPEN")
                return False
            return True
        else:  # HALF_OPEN
            return False

    def _record_success(self, operation: str):
        """Record successful operation."""
        if operation in self.circuit_breakers:
            breaker = self.circuit_breakers[operation]

            if breaker.state == "HALF_OPEN":
                breaker.success_count_in_half_open += 1
                if (
                    breaker.success_count_in_half_open
                    >= self.circuit_breaker_config["success_threshold"]
                ):
                    breaker.state = "CLOSED"
                    breaker.failure_count = 0
                    logger.info(f"Circuit breaker for {operation} CLOSED")
            elif breaker.state == "CLOSED":
                # Reset failure count on success
                breaker.failure_count = max(0, breaker.failure_count - 1)

    def _record_failure(self, operation: str):
        """Record operation failure for circuit breaker."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreakerState()

        breaker = self.circuit_breakers[operation]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()

        if breaker.failure_count >= self.circuit_breaker_config["failure_threshold"]:
            breaker.state = "OPEN"
            logger.warning(
                f"Circuit breaker for {operation} OPENED due to {breaker.failure_count} failures"
            )

    def _execute_fallback(self, operation: str, *args, **kwargs) -> Tuple[Any, bool]:
        """Execute fallback strategy for failed operation."""
        if operation in self.fallback_strategies:
            try:
                logger.info(f"Executing fallback for {operation}")
                result = self.fallback_strategies[operation](*args, **kwargs)
                return result, True
            except Exception as e:
                logger.error(f"Fallback failed for {operation}: {e}")

        # Default fallback returns None
        return None, False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        with self._error_count_lock:
            if not self.error_history:
                return {"total_errors": 0}

            total_errors = len(self.error_history)
            error_types = {}
            severity_counts = {}
            recovery_stats = {"attempted": 0, "successful": 0}

            for error in self.error_history:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
                severity_counts[error.severity.value] = (
                    severity_counts.get(error.severity.value, 0) + 1
                )

                if error.recovery_attempted:
                    recovery_stats["attempted"] += 1
                    if error.recovery_successful:
                        recovery_stats["successful"] += 1

            # Calculate error rates over time windows
            now = datetime.now()
            recent_errors = [
                e
                for e in self.error_history
                if (now - e.timestamp).total_seconds() < 3600
            ]  # Last hour

            return {
                "total_errors": total_errors,
                "recent_errors_1h": len(recent_errors),
                "error_types": error_types,
                "severity_distribution": severity_counts,
                "recovery_statistics": recovery_stats,
                "recovery_rate": recovery_stats["successful"]
                / max(recovery_stats["attempted"], 1),
                "circuit_breakers": {
                    name: state.state for name, state in self.circuit_breakers.items()
                },
            }

    def export_error_report(self, filepath: str):
        """Export comprehensive error report to file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "recent_errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "type": error.error_type,
                    "message": error.message,
                    "severity": error.severity.value,
                    "context": error.context,
                    "recovery_attempted": error.recovery_attempted,
                    "recovery_successful": error.recovery_successful,
                }
                for error in self.error_history[-100:]  # Last 100 errors
            ],
            "circuit_breaker_states": {
                name: {
                    "state": state.state,
                    "failure_count": state.failure_count,
                    "last_failure": (
                        state.last_failure_time.isoformat()
                        if state.last_failure_time
                        else None
                    ),
                }
                for name, state in self.circuit_breakers.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Error report exported to {filepath}")


# Global resilient executor instance
_global_executor = None


def get_global_resilient_executor() -> ResilientExecutor:
    """Get global resilient executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = ResilientExecutor()
    return _global_executor


def with_resilience(operation_name: str = None, context: Dict[str, Any] = None):
    """Decorator to add resilience to functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            executor = get_global_resilient_executor()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            result, success = executor.execute_with_recovery(
                func, *args, operation_name=op_name, context=context, **kwargs
            )
            return result

        return wrapper

    return decorator
