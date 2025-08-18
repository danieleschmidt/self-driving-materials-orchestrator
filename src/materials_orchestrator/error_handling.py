"""Comprehensive error handling and recovery for Materials Orchestrator."""

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Custom exception classes
class MaterialsBaseError(Exception):
    """Base exception for materials orchestrator."""

    pass


class ExperimentError(MaterialsBaseError):
    """Exception for experiment-related errors."""

    pass


class ValidationError(MaterialsBaseError):
    """Exception for validation-related errors."""

    pass


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    ROBOT = "robot"
    COMPUTATION = "computation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error."""

    experiment_id: Optional[str] = None
    campaign_id: Optional[str] = None
    robot_id: Optional[str] = None
    user_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialsError:
    """Structured error information."""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    traceback: str
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "context": {
                "experiment_id": self.context.experiment_id,
                "campaign_id": self.context.campaign_id,
                "robot_id": self.context.robot_id,
                "user_id": self.context.user_id,
                "parameters": self.context.parameters,
                "additional_data": self.context.additional_data,
            },
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "retry_count": self.retry_count,
            "resolved": self.resolved,
        }


class RecoveryStrategy:
    """Base class for error recovery strategies."""

    def __init__(self, name: str, max_retries: int = 3, backoff_seconds: float = 1.0):
        """Initialize recovery strategy.

        Args:
            name: Strategy name
            max_retries: Maximum retry attempts
            backoff_seconds: Backoff delay between retries
        """
        self.name = name
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def can_handle(self, error: MaterialsError) -> bool:
        """Check if this strategy can handle the error.

        Args:
            error: Error to check

        Returns:
            True if can handle, False otherwise
        """
        raise NotImplementedError

    def attempt_recovery(
        self, error: MaterialsError, original_function: Callable, *args, **kwargs
    ) -> Any:
        """Attempt to recover from the error.

        Args:
            error: Error to recover from
            original_function: Original function that failed
            args: Original function args
            kwargs: Original function kwargs

        Returns:
            Result of recovery attempt
        """
        raise NotImplementedError


class RetryRecoveryStrategy(RecoveryStrategy):
    """Simple retry strategy with exponential backoff."""

    def __init__(
        self, name: str = "retry", max_retries: int = 3, backoff_seconds: float = 1.0
    ):
        """Initialize retry strategy."""
        super().__init__(name, max_retries, backoff_seconds)

    def can_handle(self, error: MaterialsError) -> bool:
        """Can handle most transient errors."""
        transient_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.DATABASE,
            ErrorCategory.ROBOT,
            ErrorCategory.RESOURCE,
        }
        return (
            error.category in transient_categories
            and error.retry_count < self.max_retries
        )

    def attempt_recovery(
        self, error: MaterialsError, original_function: Callable, *args, **kwargs
    ) -> Any:
        """Attempt recovery by retrying with backoff."""
        for attempt in range(self.max_retries):
            if attempt > 0:
                delay = self.backoff_seconds * (
                    2 ** (attempt - 1)
                )  # Exponential backoff
                logger.info(
                    f"Retrying {original_function.__name__} in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(delay)

            try:
                result = original_function(*args, **kwargs)
                error.recovery_successful = True
                error.retry_count = attempt + 1
                logger.info(
                    f"Recovery successful for {error.error_id} after {attempt + 1} attempts"
                )
                return result

            except Exception as e:
                error.retry_count = attempt + 1
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    raise e

        return None


class FallbackRecoveryStrategy(RecoveryStrategy):
    """Fallback to alternative implementation."""

    def __init__(
        self, name: str = "fallback", fallback_function: Optional[Callable] = None
    ):
        """Initialize fallback strategy.

        Args:
            name: Strategy name
            fallback_function: Alternative function to use
        """
        super().__init__(name, max_retries=1)
        self.fallback_function = fallback_function

    def can_handle(self, error: MaterialsError) -> bool:
        """Can handle when fallback function is available."""
        return self.fallback_function is not None

    def attempt_recovery(
        self, error: MaterialsError, original_function: Callable, *args, **kwargs
    ) -> Any:
        """Attempt recovery using fallback function."""
        if self.fallback_function is None:
            raise ValueError("No fallback function configured")

        logger.info(f"Using fallback function for {error.error_id}")

        try:
            result = self.fallback_function(*args, **kwargs)
            error.recovery_successful = True
            logger.info(f"Fallback successful for {error.error_id}")
            return result

        except Exception as e:
            logger.error(f"Fallback also failed for {error.error_id}: {e}")
            raise


class ParameterAdjustmentStrategy(RecoveryStrategy):
    """Adjust parameters and retry."""

    def __init__(
        self,
        name: str = "parameter_adjustment",
        adjustment_rules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize parameter adjustment strategy.

        Args:
            name: Strategy name
            adjustment_rules: Rules for parameter adjustment
        """
        super().__init__(name, max_retries=2)
        self.adjustment_rules = adjustment_rules or {}

    def can_handle(self, error: MaterialsError) -> bool:
        """Can handle validation and computation errors."""
        return error.category in {ErrorCategory.VALIDATION, ErrorCategory.COMPUTATION}

    def attempt_recovery(
        self, error: MaterialsError, original_function: Callable, *args, **kwargs
    ) -> Any:
        """Attempt recovery by adjusting parameters."""
        # Extract parameters from context
        parameters = error.context.parameters.copy()

        if not parameters:
            # Try to extract from kwargs
            parameters = kwargs.get("parameters", {})

        # Apply adjustment rules
        adjusted_parameters = self._adjust_parameters(parameters, error)

        # Update kwargs
        if "parameters" in kwargs:
            kwargs["parameters"] = adjusted_parameters

        logger.info(
            f"Attempting recovery with adjusted parameters for {error.error_id}"
        )

        try:
            result = original_function(*args, **kwargs)
            error.recovery_successful = True
            logger.info(f"Parameter adjustment successful for {error.error_id}")
            return result

        except Exception as e:
            logger.error(f"Parameter adjustment failed for {error.error_id}: {e}")
            raise

    def _adjust_parameters(
        self, parameters: Dict[str, Any], error: MaterialsError
    ) -> Dict[str, Any]:
        """Adjust parameters based on error type.

        Args:
            parameters: Original parameters
            error: Error information

        Returns:
            Adjusted parameters
        """
        adjusted = parameters.copy()

        # Example adjustments based on error message
        error_msg = error.message.lower()

        if "out of bounds" in error_msg or "invalid range" in error_msg:
            # Clamp values to reasonable ranges
            for key, value in adjusted.items():
                if isinstance(value, (int, float)):
                    if value < 0:
                        adjusted[key] = 0.1
                    elif value > 1000:
                        adjusted[key] = 100.0

        elif "temperature" in error_msg:
            # Adjust temperature-related parameters
            if "temperature" in adjusted:
                temp = adjusted["temperature"]
                if temp > 500:
                    adjusted["temperature"] = 300
                elif temp < 50:
                    adjusted["temperature"] = 100

        elif "concentration" in error_msg:
            # Adjust concentration parameters
            for key in adjusted:
                if "conc" in key.lower():
                    conc = adjusted[key]
                    if conc > 5.0:
                        adjusted[key] = 2.0
                    elif conc < 0.01:
                        adjusted[key] = 0.1

        return adjusted


class ErrorHandler:
    """Central error handling and recovery system."""

    def __init__(self):
        """Initialize error handler."""
        self._errors: List[MaterialsError] = []
        self._recovery_strategies: List[RecoveryStrategy] = []
        self._error_counts: Dict[str, int] = {}
        self._category_patterns: Dict[ErrorCategory, List[str]] = {
            ErrorCategory.VALIDATION: [
                "invalid",
                "out of bounds",
                "validation",
                "parameter",
            ],
            ErrorCategory.NETWORK: ["connection", "timeout", "network", "http"],
            ErrorCategory.DATABASE: ["database", "mongodb", "query", "connection"],
            ErrorCategory.ROBOT: ["robot", "hardware", "device", "instrument"],
            ErrorCategory.COMPUTATION: ["computation", "calculation", "numpy", "scipy"],
            ErrorCategory.CONFIGURATION: ["config", "configuration", "setting"],
            ErrorCategory.RESOURCE: ["memory", "disk", "resource", "limit"],
            ErrorCategory.SECURITY: [
                "security",
                "permission",
                "access",
                "authentication",
            ],
        }

        # Add default recovery strategies
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.add_recovery_strategy(RetryRecoveryStrategy())
        self.add_recovery_strategy(ParameterAdjustmentStrategy())

    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy.

        Args:
            strategy: Recovery strategy to add
        """
        self._recovery_strategies.append(strategy)
        logger.info(f"Added recovery strategy: {strategy.name}")

    def handle_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        original_function: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> Optional[Any]:
        """Handle an error with recovery attempts.

        Args:
            exception: The exception that occurred
            context: Error context information
            severity: Error severity (auto-detected if None)
            category: Error category (auto-detected if None)
            original_function: Original function that failed
            args: Original function args
            kwargs: Original function kwargs

        Returns:
            Result of recovery attempt or None
        """
        import uuid

        # Create error record
        error = MaterialsError(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            category=category or self._categorize_error(exception),
            severity=severity or self._assess_severity(exception),
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or ErrorContext(),
        )

        # Add to error log
        self._errors.append(error)
        self._error_counts[error.category.value] = (
            self._error_counts.get(error.category.value, 0) + 1
        )

        # Keep only recent errors (last 1000)
        if len(self._errors) > 1000:
            self._errors = self._errors[-1000:]

        # Log the error
        logger.error(
            f"Error {error.error_id} ({error.category.value}, {error.severity.value}): {error.message}",
            extra={
                "error_id": error.error_id,
                "category": error.category.value,
                "severity": error.severity.value,
            },
        )

        # Attempt recovery if function provided
        if original_function:
            return self._attempt_recovery(error, original_function, *args, **kwargs)

        return None

    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message.

        Args:
            exception: Exception to categorize

        Returns:
            Error category
        """
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Check by exception type first
        type_mapping = {
            "valueerror": ErrorCategory.VALIDATION,
            "typeerror": ErrorCategory.VALIDATION,
            "connectionerror": ErrorCategory.NETWORK,
            "timeouterror": ErrorCategory.NETWORK,
            "httperror": ErrorCategory.NETWORK,
            "databaseerror": ErrorCategory.DATABASE,
            "operationerror": ErrorCategory.DATABASE,
            "memoryerror": ErrorCategory.RESOURCE,
            "permissionerror": ErrorCategory.SECURITY,
        }

        if exception_type in type_mapping:
            return type_mapping[exception_type]

        # Check by message patterns
        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                if pattern in error_msg or pattern in exception_type:
                    return category

        return ErrorCategory.UNKNOWN

    def _assess_severity(self, exception: Exception) -> ErrorSeverity:
        """Assess error severity.

        Args:
            exception: Exception to assess

        Returns:
            Error severity
        """
        exception_type = type(exception).__name__.lower()
        error_msg = str(exception).lower()

        # Critical errors
        critical_indicators = [
            "security",
            "authentication",
            "permission",
            "memory",
            "system",
            "critical",
            "fatal",
            "corruption",
        ]

        if any(
            indicator in error_msg or indicator in exception_type
            for indicator in critical_indicators
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        high_indicators = ["database", "connection", "network", "robot", "hardware"]

        if any(
            indicator in error_msg or indicator in exception_type
            for indicator in high_indicators
        ):
            return ErrorSeverity.HIGH

        # Medium severity errors
        medium_indicators = ["validation", "parameter", "computation", "timeout"]

        if any(
            indicator in error_msg or indicator in exception_type
            for indicator in medium_indicators
        ):
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    def _attempt_recovery(
        self, error: MaterialsError, original_function: Callable, *args, **kwargs
    ) -> Optional[Any]:
        """Attempt to recover from an error.

        Args:
            error: Error to recover from
            original_function: Original function that failed
            args: Original function args
            kwargs: Original function kwargs

        Returns:
            Result of recovery attempt
        """
        error.recovery_attempted = True

        # Try each recovery strategy
        for strategy in self._recovery_strategies:
            if strategy.can_handle(error):
                logger.info(f"Attempting recovery with strategy: {strategy.name}")

                try:
                    result = strategy.attempt_recovery(
                        error, original_function, *args, **kwargs
                    )
                    error.recovery_successful = True
                    logger.info(f"Recovery successful using strategy: {strategy.name}")
                    return result

                except Exception as recovery_error:
                    logger.warning(
                        f"Recovery strategy {strategy.name} failed: {recovery_error}"
                    )
                    continue

        # No recovery strategy worked
        error.recovery_successful = False
        logger.error(f"All recovery strategies failed for error {error.error_id}")
        return None

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Error statistics
        """
        if not self._errors:
            return {"total_errors": 0}

        recent_errors = [
            error
            for error in self._errors
            if error.timestamp > datetime.now() - timedelta(hours=24)
        ]

        category_counts = {}
        severity_counts = {}
        recovery_stats = {"attempted": 0, "successful": 0}

        for error in recent_errors:
            category_counts[error.category.value] = (
                category_counts.get(error.category.value, 0) + 1
            )
            severity_counts[error.severity.value] = (
                severity_counts.get(error.severity.value, 0) + 1
            )

            if error.recovery_attempted:
                recovery_stats["attempted"] += 1
                if error.recovery_successful:
                    recovery_stats["successful"] += 1

        return {
            "total_errors": len(self._errors),
            "recent_errors_24h": len(recent_errors),
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "recovery_stats": recovery_stats,
            "recovery_success_rate": (
                recovery_stats["successful"] / recovery_stats["attempted"]
                if recovery_stats["attempted"] > 0
                else 0
            ),
        }

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of recent errors
        """
        recent_errors = self._errors[-limit:]
        return [error.to_dict() for error in recent_errors]

    def clear_resolved_errors(self):
        """Clear resolved errors from memory."""
        original_count = len(self._errors)
        self._errors = [error for error in self._errors if not error.resolved]
        cleared_count = original_count - len(self._errors)

        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} resolved errors")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance.

    Returns:
        Error handler
    """
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_errors(
    context: Optional[ErrorContext] = None,
    severity: Optional[ErrorSeverity] = None,
    category: Optional[ErrorCategory] = None,
    enable_recovery: bool = True,
):
    """Decorator for automatic error handling.

    Args:
        context: Error context
        severity: Error severity
        category: Error category
        enable_recovery: Whether to attempt recovery

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()

                if enable_recovery:
                    # Try to recover
                    result = error_handler.handle_error(
                        exception=e,
                        context=context,
                        severity=severity,
                        category=category,
                        original_function=func,
                        *args,
                        **kwargs,
                    )

                    if result is not None:
                        return result
                else:
                    # Just log the error
                    error_handler.handle_error(
                        exception=e,
                        context=context,
                        severity=severity,
                        category=category,
                    )

                # Re-raise if no recovery
                raise

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    *args,
    context: Optional[ErrorContext] = None,
    default_return: Any = None,
    **kwargs,
) -> Any:
    """Safely execute a function with error handling.

    Args:
        func: Function to execute
        args: Function arguments
        context: Error context
        default_return: Value to return on error
        kwargs: Function keyword arguments

    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler = get_error_handler()
        error_handler.handle_error(exception=e, context=context)
        logger.warning(f"Function {func.__name__} failed, returning default value: {e}")
        return default_return
