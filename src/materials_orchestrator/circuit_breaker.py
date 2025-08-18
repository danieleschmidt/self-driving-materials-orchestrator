"""Circuit breaker pattern for fault tolerance."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker for fault tolerance in experiment execution."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.lock = Lock()

        logger.info(
            f"Circuit breaker initialized with threshold={self.config.failure_threshold}"
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )

            if self.state == CircuitState.HALF_OPEN:
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.config.expected_exception:
                    self._on_failure()
                    raise

            # CLOSED state
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception:
                self._on_failure()
                raise

    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker restored to CLOSED state")

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    def reset(self):
        """Manually reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# Global circuit breaker instances
_experiment_breaker = None
_database_breaker = None
_robot_breaker = None


def get_experiment_breaker() -> CircuitBreaker:
    """Get global experiment circuit breaker."""
    global _experiment_breaker
    if _experiment_breaker is None:
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        _experiment_breaker = CircuitBreaker(config)
    return _experiment_breaker


def get_database_breaker() -> CircuitBreaker:
    """Get global database circuit breaker."""
    global _database_breaker
    if _database_breaker is None:
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)
        _database_breaker = CircuitBreaker(config)
    return _database_breaker


def get_robot_breaker() -> CircuitBreaker:
    """Get global robot circuit breaker."""
    global _robot_breaker
    if _robot_breaker is None:
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=120.0)
        _robot_breaker = CircuitBreaker(config)
    return _robot_breaker
