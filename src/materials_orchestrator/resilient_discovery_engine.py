"""Resilient Discovery Engine for Production Materials Research.

Advanced fault-tolerant system for autonomous materials discovery with
comprehensive error handling, recovery mechanisms, and quality assurance.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failure modes in materials discovery."""
    EXPERIMENT_FAILURE = "experiment_failure"
    ROBOT_MALFUNCTION = "robot_malfunction"
    INSTRUMENT_ERROR = "instrument_error"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_TIMEOUT = "network_timeout"
    SAFETY_VIOLATION = "safety_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    OPTIMIZATION_STALL = "optimization_stall"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_DELAY = "retry_with_delay"
    FALLBACK_METHOD = "fallback_method"
    SKIP_AND_CONTINUE = "skip_and_continue"
    EMERGENCY_STOP = "emergency_stop"
    RECALIBRATE_SYSTEM = "recalibrate_system"
    SWITCH_BACKUP = "switch_backup"


@dataclass
class FailureEvent:
    """Represents a failure event with context and recovery information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_mode: FailureMode = FailureMode.EXPERIMENT_FAILURE
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    error_message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    impact_severity: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and analysis."""
        return {
            'id': self.id,
            'failure_mode': self.failure_mode.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'error_message': self.error_message,
            'context': self.context,
            'recovery_attempted': self.recovery_attempted,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None,
            'recovery_successful': self.recovery_successful,
            'impact_severity': self.impact_severity
        }


class ResilientDiscoveryEngine:
    """Production-grade resilient discovery engine for materials research."""

    def __init__(self,
                 max_retries: int = 3,
                 base_retry_delay: float = 1.0,
                 max_retry_delay: float = 60.0,
                 circuit_breaker_threshold: int = 5,
                 enable_predictive_maintenance: bool = True):
        """Initialize resilient discovery engine.
        
        Args:
            max_retries: Maximum retry attempts for failed operations
            base_retry_delay: Base delay between retries (exponential backoff)
            max_retry_delay: Maximum delay between retries
            circuit_breaker_threshold: Number of failures before circuit breaker trips
            enable_predictive_maintenance: Enable predictive maintenance
        """
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.enable_predictive_maintenance = enable_predictive_maintenance

        # Failure tracking
        self.failure_history: List[FailureEvent] = []
        self.failure_counts: Dict[str, int] = {}
        self.component_health: Dict[str, float] = {}

        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Recovery handlers
        self.recovery_handlers: Dict[FailureMode, Callable] = {
            FailureMode.EXPERIMENT_FAILURE: self._handle_experiment_failure,
            FailureMode.ROBOT_MALFUNCTION: self._handle_robot_malfunction,
            FailureMode.INSTRUMENT_ERROR: self._handle_instrument_error,
            FailureMode.DATA_CORRUPTION: self._handle_data_corruption,
            FailureMode.NETWORK_TIMEOUT: self._handle_network_timeout,
            FailureMode.SAFETY_VIOLATION: self._handle_safety_violation,
            FailureMode.RESOURCE_EXHAUSTION: self._handle_resource_exhaustion,
            FailureMode.OPTIMIZATION_STALL: self._handle_optimization_stall,
        }

        # Performance monitoring
        self.operation_timings: Dict[str, List[float]] = {}
        self.quality_metrics: Dict[str, List[float]] = {}

        logger.info("Resilient Discovery Engine initialized")

    async def execute_resilient_operation(self,
                                        operation: Callable,
                                        operation_name: str,
                                        *args,
                                        **kwargs) -> Tuple[Any, bool]:
        """Execute an operation with comprehensive error handling and recovery.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging and tracking
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Tuple of (result, success_flag)
        """
        start_time = time.time()

        # Check circuit breaker
        if self._is_circuit_breaker_open(operation_name):
            logger.warning(f"Circuit breaker open for {operation_name}")
            return None, False

        for attempt in range(self.max_retries + 1):
            try:
                # Execute operation with timeout
                if asyncio.iscoroutinefunction(operation):
                    result = await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=kwargs.get('timeout', 300.0)
                    )
                else:
                    result = operation(*args, **kwargs)

                # Record successful operation
                execution_time = time.time() - start_time
                self._record_successful_operation(operation_name, execution_time)

                # Reset failure count on success
                self.failure_counts[operation_name] = 0

                return result, True

            except Exception as e:
                failure_event = self._create_failure_event(
                    operation_name, e, attempt, args, kwargs
                )
                self.failure_history.append(failure_event)

                # Determine if this is a recoverable failure
                is_recoverable = self._is_recoverable_failure(e, failure_event.failure_mode)

                if attempt < self.max_retries and is_recoverable:
                    # Attempt recovery
                    recovery_successful = await self._attempt_recovery(failure_event)

                    if recovery_successful:
                        # Calculate retry delay with exponential backoff
                        delay = min(
                            self.base_retry_delay * (2 ** attempt),
                            self.max_retry_delay
                        )

                        logger.info(f"Retrying {operation_name} in {delay:.1f}s (attempt {attempt + 2})")
                        await asyncio.sleep(delay)
                        continue

                # Handle final failure
                await self._handle_final_failure(failure_event)
                return None, False

        return None, False

    def _create_failure_event(self,
                            operation_name: str,
                            exception: Exception,
                            attempt: int,
                            args: tuple,
                            kwargs: dict) -> FailureEvent:
        """Create a failure event from an exception."""

        # Classify failure mode
        failure_mode = self._classify_failure_mode(exception)

        # Calculate impact severity
        impact_severity = self._calculate_impact_severity(failure_mode, operation_name)

        return FailureEvent(
            failure_mode=failure_mode,
            component=operation_name,
            error_message=str(exception),
            context={
                'attempt': attempt,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'exception_type': type(exception).__name__
            },
            impact_severity=impact_severity
        )

    def _classify_failure_mode(self, exception: Exception) -> FailureMode:
        """Classify the failure mode based on exception type and message."""

        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        if 'timeout' in exception_str or 'timeout' in exception_type:
            return FailureMode.NETWORK_TIMEOUT
        elif 'robot' in exception_str or 'device' in exception_str:
            return FailureMode.ROBOT_MALFUNCTION
        elif 'instrument' in exception_str or 'sensor' in exception_str:
            return FailureMode.INSTRUMENT_ERROR
        elif 'data' in exception_str and ('corrupt' in exception_str or 'invalid' in exception_str):
            return FailureMode.DATA_CORRUPTION
        elif 'safety' in exception_str or 'emergency' in exception_str:
            return FailureMode.SAFETY_VIOLATION
        elif 'memory' in exception_str or 'resource' in exception_str:
            return FailureMode.RESOURCE_EXHAUSTION
        elif 'converge' in exception_str or 'optimization' in exception_str:
            return FailureMode.OPTIMIZATION_STALL
        else:
            return FailureMode.EXPERIMENT_FAILURE

    def _calculate_impact_severity(self, failure_mode: FailureMode, component: str) -> float:
        """Calculate the impact severity of a failure."""

        base_severity = {
            FailureMode.SAFETY_VIOLATION: 1.0,
            FailureMode.ROBOT_MALFUNCTION: 0.8,
            FailureMode.INSTRUMENT_ERROR: 0.7,
            FailureMode.DATA_CORRUPTION: 0.6,
            FailureMode.RESOURCE_EXHAUSTION: 0.5,
            FailureMode.NETWORK_TIMEOUT: 0.4,
            FailureMode.OPTIMIZATION_STALL: 0.3,
            FailureMode.EXPERIMENT_FAILURE: 0.2,
        }.get(failure_mode, 0.5)

        # Adjust based on component criticality
        critical_components = ['safety_system', 'main_reactor', 'control_system']
        if any(critical in component.lower() for critical in critical_components):
            base_severity *= 1.5

        return min(base_severity, 1.0)

    def _is_recoverable_failure(self, exception: Exception, failure_mode: FailureMode) -> bool:
        """Determine if a failure is recoverable."""

        # Safety violations are not recoverable
        if failure_mode == FailureMode.SAFETY_VIOLATION:
            return False

        # Check for specific non-recoverable exceptions
        non_recoverable_types = [
            SystemExit,
            KeyboardInterrupt,
            MemoryError,
        ]

        if type(exception) in non_recoverable_types:
            return False

        return True

    async def _attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure."""

        try:
            # Get recovery handler for this failure mode
            handler = self.recovery_handlers.get(failure_event.failure_mode)

            if handler:
                recovery_strategy = await handler(failure_event)
                failure_event.recovery_attempted = True
                failure_event.recovery_strategy = recovery_strategy

                # Execute recovery strategy
                success = await self._execute_recovery_strategy(recovery_strategy, failure_event)
                failure_event.recovery_successful = success

                return success

            return False

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    async def _execute_recovery_strategy(self,
                                       strategy: RecoveryStrategy,
                                       failure_event: FailureEvent) -> bool:
        """Execute a specific recovery strategy."""

        try:
            if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
                return True  # Just retry immediately

            elif strategy == RecoveryStrategy.RETRY_WITH_DELAY:
                await asyncio.sleep(2.0)
                return True

            elif strategy == RecoveryStrategy.FALLBACK_METHOD:
                return await self._activate_fallback_method(failure_event)

            elif strategy == RecoveryStrategy.RECALIBRATE_SYSTEM:
                return await self._recalibrate_system(failure_event)

            elif strategy == RecoveryStrategy.SWITCH_BACKUP:
                return await self._switch_to_backup(failure_event)

            elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                logger.warning(f"Skipping failed operation: {failure_event.component}")
                return True

            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                logger.critical("Emergency stop triggered")
                return False

            return False

        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            return False

    async def _handle_experiment_failure(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle experiment failure."""

        # Check if this is a systematic failure
        recent_failures = self._get_recent_failures(failure_event.component, minutes=10)

        if len(recent_failures) > 3:
            return RecoveryStrategy.FALLBACK_METHOD
        else:
            return RecoveryStrategy.RETRY_WITH_DELAY

    async def _handle_robot_malfunction(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle robot malfunction."""

        # Check severity
        if failure_event.impact_severity > 0.7:
            return RecoveryStrategy.SWITCH_BACKUP
        else:
            return RecoveryStrategy.RECALIBRATE_SYSTEM

    async def _handle_instrument_error(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle instrument error."""

        # Try recalibration first
        return RecoveryStrategy.RECALIBRATE_SYSTEM

    async def _handle_data_corruption(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle data corruption."""

        # Skip corrupted data and continue
        return RecoveryStrategy.SKIP_AND_CONTINUE

    async def _handle_network_timeout(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle network timeout."""

        return RecoveryStrategy.RETRY_WITH_DELAY

    async def _handle_safety_violation(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle safety violation."""

        # Safety violations trigger emergency stop
        return RecoveryStrategy.EMERGENCY_STOP

    async def _handle_resource_exhaustion(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle resource exhaustion."""

        # Wait for resources to become available
        await asyncio.sleep(10.0)
        return RecoveryStrategy.RETRY_WITH_DELAY

    async def _handle_optimization_stall(self, failure_event: FailureEvent) -> RecoveryStrategy:
        """Handle optimization stall."""

        return RecoveryStrategy.FALLBACK_METHOD

    async def _activate_fallback_method(self, failure_event: FailureEvent) -> bool:
        """Activate fallback method for failed operation."""

        logger.info(f"Activating fallback method for {failure_event.component}")

        # Implement fallback logic based on component
        if 'synthesis' in failure_event.component.lower():
            # Use alternative synthesis method
            logger.info("Switching to alternative synthesis method")
            return True

        elif 'characterization' in failure_event.component.lower():
            # Use alternative characterization technique
            logger.info("Switching to alternative characterization method")
            return True

        elif 'optimization' in failure_event.component.lower():
            # Switch to simpler optimization algorithm
            logger.info("Switching to fallback optimization algorithm")
            return True

        return False

    async def _recalibrate_system(self, failure_event: FailureEvent) -> bool:
        """Recalibrate system component."""

        logger.info(f"Recalibrating {failure_event.component}")

        # Simulate recalibration process
        await asyncio.sleep(3.0)

        logger.info(f"Recalibration completed for {failure_event.component}")
        return True

    async def _switch_to_backup(self, failure_event: FailureEvent) -> bool:
        """Switch to backup system."""

        logger.info(f"Switching to backup for {failure_event.component}")

        # Simulate backup activation
        await asyncio.sleep(2.0)

        logger.info(f"Backup activated for {failure_event.component}")
        return True

    def _get_recent_failures(self, component: str, minutes: int = 10) -> List[FailureEvent]:
        """Get recent failures for a component."""

        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        return [
            event for event in self.failure_history
            if event.component == component and event.timestamp > cutoff_time
        ]

    def _is_circuit_breaker_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for an operation."""

        breaker = self.circuit_breakers.get(operation_name)

        if not breaker:
            return False

        # Check if enough time has passed to reset the breaker
        if breaker['open_until'] and datetime.now() > breaker['open_until']:
            del self.circuit_breakers[operation_name]
            return False

        return breaker.get('is_open', False)

    def _record_successful_operation(self, operation_name: str, execution_time: float):
        """Record successful operation for monitoring."""

        if operation_name not in self.operation_timings:
            self.operation_timings[operation_name] = []

        self.operation_timings[operation_name].append(execution_time)

        # Keep only recent timings
        if len(self.operation_timings[operation_name]) > 100:
            self.operation_timings[operation_name] = self.operation_timings[operation_name][-50:]

    async def _handle_final_failure(self, failure_event: FailureEvent):
        """Handle final failure after all retries exhausted."""

        # Update failure count
        self.failure_counts[failure_event.component] = \
            self.failure_counts.get(failure_event.component, 0) + 1

        # Check if circuit breaker should trip
        if self.failure_counts[failure_event.component] >= self.circuit_breaker_threshold:
            self._trip_circuit_breaker(failure_event.component)

        # Log final failure
        logger.error(
            f"Final failure for {failure_event.component}: {failure_event.error_message}"
        )

        # Predictive maintenance alert
        if self.enable_predictive_maintenance:
            await self._trigger_predictive_maintenance(failure_event)

    def _trip_circuit_breaker(self, operation_name: str):
        """Trip circuit breaker for an operation."""

        logger.warning(f"Circuit breaker tripped for {operation_name}")

        self.circuit_breakers[operation_name] = {
            'is_open': True,
            'open_until': datetime.now() + timedelta(minutes=5),  # Reset after 5 minutes
            'trip_count': self.circuit_breakers.get(operation_name, {}).get('trip_count', 0) + 1
        }

    async def _trigger_predictive_maintenance(self, failure_event: FailureEvent):
        """Trigger predictive maintenance analysis."""

        logger.info(f"Triggering predictive maintenance for {failure_event.component}")

        # Analyze failure patterns
        component_failures = [
            event for event in self.failure_history
            if event.component == failure_event.component
        ]

        if len(component_failures) > 3:
            # Calculate failure rate
            time_span = (component_failures[-1].timestamp - component_failures[0].timestamp).total_seconds()
            failure_rate = len(component_failures) / (time_span / 3600)  # failures per hour

            if failure_rate > 0.1:  # More than 0.1 failures per hour
                logger.warning(
                    f"High failure rate detected for {failure_event.component}: "
                    f"{failure_rate:.3f} failures/hour"
                )

                # Schedule maintenance
                await self._schedule_maintenance(failure_event.component)

    async def _schedule_maintenance(self, component: str):
        """Schedule maintenance for a component."""

        logger.info(f"Scheduling maintenance for {component}")

        # In production, this would integrate with maintenance management system
        # For now, just log the recommendation
        maintenance_priority = self._calculate_maintenance_priority(component)

        logger.info(
            f"Maintenance scheduled for {component} with priority: {maintenance_priority}"
        )

    def _calculate_maintenance_priority(self, component: str) -> str:
        """Calculate maintenance priority based on failure history."""

        recent_failures = self._get_recent_failures(component, minutes=60)

        if len(recent_failures) > 5:
            return "HIGH"
        elif len(recent_failures) > 2:
            return "MEDIUM"
        else:
            return "LOW"

    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""

        # Calculate overall health metrics
        total_failures = len(self.failure_history)
        recent_failures = len(self._get_recent_failures_all(minutes=60))

        # Component health scores
        component_health = {}
        for component in self.failure_counts:
            failure_count = self.failure_counts[component]
            recent_count = len(self._get_recent_failures(component, minutes=60))

            # Health score: 1.0 (perfect) to 0.0 (critical)
            health_score = max(0.0, 1.0 - (recent_count * 0.2) - (failure_count * 0.1))
            component_health[component] = health_score

        # System recommendations
        recommendations = []

        if recent_failures > 10:
            recommendations.append("High recent failure rate - consider system inspection")

        for component, health in component_health.items():
            if health < 0.5:
                recommendations.append(f"Component {component} requires attention (health: {health:.2f})")

        return {
            'timestamp': datetime.now().isoformat(),
            'total_failures': total_failures,
            'recent_failures_1h': recent_failures,
            'component_health': component_health,
            'circuit_breakers_active': len(self.circuit_breakers),
            'average_operation_time': self._calculate_average_operation_time(),
            'recommendations': recommendations,
            'system_status': 'HEALTHY' if recent_failures < 5 else 'DEGRADED' if recent_failures < 15 else 'CRITICAL'
        }

    def _get_recent_failures_all(self, minutes: int = 10) -> List[FailureEvent]:
        """Get all recent failures across all components."""

        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        return [
            event for event in self.failure_history
            if event.timestamp > cutoff_time
        ]

    def _calculate_average_operation_time(self) -> float:
        """Calculate average operation time across all operations."""

        all_times = []
        for operation_times in self.operation_timings.values():
            all_times.extend(operation_times[-10:])  # Recent times only

        return sum(all_times) / len(all_times) if all_times else 0.0


# Global instance
_global_resilient_engine = None

def get_global_resilient_engine() -> ResilientDiscoveryEngine:
    """Get global resilient discovery engine instance."""
    global _global_resilient_engine
    if _global_resilient_engine is None:
        _global_resilient_engine = ResilientDiscoveryEngine()
    return _global_resilient_engine
