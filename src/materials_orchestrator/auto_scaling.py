"""Intelligent auto-scaling system for materials orchestrator."""

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from .performance_monitoring import PerformanceTracker, get_performance_tracker

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""

    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    comparison_operator: str = "gt"  # gt, lt, avg_gt, avg_lt
    evaluation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    min_data_points: int = 3
    enabled: bool = True
    weight: float = 1.0  # Weight for multi-rule decisions


@dataclass
class ScalingTarget:
    """Scaling target configuration."""

    name: str
    current_capacity: int
    min_capacity: int = 1
    max_capacity: int = 100
    scale_up_step: int = 1
    scale_down_step: int = 1
    target_type: str = "generic"  # generic, worker, service
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingAction:
    """Represents a scaling action."""

    target_name: str
    direction: ScalingDirection
    from_capacity: int
    to_capacity: int
    timestamp: datetime
    reason: str
    triggered_by_rules: List[str]
    confidence: float = 1.0


class ScalingStrategy(Protocol):
    """Protocol for scaling strategies."""

    def should_scale(
        self,
        target: ScalingTarget,
        metrics_data: Dict[str, List[float]]
    ) -> tuple[ScalingDirection, float, str]:
        """Determine if scaling is needed.
        
        Args:
            target: Scaling target
            metrics_data: Recent metrics data
            
        Returns:
            Tuple of (direction, confidence, reason)
        """
        ...


class ConservativeScalingStrategy:
    """Conservative scaling strategy - requires strong signals."""

    def __init__(self, confidence_threshold: float = 0.8):
        """Initialize strategy.
        
        Args:
            confidence_threshold: Minimum confidence for scaling
        """
        self.confidence_threshold = confidence_threshold

    def should_scale(
        self,
        target: ScalingTarget,
        metrics_data: Dict[str, List[float]]
    ) -> tuple[ScalingDirection, float, str]:
        """Conservative scaling decision."""
        if not metrics_data:
            return ScalingDirection.NONE, 0.0, "No metrics data"

        # Require multiple consistent signals
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0

        reasons = []

        for metric_name, values in metrics_data.items():
            if len(values) < 3:  # Need sufficient data
                continue

            avg_value = statistics.mean(values)
            recent_trend = statistics.mean(values[-3:]) - statistics.mean(values[:-3]) if len(values) >= 6 else 0

            # Example thresholds (would be configurable)
            if metric_name == "cpu_usage" and avg_value > 0.8:
                scale_up_votes += 1
                reasons.append(f"High CPU: {avg_value:.2f}")
            elif metric_name == "memory_usage" and avg_value > 0.85:
                scale_up_votes += 1
                reasons.append(f"High memory: {avg_value:.2f}")
            elif metric_name == "queue_length" and avg_value > target.current_capacity * 2:
                scale_up_votes += 1
                reasons.append(f"High queue: {avg_value}")
            elif metric_name == "response_time_ms" and avg_value > 2000:
                scale_up_votes += 1
                reasons.append(f"Slow response: {avg_value:.0f}ms")

            # Scale down conditions
            elif metric_name == "cpu_usage" and avg_value < 0.3 and recent_trend < 0:
                scale_down_votes += 1
                reasons.append(f"Low CPU: {avg_value:.2f}")
            elif metric_name == "memory_usage" and avg_value < 0.4:
                scale_down_votes += 1
                reasons.append(f"Low memory: {avg_value:.2f}")

            total_weight += 1

        # Calculate confidence based on consensus
        if scale_up_votes > scale_down_votes:
            confidence = scale_up_votes / max(total_weight, 1)
            if confidence >= self.confidence_threshold:
                return ScalingDirection.UP, confidence, "; ".join(reasons)
        elif scale_down_votes > scale_up_votes:
            confidence = scale_down_votes / max(total_weight, 1)
            if confidence >= self.confidence_threshold:
                return ScalingDirection.DOWN, confidence, "; ".join(reasons)

        return ScalingDirection.NONE, 0.0, "Insufficient confidence for scaling"


class AggressiveScalingStrategy:
    """Aggressive scaling strategy - quick to respond."""

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize strategy.
        
        Args:
            confidence_threshold: Minimum confidence for scaling
        """
        self.confidence_threshold = confidence_threshold

    def should_scale(
        self,
        target: ScalingTarget,
        metrics_data: Dict[str, List[float]]
    ) -> tuple[ScalingDirection, float, str]:
        """Aggressive scaling decision."""
        if not metrics_data:
            return ScalingDirection.NONE, 0.0, "No metrics data"

        # Quick response to any concerning metric
        for metric_name, values in metrics_data.items():
            if not values:
                continue

            latest_value = values[-1]
            avg_value = statistics.mean(values)

            # Scale up triggers
            if metric_name == "cpu_usage" and latest_value > 0.7:
                return ScalingDirection.UP, 0.9, f"High CPU spike: {latest_value:.2f}"
            elif metric_name == "memory_usage" and latest_value > 0.8:
                return ScalingDirection.UP, 0.9, f"High memory: {latest_value:.2f}"
            elif metric_name == "response_time_ms" and latest_value > 1500:
                return ScalingDirection.UP, 0.8, f"Slow response: {latest_value:.0f}ms"

            # Scale down triggers
            elif metric_name == "cpu_usage" and avg_value < 0.2:
                return ScalingDirection.DOWN, 0.7, f"Low CPU: {avg_value:.2f}"

        return ScalingDirection.NONE, 0.0, "No scaling triggers met"


class PredictiveScalingStrategy:
    """Predictive scaling using trend analysis."""

    def __init__(self, prediction_window: int = 10):
        """Initialize strategy.
        
        Args:
            prediction_window: Number of data points for trend analysis
        """
        self.prediction_window = prediction_window

    def should_scale(
        self,
        target: ScalingTarget,
        metrics_data: Dict[str, List[float]]
    ) -> tuple[ScalingDirection, float, str]:
        """Predictive scaling decision."""
        if not metrics_data:
            return ScalingDirection.NONE, 0.0, "No metrics data"

        predictions = []

        for metric_name, values in metrics_data.items():
            if len(values) < self.prediction_window:
                continue

            # Simple linear trend prediction
            recent_values = values[-self.prediction_window:]
            x = list(range(len(recent_values)))

            # Calculate linear regression slope
            n = len(recent_values)
            sum_x = sum(x)
            sum_y = sum(recent_values)
            sum_xy = sum(x[i] * recent_values[i] for i in range(n))
            sum_x2 = sum(xi**2 for xi in x)

            if n * sum_x2 - sum_x**2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

                # Predict next few values
                current_value = recent_values[-1]
                predicted_value = current_value + slope * 3  # Predict 3 steps ahead

                # Scale up predictions
                if metric_name == "cpu_usage" and predicted_value > 0.85:
                    predictions.append((ScalingDirection.UP, 0.8, f"CPU trend: {predicted_value:.2f}"))
                elif metric_name == "memory_usage" and predicted_value > 0.9:
                    predictions.append((ScalingDirection.UP, 0.8, f"Memory trend: {predicted_value:.2f}"))
                elif metric_name == "response_time_ms" and predicted_value > 3000:
                    predictions.append((ScalingDirection.UP, 0.7, f"Response time trend: {predicted_value:.0f}ms"))

                # Scale down predictions
                elif metric_name == "cpu_usage" and predicted_value < 0.15:
                    predictions.append((ScalingDirection.DOWN, 0.6, f"CPU decreasing: {predicted_value:.2f}"))

        if predictions:
            # Choose highest confidence prediction
            best_prediction = max(predictions, key=lambda x: x[1])
            return best_prediction

        return ScalingDirection.NONE, 0.0, "No predictive signals"


class AutoScalingEngine:
    """Intelligent auto-scaling engine."""

    def __init__(
        self,
        performance_tracker: Optional[PerformanceTracker] = None,
        strategy: Optional[ScalingStrategy] = None
    ):
        """Initialize auto-scaling engine.
        
        Args:
            performance_tracker: Performance tracker to use
            strategy: Scaling strategy to use
        """
        self.performance_tracker = performance_tracker or get_performance_tracker()
        self.strategy = strategy or ConservativeScalingStrategy()

        self._rules: Dict[str, ScalingRule] = {}
        self._targets: Dict[str, ScalingTarget] = {}
        self._scaling_history: List[ScalingAction] = []
        self._last_scaling: Dict[str, datetime] = {}
        self._running = False
        self._lock = threading.RLock()

        # Callbacks for scaling actions
        self._scale_up_callbacks: List[Callable] = []
        self._scale_down_callbacks: List[Callable] = []

    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule.
        
        Args:
            rule: Scaling rule to add
        """
        with self._lock:
            self._rules[rule.name] = rule
            logger.info(f"Added scaling rule: {rule.name}")

    def add_scaling_target(self, target: ScalingTarget) -> None:
        """Add a scaling target.
        
        Args:
            target: Scaling target to add
        """
        with self._lock:
            self._targets[target.name] = target
            logger.info(f"Added scaling target: {target.name}")

    def register_scale_up_callback(self, callback: Callable[[ScalingAction], None]) -> None:
        """Register callback for scale-up events.
        
        Args:
            callback: Callback function
        """
        self._scale_up_callbacks.append(callback)

    def register_scale_down_callback(self, callback: Callable[[ScalingAction], None]) -> None:
        """Register callback for scale-down events.
        
        Args:
            callback: Callback function
        """
        self._scale_down_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring."""
        if self._running:
            return

        self._running = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()

        logger.info("Started auto-scaling monitoring")

    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        self._running = False
        logger.info("Stopped auto-scaling monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._evaluate_scaling_decisions()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(60)

    def _evaluate_scaling_decisions(self) -> None:
        """Evaluate and execute scaling decisions."""
        with self._lock:
            targets = list(self._targets.values())

        for target in targets:
            try:
                # Check cooldown period
                last_scaling = self._last_scaling.get(target.name)
                if last_scaling:
                    cooldown_period = timedelta(minutes=10)  # Default cooldown
                    if datetime.now() - last_scaling < cooldown_period:
                        continue

                # Gather metrics data
                metrics_data = self._gather_metrics_for_target(target)

                # Use strategy to determine scaling decision
                direction, confidence, reason = self.strategy.should_scale(target, metrics_data)

                if direction != ScalingDirection.NONE and confidence > 0.5:
                    # Calculate new capacity
                    if direction == ScalingDirection.UP:
                        new_capacity = min(
                            target.current_capacity + target.scale_up_step,
                            target.max_capacity
                        )
                    else:  # ScalingDirection.DOWN
                        new_capacity = max(
                            target.current_capacity - target.scale_down_step,
                            target.min_capacity
                        )

                    if new_capacity != target.current_capacity:
                        self._execute_scaling_action(target, direction, new_capacity, reason, confidence)

            except Exception as e:
                logger.error(f"Error evaluating scaling for {target.name}: {e}")

    def _gather_metrics_for_target(self, target: ScalingTarget) -> Dict[str, List[float]]:
        """Gather recent metrics for a target.
        
        Args:
            target: Scaling target
            
        Returns:
            Dictionary of metric_name -> list of recent values
        """
        window = timedelta(minutes=10)
        since = datetime.now() - window

        metrics_data = {}

        # Key metrics to collect
        metric_names = [
            "system_cpu_usage",
            "system_memory_usage",
            f"{target.name}_queue_length",
            f"{target.name}_response_time_ms",
            f"{target.name}_throughput",
            f"{target.name}_error_rate"
        ]

        for metric_name in metric_names:
            metrics = self.performance_tracker.get_metrics(metric_name, since=since)
            if metrics:
                values = [m.value for m in metrics]
                metrics_data[metric_name] = values

        return metrics_data

    def _execute_scaling_action(
        self,
        target: ScalingTarget,
        direction: ScalingDirection,
        new_capacity: int,
        reason: str,
        confidence: float
    ) -> None:
        """Execute a scaling action.
        
        Args:
            target: Target to scale
            direction: Scaling direction
            new_capacity: New capacity value
            reason: Reason for scaling
            confidence: Confidence in decision
        """
        action = ScalingAction(
            target_name=target.name,
            direction=direction,
            from_capacity=target.current_capacity,
            to_capacity=new_capacity,
            timestamp=datetime.now(),
            reason=reason,
            triggered_by_rules=[],
            confidence=confidence
        )

        logger.info(
            f"Executing scaling action: {target.name} "
            f"{target.current_capacity} -> {new_capacity} "
            f"({direction.value}) - {reason} (confidence: {confidence:.2f})"
        )

        # Update target capacity
        target.current_capacity = new_capacity

        # Record action
        with self._lock:
            self._scaling_history.append(action)
            self._last_scaling[target.name] = datetime.now()

        # Execute callbacks
        if direction == ScalingDirection.UP:
            for callback in self._scale_up_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"Error in scale-up callback: {e}")
        else:
            for callback in self._scale_down_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"Error in scale-down callback: {e}")

    def get_scaling_history(self, target_name: Optional[str] = None) -> List[ScalingAction]:
        """Get scaling history.
        
        Args:
            target_name: Filter by target name
            
        Returns:
            List of scaling actions
        """
        with self._lock:
            history = self._scaling_history.copy()

        if target_name:
            history = [action for action in history if action.target_name == target_name]

        return history

    def get_current_targets(self) -> Dict[str, ScalingTarget]:
        """Get current scaling targets.
        
        Returns:
            Dictionary of current targets
        """
        with self._lock:
            return self._targets.copy()

    def force_scale(
        self,
        target_name: str,
        direction: ScalingDirection,
        reason: str = "Manual trigger"
    ) -> bool:
        """Force a scaling action.
        
        Args:
            target_name: Target to scale
            direction: Scaling direction
            reason: Reason for manual scaling
            
        Returns:
            True if scaling was executed
        """
        with self._lock:
            target = self._targets.get(target_name)

        if not target:
            logger.error(f"Target {target_name} not found")
            return False

        if direction == ScalingDirection.UP:
            new_capacity = min(
                target.current_capacity + target.scale_up_step,
                target.max_capacity
            )
        elif direction == ScalingDirection.DOWN:
            new_capacity = max(
                target.current_capacity - target.scale_down_step,
                target.min_capacity
            )
        else:
            return False

        if new_capacity != target.current_capacity:
            self._execute_scaling_action(target, direction, new_capacity, reason, 1.0)
            return True

        return False


# Global auto-scaling engine
_global_scaling_engine: Optional[AutoScalingEngine] = None


def get_auto_scaling_engine() -> AutoScalingEngine:
    """Get global auto-scaling engine.
    
    Returns:
        Auto-scaling engine instance
    """
    global _global_scaling_engine
    if _global_scaling_engine is None:
        _global_scaling_engine = AutoScalingEngine()
    return _global_scaling_engine


def setup_default_scaling_targets() -> None:
    """Set up default scaling targets for materials orchestrator."""
    engine = get_auto_scaling_engine()

    # Add default targets
    targets = [
        ScalingTarget(
            name="experiment_workers",
            current_capacity=2,
            min_capacity=1,
            max_capacity=20,
            scale_up_step=2,
            scale_down_step=1,
            target_type="worker"
        ),
        ScalingTarget(
            name="api_servers",
            current_capacity=1,
            min_capacity=1,
            max_capacity=10,
            scale_up_step=1,
            scale_down_step=1,
            target_type="service"
        ),
        ScalingTarget(
            name="optimization_queue",
            current_capacity=100,
            min_capacity=50,
            max_capacity=1000,
            scale_up_step=50,
            scale_down_step=25,
            target_type="queue"
        )
    ]

    for target in targets:
        engine.add_scaling_target(target)

    logger.info(f"Set up {len(targets)} default scaling targets")
