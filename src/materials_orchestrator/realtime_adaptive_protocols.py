"""Real-Time Adaptive Protocol Engine for Dynamic Materials Discovery.

This module implements advanced real-time adaptation capabilities that allow
the materials discovery system to dynamically adjust experimental protocols,
parameters, and strategies based on live experimental feedback.
"""

import asyncio
import logging
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class AdaptationTrigger(Enum):
    """Types of events that can trigger protocol adaptation."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    OUTLIER_DETECTION = "outlier_detection"
    CONVERGENCE_STAGNATION = "convergence_stagnation"
    RESOURCE_CONSTRAINT = "resource_constraint"
    SAFETY_CONCERN = "safety_concern"
    DISCOVERY_BREAKTHROUGH = "discovery_breakthrough"
    TIME_CONSTRAINT = "time_constraint"
    COST_OPTIMIZATION = "cost_optimization"


class AdaptationStrategy(Enum):
    """Strategies for protocol adaptation."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"
    EXPLOITATIVE = "exploitative"
    SAFETY_FIRST = "safety_first"


class ProtocolStatus(Enum):
    """Status of adaptive protocols."""
    STABLE = "stable"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    EMERGENCY = "emergency"
    LEARNING = "learning"


@dataclass
class ExperimentalCondition:
    """Represents current experimental conditions."""
    
    temperature: float = 150.0
    pressure: float = 1.0
    concentration_a: float = 1.0
    concentration_b: float = 1.0
    reaction_time: float = 3.0
    stirring_speed: float = 500.0
    ph: float = 7.0
    atmosphere: str = "air"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "concentration_a": self.concentration_a,
            "concentration_b": self.concentration_b,
            "reaction_time": self.reaction_time,
            "stirring_speed": self.stirring_speed,
            "ph": self.ph,
            "atmosphere": self.atmosphere
        }
    
    def distance_to(self, other: 'ExperimentalCondition') -> float:
        """Calculate Euclidean distance to another condition."""
        differences = [
            (self.temperature - other.temperature) / 100,  # Normalize by typical range
            (self.pressure - other.pressure) / 10,
            (self.concentration_a - other.concentration_a) / 2,
            (self.concentration_b - other.concentration_b) / 2,
            (self.reaction_time - other.reaction_time) / 10,
            (self.stirring_speed - other.stirring_speed) / 1000,
            (self.ph - other.ph) / 7
        ]
        return math.sqrt(sum(d**2 for d in differences))


@dataclass
class RealTimeResult:
    """Represents real-time experimental results."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    conditions: ExperimentalCondition = field(default_factory=ExperimentalCondition)
    properties: Dict[str, float] = field(default_factory=dict)
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    experimental_errors: List[str] = field(default_factory=list)
    instrument_status: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "conditions": self.conditions.to_dict(),
            "properties": self.properties,
            "quality_indicators": self.quality_indicators,
            "experimental_errors": self.experimental_errors,
            "instrument_status": self.instrument_status,
            "confidence_score": self.confidence_score
        }


@dataclass
class AdaptationRule:
    """Defines a rule for protocol adaptation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger: AdaptationTrigger = AdaptationTrigger.PERFORMANCE_DEGRADATION
    condition_function: Optional[Callable] = None
    adaptation_function: Optional[Callable] = None
    priority: int = 1  # Higher number = higher priority
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    enabled: bool = True
    
    def can_trigger(self) -> bool:
        """Check if rule can be triggered (respects cooldown)."""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        return datetime.now() - self.last_triggered > self.cooldown_period
    
    def trigger_rule(self) -> None:
        """Mark rule as triggered."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1


class PerformanceMonitor:
    """Monitors experimental performance in real-time."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.performance_history: deque = deque(maxlen=window_size)
        self.baseline_performance: Optional[float] = None
        self.performance_threshold = 0.1  # 10% degradation threshold
        
    def update_performance(self, result: RealTimeResult) -> Dict[str, float]:
        """Update performance metrics with new result."""
        # Calculate performance score (multi-objective)
        target_properties = {
            "band_gap": {"target": 1.4, "weight": 0.4},
            "efficiency": {"target": 0.25, "weight": 0.4},
            "stability": {"target": 0.9, "weight": 0.2}
        }
        
        performance_score = 0.0
        total_weight = 0.0
        
        for prop, config in target_properties.items():
            if prop in result.properties:
                target = config["target"]
                weight = config["weight"]
                value = result.properties[prop]
                
                # Calculate normalized error (0 = perfect, 1 = worst case)
                if prop == "band_gap":
                    error = abs(value - target) / target
                else:
                    error = abs(value - target) / target if target > 0 else abs(value)
                
                score = max(0, 1 - error) * weight
                performance_score += score
                total_weight += weight
        
        if total_weight > 0:
            performance_score /= total_weight
        
        # Apply confidence weighting
        performance_score *= result.confidence_score
        
        self.performance_history.append(performance_score)
        
        # Set baseline if not established
        if self.baseline_performance is None and len(self.performance_history) >= 3:
            self.baseline_performance = statistics.mean(self.performance_history)
        
        # Calculate performance metrics
        metrics = {
            "current_performance": performance_score,
            "average_performance": statistics.mean(self.performance_history),
            "performance_trend": self._calculate_trend(),
            "performance_volatility": statistics.stdev(self.performance_history) if len(self.performance_history) > 1 else 0,
            "baseline_performance": self.baseline_performance or 0
        }
        
        return metrics
    
    def _calculate_trend(self) -> float:
        """Calculate performance trend (-1 to 1, where 1 is improving)."""
        if len(self.performance_history) < 3:
            return 0.0
        
        # Linear regression slope
        x = list(range(len(self.performance_history)))
        y = list(self.performance_history)
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to [-1, 1] range
        return np.tanh(slope * 10)
    
    def detect_performance_issues(self) -> List[str]:
        """Detect performance-related issues."""
        issues = []
        
        if len(self.performance_history) < 3:
            return issues
        
        current_perf = self.performance_history[-1]
        avg_perf = statistics.mean(self.performance_history)
        trend = self._calculate_trend()
        
        # Check for performance degradation
        if self.baseline_performance and current_perf < self.baseline_performance * (1 - self.performance_threshold):
            issues.append("performance_below_baseline")
        
        # Check for declining trend
        if trend < -0.3:
            issues.append("declining_performance_trend")
        
        # Check for high volatility
        if len(self.performance_history) > 1:
            volatility = statistics.stdev(self.performance_history)
            if volatility > 0.2:
                issues.append("high_performance_volatility")
        
        # Check for stagnation
        if abs(trend) < 0.1 and len(self.performance_history) >= 5:
            recent_range = max(self.performance_history[-5:]) - min(self.performance_history[-5:])
            if recent_range < 0.05:
                issues.append("performance_stagnation")
        
        return issues


class OutlierDetector:
    """Detects outliers in experimental results."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.result_history: deque = deque(maxlen=window_size)
        
    def add_result(self, result: RealTimeResult) -> Dict[str, Any]:
        """Add result and detect outliers."""
        self.result_history.append(result)
        
        if len(self.result_history) < 5:
            return {"is_outlier": False, "outlier_type": None, "outlier_score": 0.0}
        
        # Extract property values for outlier detection
        outlier_analysis = {}
        
        for prop in ["band_gap", "efficiency", "stability"]:
            if prop in result.properties:
                outlier_info = self._detect_property_outlier(prop, result.properties[prop])
                if outlier_info["is_outlier"]:
                    outlier_analysis[prop] = outlier_info
        
        # Overall outlier assessment
        is_outlier = len(outlier_analysis) > 0
        outlier_score = max([info["outlier_score"] for info in outlier_analysis.values()]) if outlier_analysis else 0.0
        
        # Determine outlier type
        outlier_type = None
        if is_outlier:
            if outlier_score > 0.8:
                outlier_type = "extreme_outlier"
            elif outlier_score > 0.5:
                outlier_type = "moderate_outlier"
            else:
                outlier_type = "mild_outlier"
        
        return {
            "is_outlier": is_outlier,
            "outlier_type": outlier_type,
            "outlier_score": outlier_score,
            "property_analysis": outlier_analysis
        }
    
    def _detect_property_outlier(self, property_name: str, value: float) -> Dict[str, Any]:
        """Detect outlier for specific property."""
        # Extract historical values for this property
        historical_values = []
        for result in self.result_history:
            if property_name in result.properties:
                historical_values.append(result.properties[property_name])
        
        if len(historical_values) < 3:
            return {"is_outlier": False, "outlier_score": 0.0}
        
        # Statistical outlier detection using IQR method
        sorted_values = sorted(historical_values)
        n = len(sorted_values)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        # Calculate outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if value is outlier
        is_outlier = value < lower_bound or value > upper_bound
        
        # Calculate outlier score (0-1, where 1 is extreme outlier)
        if not is_outlier:
            outlier_score = 0.0
        else:
            if value < lower_bound:
                outlier_score = min(1.0, (lower_bound - value) / (iqr + 1e-10))
            else:
                outlier_score = min(1.0, (value - upper_bound) / (iqr + 1e-10))
        
        return {
            "is_outlier": is_outlier,
            "outlier_score": outlier_score,
            "bounds": (lower_bound, upper_bound),
            "outlier_direction": "low" if value < lower_bound else "high" if value > upper_bound else "normal"
        }


class AdaptiveProtocolEngine:
    """Main engine for real-time protocol adaptation."""
    
    def __init__(self, 
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
                 max_adaptation_rate: float = 0.2):
        self.adaptation_strategy = adaptation_strategy
        self.max_adaptation_rate = max_adaptation_rate  # Maximum relative change per adaptation
        
        # Components
        self.performance_monitor = PerformanceMonitor()
        self.outlier_detector = OutlierDetector()
        
        # State
        self.current_conditions = ExperimentalCondition()
        self.adaptation_rules: List[AdaptationRule] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        self.protocol_status = ProtocolStatus.STABLE
        
        # Learning
        self.successful_adaptations: List[Dict[str, Any]] = []
        self.failed_adaptations: List[Dict[str, Any]] = []
        
        # Initialize default adaptation rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self) -> None:
        """Initialize default adaptation rules."""
        
        # Performance degradation rule
        performance_rule = AdaptationRule(
            name="performance_degradation_response",
            trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
            condition_function=self._check_performance_degradation,
            adaptation_function=self._adapt_for_performance,
            priority=3,
            cooldown_period=timedelta(minutes=15)
        )
        
        # Outlier detection rule
        outlier_rule = AdaptationRule(
            name="outlier_investigation",
            trigger=AdaptationTrigger.OUTLIER_DETECTION,
            condition_function=self._check_outlier_conditions,
            adaptation_function=self._adapt_for_outliers,
            priority=2,
            cooldown_period=timedelta(minutes=5)
        )
        
        # Convergence stagnation rule
        stagnation_rule = AdaptationRule(
            name="convergence_boost",
            trigger=AdaptationTrigger.CONVERGENCE_STAGNATION,
            condition_function=self._check_stagnation,
            adaptation_function=self._adapt_for_stagnation,
            priority=1,
            cooldown_period=timedelta(minutes=30)
        )
        
        # Safety rule (highest priority)
        safety_rule = AdaptationRule(
            name="safety_response",
            trigger=AdaptationTrigger.SAFETY_CONCERN,
            condition_function=self._check_safety_conditions,
            adaptation_function=self._adapt_for_safety,
            priority=5,
            cooldown_period=timedelta(minutes=1)
        )
        
        self.adaptation_rules = [performance_rule, outlier_rule, stagnation_rule, safety_rule]
    
    async def process_realtime_result(self, result: RealTimeResult) -> Dict[str, Any]:
        """Process real-time experimental result and trigger adaptations if needed.
        
        Args:
            result: Real-time experimental result
            
        Returns:
            Processing summary including any adaptations made
        """
        logger.info(f"Processing real-time result: {result.id}")
        
        # Update monitoring systems
        performance_metrics = self.performance_monitor.update_performance(result)
        outlier_analysis = self.outlier_detector.add_result(result)
        
        # Check for adaptation triggers
        triggered_rules = []
        
        for rule in sorted(self.adaptation_rules, key=lambda r: r.priority, reverse=True):
            if not rule.can_trigger():
                continue
                
            try:
                if rule.condition_function and await self._evaluate_condition_async(rule.condition_function, result, performance_metrics, outlier_analysis):
                    triggered_rules.append(rule)
                    rule.trigger_rule()
                    logger.info(f"Adaptation rule triggered: {rule.name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        # Execute adaptations
        adaptations_made = []
        
        if triggered_rules:
            self.protocol_status = ProtocolStatus.ADAPTING
            
            for rule in triggered_rules:
                try:
                    if rule.adaptation_function:
                        adaptation_result = await self._execute_adaptation_async(
                            rule.adaptation_function, 
                            result, 
                            performance_metrics, 
                            outlier_analysis
                        )
                        
                        if adaptation_result:
                            adaptations_made.append({
                                "rule_name": rule.name,
                                "trigger": rule.trigger.value,
                                "adaptation": adaptation_result,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Update adaptation history
                            self.adaptation_history.append(adaptations_made[-1])
                            
                except Exception as e:
                    logger.error(f"Error executing adaptation {rule.name}: {e}")
        
        # Update protocol status
        if not adaptations_made:
            self.protocol_status = ProtocolStatus.STABLE
        
        # Prepare response
        response = {
            "result_id": result.id,
            "performance_metrics": performance_metrics,
            "outlier_analysis": outlier_analysis,
            "triggered_rules": [rule.name for rule in triggered_rules],
            "adaptations_made": adaptations_made,
            "protocol_status": self.protocol_status.value,
            "current_conditions": self.current_conditions.to_dict()
        }
        
        logger.info(f"Real-time processing complete. Adaptations: {len(adaptations_made)}")
        
        return response
    
    async def _evaluate_condition_async(self, 
                                      condition_func: Callable,
                                      result: RealTimeResult,
                                      performance_metrics: Dict[str, float],
                                      outlier_analysis: Dict[str, Any]) -> bool:
        """Evaluate condition function asynchronously."""
        if asyncio.iscoroutinefunction(condition_func):
            return await condition_func(result, performance_metrics, outlier_analysis)
        else:
            return condition_func(result, performance_metrics, outlier_analysis)
    
    async def _execute_adaptation_async(self,
                                      adaptation_func: Callable,
                                      result: RealTimeResult,
                                      performance_metrics: Dict[str, float],
                                      outlier_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute adaptation function asynchronously."""
        if asyncio.iscoroutinefunction(adaptation_func):
            return await adaptation_func(result, performance_metrics, outlier_analysis)
        else:
            return adaptation_func(result, performance_metrics, outlier_analysis)
    
    # Condition checking functions
    def _check_performance_degradation(self, 
                                     result: RealTimeResult,
                                     performance_metrics: Dict[str, float],
                                     outlier_analysis: Dict[str, Any]) -> bool:
        """Check if performance degradation occurred."""
        issues = self.performance_monitor.detect_performance_issues()
        return "performance_below_baseline" in issues or "declining_performance_trend" in issues
    
    def _check_outlier_conditions(self,
                                result: RealTimeResult,
                                performance_metrics: Dict[str, float],
                                outlier_analysis: Dict[str, Any]) -> bool:
        """Check if outlier conditions are met."""
        return outlier_analysis.get("is_outlier", False) and outlier_analysis.get("outlier_score", 0) > 0.6
    
    def _check_stagnation(self,
                         result: RealTimeResult,
                         performance_metrics: Dict[str, float],
                         outlier_analysis: Dict[str, Any]) -> bool:
        """Check for convergence stagnation."""
        issues = self.performance_monitor.detect_performance_issues()
        return "performance_stagnation" in issues
    
    def _check_safety_conditions(self,
                               result: RealTimeResult,
                               performance_metrics: Dict[str, float],
                               outlier_analysis: Dict[str, Any]) -> bool:
        """Check for safety concerns."""
        # Check for experimental errors
        if result.experimental_errors:
            return True
        
        # Check for extreme outliers that might indicate safety issues
        if outlier_analysis.get("outlier_score", 0) > 0.9:
            return True
        
        # Check for extreme conditions
        conditions = result.conditions
        if (conditions.temperature > 300 or conditions.temperature < 0 or
            conditions.pressure > 50 or conditions.pressure < 0 or
            conditions.ph < 1 or conditions.ph > 13):
            return True
        
        return False
    
    # Adaptation functions
    def _adapt_for_performance(self,
                             result: RealTimeResult,
                             performance_metrics: Dict[str, float],
                             outlier_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt conditions to improve performance."""
        logger.info("Adapting for performance improvement")
        
        old_conditions = self.current_conditions
        new_conditions = ExperimentalCondition()
        
        # Copy current conditions
        new_conditions.__dict__.update(old_conditions.__dict__)
        
        # Adaptation strategy based on performance trend
        trend = performance_metrics.get("performance_trend", 0)
        
        if trend < -0.3:  # Declining performance
            # Try opposite direction changes
            if self.adaptation_strategy in [AdaptationStrategy.AGGRESSIVE, AdaptationStrategy.EXPLORATORY]:
                adaptation_factor = 0.15
            else:
                adaptation_factor = 0.08
            
            # Adjust key parameters
            if np.random.random() < 0.5:
                new_conditions.temperature += np.random.choice([-1, 1]) * adaptation_factor * old_conditions.temperature
            
            if np.random.random() < 0.5:
                new_conditions.concentration_a += np.random.choice([-1, 1]) * adaptation_factor * old_conditions.concentration_a
            
            if np.random.random() < 0.3:
                new_conditions.reaction_time += np.random.choice([-1, 1]) * adaptation_factor * old_conditions.reaction_time
        
        else:  # Stable or improving - make smaller adjustments
            adaptation_factor = 0.05
            
            # Fine-tune based on recent performance
            current_perf = performance_metrics.get("current_performance", 0.5)
            if current_perf < 0.7:  # Room for improvement
                new_conditions.temperature += np.random.normal(0, adaptation_factor * old_conditions.temperature)
                new_conditions.concentration_a += np.random.normal(0, adaptation_factor * old_conditions.concentration_a)
        
        # Apply safety bounds
        new_conditions = self._apply_safety_bounds(new_conditions)
        
        # Update current conditions
        self.current_conditions = new_conditions
        
        return {
            "adaptation_type": "performance_optimization",
            "old_conditions": old_conditions.to_dict(),
            "new_conditions": new_conditions.to_dict(),
            "adaptation_factor": adaptation_factor,
            "performance_trend": trend
        }
    
    def _adapt_for_outliers(self,
                          result: RealTimeResult,
                          performance_metrics: Dict[str, float],
                          outlier_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt conditions based on outlier analysis."""
        logger.info("Adapting for outlier investigation")
        
        old_conditions = self.current_conditions
        new_conditions = ExperimentalCondition()
        new_conditions.__dict__.update(old_conditions.__dict__)
        
        # Analyze outlier characteristics
        outlier_score = outlier_analysis.get("outlier_score", 0)
        property_analysis = outlier_analysis.get("property_analysis", {})
        
        if outlier_score > 0.8:  # Extreme outlier - investigate cautiously
            adaptation_factor = 0.1
            
            # Move toward outlier conditions to investigate
            for prop, analysis in property_analysis.items():
                if analysis.get("outlier_direction") == "high":
                    # Outlier had high value - might be interesting
                    if prop == "efficiency" or prop == "stability":
                        # Positive outliers - explore this direction
                        new_conditions.temperature += adaptation_factor * old_conditions.temperature * 0.5
                        new_conditions.concentration_a += adaptation_factor * old_conditions.concentration_a * 0.3
                
        else:  # Moderate outlier - make small adjustments
            adaptation_factor = 0.05
            
            # Small exploratory changes
            new_conditions.concentration_b += np.random.normal(0, adaptation_factor * old_conditions.concentration_b)
            new_conditions.stirring_speed += np.random.normal(0, adaptation_factor * old_conditions.stirring_speed)
        
        # Apply safety bounds
        new_conditions = self._apply_safety_bounds(new_conditions)
        
        # Update current conditions
        self.current_conditions = new_conditions
        
        return {
            "adaptation_type": "outlier_investigation",
            "old_conditions": old_conditions.to_dict(),
            "new_conditions": new_conditions.to_dict(),
            "outlier_score": outlier_score,
            "investigation_direction": "exploratory"
        }
    
    def _adapt_for_stagnation(self,
                            result: RealTimeResult,
                            performance_metrics: Dict[str, float],
                            outlier_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt conditions to break out of stagnation."""
        logger.info("Adapting to break performance stagnation")
        
        old_conditions = self.current_conditions
        new_conditions = ExperimentalCondition()
        new_conditions.__dict__.update(old_conditions.__dict__)
        
        # More aggressive exploration to break stagnation
        if self.adaptation_strategy == AdaptationStrategy.EXPLORATORY:
            adaptation_factor = 0.25
        elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            adaptation_factor = 0.2
        else:
            adaptation_factor = 0.15
        
        # Make larger changes to multiple parameters
        new_conditions.temperature += np.random.normal(0, adaptation_factor * old_conditions.temperature)
        new_conditions.concentration_a += np.random.normal(0, adaptation_factor * old_conditions.concentration_a)
        new_conditions.concentration_b += np.random.normal(0, adaptation_factor * old_conditions.concentration_b)
        new_conditions.reaction_time += np.random.normal(0, adaptation_factor * old_conditions.reaction_time)
        
        # Occasionally try completely different regimes
        if np.random.random() < 0.3:
            # Jump to different parameter regime
            new_conditions.temperature += np.random.choice([-1, 1]) * 0.3 * old_conditions.temperature
            new_conditions.ph += np.random.choice([-1, 1]) * 1.0
        
        # Apply safety bounds
        new_conditions = self._apply_safety_bounds(new_conditions)
        
        # Update current conditions
        self.current_conditions = new_conditions
        
        return {
            "adaptation_type": "stagnation_breakthrough",
            "old_conditions": old_conditions.to_dict(),
            "new_conditions": new_conditions.to_dict(),
            "adaptation_factor": adaptation_factor,
            "exploration_type": "aggressive"
        }
    
    def _adapt_for_safety(self,
                        result: RealTimeResult,
                        performance_metrics: Dict[str, float],
                        outlier_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt conditions to address safety concerns."""
        logger.warning("Adapting for safety concerns")
        
        old_conditions = self.current_conditions
        new_conditions = ExperimentalCondition()
        new_conditions.__dict__.update(old_conditions.__dict__)
        
        # Move to safer conditions
        safety_adaptations = []
        
        # Temperature safety
        if old_conditions.temperature > 250:
            new_conditions.temperature = min(200, old_conditions.temperature * 0.9)
            safety_adaptations.append("temperature_reduction")
        elif old_conditions.temperature < 50:
            new_conditions.temperature = max(100, old_conditions.temperature * 1.1)
            safety_adaptations.append("temperature_increase")
        
        # Pressure safety
        if old_conditions.pressure > 10:
            new_conditions.pressure = min(5, old_conditions.pressure * 0.8)
            safety_adaptations.append("pressure_reduction")
        
        # pH safety
        if old_conditions.ph < 3:
            new_conditions.ph = max(4, old_conditions.ph + 1)
            safety_adaptations.append("ph_increase")
        elif old_conditions.ph > 11:
            new_conditions.ph = min(10, old_conditions.ph - 1)
            safety_adaptations.append("ph_decrease")
        
        # Concentration safety
        if old_conditions.concentration_a > 1.8:
            new_conditions.concentration_a = min(1.5, old_conditions.concentration_a * 0.9)
            safety_adaptations.append("concentration_a_reduction")
        
        if old_conditions.concentration_b > 1.8:
            new_conditions.concentration_b = min(1.5, old_conditions.concentration_b * 0.9)
            safety_adaptations.append("concentration_b_reduction")
        
        # Update current conditions
        self.current_conditions = new_conditions
        self.protocol_status = ProtocolStatus.EMERGENCY
        
        return {
            "adaptation_type": "safety_response",
            "old_conditions": old_conditions.to_dict(),
            "new_conditions": new_conditions.to_dict(),
            "safety_adaptations": safety_adaptations,
            "experimental_errors": result.experimental_errors
        }
    
    def _apply_safety_bounds(self, conditions: ExperimentalCondition) -> ExperimentalCondition:
        """Apply safety bounds to experimental conditions."""
        # Temperature bounds
        conditions.temperature = np.clip(conditions.temperature, 50, 300)
        
        # Pressure bounds
        conditions.pressure = np.clip(conditions.pressure, 0.1, 20)
        
        # Concentration bounds
        conditions.concentration_a = np.clip(conditions.concentration_a, 0.1, 2.0)
        conditions.concentration_b = np.clip(conditions.concentration_b, 0.1, 2.0)
        
        # Time bounds
        conditions.reaction_time = np.clip(conditions.reaction_time, 0.5, 48)
        
        # Stirring speed bounds
        conditions.stirring_speed = np.clip(conditions.stirring_speed, 100, 2000)
        
        # pH bounds
        conditions.ph = np.clip(conditions.ph, 2, 12)
        
        return conditions
    
    def learn_from_adaptation_outcomes(self, 
                                     adaptation_id: str,
                                     outcome_result: RealTimeResult,
                                     success: bool) -> None:
        """Learn from adaptation outcomes to improve future adaptations."""
        # Find the adaptation in history
        adaptation_record = None
        for record in self.adaptation_history:
            if record.get("adaptation", {}).get("adaptation_type") == adaptation_id:
                adaptation_record = record
                break
        
        if not adaptation_record:
            return
        
        # Create learning record
        learning_record = {
            "adaptation": adaptation_record,
            "outcome": outcome_result.to_dict(),
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            self.successful_adaptations.append(learning_record)
            logger.info(f"Learned successful adaptation: {adaptation_id}")
        else:
            self.failed_adaptations.append(learning_record)
            logger.info(f"Learned failed adaptation: {adaptation_id}")
        
        # Update adaptation strategies based on learning
        self._update_adaptation_strategies()
    
    def _update_adaptation_strategies(self) -> None:
        """Update adaptation strategies based on learned outcomes."""
        if len(self.successful_adaptations) + len(self.failed_adaptations) < 10:
            return  # Need more data to learn
        
        # Analyze success rates by adaptation type
        adaptation_types = set()
        for adaptation in self.successful_adaptations + self.failed_adaptations:
            adaptation_types.add(adaptation["adaptation"]["adaptation_type"])
        
        for adaptation_type in adaptation_types:
            successes = len([a for a in self.successful_adaptations 
                           if a["adaptation"]["adaptation_type"] == adaptation_type])
            failures = len([a for a in self.failed_adaptations 
                          if a["adaptation"]["adaptation_type"] == adaptation_type])
            
            success_rate = successes / (successes + failures) if (successes + failures) > 0 else 0
            
            # Adjust adaptation parameters based on success rate
            if success_rate > 0.7:
                # High success rate - can be more aggressive
                self.max_adaptation_rate = min(0.3, self.max_adaptation_rate * 1.1)
            elif success_rate < 0.3:
                # Low success rate - be more conservative
                self.max_adaptation_rate = max(0.05, self.max_adaptation_rate * 0.9)
        
        logger.info(f"Updated adaptation strategies. Max rate: {self.max_adaptation_rate:.3f}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation performance and current state."""
        total_adaptations = len(self.adaptation_history)
        
        # Calculate success metrics
        if self.successful_adaptations or self.failed_adaptations:
            total_learned = len(self.successful_adaptations) + len(self.failed_adaptations)
            success_rate = len(self.successful_adaptations) / total_learned if total_learned > 0 else 0
        else:
            success_rate = 0
        
        # Recent performance
        recent_performance = []
        for result in list(self.performance_monitor.performance_history)[-5:]:
            recent_performance.append(result)
        
        # Rule statistics
        rule_stats = {}
        for rule in self.adaptation_rules:
            rule_stats[rule.name] = {
                "trigger_count": rule.trigger_count,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
            }
        
        return {
            "protocol_status": self.protocol_status.value,
            "adaptation_strategy": self.adaptation_strategy.value,
            "total_adaptations": total_adaptations,
            "success_rate": success_rate,
            "max_adaptation_rate": self.max_adaptation_rate,
            "current_conditions": self.current_conditions.to_dict(),
            "recent_performance": recent_performance,
            "rule_statistics": rule_stats,
            "performance_issues_detected": self.performance_monitor.detect_performance_issues(),
            "baseline_performance": self.performance_monitor.baseline_performance
        }


# Global instance for easy access
_global_adaptive_engine: Optional[AdaptiveProtocolEngine] = None


def get_global_adaptive_engine() -> AdaptiveProtocolEngine:
    """Get the global adaptive protocol engine instance."""
    global _global_adaptive_engine
    if _global_adaptive_engine is None:
        _global_adaptive_engine = AdaptiveProtocolEngine()
    return _global_adaptive_engine


async def process_realtime_experiment_data(experimental_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to process real-time experimental data.
    
    Args:
        experimental_data: Dictionary containing experimental conditions and results
        
    Returns:
        Processing results including any adaptations made
    """
    # Convert to RealTimeResult
    conditions = ExperimentalCondition()
    if "conditions" in experimental_data:
        conditions.__dict__.update(experimental_data["conditions"])
    
    result = RealTimeResult(
        conditions=conditions,
        properties=experimental_data.get("properties", {}),
        quality_indicators=experimental_data.get("quality_indicators", {}),
        experimental_errors=experimental_data.get("experimental_errors", []),
        instrument_status=experimental_data.get("instrument_status", {}),
        confidence_score=experimental_data.get("confidence_score", 1.0)
    )
    
    # Process with adaptive engine
    engine = get_global_adaptive_engine()
    return await engine.process_realtime_result(result)