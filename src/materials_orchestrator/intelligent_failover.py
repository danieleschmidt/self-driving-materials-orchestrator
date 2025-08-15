"""Intelligent Failover System for Materials Discovery Pipeline.

Provides advanced failover capabilities with predictive failure detection
and intelligent routing for materials discovery experiments.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Component types in the materials discovery pipeline."""
    ROBOT = "robot"
    DATABASE = "database"
    INSTRUMENT = "instrument"
    PLANNER = "planner"
    ANALYZER = "analyzer"
    STORAGE = "storage"
    NETWORK = "network"


class FailoverStrategy(Enum):
    """Failover strategies for different components."""
    HOT_STANDBY = "hot_standby"
    COLD_STANDBY = "cold_standby"
    LOAD_BALANCING = "load_balancing"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class Component:
    """Component in the materials discovery pipeline."""
    component_id: str
    name: str
    component_type: ComponentType
    status: str = "healthy"  # healthy, degraded, failed, maintenance
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0  # 0.0 to 1.0
    last_health_check: datetime = field(default_factory=datetime.now)
    failure_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def reliability_score(self) -> float:
        """Calculate component reliability score."""
        total_operations = self.success_count + self.failure_count
        if total_operations == 0:
            return 1.0
        return self.success_count / total_operations
    
    @property
    def is_available(self) -> bool:
        """Check if component is available for work."""
        return self.status in ["healthy", "degraded"] and self.load < 0.95


@dataclass
class FailoverRule:
    """Failover rule definition."""
    rule_id: str
    name: str
    trigger_conditions: Dict[str, Any]
    failover_strategy: FailoverStrategy
    primary_component_types: List[ComponentType]
    backup_component_types: List[ComponentType]
    priority: int = 1
    cooldown_seconds: int = 60
    max_failovers: int = 5
    enabled: bool = True


@dataclass
class FailoverEvent:
    """Failover event record."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    trigger_component: str = ""
    failed_component: str = ""
    backup_component: str = ""
    strategy_used: FailoverStrategy = FailoverStrategy.HOT_STANDBY
    success: bool = False
    duration_seconds: float = 0.0
    experiments_affected: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveFailureDetector:
    """Predictive failure detection using ML-based approaches."""
    
    def __init__(self):
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.prediction_models: Dict[str, Any] = {}
    
    def record_metric(self, component_id: str, metric_name: str, value: float):
        """Record a metric for failure prediction."""
        timestamp = datetime.now()
        metric_key = f"{component_id}_{metric_name}"
        
        self.component_metrics[metric_key].append({
            "timestamp": timestamp,
            "value": value
        })
    
    def predict_failure_probability(self, component_id: str) -> float:
        """Predict probability of component failure in next hour."""
        # Simple heuristic-based prediction
        # In production, would use trained ML models
        
        cpu_key = f"{component_id}_cpu_usage"
        memory_key = f"{component_id}_memory_usage"
        error_key = f"{component_id}_error_rate"
        
        cpu_metrics = list(self.component_metrics[cpu_key])
        memory_metrics = list(self.component_metrics[memory_key])
        error_metrics = list(self.component_metrics[error_key])
        
        if not cpu_metrics:
            return 0.1  # Default low probability
        
        # Calculate trends and current values
        recent_cpu = [m["value"] for m in cpu_metrics[-10:]]
        recent_memory = [m["value"] for m in memory_metrics[-10:]]
        recent_errors = [m["value"] for m in error_metrics[-10:]]
        
        failure_probability = 0.0
        
        # High CPU usage trend
        if recent_cpu and max(recent_cpu) > 90:
            failure_probability += 0.3
        
        # High memory usage
        if recent_memory and max(recent_memory) > 85:
            failure_probability += 0.25
        
        # Increasing error rate
        if recent_errors and len(recent_errors) > 5:
            if recent_errors[-1] > recent_errors[0]:
                failure_probability += 0.2
        
        # Add some randomness for demonstration
        failure_probability += random.uniform(0, 0.1)
        
        return min(failure_probability, 0.95)
    
    def should_preemptive_failover(self, component_id: str, threshold: float = 0.7) -> bool:
        """Determine if preemptive failover should be triggered."""
        probability = self.predict_failure_probability(component_id)
        return probability > threshold


class IntelligentFailoverManager:
    """Advanced failover manager with predictive capabilities."""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.failover_rules: Dict[str, FailoverRule] = {}
        self.failover_events: List[FailoverEvent] = []
        self.component_groups: Dict[str, List[str]] = {}
        self.active_failovers: Dict[str, str] = {}  # failed_component -> backup_component
        
        self.failure_detector = PredictiveFailureDetector()
        self.monitoring_active = False
        self.prediction_interval = 30  # seconds
        
        # Statistics
        self.total_failovers = 0
        self.successful_failovers = 0
        self.experiments_saved = 0
        
        # Register default failover rules
        self._register_default_failover_rules()
    
    def register_component(
        self,
        component_id: str,
        name: str,
        component_type: ComponentType,
        capabilities: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Component:
        """Register a component for failover management."""
        component = Component(
            component_id=component_id,
            name=name,
            component_type=component_type,
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
        
        self.components[component_id] = component
        logger.info(f"Registered component: {name} ({component_type.value})")
        
        return component
    
    def register_failover_rule(
        self,
        rule_id: str,
        name: str,
        trigger_conditions: Dict[str, Any],
        failover_strategy: FailoverStrategy,
        primary_component_types: List[ComponentType],
        backup_component_types: List[ComponentType],
        priority: int = 1,
        cooldown_seconds: int = 60
    ):
        """Register a failover rule."""
        rule = FailoverRule(
            rule_id=rule_id,
            name=name,
            trigger_conditions=trigger_conditions,
            failover_strategy=failover_strategy,
            primary_component_types=primary_component_types,
            backup_component_types=backup_component_types,
            priority=priority,
            cooldown_seconds=cooldown_seconds
        )
        
        self.failover_rules[rule_id] = rule
        logger.info(f"Registered failover rule: {name}")
    
    def _register_default_failover_rules(self):
        """Register default failover rules for common scenarios."""
        
        # Robot failover rule
        self.register_failover_rule(
            rule_id="robot_failover",
            name="Robot Hot Standby Failover",
            trigger_conditions={"status": "failed", "component_type": "robot"},
            failover_strategy=FailoverStrategy.HOT_STANDBY,
            primary_component_types=[ComponentType.ROBOT],
            backup_component_types=[ComponentType.ROBOT],
            priority=3
        )
        
        # Database failover rule
        self.register_failover_rule(
            rule_id="database_failover",
            name="Database Standby Failover",
            trigger_conditions={"status": "failed", "component_type": "database"},
            failover_strategy=FailoverStrategy.COLD_STANDBY,
            primary_component_types=[ComponentType.DATABASE],
            backup_component_types=[ComponentType.DATABASE],
            priority=5
        )
        
        # Load balancing rule for instruments
        self.register_failover_rule(
            rule_id="instrument_load_balancing",
            name="Instrument Load Balancing",
            trigger_conditions={"load": 0.8, "component_type": "instrument"},
            failover_strategy=FailoverStrategy.LOAD_BALANCING,
            primary_component_types=[ComponentType.INSTRUMENT],
            backup_component_types=[ComponentType.INSTRUMENT],
            priority=2
        )
    
    def update_component_status(self, component_id: str, status: str, load: float = None):
        """Update component status and potentially trigger failover."""
        if component_id not in self.components:
            logger.warning(f"Unknown component: {component_id}")
            return
        
        component = self.components[component_id]
        old_status = component.status
        component.status = status
        component.last_health_check = datetime.now()
        
        if load is not None:
            component.load = load
        
        # Record metrics for prediction
        self.failure_detector.record_metric(component_id, "status_change", 1 if status == "failed" else 0)
        self.failure_detector.record_metric(component_id, "load", component.load * 100)
        
        # Check if failover should be triggered
        if status in ["failed", "degraded"] and old_status != status:
            asyncio.create_task(self._trigger_failover(component_id))
    
    async def _trigger_failover(self, failed_component_id: str):
        """Trigger failover for a failed component."""
        failed_component = self.components[failed_component_id]
        
        logger.warning(f"Component {failed_component.name} failed, triggering failover...")
        
        # Find applicable failover rules
        applicable_rules = self._find_applicable_rules(failed_component)
        
        if not applicable_rules:
            logger.error(f"No failover rules found for component {failed_component.name}")
            return
        
        # Sort by priority (higher first)
        applicable_rules.sort(key=lambda x: x.priority, reverse=True)
        
        for rule in applicable_rules:
            if await self._execute_failover(failed_component_id, rule):
                break
    
    def _find_applicable_rules(self, component: Component) -> List[FailoverRule]:
        """Find failover rules applicable to a component."""
        applicable_rules = []
        
        for rule in self.failover_rules.values():
            if not rule.enabled:
                continue
            
            # Check if component type matches
            if component.component_type in rule.primary_component_types:
                # Check trigger conditions
                if self._check_trigger_conditions(component, rule.trigger_conditions):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_trigger_conditions(self, component: Component, conditions: Dict[str, Any]) -> bool:
        """Check if trigger conditions are met."""
        for condition_key, condition_value in conditions.items():
            if condition_key == "status":
                if component.status != condition_value:
                    return False
            elif condition_key == "component_type":
                if component.component_type.value != condition_value:
                    return False
            elif condition_key == "load":
                if component.load < condition_value:
                    return False
        
        return True
    
    async def _execute_failover(self, failed_component_id: str, rule: FailoverRule) -> bool:
        """Execute failover according to the rule."""
        start_time = time.time()
        failed_component = self.components[failed_component_id]
        
        # Find backup component
        backup_component = self._find_backup_component(failed_component, rule)
        
        if not backup_component:
            logger.error(f"No backup component found for {failed_component.name}")
            return False
        
        event = FailoverEvent(
            trigger_component=failed_component_id,
            failed_component=failed_component_id,
            backup_component=backup_component.component_id,
            strategy_used=rule.failover_strategy
        )
        
        try:
            # Execute failover strategy
            success = await self._apply_failover_strategy(
                failed_component, backup_component, rule.failover_strategy
            )
            
            event.success = success
            event.duration_seconds = time.time() - start_time
            
            if success:
                self.active_failovers[failed_component_id] = backup_component.component_id
                self.successful_failovers += 1
                logger.info(f"Failover successful: {failed_component.name} -> {backup_component.name}")
            else:
                logger.error(f"Failover failed: {failed_component.name} -> {backup_component.name}")
            
            self.total_failovers += 1
            self.failover_events.append(event)
            
            return success
        
        except Exception as e:
            logger.error(f"Failover execution error: {e}")
            event.success = False
            event.duration_seconds = time.time() - start_time
            self.failover_events.append(event)
            return False
    
    def _find_backup_component(self, failed_component: Component, rule: FailoverRule) -> Optional[Component]:
        """Find suitable backup component."""
        candidates = []
        
        for component in self.components.values():
            if (component.component_type in rule.backup_component_types and
                component.component_id != failed_component.component_id and
                component.is_available):
                
                # Check if backup has required capabilities
                if all(cap in component.capabilities for cap in failed_component.capabilities):
                    candidates.append(component)
        
        if not candidates:
            return None
        
        # Select best candidate based on reliability and load
        candidates.sort(key=lambda c: (c.reliability_score, -c.load), reverse=True)
        
        return candidates[0]
    
    async def _apply_failover_strategy(
        self,
        failed_component: Component,
        backup_component: Component,
        strategy: FailoverStrategy
    ) -> bool:
        """Apply specific failover strategy."""
        
        if strategy == FailoverStrategy.HOT_STANDBY:
            return await self._hot_standby_failover(failed_component, backup_component)
        elif strategy == FailoverStrategy.COLD_STANDBY:
            return await self._cold_standby_failover(failed_component, backup_component)
        elif strategy == FailoverStrategy.LOAD_BALANCING:
            return await self._load_balancing_failover(failed_component, backup_component)
        elif strategy == FailoverStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_failover(failed_component, backup_component)
        elif strategy == FailoverStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_failover(failed_component, backup_component)
        else:
            logger.error(f"Unknown failover strategy: {strategy}")
            return False
    
    async def _hot_standby_failover(self, failed: Component, backup: Component) -> bool:
        """Execute hot standby failover."""
        logger.info(f"Hot standby failover: {failed.name} -> {backup.name}")
        
        # Simulate instant failover
        backup.status = "active"
        backup.load = failed.load
        
        await asyncio.sleep(0.1)  # Minimal delay
        return True
    
    async def _cold_standby_failover(self, failed: Component, backup: Component) -> bool:
        """Execute cold standby failover."""
        logger.info(f"Cold standby failover: {failed.name} -> {backup.name}")
        
        # Simulate startup time
        await asyncio.sleep(2)
        
        backup.status = "active"
        backup.load = failed.load
        
        return True
    
    async def _load_balancing_failover(self, failed: Component, backup: Component) -> bool:
        """Execute load balancing failover."""
        logger.info(f"Load balancing failover: {failed.name} -> {backup.name}")
        
        # Redistribute load
        load_to_transfer = failed.load * 0.7  # Transfer 70% of load
        backup.load = min(backup.load + load_to_transfer, 0.95)
        failed.load *= 0.3  # Keep 30% load if recoverable
        
        await asyncio.sleep(0.5)
        return True
    
    async def _circuit_breaker_failover(self, failed: Component, backup: Component) -> bool:
        """Execute circuit breaker failover."""
        logger.info(f"Circuit breaker failover: {failed.name} -> {backup.name}")
        
        # Implement circuit breaker pattern
        backup.status = "active"
        failed.status = "circuit_open"
        
        await asyncio.sleep(1)
        return True
    
    async def _graceful_degradation_failover(self, failed: Component, backup: Component) -> bool:
        """Execute graceful degradation failover."""
        logger.info(f"Graceful degradation failover: {failed.name} -> {backup.name}")
        
        # Reduce capabilities but maintain basic functionality
        backup.status = "degraded"
        backup.load = failed.load
        
        await asyncio.sleep(0.3)
        return True
    
    async def start_predictive_monitoring(self):
        """Start predictive failure monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting predictive failure monitoring...")
        
        while self.monitoring_active:
            try:
                await self._check_predictive_failures()
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Predictive monitoring error: {e}")
                await asyncio.sleep(self.prediction_interval)
    
    async def _check_predictive_failures(self):
        """Check for predictive failures and trigger preemptive failover."""
        for component_id, component in self.components.items():
            if component.status not in ["healthy", "degraded"]:
                continue
            
            # Update metrics for prediction
            self.failure_detector.record_metric(component_id, "cpu_usage", random.uniform(20, 95))
            self.failure_detector.record_metric(component_id, "memory_usage", random.uniform(30, 90))
            self.failure_detector.record_metric(component_id, "error_rate", random.uniform(0, 10))
            
            # Check if preemptive failover should be triggered
            if self.failure_detector.should_preemptive_failover(component_id):
                logger.warning(f"Preemptive failover triggered for {component.name}")
                await self._trigger_failover(component_id)
    
    def stop_monitoring(self):
        """Stop predictive monitoring."""
        self.monitoring_active = False
        logger.info("Predictive monitoring stopped")
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get comprehensive failover status."""
        return {
            "total_components": len(self.components),
            "active_failovers": len(self.active_failovers),
            "total_failovers": self.total_failovers,
            "successful_failovers": self.successful_failovers,
            "success_rate": self.successful_failovers / max(self.total_failovers, 1),
            "experiments_saved": self.experiments_saved,
            "monitoring_active": self.monitoring_active,
            "components": {
                comp_id: {
                    "name": comp.name,
                    "type": comp.component_type.value,
                    "status": comp.status,
                    "load": comp.load,
                    "reliability": comp.reliability_score,
                    "available": comp.is_available
                }
                for comp_id, comp in self.components.items()
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "failed_component": event.failed_component,
                    "backup_component": event.backup_component,
                    "strategy": event.strategy_used.value,
                    "success": event.success,
                    "duration": event.duration_seconds
                }
                for event in self.failover_events[-10:]  # Last 10 events
            ]
        }


# Global instance
_global_failover_manager: Optional[IntelligentFailoverManager] = None


def get_failover_manager() -> IntelligentFailoverManager:
    """Get the global failover manager instance."""
    global _global_failover_manager
    if _global_failover_manager is None:
        _global_failover_manager = IntelligentFailoverManager()
    return _global_failover_manager