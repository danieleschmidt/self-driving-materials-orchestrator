"""Self-Driving Materials Orchestrator.

End-to-end agentic pipeline for autonomous materials-discovery experiments.
"""

import logging

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

logger = logging.getLogger(__name__)

from .advanced_monitoring import (
    AdvancedMonitoringSystem,
    AlertSeverity,
    MetricType,
    get_monitoring_system,
)
from .caching_system import ExperimentResultCache, get_global_experiment_cache
from .core import AutonomousLab, CampaignResult, Experiment, MaterialsObjective
from .database import ExperimentDatabase, ExperimentTracker
from .distributed_self_healing import (
    GlobalCoordinationLayer,
    GlobalRegion,
    NodeStatus,
    get_global_coordination_layer,
)
from .global_compliance import (
    ComplianceRegulation,
    GlobalComplianceManager,
    InternationalizationManager,
    get_compliance_manager,
    get_i18n_manager,
    t,
)
from .intelligent_failover import (
    ComponentType,
    FailoverStrategy,
    IntelligentFailoverManager,
    get_failover_manager,
)
from .ml_acceleration import IntelligentOptimizer, PropertyPredictor
from .monitoring import HealthMonitor
from .multi_region_deployment import (
    DeploymentRegion,
    MultiRegionDeploymentManager,
    RegionStatus,
    get_deployment_manager,
)
from .optimization import AdaptiveCache, ConcurrentExecutor
from .pipeline_guard import (
    FailureType,
    PipelineStatus,
    SelfHealingPipelineGuard,
    get_pipeline_guard,
)
from .planners import BayesianPlanner, GridPlanner, RandomPlanner
from .quantum_enhanced_pipeline_guard import (
    DistributedQuantumPipelineGuard,
    PipelineOptimizationProblem,
    QuantumConfiguration,
    create_quantum_optimization_problem,
    get_quantum_pipeline_guard,
)
from .robots import RobotOrchestrator
from .robots import SimulatedDriver as SimulatedRobot
from .robust_error_handling import (
    ErrorCategory,
    ErrorSeverity,
    RobustErrorHandler,
    get_global_error_handler,
    with_error_handling,
)
from .scaling_optimizer import PerformanceOptimizer, get_global_optimizer
from .security import SecurityManager
from .validation import ExperimentValidator

# Next-Generation AI Enhancements (Generation 4+)
try:
    from .autonomous_hypothesis_generator import (
        AutonomousHypothesisGenerator,
        HypothesisConfidence,
        HypothesisType,
        ScientificHypothesis,
        generate_scientific_hypotheses,
        get_global_hypothesis_generator,
    )
except ImportError as e:
    logger.warning(f"Autonomous hypothesis generator not available: {e}")

try:
    from .quantum_hybrid_optimizer import (
        OptimizationStrategy,
        QuantumBackend,
        QuantumHybridOptimizer,
        QuantumOptimizationProblem,
        QuantumOptimizationResult,
        get_global_quantum_optimizer,
        optimize_with_quantum_hybrid,
    )
except ImportError as e:
    logger.warning(f"Quantum hybrid optimizer not available: {e}")

try:
    from .federated_learning_coordinator import (
        FederatedLearningCoordinator,
        FederatedModel,
        FederationStatus,
        LabNode,
        LabRole,
        ModelUpdate,
        PrivacyLevel,
        create_federated_materials_network,
        get_global_federation_coordinator,
    )
except ImportError as e:
    logger.warning(f"Federated learning coordinator not available: {e}")

try:
    from .realtime_adaptive_protocols import (
        AdaptationStrategy,
        AdaptationTrigger,
        AdaptiveProtocolEngine,
        ExperimentalCondition,
        ProtocolStatus,
        RealTimeResult,
        get_global_adaptive_engine,
        process_realtime_experiment_data,
    )
except ImportError as e:
    logger.warning(f"Realtime adaptive protocols not available: {e}")

# Optional dashboard import (requires streamlit)
try:
    from .dashboard import LabDashboard

    DASHBOARD_AVAILABLE = True
except ImportError:
    LabDashboard = None
    DASHBOARD_AVAILABLE = False

__all__ = [
    "AutonomousLab",
    "MaterialsObjective",
    "Experiment",
    "CampaignResult",
    "BayesianPlanner",
    "RandomPlanner",
    "GridPlanner",
    "RobotOrchestrator",
    "SimulatedRobot",
    "ExperimentTracker",
    "ExperimentDatabase",
    "HealthMonitor",
    "SecurityManager",
    "ExperimentValidator",
    "AdaptiveCache",
    "ConcurrentExecutor",
    "IntelligentOptimizer",
    "PropertyPredictor",
    "PerformanceOptimizer",
    "ExperimentResultCache",
    "get_global_optimizer",
    "get_global_experiment_cache",
    "SelfHealingPipelineGuard",
    "get_pipeline_guard",
    "PipelineStatus",
    "FailureType",
    "IntelligentFailoverManager",
    "get_failover_manager",
    "ComponentType",
    "FailoverStrategy",
    "RobustErrorHandler",
    "get_global_error_handler",
    "ErrorSeverity",
    "ErrorCategory",
    "with_error_handling",
    "AdvancedMonitoringSystem",
    "get_monitoring_system",
    "MetricType",
    "AlertSeverity",
    "DistributedQuantumPipelineGuard",
    "get_quantum_pipeline_guard",
    "QuantumConfiguration",
    "PipelineOptimizationProblem",
    "GlobalCoordinationLayer",
    "get_global_coordination_layer",
    "GlobalRegion",
    "NodeStatus",
    "GlobalComplianceManager",
    "get_compliance_manager",
    "InternationalizationManager",
    "get_i18n_manager",
    "ComplianceRegulation",
    "t",
    "MultiRegionDeploymentManager",
    "get_deployment_manager",
    "DeploymentRegion",
    "RegionStatus",
    "DASHBOARD_AVAILABLE",
    # Next-Generation AI Enhancements
    "AutonomousHypothesisGenerator",
    "ScientificHypothesis",
    "HypothesisType",
    "HypothesisConfidence",
    "generate_scientific_hypotheses",
    "get_global_hypothesis_generator",
    "QuantumHybridOptimizer",
    "QuantumOptimizationProblem",
    "QuantumOptimizationResult",
    "OptimizationStrategy",
    "QuantumBackend",
    "optimize_with_quantum_hybrid",
    "get_global_quantum_optimizer",
    "FederatedLearningCoordinator",
    "LabNode",
    "FederatedModel",
    "ModelUpdate",
    "LabRole",
    "PrivacyLevel",
    "FederationStatus",
    "create_federated_materials_network",
    "get_global_federation_coordinator",
    "AdaptiveProtocolEngine",
    "ExperimentalCondition",
    "RealTimeResult",
    "AdaptationStrategy",
    "AdaptationTrigger",
    "ProtocolStatus",
    "process_realtime_experiment_data",
    "get_global_adaptive_engine",
    # Factory functions
    "create_database",
    "create_health_monitor",
    "create_security_manager",
    "create_validator",
    "create_intelligent_optimizer",
    "get_global_cache",
    "get_global_executor",
    "create_default_robots",
]

# Add LabDashboard to __all__ if available
if DASHBOARD_AVAILABLE:
    __all__.append("LabDashboard")


# Factory functions for convenience
def create_database(url: str = "mongodb://localhost:27017/") -> ExperimentTracker:
    """Create experiment database tracker."""
    return ExperimentTracker(connection_string=url)


def create_health_monitor() -> HealthMonitor:
    """Create health monitor instance."""
    return HealthMonitor()


def create_security_manager() -> SecurityManager:
    """Create security manager instance."""
    return SecurityManager()


def create_validator() -> ExperimentValidator:
    """Create experiment validator instance."""
    return ExperimentValidator()


def create_intelligent_optimizer(target_property: str) -> IntelligentOptimizer:
    """Create intelligent optimizer instance."""
    return IntelligentOptimizer(target_property=target_property)


def get_global_cache() -> AdaptiveCache:
    """Get global adaptive cache instance."""
    return AdaptiveCache()


def get_global_executor() -> ConcurrentExecutor:
    """Get global concurrent executor instance."""
    return ConcurrentExecutor()


def create_default_robots() -> list:
    """Create default robot configuration."""
    return [SimulatedRobot()]
