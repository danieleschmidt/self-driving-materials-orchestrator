"""Self-Driving Materials Orchestrator.

End-to-end agentic pipeline for autonomous materials-discovery experiments.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import AutonomousLab, MaterialsObjective, Experiment, CampaignResult
from .planners import BayesianPlanner, RandomPlanner, GridPlanner
from .robots import RobotOrchestrator, SimulatedDriver as SimulatedRobot
from .database import ExperimentTracker, ExperimentDatabase
from .monitoring import HealthMonitor
from .security import SecurityManager
from .validation import ExperimentValidator
from .optimization import AdaptiveCache, ConcurrentExecutor
from .scaling_optimizer import PerformanceOptimizer, get_global_optimizer
from .caching_system import ExperimentResultCache, get_global_experiment_cache
from .ml_acceleration import IntelligentOptimizer, PropertyPredictor

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
    "DASHBOARD_AVAILABLE",
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