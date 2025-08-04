"""Self-Driving Materials Orchestrator.

End-to-end agentic pipeline for autonomous materials-discovery experiments.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import AutonomousLab, MaterialsObjective, Experiment, CampaignResult
from .planners import BayesianPlanner, RandomPlanner, GridPlanner
from .robots import RobotOrchestrator, SimulatedRobot, create_default_robots
from .database import ExperimentTracker, create_database
from .monitoring import HealthMonitor, create_health_monitor
from .security import SecurityManager, create_security_manager
from .validation import ExperimentValidator, create_validator
from .optimization import AdaptiveCache, ConcurrentExecutor, get_global_cache, get_global_executor
from .ml_acceleration import IntelligentOptimizer, PropertyPredictor, create_intelligent_optimizer

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
    "create_default_robots",
    "ExperimentTracker",
    "create_database",
    "HealthMonitor",
    "create_health_monitor",
    "SecurityManager",
    "create_security_manager",
    "ExperimentValidator",
    "create_validator",
    "AdaptiveCache",
    "ConcurrentExecutor",
    "get_global_cache",
    "get_global_executor",
    "IntelligentOptimizer",
    "PropertyPredictor",
    "create_intelligent_optimizer",
    "DASHBOARD_AVAILABLE",
]

# Add LabDashboard to __all__ if available
if DASHBOARD_AVAILABLE:
    __all__.append("LabDashboard")