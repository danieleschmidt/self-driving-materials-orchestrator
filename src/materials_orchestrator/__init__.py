"""Self-Driving Materials Orchestrator.

End-to-end agentic pipeline for autonomous materials-discovery experiments.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import AutonomousLab, MaterialsObjective, Experiment, CampaignResult
from .planners import BayesianPlanner, RandomPlanner, GridPlanner

__all__ = [
    "AutonomousLab",
    "MaterialsObjective", 
    "Experiment",
    "CampaignResult",
    "BayesianPlanner",
    "RandomPlanner", 
    "GridPlanner",
]