"""Core classes for autonomous materials discovery."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LabStatus(Enum):
    """Laboratory status enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class MaterialsObjective:
    """Define optimization objective for materials discovery."""
    
    target_property: str
    target_range: tuple[float, float]
    optimization_direction: str = "minimize_variance"
    material_system: str = "general"
    constraints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate objective parameters."""
        valid_directions = ["minimize", "maximize", "minimize_variance", "target"]
        if self.optimization_direction not in valid_directions:
            raise ValueError(f"Invalid optimization direction: {self.optimization_direction}")


class AutonomousLab:
    """Main orchestrator for autonomous materials discovery experiments."""
    
    def __init__(
        self,
        robots: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        planner: Optional[Any] = None,
        database_url: str = "mongodb://localhost:27017/",
    ):
        """Initialize autonomous laboratory.
        
        Args:
            robots: List of robot identifiers to connect
            instruments: List of instrument identifiers
            planner: Experiment planning algorithm
            database_url: MongoDB connection string
        """
        self.robots = robots or []
        self.instruments = instruments or []
        self.planner = planner
        self.database_url = database_url
        self.status = LabStatus.IDLE
        self._experiments_run = 0
        self._best_material: Optional[Dict[str, Any]] = None
        
    @property
    def total_experiments(self) -> int:
        """Total number of experiments executed."""
        return self._experiments_run
    
    @property
    def best_material(self) -> Optional[Dict[str, Any]]:
        """Best material discovered so far."""
        return self._best_material
    
    def run_campaign(
        self,
        objective: MaterialsObjective,
        initial_samples: int = 20,
        max_experiments: int = 500,
        stop_on_target: bool = True,
    ) -> "CampaignResult":
        """Run autonomous discovery campaign.
        
        Args:
            objective: Materials discovery objective
            initial_samples: Initial random samples
            max_experiments: Maximum experiments to run
            stop_on_target: Stop when target is reached
            
        Returns:
            Campaign results summary
        """
        self.status = LabStatus.RUNNING
        
        # Placeholder implementation
        campaign_result = CampaignResult(
            best_material={"composition": "Pb0.5Sn0.5I3", "properties": {}},
            total_experiments=50,
            best_properties={"band_gap": 1.42},
            convergence_history=[],
        )
        
        self._experiments_run = campaign_result.total_experiments
        self._best_material = campaign_result.best_material
        self.status = LabStatus.IDLE
        
        return campaign_result


@dataclass
class CampaignResult:
    """Results from an autonomous discovery campaign."""
    
    best_material: Dict[str, Any]
    total_experiments: int
    best_properties: Dict[str, float]
    convergence_history: List[Dict[str, Any]]