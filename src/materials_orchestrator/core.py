"""Core classes for autonomous materials discovery."""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


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
    success_threshold: Optional[float] = None
    
    def __post_init__(self):
        """Validate objective parameters."""
        valid_directions = ["minimize", "maximize", "minimize_variance", "target"]
        if self.optimization_direction not in valid_directions:
            raise ValueError(f"Invalid optimization direction: {self.optimization_direction}")
            
        if self.target_range[0] >= self.target_range[1]:
            raise ValueError("target_range must be (min, max) with min < max")
    
    def evaluate_success(self, property_value: float) -> bool:
        """Check if a property value meets the objective."""
        if self.success_threshold is None:
            return self.target_range[0] <= property_value <= self.target_range[1]
        
        if self.optimization_direction == "minimize":
            return property_value <= self.success_threshold
        elif self.optimization_direction == "maximize":
            return property_value >= self.success_threshold
        else:
            return abs(property_value - self.success_threshold) <= 0.1
    
    def calculate_fitness(self, property_value: float) -> float:
        """Calculate fitness score for a property value."""
        if self.optimization_direction == "minimize":
            return -property_value
        elif self.optimization_direction == "maximize":
            return property_value
        elif self.optimization_direction == "target":
            target = sum(self.target_range) / 2
            return -abs(property_value - target)
        else:  # minimize_variance
            target = sum(self.target_range) / 2
            return -abs(property_value - target)


class AutonomousLab:
    """Main orchestrator for autonomous materials discovery experiments."""
    
    def __init__(
        self,
        robots: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        planner: Optional[Any] = None,
        database_url: str = "mongodb://localhost:27017/",
        experiment_simulator: Optional[Callable] = None,
    ):
        """Initialize autonomous laboratory.
        
        Args:
            robots: List of robot identifiers to connect
            instruments: List of instrument identifiers
            planner: Experiment planning algorithm
            database_url: MongoDB connection string
            experiment_simulator: Function to simulate experiments
        """
        self.robots = robots or []
        self.instruments = instruments or []
        self.planner = planner
        self.database_url = database_url
        self.experiment_simulator = experiment_simulator or self._default_simulator
        self.status = LabStatus.IDLE
        self._experiments_run = 0
        self._successful_experiments = 0
        self._best_material: Optional[Dict[str, Any]] = None
        self._experiments_history: List[Experiment] = []
        
    @property
    def total_experiments(self) -> int:
        """Total number of experiments executed."""
        return self._experiments_run
    
    @property
    def successful_experiments(self) -> int:
        """Number of successful experiments."""
        return self._successful_experiments
    
    @property
    def success_rate(self) -> float:
        """Experiment success rate."""
        if self._experiments_run == 0:
            return 0.0
        return self._successful_experiments / self._experiments_run
    
    @property
    def best_material(self) -> Optional[Dict[str, Any]]:
        """Best material discovered so far."""
        return self._best_material
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all experiment results for optimization."""
        results = []
        for exp in self._experiments_history:
            if exp.status == "completed" and exp.results:
                result = exp.parameters.copy()
                result.update(exp.results)
                results.append(result)
        return results
    
    def _default_simulator(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Default experiment simulator for testing.
        
        Simulates perovskite band gap based on composition and processing.
        """
        import random
        import math
        
        # Simulate band gap based on parameters
        # This is a simplified model for demonstration
        base_gap = 1.5
        
        # Temperature effect
        temp = parameters.get("temperature", 150)
        temp_effect = (temp - 150) / 100 * 0.1
        
        # Concentration effects
        conc_a = parameters.get("precursor_A_conc", 1.0)
        conc_b = parameters.get("precursor_B_conc", 1.0)
        conc_effect = (conc_a + conc_b - 2.0) * 0.05
        
        # Time effect
        time_hrs = parameters.get("reaction_time", 3)
        time_effect = math.log(time_hrs) * 0.02
        
        # Add realistic noise
        noise = random.gauss(0, 0.05)
        
        band_gap = base_gap + temp_effect + conc_effect + time_effect + noise
        band_gap = max(0.5, min(3.0, band_gap))  # Physical limits
        
        # Simulate additional properties
        efficiency = max(0, min(30, 25 * math.exp(-(band_gap - 1.4)**2 / 0.1) + random.gauss(0, 2)))
        stability = max(0, min(1, 0.9 - abs(band_gap - 1.5) * 0.3 + random.gauss(0, 0.1)))
        
        # Random failure rate
        if random.random() < 0.05:  # 5% failure rate
            return {}
        
        return {
            "band_gap": round(band_gap, 3),
            "efficiency": round(efficiency, 2),
            "stability": round(stability, 3),
        }
    
    def run_experiment(self, parameters: Dict[str, Any]) -> "Experiment":
        """Run a single experiment."""
        experiment = Experiment(
            parameters=parameters,
            status="running",
            metadata={"lab_id": "autonomous_lab_1"}
        )
        
        start_time = time.time()
        
        try:
            # Simulate experiment execution time
            time.sleep(0.1)  # Simulate processing time
            
            results = self.experiment_simulator(parameters)
            
            if results:
                experiment.results = results
                experiment.status = "completed"
                self._successful_experiments += 1
            else:
                experiment.status = "failed"
                
        except Exception as e:
            logger.error(f"Experiment {experiment.id} failed: {e}")
            experiment.status = "failed"
            experiment.metadata["error"] = str(e)
        
        experiment.duration = time.time() - start_time
        self._experiments_run += 1
        self._experiments_history.append(experiment)
        
        return experiment
    
    def run_campaign(
        self,
        objective: MaterialsObjective,
        param_space: Dict[str, tuple],
        initial_samples: int = 20,
        max_experiments: int = 500,
        stop_on_target: bool = True,
        convergence_patience: int = 50,
    ) -> "CampaignResult":
        """Run autonomous discovery campaign.
        
        Args:
            objective: Materials discovery objective
            param_space: Parameter space for optimization
            initial_samples: Initial random samples
            max_experiments: Maximum experiments to run
            stop_on_target: Stop when target is reached
            convergence_patience: Stop if no improvement for N experiments
            
        Returns:
            Campaign results summary
        """
        logger.info(f"Starting campaign for {objective.target_property}")
        self.status = LabStatus.RUNNING
        
        campaign_id = str(uuid.uuid4())
        start_time = datetime.now()
        convergence_history = []
        best_fitness = float('-inf')
        no_improvement_count = 0
        
        # Reset lab state
        self._experiments_run = 0
        self._successful_experiments = 0
        self._experiments_history = []
        
        try:
            # Phase 1: Initial random sampling
            logger.info(f"Phase 1: Running {initial_samples} initial experiments")
            from .planners import RandomPlanner
            random_planner = RandomPlanner()
            
            initial_params = random_planner.suggest_next(
                initial_samples, param_space, []
            )
            
            for params in initial_params:
                experiment = self.run_experiment(params)
                
                if experiment.status == "completed":
                    property_value = experiment.results.get(objective.target_property)
                    if property_value is not None:
                        fitness = objective.calculate_fitness(property_value)
                        
                        if fitness > best_fitness:
                            best_fitness = fitness
                            self._best_material = {
                                "parameters": experiment.parameters,
                                "properties": experiment.results,
                                "experiment_id": experiment.id
                            }
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        # Check if target reached
                        if stop_on_target and objective.evaluate_success(property_value):
                            logger.info(f"Target reached! Property: {property_value}")
                            break
                            
                        convergence_history.append({
                            "experiment": self._experiments_run,
                            "best_fitness": best_fitness,
                            "current_value": property_value,
                        })
            
            # Phase 2: Bayesian optimization (if planner available)
            if self.planner and self._experiments_run < max_experiments:
                logger.info("Phase 2: Bayesian optimization")
                
                while (self._experiments_run < max_experiments and 
                       no_improvement_count < convergence_patience):
                    
                    # Get suggestions from planner
                    current_results = self.get_results()
                    if not current_results:
                        break
                        
                    suggestions = self.planner.suggest_next(
                        min(5, max_experiments - self._experiments_run),
                        param_space,
                        current_results
                    )
                    
                    # Run suggested experiments
                    for params in suggestions:
                        if self._experiments_run >= max_experiments:
                            break
                            
                        experiment = self.run_experiment(params)
                        
                        if experiment.status == "completed":
                            property_value = experiment.results.get(objective.target_property)
                            if property_value is not None:
                                fitness = objective.calculate_fitness(property_value)
                                
                                if fitness > best_fitness:
                                    best_fitness = fitness
                                    self._best_material = {
                                        "parameters": experiment.parameters,
                                        "properties": experiment.results,
                                        "experiment_id": experiment.id
                                    }
                                    no_improvement_count = 0
                                    logger.info(f"New best: {property_value} (experiment {self._experiments_run})")
                                else:
                                    no_improvement_count += 1
                                
                                # Check if target reached
                                if stop_on_target and objective.evaluate_success(property_value):
                                    logger.info(f"Target reached! Property: {property_value}")
                                    break
                                    
                                convergence_history.append({
                                    "experiment": self._experiments_run,
                                    "best_fitness": best_fitness,
                                    "current_value": property_value,
                                })
                    
                    # Check convergence
                    if no_improvement_count >= convergence_patience:
                        logger.info(f"Converged after {convergence_patience} experiments without improvement")
                        break
        
        except KeyboardInterrupt:
            logger.info("Campaign interrupted by user")
        except Exception as e:
            logger.error(f"Campaign failed: {e}")
            self.status = LabStatus.ERROR
            raise
        finally:
            self.status = LabStatus.IDLE
        
        end_time = datetime.now()
        
        # Create campaign result
        campaign_result = CampaignResult(
            campaign_id=campaign_id,
            objective=objective,
            best_material=self._best_material or {"parameters": {}, "properties": {}},
            total_experiments=self._experiments_run,
            successful_experiments=self._successful_experiments,
            best_properties=self._best_material["properties"] if self._best_material else {},
            convergence_history=convergence_history,
            experiments=self._experiments_history.copy(),
            start_time=start_time,
            end_time=end_time,
        )
        
        logger.info(f"Campaign completed: {self._experiments_run} experiments, "
                   f"{self.success_rate:.1%} success rate")
        
        return campaign_result


@dataclass
class Experiment:
    """Individual experiment record."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "results": self.results,
            "status": self.status,
            "duration": self.duration,
            "metadata": self.metadata,
        }


@dataclass
class CampaignResult:
    """Results from an autonomous discovery campaign."""
    
    campaign_id: str
    objective: MaterialsObjective
    best_material: Dict[str, Any]
    total_experiments: int
    successful_experiments: int
    best_properties: Dict[str, float]
    convergence_history: List[Dict[str, Any]]
    experiments: List[Experiment] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate experiment success rate."""
        if self.total_experiments == 0:
            return 0.0
        return self.successful_experiments / self.total_experiments
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate campaign duration in hours."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() / 3600
    
    def get_best_fitness(self) -> float:
        """Get best fitness score achieved."""
        if not self.best_properties:
            return float('-inf')
        
        property_value = self.best_properties.get(self.objective.target_property, 0)
        return self.objective.calculate_fitness(property_value)