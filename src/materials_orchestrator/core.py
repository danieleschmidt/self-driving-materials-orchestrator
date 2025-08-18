"""Core classes for autonomous materials discovery."""

import math
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    pass
import logging
from dataclasses import dataclass, field
from enum import Enum

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
            raise ValueError(
                f"Invalid optimization direction: {self.optimization_direction}"
            )

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
        enable_monitoring: bool = True,
    ):
        """Initialize autonomous laboratory.

        Args:
            robots: List of robot identifiers to connect
            instruments: List of instrument identifiers
            planner: Experiment planning algorithm
            database_url: MongoDB connection string
            experiment_simulator: Function to simulate experiments
            enable_monitoring: Enable health monitoring
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

        # Initialize enhanced monitoring and security
        if enable_monitoring:
            self._setup_monitoring()

        # Initialize performance optimization
        self._setup_performance_optimization()

        logger.info(
            f"Autonomous lab initialized with {len(self.robots)} robots and {len(self.instruments)} instruments"
        )

    def _setup_monitoring(self):
        """Setup health monitoring and security systems."""
        try:
            from .health_monitoring import get_global_health_monitor

            self.health_monitor = get_global_health_monitor()

            # Register custom health checks for this lab
            from .health_monitoring import ComponentType

            self.health_monitor.register_health_check(
                f"lab_{id(self)}_experiments",
                self._check_lab_health,
                ComponentType.CORE,
            )

            # Start monitoring if not already running
            if not self.health_monitor.monitoring_active:
                self.health_monitor.start_monitoring()

            logger.info("Health monitoring enabled for laboratory")

        except ImportError as e:
            logger.warning(f"Health monitoring not available: {e}")

    def _check_lab_health(self):
        """Custom health check for this laboratory instance."""
        from .health_monitoring import (
            ComponentHealth,
            ComponentType,
            HealthMetric,
            HealthStatus,
        )

        metrics = []

        # Check experiment success rate
        success_rate = self.success_rate
        if success_rate < 0.5:
            status = HealthStatus.CRITICAL
        elif success_rate < 0.8:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        metrics.append(
            HealthMetric(
                name="experiment_success_rate",
                value=success_rate,
                unit="fraction",
                status=status,
            )
        )

        # Check recent experiment count
        metrics.append(
            HealthMetric(
                name="total_experiments",
                value=self.total_experiments,
                unit="count",
                status=HealthStatus.HEALTHY,
            )
        )

        # Overall lab status
        overall_status = HealthStatus.HEALTHY
        if self.status == LabStatus.ERROR:
            overall_status = HealthStatus.CRITICAL
        elif self.status == LabStatus.PAUSED:
            overall_status = HealthStatus.WARNING

        return ComponentHealth(
            name=f"lab_{id(self)}",
            component_type=ComponentType.CORE,
            status=overall_status,
            metrics=metrics,
        )

    def _setup_performance_optimization(self):
        """Setup performance optimization systems."""
        try:
            from .performance_optimizer import get_global_performance_optimizer

            self.performance_optimizer = get_global_performance_optimizer()

            # Start performance monitoring
            self.performance_optimizer.start_performance_monitoring()

            logger.info("Performance optimization enabled for laboratory")

        except ImportError as e:
            logger.warning(f"Performance optimization not available: {e}")
            self.performance_optimizer = None

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
        import math
        import random

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
        efficiency = max(
            0,
            min(30, 25 * math.exp(-((band_gap - 1.4) ** 2) / 0.1) + random.gauss(0, 2)),
        )
        stability = max(
            0, min(1, 0.9 - abs(band_gap - 1.5) * 0.3 + random.gauss(0, 0.1))
        )

        # Random failure rate
        if random.random() < 0.05:  # 5% failure rate
            return {}

        return {
            "band_gap": round(band_gap, 3),
            "efficiency": round(efficiency, 2),
            "stability": round(stability, 3),
        }

    def run_experiment(self, parameters: Dict[str, Any]) -> "Experiment":
        """Run a single experiment with enhanced error handling and validation."""
        experiment = Experiment(
            parameters=parameters,
            status="running",
            metadata={"lab_id": "autonomous_lab_1"},
        )

        start_time = time.time()

        try:
            # Import validation here to avoid circular imports
            from .error_recovery import get_global_resilient_executor
            from .security_enhanced import get_global_security_manager
            from .validation import create_validator

            # Enhanced security validation
            security_manager = get_global_security_manager()
            is_valid, error_msg = security_manager.validate_request(
                {"parameters": parameters}, f"lab_{id(self)}", "run_experiment"
            )

            if not is_valid:
                experiment.status = "failed"
                experiment.metadata["security_error"] = error_msg
                logger.error(
                    f"Experiment {experiment.id} failed security validation: {error_msg}"
                )
                return experiment

            # Validate and sanitize input parameters
            validator = create_validator()
            validation_results = validator.validate_parameters(parameters)

            # Check for critical validation errors
            critical_errors = [
                r for r in validation_results if r.status.value == "invalid"
            ]
            if critical_errors:
                experiment.status = "failed"
                experiment.metadata["validation_errors"] = [
                    r.message for r in critical_errors
                ]
                logger.error(
                    f"Experiment {experiment.id} failed validation: {critical_errors[0].message}"
                )
                return experiment

            # Sanitize parameters for safety
            safe_params = security_manager.input_validator.sanitize_parameters(
                parameters
            )

            # Add safety checks
            if not self._safety_check(safe_params):
                experiment.status = "failed"
                experiment.metadata["error"] = "Safety check failed"
                logger.error(f"Experiment {experiment.id} failed safety check")
                return experiment

            # Simulate experiment execution time
            time.sleep(0.1)  # Simulate processing time

            # Run experiment with enhanced error handling using resilient executor
            resilient_executor = get_global_resilient_executor()
            results, success = resilient_executor.execute_with_recovery(
                self.experiment_simulator,
                safe_params,
                operation_name="experiment_simulation",
                context={"experiment_id": experiment.id, "parameters": safe_params},
            )

            # Fallback to retry mechanism if resilient executor fails
            if not success:
                results = self._run_experiment_with_retry(safe_params, max_retries=2)

            if results:
                experiment.results = results
                experiment.status = "completed"
                self._successful_experiments += 1

                # Validate results
                result_validation = validator.validate_results(results)
                warning_count = sum(
                    1 for r in result_validation if r.status.value == "warning"
                )
                if warning_count > 0:
                    experiment.metadata["result_warnings"] = warning_count
                    logger.warning(
                        f"Experiment {experiment.id} completed with {warning_count} result warnings"
                    )
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

    def _safety_check(self, parameters: Dict[str, Any]) -> bool:
        """Perform safety checks on experiment parameters."""
        # Temperature safety check
        temp = parameters.get("temperature", 0)
        if temp > 400:  # Above 400°C requires special handling
            logger.warning(f"High temperature experiment: {temp}°C")
            # In a real system, this would check for proper safety equipment

        # pH safety check
        ph = parameters.get("pH", 7)
        if ph < 2 or ph > 12:  # Extreme pH values
            logger.warning(f"Extreme pH experiment: {ph}")

        # Concentration safety check
        conc_a = parameters.get("precursor_A_conc", 0)
        conc_b = parameters.get("precursor_B_conc", 0)
        if conc_a > 3.0 or conc_b > 3.0:  # High concentrations
            logger.warning(f"High concentration experiment: A={conc_a}, B={conc_b}")

        return True  # All safety checks passed

    def _run_experiment_with_retry(
        self, parameters: Dict[str, Any], max_retries: int = 2
    ) -> Dict[str, float]:
        """Run experiment with retry logic for robustness."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                results = self.experiment_simulator(parameters)

                # Validate results are reasonable
                if results and self._validate_results(results):
                    return results
                elif attempt < max_retries:
                    logger.warning(
                        f"Experiment attempt {attempt + 1} produced invalid results, retrying..."
                    )
                    continue
                else:
                    return {}  # Failed all attempts

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Experiment attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    time.sleep(0.1)  # Brief delay before retry
                    continue
                else:
                    logger.error(
                        f"Experiment failed after {max_retries + 1} attempts: {e}"
                    )
                    raise

        return {}

    def _validate_results(self, results: Dict[str, float]) -> bool:
        """Validate experiment results are physically reasonable."""
        if not results:
            return False

        # Check for NaN or infinite values
        for key, value in results.items():
            if (
                not isinstance(value, (int, float))
                or math.isnan(float(value))
                or math.isinf(float(value))
            ):
                logger.warning(f"Invalid result value: {key}={value}")
                return False

        # Physical validation
        band_gap = results.get("band_gap")
        if band_gap and (band_gap < 0.1 or band_gap > 4.0):
            logger.warning(f"Unrealistic band gap: {band_gap} eV")
            return False

        efficiency = results.get("efficiency")
        if efficiency and (efficiency < 0 or efficiency > 50):
            logger.warning(f"Unrealistic efficiency: {efficiency}%")
            return False

        stability = results.get("stability")
        if stability and (stability < 0 or stability > 1):
            logger.warning(f"Invalid stability: {stability}")
            return False

        return True

    def run_campaign(
        self,
        objective: MaterialsObjective,
        param_space: Dict[str, tuple],
        initial_samples: int = 20,
        max_experiments: int = 500,
        stop_on_target: bool = True,
        convergence_patience: int = 50,
        concurrent_experiments: int = 1,
        enable_autonomous_reasoning: bool = True,
    ) -> "CampaignResult":
        """Run autonomous discovery campaign.

        Args:
            objective: Materials discovery objective
            param_space: Parameter space for optimization
            initial_samples: Initial random samples
            max_experiments: Maximum experiments to run
            stop_on_target: Stop when target is reached
            convergence_patience: Stop if no improvement for N experiments
            concurrent_experiments: Number of experiments to run concurrently
            enable_autonomous_reasoning: Enable autonomous decision making

        Returns:
            Campaign results summary
        """
        logger.info(f"Starting campaign for {objective.target_property}")
        self.status = LabStatus.RUNNING

        campaign_id = str(uuid.uuid4())
        start_time = datetime.now()
        convergence_history = []
        best_fitness = float("-inf")
        no_improvement_count = 0

        # Reset lab state
        self._experiments_run = 0
        self._successful_experiments = 0
        self._experiments_history = []

        # Initialize autonomous reasoning if enabled
        autonomous_reasoner = None
        if enable_autonomous_reasoning:
            try:
                from .autonomous_reasoning import ReasoningContext, get_global_reasoner

                autonomous_reasoner = get_global_reasoner()
                logger.info("Autonomous reasoning enabled for campaign")
            except ImportError:
                logger.warning("Autonomous reasoning not available")

        try:
            # Phase 1: Initial random sampling
            logger.info(f"Phase 1: Running {initial_samples} initial experiments")
            from .planners import RandomPlanner

            random_planner = RandomPlanner()

            initial_params = random_planner.suggest_next(
                initial_samples, param_space, []
            )

            # Run initial experiments with concurrency
            if (
                concurrent_experiments > 1
                and hasattr(self, "performance_optimizer")
                and self.performance_optimizer
            ):
                # Concurrent execution using performance optimizer
                experiment_tasks = [
                    (self.run_experiment, (params,), {}) for params in initial_params
                ]
                experiments = self.performance_optimizer.concurrent_execute(
                    experiment_tasks,
                    max_concurrent=min(concurrent_experiments, len(initial_params)),
                )
            else:
                # Sequential execution
                experiments = [self.run_experiment(params) for params in initial_params]

            # Process experiment results
            for experiment in experiments:
                if experiment and experiment.status == "completed":
                    property_value = experiment.results.get(objective.target_property)
                    if property_value is not None:
                        fitness = objective.calculate_fitness(property_value)

                        if fitness > best_fitness:
                            best_fitness = fitness
                            self._best_material = {
                                "parameters": experiment.parameters,
                                "properties": experiment.results,
                                "experiment_id": experiment.id,
                            }
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1

                        # Check if target reached
                        if stop_on_target and objective.evaluate_success(
                            property_value
                        ):
                            logger.info(f"Target reached! Property: {property_value}")
                            break

                        convergence_history.append(
                            {
                                "experiment": self._experiments_run,
                                "best_fitness": best_fitness,
                                "current_value": property_value,
                            }
                        )

            # Phase 2: Bayesian optimization (if planner available)
            if self.planner and self._experiments_run < max_experiments:
                logger.info("Phase 2: Bayesian optimization")

                while (
                    self._experiments_run < max_experiments
                    and no_improvement_count < convergence_patience
                ):

                    # Get suggestions from planner
                    current_results = self.get_results()
                    if not current_results:
                        break

                    suggestions = self.planner.suggest_next(
                        min(5, max_experiments - self._experiments_run),
                        param_space,
                        current_results,
                    )

                    # Run suggested experiments with concurrency
                    if (
                        concurrent_experiments > 1
                        and hasattr(self, "performance_optimizer")
                        and self.performance_optimizer
                    ):
                        # Concurrent execution
                        batch_size = min(
                            concurrent_experiments,
                            len(suggestions),
                            max_experiments - self._experiments_run,
                        )
                        batch_suggestions = suggestions[:batch_size]
                        experiment_tasks = [
                            (self.run_experiment, (params,), {})
                            for params in batch_suggestions
                        ]
                        batch_experiments = (
                            self.performance_optimizer.concurrent_execute(
                                experiment_tasks, max_concurrent=batch_size
                            )
                        )
                        experiments_to_process = batch_experiments
                    else:
                        # Sequential execution
                        experiments_to_process = []
                        for params in suggestions:
                            if self._experiments_run >= max_experiments:
                                break
                            experiment = self.run_experiment(params)
                            experiments_to_process.append(experiment)

                    # Process batch of experiments
                    for experiment in experiments_to_process:
                        if not experiment:
                            continue

                        if experiment.status == "completed":
                            property_value = experiment.results.get(
                                objective.target_property
                            )
                            if property_value is not None:
                                fitness = objective.calculate_fitness(property_value)

                                if fitness > best_fitness:
                                    best_fitness = fitness
                                    self._best_material = {
                                        "parameters": experiment.parameters,
                                        "properties": experiment.results,
                                        "experiment_id": experiment.id,
                                    }
                                    no_improvement_count = 0
                                    logger.info(
                                        f"New best: {property_value} (experiment {self._experiments_run})"
                                    )
                                else:
                                    no_improvement_count += 1

                                # Check if target reached
                                if stop_on_target and objective.evaluate_success(
                                    property_value
                                ):
                                    logger.info(
                                        f"Target reached! Property: {property_value}"
                                    )
                                    break

                                convergence_history.append(
                                    {
                                        "experiment": self._experiments_run,
                                        "best_fitness": best_fitness,
                                        "current_value": property_value,
                                    }
                                )

                    # Autonomous reasoning checkpoint
                    if autonomous_reasoner and self._experiments_run % 10 == 0:
                        reasoning_context = ReasoningContext(
                            current_state={
                                "experiments_run": self._experiments_run,
                                "success_rate": self.success_rate,
                                "best_fitness": best_fitness,
                                "no_improvement_count": no_improvement_count,
                            },
                            experiment_history=self._experiments_history,
                            objective=objective,
                        )

                        decision = autonomous_reasoner.make_decision(reasoning_context)
                        if decision:
                            logger.info(f"Autonomous decision: {decision.description}")

                            # Execute autonomous decisions
                            if decision.decision_type.value == "early_stopping":
                                logger.info("Autonomous early stopping decision made")
                                break
                            elif decision.decision_type.value == "parameter_adjustment":
                                # Could adjust convergence patience or other parameters
                                if "convergence_patience" in decision.parameters:
                                    convergence_patience = decision.parameters[
                                        "convergence_patience"
                                    ]

                    # Check convergence
                    if no_improvement_count >= convergence_patience:
                        logger.info(
                            f"Converged after {convergence_patience} experiments without improvement"
                        )
                        break

        except KeyboardInterrupt:
            logger.info("Campaign interrupted by user")
        except Exception as e:
            logger.error(f"Campaign failed: {e}")
            self.status = LabStatus.ERROR
            raise
        finally:
            self.status = LabStatus.IDLE

            # Update ML optimizer with campaign results if available
            try:
                from .ml_acceleration import create_intelligent_optimizer

                ml_optimizer = create_intelligent_optimizer(objective.target_property)
                for experiment in self._experiments_history:
                    if experiment.status == "completed":
                        exp_data = {
                            "parameters": experiment.parameters,
                            "results": experiment.results,
                        }
                        ml_optimizer.add_experiment_result(exp_data)
            except ImportError:
                pass  # ML acceleration not available

        end_time = datetime.now()

        # Create campaign result
        campaign_result = CampaignResult(
            campaign_id=campaign_id,
            objective=objective,
            best_material=self._best_material or {"parameters": {}, "properties": {}},
            total_experiments=self._experiments_run,
            successful_experiments=self._successful_experiments,
            best_properties=(
                self._best_material["properties"] if self._best_material else {}
            ),
            convergence_history=convergence_history,
            experiments=self._experiments_history.copy(),
            start_time=start_time,
            end_time=end_time,
        )

        # Generate advanced analytics if available
        try:
            from .advanced_analytics import get_global_analyzer

            analyzer = get_global_analyzer()
            analytics = analyzer.analyze_campaign(campaign_result)
            insights = analyzer.generate_insights_report(analytics)

            # Log key insights
            if insights["performance_summary"]["overall_rating"]:
                logger.info(
                    f"Campaign rating: {insights['performance_summary']['overall_rating']}"
                )

            # Store analytics in campaign result metadata
            if hasattr(campaign_result, "metadata"):
                campaign_result.metadata.update(
                    {"analytics": analytics, "insights": insights}
                )
            else:
                # Add metadata field if not exists
                campaign_result.analytics = analytics
                campaign_result.insights = insights

        except ImportError:
            logger.warning("Advanced analytics not available")

        logger.info(
            f"Campaign completed: {self._experiments_run} experiments, "
            f"{self.success_rate:.1%} success rate"
        )

        return campaign_result

    def _validate_experiment_parameters(self, parameters: Dict[str, float]) -> bool:
        """Validate experiment parameters."""
        from .error_handling import ValidationError

        for key, value in parameters.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"Parameter {key} must be numeric, got {type(value)}"
                )

            if key == "temperature" and value < 0:
                raise ValidationError(f"Temperature cannot be negative: {value}")

            if key == "concentration" and (value < 0 or value > 10):
                raise ValidationError(
                    f"Concentration must be between 0 and 10: {value}"
                )

        return True


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
            return float("-inf")

        property_value = self.best_properties.get(self.objective.target_property, 0)
        return self.objective.calculate_fitness(property_value)
