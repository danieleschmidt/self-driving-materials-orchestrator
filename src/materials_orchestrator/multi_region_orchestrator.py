"""Multi-region orchestrator for global materials discovery campaigns."""

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .global_deployment import (
    GlobalDeploymentManager,
    Region,
)

logger = logging.getLogger(__name__)


class CampaignStrategy(Enum):
    """Multi-region campaign strategies."""

    CENTRALIZED = "centralized"  # Single region coordination
    DISTRIBUTED = "distributed"  # Full distribution across regions
    FEDERATED = "federated"  # Coordinated but autonomous regions
    EDGE_COMPUTING = "edge"  # Edge-based processing


@dataclass
class RegionWorkload:
    """Workload assignment for a region."""

    region: Region
    experiment_count: int
    parameter_ranges: Dict[str, Tuple[float, float]]
    priority: int = 1
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0


@dataclass
class MultiRegionCampaign:
    """Multi-region campaign configuration."""

    campaign_id: str
    global_objective: Any
    strategy: CampaignStrategy
    workload_distribution: Dict[Region, RegionWorkload]
    coordination_region: Region
    sync_interval: int = 300  # seconds
    fault_tolerance: bool = True
    data_aggregation_strategy: str = "weighted_average"


class MultiRegionOrchestrator:
    """Orchestrates materials discovery across multiple regions."""

    def __init__(self, deployment_manager: GlobalDeploymentManager):
        """Initialize multi-region orchestrator."""
        self.deployment_manager = deployment_manager
        self.active_campaigns = {}
        self.region_coordinators = {}
        self.global_state = {}

        # Performance tracking
        self.region_performance = {}
        self.latency_matrix = {}

        # Synchronization
        self._sync_lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread = None

        # Initialize region coordinators
        self._initialize_region_coordinators()

        logger.info("Multi-region orchestrator initialized")

    def _initialize_region_coordinators(self):
        """Initialize coordinators for each active region."""
        for region in self.deployment_manager.active_regions:
            self.region_coordinators[region] = RegionCoordinator(
                region=region, deployment_manager=self.deployment_manager
            )

            # Initialize performance tracking
            self.region_performance[region] = {
                "throughput": 0.0,
                "success_rate": 0.0,
                "average_latency": 0.0,
                "resource_utilization": 0.0,
                "experiment_count": 0,
                "last_update": datetime.now(timezone.utc),
            }

    def create_multi_region_campaign(
        self,
        objective: Any,
        param_space: Dict[str, Tuple[float, float]],
        total_experiments: int = 100,
        strategy: CampaignStrategy = CampaignStrategy.DISTRIBUTED,
        preferred_regions: Optional[List[Region]] = None,
    ) -> MultiRegionCampaign:
        """Create a multi-region campaign."""

        campaign_id = f"global_campaign_{int(time.time())}"

        # Select regions for the campaign
        if preferred_regions:
            available_regions = [
                r
                for r in preferred_regions
                if r in self.deployment_manager.active_regions
            ]
        else:
            # Use all healthy and compliant regions
            available_regions = [
                r
                for r, data in self.deployment_manager.active_regions.items()
                if data["status"] == "active"
                and data["compliance_status"] == "compliant"
            ]

        if not available_regions:
            raise ValueError("No available regions for campaign")

        # Distribute workload across regions
        workload_distribution = self._distribute_workload(
            available_regions, param_space, total_experiments, strategy
        )

        # Select coordination region (lowest latency, highest performance)
        coordination_region = self._select_coordination_region(available_regions)

        campaign = MultiRegionCampaign(
            campaign_id=campaign_id,
            global_objective=objective,
            strategy=strategy,
            workload_distribution=workload_distribution,
            coordination_region=coordination_region,
        )

        self.active_campaigns[campaign_id] = campaign

        logger.info(
            f"Multi-region campaign {campaign_id} created with {len(available_regions)} regions"
        )
        return campaign

    def _distribute_workload(
        self,
        regions: List[Region],
        param_space: Dict[str, Tuple[float, float]],
        total_experiments: int,
        strategy: CampaignStrategy,
    ) -> Dict[Region, RegionWorkload]:
        """Distribute workload across regions."""

        workload_distribution = {}

        if strategy == CampaignStrategy.CENTRALIZED:
            # Assign all work to the primary region
            primary_region = self.deployment_manager.config.primary_region
            if primary_region in regions:
                workload_distribution[primary_region] = RegionWorkload(
                    region=primary_region,
                    experiment_count=total_experiments,
                    parameter_ranges=param_space.copy(),
                )
            else:
                # Fallback to first available region
                workload_distribution[regions[0]] = RegionWorkload(
                    region=regions[0],
                    experiment_count=total_experiments,
                    parameter_ranges=param_space.copy(),
                )

        elif strategy == CampaignStrategy.DISTRIBUTED:
            # Distribute experiments evenly across regions
            base_experiments = total_experiments // len(regions)
            remaining_experiments = total_experiments % len(regions)

            for i, region in enumerate(regions):
                experiment_count = base_experiments
                if i < remaining_experiments:
                    experiment_count += 1

                workload_distribution[region] = RegionWorkload(
                    region=region,
                    experiment_count=experiment_count,
                    parameter_ranges=param_space.copy(),
                    priority=1,
                )

        elif strategy == CampaignStrategy.FEDERATED:
            # Distribute based on region performance and capacity
            region_weights = self._calculate_region_weights(regions)

            for region in regions:
                weight = region_weights.get(region, 1.0)
                experiment_count = int(total_experiments * weight)

                workload_distribution[region] = RegionWorkload(
                    region=region,
                    experiment_count=experiment_count,
                    parameter_ranges=param_space.copy(),
                    priority=int(weight * 10),  # Convert to priority scale
                )

        elif strategy == CampaignStrategy.EDGE_COMPUTING:
            # Partition parameter space geographically
            workload_distribution = self._partition_parameter_space(
                regions, param_space, total_experiments
            )

        return workload_distribution

    def _calculate_region_weights(self, regions: List[Region]) -> Dict[Region, float]:
        """Calculate relative weights for regions based on performance."""
        weights = {}
        total_performance = 0.0

        for region in regions:
            performance = self.region_performance.get(region, {})

            # Calculate composite performance score
            throughput = performance.get("throughput", 1.0)
            success_rate = performance.get("success_rate", 0.8)
            utilization = 1.0 - performance.get(
                "resource_utilization", 0.5
            )  # Lower utilization = higher weight

            score = throughput * success_rate * utilization
            weights[region] = max(score, 0.1)  # Minimum weight
            total_performance += weights[region]

        # Normalize weights
        if total_performance > 0:
            weights = {
                region: weight / total_performance for region, weight in weights.items()
            }
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(regions)
            weights = dict.fromkeys(regions, equal_weight)

        return weights

    def _partition_parameter_space(
        self,
        regions: List[Region],
        param_space: Dict[str, Tuple[float, float]],
        total_experiments: int,
    ) -> Dict[Region, RegionWorkload]:
        """Partition parameter space across regions."""
        workload_distribution = {}

        # Simple partitioning: divide the first parameter equally
        if not param_space:
            return workload_distribution

        first_param = list(param_space.keys())[0]
        param_min, param_max = param_space[first_param]
        param_range = param_max - param_min

        experiments_per_region = total_experiments // len(regions)

        for i, region in enumerate(regions):
            # Calculate parameter subset for this region
            region_param_min = param_min + (i * param_range / len(regions))
            region_param_max = param_min + ((i + 1) * param_range / len(regions))

            # Create regional parameter space
            regional_param_space = param_space.copy()
            regional_param_space[first_param] = (region_param_min, region_param_max)

            workload_distribution[region] = RegionWorkload(
                region=region,
                experiment_count=experiments_per_region,
                parameter_ranges=regional_param_space,
                priority=1,
            )

        return workload_distribution

    def _select_coordination_region(self, regions: List[Region]) -> Region:
        """Select the best region for coordination."""
        # Prefer primary region if available
        primary_region = self.deployment_manager.config.primary_region
        if primary_region in regions:
            return primary_region

        # Otherwise select based on performance
        best_region = regions[0]
        best_score = 0.0

        for region in regions:
            performance = self.region_performance.get(region, {})
            score = (
                performance.get("throughput", 1.0) * 0.4
                + performance.get("success_rate", 0.8) * 0.3
                + (1.0 - performance.get("average_latency", 0.1)) * 0.3
            )

            if score > best_score:
                best_score = score
                best_region = region

        return best_region

    def execute_multi_region_campaign(
        self, campaign: MultiRegionCampaign
    ) -> Dict[str, Any]:
        """Execute a multi-region campaign."""
        logger.info(f"Starting multi-region campaign {campaign.campaign_id}")

        start_time = datetime.now(timezone.utc)

        # Initialize global state
        self.global_state[campaign.campaign_id] = {
            "status": "running",
            "start_time": start_time,
            "completed_regions": set(),
            "total_experiments": 0,
            "successful_experiments": 0,
            "best_result": None,
            "region_results": {},
        }

        # Execute experiments in each region
        if campaign.strategy == CampaignStrategy.CENTRALIZED:
            results = self._execute_centralized_campaign(campaign)
        elif campaign.strategy == CampaignStrategy.DISTRIBUTED:
            results = self._execute_distributed_campaign(campaign)
        elif campaign.strategy == CampaignStrategy.FEDERATED:
            results = self._execute_federated_campaign(campaign)
        else:  # EDGE_COMPUTING
            results = self._execute_edge_campaign(campaign)

        end_time = datetime.now(timezone.utc)

        # Aggregate results
        aggregated_results = self._aggregate_regional_results(campaign, results)
        aggregated_results.update(
            {
                "campaign_id": campaign.campaign_id,
                "strategy": campaign.strategy.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "regions_used": list(campaign.workload_distribution.keys()),
            }
        )

        # Cleanup
        if campaign.campaign_id in self.global_state:
            del self.global_state[campaign.campaign_id]

        if campaign.campaign_id in self.active_campaigns:
            del self.active_campaigns[campaign.campaign_id]

        logger.info(f"Multi-region campaign {campaign.campaign_id} completed")
        return aggregated_results

    def _execute_distributed_campaign(
        self, campaign: MultiRegionCampaign
    ) -> Dict[Region, Any]:
        """Execute campaign with distributed strategy."""
        regional_results = {}

        # Execute in parallel across regions
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(campaign.workload_distribution)
        ) as executor:
            future_to_region = {}

            for region, workload in campaign.workload_distribution.items():
                coordinator = self.region_coordinators[region]
                future = executor.submit(
                    coordinator.execute_regional_campaign,
                    campaign.global_objective,
                    workload,
                )
                future_to_region[future] = region

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    result = future.result()
                    regional_results[region] = result

                    # Update global state
                    self._update_global_state(campaign.campaign_id, region, result)

                except Exception as e:
                    logger.error(f"Region {region.value} failed: {e}")
                    regional_results[region] = {"error": str(e), "success": False}

        return regional_results

    def _execute_federated_campaign(
        self, campaign: MultiRegionCampaign
    ) -> Dict[Region, Any]:
        """Execute campaign with federated strategy (coordinated optimization)."""
        regional_results = {}

        # Federated learning approach - iterative coordination
        max_iterations = 5
        convergence_threshold = 0.01

        for iteration in range(max_iterations):
            logger.info(f"Federated iteration {iteration + 1}/{max_iterations}")

            # Execute experiments in parallel
            iteration_results = {}

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(campaign.workload_distribution)
            ) as executor:
                future_to_region = {}

                for region, workload in campaign.workload_distribution.items():
                    coordinator = self.region_coordinators[region]

                    # Adjust workload based on global knowledge
                    adjusted_workload = self._adjust_workload_for_federation(
                        workload, iteration, campaign.campaign_id
                    )

                    future = executor.submit(
                        coordinator.execute_regional_campaign,
                        campaign.global_objective,
                        adjusted_workload,
                    )
                    future_to_region[future] = region

                # Collect iteration results
                for future in concurrent.futures.as_completed(future_to_region):
                    region = future_to_region[future]
                    try:
                        result = future.result()
                        iteration_results[region] = result
                    except Exception as e:
                        logger.error(
                            f"Region {region.value} failed in iteration {iteration}: {e}"
                        )
                        iteration_results[region] = {"error": str(e), "success": False}

            # Update global knowledge
            self._update_federated_knowledge(campaign.campaign_id, iteration_results)

            # Check convergence
            if self._check_federated_convergence(
                iteration_results, convergence_threshold
            ):
                logger.info(
                    f"Federated campaign converged after {iteration + 1} iterations"
                )
                break

        # Aggregate all iteration results
        for region in campaign.workload_distribution:
            if region in iteration_results:
                regional_results[region] = iteration_results[region]

        return regional_results

    def _execute_centralized_campaign(
        self, campaign: MultiRegionCampaign
    ) -> Dict[Region, Any]:
        """Execute campaign with centralized strategy."""
        # Execute only in coordination region
        coordination_region = campaign.coordination_region
        workload = campaign.workload_distribution[coordination_region]

        coordinator = self.region_coordinators[coordination_region]
        result = coordinator.execute_regional_campaign(
            campaign.global_objective, workload
        )

        return {coordination_region: result}

    def _execute_edge_campaign(
        self, campaign: MultiRegionCampaign
    ) -> Dict[Region, Any]:
        """Execute campaign with edge computing strategy."""
        # Similar to distributed but with parameter space partitioning
        return self._execute_distributed_campaign(campaign)

    def _aggregate_regional_results(
        self, campaign: MultiRegionCampaign, regional_results: Dict[Region, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple regions."""
        aggregated = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "success_rate": 0.0,
            "best_result": None,
            "regional_summary": {},
            "performance_comparison": {},
        }

        all_experiments = []
        best_fitness = float("-inf")

        for region, result in regional_results.items():
            if result.get("success", False):
                experiments = result.get("total_experiments", 0)
                successes = result.get("successful_experiments", 0)

                aggregated["total_experiments"] += experiments
                aggregated["successful_experiments"] += successes

                # Track regional performance
                aggregated["regional_summary"][region.value] = {
                    "experiments": experiments,
                    "success_rate": successes / experiments if experiments > 0 else 0,
                    "best_result": result.get("best_result"),
                    "duration": result.get("duration", 0),
                }

                # Find global best
                regional_best = result.get("best_result")
                if regional_best:
                    regional_fitness = regional_best.get("fitness", float("-inf"))
                    if regional_fitness > best_fitness:
                        best_fitness = regional_fitness
                        aggregated["best_result"] = regional_best

                # Collect all experiments for analysis
                if "experiments" in result:
                    all_experiments.extend(result["experiments"])

        # Calculate overall success rate
        if aggregated["total_experiments"] > 0:
            aggregated["success_rate"] = (
                aggregated["successful_experiments"] / aggregated["total_experiments"]
            )

        # Performance comparison
        if len(regional_results) > 1:
            aggregated["performance_comparison"] = self._compare_regional_performance(
                regional_results
            )

        # Add experiment details
        aggregated["all_experiments"] = all_experiments

        return aggregated

    def _compare_regional_performance(
        self, regional_results: Dict[Region, Any]
    ) -> Dict[str, Any]:
        """Compare performance across regions."""
        comparison = {
            "throughput_ranking": [],
            "success_rate_ranking": [],
            "efficiency_ranking": [],
            "latency_comparison": {},
        }

        # Calculate metrics for each region
        region_metrics = {}
        for region, result in regional_results.items():
            if result.get("success", False):
                experiments = result.get("total_experiments", 0)
                duration = result.get("duration", 1)
                successes = result.get("successful_experiments", 0)

                throughput = experiments / duration if duration > 0 else 0
                success_rate = successes / experiments if experiments > 0 else 0
                efficiency = throughput * success_rate

                region_metrics[region] = {
                    "throughput": throughput,
                    "success_rate": success_rate,
                    "efficiency": efficiency,
                }

        # Create rankings
        for metric in ["throughput", "success_rate", "efficiency"]:
            ranking = sorted(
                region_metrics.items(), key=lambda x: x[1][metric], reverse=True
            )
            comparison[f"{metric}_ranking"] = [
                {"region": r.value, "value": metrics[metric]} for r, metrics in ranking
            ]

        return comparison

    def _update_global_state(
        self, campaign_id: str, region: Region, result: Dict[str, Any]
    ):
        """Update global campaign state."""
        with self._sync_lock:
            if campaign_id in self.global_state:
                state = self.global_state[campaign_id]
                state["completed_regions"].add(region)
                state["region_results"][region] = result

                if result.get("success", False):
                    state["total_experiments"] += result.get("total_experiments", 0)
                    state["successful_experiments"] += result.get(
                        "successful_experiments", 0
                    )

                    # Update best result
                    regional_best = result.get("best_result")
                    if regional_best:
                        if state["best_result"] is None:
                            state["best_result"] = regional_best
                        else:
                            current_fitness = state["best_result"].get(
                                "fitness", float("-inf")
                            )
                            regional_fitness = regional_best.get(
                                "fitness", float("-inf")
                            )
                            if regional_fitness > current_fitness:
                                state["best_result"] = regional_best

    def _adjust_workload_for_federation(
        self, workload: RegionWorkload, iteration: int, campaign_id: str
    ) -> RegionWorkload:
        """Adjust workload based on federated learning knowledge."""
        # In a real implementation, this would adjust the parameter ranges
        # based on global knowledge from previous iterations
        adjusted_workload = RegionWorkload(
            region=workload.region,
            experiment_count=max(workload.experiment_count // (iteration + 1), 5),
            parameter_ranges=workload.parameter_ranges.copy(),
            priority=workload.priority,
            constraints=workload.constraints,
        )

        return adjusted_workload

    def _update_federated_knowledge(
        self, campaign_id: str, iteration_results: Dict[Region, Any]
    ):
        """Update global knowledge from federated iteration."""
        # Aggregate knowledge from all regions
        # This would typically involve updating global models or parameter distributions
        pass

    def _check_federated_convergence(
        self, iteration_results: Dict[Region, Any], threshold: float
    ) -> bool:
        """Check if federated optimization has converged."""
        # Simple convergence check - in practice this would be more sophisticated
        best_values = []
        for result in iteration_results.values():
            if result.get("success", False) and result.get("best_result"):
                best_values.append(result["best_result"].get("fitness", 0))

        if len(best_values) < 2:
            return False

        # Check if variance is below threshold
        import statistics

        variance = (
            statistics.variance(best_values) if len(best_values) > 1 else float("inf")
        )
        return variance < threshold

    def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get status of a multi-region campaign."""
        if campaign_id in self.global_state:
            return self.global_state[campaign_id].copy()
        elif campaign_id in self.active_campaigns:
            return {"status": "pending", "campaign_id": campaign_id}
        else:
            return {"status": "not_found", "campaign_id": campaign_id}


class RegionCoordinator:
    """Coordinates experiments within a single region."""

    def __init__(self, region: Region, deployment_manager: GlobalDeploymentManager):
        """Initialize region coordinator."""
        self.region = region
        self.deployment_manager = deployment_manager
        self.local_state = {}

    def execute_regional_campaign(
        self, objective: Any, workload: RegionWorkload
    ) -> Dict[str, Any]:
        """Execute campaign within this region."""
        try:
            # Import here to avoid circular imports
            from .core import AutonomousLab

            # Create regional lab instance
            lab = AutonomousLab()

            # Execute the workload
            start_time = datetime.now(timezone.utc)

            # For demonstration, run a simplified campaign
            campaign_result = lab.run_campaign(
                objective=objective,
                param_space=workload.parameter_ranges,
                max_experiments=workload.experiment_count,
                initial_samples=min(workload.experiment_count // 4, 10),
            )

            end_time = datetime.now(timezone.utc)

            # Format results for multi-region aggregation
            result = {
                "success": True,
                "region": self.region.value,
                "total_experiments": campaign_result.total_experiments,
                "successful_experiments": campaign_result.successful_experiments,
                "success_rate": campaign_result.success_rate,
                "best_result": {
                    "parameters": campaign_result.best_material.get("parameters", {}),
                    "properties": campaign_result.best_properties,
                    "fitness": campaign_result.get_best_fitness(),
                },
                "duration": (end_time - start_time).total_seconds(),
                "experiments": [exp.to_dict() for exp in campaign_result.experiments],
                "convergence_history": campaign_result.convergence_history,
            }

            return result

        except Exception as e:
            logger.error(f"Regional campaign failed in {self.region.value}: {e}")
            return {
                "success": False,
                "region": self.region.value,
                "error": str(e),
                "total_experiments": 0,
                "successful_experiments": 0,
            }


# Global multi-region orchestrator
_global_multi_region_orchestrator = None


def get_global_multi_region_orchestrator() -> Optional[MultiRegionOrchestrator]:
    """Get global multi-region orchestrator instance."""
    return _global_multi_region_orchestrator


def initialize_multi_region_orchestrator(
    deployment_manager: GlobalDeploymentManager,
) -> MultiRegionOrchestrator:
    """Initialize global multi-region orchestrator."""
    global _global_multi_region_orchestrator
    _global_multi_region_orchestrator = MultiRegionOrchestrator(deployment_manager)
    return _global_multi_region_orchestrator
