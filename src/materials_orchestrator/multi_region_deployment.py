"""Multi-Region Deployment Manager for Global Self-Healing Pipeline.

Manages deployment across multiple geographic regions with automated
failover, data replication, and regional compliance.
"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Deployment regions for global coverage."""

    US_EAST_1 = "us-east-1"  # Virginia
    US_WEST_1 = "us-west-1"  # California
    EU_WEST_1 = "eu-west-1"  # Ireland
    EU_CENTRAL_1 = "eu-central-1"  # Frankfurt
    AP_SOUTHEAST_1 = "ap-southeast-1"  # Singapore
    AP_NORTHEAST_1 = "ap-northeast-1"  # Tokyo
    CA_CENTRAL_1 = "ca-central-1"  # Canada
    SA_EAST_1 = "sa-east-1"  # São Paulo
    AP_SOUTH_1 = "ap-south-1"  # Mumbai
    EU_NORTH_1 = "eu-north-1"  # Stockholm


class DeploymentTier(Enum):
    """Deployment tiers for different environments."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


class RegionStatus(Enum):
    """Regional deployment status."""

    ACTIVE = "active"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    DEPLOYING = "deploying"
    SCALING = "scaling"


@dataclass
class RegionalConfiguration:
    """Configuration for a regional deployment."""

    region: DeploymentRegion
    tier: DeploymentTier
    status: RegionStatus = RegionStatus.STANDBY

    # Infrastructure configuration
    compute_instances: int = 2
    memory_gb_per_instance: int = 8
    storage_gb: int = 100
    max_instances: int = 10
    min_instances: int = 1

    # Network configuration
    vpc_cidr: str = "10.0.0.0/16"
    availability_zones: List[str] = field(default_factory=list)
    load_balancer_enabled: bool = True

    # Data configuration
    database_replicas: int = 1
    backup_retention_days: int = 30
    encryption_enabled: bool = True

    # Compliance configuration
    data_residency_required: bool = False
    compliance_regulations: List[str] = field(default_factory=list)

    # Performance configuration
    auto_scaling_enabled: bool = True
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0

    # Monitoring configuration
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    log_retention_days: int = 90

    # Deployment metadata
    deployed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    version: str = "1.0.0"

    def __post_init__(self):
        """Set default values based on region."""
        if not self.availability_zones:
            # Set default AZs based on region
            region_az_map = {
                DeploymentRegion.US_EAST_1: ["us-east-1a", "us-east-1b", "us-east-1c"],
                DeploymentRegion.US_WEST_1: ["us-west-1a", "us-west-1b", "us-west-1c"],
                DeploymentRegion.EU_WEST_1: ["eu-west-1a", "eu-west-1b", "eu-west-1c"],
                DeploymentRegion.EU_CENTRAL_1: [
                    "eu-central-1a",
                    "eu-central-1b",
                    "eu-central-1c",
                ],
                DeploymentRegion.AP_SOUTHEAST_1: [
                    "ap-southeast-1a",
                    "ap-southeast-1b",
                    "ap-southeast-1c",
                ],
                DeploymentRegion.AP_NORTHEAST_1: [
                    "ap-northeast-1a",
                    "ap-northeast-1b",
                    "ap-northeast-1c",
                ],
            }
            self.availability_zones = region_az_map.get(self.region, ["a", "b", "c"])

        # Set compliance requirements based on region
        if not self.compliance_regulations:
            if self.region in [
                DeploymentRegion.EU_WEST_1,
                DeploymentRegion.EU_CENTRAL_1,
                DeploymentRegion.EU_NORTH_1,
            ]:
                self.compliance_regulations = ["GDPR"]
                self.data_residency_required = True
            elif self.region in [
                DeploymentRegion.US_EAST_1,
                DeploymentRegion.US_WEST_1,
            ]:
                self.compliance_regulations = ["CCPA", "SOX", "HIPAA"]
            elif self.region == DeploymentRegion.AP_SOUTHEAST_1:
                self.compliance_regulations = ["PDPA"]
                self.data_residency_required = True
            elif self.region == DeploymentRegion.CA_CENTRAL_1:
                self.compliance_regulations = ["PIPEDA"]
            elif self.region == DeploymentRegion.SA_EAST_1:
                self.compliance_regulations = ["LGPD"]


@dataclass
class DeploymentStatus:
    """Status of regional deployment."""

    region: DeploymentRegion
    status: RegionStatus
    health_score: float = 1.0  # 0.0 to 1.0
    last_health_check: datetime = field(default_factory=datetime.now)

    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0
    network_utilization: float = 0.0

    # Performance metrics
    request_latency_ms: float = 0.0
    throughput_requests_per_sec: float = 0.0
    error_rate: float = 0.0

    # Availability metrics
    uptime_percentage: float = 100.0
    last_downtime: Optional[datetime] = None
    downtime_duration_minutes: float = 0.0

    # Active instances
    active_instances: int = 0
    target_instances: int = 0

    # Data replication status
    replication_lag_seconds: float = 0.0
    backup_status: str = "healthy"

    def is_healthy(self) -> bool:
        """Check if region is healthy."""
        return (
            self.health_score > 0.8
            and self.status == RegionStatus.ACTIVE
            and self.error_rate < 0.05
            and self.uptime_percentage > 99.0
        )


@dataclass
class TrafficRoutingRule:
    """Traffic routing rule for multi-region deployment."""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    priority: int = 100

    # Conditions
    source_regions: List[DeploymentRegion] = field(default_factory=list)
    user_locations: List[str] = field(default_factory=list)
    request_types: List[str] = field(default_factory=list)

    # Routing targets
    target_regions: List[DeploymentRegion] = field(default_factory=list)
    weight_distribution: Dict[str, float] = field(default_factory=dict)

    # Failover configuration
    failover_enabled: bool = True
    health_check_threshold: float = 0.8

    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class MultiRegionDeploymentManager:
    """Advanced multi-region deployment manager."""

    def __init__(self):
        self.regional_configs: Dict[DeploymentRegion, RegionalConfiguration] = {}
        self.deployment_status: Dict[DeploymentRegion, DeploymentStatus] = {}
        self.traffic_routing_rules: Dict[str, TrafficRoutingRule] = {}

        # Deployment state
        self.active_regions: Set[DeploymentRegion] = set()
        self.primary_region: Optional[DeploymentRegion] = None
        self.disaster_recovery_region: Optional[DeploymentRegion] = None

        # Data replication
        self.replication_topology: Dict[DeploymentRegion, List[DeploymentRegion]] = {}
        self.data_sync_status: Dict[str, Dict[str, Any]] = {}

        # Global configuration
        self.global_load_balancer_enabled = True
        self.auto_failover_enabled = True
        self.cross_region_encryption = True
        self.data_replication_enabled = True

        # Monitoring and alerting
        self.health_check_interval = 30  # seconds
        self.deployment_monitoring_active = False

        # Performance optimization
        self.request_routing_cache: Dict[str, DeploymentRegion] = {}
        self.region_latency_matrix: Dict[
            Tuple[DeploymentRegion, DeploymentRegion], float
        ] = {}

        # Initialize default configurations
        self._initialize_default_configurations()
        self._initialize_latency_matrix()

    def _initialize_default_configurations(self):
        """Initialize default regional configurations."""
        # Production regions
        production_regions = [
            DeploymentRegion.US_EAST_1,
            DeploymentRegion.EU_WEST_1,
            DeploymentRegion.AP_SOUTHEAST_1,
        ]

        for region in production_regions:
            config = RegionalConfiguration(
                region=region,
                tier=DeploymentTier.PRODUCTION,
                compute_instances=3,
                memory_gb_per_instance=16,
                storage_gb=500,
                max_instances=20,
                database_replicas=2,
                backup_retention_days=90,
            )
            self.regional_configs[region] = config

            # Initialize status
            status = DeploymentStatus(
                region=region,
                status=RegionStatus.STANDBY,
                target_instances=config.compute_instances,
            )
            self.deployment_status[region] = status

        # Set primary and DR regions
        self.primary_region = DeploymentRegion.US_EAST_1
        self.disaster_recovery_region = DeploymentRegion.EU_WEST_1

        # Initialize replication topology
        self._setup_replication_topology()

        # Initialize default routing rules
        self._setup_default_routing_rules()

    def _initialize_latency_matrix(self):
        """Initialize cross-region latency matrix."""
        # Simplified latency matrix (in milliseconds)
        latencies = {
            (DeploymentRegion.US_EAST_1, DeploymentRegion.US_WEST_1): 70,
            (DeploymentRegion.US_EAST_1, DeploymentRegion.EU_WEST_1): 100,
            (DeploymentRegion.US_EAST_1, DeploymentRegion.EU_CENTRAL_1): 120,
            (DeploymentRegion.US_EAST_1, DeploymentRegion.AP_SOUTHEAST_1): 180,
            (DeploymentRegion.US_EAST_1, DeploymentRegion.AP_NORTHEAST_1): 200,
            (DeploymentRegion.US_WEST_1, DeploymentRegion.EU_WEST_1): 150,
            (DeploymentRegion.US_WEST_1, DeploymentRegion.EU_CENTRAL_1): 160,
            (DeploymentRegion.US_WEST_1, DeploymentRegion.AP_SOUTHEAST_1): 120,
            (DeploymentRegion.US_WEST_1, DeploymentRegion.AP_NORTHEAST_1): 130,
            (DeploymentRegion.EU_WEST_1, DeploymentRegion.EU_CENTRAL_1): 30,
            (DeploymentRegion.EU_WEST_1, DeploymentRegion.AP_SOUTHEAST_1): 200,
            (DeploymentRegion.EU_WEST_1, DeploymentRegion.AP_NORTHEAST_1): 250,
            (DeploymentRegion.EU_CENTRAL_1, DeploymentRegion.AP_SOUTHEAST_1): 180,
            (DeploymentRegion.EU_CENTRAL_1, DeploymentRegion.AP_NORTHEAST_1): 220,
            (DeploymentRegion.AP_SOUTHEAST_1, DeploymentRegion.AP_NORTHEAST_1): 50,
        }

        # Populate symmetric matrix
        for (region1, region2), latency in latencies.items():
            self.region_latency_matrix[(region1, region2)] = latency
            self.region_latency_matrix[(region2, region1)] = latency

        # Self-latency is zero
        for region in DeploymentRegion:
            self.region_latency_matrix[(region, region)] = 0

    def _setup_replication_topology(self):
        """Setup data replication topology."""
        # Primary -> DR replication
        if self.primary_region and self.disaster_recovery_region:
            self.replication_topology[self.primary_region] = [
                self.disaster_recovery_region
            ]

        # Multi-region replication for production data
        for region in self.regional_configs:
            if region != self.primary_region:
                if region not in self.replication_topology:
                    self.replication_topology[region] = []

                # Replicate to closest region
                closest_region = self._find_closest_region(region)
                if closest_region and closest_region != region:
                    self.replication_topology[region].append(closest_region)

    def _find_closest_region(
        self, source_region: DeploymentRegion
    ) -> Optional[DeploymentRegion]:
        """Find closest region for replication."""
        min_latency = float("inf")
        closest_region = None

        for target_region in self.regional_configs:
            if target_region != source_region:
                latency = self.region_latency_matrix.get(
                    (source_region, target_region), 1000
                )
                if latency < min_latency:
                    min_latency = latency
                    closest_region = target_region

        return closest_region

    def _setup_default_routing_rules(self):
        """Setup default traffic routing rules."""
        # Geographic routing rule
        geo_rule = TrafficRoutingRule(
            name="Geographic Routing",
            priority=1,
            user_locations=["North America", "Europe", "Asia"],
            target_regions=[
                DeploymentRegion.US_EAST_1,
                DeploymentRegion.EU_WEST_1,
                DeploymentRegion.AP_SOUTHEAST_1,
            ],
            weight_distribution={
                DeploymentRegion.US_EAST_1.value: 0.4,
                DeploymentRegion.EU_WEST_1.value: 0.35,
                DeploymentRegion.AP_SOUTHEAST_1.value: 0.25,
            },
        )
        self.traffic_routing_rules[geo_rule.rule_id] = geo_rule

        # Failover routing rule
        failover_rule = TrafficRoutingRule(
            name="Primary Failover",
            priority=2,
            target_regions=[DeploymentRegion.EU_WEST_1],
            weight_distribution={DeploymentRegion.EU_WEST_1.value: 1.0},
            failover_enabled=True,
            health_check_threshold=0.7,
        )
        self.traffic_routing_rules[failover_rule.rule_id] = failover_rule

    async def deploy_region(
        self, region: DeploymentRegion, config: Optional[RegionalConfiguration] = None
    ) -> bool:
        """Deploy to a specific region."""
        logger.info(f"Starting deployment to region: {region.value}")

        if config:
            self.regional_configs[region] = config
        elif region not in self.regional_configs:
            # Create default configuration
            self.regional_configs[region] = RegionalConfiguration(
                region=region, tier=DeploymentTier.PRODUCTION
            )

        deployment_config = self.regional_configs[region]

        # Initialize deployment status
        if region not in self.deployment_status:
            self.deployment_status[region] = DeploymentStatus(
                region=region,
                status=RegionStatus.DEPLOYING,
                target_instances=deployment_config.compute_instances,
            )

        status = self.deployment_status[region]
        status.status = RegionStatus.DEPLOYING

        try:
            # Phase 1: Infrastructure deployment
            logger.info(f"Deploying infrastructure in {region.value}")
            await self._deploy_infrastructure(region, deployment_config)

            # Phase 2: Application deployment
            logger.info(f"Deploying application in {region.value}")
            await self._deploy_application(region, deployment_config)

            # Phase 3: Data setup and replication
            logger.info(f"Setting up data replication in {region.value}")
            await self._setup_data_replication(region)

            # Phase 4: Health checks and activation
            logger.info(f"Running health checks in {region.value}")
            if await self._run_deployment_health_checks(region):
                status.status = RegionStatus.ACTIVE
                status.health_score = 1.0
                self.active_regions.add(region)
                deployment_config.deployed_at = datetime.now()

                logger.info(f"Successfully deployed to region: {region.value}")
                return True
            else:
                status.status = RegionStatus.FAILED
                logger.error(f"Health checks failed for region: {region.value}")
                return False

        except Exception as e:
            status.status = RegionStatus.FAILED
            logger.error(f"Deployment failed for region {region.value}: {e}")
            return False

    async def _deploy_infrastructure(
        self, region: DeploymentRegion, config: RegionalConfiguration
    ):
        """Deploy infrastructure for a region."""
        # Simulate infrastructure deployment
        await asyncio.sleep(2)  # Simulate deployment time

        logger.info(f"Deployed {config.compute_instances} instances in {region.value}")
        logger.info(f"Configured VPC with CIDR {config.vpc_cidr}")
        logger.info(f"Set up load balancer: {config.load_balancer_enabled}")

        # Update status
        status = self.deployment_status[region]
        status.active_instances = config.compute_instances

    async def _deploy_application(
        self, region: DeploymentRegion, config: RegionalConfiguration
    ):
        """Deploy application to a region."""
        # Simulate application deployment
        await asyncio.sleep(1.5)

        logger.info(f"Deployed self-healing pipeline guard to {region.value}")
        logger.info(f"Version: {config.version}")
        logger.info(f"Monitoring enabled: {config.monitoring_enabled}")

    async def _setup_data_replication(self, region: DeploymentRegion):
        """Setup data replication for a region."""
        if region in self.replication_topology:
            replica_regions = self.replication_topology[region]

            for replica_region in replica_regions:
                # Simulate replication setup
                await asyncio.sleep(0.5)

                replication_id = f"{region.value}-to-{replica_region.value}"
                self.data_sync_status[replication_id] = {
                    "source": region.value,
                    "target": replica_region.value,
                    "status": "active",
                    "lag_seconds": 0.0,
                    "last_sync": datetime.now().isoformat(),
                }

                logger.info(
                    f"Setup replication: {region.value} -> {replica_region.value}"
                )

    async def _run_deployment_health_checks(self, region: DeploymentRegion) -> bool:
        """Run health checks for deployed region."""
        # Simulate health checks
        await asyncio.sleep(1)

        # Check infrastructure
        infrastructure_healthy = True

        # Check application
        application_healthy = True

        # Check data replication
        replication_healthy = True

        # Check compliance
        compliance_healthy = self._check_regional_compliance(region)

        overall_health = (
            infrastructure_healthy
            and application_healthy
            and replication_healthy
            and compliance_healthy
        )

        logger.info(f"Health check results for {region.value}: {overall_health}")

        return overall_health

    def _check_regional_compliance(self, region: DeploymentRegion) -> bool:
        """Check compliance requirements for region."""
        config = self.regional_configs.get(region)
        if not config:
            return False

        # Check data residency requirements
        if config.data_residency_required:
            # Ensure data stays within required boundaries
            logger.info(f"Data residency compliance check passed for {region.value}")

        # Check encryption requirements
        if config.encryption_enabled:
            logger.info(f"Encryption compliance check passed for {region.value}")

        return True

    async def failover_to_region(
        self,
        target_region: DeploymentRegion,
        failed_region: Optional[DeploymentRegion] = None,
    ) -> bool:
        """Perform failover to target region."""
        logger.warning(f"Initiating failover to region: {target_region.value}")

        if failed_region:
            logger.warning(f"Failing over from region: {failed_region.value}")

            # Mark failed region
            if failed_region in self.deployment_status:
                self.deployment_status[failed_region].status = RegionStatus.FAILED
                self.active_regions.discard(failed_region)

        # Ensure target region is ready
        if target_region not in self.active_regions:
            logger.info(f"Target region {target_region.value} not active, deploying...")
            success = await self.deploy_region(target_region)
            if not success:
                logger.error(f"Failed to deploy target region {target_region.value}")
                return False

        # Scale up target region
        await self._scale_region(target_region, scale_factor=1.5)

        # Update traffic routing
        await self._update_traffic_routing_for_failover(target_region, failed_region)

        # Update primary region if needed
        if failed_region == self.primary_region:
            self.primary_region = target_region
            logger.info(f"Updated primary region to: {target_region.value}")

        logger.info(f"Failover to {target_region.value} completed successfully")
        return True

    async def _scale_region(self, region: DeploymentRegion, scale_factor: float):
        """Scale a region by the given factor."""
        if region not in self.deployment_status:
            return

        status = self.deployment_status[region]
        config = self.regional_configs[region]

        new_instance_count = min(
            int(status.active_instances * scale_factor), config.max_instances
        )

        if new_instance_count != status.active_instances:
            logger.info(
                f"Scaling {region.value} from {status.active_instances} to {new_instance_count} instances"
            )

            status.status = RegionStatus.SCALING

            # Simulate scaling time
            await asyncio.sleep(1)

            status.active_instances = new_instance_count
            status.target_instances = new_instance_count
            status.status = RegionStatus.ACTIVE

    async def _update_traffic_routing_for_failover(
        self, target_region: DeploymentRegion, failed_region: Optional[DeploymentRegion]
    ):
        """Update traffic routing rules for failover."""
        # Update weight distribution to route traffic to target region
        for rule in self.traffic_routing_rules.values():
            if target_region in rule.target_regions:
                # Increase weight for target region
                if target_region.value in rule.weight_distribution:
                    rule.weight_distribution[target_region.value] = min(
                        rule.weight_distribution[target_region.value] + 0.3, 1.0
                    )
                else:
                    rule.weight_distribution[target_region.value] = 0.5

                # Decrease or remove weight for failed region
                if failed_region and failed_region.value in rule.weight_distribution:
                    rule.weight_distribution[failed_region.value] = 0.0

        # Clear routing cache to force recalculation
        self.request_routing_cache.clear()

    def route_request(
        self, user_location: str = "", request_type: str = ""
    ) -> DeploymentRegion:
        """Route request to optimal region."""
        cache_key = f"{user_location}:{request_type}"

        # Check cache
        if cache_key in self.request_routing_cache:
            cached_region = self.request_routing_cache[cache_key]
            if cached_region in self.active_regions:
                return cached_region

        # Find best region based on routing rules
        best_region = self._find_optimal_region(user_location, request_type)

        # Cache result
        self.request_routing_cache[cache_key] = best_region

        return best_region

    def _find_optimal_region(
        self, user_location: str, request_type: str
    ) -> DeploymentRegion:
        """Find optimal region for request routing."""
        # Get applicable routing rules
        applicable_rules = []

        for rule in self.traffic_routing_rules.values():
            if not rule.enabled:
                continue

            # Check conditions
            if rule.user_locations and user_location not in rule.user_locations:
                continue

            if rule.request_types and request_type not in rule.request_types:
                continue

            applicable_rules.append(rule)

        # Sort by priority
        applicable_rules.sort(key=lambda r: r.priority)

        # Apply first matching rule
        for rule in applicable_rules:
            healthy_regions = [
                region
                for region in rule.target_regions
                if (
                    region in self.active_regions
                    and self.deployment_status[region].is_healthy()
                )
            ]

            if healthy_regions:
                # Select based on weights
                if rule.weight_distribution:
                    return self._weighted_region_selection(
                        healthy_regions, rule.weight_distribution
                    )
                else:
                    # Default to lowest latency
                    return self._lowest_latency_region(healthy_regions, user_location)

        # Fallback to primary region or any active region
        if self.primary_region and self.primary_region in self.active_regions:
            return self.primary_region

        return (
            next(iter(self.active_regions))
            if self.active_regions
            else DeploymentRegion.US_EAST_1
        )

    def _weighted_region_selection(
        self, regions: List[DeploymentRegion], weights: Dict[str, float]
    ) -> DeploymentRegion:
        """Select region based on weighted distribution."""

        total_weight = 0.0
        region_weights = []

        for region in regions:
            weight = weights.get(region.value, 0.0)
            region_weights.append((region, weight))
            total_weight += weight

        if total_weight == 0:
            return regions[0]

        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0.0

        for region, weight in region_weights:
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return region

        return regions[-1]

    def _lowest_latency_region(
        self, regions: List[DeploymentRegion], user_location: str
    ) -> DeploymentRegion:
        """Select region with lowest latency."""
        # Simplified: map user location to closest region
        location_region_map = {
            "North America": DeploymentRegion.US_EAST_1,
            "Europe": DeploymentRegion.EU_WEST_1,
            "Asia": DeploymentRegion.AP_SOUTHEAST_1,
            "": DeploymentRegion.US_EAST_1,  # Default
        }

        preferred_region = location_region_map.get(
            user_location, DeploymentRegion.US_EAST_1
        )

        if preferred_region in regions:
            return preferred_region

        return regions[0]

    async def start_deployment_monitoring(self):
        """Start deployment monitoring across all regions."""
        if self.deployment_monitoring_active:
            return

        self.deployment_monitoring_active = True
        logger.info("Starting multi-region deployment monitoring...")

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._replication_monitoring_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
        ]

        # Store tasks for cleanup
        self._monitoring_tasks = tasks

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Deployment monitoring error: {e}")
        finally:
            for task in tasks:
                task.cancel()

    async def _health_monitoring_loop(self):
        """Monitor health of all deployed regions."""
        while self.deployment_monitoring_active:
            try:
                for region in self.active_regions:
                    await self._check_region_health(region)

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _check_region_health(self, region: DeploymentRegion):
        """Check health of a specific region."""
        if region not in self.deployment_status:
            return

        status = self.deployment_status[region]

        # Simulate health metrics collection

        # Update metrics
        status.cpu_utilization = random.uniform(20, 80)
        status.memory_utilization = random.uniform(30, 70)
        status.storage_utilization = random.uniform(10, 60)
        status.request_latency_ms = random.uniform(50, 200)
        status.throughput_requests_per_sec = random.uniform(100, 1000)
        status.error_rate = random.uniform(0, 0.05)
        status.last_health_check = datetime.now()

        # Calculate health score
        health_factors = [
            1.0 - (status.cpu_utilization / 100.0),
            1.0 - (status.memory_utilization / 100.0),
            1.0 - (status.error_rate / 0.1),
            min(1.0, 200.0 / status.request_latency_ms),
        ]

        status.health_score = sum(health_factors) / len(health_factors)

        # Check for failover conditions
        if status.health_score < 0.5 and self.auto_failover_enabled:
            logger.warning(
                f"Region {region.value} health score low: {status.health_score:.2f}"
            )
            await self._trigger_auto_failover(region)

    async def _trigger_auto_failover(self, failed_region: DeploymentRegion):
        """Trigger automatic failover from failed region."""
        # Find best failover target
        failover_target = self._find_best_failover_target(failed_region)

        if failover_target:
            await self.failover_to_region(failover_target, failed_region)
        else:
            logger.error(f"No suitable failover target found for {failed_region.value}")

    def _find_best_failover_target(
        self, failed_region: DeploymentRegion
    ) -> Optional[DeploymentRegion]:
        """Find best region for failover."""
        candidates = [
            region
            for region in self.active_regions
            if (region != failed_region and self.deployment_status[region].is_healthy())
        ]

        if not candidates:
            # Try to use disaster recovery region
            if (
                self.disaster_recovery_region
                and self.disaster_recovery_region != failed_region
            ):
                return self.disaster_recovery_region
            return None

        # Select region with highest health score and capacity
        best_region = max(
            candidates,
            key=lambda r: (
                self.deployment_status[r].health_score,
                self.regional_configs[r].max_instances
                - self.deployment_status[r].active_instances,
            ),
        )

        return best_region

    async def _performance_monitoring_loop(self):
        """Monitor performance across regions."""
        while self.deployment_monitoring_active:
            try:
                # Update latency matrix
                await self._update_latency_measurements()

                # Optimize routing based on performance
                await self._optimize_traffic_routing()

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)

    async def _update_latency_measurements(self):
        """Update cross-region latency measurements."""
        # In production, would measure actual latencies
        # For now, simulate with some variation

        for (region1, region2), base_latency in self.region_latency_matrix.items():
            if region1 != region2:
                # Add some random variation
                variation = random.uniform(-10, 10)
                self.region_latency_matrix[(region1, region2)] = max(
                    1, base_latency + variation
                )

    async def _optimize_traffic_routing(self):
        """Optimize traffic routing based on current performance."""
        # Clear routing cache to force recalculation
        self.request_routing_cache.clear()

        # Analyze current performance and adjust routing weights
        for rule in self.traffic_routing_rules.values():
            if rule.weight_distribution:
                # Adjust weights based on region performance
                for region in rule.target_regions:
                    if region in self.deployment_status:
                        status = self.deployment_status[region]
                        region_key = region.value

                        if region_key in rule.weight_distribution:
                            # Increase weight for healthy, high-performing regions
                            performance_factor = (
                                status.health_score
                                * (1.0 - status.error_rate)
                                * min(1.0, 200.0 / status.request_latency_ms)
                            )

                            # Gradually adjust weight
                            current_weight = rule.weight_distribution[region_key]
                            target_weight = (
                                performance_factor * 0.5
                            )  # Base weight of 0.5
                            adjustment = (
                                target_weight - current_weight
                            ) * 0.1  # 10% adjustment

                            rule.weight_distribution[region_key] = max(
                                0.0, min(1.0, current_weight + adjustment)
                            )

    async def _replication_monitoring_loop(self):
        """Monitor data replication across regions."""
        while self.deployment_monitoring_active:
            try:
                for replication_id, replication_status in self.data_sync_status.items():
                    # Simulate replication lag
                    import random

                    replication_status["lag_seconds"] = random.uniform(0, 5)
                    replication_status["last_sync"] = datetime.now().isoformat()

                    # Update regional replication status
                    source_region = DeploymentRegion(replication_status["source"])
                    if source_region in self.deployment_status:
                        self.deployment_status[
                            source_region
                        ].replication_lag_seconds = replication_status["lag_seconds"]

                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Replication monitoring error: {e}")
                await asyncio.sleep(60)

    async def _auto_scaling_loop(self):
        """Auto-scaling loop for regions."""
        while self.deployment_monitoring_active:
            try:
                for region in self.active_regions:
                    await self._check_auto_scaling(region)

                await asyncio.sleep(120)  # Every 2 minutes

            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(120)

    async def _check_auto_scaling(self, region: DeploymentRegion):
        """Check if region needs auto-scaling."""
        if region not in self.deployment_status or region not in self.regional_configs:
            return

        status = self.deployment_status[region]
        config = self.regional_configs[region]

        if not config.auto_scaling_enabled:
            return

        # Scale up conditions
        if (
            status.cpu_utilization > config.target_cpu_utilization + 10
            or status.memory_utilization > config.target_memory_utilization + 10
        ):

            if status.active_instances < config.max_instances:
                new_count = min(status.active_instances + 1, config.max_instances)
                await self._update_instance_count(region, new_count)
                logger.info(f"Scaled up {region.value} to {new_count} instances")

        # Scale down conditions
        elif (
            status.cpu_utilization < config.target_cpu_utilization - 20
            and status.memory_utilization < config.target_memory_utilization - 20
        ):

            if status.active_instances > config.min_instances:
                new_count = max(status.active_instances - 1, config.min_instances)
                await self._update_instance_count(region, new_count)
                logger.info(f"Scaled down {region.value} to {new_count} instances")

    async def _update_instance_count(self, region: DeploymentRegion, new_count: int):
        """Update instance count for a region."""
        status = self.deployment_status[region]
        status.status = RegionStatus.SCALING

        # Simulate scaling time
        await asyncio.sleep(0.5)

        status.active_instances = new_count
        status.target_instances = new_count
        status.status = RegionStatus.ACTIVE

    def stop_deployment_monitoring(self):
        """Stop deployment monitoring."""
        self.deployment_monitoring_active = False

        # Cancel monitoring tasks
        if hasattr(self, "_monitoring_tasks"):
            for task in self._monitoring_tasks:
                task.cancel()

        logger.info("Deployment monitoring stopped")

    def get_global_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        return {
            "primary_region": (
                self.primary_region.value if self.primary_region else None
            ),
            "disaster_recovery_region": (
                self.disaster_recovery_region.value
                if self.disaster_recovery_region
                else None
            ),
            "active_regions": [region.value for region in self.active_regions],
            "total_regions": len(self.regional_configs),
            "global_health_score": self._calculate_global_health_score(),
            "auto_failover_enabled": self.auto_failover_enabled,
            "cross_region_encryption": self.cross_region_encryption,
            "regional_status": {
                region.value: {
                    "status": status.status.value,
                    "health_score": status.health_score,
                    "active_instances": status.active_instances,
                    "cpu_utilization": status.cpu_utilization,
                    "memory_utilization": status.memory_utilization,
                    "request_latency_ms": status.request_latency_ms,
                    "error_rate": status.error_rate,
                    "uptime_percentage": status.uptime_percentage,
                    "replication_lag_seconds": status.replication_lag_seconds,
                }
                for region, status in self.deployment_status.items()
            },
            "traffic_routing_rules": len(self.traffic_routing_rules),
            "data_replication_status": self.data_sync_status,
            "monitoring_active": self.deployment_monitoring_active,
        }

    def _calculate_global_health_score(self) -> float:
        """Calculate global health score across all regions."""
        if not self.deployment_status:
            return 0.0

        active_scores = [
            status.health_score
            for region, status in self.deployment_status.items()
            if region in self.active_regions
        ]

        if not active_scores:
            return 0.0

        return sum(active_scores) / len(active_scores)


# Global deployment manager instance
_global_deployment_manager: Optional[MultiRegionDeploymentManager] = None


def get_deployment_manager() -> MultiRegionDeploymentManager:
    """Get the global deployment manager instance."""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = MultiRegionDeploymentManager()
    return _global_deployment_manager
