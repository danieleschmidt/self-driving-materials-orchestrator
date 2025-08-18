"""Distributed Self-Healing System for Global Materials Discovery Networks.

Implements distributed self-healing capabilities across multiple regions with
advanced coordination, consensus mechanisms, and global optimization.
"""

import asyncio
import logging
import random
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Distributed node status."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    SYNCHRONIZING = "synchronizing"
    LEADER = "leader"
    FOLLOWER = "follower"


class ConsensusAlgorithm(Enum):
    """Consensus algorithms for distributed coordination."""

    RAFT = "raft"
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    TENDERMINT = "tendermint"
    QUANTUM_CONSENSUS = "quantum_consensus"


class GlobalRegion(Enum):
    """Global regions for distributed deployment."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    ASIA_NORTHEAST = "asia-northeast"


@dataclass
class DistributedNode:
    """Distributed self-healing node."""

    node_id: str
    region: GlobalRegion
    endpoint: str
    status: NodeStatus = NodeStatus.ONLINE
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    failure_count: int = 0
    recovery_count: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        heartbeat_age = (datetime.now() - self.last_heartbeat).total_seconds()
        return (
            self.status in [NodeStatus.ONLINE, NodeStatus.LEADER, NodeStatus.FOLLOWER]
            and heartbeat_age < 60
        )  # 60 second heartbeat timeout

    @property
    def reliability_score(self) -> float:
        """Calculate node reliability score."""
        total_operations = self.failure_count + self.recovery_count
        if total_operations == 0:
            return 1.0
        return self.recovery_count / total_operations


@dataclass
class DistributedFailure:
    """Distributed failure event."""

    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    affected_nodes: List[str] = field(default_factory=list)
    failure_type: str = "unknown"
    severity: str = "medium"
    description: str = ""
    global_impact: bool = False
    consensus_required: bool = False
    resolved: bool = False
    resolution_strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusMessage:
    """Consensus protocol message."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    message_type: str = ""  # propose, vote, commit, heartbeat
    term: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str = ""


@dataclass
class GlobalState:
    """Global distributed system state."""

    term: int = 0
    leader_id: Optional[str] = None
    nodes: Dict[str, DistributedNode] = field(default_factory=dict)
    active_failures: Dict[str, DistributedFailure] = field(default_factory=dict)
    global_configuration: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    consensus_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "term": self.term,
            "leader_id": self.leader_id,
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "active_failures": {
                f_id: asdict(failure) for f_id, failure in self.active_failures.items()
            },
            "global_configuration": self.global_configuration,
            "last_updated": self.last_updated.isoformat(),
            "consensus_log_size": len(self.consensus_log),
        }


class DistributedConsensusEngine:
    """Distributed consensus engine for coordinating self-healing actions."""

    def __init__(
        self, node_id: str, algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT
    ):
        self.node_id = node_id
        self.algorithm = algorithm

        # Consensus state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[Dict[str, Any]] = []
        self.commit_index = 0
        self.last_applied = 0

        # Leader state
        self.is_leader = False
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # Timing
        self.election_timeout = random.uniform(5, 10)  # seconds
        self.heartbeat_interval = 2.0  # seconds
        self.last_heartbeat = datetime.now()

        # Message queues
        self.incoming_messages: asyncio.Queue = asyncio.Queue()
        self.outgoing_messages: asyncio.Queue = asyncio.Queue()

        # Consensus statistics
        self.elections_participated = 0
        self.votes_cast = 0
        self.messages_sent = 0
        self.messages_received = 0

    async def start_consensus(self, peers: List[str]):
        """Start consensus protocol."""
        self.peers = peers
        logger.info(f"Starting consensus engine for node {self.node_id}")

        # Start consensus tasks
        tasks = [
            asyncio.create_task(self._election_timeout_handler()),
            asyncio.create_task(self._heartbeat_sender()),
            asyncio.create_task(self._message_processor()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Consensus error: {e}")
        finally:
            for task in tasks:
                task.cancel()

    async def _election_timeout_handler(self):
        """Handle election timeouts."""
        while True:
            try:
                await asyncio.sleep(self.election_timeout)

                # Check if we've received heartbeat recently
                time_since_heartbeat = (
                    datetime.now() - self.last_heartbeat
                ).total_seconds()

                if time_since_heartbeat > self.election_timeout and not self.is_leader:
                    await self._start_election()

            except Exception as e:
                logger.error(f"Election timeout error: {e}")

    async def _start_election(self):
        """Start leader election."""
        logger.info(
            f"Node {self.node_id} starting election for term {self.current_term + 1}"
        )

        # Increment term and vote for self
        self.current_term += 1
        self.voted_for = self.node_id
        self.elections_participated += 1

        # Reset election timeout
        self.election_timeout = random.uniform(5, 10)

        # Send vote requests to all peers
        vote_request = ConsensusMessage(
            sender_id=self.node_id,
            message_type="vote_request",
            term=self.current_term,
            payload={
                "last_log_index": len(self.log) - 1,
                "last_log_term": self.log[-1].get("term", 0) if self.log else 0,
            },
        )

        votes_received = 1  # Vote for self
        majority = len(self.peers) // 2 + 1

        # Send vote requests (simplified - in production would use actual network)
        for peer in self.peers:
            if peer != self.node_id:
                await self.outgoing_messages.put(vote_request)
                self.messages_sent += 1

        # Simulate vote collection (in production would wait for actual responses)
        await asyncio.sleep(1)

        # Simulate receiving votes (random for demo)
        for peer in self.peers:
            if peer != self.node_id and random.random() > 0.3:  # 70% chance of vote
                votes_received += 1

        # Check if won election
        if votes_received >= majority:
            await self._become_leader()
        else:
            logger.info(
                f"Node {self.node_id} lost election for term {self.current_term}"
            )

    async def _become_leader(self):
        """Become cluster leader."""
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")

        self.is_leader = True
        self.last_heartbeat = datetime.now()

        # Initialize leader state
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = 0

    async def _heartbeat_sender(self):
        """Send heartbeat messages as leader."""
        while True:
            try:
                if self.is_leader:
                    heartbeat = ConsensusMessage(
                        sender_id=self.node_id,
                        message_type="heartbeat",
                        term=self.current_term,
                        payload={"commit_index": self.commit_index},
                    )

                    # Send to all peers
                    for peer in self.peers:
                        if peer != self.node_id:
                            await self.outgoing_messages.put(heartbeat)
                            self.messages_sent += 1

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _message_processor(self):
        """Process incoming consensus messages."""
        while True:
            try:
                message = await self.incoming_messages.get()
                self.messages_received += 1

                if message.message_type == "vote_request":
                    await self._handle_vote_request(message)
                elif message.message_type == "vote_response":
                    await self._handle_vote_response(message)
                elif message.message_type == "heartbeat":
                    await self._handle_heartbeat(message)
                elif message.message_type == "append_entries":
                    await self._handle_append_entries(message)

            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _handle_vote_request(self, message: ConsensusMessage):
        """Handle vote request message."""
        grant_vote = False

        # Grant vote if:
        # 1. Haven't voted in this term OR voted for this candidate
        # 2. Candidate's log is at least as up-to-date as ours
        if message.term > self.current_term and (
            self.voted_for is None or self.voted_for == message.sender_id
        ):

            grant_vote = True
            self.voted_for = message.sender_id
            self.current_term = message.term
            self.votes_cast += 1

        # Send vote response
        response = ConsensusMessage(
            sender_id=self.node_id,
            message_type="vote_response",
            term=self.current_term,
            payload={"vote_granted": grant_vote},
        )

        await self.outgoing_messages.put(response)
        self.messages_sent += 1

    async def _handle_vote_response(self, message: ConsensusMessage):
        """Handle vote response message."""
        # Implementation would count votes and become leader if majority
        pass

    async def _handle_heartbeat(self, message: ConsensusMessage):
        """Handle heartbeat message."""
        if message.term >= self.current_term:
            self.current_term = message.term
            self.is_leader = False
            self.last_heartbeat = datetime.now()

            # Update commit index
            if message.payload.get("commit_index", 0) > self.commit_index:
                self.commit_index = message.payload["commit_index"]

    async def _handle_append_entries(self, message: ConsensusMessage):
        """Handle append entries message."""
        # Implementation would handle log replication
        pass

    async def propose_action(self, action: Dict[str, Any]) -> bool:
        """Propose a distributed action for consensus."""
        if not self.is_leader:
            return False

        # Add to log
        log_entry = {
            "term": self.current_term,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        }

        self.log.append(log_entry)

        # Replicate to followers (simplified)
        return True

    def get_consensus_status(self) -> Dict[str, Any]:
        """Get consensus engine status."""
        return {
            "node_id": self.node_id,
            "algorithm": self.algorithm.value,
            "current_term": self.current_term,
            "is_leader": self.is_leader,
            "voted_for": self.voted_for,
            "log_size": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "elections_participated": self.elections_participated,
            "votes_cast": self.votes_cast,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


class GlobalCoordinationLayer:
    """Global coordination layer for distributed self-healing."""

    def __init__(self, region: GlobalRegion, node_id: str = None):
        self.region = region
        self.node_id = node_id or f"{region.value}-{uuid.uuid4().hex[:8]}"

        # Distributed state
        self.global_state = GlobalState()
        self.local_node = DistributedNode(
            node_id=self.node_id,
            region=region,
            endpoint=f"https://{region.value}.materials-discovery.com",
        )

        # Consensus engine
        self.consensus_engine = DistributedConsensusEngine(self.node_id)

        # Peer discovery and management
        self.peer_nodes: Dict[str, DistributedNode] = {}
        self.region_leaders: Dict[GlobalRegion, str] = {}

        # Cross-region communication
        self.cross_region_latencies: Dict[GlobalRegion, float] = {}
        self.data_replication_factor = 3

        # Global failure tracking
        self.global_failures: Dict[str, DistributedFailure] = {}
        self.failure_correlation_graph = defaultdict(list)

        # Performance optimization
        self.request_router = GlobalRequestRouter()
        self.load_balancer = GlobalLoadBalancer()

        # Monitoring
        self.metrics_collector = GlobalMetricsCollector()
        self.coordination_enabled = False

    async def join_global_network(self, bootstrap_nodes: List[str]):
        """Join the global distributed network."""
        logger.info(f"Node {self.node_id} joining global network...")

        # Register with global state
        self.global_state.nodes[self.node_id] = self.local_node

        # Start peer discovery
        await self._discover_peers(bootstrap_nodes)

        # Start consensus engine
        peer_ids = list(self.peer_nodes.keys())
        consensus_task = asyncio.create_task(
            self.consensus_engine.start_consensus(peer_ids)
        )

        # Start coordination services
        self.coordination_enabled = True
        coordination_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._failure_detection_loop()),
            asyncio.create_task(self._global_state_sync_loop()),
            asyncio.create_task(self._cross_region_coordination_loop()),
        ]

        # Store tasks for cleanup
        self._coordination_tasks = [consensus_task] + coordination_tasks

        logger.info(f"Node {self.node_id} successfully joined global network")

    async def _discover_peers(self, bootstrap_nodes: List[str]):
        """Discover peer nodes in the network."""
        # Simulate peer discovery
        regions = list(GlobalRegion)

        for region in regions:
            if region != self.region:
                peer_id = f"{region.value}-{uuid.uuid4().hex[:8]}"
                peer_node = DistributedNode(
                    node_id=peer_id,
                    region=region,
                    endpoint=f"https://{region.value}.materials-discovery.com",
                    latency_ms=self._calculate_cross_region_latency(region),
                )
                self.peer_nodes[peer_id] = peer_node
                self.cross_region_latencies[region] = peer_node.latency_ms

        logger.info(f"Discovered {len(self.peer_nodes)} peer nodes")

    def _calculate_cross_region_latency(self, target_region: GlobalRegion) -> float:
        """Calculate estimated latency to target region."""
        # Simplified latency model based on geographical distance
        latency_matrix = {
            (GlobalRegion.US_EAST, GlobalRegion.US_WEST): 70,
            (GlobalRegion.US_EAST, GlobalRegion.EU_WEST): 100,
            (GlobalRegion.US_EAST, GlobalRegion.EU_CENTRAL): 120,
            (GlobalRegion.US_EAST, GlobalRegion.ASIA_PACIFIC): 180,
            (GlobalRegion.US_EAST, GlobalRegion.ASIA_NORTHEAST): 200,
            (GlobalRegion.US_WEST, GlobalRegion.EU_WEST): 150,
            (GlobalRegion.US_WEST, GlobalRegion.EU_CENTRAL): 160,
            (GlobalRegion.US_WEST, GlobalRegion.ASIA_PACIFIC): 120,
            (GlobalRegion.US_WEST, GlobalRegion.ASIA_NORTHEAST): 130,
            (GlobalRegion.EU_WEST, GlobalRegion.EU_CENTRAL): 30,
            (GlobalRegion.EU_WEST, GlobalRegion.ASIA_PACIFIC): 200,
            (GlobalRegion.EU_WEST, GlobalRegion.ASIA_NORTHEAST): 250,
            (GlobalRegion.EU_CENTRAL, GlobalRegion.ASIA_PACIFIC): 180,
            (GlobalRegion.EU_CENTRAL, GlobalRegion.ASIA_NORTHEAST): 220,
            (GlobalRegion.ASIA_PACIFIC, GlobalRegion.ASIA_NORTHEAST): 50,
        }

        key = (self.region, target_region)
        reverse_key = (target_region, self.region)

        return latency_matrix.get(key, latency_matrix.get(reverse_key, 100))

    async def _heartbeat_loop(self):
        """Send heartbeat to maintain node liveness."""
        while self.coordination_enabled:
            try:
                # Update local node heartbeat
                self.local_node.last_heartbeat = datetime.now()

                # Send heartbeat to peers (simplified)
                heartbeat_data = {
                    "node_id": self.node_id,
                    "region": self.region.value,
                    "status": self.local_node.status.value,
                    "load": self.local_node.load,
                    "timestamp": datetime.now().isoformat(),
                }

                # In production, would send actual network requests
                logger.debug(f"Sending heartbeat from {self.node_id}")

                await asyncio.sleep(10)  # 10 second heartbeat interval

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)

    async def _failure_detection_loop(self):
        """Detect failures across the distributed system."""
        while self.coordination_enabled:
            try:
                # Check peer node health
                failed_nodes = []

                for node_id, node in self.peer_nodes.items():
                    if not node.is_healthy:
                        failed_nodes.append(node_id)

                        # Create distributed failure record
                        failure = DistributedFailure(
                            affected_nodes=[node_id],
                            failure_type="node_failure",
                            severity="high",
                            description=f"Node {node_id} in region {node.region.value} is unresponsive",
                            global_impact=self._assess_global_impact([node_id]),
                            consensus_required=True,
                        )

                        await self._handle_distributed_failure(failure)

                if failed_nodes:
                    logger.warning(
                        f"Detected {len(failed_nodes)} failed nodes: {failed_nodes}"
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Failure detection error: {e}")
                await asyncio.sleep(30)

    async def _global_state_sync_loop(self):
        """Synchronize global state across nodes."""
        while self.coordination_enabled:
            try:
                # Update global state
                self.global_state.last_updated = datetime.now()
                self.global_state.nodes[self.node_id] = self.local_node

                # Sync with leader if not leader
                if not self.consensus_engine.is_leader:
                    await self._sync_with_leader()
                else:
                    # As leader, broadcast state updates
                    await self._broadcast_state_updates()

                await asyncio.sleep(60)  # Sync every minute

            except Exception as e:
                logger.error(f"State sync error: {e}")
                await asyncio.sleep(60)

    async def _cross_region_coordination_loop(self):
        """Coordinate actions across regions."""
        while self.coordination_enabled:
            try:
                # Check for cross-region coordination needs
                await self._coordinate_cross_region_healing()
                await self._optimize_cross_region_routing()
                await self._balance_global_load()

                await asyncio.sleep(120)  # Coordinate every 2 minutes

            except Exception as e:
                logger.error(f"Cross-region coordination error: {e}")
                await asyncio.sleep(120)

    async def _handle_distributed_failure(self, failure: DistributedFailure):
        """Handle a distributed failure event."""
        logger.info(f"Handling distributed failure: {failure.failure_id}")

        # Add to global failures
        self.global_failures[failure.failure_id] = failure

        # Check if consensus is required
        if failure.consensus_required:
            # Propose healing action through consensus
            healing_action = {
                "type": "distributed_healing",
                "failure_id": failure.failure_id,
                "affected_nodes": failure.affected_nodes,
                "strategy": await self._determine_healing_strategy(failure),
            }

            success = await self.consensus_engine.propose_action(healing_action)
            if success:
                await self._execute_distributed_healing(healing_action)
        else:
            # Handle locally
            await self._execute_local_healing(failure)

    async def _determine_healing_strategy(self, failure: DistributedFailure) -> str:
        """Determine optimal healing strategy for distributed failure."""
        if failure.global_impact:
            return "global_failover"
        elif len(failure.affected_nodes) > 1:
            return "multi_node_recovery"
        else:
            return "single_node_recovery"

    async def _execute_distributed_healing(self, healing_action: Dict[str, Any]):
        """Execute distributed healing action."""
        strategy = healing_action["strategy"]

        if strategy == "global_failover":
            await self._execute_global_failover(healing_action)
        elif strategy == "multi_node_recovery":
            await self._execute_multi_node_recovery(healing_action)
        elif strategy == "single_node_recovery":
            await self._execute_single_node_recovery(healing_action)

        logger.info(f"Executed distributed healing: {strategy}")

    async def _execute_global_failover(self, healing_action: Dict[str, Any]):
        """Execute global failover across regions."""
        affected_regions = set()

        # Determine affected regions
        for node_id in healing_action["affected_nodes"]:
            if node_id in self.peer_nodes:
                affected_regions.add(self.peer_nodes[node_id].region)

        # Redirect traffic away from affected regions
        for region in affected_regions:
            await self.request_router.redirect_region_traffic(region)

        # Scale up healthy regions
        healthy_regions = set(GlobalRegion) - affected_regions
        for region in healthy_regions:
            await self.load_balancer.scale_region(region, scale_factor=1.5)

    async def _execute_multi_node_recovery(self, healing_action: Dict[str, Any]):
        """Execute multi-node recovery."""
        # Coordinate recovery across multiple nodes
        for node_id in healing_action["affected_nodes"]:
            if node_id in self.peer_nodes:
                await self._recover_node(node_id)

    async def _execute_single_node_recovery(self, healing_action: Dict[str, Any]):
        """Execute single node recovery."""
        node_id = healing_action["affected_nodes"][0]
        await self._recover_node(node_id)

    async def _recover_node(self, node_id: str):
        """Recover a specific node."""
        logger.info(f"Recovering node: {node_id}")

        if node_id in self.peer_nodes:
            node = self.peer_nodes[node_id]

            # Simulate recovery process
            await asyncio.sleep(2)

            # Update node status
            node.status = NodeStatus.ONLINE
            node.last_heartbeat = datetime.now()
            node.recovery_count += 1

            logger.info(f"Node {node_id} recovery completed")

    def _assess_global_impact(self, affected_nodes: List[str]) -> bool:
        """Assess if failure has global impact."""
        # Check if failure affects multiple regions
        affected_regions = set()

        for node_id in affected_nodes:
            if node_id in self.peer_nodes:
                affected_regions.add(self.peer_nodes[node_id].region)

        # Global impact if affects multiple regions or critical nodes
        return len(affected_regions) > 1 or len(affected_nodes) > 2

    async def _sync_with_leader(self):
        """Sync state with cluster leader."""
        # In production, would send actual sync request to leader
        pass

    async def _broadcast_state_updates(self):
        """Broadcast state updates as leader."""
        # In production, would send state updates to all followers
        pass

    async def _coordinate_cross_region_healing(self):
        """Coordinate healing actions across regions."""
        # Check for failures that require cross-region coordination
        cross_region_failures = [
            failure
            for failure in self.global_failures.values()
            if failure.global_impact and not failure.resolved
        ]

        for failure in cross_region_failures:
            await self._handle_distributed_failure(failure)

    async def _optimize_cross_region_routing(self):
        """Optimize routing across regions."""
        # Update routing tables based on latency and load
        await self.request_router.update_routing_tables(
            self.cross_region_latencies,
            {node.region: node.load for node in self.peer_nodes.values()},
        )

    async def _balance_global_load(self):
        """Balance load across global regions."""
        # Calculate load distribution
        region_loads = defaultdict(float)
        for node in self.peer_nodes.values():
            region_loads[node.region] += node.load

        # Balance if needed
        max_load = max(region_loads.values()) if region_loads else 0
        min_load = min(region_loads.values()) if region_loads else 0

        if max_load - min_load > 0.3:  # Significant imbalance
            await self.load_balancer.rebalance_global_load(region_loads)

    async def _execute_local_healing(self, failure: DistributedFailure):
        """Execute local healing for non-consensus failures."""
        logger.info(f"Executing local healing for failure: {failure.failure_id}")

        # Simulate local healing
        await asyncio.sleep(1)

        # Mark as resolved
        failure.resolved = True
        failure.resolution_strategy = "local_healing"

    def stop_coordination(self):
        """Stop global coordination."""
        self.coordination_enabled = False

        # Cancel coordination tasks
        if hasattr(self, "_coordination_tasks"):
            for task in self._coordination_tasks:
                task.cancel()

        logger.info(f"Node {self.node_id} stopped global coordination")

    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global status."""
        return {
            "node_info": {
                "node_id": self.node_id,
                "region": self.region.value,
                "status": self.local_node.status.value,
                "load": self.local_node.load,
                "reliability": self.local_node.reliability_score,
            },
            "global_state": self.global_state.to_dict(),
            "peer_nodes": {
                node_id: {
                    "region": node.region.value,
                    "status": node.status.value,
                    "load": node.load,
                    "latency_ms": node.latency_ms,
                    "is_healthy": node.is_healthy,
                }
                for node_id, node in self.peer_nodes.items()
            },
            "consensus": self.consensus_engine.get_consensus_status(),
            "failures": {
                "active_count": len(
                    [f for f in self.global_failures.values() if not f.resolved]
                ),
                "total_count": len(self.global_failures),
                "global_impact_count": len(
                    [f for f in self.global_failures.values() if f.global_impact]
                ),
            },
            "cross_region_latencies": {
                region.value: latency
                for region, latency in self.cross_region_latencies.items()
            },
        }


class GlobalRequestRouter:
    """Global request router for optimal traffic distribution."""

    def __init__(self):
        self.routing_tables: Dict[GlobalRegion, float] = {}
        self.traffic_redirections: Dict[GlobalRegion, GlobalRegion] = {}

    async def redirect_region_traffic(self, from_region: GlobalRegion):
        """Redirect traffic from a failed region."""
        # Find best alternative region
        alternatives = [r for r in GlobalRegion if r != from_region]
        best_alternative = min(
            alternatives, key=lambda r: self.routing_tables.get(r, 1.0)
        )

        self.traffic_redirections[from_region] = best_alternative
        logger.info(
            f"Redirecting traffic from {from_region.value} to {best_alternative.value}"
        )

    async def update_routing_tables(
        self, latencies: Dict[GlobalRegion, float], loads: Dict[GlobalRegion, float]
    ):
        """Update routing tables based on current conditions."""
        for region in GlobalRegion:
            latency = latencies.get(region, 100)
            load = loads.get(region, 0.5)

            # Combined routing score (lower is better)
            self.routing_tables[region] = latency * (1 + load)


class GlobalLoadBalancer:
    """Global load balancer for distributed systems."""

    def __init__(self):
        self.region_capacities: Dict[GlobalRegion, float] = dict.fromkeys(
            GlobalRegion, 1.0
        )

    async def scale_region(self, region: GlobalRegion, scale_factor: float):
        """Scale capacity in a specific region."""
        self.region_capacities[region] *= scale_factor
        logger.info(f"Scaled region {region.value} by factor {scale_factor}")

    async def rebalance_global_load(self, region_loads: Dict[GlobalRegion, float]):
        """Rebalance load across global regions."""
        total_load = sum(region_loads.values())
        target_load_per_region = total_load / len(GlobalRegion)

        for region, current_load in region_loads.items():
            if current_load > target_load_per_region * 1.2:
                # Overloaded region - scale up
                await self.scale_region(region, 1.3)
            elif current_load < target_load_per_region * 0.8:
                # Underloaded region - scale down
                await self.scale_region(region, 0.8)


class GlobalMetricsCollector:
    """Global metrics collector for distributed monitoring."""

    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def record_cross_region_latency(
        self, from_region: GlobalRegion, to_region: GlobalRegion, latency: float
    ):
        """Record cross-region latency metric."""
        metric_name = f"latency_{from_region.value}_to_{to_region.value}"
        self.metrics[metric_name].append(
            {"timestamp": datetime.now(), "value": latency}
        )

    def record_healing_event(self, event_type: str, success: bool, duration: float):
        """Record healing event metrics."""
        self.metrics[f"healing_{event_type}_duration"].append(
            {"timestamp": datetime.now(), "value": duration}
        )

        self.metrics[f"healing_{event_type}_success"].append(
            {"timestamp": datetime.now(), "value": 1.0 if success else 0.0}
        )


# Global coordination layer instance
_global_coordination_layer: Optional[GlobalCoordinationLayer] = None


def get_global_coordination_layer(
    region: GlobalRegion = GlobalRegion.US_EAST,
) -> GlobalCoordinationLayer:
    """Get the global coordination layer instance."""
    global _global_coordination_layer
    if _global_coordination_layer is None:
        _global_coordination_layer = GlobalCoordinationLayer(region)
    return _global_coordination_layer


async def create_distributed_healing_network(
    regions: List[GlobalRegion],
) -> Dict[str, GlobalCoordinationLayer]:
    """Create distributed healing network across multiple regions."""
    network = {}

    for region in regions:
        coordination_layer = GlobalCoordinationLayer(region)
        network[region.value] = coordination_layer

        # Join global network (simplified)
        bootstrap_nodes = [f"{r.value}-bootstrap" for r in regions if r != region]
        await coordination_layer.join_global_network(bootstrap_nodes)

    return network
