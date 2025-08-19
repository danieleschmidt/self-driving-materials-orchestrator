"""Ultra-high performance optimization system for materials orchestrator."""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import hashlib
import pickle
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None


@dataclass
class WorkerNode:
    """Worker node for distributed processing."""
    node_id: str
    capacity: int
    current_load: int
    performance_score: float
    last_health_check: datetime
    is_healthy: bool = True


class UltraHighPerformanceCache:
    """Ultra-high performance caching system with intelligent eviction."""
    
    def __init__(self, max_size: int = 100000, max_memory_mb: int = 1000, 
                 default_ttl: int = 3600, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.policy = policy
        self.cache = {}
        self.access_order = deque()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.current_memory = 0
        self._lock = RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_evictions = 0
        
        logger.info(f"Ultra-high performance cache initialized: {max_size} entries, {max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(key, entry)
                    self.misses += 1
                    return None
                
                # Update access patterns
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.frequency_counter[key] += 1
                
                # Update LRU order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache with intelligent eviction."""
        with self._lock:
            # Calculate entry size
            try:
                serialized_value = pickle.dumps(value)
                entry_size = len(serialized_value)
            except:
                # Fallback size estimation
                entry_size = len(str(value)) * 4
            
            # Check if single entry exceeds memory limit
            if entry_size > self.max_memory_bytes:
                logger.warning(f"Entry too large for cache: {entry_size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
                self.access_order.remove(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=entry_size,
                ttl_seconds=ttl or self.default_ttl
            )
            
            # Ensure space is available
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + entry_size > self.max_memory_bytes):
                if not self._evict_entry():
                    logger.error("Failed to evict entry for cache space")
                    return False
            
            # Add entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.frequency_counter[key] += 1
            self.current_memory += entry_size
            
            return True
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_order.clear()
                self.frequency_counter.clear()
                self.current_memory = 0
                return count
            
            # Pattern-based invalidation
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                entry = self.cache[key]
                self._remove_entry(key, entry)
            
            return len(keys_to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'memory_evictions': self.memory_evictions,
                'policy': self.policy.value
            }
    
    def _evict_entry(self) -> bool:
        """Evict entry based on policy."""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            key_to_evict = self.access_order[0]
        elif self.policy == CachePolicy.LFU:
            key_to_evict = min(self.frequency_counter.keys(), 
                              key=lambda k: self.frequency_counter[k])
        elif self.policy == CachePolicy.TTL:
            # Evict expired entries first
            now = datetime.now()
            expired_keys = [k for k, entry in self.cache.items() if self._is_expired(entry)]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = self.access_order[0]  # Fallback to LRU
        else:  # ADAPTIVE
            key_to_evict = self._adaptive_eviction()
        
        if key_to_evict in self.cache:
            entry = self.cache[key_to_evict]
            self._remove_entry(key_to_evict, entry)
            self.evictions += 1
            return True
        
        return False
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on multiple factors."""
        if not self.cache:
            return ""
        
        scores = {}
        now = datetime.now()
        
        for key, entry in self.cache.items():
            # Score based on access frequency, recency, and size
            frequency_score = self.frequency_counter[key]
            recency_score = (now - entry.last_accessed).total_seconds()
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Combined score (lower is better for eviction)
            scores[key] = frequency_score - (recency_score / 3600) - size_penalty
        
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str, entry: CacheEntry):
        """Remove entry and clean up references."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.frequency_counter:
            del self.frequency_counter[key]
        self.current_memory -= entry.size_bytes
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl_seconds is None:
            return False
        return (datetime.now() - entry.created_at).total_seconds() > entry.ttl_seconds


class DistributedLoadBalancer:
    """Intelligent load balancer for distributed processing."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED):
        self.strategy = strategy
        self.workers = {}
        self.round_robin_index = 0
        self.performance_history = defaultdict(list)
        self._lock = Lock()
        
        logger.info(f"Load balancer initialized with {strategy.value} strategy")
    
    def add_worker(self, node_id: str, capacity: int, initial_performance: float = 1.0):
        """Add worker node to load balancer."""
        with self._lock:
            self.workers[node_id] = WorkerNode(
                node_id=node_id,
                capacity=capacity,
                current_load=0,
                performance_score=initial_performance,
                last_health_check=datetime.now(),
                is_healthy=True
            )
        logger.info(f"Added worker {node_id} with capacity {capacity}")
    
    def remove_worker(self, node_id: str):
        """Remove worker node from load balancer."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                logger.info(f"Removed worker {node_id}")
    
    def select_worker(self, task_weight: int = 1) -> Optional[str]:
        """Select best worker for task based on strategy."""
        with self._lock:
            available_workers = [w for w in self.workers.values() 
                               if w.is_healthy and w.current_load + task_weight <= w.capacity]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected = available_workers[self.round_robin_index % len(available_workers)]
                self.round_robin_index += 1
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                selected = min(available_workers, key=lambda w: w.current_load)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                # Select based on capacity
                selected = max(available_workers, key=lambda w: w.capacity - w.current_load)
            else:  # PERFORMANCE_BASED
                selected = max(available_workers, key=lambda w: w.performance_score)
            
            return selected.node_id
    
    def start_task(self, node_id: str, task_weight: int = 1):
        """Mark task as started on worker."""
        with self._lock:
            if node_id in self.workers:
                self.workers[node_id].current_load += task_weight
    
    def complete_task(self, node_id: str, task_weight: int = 1, duration_seconds: float = 0):
        """Mark task as completed and update performance metrics."""
        with self._lock:
            if node_id in self.workers:
                worker = self.workers[node_id]
                worker.current_load = max(0, worker.current_load - task_weight)
                
                # Update performance score
                if duration_seconds > 0:
                    performance = task_weight / duration_seconds  # Tasks per second
                    self.performance_history[node_id].append(performance)
                    
                    # Keep only recent performance data
                    if len(self.performance_history[node_id]) > 100:
                        self.performance_history[node_id] = self.performance_history[node_id][-100:]
                    
                    # Update average performance
                    worker.performance_score = sum(self.performance_history[node_id]) / len(self.performance_history[node_id])
    
    def health_check(self, node_id: str, is_healthy: bool):
        """Update worker health status."""
        with self._lock:
            if node_id in self.workers:
                self.workers[node_id].is_healthy = is_healthy
                self.workers[node_id].last_health_check = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        with self._lock:
            total_capacity = sum(w.capacity for w in self.workers.values())
            total_load = sum(w.current_load for w in self.workers.values())
            healthy_workers = sum(1 for w in self.workers.values() if w.is_healthy)
            
            return {
                'total_workers': len(self.workers),
                'healthy_workers': healthy_workers,
                'total_capacity': total_capacity,
                'current_load': total_load,
                'utilization': (total_load / total_capacity) if total_capacity > 0 else 0,
                'strategy': self.strategy.value,
                'workers': {
                    w.node_id: {
                        'capacity': w.capacity,
                        'load': w.current_load,
                        'performance': w.performance_score,
                        'healthy': w.is_healthy
                    } for w in self.workers.values()
                }
            }


class ConcurrentExperimentProcessor:
    """High-performance concurrent experiment processing engine."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.active_experiments = {}
        self.completed_experiments = {}
        self._lock = Lock()
        
        # Initialize executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Concurrent processor initialized: {self.max_workers} workers ({'processes' if use_processes else 'threads'})")
    
    async def process_experiments_batch(self, experiments: List[Dict[str, Any]], 
                                      experiment_function: Callable) -> List[Dict[str, Any]]:
        """Process batch of experiments concurrently."""
        start_time = time.time()
        
        # Submit all experiments
        future_to_experiment = {}
        for exp in experiments:
            future = self.executor.submit(self._run_experiment_safe, experiment_function, exp)
            future_to_experiment[future] = exp
            
            with self._lock:
                self.active_experiments[exp.get('id', str(hash(str(exp))))] = {
                    'experiment': exp,
                    'start_time': time.time(),
                    'future': future
                }
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            exp_id = experiment.get('id', str(hash(str(experiment))))
            
            try:
                result = future.result()
                result['experiment_id'] = exp_id
                result['success'] = True
                results.append(result)
                
                with self._lock:
                    if exp_id in self.active_experiments:
                        duration = time.time() - self.active_experiments[exp_id]['start_time']
                        self.completed_experiments[exp_id] = {
                            'experiment': experiment,
                            'result': result,
                            'duration': duration,
                            'success': True
                        }
                        del self.active_experiments[exp_id]
                        
            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
                results.append({
                    'experiment_id': exp_id,
                    'success': False,
                    'error': str(e),
                    'experiment': experiment
                })
                
                with self._lock:
                    if exp_id in self.active_experiments:
                        duration = time.time() - self.active_experiments[exp_id]['start_time']
                        self.completed_experiments[exp_id] = {
                            'experiment': experiment,
                            'error': str(e),
                            'duration': duration,
                            'success': False
                        }
                        del self.active_experiments[exp_id]
        
        total_time = time.time() - start_time
        logger.info(f"Processed {len(experiments)} experiments in {total_time:.2f}s")
        
        return results
    
    def _run_experiment_safe(self, experiment_function: Callable, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run single experiment with error handling."""
        try:
            return experiment_function(experiment)
        except Exception as e:
            logger.error(f"Experiment function failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            completed_count = len(self.completed_experiments)
            if completed_count == 0:
                return {'completed_experiments': 0}
            
            durations = [exp['duration'] for exp in self.completed_experiments.values()]
            success_count = sum(1 for exp in self.completed_experiments.values() if exp['success'])
            
            return {
                'active_experiments': len(self.active_experiments),
                'completed_experiments': completed_count,
                'success_rate': success_count / completed_count,
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'experiments_per_second': completed_count / sum(durations) if durations else 0
            }
    
    def shutdown(self):
        """Shutdown executor gracefully."""
        self.executor.shutdown(wait=True)
        logger.info("Concurrent processor shutdown complete")


class AutoScalingManager:
    """Intelligent auto-scaling for dynamic resource management."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 64, target_utilization: float = 0.7):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.current_workers = min_workers
        self.load_history = deque(maxlen=100)
        self.scaling_decisions = []
        
        logger.info(f"Auto-scaling manager initialized: {min_workers}-{max_workers} workers, target {target_utilization}")
    
    def report_load(self, current_load: float, queue_size: int = 0):
        """Report current system load for scaling decisions."""
        self.load_history.append({
            'timestamp': datetime.now(),
            'load': current_load,
            'queue_size': queue_size,
            'workers': self.current_workers
        })
        
        # Make scaling decision if enough data points
        if len(self.load_history) >= 10:
            self._make_scaling_decision()
    
    def _make_scaling_decision(self):
        """Make intelligent scaling decision based on load history."""
        recent_loads = [point['load'] for point in list(self.load_history)[-10:]]
        recent_queues = [point['queue_size'] for point in list(self.load_history)[-10:]]
        
        avg_load = sum(recent_loads) / len(recent_loads)
        avg_queue = sum(recent_queues) / len(recent_queues)
        
        # Scale up conditions
        should_scale_up = (
            avg_load > self.target_utilization * 1.2 or  # High utilization
            avg_queue > 20 or  # Large queue
            (avg_load > self.target_utilization and self._is_trend_increasing())
        )
        
        # Scale down conditions
        should_scale_down = (
            avg_load < self.target_utilization * 0.5 and  # Low utilization
            avg_queue == 0 and  # No queue
            self._is_trend_decreasing()
        )
        
        if should_scale_up and self.current_workers < self.max_workers:
            new_workers = min(self.max_workers, int(self.current_workers * 1.5))
            self._scale_to(new_workers, 'scale_up', avg_load)
            
        elif should_scale_down and self.current_workers > self.min_workers:
            new_workers = max(self.min_workers, int(self.current_workers * 0.8))
            self._scale_to(new_workers, 'scale_down', avg_load)
    
    def _scale_to(self, new_worker_count: int, action: str, trigger_load: float):
        """Execute scaling action."""
        old_count = self.current_workers
        self.current_workers = new_worker_count
        
        decision = {
            'timestamp': datetime.now(),
            'action': action,
            'old_workers': old_count,
            'new_workers': new_worker_count,
            'trigger_load': trigger_load
        }
        self.scaling_decisions.append(decision)
        
        logger.info(f"Auto-scaling: {action} from {old_count} to {new_worker_count} workers (load: {trigger_load:.2f})")
    
    def _is_trend_increasing(self) -> bool:
        """Check if load trend is increasing."""
        if len(self.load_history) < 5:
            return False
        
        recent = list(self.load_history)[-5:]
        loads = [point['load'] for point in recent]
        
        # Simple trend detection
        return loads[-1] > loads[0] and loads[-2] > loads[1]
    
    def _is_trend_decreasing(self) -> bool:
        """Check if load trend is decreasing.""" 
        if len(self.load_history) < 5:
            return False
        
        recent = list(self.load_history)[-5:]
        loads = [point['load'] for point in recent]
        
        return loads[-1] < loads[0] and loads[-2] < loads[1]
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status and metrics."""
        if not self.load_history:
            return {'current_workers': self.current_workers}
        
        recent_load = self.load_history[-1]['load']
        recent_decisions = self.scaling_decisions[-10:] if self.scaling_decisions else []
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'target_utilization': self.target_utilization,
            'current_load': recent_load,
            'recent_decisions': len(recent_decisions),
            'scaling_history': recent_decisions
        }


def create_ultra_high_performance_system() -> Tuple[UltraHighPerformanceCache, DistributedLoadBalancer, 
                                                  ConcurrentExperimentProcessor, AutoScalingManager]:
    """Factory function to create ultra-high performance system."""
    # Create optimized cache
    cache = UltraHighPerformanceCache(
        max_size=50000,
        max_memory_mb=2000,
        policy=CachePolicy.ADAPTIVE
    )
    
    # Create load balancer
    load_balancer = DistributedLoadBalancer(LoadBalancingStrategy.PERFORMANCE_BASED)
    
    # Add some worker nodes
    load_balancer.add_worker("worker_1", capacity=10, initial_performance=1.0)
    load_balancer.add_worker("worker_2", capacity=8, initial_performance=1.2)
    load_balancer.add_worker("worker_3", capacity=12, initial_performance=0.9)
    
    # Create concurrent processor
    processor = ConcurrentExperimentProcessor(max_workers=16, use_processes=False)
    
    # Create auto-scaling manager
    auto_scaler = AutoScalingManager(min_workers=4, max_workers=32, target_utilization=0.75)
    
    logger.info("Ultra-high performance system created")
    return cache, load_balancer, processor, auto_scaler