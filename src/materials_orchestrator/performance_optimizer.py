"""Advanced performance optimization and scaling system."""

import concurrent.futures
import logging
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RESPONSE_TIME = "response_time"


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""

    operation_count: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0  # operations per second
    error_rate: float = 0.0
    concurrent_operations: int = 0
    resource_utilization: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0


class AdaptivePerformanceCache:
    """High-performance adaptive cache with intelligent eviction."""

    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.strategy = CacheStrategy.ADAPTIVE
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._access_pattern_history: List[str] = []

        # Performance tracking
        self._operation_times: List[float] = []
        self._last_cleanup = time.time()

        logger.info(f"Adaptive cache initialized with max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking."""
        start_time = time.time()

        with self._lock:
            if key not in self.cache:
                self._miss_count += 1
                self._operation_times.append(time.time() - start_time)
                return None

            entry = self.cache[key]

            # Check TTL expiration
            if self._is_expired(entry):
                del self.cache[key]
                self._miss_count += 1
                self._operation_times.append(time.time() - start_time)
                return None

            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._hit_count += 1

            # Track access pattern
            self._access_pattern_history.append(key)
            if len(self._access_pattern_history) > 1000:
                self._access_pattern_history = self._access_pattern_history[-500:]

            self._operation_times.append(time.time() - start_time)
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value in cache with intelligent eviction."""
        start_time = time.time()

        with self._lock:
            # Calculate value size (approximate)
            try:
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                size_bytes = len(str(value).encode("utf-8"))

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds or self.default_ttl,
                size_bytes=size_bytes,
            )

            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                self._evict_entries()

            self.cache[key] = entry

            # Periodic cleanup
            if time.time() - self._last_cleanup > 300:  # Every 5 minutes
                self._cleanup_expired()
                self._last_cleanup = time.time()

            self._operation_times.append(time.time() - start_time)
            return True

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl_seconds is None:
            return False

        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds

    def _evict_entries(self):
        """Evict entries based on adaptive strategy."""
        eviction_count = max(1, len(self.cache) // 10)  # Evict 10% of entries

        if self.strategy == CacheStrategy.ADAPTIVE:
            # Use adaptive strategy based on access patterns
            self._adaptive_eviction(eviction_count)
        elif self.strategy == CacheStrategy.LRU:
            self._lru_eviction(eviction_count)
        elif self.strategy == CacheStrategy.LFU:
            self._lfu_eviction(eviction_count)
        else:  # TTL
            self._ttl_eviction(eviction_count)

    def _adaptive_eviction(self, count: int):
        """Adaptive eviction based on access patterns and entry characteristics."""
        # Score entries based on multiple factors
        scored_entries = []

        for key, entry in self.cache.items():
            score = self._calculate_eviction_score(entry)
            scored_entries.append((score, key))

        # Sort by score (higher score = more likely to evict)
        scored_entries.sort(reverse=True)

        # Evict highest scoring entries
        for i in range(min(count, len(scored_entries))):
            key_to_evict = scored_entries[i][1]
            del self.cache[key_to_evict]
            logger.debug(f"Evicted cache entry: {key_to_evict}")

    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score for adaptive cache management."""
        now = datetime.now()

        # Time since last access (higher = more likely to evict)
        time_score = (now - entry.last_accessed).total_seconds() / 3600  # Hours

        # Access frequency (lower = more likely to evict)
        frequency_score = 1.0 / max(entry.access_count, 1)

        # Size factor (larger entries slightly more likely to evict)
        size_score = entry.size_bytes / 10000  # Normalize by 10KB

        # TTL factor (closer to expiration = more likely to evict)
        ttl_score = 0
        if entry.ttl_seconds:
            age = (now - entry.created_at).total_seconds()
            remaining_ratio = max(0, 1 - age / entry.ttl_seconds)
            ttl_score = 1 - remaining_ratio

        # Weighted combination
        total_score = (
            time_score * 0.4
            + frequency_score * 0.3
            + size_score * 0.1
            + ttl_score * 0.2
        )

        return total_score

    def _lru_eviction(self, count: int):
        """Least Recently Used eviction."""
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        for i in range(min(count, len(sorted_entries))):
            key_to_evict = sorted_entries[i][0]
            del self.cache[key_to_evict]

    def _lfu_eviction(self, count: int):
        """Least Frequently Used eviction."""
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].access_count)

        for i in range(min(count, len(sorted_entries))):
            key_to_evict = sorted_entries[i][0]
            del self.cache[key_to_evict]

    def _ttl_eviction(self, count: int):
        """Time-To-Live based eviction (oldest first)."""
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].created_at)

        for i in range(min(count, len(sorted_entries))):
            key_to_evict = sorted_entries[i][0]
            del self.cache[key_to_evict]

    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = []

        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total_requests, 1)

            avg_operation_time = sum(self._operation_times) / max(
                len(self._operation_times), 1
            )

            total_size = sum(entry.size_bytes for entry in self.cache.values())

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "average_operation_time_ms": avg_operation_time * 1000,
                "total_size_bytes": total_size,
                "strategy": self.strategy.value,
                "utilization": len(self.cache) / self.max_size,
            }

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._operation_times.clear()
            self._access_pattern_history.clear()


class ConcurrentExecutionPool:
    """High-performance concurrent execution pool with load balancing."""

    def __init__(self, max_workers: int = None, queue_size: int = 1000):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.queue_size = queue_size
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.task_queue = queue.Queue(maxsize=queue_size)

        # Performance tracking
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        self.task_response_times: List[float] = []
        self._lock = threading.Lock()

        # Load balancing
        self.worker_loads: Dict[int, int] = dict.fromkeys(range(self.max_workers), 0)
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS

        logger.info(
            f"Concurrent execution pool initialized with {self.max_workers} workers"
        )

    def submit_task(
        self, func: Callable, *args, priority: int = 0, **kwargs
    ) -> concurrent.futures.Future:
        """Submit a task for concurrent execution."""

        task_id = f"task_{time.time()}_{id(func)}"
        start_time = time.time()

        def wrapped_func():
            try:
                with self._lock:
                    self.active_tasks.add(task_id)

                result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                with self._lock:
                    self.active_tasks.discard(task_id)
                    self.completed_tasks += 1
                    self.total_execution_time += execution_time
                    self.task_response_times.append(execution_time)

                    # Keep response time history manageable
                    if len(self.task_response_times) > 1000:
                        self.task_response_times = self.task_response_times[-500:]

                return result

            except Exception as e:
                with self._lock:
                    self.active_tasks.discard(task_id)
                    self.failed_tasks += 1
                raise e

        return self.executor.submit(wrapped_func)

    def submit_batch(
        self, tasks: List[Tuple[Callable, tuple, dict]], max_concurrent: int = None
    ) -> List[concurrent.futures.Future]:
        """Submit a batch of tasks with concurrency control."""
        max_concurrent = max_concurrent or self.max_workers

        futures = []
        semaphore = threading.Semaphore(max_concurrent)

        def semaphore_wrapped_func(func, args, kwargs):
            with semaphore:
                return func(*args, **kwargs)

        for func, args, kwargs in tasks:
            future = self.submit_task(semaphore_wrapped_func, func, args, kwargs)
            futures.append(future)

        return futures

    def map_concurrent(
        self, func: Callable, iterable, max_workers: int = None, timeout: float = None
    ) -> List[Any]:
        """Map function over iterable with concurrent execution."""
        max_workers = max_workers or self.max_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            results = []

            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Concurrent task failed: {e}")
                    results.append(None)

        return results

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get execution pool performance metrics."""
        with self._lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            avg_response_time = (
                sum(self.task_response_times) / len(self.task_response_times)
                if self.task_response_times
                else 0.0
            )

            error_rate = self.failed_tasks / max(total_tasks, 1)
            throughput = total_tasks / max(self.total_execution_time, 1)

            return PerformanceMetrics(
                operation_count=total_tasks,
                average_response_time=avg_response_time,
                throughput=throughput,
                error_rate=error_rate,
                concurrent_operations=len(self.active_tasks),
                resource_utilization=len(self.active_tasks) / self.max_workers,
            )

    def shutdown(self, wait: bool = True):
        """Shutdown the execution pool."""
        self.executor.shutdown(wait=wait)
        logger.info("Concurrent execution pool shutdown")


class AutoScalingManager:
    """Automatic scaling manager based on performance metrics."""

    def __init__(self, min_workers: int = 2, max_workers: int = 32):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_time = time.time()
        self.scale_cooldown = 60.0  # seconds

        # Scaling thresholds
        self.scale_up_cpu_threshold = 0.8
        self.scale_down_cpu_threshold = 0.3
        self.scale_up_response_time_threshold = 2.0  # seconds
        self.scale_down_response_time_threshold = 0.5

        logger.info(
            f"Auto-scaling manager initialized ({min_workers}-{max_workers} workers)"
        )

    def should_scale(self, metrics: PerformanceMetrics) -> Tuple[bool, int]:
        """Determine if scaling is needed based on metrics."""

        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False, self.current_workers  # Still in cooldown

        # Scale up conditions
        should_scale_up = (
            metrics.resource_utilization > self.scale_up_cpu_threshold
            or metrics.average_response_time > self.scale_up_response_time_threshold
            or metrics.concurrent_operations > self.current_workers * 0.9
        )

        # Scale down conditions
        should_scale_down = (
            metrics.resource_utilization < self.scale_down_cpu_threshold
            and metrics.average_response_time < self.scale_down_response_time_threshold
            and metrics.concurrent_operations < self.current_workers * 0.3
        )

        new_worker_count = self.current_workers

        if should_scale_up and self.current_workers < self.max_workers:
            # Scale up by 50% or at least 1 worker
            scale_factor = max(1, int(self.current_workers * 0.5))
            new_worker_count = min(
                self.max_workers, self.current_workers + scale_factor
            )

        elif should_scale_down and self.current_workers > self.min_workers:
            # Scale down by 25% or at least 1 worker
            scale_factor = max(1, int(self.current_workers * 0.25))
            new_worker_count = max(
                self.min_workers, self.current_workers - scale_factor
            )

        needs_scaling = new_worker_count != self.current_workers

        if needs_scaling:
            # Record scaling decision
            scaling_event = {
                "timestamp": datetime.now().isoformat(),
                "old_workers": self.current_workers,
                "new_workers": new_worker_count,
                "trigger_metrics": {
                    "resource_utilization": metrics.resource_utilization,
                    "average_response_time": metrics.average_response_time,
                    "concurrent_operations": metrics.concurrent_operations,
                    "throughput": metrics.throughput,
                },
                "scale_direction": (
                    "up" if new_worker_count > self.current_workers else "down"
                ),
            }

            self.scaling_history.append(scaling_event)
            self.current_workers = new_worker_count
            self.last_scale_time = current_time

            logger.info(
                f"Auto-scaling: {scaling_event['old_workers']} -> {new_worker_count} workers"
            )

        return needs_scaling, new_worker_count

    def get_scaling_recommendations(
        self, metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Get scaling recommendations without actually scaling."""

        needs_scaling, recommended_workers = self.should_scale(metrics)

        return {
            "needs_scaling": needs_scaling,
            "current_workers": self.current_workers,
            "recommended_workers": recommended_workers,
            "scale_direction": (
                "up"
                if recommended_workers > self.current_workers
                else "down" if recommended_workers < self.current_workers else "none"
            ),
            "scaling_factor": abs(recommended_workers - self.current_workers),
            "cooldown_remaining": max(
                0, self.scale_cooldown - (time.time() - self.last_scale_time)
            ),
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Initialize components
        self.cache = AdaptivePerformanceCache(
            max_size=self.config["cache"]["max_size"],
            default_ttl=self.config["cache"]["default_ttl"],
        )

        self.executor_pool = ConcurrentExecutionPool(
            max_workers=self.config["concurrency"]["max_workers"],
            queue_size=self.config["concurrency"]["queue_size"],
        )

        self.auto_scaler = AutoScalingManager(
            min_workers=self.config["scaling"]["min_workers"],
            max_workers=self.config["scaling"]["max_workers"],
        )

        # Performance monitoring
        self.performance_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None

        logger.info("Performance optimizer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default performance configuration."""
        return {
            "cache": {"max_size": 10000, "default_ttl": 3600, "cleanup_interval": 300},
            "concurrency": {
                "max_workers": min(32, (os.cpu_count() or 1) + 4),
                "queue_size": 1000,
                "batch_size": 100,
            },
            "scaling": {
                "min_workers": 2,
                "max_workers": 32,
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3,
                "cooldown_seconds": 60,
            },
            "monitoring": {
                "enabled": True,
                "interval_seconds": 30,
                "history_size": 1000,
            },
        }

    def cached_execute(self, cache_key: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with caching."""

        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_result

        # Execute function
        logger.debug(f"Cache miss for key: {cache_key}, executing function")
        result = func(*args, **kwargs)

        # Cache result
        self.cache.put(cache_key, result)

        return result

    def concurrent_execute(
        self, tasks: List[Tuple[Callable, tuple, dict]], max_concurrent: int = None
    ) -> List[Any]:
        """Execute tasks concurrently with performance optimization."""

        futures = self.executor_pool.submit_batch(tasks, max_concurrent=max_concurrent)

        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Concurrent task failed: {e}")
                results.append(None)

        return results

    def start_performance_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                executor_metrics = self.executor_pool.get_performance_metrics()
                cache_stats = self.cache.get_performance_stats()

                # Enhanced metrics with cache information
                combined_metrics = PerformanceMetrics(
                    operation_count=executor_metrics.operation_count,
                    average_response_time=executor_metrics.average_response_time,
                    cache_hit_rate=cache_stats["hit_rate"],
                    throughput=executor_metrics.throughput,
                    error_rate=executor_metrics.error_rate,
                    concurrent_operations=executor_metrics.concurrent_operations,
                    resource_utilization=executor_metrics.resource_utilization,
                )

                self.performance_history.append(combined_metrics)

                # Keep history manageable
                if (
                    len(self.performance_history)
                    > self.config["monitoring"]["history_size"]
                ):
                    self.performance_history = self.performance_history[-500:]

                # Check auto-scaling
                needs_scaling, new_worker_count = self.auto_scaler.should_scale(
                    combined_metrics
                )
                if needs_scaling:
                    logger.info(
                        f"Performance-based scaling triggered: {new_worker_count} workers"
                    )

                time.sleep(self.config["monitoring"]["interval_seconds"])

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(self.config["monitoring"]["interval_seconds"])

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        cache_stats = self.cache.get_performance_stats()
        executor_metrics = self.executor_pool.get_performance_metrics()
        scaling_recommendations = self.auto_scaler.get_scaling_recommendations(
            executor_metrics
        )

        # Historical analysis
        if self.performance_history:
            recent_metrics = self.performance_history[-10:]  # Last 10 measurements
            avg_response_time = sum(
                m.average_response_time for m in recent_metrics
            ) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(
                recent_metrics
            )
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(
                recent_metrics
            )
        else:
            avg_response_time = executor_metrics.average_response_time
            avg_throughput = executor_metrics.throughput
            avg_cache_hit_rate = cache_stats["hit_rate"]

        return {
            "timestamp": datetime.now().isoformat(),
            "cache_performance": cache_stats,
            "execution_performance": {
                "active_tasks": executor_metrics.concurrent_operations,
                "completed_tasks": executor_metrics.operation_count,
                "average_response_time": avg_response_time,
                "throughput": avg_throughput,
                "error_rate": executor_metrics.error_rate,
                "resource_utilization": executor_metrics.resource_utilization,
            },
            "scaling_status": scaling_recommendations,
            "historical_averages": {
                "response_time": avg_response_time,
                "throughput": avg_throughput,
                "cache_hit_rate": avg_cache_hit_rate,
            },
            "optimization_recommendations": self._generate_optimization_recommendations(),
        }

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        cache_stats = self.cache.get_performance_stats()
        executor_metrics = self.executor_pool.get_performance_metrics()

        # Cache recommendations
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append(
                "Consider increasing cache size or TTL - low hit rate detected"
            )

        if cache_stats["utilization"] > 0.9:
            recommendations.append(
                "Cache is near capacity - consider increasing max_size"
            )

        # Concurrency recommendations
        if executor_metrics.resource_utilization > 0.9:
            recommendations.append(
                "High resource utilization - consider increasing worker count"
            )

        if executor_metrics.error_rate > 0.1:
            recommendations.append(
                "High error rate detected - review error handling and retry logic"
            )

        if executor_metrics.average_response_time > 2.0:
            recommendations.append(
                "High average response time - consider optimizing task execution or scaling up"
            )

        # Scaling recommendations
        scaling_rec = self.auto_scaler.get_scaling_recommendations(executor_metrics)
        if scaling_rec["needs_scaling"]:
            recommendations.append(
                f"Auto-scaling recommended: {scaling_rec['scale_direction']} to {scaling_rec['recommended_workers']} workers"
            )

        if not recommendations:
            recommendations.append("System performance is optimal")

        return recommendations

    def shutdown(self):
        """Shutdown performance optimizer."""
        self.stop_performance_monitoring()
        self.executor_pool.shutdown()
        logger.info("Performance optimizer shutdown")


# Global performance optimizer instance
_global_optimizer = None


def get_global_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer
