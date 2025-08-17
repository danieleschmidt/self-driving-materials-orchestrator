"""Advanced scaling and optimization for high-performance materials discovery."""

import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
import threading
import time
import logging
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue
from .utils import np, NUMPY_AVAILABLE
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for scaling performance."""

    concurrent_experiments: int = 0
    max_concurrent_experiments: int = 0
    total_throughput: float = 0.0  # experiments/hour
    average_response_time: float = 0.0  # seconds
    memory_usage: float = 0.0  # MB
    cpu_utilization: float = 0.0  # percentage
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    auto_scale_events: int = 0


class ExperimentPriority:
    """Priority queue for experiment scheduling."""

    def __init__(self):
        self.queue = PriorityQueue()
        self.counter = 0

    def add_experiment(self, experiment: Dict[str, Any], priority: int = 5):
        """Add experiment with priority (1=highest, 10=lowest)."""
        self.counter += 1
        # Use counter for tie-breaking to maintain FIFO for same priorities
        self.queue.put((priority, self.counter, experiment))

    def get_next_experiment(self) -> Optional[Dict[str, Any]]:
        """Get highest priority experiment."""
        if self.queue.empty():
            return None
        _, _, experiment = self.queue.get()
        return experiment

    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()


class AdaptiveLoadBalancer:
    """Adaptive load balancing for experiment execution."""

    def __init__(self, initial_workers: int = 4, max_workers: int = 32):
        self.min_workers = max(1, initial_workers // 2)
        self.max_workers = max_workers
        self.current_workers = initial_workers
        self.executor = ThreadPoolExecutor(max_workers=initial_workers)

        # Metrics for adaptive scaling
        self.queue_sizes = []
        self.response_times = []
        self.last_scale_event = time.time()
        self.scale_cooldown = 30.0  # seconds

        logger.info(f"Load balancer initialized: {initial_workers} workers")

    def submit_experiment(self, func: Callable, *args, **kwargs):
        """Submit experiment for execution."""
        start_time = time.time()
        future = self.executor.submit(func, *args, **kwargs)

        def track_completion(f):
            try:
                result = f.result()
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self._consider_scaling()
                return result
            except Exception as e:
                logger.error(f"Experiment execution failed: {e}")
                raise

        future.add_done_callback(track_completion)
        return future

    def _consider_scaling(self):
        """Consider auto-scaling based on metrics."""
        current_time = time.time()
        if current_time - self.last_scale_event < self.scale_cooldown:
            return

        # Get recent metrics
        recent_times = (
            self.response_times[-20:]
            if len(self.response_times) >= 20
            else self.response_times
        )
        if not recent_times:
            return

        avg_response_time = np.mean(recent_times)
        queue_size = self.executor._work_queue.qsize()

        # Scale up if response time is high or queue is building
        if (
            avg_response_time > 5.0 or queue_size > self.current_workers * 2
        ) and self.current_workers < self.max_workers:
            self._scale_up()

        # Scale down if response time is low and queue is empty
        elif (
            avg_response_time < 1.0
            and queue_size == 0
            and self.current_workers > self.min_workers
        ):
            self._scale_down()

    def _scale_up(self):
        """Scale up workers."""
        new_workers = min(self.current_workers * 2, self.max_workers)
        if new_workers > self.current_workers:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=new_workers)
            self.current_workers = new_workers
            self.last_scale_event = time.time()
            logger.info(f"Scaled up to {new_workers} workers")

    def _scale_down(self):
        """Scale down workers."""
        new_workers = max(self.current_workers // 2, self.min_workers)
        if new_workers < self.current_workers:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=new_workers)
            self.current_workers = new_workers
            self.last_scale_event = time.time()
            logger.info(f"Scaled down to {new_workers} workers")


class DistributedExperimentManager:
    """Manages distributed experiment execution across multiple processes."""

    def __init__(self, num_processes: Optional[int] = None):
        self.num_processes = num_processes or mp.cpu_count()
        self.process_executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.num_processes * 2)

        # Coordination
        self.experiment_queue = ExperimentPriority()
        self.active_experiments = {}
        self.completed_experiments = []

        # Scaling components
        self.load_balancer = AdaptiveLoadBalancer()

        # Metrics
        self.metrics = ScalingMetrics()
        self.metrics_lock = threading.Lock()

        logger.info(
            f"Distributed manager initialized with {self.num_processes} processes"
        )

    async def execute_campaign_distributed(
        self, experiments: List[Dict[str, Any]], max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute experiments in distributed manner."""

        # Priority assignment based on expected value
        for i, exp in enumerate(experiments):
            priority = self._calculate_priority(exp)
            self.experiment_queue.add_experiment(exp, priority)

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_single(experiment):
            async with semaphore:
                return await self._execute_experiment_async(experiment)

        # Execute all experiments concurrently
        tasks = [execute_single(exp) for exp in experiments]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]

        logger.info(
            f"Distributed execution completed: {len(successful_results)}/{len(experiments)} successful"
        )
        return successful_results

    def _calculate_priority(self, experiment: Dict[str, Any]) -> int:
        """Calculate experiment priority (1=highest, 10=lowest)."""
        # Simple heuristic - can be enhanced with ML prediction
        params = experiment.get("parameters", {})

        # Prioritize experiments in promising regions
        if "expected_performance" in experiment:
            perf = experiment["expected_performance"]
            if perf > 0.8:
                return 1  # High priority
            elif perf > 0.6:
                return 3  # Medium-high priority
            elif perf > 0.4:
                return 5  # Medium priority
            else:
                return 7  # Lower priority

        # Default priority
        return 5

    async def _execute_experiment_async(
        self, experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single experiment asynchronously."""
        loop = asyncio.get_event_loop()

        try:
            # Update metrics
            with self.metrics_lock:
                self.metrics.concurrent_experiments += 1
                self.metrics.max_concurrent_experiments = max(
                    self.metrics.max_concurrent_experiments,
                    self.metrics.concurrent_experiments,
                )

            # Execute in thread pool to avoid blocking
            result = await loop.run_in_executor(
                self.thread_executor, self._execute_experiment_sync, experiment
            )

            return result

        finally:
            # Update metrics
            with self.metrics_lock:
                self.metrics.concurrent_experiments -= 1

    def _execute_experiment_sync(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous experiment execution."""
        from . import simulate_perovskite_experiment

        try:
            start_time = time.time()

            parameters = experiment.get("parameters", {})
            result = simulate_perovskite_experiment(parameters)

            duration = time.time() - start_time

            return {
                "experiment_id": experiment.get("id", "unknown"),
                "parameters": parameters,
                "results": result,
                "duration": duration,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            return {
                "experiment_id": experiment.get("id", "unknown"),
                "parameters": experiment.get("parameters", {}),
                "results": {},
                "error": str(e),
                "status": "failed",
            }


class PerformanceOptimizer:
    """Optimizes performance across the entire system."""

    def __init__(self):
        self.distributed_manager = DistributedExperimentManager()
        self.optimization_cache = {}
        self.performance_history = []

        # Auto-tuning parameters
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.current_batch_size = 4
        self.batch_performance = {}

        logger.info("Performance optimizer initialized")

    def optimize_batch_size(self, experiment_count: int) -> int:
        """Determine optimal batch size based on experiment count and history."""

        # For small experiment counts, use smaller batches
        if experiment_count < 10:
            return min(2, experiment_count)

        # Use performance history to select best batch size
        if self.batch_performance:
            best_batch = max(
                self.batch_performance.keys(), key=lambda x: self.batch_performance[x]
            )
            return min(best_batch, experiment_count // 2)

        # Default adaptive sizing
        return min(self.current_batch_size, experiment_count // 4, 8)

    def optimize_memory_usage(self):
        """Optimize memory usage by clearing caches and cleaning up."""

        # Clear old cache entries
        current_time = time.time()
        cache_ttl = 3600  # 1 hour

        expired_keys = [
            key
            for key, (value, timestamp) in self.optimization_cache.items()
            if current_time - timestamp > cache_ttl
        ]

        for key in expired_keys:
            del self.optimization_cache[key]

        # Limit cache size
        if len(self.optimization_cache) > 1000:
            # Keep only most recent 500 entries
            sorted_items = sorted(
                self.optimization_cache.items(),
                key=lambda x: x[1][1],  # Sort by timestamp
                reverse=True,
            )
            self.optimization_cache = dict(sorted_items[:500])

        logger.debug(
            f"Memory optimization: cache size = {len(self.optimization_cache)}"
        )

    async def execute_optimized_campaign(
        self, experiments: List[Dict[str, Any]], optimization_level: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """Execute campaign with performance optimizations."""

        start_time = time.time()

        # Determine optimal concurrency based on system resources
        if optimization_level == "speed":
            max_concurrent = min(32, len(experiments))
        elif optimization_level == "memory":
            max_concurrent = min(4, len(experiments))
        else:  # balanced
            max_concurrent = min(16, len(experiments))

        # Execute with optimization
        results = await self.distributed_manager.execute_campaign_distributed(
            experiments, max_concurrent
        )

        # Record performance metrics
        duration = time.time() - start_time
        throughput = len(results) / duration if duration > 0 else 0

        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "experiment_count": len(experiments),
                "successful_count": len(results),
                "duration": duration,
                "throughput": throughput,
                "max_concurrent": max_concurrent,
                "optimization_level": optimization_level,
            }
        )

        logger.info(
            f"Optimized execution: {len(results)} experiments in {duration:.2f}s "
            f"({throughput:.1f} exp/s)"
        )

        return results


# Global optimizer instance
_global_optimizer = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer
