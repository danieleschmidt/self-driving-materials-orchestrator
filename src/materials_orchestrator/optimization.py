"""Performance optimization and caching for autonomous lab operations."""

import time
import threading
import concurrent.futures
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import logging
import hashlib
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def access(self) -> Any:
        """Access cache entry and update metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value


class AdaptiveCache:
    """Adaptive caching system that learns from access patterns."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        self._lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate cache key from function call."""
        # Create a stable hash of the arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            self._record_access(key)
            return entry.access()
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Use adaptive TTL based on access patterns
            if ttl is None:
                ttl = self._calculate_adaptive_ttl(key)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                ttl=ttl
            )
            
            self.cache[key] = entry
            
            # Evict if over size limit
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _record_access(self, key: str) -> None:
        """Record access pattern for adaptive caching."""
        now = datetime.now()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(now)
        
        # Keep only recent access history
        cutoff = now - timedelta(hours=24)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def _calculate_adaptive_ttl(self, key: str) -> float:
        """Calculate adaptive TTL based on access patterns."""
        if key not in self.access_patterns or not self.access_patterns[key]:
            return self.default_ttl
        
        accesses = self.access_patterns[key]
        
        # Calculate access frequency (accesses per hour)
        if len(accesses) >= 2:
            time_span = (accesses[-1] - accesses[0]).total_seconds() / 3600
            frequency = len(accesses) / max(time_span, 0.1)
            
            # Higher frequency = longer TTL
            adaptive_ttl = self.default_ttl * (1 + frequency / 10)
            return min(adaptive_ttl, self.default_ttl * 10)  # Cap at 10x default
        
        return self.default_ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 10% of entries
        evict_count = max(1, len(self.cache) // 10)
        for i in range(evict_count):
            key, _ = sorted_entries[i]
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


# Global cache instance
_global_cache = AdaptiveCache()


def cached(ttl: Optional[float] = None, cache_instance: Optional[AdaptiveCache] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _global_cache
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: cache.get_stats()
        
        return wrapper
    return decorator


class ConcurrentExecutor:
    """Enhanced concurrent execution with intelligent load balancing."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_queue: List[Tuple[Callable, Tuple, Dict]] = []
        self.results: Dict[str, Any] = {}
        self.task_times: List[float] = []
        self._lock = threading.Lock()
        
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for concurrent execution."""
        task_id = hashlib.md5(
            f"{func.__name__}{args}{kwargs}{time.time()}".encode()
        ).hexdigest()
        
        future = self.executor.submit(self._execute_with_timing, func, task_id, *args, **kwargs)
        
        return task_id
    
    def _execute_with_timing(self, func: Callable, task_id: str, *args, **kwargs) -> Any:
        """Execute function with timing."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.results[task_id] = {'status': 'completed', 'result': result, 'error': None}
        except Exception as e:
            with self._lock:
                self.results[task_id] = {'status': 'failed', 'result': None, 'error': str(e)}
            logger.error(f"Task {task_id} failed: {e}")
        finally:
            execution_time = time.time() - start_time
            with self._lock:
                self.task_times.append(execution_time)
                if len(self.task_times) > 1000:  # Keep last 1000 times
                    self.task_times = self.task_times[-500:]
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        start_time = time.time()
        while True:
            with self._lock:
                if task_id in self.results:
                    result_info = self.results[task_id]
                    if result_info['status'] == 'completed':
                        return result_info['result']
                    elif result_info['status'] == 'failed':
                        raise Exception(f"Task failed: {result_info['error']}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            time.sleep(0.1)
    
    def submit_batch(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[str]:
        """Submit multiple tasks concurrently."""
        task_ids = []
        for func, args, kwargs in tasks:
            task_id = self.submit_task(func, *args, **kwargs)
            task_ids.append(task_id)
        return task_ids
    
    def wait_for_batch(self, task_ids: List[str], timeout: Optional[float] = None) -> List[Any]:
        """Wait for batch of tasks to complete."""
        results = []
        for task_id in task_ids:
            try:
                result = self.get_result(task_id, timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch task {task_id} failed: {e}")
                results.append(None)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self.task_times:
                return {'avg_time': 0, 'min_time': 0, 'max_time': 0, 'total_tasks': 0}
            
            return {
                'avg_time': sum(self.task_times) / len(self.task_times),
                'min_time': min(self.task_times),
                'max_time': max(self.task_times),
                'total_tasks': len(self.task_times),
                'active_workers': self.max_workers
            }
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


class ResourcePool:
    """Resource pool for managing expensive objects (DB connections, etc.)."""
    
    def __init__(self, factory: Callable, max_size: int = 10, timeout: float = 30.0):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.pool: List[Any] = []
        self.in_use: set = set()
        self._lock = threading.Semaphore(max_size)
        self._pool_lock = threading.Lock()
        
    def acquire(self) -> Any:
        """Acquire resource from pool."""
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError("Failed to acquire resource from pool")
        
        with self._pool_lock:
            if self.pool:
                resource = self.pool.pop()
            else:
                resource = self.factory()
            
            self.in_use.add(id(resource))
            return resource
    
    def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        with self._pool_lock:
            if id(resource) in self.in_use:
                self.in_use.remove(id(resource))
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
                else:
                    # Pool is full, dispose of resource
                    if hasattr(resource, 'close'):
                        try:
                            resource.close()
                        except Exception:
                            pass
        
        self._lock.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._pool_lock:
            return {
                'pool_size': len(self.pool),
                'in_use': len(self.in_use),
                'max_size': self.max_size,
                'available': len(self.pool)
            }


class BatchProcessor:
    """Intelligent batch processing for experiments."""
    
    def __init__(self, batch_size: int = 10, timeout: float = 60.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_items: List[Any] = []
        self.batch_results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.executor = ConcurrentExecutor()
        
    def add_item(self, item: Any, item_id: str) -> None:
        """Add item to batch queue."""
        with self._lock:
            self.pending_items.append((item, item_id))
            
            # Process batch if full
            if len(self.pending_items) >= self.batch_size:
                self._process_batch()
    
    def _process_batch(self) -> None:
        """Process current batch of items."""
        if not self.pending_items:
            return
        
        batch = self.pending_items[:self.batch_size]
        self.pending_items = self.pending_items[self.batch_size:]
        
        # Process batch concurrently
        tasks = [(self._process_single_item, (item,), {}) for item, _ in batch]
        task_ids = self.executor.submit_batch(tasks)
        results = self.executor.wait_for_batch(task_ids, self.timeout)
        
        # Store results
        for (item, item_id), result in zip(batch, results):
            self.batch_results[item_id] = result
    
    def _process_single_item(self, item: Any) -> Any:
        """Process a single item (override in subclass)."""
        # Default implementation - just return the item
        return item
    
    def get_result(self, item_id: str) -> Optional[Any]:
        """Get result for processed item."""
        with self._lock:
            return self.batch_results.get(item_id)
    
    def force_process(self) -> None:
        """Force process remaining items in queue."""
        with self._lock:
            if self.pending_items:
                self._process_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                'pending_items': len(self.pending_items),
                'completed_items': len(self.batch_results),
                'batch_size': self.batch_size,
                'executor_stats': self.executor.get_performance_stats()
            }


class AutoScaler:
    """Auto-scaling for lab operations based on load."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.load_history: List[float] = []
        self.scaling_decisions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def report_load(self, queue_size: int, active_experiments: int) -> None:
        """Report current system load."""
        # Calculate load as ratio of work to capacity
        load = (queue_size + active_experiments) / max(self.current_workers, 1)
        
        with self._lock:
            self.load_history.append(load)
            if len(self.load_history) > 100:  # Keep last 100 measurements
                self.load_history = self.load_history[-50:]
            
            # Make scaling decision
            self._make_scaling_decision(load)
    
    def _make_scaling_decision(self, current_load: float) -> None:
        """Make auto-scaling decision based on load."""
        if len(self.load_history) < 5:
            return  # Need more data
        
        avg_load = sum(self.load_history[-5:]) / 5  # Average of last 5 measurements
        
        decision = {
            'timestamp': datetime.now(),
            'current_load': current_load,
            'avg_load': avg_load,
            'current_workers': self.current_workers,
            'action': 'none'
        }
        
        # Scale up if high load
        if avg_load > 2.0 and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 2, self.max_workers)
            decision['action'] = 'scale_up'
            decision['new_workers'] = new_workers
            self.current_workers = new_workers
            logger.info(f"Scaling up to {new_workers} workers (load: {avg_load:.2f})")
        
        # Scale down if low load
        elif avg_load < 0.5 and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
            decision['action'] = 'scale_down'
            decision['new_workers'] = new_workers
            self.current_workers = new_workers
            logger.info(f"Scaling down to {new_workers} workers (load: {avg_load:.2f})")
        
        self.scaling_decisions.append(decision)
        
        # Keep decision history limited
        if len(self.scaling_decisions) > 1000:
            self.scaling_decisions = self.scaling_decisions[-500:]
    
    def get_recommended_workers(self) -> int:
        """Get current recommended number of workers."""
        return self.current_workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self._lock:
            recent_decisions = self.scaling_decisions[-10:] if self.scaling_decisions else []
            scale_ups = sum(1 for d in recent_decisions if d['action'] == 'scale_up')
            scale_downs = sum(1 for d in recent_decisions if d['action'] == 'scale_down')
            
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'avg_load': sum(self.load_history[-10:]) / len(self.load_history[-10:]) if self.load_history else 0,
                'recent_scale_ups': scale_ups,
                'recent_scale_downs': scale_downs,
                'total_decisions': len(self.scaling_decisions)
            }


# Global instances
_global_executor = ConcurrentExecutor()
_global_autoscaler = AutoScaler()


def get_global_cache() -> AdaptiveCache:
    """Get global cache instance."""
    return _global_cache


def get_global_executor() -> ConcurrentExecutor:
    """Get global executor instance."""
    return _global_executor


def get_global_autoscaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    return _global_autoscaler
