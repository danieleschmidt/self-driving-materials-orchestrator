"""High-performance computing optimizations for materials discovery."""

import logging
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
import concurrent.futures
import psutil
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    task_throughput: float = 0.0
    average_task_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0

class HighPerformanceOptimizer:
    """Optimize system performance for materials discovery workloads."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.metrics = PerformanceMetrics()
        self.optimization_history = []
        self.performance_targets = {
            'cpu_usage_target': 80.0,
            'memory_usage_target': 75.0,
            'task_throughput_target': 10.0,  # tasks per second
            'cache_hit_rate_target': 80.0
        }
        
        # Resource monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._optimization_thread = None
        
        # Adaptive settings
        self.adaptive_batch_size = 10
        self.adaptive_worker_count = mp.cpu_count()
        self.adaptive_cache_size = 1000
        
        # Performance caches
        self.result_cache = {}
        self.computation_cache = {}
        self.max_cache_size = 10000
        
        logger.info("High-performance optimizer initialized")
    
    def start_optimization(self):
        """Start continuous performance optimization."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self._optimization_thread = threading.Thread(target=self._optimize_performance, daemon=True)
        
        self._monitor_thread.start()
        self._optimization_thread.start()
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self._monitoring_active = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)
        
        logger.info("Performance optimization stopped")
    
    def _monitor_performance(self):
        """Monitor system performance metrics."""
        while self._monitoring_active:
            try:
                # CPU usage
                self.metrics.cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.memory_usage = memory.percent
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if hasattr(self, '_last_disk_io'):
                    self.metrics.disk_io = (
                        disk_io.read_bytes + disk_io.write_bytes - 
                        self._last_disk_io.read_bytes - self._last_disk_io.write_bytes
                    ) / 1024 / 1024  # MB/s
                self._last_disk_io = disk_io
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_io'):
                    self.metrics.network_io = (
                        net_io.bytes_sent + net_io.bytes_recv - 
                        self._last_net_io.bytes_sent - self._last_net_io.bytes_recv
                    ) / 1024 / 1024  # MB/s
                self._last_net_io = net_io
                
                # Cache hit rate
                total_cache_requests = getattr(self, '_cache_hits', 0) + getattr(self, '_cache_misses', 0)
                if total_cache_requests > 0:
                    self.metrics.cache_hit_rate = (getattr(self, '_cache_hits', 0) / total_cache_requests) * 100
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _optimize_performance(self):
        """Continuously optimize performance."""
        while self._monitoring_active:
            try:
                # CPU optimization
                if self.metrics.cpu_usage < 60:
                    self._scale_up_workers()
                elif self.metrics.cpu_usage > 90:
                    self._scale_down_workers()
                
                # Memory optimization
                if self.metrics.memory_usage > 85:
                    self._optimize_memory_usage()
                
                # Cache optimization
                if self.metrics.cache_hit_rate < 70:
                    self._optimize_cache_strategy()
                
                # Batch size optimization
                self._optimize_batch_size()
                
                time.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                time.sleep(60)
    
    def _scale_up_workers(self):
        """Increase number of workers."""
        max_workers = min(mp.cpu_count() * 2, 32)  # Cap at 32 workers
        if self.adaptive_worker_count < max_workers:
            self.adaptive_worker_count += 1
            logger.info(f"Scaled up to {self.adaptive_worker_count} workers")
    
    def _scale_down_workers(self):
        """Decrease number of workers."""
        min_workers = max(1, mp.cpu_count() // 2)
        if self.adaptive_worker_count > min_workers:
            self.adaptive_worker_count -= 1
            logger.info(f"Scaled down to {self.adaptive_worker_count} workers")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage."""
        # Clear caches
        if len(self.result_cache) > self.max_cache_size // 2:
            # Remove oldest entries
            oldest_keys = list(self.result_cache.keys())[:len(self.result_cache) // 4]
            for key in oldest_keys:
                del self.result_cache[key]
        
        if len(self.computation_cache) > self.max_cache_size // 2:
            oldest_keys = list(self.computation_cache.keys())[:len(self.computation_cache) // 4]
            for key in oldest_keys:
                del self.computation_cache[key]
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory optimization performed")
    
    def _optimize_cache_strategy(self):
        """Optimize caching strategy."""
        # Increase cache size if hit rate is low
        if self.metrics.cache_hit_rate < 50:
            self.adaptive_cache_size = min(self.adaptive_cache_size * 1.2, self.max_cache_size)
        
        logger.info(f"Cache strategy optimized: size={self.adaptive_cache_size}")
    
    def _optimize_batch_size(self):
        """Optimize batch size based on performance."""
        # Increase batch size if CPU usage is low
        if self.metrics.cpu_usage < 50 and self.adaptive_batch_size < 50:
            self.adaptive_batch_size = min(self.adaptive_batch_size + 2, 50)
        # Decrease if CPU usage is too high
        elif self.metrics.cpu_usage > 90 and self.adaptive_batch_size > 5:
            self.adaptive_batch_size = max(self.adaptive_batch_size - 2, 5)
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance tracking."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.optimization_history.append({
                'operation': operation_name,
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': datetime.now()
            })
            
            logger.debug(f"Performance: {operation_name} took {duration:.2f}s, memory delta: {memory_delta:.1f}MB")
    
    def cache_result(self, key: str, result: Any) -> Any:
        """Cache computation result with intelligent eviction."""
        if len(self.result_cache) >= self.adaptive_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[key] = {
            'result': result,
            'timestamp': time.time(),
            'access_count': 1
        }
        
        return result
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        if key in self.result_cache:
            entry = self.result_cache[key]
            entry['access_count'] += 1
            entry['last_access'] = time.time()
            
            # Track cache hit
            self._cache_hits = getattr(self, '_cache_hits', 0) + 1
            
            return entry['result']
        
        # Track cache miss
        self._cache_misses = getattr(self, '_cache_misses', 0) + 1
        return None
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    max_workers: Optional[int] = None) -> List[Any]:
        """High-performance parallel mapping."""
        if not items:
            return []
        
        max_workers = max_workers or self.adaptive_worker_count
        
        # Use adaptive batch size for better performance
        if len(items) > self.adaptive_batch_size:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Split into batches
                batches = [items[i:i + self.adaptive_batch_size] 
                          for i in range(0, len(items), self.adaptive_batch_size)]
                
                # Process batches in parallel
                batch_futures = [executor.submit(self._process_batch, func, batch) 
                               for batch in batches]
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(batch_futures):
                    results.extend(future.result())
                
                return results
        else:
            # Small dataset, process directly
            return [func(item) for item in items]
    
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items."""
        return [func(item) for item in batch]
    
    def optimize_numpy_operations(self):
        """Optimize NumPy operations."""
        # Set optimal thread count for NumPy
        import os
        os.environ['MKL_NUM_THREADS'] = str(min(4, mp.cpu_count()))
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(4, mp.cpu_count()))
        os.environ['OMP_NUM_THREADS'] = str(min(4, mp.cpu_count()))
        
        # Use single-threaded BLAS for better control
        if hasattr(np, '__config__'):
            logger.info(f"NumPy configuration optimized for {mp.cpu_count()} cores")
    
    def memory_efficient_operation(self, operation: Callable, data: Any, 
                                 chunk_size: Optional[int] = None) -> Any:
        """Perform memory-efficient operations on large datasets."""
        if not chunk_size:
            # Adaptive chunk size based on available memory
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            chunk_size = max(100, int(available_memory / 100))  # Conservative estimate
        
        if hasattr(data, '__len__') and len(data) > chunk_size:
            # Process in chunks
            results = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                chunk_result = operation(chunk)
                results.append(chunk_result)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine results
            if isinstance(results[0], np.ndarray):
                return np.concatenate(results)
            elif isinstance(results[0], list):
                combined = []
                for r in results:
                    combined.extend(r)
                return combined
            else:
                return results
        else:
            return operation(data)
    
    def benchmark_operation(self, operation: Callable, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
        """Benchmark an operation."""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            operation(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage)
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if self.metrics.cpu_usage < 50:
            recommendations.append("Consider increasing parallelism - CPU underutilized")
        
        if self.metrics.memory_usage > 80:
            recommendations.append("High memory usage detected - consider reducing cache size")
        
        if self.metrics.cache_hit_rate < 60:
            recommendations.append("Low cache hit rate - review caching strategy")
        
        if self.metrics.task_throughput < 5:
            recommendations.append("Low task throughput - optimize algorithms or increase workers")
        
        # Analyze optimization history
        if len(self.optimization_history) > 10:
            recent_durations = [op['duration'] for op in self.optimization_history[-10:]]
            if np.mean(recent_durations) > 5.0:
                recommendations.append("Long operation durations detected - consider optimization")
        
        return recommendations
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'current_metrics': {
                'cpu_usage': f"{self.metrics.cpu_usage:.1f}%",
                'memory_usage': f"{self.metrics.memory_usage:.1f}%",
                'cache_hit_rate': f"{self.metrics.cache_hit_rate:.1f}%",
                'task_throughput': f"{self.metrics.task_throughput:.2f} tasks/sec"
            },
            'adaptive_settings': {
                'worker_count': self.adaptive_worker_count,
                'batch_size': self.adaptive_batch_size,
                'cache_size': self.adaptive_cache_size
            },
            'cache_statistics': {
                'result_cache_size': len(self.result_cache),
                'computation_cache_size': len(self.computation_cache),
                'cache_hits': getattr(self, '_cache_hits', 0),
                'cache_misses': getattr(self, '_cache_misses', 0)
            },
            'optimization_history': self.optimization_history[-10:],  # Last 10 operations
            'recommendations': self.get_optimization_recommendations(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'total_memory': f"{psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB",
                'available_memory': f"{psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB"
            }
        }

class GPUAccelerator:
    """GPU acceleration for compute-intensive operations."""
    
    def __init__(self):
        """Initialize GPU accelerator."""
        self.gpu_available = self._check_gpu_availability()
        self.gpu_memory_limit = None
        
        if self.gpu_available:
            self._initialize_gpu()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            logger.info("CuPy not available - GPU acceleration disabled")
            return False
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _initialize_gpu(self):
        """Initialize GPU for computation."""
        try:
            import cupy as cp
            
            # Set memory pool
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Get GPU memory info
            gpu_memory = cp.cuda.Device().mem_info
            total_memory = gpu_memory[1] / 1024 / 1024  # MB
            self.gpu_memory_limit = int(total_memory * 0.8)  # Use 80% of GPU memory
            
            logger.info(f"GPU acceleration initialized - {total_memory:.0f}MB available")
            
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self.gpu_available = False
    
    def accelerate_array_operations(self, operation: Callable, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Accelerate array operations using GPU."""
        if not self.gpu_available:
            return [operation(arr) for arr in arrays]
        
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_arrays = [cp.asarray(arr) for arr in arrays]
            
            # Perform operations on GPU
            gpu_results = [operation(arr) for arr in gpu_arrays]
            
            # Transfer back to CPU
            cpu_results = [cp.asnumpy(result) for result in gpu_results]
            
            return cpu_results
            
        except Exception as e:
            logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
            return [operation(arr) for arr in arrays]
    
    def accelerate_matrix_operations(self, matrices: List[np.ndarray], 
                                   operation: str = "multiply") -> np.ndarray:
        """Accelerate matrix operations."""
        if not self.gpu_available or len(matrices) < 2:
            # Fallback to NumPy
            if operation == "multiply":
                result = matrices[0]
                for mat in matrices[1:]:
                    result = np.dot(result, mat)
                return result
            return matrices[0]
        
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_matrices = [cp.asarray(mat) for mat in matrices]
            
            # Perform matrix operations
            if operation == "multiply":
                result = gpu_matrices[0]
                for mat in gpu_matrices[1:]:
                    result = cp.dot(result, mat)
            else:
                result = gpu_matrices[0]
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except Exception as e:
            logger.warning(f"GPU matrix operations failed: {e}")
            # Fallback to CPU
            if operation == "multiply":
                result = matrices[0]
                for mat in matrices[1:]:
                    result = np.dot(result, mat)
                return result
            return matrices[0]

# Global performance optimizer
_global_optimizer = None
_global_gpu_accelerator = None

def get_global_optimizer() -> HighPerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = HighPerformanceOptimizer()
        _global_optimizer.start_optimization()
    return _global_optimizer

def get_global_gpu_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance."""
    global _global_gpu_accelerator
    if _global_gpu_accelerator is None:
        _global_gpu_accelerator = GPUAccelerator()
    return _global_gpu_accelerator