"""Advanced caching system for materials discovery optimization."""

import hashlib
import json
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .utils import np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""

    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """LRU (Least Recently Used) cache with size and TTL support."""

    def __init__(self, max_size: int = 10000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(f"LRU Cache initialized: max_size={max_size}, ttl={default_ttl}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                self._remove_key(key)
                self.misses += 1
                return None

            # Update access info
            entry.access()
            self._move_to_end(key)
            self.hits += 1

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1  # Fallback

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl,
            )

            # Remove existing entry if present
            if key in self.cache:
                self._remove_key(key)

            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)

            # Ensure size limit
            self._ensure_size_limit()

    def _move_to_end(self, key: str) -> None:
        """Move key to end of access order."""
        if key in self.access_order:
            self.access_order.remove(key)
            self.access_order.append(key)

    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access order."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def _ensure_size_limit(self) -> None:
        """Ensure cache doesn't exceed size limit."""
        while len(self.cache) > self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order[0]
                self._remove_key(lru_key)
                self.evictions += 1

    def clear_expired(self) -> int:
        """Clear expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_key(key)

            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            )

            total_size = sum(entry.size_bytes for entry in self.cache.values())

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_size_bytes": total_size,
            }


class ExperimentResultCache:
    """Specialized cache for experiment results with intelligent key generation."""

    def __init__(self, max_size: int = 50000, ttl: float = 86400):  # 24 hours TTL
        self.lru_cache = LRUCache(max_size=max_size, default_ttl=ttl)
        self.parameter_tolerance = 0.01  # 1% tolerance for parameter matching

        # Predictive caching
        self.prediction_model = None
        self.cached_predictions: Dict[str, float] = {}

        logger.info(f"Experiment cache initialized: max_size={max_size}, ttl={ttl}s")

    def _generate_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key from experiment parameters."""
        # Sort parameters for consistent key generation
        sorted_params = sorted(parameters.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _generate_fuzzy_keys(self, parameters: Dict[str, Any]) -> List[str]:
        """Generate fuzzy keys for approximate matching."""
        keys = []

        # Generate keys with parameter variations within tolerance
        for tolerance in [0.005, 0.01, 0.02]:  # 0.5%, 1%, 2%
            for param_name, param_value in parameters.items():
                if isinstance(param_value, (int, float)):
                    # Create variations
                    variations = [
                        param_value * (1 + tolerance),
                        param_value * (1 - tolerance),
                    ]

                    for variation in variations:
                        fuzzy_params = parameters.copy()
                        fuzzy_params[param_name] = variation
                        keys.append(self._generate_key(fuzzy_params))

        return keys

    def get_experiment_result(
        self, parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get experiment result from cache."""
        # Try exact match first
        key = self._generate_key(parameters)
        result = self.lru_cache.get(key)

        if result is not None:
            logger.debug(f"Cache hit (exact): {key[:8]}...")
            return result

        # Try fuzzy matching
        fuzzy_keys = self._generate_fuzzy_keys(parameters)
        for fuzzy_key in fuzzy_keys:
            result = self.lru_cache.get(fuzzy_key)
            if result is not None:
                logger.debug(f"Cache hit (fuzzy): {fuzzy_key[:8]}...")
                return result

        return None

    def cache_experiment_result(
        self, parameters: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Cache experiment result."""
        key = self._generate_key(parameters)

        # Add metadata
        cached_result = {
            "parameters": parameters,
            "results": result,
            "cached_at": datetime.now().isoformat(),
        }

        self.lru_cache.put(key, cached_result)
        logger.debug(f"Cached result: {key[:8]}...")

    def precompute_promising_regions(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective: str,
        n_samples: int = 1000,
    ) -> None:
        """Precompute results for promising parameter regions."""
        if not self.prediction_model:
            logger.warning("No prediction model available for precomputation")
            return

        logger.info(f"Precomputing {n_samples} experiments in promising regions...")

        # Generate samples in promising regions
        samples = self._sample_promising_regions(parameter_space, n_samples)

        # Predict and cache results
        for sample in samples:
            if self.get_experiment_result(sample) is None:  # Don't overwrite existing
                predicted_result = self._predict_experiment_result(sample, objective)
                if predicted_result:
                    self.cache_experiment_result(sample, predicted_result)

    def _sample_promising_regions(
        self, parameter_space: Dict[str, Tuple[float, float]], n_samples: int
    ) -> List[Dict[str, float]]:
        """Sample parameters from promising regions."""
        samples = []

        # Use Latin hypercube sampling for good coverage
        from scipy.stats import qmc

        param_names = list(parameter_space.keys())
        bounds = [parameter_space[name] for name in param_names]

        sampler = qmc.LatinHypercube(d=len(param_names))
        unit_samples = sampler.random(n_samples)

        # Scale to actual bounds
        for unit_sample in unit_samples:
            sample = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = bounds[i]
                sample[param_name] = min_val + unit_sample[i] * (max_val - min_val)
            samples.append(sample)

        return samples

    def _predict_experiment_result(
        self, parameters: Dict[str, Any], objective: str
    ) -> Optional[Dict[str, Any]]:
        """Predict experiment result using ML model."""
        # Placeholder for ML prediction
        # In a real implementation, this would use trained models

        # Simple heuristic for demonstration
        if objective == "band_gap":
            temp = parameters.get("temperature", 150)
            conc = parameters.get("concentration", 1.0)

            # Synthetic prediction based on parameter values
            predicted_bandgap = (
                1.5 - 0.001 * temp + 0.1 * conc + np.random.normal(0, 0.05)
            )
            predicted_efficiency = max(0, 25 - abs(predicted_bandgap - 1.4) * 50)

            return {
                "band_gap": max(0.5, min(3.0, predicted_bandgap)),
                "efficiency": max(0, min(30, predicted_efficiency)),
                "stability": 0.8 + np.random.normal(0, 0.1),
                "predicted": True,
            }

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        base_stats = self.lru_cache.stats()
        base_stats["parameter_tolerance"] = self.parameter_tolerance
        base_stats["prediction_cache_size"] = len(self.cached_predictions)

        return base_stats


class MultiLevelCache:
    """Multi-level caching system with L1 (memory) and L2 (persistent) caches."""

    def __init__(self, l1_size: int = 1000, l2_size: int = 10000):
        # L1 cache - fast memory cache
        self.l1_cache = LRUCache(max_size=l1_size, default_ttl=3600)  # 1 hour

        # L2 cache - larger memory cache
        self.l2_cache = LRUCache(max_size=l2_size, default_ttl=86400)  # 24 hours

        logger.info(f"Multi-level cache initialized: L1={l1_size}, L2={l2_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in multi-level cache."""
        # Always put in L1
        self.l1_cache.put(key, value)

        # Also put in L2 for persistence
        self.l2_cache.put(key, value)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = self.l1_cache.stats()
        l2_stats = self.l2_cache.stats()

        return {
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "total_hits": l1_stats["hits"] + l2_stats["hits"],
            "total_misses": l1_stats["misses"] + l2_stats["misses"],
            "combined_hit_rate": (
                (l1_stats["hits"] + l2_stats["hits"])
                / (
                    l1_stats["hits"]
                    + l2_stats["hits"]
                    + l1_stats["misses"]
                    + l2_stats["misses"]
                )
                if (
                    l1_stats["hits"]
                    + l2_stats["hits"]
                    + l1_stats["misses"]
                    + l2_stats["misses"]
                )
                > 0
                else 0
            ),
        }


# Global cache instances
_global_experiment_cache = None
_global_multi_level_cache = None


def get_global_experiment_cache() -> ExperimentResultCache:
    """Get global experiment result cache."""
    global _global_experiment_cache
    if _global_experiment_cache is None:
        _global_experiment_cache = ExperimentResultCache()
    return _global_experiment_cache


def get_global_multi_level_cache() -> MultiLevelCache:
    """Get global multi-level cache."""
    global _global_multi_level_cache
    if _global_multi_level_cache is None:
        _global_multi_level_cache = MultiLevelCache()
    return _global_multi_level_cache
