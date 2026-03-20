"""ExperimentDesigner: Latin hypercube sampling for parameter space exploration."""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ExperimentDesigner:
    """Design experiments using Latin Hypercube Sampling (LHS)."""

    def __init__(self, parameters: Dict[str, Tuple[float, float]], seed: int = 42):
        """
        Args:
            parameters: dict of {param_name: (min, max)}
            seed: random seed for reproducibility
        """
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.n_params = len(self.param_names)
        self.rng = np.random.default_rng(seed)

    def latin_hypercube_sample(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate n_samples via Latin Hypercube Sampling."""
        n = n_samples
        d = self.n_params

        # LHS: divide each dimension into n equal intervals, pick one sample per interval
        result = np.zeros((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            # Random position within each interval
            u = (perm + self.rng.uniform(size=n)) / n
            lo, hi = self.parameters[self.param_names[j]]
            result[:, j] = lo + u * (hi - lo)

        samples = []
        for i in range(n):
            sample = {self.param_names[j]: float(result[i, j]) for j in range(d)}
            samples.append(sample)
        return samples

    def grid_sample(self, n_per_dim: int = 3) -> List[Dict[str, float]]:
        """Generate a regular grid of samples."""
        grids = []
        for name, (lo, hi) in self.parameters.items():
            grids.append(np.linspace(lo, hi, n_per_dim))

        mesh = np.meshgrid(*grids, indexing="ij")
        flat = [m.flatten() for m in mesh]
        samples = []
        for i in range(len(flat[0])):
            sample = {self.param_names[j]: float(flat[j][i]) for j in range(self.n_params)}
            samples.append(sample)
        return samples

    def suggest_next(
        self, existing_points: List[Dict[str, float]], n: int = 1
    ) -> List[Dict[str, float]]:
        """Suggest next experiments by maximizing minimum distance to existing points."""
        candidates = self.latin_hypercube_sample(max(n * 10, 50))
        if not existing_points:
            return candidates[:n]

        existing = np.array([[p[k] for k in self.param_names] for p in existing_points])
        # Normalize
        ranges = np.array([(hi - lo) for _, (lo, hi) in self.parameters.items()])
        ranges = np.where(ranges == 0, 1, ranges)

        best = []
        for _ in range(n):
            best_dist = -1
            best_cand = None
            for cand in candidates:
                cand_arr = np.array([cand[k] for k in self.param_names]) / ranges
                if len(existing) > 0:
                    dists = np.linalg.norm(existing / ranges - cand_arr, axis=1)
                    min_dist = dists.min()
                else:
                    min_dist = float("inf")
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_cand = cand
            if best_cand:
                best.append(best_cand)
                existing = np.vstack([existing, [best_cand[k] for k in self.param_names]])

        return best
