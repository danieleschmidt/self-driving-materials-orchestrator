"""ResultsTracker: JSON log of all experiments and Pareto front tracking."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ResultsTracker:
    """Track experiment results and maintain Pareto-optimal front."""

    def __init__(self, log_path: Optional[str] = None, objectives: List[str] = None):
        """
        Args:
            log_path: optional JSON file path for persistence
            objectives: list of property names to track in Pareto front
        """
        self.log_path = Path(log_path) if log_path else None
        self.objectives = objectives or ["conductivity", "strength"]
        self._experiments: List[Dict] = []

        if self.log_path and self.log_path.exists():
            self._load()

    def log(self, params: Dict[str, float], results: Dict[str, float], iteration: int = 0) -> Dict:
        """Log an experiment."""
        entry = {
            "iteration": iteration,
            "params": dict(params),
            "results": dict(results),
        }
        self._experiments.append(entry)
        if self.log_path:
            self._save()
        return entry

    def get_all(self) -> List[Dict]:
        """Get all logged experiments."""
        return list(self._experiments)

    def count(self) -> int:
        return len(self._experiments)

    def best(self, objective: str) -> Optional[Dict]:
        """Get experiment with best value for a single objective."""
        if not self._experiments:
            return None
        return max(self._experiments, key=lambda e: e["results"].get(objective, float("-inf")))

    def pareto_front(self) -> List[Dict]:
        """Compute the Pareto front (maximizing all objectives)."""
        if not self._experiments:
            return []

        dominated = set()
        n = len(self._experiments)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                r_i = self._experiments[i]["results"]
                r_j = self._experiments[j]["results"]
                j_dominates = all(
                    r_j.get(obj, 0) >= r_i.get(obj, 0) for obj in self.objectives
                ) and any(
                    r_j.get(obj, 0) > r_i.get(obj, 0) for obj in self.objectives
                )
                if j_dominates:
                    dominated.add(i)
                    break

        return [self._experiments[i] for i in range(n) if i not in dominated]

    def summary(self) -> Dict:
        """Compute summary statistics for all results."""
        if not self._experiments:
            return {}
        import numpy as np
        summary = {}
        for obj in self.objectives:
            vals = [e["results"].get(obj, 0) for e in self._experiments]
            summary[obj] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        return summary

    def _save(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self._experiments, f, indent=2)

    def _load(self):
        with open(self.log_path, "r") as f:
            self._experiments = json.load(f)
