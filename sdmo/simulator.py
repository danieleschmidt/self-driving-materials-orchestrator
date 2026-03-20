"""MaterialsSimulator: evaluate synthetic material properties via polynomial model."""

import numpy as np
from typing import Dict, List, Optional


class MaterialsSimulator:
    """
    Simulate material properties using a polynomial surrogate model.
    
    Properties computed:
    - conductivity: electrical conductivity (S/m)
    - strength: mechanical tensile strength (MPa)
    - stability: thermal stability score (0-1)
    """

    def __init__(self, noise_level: float = 0.05, seed: int = 42):
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def _conductivity(self, x: float, y: float, z: float = 0.5) -> float:
        """Polynomial model for electrical conductivity."""
        return (
            10.0 * x ** 2
            - 5.0 * y ** 2
            + 8.0 * x * y
            - 3.0 * z
            + 2.0 * x
            + 15.0
        )

    def _strength(self, x: float, y: float, z: float = 0.5) -> float:
        """Polynomial model for tensile strength."""
        return (
            -6.0 * x ** 2
            + 4.0 * y ** 2
            + 3.0 * x * y
            + 12.0 * x
            - 2.0 * y
            + 8.0 * z
            + 50.0
        )

    def _stability(self, x: float, y: float, z: float = 0.5) -> float:
        """Model thermal stability as a sigmoid-like score."""
        raw = -(x - 0.5) ** 2 - (y - 0.5) ** 2 + 0.5 * z + 0.5
        return float(np.clip(raw, 0.0, 1.0))

    def simulate(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate material properties for a parameter set.
        
        Args:
            params: dict with keys like 'composition_x', 'temperature', 'pressure'
        
        Returns:
            dict with properties: conductivity, strength, stability
        """
        keys = sorted(params.keys())
        vals = [params[k] for k in keys]

        # Normalize to [0,1] range for model evaluation
        x = float(np.clip(vals[0] if len(vals) > 0 else 0.5, 0, 1))
        y = float(np.clip(vals[1] if len(vals) > 1 else 0.5, 0, 1))
        z = float(np.clip(vals[2] if len(vals) > 2 else 0.5, 0, 1))

        noise = self.rng.normal(0, self.noise_level, 3)

        conductivity = float(np.clip(self._conductivity(x, y, z) + noise[0], 0, 100))
        strength = float(np.clip(self._strength(x, y, z) + noise[1], 0, 200))
        stability = float(np.clip(self._stability(x, y, z) + noise[2] * 0.1, 0, 1))

        return {
            "conductivity": conductivity,
            "strength": strength,
            "stability": stability,
        }

    def batch_simulate(self, param_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Simulate a batch of parameter sets."""
        return [self.simulate(p) for p in param_list]
