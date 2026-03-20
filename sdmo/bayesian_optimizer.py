"""BayesianOptimizer: GP surrogate + Expected Improvement acquisition (numpy-based)."""

import numpy as np
from typing import Dict, List, Optional, Tuple


class GaussianProcessSurrogate:
    """Simple RBF-kernel Gaussian Process for regression."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared exponential (RBF) kernel."""
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self._rbf_kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std at new points."""
        if self.X_train is None:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        K_star = self._rbf_kernel(X_new, self.X_train)
        K_star_star = self._rbf_kernel(X_new, X_new)

        mu = K_star @ self.K_inv @ self.y_train
        cov = K_star_star - K_star @ self.K_inv @ K_star.T
        std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        return mu, std


class BayesianOptimizer:
    """GP surrogate + Expected Improvement acquisition for black-box optimization."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        objective: str = "conductivity",
        xi: float = 0.01,
        seed: int = 42,
    ):
        """
        Args:
            param_bounds: {param_name: (min, max)}
            objective: which property to optimize
            xi: exploration-exploitation tradeoff for EI
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.objective = objective
        self.xi = xi
        self.rng = np.random.default_rng(seed)
        self.gp = GaussianProcessSurrogate()

        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[k] for k in self.param_names])

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        return {k: float(arr[i]) for i, k in enumerate(self.param_names)}

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize to [0,1]."""
        lo = np.array([self.param_bounds[k][0] for k in self.param_names])
        hi = np.array([self.param_bounds[k][1] for k in self.param_names])
        return (X - lo) / (hi - lo + 1e-10)

    def update(self, params: Dict[str, float], result: Dict[str, float]):
        """Add a new observation."""
        x = self._params_to_array(params)
        y = result.get(self.objective, 0.0)
        self.X_observed.append(x)
        self.y_observed.append(y)

        if len(self.X_observed) >= 2:
            X = self._normalize(np.array(self.X_observed))
            y_arr = np.array(self.y_observed)
            self.gp.fit(X, y_arr)

    def expected_improvement(self, X_candidates: np.ndarray) -> np.ndarray:
        """Compute EI for candidate points."""
        if len(self.y_observed) < 2:
            return self.rng.uniform(size=len(X_candidates))

        X_norm = self._normalize(X_candidates)
        mu, sigma = self.gp.predict(X_norm)
        y_best = max(self.y_observed)

        Z = (mu - y_best - self.xi) / (sigma + 1e-10)
        from scipy.stats import norm as scipy_norm
        try:
            ei = (mu - y_best - self.xi) * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)
        except ImportError:
            # Manual normal CDF/PDF approximation if scipy not available
            def _norm_cdf(z):
                return 0.5 * (1 + np.sign(z) * np.sqrt(1 - np.exp(-2 * z**2 / np.pi)))
            def _norm_pdf(z):
                return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            ei = (mu - y_best - self.xi) * _norm_cdf(Z) + sigma * _norm_pdf(Z)

        return np.maximum(ei, 0)

    def suggest(self, n_candidates: int = 200) -> Dict[str, float]:
        """Suggest next experiment via EI maximization."""
        lo = np.array([self.param_bounds[k][0] for k in self.param_names])
        hi = np.array([self.param_bounds[k][1] for k in self.param_names])
        candidates = self.rng.uniform(lo, hi, size=(n_candidates, len(self.param_names)))

        ei = self.expected_improvement(candidates)
        best_idx = np.argmax(ei)
        return self._array_to_params(candidates[best_idx])

    def best_observed(self) -> Optional[Tuple[Dict[str, float], float]]:
        """Return the best (params, value) observed so far."""
        if not self.y_observed:
            return None
        best_idx = int(np.argmax(self.y_observed))
        return self._array_to_params(self.X_observed[best_idx]), self.y_observed[best_idx]
