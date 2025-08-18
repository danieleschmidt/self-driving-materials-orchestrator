"""Experiment planning algorithms for autonomous discovery."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from .utils import (
    NUMPY_AVAILABLE,
    RBF,
    SKLEARN_AVAILABLE,
    GaussianProcessRegressor,
    Matern,
    np,
)

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    from sklearn.gaussian_process.kernels import ConstantKernel

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using simplified Bayesian optimization")
    SKLEARN_AVAILABLE = False


class BasePlanner(ABC):
    """Base class for experiment planning algorithms."""

    @abstractmethod
    def suggest_next(
        self,
        n_suggestions: int,
        param_space: Dict[str, tuple],
        previous_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Suggest next experiments to run."""
        pass


class BayesianPlanner(BasePlanner):
    """Bayesian optimization-based experiment planner with Gaussian Process."""

    def __init__(
        self,
        acquisition_function: str = "expected_improvement",
        batch_size: int = 1,
        kernel: str = "matern",
        exploration_factor: float = 0.1,
        target_property: str = "band_gap",
    ):
        """Initialize Bayesian planner.

        Args:
            acquisition_function: Acquisition function to use
            batch_size: Number of experiments to suggest at once
            kernel: Gaussian process kernel type
            exploration_factor: Balance exploration vs exploitation
            target_property: Property to optimize
        """
        self.acquisition_function = acquisition_function
        self.batch_size = batch_size
        self.kernel = kernel
        self.exploration_factor = exploration_factor
        self.target_property = target_property
        self.gp_model = None
        self._param_names = []

    def _prepare_data(
        self, previous_results: List[Dict[str, Any]], param_space: Dict[str, tuple]
    ):
        """Prepare training data for Gaussian Process."""
        if not previous_results:
            return None, None

        self._param_names = list(param_space.keys())
        X, y = [], []

        for result in previous_results:
            if self.target_property in result:
                x_row = [result.get(param_name, 0) for param_name in self._param_names]
                X.append(x_row)
                y.append(result[self.target_property])

        if not NUMPY_AVAILABLE:
            return X if X else None, y if y else None
        return np.array(X) if X else None, np.array(y) if y else None

    def _create_gp_model(self):
        """Create Gaussian Process model."""
        if not SKLEARN_AVAILABLE:
            return None

        if self.kernel == "matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42,
        )

    def _acquisition_expected_improvement(self, X_candidate: Any, y_train: Any) -> Any:
        """Calculate Expected Improvement acquisition function."""
        y_pred, y_std = self.gp_model.predict(X_candidate, return_std=True)
        y_best = np.max(y_train)
        xi = self.exploration_factor

        with np.errstate(divide="ignore"):
            imp = y_pred - y_best - xi
            Z = imp / y_std
            ei = imp * norm.cdf(Z) + y_std * norm.pdf(Z)
            ei[y_std == 0.0] = 0.0

        return ei

    def suggest_next(
        self,
        n_suggestions: int,
        param_space: Dict[str, tuple],
        previous_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Suggest next experiments using Bayesian optimization."""
        if not NUMPY_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.info("Missing dependencies, using random sampling")
            return RandomPlanner().suggest_next(
                n_suggestions, param_space, previous_results
            )

        X_train, y_train = self._prepare_data(previous_results, param_space)

        if X_train is None or len(X_train) < 3:
            logger.info("Using random sampling (insufficient data)")
            return RandomPlanner().suggest_next(
                n_suggestions, param_space, previous_results
            )

        # Fit GP model
        self.gp_model = self._create_gp_model()
        try:
            self.gp_model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}")
            return RandomPlanner().suggest_next(
                n_suggestions, param_space, previous_results
            )

        suggestions = []
        for _ in range(n_suggestions):
            # Generate candidates
            candidates = []
            for _ in range(1000):
                candidate = [
                    np.random.uniform(low, high) for low, high in param_space.values()
                ]
                candidates.append(candidate)

            candidates = np.array(candidates)

            # Find best candidate using acquisition function
            if self.acquisition_function == "expected_improvement":
                acq_values = self._acquisition_expected_improvement(candidates, y_train)
            else:  # Upper confidence bound
                y_pred, y_std = self.gp_model.predict(candidates, return_std=True)
                acq_values = y_pred + self.exploration_factor * y_std

            best_idx = np.argmax(acq_values)
            best_candidate = candidates[best_idx]

            # Convert to dict
            suggestion = {
                name: float(val) for name, val in zip(self._param_names, best_candidate)
            }
            suggestions.append(suggestion)

        return suggestions


class RandomPlanner(BasePlanner):
    """Random sampling experiment planner."""

    def suggest_next(
        self,
        n_suggestions: int,
        param_space: Dict[str, tuple],
        previous_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Suggest random experiments."""
        import random

        suggestions = []
        for _ in range(n_suggestions):
            suggestion = {}
            for param, (low, high) in param_space.items():
                suggestion[param] = random.uniform(low, high)
            suggestions.append(suggestion)

        return suggestions


class GridPlanner(BasePlanner):
    """Grid search experiment planner."""

    def __init__(self, grid_density: int = 10):
        """Initialize grid planner.

        Args:
            grid_density: Number of points per dimension
        """
        self.grid_density = grid_density

    def suggest_next(
        self,
        n_suggestions: int,
        param_space: Dict[str, tuple],
        previous_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Suggest experiments from grid points."""
        import itertools
        import random

        # Generate grid points
        param_grids = []
        param_names = list(param_space.keys())

        for param, (low, high) in param_space.items():
            if NUMPY_AVAILABLE:
                param_grids.append(np.linspace(low, high, self.grid_density))
            else:
                # Fallback: manual linspace
                step = (high - low) / (self.grid_density - 1)
                grid = [low + i * step for i in range(self.grid_density)]
                param_grids.append(grid)

        # Create all combinations
        grid_points = list(itertools.product(*param_grids))

        # Select random subset
        selected_points = random.sample(
            grid_points, min(n_suggestions, len(grid_points))
        )

        # Convert to dict format
        suggestions = []
        for point in selected_points:
            suggestion = dict(zip(param_names, point))
            suggestions.append(suggestion)

        return suggestions
