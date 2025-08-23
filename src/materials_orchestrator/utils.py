"""Utility functions and fallback implementations for materials orchestrator."""

import logging
import math

logger = logging.getLogger(__name__)

# Numpy fallback implementation
try:
    import numpy as np

    NUMPY_AVAILABLE = True
    logger.info("NumPy available - using optimized numerical operations")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - using fallback implementations")

    class MockNumpy:
        """Mock numpy implementation with basic functionality."""

        @staticmethod
        def array(x):
            """Convert to array-like structure."""
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
                return list(x)
            return [x]

        @staticmethod
        def mean(x):
            """Calculate mean of array."""
            if not x:
                return 0.0
            return sum(x) / len(x)

        @staticmethod
        def std(x):
            """Calculate standard deviation."""
            if not x:
                return 0.0
            mean_val = sum(x) / len(x)
            variance = sum((i - mean_val) ** 2 for i in x) / len(x)
            return variance**0.5

        @staticmethod
        def var(x):
            """Calculate variance."""
            if not x:
                return 0.0
            mean_val = sum(x) / len(x)
            return sum((i - mean_val) ** 2 for i in x) / len(x)

        @staticmethod
        def corrcoef(x, y):
            """Calculate correlation coefficient matrix."""
            if not x or not y or len(x) != len(y):
                return MockArray([[0.0, 0.0], [0.0, 0.0]])

            n = len(x)
            if n == 1:
                return MockArray([[1.0, 0.0], [0.0, 1.0]])

            mean_x = sum(x) / n
            mean_y = sum(y) / n

            # Calculate covariance and standard deviations
            cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
            std_x = (sum((xi - mean_x) ** 2 for xi in x) / (n - 1)) ** 0.5
            std_y = (sum((yi - mean_y) ** 2 for yi in y) / (n - 1)) ** 0.5

            if std_x == 0 or std_y == 0:
                corr = 0.0
            else:
                corr = cov / (std_x * std_y)

            return MockArray([[1.0, corr], [corr, 1.0]])

        @staticmethod
        def isnan(x):
            """Check if value is NaN."""
            if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                return [math.isnan(float(v)) if isinstance(v, (int, float)) else False for v in x]
            try:
                return math.isnan(float(x))
            except (ValueError, TypeError):
                return False


    class MockArray:
        """Mock numpy array that supports 2D indexing."""

        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                row, col = key
                return self.data[row][col]
            return self.data[key]

        def __setitem__(self, key, value):
            if isinstance(key, tuple) and len(key) == 2:
                row, col = key
                self.data[row][col] = value
            else:
                self.data[key] = value

        @staticmethod
        def zeros(shape):
            """Create array of zeros."""
            if isinstance(shape, int):
                return [0.0] * shape
            elif isinstance(shape, (list, tuple)) and len(shape) == 1:
                return [0.0] * shape[0]
            else:
                # For multi-dimensional, return nested lists
                return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]

        @staticmethod
        def ones(shape):
            """Create array of ones."""
            if isinstance(shape, int):
                return [1.0] * shape
            elif isinstance(shape, (list, tuple)) and len(shape) == 1:
                return [1.0] * shape[0]
            else:
                return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]

        @staticmethod
        def random():
            """Mock random module."""
            import random

            class MockRandom:
                @staticmethod
                def rand(*args):
                    if len(args) == 0:
                        return random.random()
                    elif len(args) == 1:
                        return [random.random() for _ in range(args[0])]
                    else:
                        return [
                            [random.random() for _ in range(args[1])]
                            for _ in range(args[0])
                        ]

                @staticmethod
                def normal(loc=0.0, scale=1.0, size=None):
                    import random

                    if size is None:
                        return random.gauss(loc, scale)
                    elif isinstance(size, int):
                        return [random.gauss(loc, scale) for _ in range(size)]
                    else:
                        return [
                            [random.gauss(loc, scale) for _ in range(size[1])]
                            for _ in range(size[0])
                        ]

            return MockRandom()

        @staticmethod
        def exp(x):
            """Exponential function."""
            if isinstance(x, (list, tuple)):
                return [math.exp(val) for val in x]
            return math.exp(x)

        @staticmethod
        def log(x):
            """Natural logarithm."""
            if isinstance(x, (list, tuple)):
                return [math.log(val) for val in x]
            return math.log(x)

        @staticmethod
        def sqrt(x):
            """Square root."""
            if isinstance(x, (list, tuple)):
                return [math.sqrt(val) for val in x]
            return math.sqrt(x)

        @staticmethod
        def sum(x):
            """Sum of array elements."""
            return sum(x)

        @staticmethod
        def max(x):
            """Maximum of array elements."""
            return max(x)

        @staticmethod
        def min(x):
            """Minimum of array elements."""
            return min(x)

        @staticmethod
        def argmax(x):
            """Index of maximum value."""
            return x.index(max(x))

        @staticmethod
        def argmin(x):
            """Index of minimum value."""
            return x.index(min(x))

    np = MockNumpy()

# Scipy fallback
try:
    from scipy import optimize, stats

    SCIPY_AVAILABLE = True
    logger.info("SciPy available - using advanced optimization")
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - using basic optimization")

    class MockOptimize:
        @staticmethod
        def minimize(fun, x0, **kwargs):
            """Basic gradient descent minimization."""
            import random

            class Result:
                def __init__(self, x, fun_val, success=True):
                    self.x = x
                    self.fun = fun_val
                    self.success = success

            # Simple random search for optimization
            best_x = x0
            best_val = fun(x0)

            for _ in range(100):
                # Random perturbation
                noise = [random.gauss(0, 0.1) for _ in range(len(x0))]
                test_x = [x + n for x, n in zip(best_x, noise)]

                try:
                    test_val = fun(test_x)
                    if test_val < best_val:
                        best_x = test_x
                        best_val = test_val
                except:
                    continue

            return Result(best_x, best_val)

    class MockStats:
        @staticmethod
        def norm():
            class MockNorm:
                @staticmethod
                def pdf(x, loc=0, scale=1):
                    # Gaussian PDF approximation
                    return math.exp(-0.5 * ((x - loc) / scale) ** 2) / (
                        scale * math.sqrt(2 * math.pi)
                    )

            return MockNorm()

    optimize = MockOptimize()
    stats = MockStats()

# Scikit-learn fallback
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern

    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn available - using advanced ML models")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using simplified ML models")

    class MockGaussianProcessRegressor:
        """Simplified Gaussian Process for Bayesian optimization."""

        def __init__(self, **kwargs):
            self.X_train = []
            self.y_train = []
            self.fitted = False

        def fit(self, X, y):
            """Fit the model to training data."""
            self.X_train = X
            self.y_train = y
            self.fitted = True
            return self

        def predict(self, X, return_std=False):
            """Predict with uncertainty estimation."""
            if not self.fitted:
                if return_std:
                    return [0.0] * len(X), [1.0] * len(X)
                return [0.0] * len(X)

            # Simple nearest neighbor prediction
            predictions = []
            uncertainties = []

            for x_test in X:
                if not self.X_train:
                    pred = 0.0
                    std = 1.0
                else:
                    # Find nearest neighbor
                    distances = []
                    for x_train in self.X_train:
                        dist = sum((a - b) ** 2 for a, b in zip(x_test, x_train)) ** 0.5
                        distances.append(dist)

                    nearest_idx = distances.index(min(distances))
                    pred = self.y_train[nearest_idx]

                    # Uncertainty based on distance to nearest neighbor
                    min_dist = min(distances)
                    std = max(0.1, min(1.0, min_dist))

                predictions.append(pred)
                uncertainties.append(std)

            if return_std:
                return predictions, uncertainties
            return predictions

    class MockMatern:
        def __init__(self, **kwargs):
            pass

    class MockRBF:
        def __init__(self, **kwargs):
            pass

    GaussianProcessRegressor = MockGaussianProcessRegressor
    Matern = MockMatern
    RBF = MockRBF

# MongoDB fallback
try:
    import pymongo

    PYMONGO_AVAILABLE = True
    logger.info("PyMongo available - using MongoDB storage")
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("PyMongo not available - using file-based storage")

# PSUtil fallback
try:
    import psutil

    PSUTIL_AVAILABLE = True
    logger.info("PSUtil available - using system monitoring")
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("PSUtil not available - using basic monitoring")

    class MockPsutil:
        @staticmethod
        def cpu_percent():
            return 50.0

        @staticmethod
        def virtual_memory():
            class Memory:
                percent = 60.0
                available = 8 * 1024 * 1024 * 1024  # 8GB
                used = 4 * 1024 * 1024 * 1024  # 4GB

            return Memory()

        @staticmethod
        def disk_usage(path):
            class Disk:
                free = 100 * 1024 * 1024 * 1024  # 100GB
                used = 50 * 1024 * 1024 * 1024  # 50GB
                total = 150 * 1024 * 1024 * 1024  # 150GB

            return Disk()

    psutil = MockPsutil()


def ensure_numpy_compatible(x):
    """Ensure data is compatible with both numpy and fallback implementations."""
    if isinstance(x, (list, tuple)):
        return x
    elif hasattr(x, "tolist"):
        return x.tolist()
    else:
        return [x]


def safe_numerical_operation(operation, *args, default_value=0.0):
    """Safely perform numerical operations with error handling."""
    try:
        return operation(*args)
    except Exception as e:
        logger.warning(
            f"Numerical operation failed: {e}, returning default: {default_value}"
        )
        return default_value


# Export availability flags and implementations
__all__ = [
    "np",
    "optimize",
    "stats",
    "GaussianProcessRegressor",
    "Matern",
    "RBF",
    "psutil",
    "NUMPY_AVAILABLE",
    "SCIPY_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "PYMONGO_AVAILABLE",
    "PSUTIL_AVAILABLE",
    "ensure_numpy_compatible",
    "safe_numerical_operation",
]
