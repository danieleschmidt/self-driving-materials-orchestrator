"""Enhanced ML capabilities with comprehensive fallbacks."""

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Machine learning model performance metrics."""

    accuracy: float = 0.0
    r2_score: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    training_time: float = 0.0
    prediction_count: int = 0


class EnhancedMLOptimizer:
    """Enhanced ML optimizer with advanced capabilities."""

    def __init__(self, target_property: str, model_type: str = "gaussian_process"):
        self.target_property = target_property
        self.model_type = model_type
        self.training_data: List[Dict[str, Any]] = []
        self.metrics = ModelMetrics()
        self.model_trained = False

        # Try to import real ML libraries
        self.has_sklearn = self._check_sklearn()
        self.has_numpy = self._check_numpy()

        logger.info(f"Enhanced ML optimizer initialized for {target_property}")
        if not self.has_sklearn:
            logger.warning(
                "scikit-learn not available, using advanced fallback algorithms"
            )

    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn

            return True
        except ImportError:
            return False

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy

            return True
        except ImportError:
            return False

    def add_training_data(self, experiment: Dict[str, Any]) -> None:
        """Add experiment data for model training."""
        if "parameters" in experiment and "results" in experiment:
            self.training_data.append(experiment)

            # Retrain model every 10 experiments
            if len(self.training_data) % 10 == 0:
                self.train_model()

    def train_model(self) -> ModelMetrics:
        """Train the ML model with current data."""
        if len(self.training_data) < 3:
            logger.warning("Insufficient training data for ML model")
            return self.metrics

        logger.info(f"Training ML model with {len(self.training_data)} experiments")

        if self.has_sklearn:
            return self._train_sklearn_model()
        else:
            return self._train_fallback_model()

    def _train_sklearn_model(self) -> ModelMetrics:
        """Train model using scikit-learn."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            # Prepare data
            X, y = self._prepare_training_data()

            if len(X) < 3:
                return self.metrics

            # Choose model based on type
            if self.model_type == "gaussian_process":
                model = GaussianProcessRegressor(random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            # Train model
            model.fit(X, y)

            # Calculate metrics
            y_pred = model.predict(X)
            self.metrics.accuracy = max(
                0, 1 - mean_absolute_error(y, y_pred) / (max(y) - min(y) + 1e-10)
            )
            self.metrics.r2_score = r2_score(y, y_pred)
            self.metrics.mae = mean_absolute_error(y, y_pred)
            self.metrics.rmse = math.sqrt(mean_squared_error(y, y_pred))

            self.model = model
            self.model_trained = True

            logger.info(
                f"Model trained - R²: {self.metrics.r2_score:.3f}, MAE: {self.metrics.mae:.3f}"
            )

        except Exception as e:
            logger.error(f"sklearn model training failed: {e}")
            return self._train_fallback_model()

        return self.metrics

    def _train_fallback_model(self) -> ModelMetrics:
        """Train model using advanced fallback algorithms."""
        # Implement sophisticated fallback with statistical modeling

        # Extract target values
        targets = []
        for exp in self.training_data:
            if self.target_property in exp["results"]:
                targets.append(exp["results"][self.target_property])

        if not targets:
            return self.metrics

        # Calculate sophisticated statistics
        mean_target = sum(targets) / len(targets)
        variance = sum((x - mean_target) ** 2 for x in targets) / len(targets)
        std_dev = math.sqrt(variance)

        # Advanced correlation analysis
        param_correlations = self._calculate_parameter_correlations()

        # Create statistical model
        self.fallback_model = {
            "mean": mean_target,
            "std": std_dev,
            "correlations": param_correlations,
            "data_points": len(targets),
            "target_range": (min(targets), max(targets)),
        }

        # Calculate fallback metrics
        predictions = [
            self._predict_fallback(exp["parameters"]) for exp in self.training_data
        ]
        actual_values = [
            exp["results"][self.target_property]
            for exp in self.training_data
            if self.target_property in exp["results"]
        ]

        if len(predictions) == len(actual_values) and len(actual_values) > 0:
            mae = sum(abs(p - a) for p, a in zip(predictions, actual_values)) / len(
                actual_values
            )
            self.metrics.mae = mae
            self.metrics.accuracy = max(
                0, 1 - mae / (max(actual_values) - min(actual_values) + 1e-10)
            )

        self.model_trained = True
        logger.info(
            f"Fallback model trained with {len(self.training_data)} data points"
        )

        return self.metrics

    def _prepare_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare training data for sklearn models."""
        X, y = [], []

        # Get parameter names consistently
        all_params = set()
        for exp in self.training_data:
            all_params.update(exp["parameters"].keys())
        param_names = sorted(list(all_params))

        for exp in self.training_data:
            if self.target_property in exp["results"]:
                # Create feature vector
                features = []
                for param in param_names:
                    features.append(exp["parameters"].get(param, 0.0))

                X.append(features)
                y.append(exp["results"][self.target_property])

        return X, y

    def _calculate_parameter_correlations(self) -> Dict[str, float]:
        """Calculate parameter correlations with target property."""
        correlations = {}

        # Get all parameter names
        all_params = set()
        for exp in self.training_data:
            all_params.update(exp["parameters"].keys())

        for param in all_params:
            param_values = []
            target_values = []

            for exp in self.training_data:
                if (
                    param in exp["parameters"]
                    and self.target_property in exp["results"]
                ):
                    param_values.append(exp["parameters"][param])
                    target_values.append(exp["results"][self.target_property])

            if len(param_values) > 1:
                # Calculate Pearson correlation coefficient
                corr = self._pearson_correlation(param_values, target_values)
                correlations[param] = corr

        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt(
            (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def predict(self, parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Predict property value and uncertainty."""
        if not self.model_trained:
            if len(self.training_data) > 0:
                self.train_model()
            else:
                return self._default_prediction(parameters)

        if self.has_sklearn and hasattr(self, "model"):
            return self._predict_sklearn(parameters)
        else:
            return self._predict_fallback(parameters)

    def _predict_sklearn(self, parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction using scikit-learn model."""
        try:
            # Prepare feature vector
            X, _ = self._prepare_training_data()
            if not X:
                return self._default_prediction(parameters)

            param_names = sorted(
                set().union(*(exp["parameters"].keys() for exp in self.training_data))
            )
            features = [parameters.get(param, 0.0) for param in param_names]

            prediction = self.model.predict([features])[0]

            # Estimate uncertainty (simplified)
            uncertainty = self.metrics.mae if self.metrics.mae > 0 else 0.1

            return float(prediction), float(uncertainty)

        except Exception as e:
            logger.error(f"sklearn prediction failed: {e}")
            return self._predict_fallback(parameters)

    def _predict_fallback(self, parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Make prediction using fallback statistical model."""
        if not hasattr(self, "fallback_model"):
            return self._default_prediction(parameters)

        model = self.fallback_model
        base_prediction = model["mean"]

        # Apply correlation adjustments
        for param, value in parameters.items():
            if param in model["correlations"]:
                corr = model["correlations"][param]
                # Simple linear adjustment based on correlation
                param_avg = self._get_parameter_average(param)
                if param_avg is not None:
                    normalized_value = (value - param_avg) / (param_avg + 1e-10)
                    adjustment = corr * normalized_value * model["std"] * 0.5
                    base_prediction += adjustment

        uncertainty = model["std"] * 0.5

        return float(base_prediction), float(uncertainty)

    def _get_parameter_average(self, param: str) -> Optional[float]:
        """Get average value for a parameter."""
        values = []
        for exp in self.training_data:
            if param in exp["parameters"]:
                values.append(exp["parameters"][param])

        return sum(values) / len(values) if values else None

    def _default_prediction(self, parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Default prediction when no model is available."""
        # Use heuristic based on parameter ranges
        if "temperature" in parameters:
            # Temperature-based prediction for materials
            temp = parameters["temperature"]
            base_value = 1.5 + (temp - 150) / 500  # Simple heuristic
            uncertainty = 0.2
        else:
            base_value = 1.5  # Default value
            uncertainty = 0.3

        return base_value, uncertainty

    def suggest_next_experiments(
        self, param_space: Dict[str, Tuple[float, float]], n_suggestions: int = 5
    ) -> List[Dict[str, float]]:
        """Suggest next experiments using acquisition function."""
        suggestions = []

        # Use different strategies for variety
        strategies = ["exploitation", "exploration", "balanced"]

        for i in range(n_suggestions):
            strategy = strategies[i % len(strategies)]
            suggestion = self._generate_suggestion(param_space, strategy)
            suggestions.append(suggestion)

        return suggestions

    def _generate_suggestion(
        self, param_space: Dict[str, Tuple[float, float]], strategy: str
    ) -> Dict[str, float]:
        """Generate a single experiment suggestion."""
        suggestion = {}

        for param, (low, high) in param_space.items():
            if strategy == "exploitation" and self.model_trained:
                # Focus on promising regions
                if hasattr(self, "fallback_model"):
                    correlations = self.fallback_model.get("correlations", {})
                    if param in correlations:
                        corr = correlations[param]
                        if abs(corr) > 0.3:  # Strong correlation
                            param_avg = self._get_parameter_average(param)
                            if param_avg is not None:
                                # Bias towards correlated direction
                                if corr > 0:
                                    value = (
                                        param_avg
                                        + (high - param_avg) * random.random() * 0.7
                                    )
                                else:
                                    value = (
                                        param_avg
                                        - (param_avg - low) * random.random() * 0.7
                                    )
                                suggestion[param] = max(low, min(high, value))
                                continue

            elif strategy == "exploration":
                # Random exploration with slight bias to unexplored regions
                value = low + (high - low) * random.random()

            else:  # balanced
                # Gaussian around current best region
                if hasattr(self, "fallback_model"):
                    param_avg = self._get_parameter_average(param)
                    if param_avg is not None:
                        std = (high - low) * 0.2
                        value = random.gauss(param_avg, std)
                        value = max(low, min(high, value))
                    else:
                        value = low + (high - low) * random.random()
                else:
                    value = low + (high - low) * random.random()

            suggestion[param] = float(value)

        return suggestion

    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights from the trained model."""
        insights = {
            "model_type": self.model_type,
            "training_samples": len(self.training_data),
            "model_trained": self.model_trained,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "r2_score": self.metrics.r2_score,
                "mae": self.metrics.mae,
                "rmse": self.metrics.rmse,
            },
        }

        if hasattr(self, "fallback_model"):
            insights["parameter_correlations"] = self.fallback_model["correlations"]
            insights["target_statistics"] = {
                "mean": self.fallback_model["mean"],
                "std": self.fallback_model["std"],
                "range": self.fallback_model["target_range"],
            }

        return insights
