"""Machine learning acceleration and intelligent optimization strategies."""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of property prediction."""
    predicted_value: float
    confidence: float
    uncertainty: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_used: str = "unknown"


@dataclass
class OptimizationSuggestion:
    """Suggestion for next experiment."""
    parameters: Dict[str, float]
    expected_improvement: float
    uncertainty: float
    strategy: str
    confidence: float


class PropertyPredictor:
    """Fast property prediction using ML models."""
    
    def __init__(self, target_property: str = "band_gap"):
        self.target_property = target_property
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.feature_names: List[str] = []
        self.model_performance: Dict[str, float] = {}
        
    def add_training_data(self, experiment_data: Dict[str, Any]) -> None:
        """Add experiment data for model training."""
        if self.target_property in experiment_data.get('results', {}):
            self.training_data.append(experiment_data)
            
            # Retrain if we have enough data
            if len(self.training_data) % 10 == 0:  # Retrain every 10 experiments
                self._train_models()
    
    def _extract_features(self, experiment_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features from experiment data."""
        parameters = experiment_data.get('parameters', {})
        
        # Define feature extraction for common parameters
        feature_map = {
            'precursor_A_conc': parameters.get('precursor_A_conc', 0),
            'precursor_B_conc': parameters.get('precursor_B_conc', 0),
            'temperature': parameters.get('temperature', 0),
            'reaction_time': parameters.get('reaction_time', 0),
            'pH': parameters.get('pH', 7),
            'solvent_ratio': parameters.get('solvent_ratio', 0.5),
        }
        
        # Add derived features
        feature_map['total_conc'] = feature_map['precursor_A_conc'] + feature_map['precursor_B_conc']
        feature_map['conc_ratio'] = (
            feature_map['precursor_A_conc'] / max(feature_map['precursor_B_conc'], 0.001)
        )
        feature_map['temp_time_product'] = feature_map['temperature'] * feature_map['reaction_time']
        
        if not self.feature_names:
            self.feature_names = list(feature_map.keys())
        
        return [feature_map[name] for name in self.feature_names]
    
    def _train_models(self) -> None:
        """Train ML models on available data."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("ML libraries not available, skipping model training")
            return
        
        if len(self.training_data) < 5:
            logger.info("Not enough training data for ML models")
            return
        
        # Prepare training data
        X, y = [], []
        for exp in self.training_data:
            features = self._extract_features(exp)
            target = exp.get('results', {}).get(self.target_property)
            
            if features and target is not None:
                X.append(features)
                y.append(target)
        
        if len(X) < 5:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'gp': GaussianProcessRegressor(random_state=42) if len(X) < 100 else None
        }
        
        for model_name, model in models_to_train.items():
            if model is None:
                continue
                
            try:
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                # Evaluate performance
                if len(X) > 10:
                    scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)//2), scoring='r2')
                    performance = scores.mean()
                else:
                    performance = 0.5  # Default for small datasets
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                self.model_performance[model_name] = performance
                
                logger.info(f"Trained {model_name} model with R² = {performance:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name} model: {e}")
    
    def predict(self, parameters: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict property value for given parameters."""
        if not self.models or not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        
        # Extract features
        experiment_data = {'parameters': parameters}
        features = self._extract_features(experiment_data)
        
        if not features:
            return None
        
        # Use best performing model
        best_model_name = max(self.model_performance, key=self.model_performance.get)
        model = self.models[best_model_name]
        scaler = self.scalers[best_model_name]
        
        try:
            # Scale features and predict
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on model performance and prediction variance
            confidence = self.model_performance[best_model_name]
            
            # Get feature importance for tree-based models
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            
            # Get uncertainty for GP models
            uncertainty = None
            if hasattr(model, 'predict') and 'gp' in best_model_name:
                try:
                    _, std = model.predict(features_scaled, return_std=True)
                    uncertainty = std[0]
                except:
                    pass
            
            return PredictionResult(
                predicted_value=float(prediction),
                confidence=float(confidence),
                uncertainty=uncertainty,
                feature_importance=feature_importance,
                model_used=best_model_name
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        return {
            'available_models': list(self.models.keys()),
            'model_performance': self.model_performance,
            'training_data_size': len(self.training_data),
            'feature_names': self.feature_names,
            'target_property': self.target_property
        }


class IntelligentOptimizer:
    """Intelligent optimization using multiple strategies."""
    
    def __init__(self, target_property: str = "band_gap"):
        self.target_property = target_property
        self.predictor = PropertyPredictor(target_property)
        self.exploration_strategies = [
            self._uncertainty_sampling,
            self._diversity_sampling,
            self._expected_improvement,
            self._random_exploration
        ]
        self.strategy_performance: Dict[str, List[float]] = {}
        self.recent_suggestions: List[OptimizationSuggestion] = []
        
    def add_experiment_result(self, experiment_data: Dict[str, Any]) -> None:
        """Add experiment result to improve optimization."""
        self.predictor.add_training_data(experiment_data)
        self._update_strategy_performance(experiment_data)
    
    def _update_strategy_performance(self, experiment_data: Dict[str, Any]) -> None:
        """Update performance tracking for different strategies."""
        # Find which suggestion led to this result
        parameters = experiment_data.get('parameters', {})
        target_value = experiment_data.get('results', {}).get(self.target_property)
        
        if target_value is None:
            return
        
        # Match to recent suggestions (simplified)
        for suggestion in self.recent_suggestions[-10:]:  # Check last 10 suggestions
            if self._parameters_match(suggestion.parameters, parameters, tolerance=0.1):
                strategy = suggestion.strategy
                if strategy not in self.strategy_performance:
                    self.strategy_performance[strategy] = []
                
                # Score based on expected vs actual improvement
                improvement_score = min(suggestion.expected_improvement / max(abs(target_value - 1.4), 0.1), 2.0)
                self.strategy_performance[strategy].append(improvement_score)
                
                # Keep only recent performance
                if len(self.strategy_performance[strategy]) > 50:
                    self.strategy_performance[strategy] = self.strategy_performance[strategy][-25:]
                break
    
    def _parameters_match(self, params1: Dict[str, float], params2: Dict[str, float], tolerance: float = 0.1) -> bool:
        """Check if parameter sets match within tolerance."""
        for key in params1:
            if key in params2:
                if abs(params1[key] - params2[key]) / max(abs(params1[key]), 0.1) > tolerance:
                    return False
        return True
    
    def suggest_next_experiments(
        self, 
        param_space: Dict[str, Tuple[float, float]], 
        n_suggestions: int = 5,
        previous_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[OptimizationSuggestion]:
        """Suggest next experiments using intelligent strategies."""
        
        suggestions = []
        
        # Select best performing strategies
        strategy_scores = self._get_strategy_scores()
        selected_strategies = self._select_strategies(strategy_scores, n_suggestions)
        
        for strategy_func, strategy_name in selected_strategies:
            try:
                suggestion = strategy_func(param_space, previous_results)
                if suggestion:
                    suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
        
        # Fill remaining slots with random exploration
        while len(suggestions) < n_suggestions:
            random_suggestion = self._random_exploration(param_space, previous_results)
            if random_suggestion:
                suggestions.append(random_suggestion)
        
        # Store suggestions for performance tracking
        self.recent_suggestions.extend(suggestions)
        if len(self.recent_suggestions) > 100:
            self.recent_suggestions = self.recent_suggestions[-50:]
        
        return suggestions[:n_suggestions]
    
    def _get_strategy_scores(self) -> Dict[str, float]:
        """Get performance scores for each strategy."""
        scores = {}
        for strategy_name, performances in self.strategy_performance.items():
            if performances:
                scores[strategy_name] = sum(performances) / len(performances)
            else:
                scores[strategy_name] = 0.5  # Default score
        return scores
    
    def _select_strategies(self, strategy_scores: Dict[str, float], n_needed: int) -> List[Tuple[Callable, str]]:
        """Select best performing strategies."""
        strategy_map = {
            'uncertainty_sampling': self._uncertainty_sampling,
            'diversity_sampling': self._diversity_sampling,
            'expected_improvement': self._expected_improvement,
            'random_exploration': self._random_exploration
        }
        
        # Sort strategies by performance
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        for strategy_name, score in sorted_strategies[:n_needed]:
            if strategy_name in strategy_map:
                selected.append((strategy_map[strategy_name], strategy_name))
        
        return selected
    
    def _uncertainty_sampling(self, param_space: Dict[str, Tuple[float, float]], previous_results: Optional[List[Dict[str, Any]]]) -> Optional[OptimizationSuggestion]:
        """Sample from regions with high prediction uncertainty."""
        if not NUMPY_AVAILABLE:
            return None
        
        # Generate candidate points
        candidates = self._generate_candidates(param_space, 100)
        best_candidate = None
        max_uncertainty = 0
        
        for candidate in candidates:
            prediction = self.predictor.predict(candidate)
            if prediction and prediction.uncertainty:
                if prediction.uncertainty > max_uncertainty:
                    max_uncertainty = prediction.uncertainty
                    best_candidate = candidate
        
        if best_candidate:
            return OptimizationSuggestion(
                parameters=best_candidate,
                expected_improvement=max_uncertainty,
                uncertainty=max_uncertainty,
                strategy="uncertainty_sampling",
                confidence=0.7
            )
        
        return None
    
    def _diversity_sampling(self, param_space: Dict[str, Tuple[float, float]], previous_results: Optional[List[Dict[str, Any]]]) -> Optional[OptimizationSuggestion]:
        """Sample from unexplored regions of parameter space."""
        if not previous_results or not NUMPY_AVAILABLE:
            return self._random_exploration(param_space, previous_results)
        
        # Get existing parameter combinations
        existing_params = []
        for result in previous_results:
            params = result.get('parameters', {})
            if params:
                existing_params.append(params)
        
        if not existing_params:
            return self._random_exploration(param_space, previous_results)
        
        # Find most diverse candidate
        candidates = self._generate_candidates(param_space, 50)
        best_candidate = None
        max_diversity = 0
        
        for candidate in candidates:
            diversity = self._calculate_diversity(candidate, existing_params)
            if diversity > max_diversity:
                max_diversity = diversity
                best_candidate = candidate
        
        if best_candidate:
            return OptimizationSuggestion(
                parameters=best_candidate,
                expected_improvement=max_diversity * 0.5,
                uncertainty=0.8,
                strategy="diversity_sampling",
                confidence=0.6
            )
        
        return None
    
    def _expected_improvement(self, param_space: Dict[str, Tuple[float, float]], previous_results: Optional[List[Dict[str, Any]]]) -> Optional[OptimizationSuggestion]:
        """Sample based on expected improvement over current best."""
        if not previous_results:
            return self._random_exploration(param_space, previous_results)
        
        # Find current best
        best_value = None
        for result in previous_results:
            value = result.get('results', {}).get(self.target_property)
            if value is not None:
                if best_value is None or abs(value - 1.4) < abs(best_value - 1.4):  # Closer to target
                    best_value = value
        
        if best_value is None:
            return self._random_exploration(param_space, previous_results)
        
        # Find candidate with highest expected improvement
        candidates = self._generate_candidates(param_space, 50)
        best_candidate = None
        max_ei = 0
        
        for candidate in candidates:
            prediction = self.predictor.predict(candidate)
            if prediction:
                # Calculate expected improvement (simplified)
                improvement = max(0, abs(prediction.predicted_value - 1.4) - abs(best_value - 1.4))
                ei = improvement * prediction.confidence
                
                if ei > max_ei:
                    max_ei = ei
                    best_candidate = candidate
        
        if best_candidate:
            return OptimizationSuggestion(
                parameters=best_candidate,
                expected_improvement=max_ei,
                uncertainty=0.5,
                strategy="expected_improvement",
                confidence=0.8
            )
        
        return None
    
    def _random_exploration(self, param_space: Dict[str, Tuple[float, float]], previous_results: Optional[List[Dict[str, Any]]]) -> OptimizationSuggestion:
        """Random exploration as fallback strategy."""
        import random
        
        parameters = {}
        for param, (low, high) in param_space.items():
            parameters[param] = random.uniform(low, high)
        
        return OptimizationSuggestion(
            parameters=parameters,
            expected_improvement=0.3,
            uncertainty=1.0,
            strategy="random_exploration",
            confidence=0.4
        )
    
    def _generate_candidates(self, param_space: Dict[str, Tuple[float, float]], n_candidates: int) -> List[Dict[str, float]]:
        """Generate candidate parameter combinations."""
        if not NUMPY_AVAILABLE:
            import random
            candidates = []
            for _ in range(n_candidates):
                candidate = {}
                for param, (low, high) in param_space.items():
                    candidate[param] = random.uniform(low, high)
                candidates.append(candidate)
            return candidates
        
        candidates = []
        for _ in range(n_candidates):
            candidate = {}
            for param, (low, high) in param_space.items():
                candidate[param] = np.random.uniform(low, high)
            candidates.append(candidate)
        
        return candidates
    
    def _calculate_diversity(self, candidate: Dict[str, float], existing_params: List[Dict[str, float]]) -> float:
        """Calculate diversity score for a candidate."""
        if not existing_params:
            return 1.0
        
        min_distance = float('inf')
        for existing in existing_params:
            distance = 0
            count = 0
            for param, value in candidate.items():
                if param in existing:
                    distance += (value - existing[param]) ** 2
                    count += 1
            
            if count > 0:
                distance = (distance / count) ** 0.5
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'strategy_performance': {k: sum(v)/len(v) if v else 0 for k, v in self.strategy_performance.items()},
            'recent_suggestions': len(self.recent_suggestions),
            'model_info': self.predictor.get_model_info(),
            'total_experiments_learned': len(self.predictor.training_data)
        }


def create_intelligent_optimizer(target_property: str = "band_gap") -> IntelligentOptimizer:
    """Create an intelligent optimizer instance."""
    return IntelligentOptimizer(target_property=target_property)
