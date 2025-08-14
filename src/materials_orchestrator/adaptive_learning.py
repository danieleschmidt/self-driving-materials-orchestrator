"""Adaptive learning system for continuous improvement of materials discovery."""

import logging
from .utils import np, NUMPY_AVAILABLE
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning modes for adaptive system."""
    PASSIVE = "passive"  # Learn from outcomes
    ACTIVE = "active"    # Actively seek learning opportunities
    HYBRID = "hybrid"    # Combination of both

@dataclass
class LearningPattern:
    """Represents a learned pattern."""
    pattern_id: str
    pattern_type: str
    confidence: float
    observations: int
    success_rate: float
    context: Dict[str, Any]
    last_updated: datetime
    description: str

@dataclass
class ExperimentOutcome:
    """Standardized experiment outcome for learning."""
    experiment_id: str
    parameters: Dict[str, float]
    results: Dict[str, float]
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdaptiveLearningEngine:
    """Adaptive learning engine that improves over time."""
    
    def __init__(self, learning_mode: LearningMode = LearningMode.HYBRID):
        """Initialize adaptive learning engine."""
        self.learning_mode = learning_mode
        self.patterns = {}
        self.experiment_history = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.max_history = 10000
        
        # Performance tracking
        self.performance_metrics = {
            'prediction_accuracy': 0.5,
            'recommendation_success_rate': 0.5,
            'learning_speed': 0.5,
            'pattern_discovery_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize learning components
        self._initialize_learning_modules()
        
    def _initialize_learning_modules(self):
        """Initialize learning modules."""
        self.pattern_detectors = {
            'parameter_correlation': self._detect_parameter_correlations,
            'success_prediction': self._learn_success_predictors,
            'optimization_strategy': self._learn_optimization_strategies,
            'failure_pattern': self._detect_failure_patterns,
            'convergence_pattern': self._learn_convergence_patterns
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'correlation_threshold': 0.6,
            'success_threshold': 0.8,
            'pattern_strength_threshold': 0.7
        }
        
    def learn_from_experiment(self, outcome: ExperimentOutcome):
        """Learn from a single experiment outcome."""
        with self._lock:
            # Add to history
            self.experiment_history.append(outcome)
            
            # Maintain history size
            if len(self.experiment_history) > self.max_history:
                self.experiment_history = self.experiment_history[-self.max_history:]
            
            # Update patterns
            self._update_patterns(outcome)
            
            # Adaptive threshold adjustment
            self._adjust_thresholds()
            
            logger.debug(f"Learned from experiment {outcome.experiment_id}")
    
    def learn_from_campaign(self, campaign_results):
        """Learn from an entire campaign."""
        logger.info(f"Learning from campaign {campaign_results.campaign_id}")
        
        # Extract experiment outcomes
        outcomes = []
        for exp in campaign_results.experiments:
            if exp.status == "completed":
                outcome = ExperimentOutcome(
                    experiment_id=exp.id,
                    parameters=exp.parameters,
                    results=exp.results,
                    success=campaign_results.objective.evaluate_success(
                        exp.results.get(campaign_results.objective.target_property, 0)
                    ),
                    duration=exp.duration or 0.0,
                    metadata=exp.metadata
                )
                outcomes.append(outcome)
        
        # Batch learning
        for outcome in outcomes:
            self.learn_from_experiment(outcome)
        
        # Campaign-level pattern detection
        self._detect_campaign_patterns(campaign_results)
        
        # Update performance metrics
        self._update_performance_metrics(campaign_results)
        
    def predict_experiment_success(self, parameters: Dict[str, float]) -> Tuple[float, str]:
        """Predict probability of experiment success."""
        if len(self.experiment_history) < 10:
            return 0.5, "Insufficient data for prediction"
        
        # Use learned patterns for prediction
        success_probability = 0.5  # Default
        reasoning = []
        
        # Check parameter correlation patterns
        for pattern_id, pattern in self.patterns.items():
            if pattern.pattern_type == 'parameter_correlation' and pattern.confidence > 0.6:
                # Simple pattern matching
                match_score = self._calculate_pattern_match(parameters, pattern.context)
                if match_score > 0.7:
                    success_probability = max(success_probability, pattern.success_rate * match_score)
                    reasoning.append(f"Pattern {pattern_id}: {pattern.description}")
        
        # Historical similarity
        similar_experiments = self._find_similar_experiments(parameters, top_k=5)
        if similar_experiments:
            similar_success_rate = sum(exp.success for exp in similar_experiments) / len(similar_experiments)
            success_probability = (success_probability + similar_success_rate) / 2
            reasoning.append(f"Similar experiments success rate: {similar_success_rate:.2f}")
        
        reasoning_text = "; ".join(reasoning) if reasoning else "No specific patterns found"
        return success_probability, reasoning_text
    
    def recommend_parameters(self, objective, param_space: Dict[str, Tuple], 
                           num_recommendations: int = 3) -> List[Dict[str, float]]:
        """Recommend promising parameter combinations."""
        recommendations = []
        
        if len(self.experiment_history) < 5:
            # Random recommendations for cold start
            for _ in range(num_recommendations):
                params = {}
                for param, (low, high) in param_space.items():
                    params[param] = np.random.uniform(low, high)
                recommendations.append(params)
            return recommendations
        
        # Strategy 1: Exploit successful regions
        successful_experiments = [exp for exp in self.experiment_history if exp.success]
        if successful_experiments:
            # Find best performing experiments
            target_property = objective.target_property
            best_experiments = sorted(
                successful_experiments,
                key=lambda x: x.results.get(target_property, 0),
                reverse=True
            )[:5]
            
            # Generate variations around best experiments
            for exp in best_experiments[:num_recommendations]:
                variation = self._generate_parameter_variation(exp.parameters, param_space)
                recommendations.append(variation)
        
        # Strategy 2: Explore based on learned patterns
        if len(recommendations) < num_recommendations:
            pattern_recommendations = self._generate_pattern_based_recommendations(
                objective, param_space, num_recommendations - len(recommendations)
            )
            recommendations.extend(pattern_recommendations)
        
        # Ensure we have enough recommendations
        while len(recommendations) < num_recommendations:
            random_params = {}
            for param, (low, high) in param_space.items():
                random_params[param] = np.random.uniform(low, high)
            recommendations.append(random_params)
        
        return recommendations[:num_recommendations]
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what the system has learned."""
        insights = {
            'total_experiments': len(self.experiment_history),
            'patterns_discovered': len(self.patterns),
            'performance_metrics': self.performance_metrics.copy(),
            'key_patterns': [],
            'recommendations': []
        }
        
        # Extract key patterns
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda x: x.confidence * x.observations,
            reverse=True
        )
        
        for pattern in sorted_patterns[:5]:
            insights['key_patterns'].append({
                'type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'observations': pattern.observations,
                'description': pattern.description
            })
        
        # Generate learning recommendations
        if self.performance_metrics['prediction_accuracy'] < 0.6:
            insights['recommendations'].append("Collect more diverse experimental data")
        
        if len(self.patterns) < 3:
            insights['recommendations'].append("Increase experiment variety to discover patterns")
        
        return insights
    
    def _update_patterns(self, outcome: ExperimentOutcome):
        """Update patterns based on new outcome."""
        # Run all pattern detectors
        for detector_name, detector_func in self.pattern_detectors.items():
            try:
                new_patterns = detector_func(outcome)
                for pattern in new_patterns:
                    self._integrate_pattern(pattern)
            except Exception as e:
                logger.warning(f"Pattern detector {detector_name} failed: {e}")
    
    def _detect_parameter_correlations(self, outcome: ExperimentOutcome) -> List[LearningPattern]:
        """Detect correlations between parameters and outcomes."""
        patterns = []
        
        if len(self.experiment_history) < 10:
            return patterns
        
        # Analyze correlation for each parameter
        for param_name in outcome.parameters.keys():
            param_values = []
            target_values = []
            
            for exp in self.experiment_history[-50:]:  # Recent history
                if param_name in exp.parameters and exp.results:
                    param_values.append(exp.parameters[param_name])
                    # Use first result as target (simplified)
                    target_value = list(exp.results.values())[0] if exp.results else 0
                    target_values.append(target_value)
            
            if len(param_values) >= 10:
                # Calculate correlation
                correlation = np.corrcoef(param_values, target_values)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > self.adaptive_thresholds['correlation_threshold']:
                    pattern = LearningPattern(
                        pattern_id=f"param_corr_{param_name}",
                        pattern_type="parameter_correlation",
                        confidence=abs(correlation),
                        observations=len(param_values),
                        success_rate=sum(exp.success for exp in self.experiment_history[-50:]) / min(50, len(self.experiment_history)),
                        context={'parameter': param_name, 'correlation': correlation},
                        last_updated=datetime.now(),
                        description=f"Parameter {param_name} shows correlation {correlation:.2f}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _learn_success_predictors(self, outcome: ExperimentOutcome) -> List[LearningPattern]:
        """Learn predictors of experimental success."""
        patterns = []
        
        if len(self.experiment_history) < 20:
            return patterns
        
        # Analyze success patterns by parameter ranges
        successful_params = {}
        failed_params = {}
        
        for exp in self.experiment_history[-100:]:
            param_dict = successful_params if exp.success else failed_params
            for param, value in exp.parameters.items():
                if param not in param_dict:
                    param_dict[param] = []
                param_dict[param].append(value)
        
        # Find parameter ranges with high success rates
        for param in successful_params.keys():
            if param in failed_params and len(successful_params[param]) >= 5:
                success_range = (min(successful_params[param]), max(successful_params[param]))
                success_mean = np.mean(successful_params[param])
                
                # Check if this range is significantly different from failures
                if len(failed_params[param]) >= 5:
                    failed_mean = np.mean(failed_params[param])
                    if abs(success_mean - failed_mean) > np.std(failed_params[param]):
                        pattern = LearningPattern(
                            pattern_id=f"success_range_{param}",
                            pattern_type="success_prediction",
                            confidence=0.7,  # Conservative estimate
                            observations=len(successful_params[param]),
                            success_rate=1.0,  # By definition for successful range
                            context={'parameter': param, 'optimal_range': success_range, 'optimal_mean': success_mean},
                            last_updated=datetime.now(),
                            description=f"Parameter {param} optimal range: {success_range}"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _learn_optimization_strategies(self, outcome: ExperimentOutcome) -> List[LearningPattern]:
        """Learn effective optimization strategies."""
        patterns = []
        
        # Placeholder for more sophisticated strategy learning
        # This could analyze convergence patterns, parameter exploration strategies, etc.
        
        return patterns
    
    def _detect_failure_patterns(self, outcome: ExperimentOutcome) -> List[LearningPattern]:
        """Detect patterns that lead to experimental failures."""
        patterns = []
        
        if not outcome.success and len(self.experiment_history) >= 10:
            # Look for common failure patterns
            recent_failures = [exp for exp in self.experiment_history[-20:] if not exp.success]
            
            if len(recent_failures) >= 3:
                # Analyze common parameter characteristics in failures
                failure_params = {}
                for exp in recent_failures:
                    for param, value in exp.parameters.items():
                        if param not in failure_params:
                            failure_params[param] = []
                        failure_params[param].append(value)
                
                # Check for parameter ranges that consistently fail
                for param, values in failure_params.items():
                    if len(values) >= 3:
                        failure_range = (min(values), max(values))
                        pattern = LearningPattern(
                            pattern_id=f"failure_pattern_{param}",
                            pattern_type="failure_pattern",
                            confidence=len(values) / len(recent_failures),
                            observations=len(values),
                            success_rate=0.0,
                            context={'parameter': param, 'failure_range': failure_range},
                            last_updated=datetime.now(),
                            description=f"Parameter {param} failure range: {failure_range}"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _learn_convergence_patterns(self, outcome: ExperimentOutcome) -> List[LearningPattern]:
        """Learn patterns related to optimization convergence."""
        patterns = []
        
        # Placeholder for convergence pattern learning
        # This could analyze how different parameter combinations affect convergence speed
        
        return patterns
    
    def _integrate_pattern(self, pattern: LearningPattern):
        """Integrate a new pattern into the knowledge base."""
        if pattern.pattern_id in self.patterns:
            # Update existing pattern
            existing = self.patterns[pattern.pattern_id]
            
            # Weighted average for confidence
            total_obs = existing.observations + pattern.observations
            existing.confidence = (
                (existing.confidence * existing.observations + pattern.confidence * pattern.observations) 
                / total_obs
            )
            existing.observations = total_obs
            existing.success_rate = (
                (existing.success_rate * existing.observations + pattern.success_rate * pattern.observations)
                / total_obs
            )
            existing.last_updated = datetime.now()
            
            # Update context with new information
            existing.context.update(pattern.context)
        else:
            # Add new pattern
            self.patterns[pattern.pattern_id] = pattern
    
    def _calculate_pattern_match(self, parameters: Dict[str, float], pattern_context: Dict[str, Any]) -> float:
        """Calculate how well parameters match a pattern."""
        if 'parameter' in pattern_context:
            param_name = pattern_context['parameter']
            if param_name in parameters:
                if 'optimal_range' in pattern_context:
                    optimal_range = pattern_context['optimal_range']
                    param_value = parameters[param_name]
                    if optimal_range[0] <= param_value <= optimal_range[1]:
                        return 1.0
                    else:
                        # Distance-based similarity
                        range_center = (optimal_range[0] + optimal_range[1]) / 2
                        range_width = optimal_range[1] - optimal_range[0]
                        distance = abs(param_value - range_center)
                        return max(0.0, 1.0 - distance / range_width)
        
        return 0.0
    
    def _find_similar_experiments(self, parameters: Dict[str, float], top_k: int = 5) -> List[ExperimentOutcome]:
        """Find experiments with similar parameters."""
        if not self.experiment_history:
            return []
        
        similarities = []
        for exp in self.experiment_history:
            similarity = self._calculate_parameter_similarity(parameters, exp.parameters)
            similarities.append((similarity, exp))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:top_k]]
    
    def _calculate_parameter_similarity(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """Calculate similarity between parameter sets."""
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 0.0
        
        similarities = []
        for param in common_params:
            val1, val2 = params1[param], params2[param]
            max_val = max(abs(val1), abs(val2), 1e-6)
            similarity = 1.0 - abs(val1 - val2) / max_val
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _generate_parameter_variation(self, base_params: Dict[str, float], 
                                    param_space: Dict[str, Tuple]) -> Dict[str, float]:
        """Generate a variation of base parameters."""
        variation = base_params.copy()
        
        # Add small random variations
        for param, (low, high) in param_space.items():
            if param in variation:
                current_val = variation[param]
                variation_range = (high - low) * 0.1  # 10% of range
                noise = np.random.normal(0, variation_range / 3)  # 3-sigma within range
                
                new_val = current_val + noise
                new_val = max(low, min(high, new_val))  # Clamp to bounds
                variation[param] = new_val
        
        return variation
    
    def _generate_pattern_based_recommendations(self, objective, param_space: Dict[str, Tuple], 
                                              num_recommendations: int) -> List[Dict[str, float]]:
        """Generate recommendations based on learned patterns."""
        recommendations = []
        
        # Use success prediction patterns
        success_patterns = [p for p in self.patterns.values() if p.pattern_type == "success_prediction"]
        
        for pattern in success_patterns[:num_recommendations]:
            if 'parameter' in pattern.context and 'optimal_range' in pattern.context:
                param_name = pattern.context['parameter']
                optimal_range = pattern.context['optimal_range']
                
                if param_name in param_space:
                    # Generate recommendation within optimal range
                    params = {}
                    for param, (low, high) in param_space.items():
                        if param == param_name:
                            # Use optimal range, constrained by param_space
                            opt_low = max(low, optimal_range[0])
                            opt_high = min(high, optimal_range[1])
                            params[param] = np.random.uniform(opt_low, opt_high)
                        else:
                            params[param] = np.random.uniform(low, high)
                    
                    recommendations.append(params)
        
        return recommendations
    
    def _adjust_thresholds(self):
        """Adaptively adjust thresholds based on learning progress."""
        if len(self.experiment_history) > 50:
            # Adjust correlation threshold based on discovered patterns
            if len(self.patterns) < 2:
                # Lower threshold to discover more patterns
                self.adaptive_thresholds['correlation_threshold'] *= 0.95
            elif len(self.patterns) > 10:
                # Raise threshold to focus on strong patterns
                self.adaptive_thresholds['correlation_threshold'] *= 1.05
            
            # Keep threshold in reasonable range
            self.adaptive_thresholds['correlation_threshold'] = max(0.3, min(0.8, self.adaptive_thresholds['correlation_threshold']))
    
    def _detect_campaign_patterns(self, campaign_results):
        """Detect patterns at the campaign level."""
        # Analyze campaign-wide patterns like convergence behavior, exploration strategies, etc.
        pass
    
    def _update_performance_metrics(self, campaign_results):
        """Update performance metrics based on campaign results."""
        # Simple update for now
        self.performance_metrics['prediction_accuracy'] = min(1.0, self.performance_metrics['prediction_accuracy'] + 0.05)
        
    def save_learning_state(self, filepath: Path):
        """Save learning state to file."""
        try:
            state = {
                'patterns': self.patterns,
                'performance_metrics': self.performance_metrics,
                'adaptive_thresholds': self.adaptive_thresholds,
                'experiment_history': self.experiment_history[-1000:]  # Save recent history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Learning state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
    
    def load_learning_state(self, filepath: Path):
        """Load learning state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.patterns = state.get('patterns', {})
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            self.adaptive_thresholds = state.get('adaptive_thresholds', self.adaptive_thresholds)
            self.experiment_history = state.get('experiment_history', [])
            
            logger.info(f"Learning state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")

# Global learning engine instance
_global_learning_engine = None

def get_global_learning_engine() -> AdaptiveLearningEngine:
    """Get global adaptive learning engine instance."""
    global _global_learning_engine
    if _global_learning_engine is None:
        _global_learning_engine = AdaptiveLearningEngine()
    return _global_learning_engine