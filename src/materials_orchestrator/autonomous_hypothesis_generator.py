"""Autonomous Hypothesis Generation for Scientific Discovery.

This module implements advanced AI reasoning capabilities that can autonomously
generate, test, and refine scientific hypotheses based on experimental data.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HypothesisConfidence(Enum):
    """Confidence levels for generated hypotheses."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class HypothesisType(Enum):
    """Types of scientific hypotheses."""
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"  
    PREDICTIVE = "predictive"
    MECHANISTIC = "mechanistic"
    COMPOSITIONAL = "compositional"


@dataclass
class ScientificHypothesis:
    """Represents a scientific hypothesis with supporting evidence."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_text: str = ""
    hypothesis_type: HypothesisType = HypothesisType.PREDICTIVE
    confidence: HypothesisConfidence = HypothesisConfidence.LOW
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    falsifiable_predictions: List[str] = field(default_factory=list)
    experimental_tests: List[Dict[str, Any]] = field(default_factory=list)
    parameters_of_interest: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    validation_score: float = 0.0
    statistical_significance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary."""
        return {
            "id": self.id,
            "hypothesis_text": self.hypothesis_text,
            "hypothesis_type": self.hypothesis_type.value,
            "confidence": self.confidence.value,
            "supporting_evidence": self.supporting_evidence,
            "falsifiable_predictions": self.falsifiable_predictions,
            "experimental_tests": self.experimental_tests,
            "parameters_of_interest": self.parameters_of_interest,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "validation_score": self.validation_score,
            "statistical_significance": self.statistical_significance
        }


class AutonomousHypothesisGenerator:
    """AI-powered hypothesis generation for autonomous scientific discovery."""
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.7,
                 statistical_significance_threshold: float = 0.05,
                 max_hypotheses_per_session: int = 10):
        """Initialize the hypothesis generator.
        
        Args:
            min_confidence_threshold: Minimum confidence for hypothesis acceptance
            statistical_significance_threshold: P-value threshold for statistical tests
            max_hypotheses_per_session: Maximum hypotheses to generate per session
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.statistical_significance_threshold = statistical_significance_threshold
        self.max_hypotheses_per_session = max_hypotheses_per_session
        self.generated_hypotheses: List[ScientificHypothesis] = []
        self.experimental_history: List[Dict[str, Any]] = []
        self.pattern_database: Dict[str, Any] = {}
        
    def analyze_experimental_patterns(self, 
                                    experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experimental data to identify patterns and relationships.
        
        Args:
            experiments: List of experimental results
            
        Returns:
            Dictionary containing identified patterns
        """
        if len(experiments) < 5:
            logger.warning("Insufficient data for pattern analysis")
            return {}
            
        # Extract parameter-property relationships
        patterns = {
            "correlations": self._identify_correlations(experiments),
            "clusters": self._identify_clusters(experiments),
            "outliers": self._identify_outliers(experiments),
            "trends": self._identify_trends(experiments),
            "phase_spaces": self._identify_phase_spaces(experiments)
        }
        
        return patterns
    
    def _identify_correlations(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify correlations between parameters and properties."""
        correlations = {}
        
        # Extract numerical data
        params = {}
        properties = {}
        
        for exp in experiments:
            for param, value in exp.get("parameters", {}).items():
                if isinstance(value, (int, float)):
                    if param not in params:
                        params[param] = []
                    params[param].append(value)
                    
            for prop, value in exp.get("results", {}).items():
                if isinstance(value, (int, float)):
                    if prop not in properties:
                        properties[prop] = []
                    properties[prop].append(value)
        
        # Calculate correlations
        for param_name, param_values in params.items():
            for prop_name, prop_values in properties.items():
                if len(param_values) == len(prop_values) and len(param_values) > 2:
                    correlation, p_value = stats.pearsonr(param_values, prop_values)
                    
                    if abs(correlation) > 0.3 and p_value < 0.05:
                        correlations[f"{param_name}_{prop_name}"] = {
                            "correlation": correlation,
                            "p_value": p_value,
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate"
                        }
        
        return correlations
    
    def _identify_clusters(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify clusters in experimental data."""
        # Prepare data for clustering
        features = []
        feature_names = []
        
        # Extract consistent numerical features
        for exp in experiments:
            row = []
            if not feature_names:  # First iteration - establish feature names
                for param, value in exp.get("parameters", {}).items():
                    if isinstance(value, (int, float)):
                        feature_names.append(f"param_{param}")
                for prop, value in exp.get("results", {}).items():
                    if isinstance(value, (int, float)):
                        feature_names.append(f"result_{prop}")
            
            # Extract values in consistent order
            for name in feature_names:
                if name.startswith("param_"):
                    param_name = name[6:]  # Remove "param_" prefix
                    value = exp.get("parameters", {}).get(param_name, 0)
                else:  # result_
                    prop_name = name[7:]  # Remove "result_" prefix
                    value = exp.get("results", {}).get(prop_name, 0)
                row.append(float(value))
            
            features.append(row)
        
        if len(features) < 3 or len(feature_names) == 0:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(features_scaled)
        
        # Analyze clusters
        clusters = {}
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_points = [i for i, l in enumerate(cluster_labels) if l == label]
            cluster_data = [features[i] for i in cluster_points]
            
            # Calculate cluster statistics
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            clusters[f"cluster_{label}"] = {
                "size": len(cluster_points),
                "mean_values": dict(zip(feature_names, cluster_mean)),
                "std_values": dict(zip(feature_names, cluster_std)),
                "member_experiments": cluster_points
            }
        
        return clusters
    
    def _identify_outliers(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify outlier experiments that may indicate novel behavior."""
        outliers = []
        
        # Extract property values for outlier detection
        for prop_name in ["band_gap", "efficiency", "stability"]:
            values = []
            exp_indices = []
            
            for i, exp in enumerate(experiments):
                value = exp.get("results", {}).get(prop_name)
                if isinstance(value, (int, float)):
                    values.append(value)
                    exp_indices.append(i)
            
            if len(values) < 5:
                continue
                
            # Statistical outlier detection using IQR
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    exp_idx = exp_indices[i]
                    outliers.append({
                        "experiment_index": exp_idx,
                        "property": prop_name,
                        "value": value,
                        "bounds": (lower_bound, upper_bound),
                        "outlier_type": "high" if value > upper_bound else "low"
                    })
        
        return outliers
    
    def _identify_trends(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify temporal trends in experimental outcomes."""
        trends = {}
        
        # Sort experiments by timestamp if available
        sorted_experiments = sorted(
            experiments,
            key=lambda x: x.get("timestamp", datetime.now())
        )
        
        # Analyze trends for each property
        for prop_name in ["band_gap", "efficiency", "stability"]:
            values = []
            timestamps = []
            
            for exp in sorted_experiments:
                value = exp.get("results", {}).get(prop_name)
                if isinstance(value, (int, float)):
                    values.append(value)
                    timestamps.append(len(values))  # Use index as time
            
            if len(values) < 3:
                continue
                
            # Calculate trend slope
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
            
            if abs(r_value) > 0.3 and p_value < 0.1:
                trends[prop_name] = {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "p_value": p_value,
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                    "significance": "significant" if p_value < 0.05 else "marginally_significant"
                }
        
        return trends
    
    def _identify_phase_spaces(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify distinct regions in parameter space with different behaviors."""
        phase_spaces = {}
        
        # Focus on key parameters
        key_params = ["temperature", "concentration", "pressure", "time"]
        
        for param in key_params:
            param_values = []
            property_values = []
            
            for exp in experiments:
                param_val = exp.get("parameters", {}).get(param)
                prop_val = exp.get("results", {}).get("band_gap")  # Use band_gap as reference
                
                if isinstance(param_val, (int, float)) and isinstance(prop_val, (int, float)):
                    param_values.append(param_val)
                    property_values.append(prop_val)
            
            if len(param_values) < 5:
                continue
                
            # Identify phase boundaries using derivative analysis
            sorted_data = sorted(zip(param_values, property_values))
            params_sorted = [x[0] for x in sorted_data]
            props_sorted = [x[1] for x in sorted_data]
            
            # Calculate moving average derivative
            window_size = min(3, len(params_sorted) // 3)
            if window_size < 2:
                continue
                
            derivatives = []
            param_points = []
            
            for i in range(window_size, len(params_sorted) - window_size):
                x1, y1 = params_sorted[i - window_size], props_sorted[i - window_size]
                x2, y2 = params_sorted[i + window_size], props_sorted[i + window_size]
                
                if x2 != x1:
                    derivative = (y2 - y1) / (x2 - x1)
                    derivatives.append(derivative)
                    param_points.append(params_sorted[i])
            
            # Find phase transitions (rapid changes in derivative)
            if len(derivatives) > 2:
                derivative_changes = []
                for i in range(1, len(derivatives)):
                    change = abs(derivatives[i] - derivatives[i-1])
                    derivative_changes.append((param_points[i], change))
                
                # Identify significant phase boundaries
                mean_change = np.mean([x[1] for x in derivative_changes])
                std_change = np.std([x[1] for x in derivative_changes])
                threshold = mean_change + 2 * std_change
                
                phase_boundaries = [x[0] for x in derivative_changes if x[1] > threshold]
                
                if phase_boundaries:
                    phase_spaces[param] = {
                        "boundaries": phase_boundaries,
                        "num_phases": len(phase_boundaries) + 1,
                        "parameter_range": (min(param_values), max(param_values))
                    }
        
        return phase_spaces
    
    async def generate_hypotheses(self, 
                                 experiments: List[Dict[str, Any]],
                                 target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate scientific hypotheses based on experimental data.
        
        Args:
            experiments: Historical experimental data
            target_properties: Properties of interest for hypothesis generation
            
        Returns:
            List of generated hypotheses
        """
        logger.info(f"Generating hypotheses from {len(experiments)} experiments")
        
        # Analyze patterns in the data
        patterns = self.analyze_experimental_patterns(experiments)
        
        generated_hypotheses = []
        
        # Generate correlation-based hypotheses
        hypotheses = await self._generate_correlation_hypotheses(patterns, target_properties)
        generated_hypotheses.extend(hypotheses)
        
        # Generate cluster-based hypotheses
        hypotheses = await self._generate_cluster_hypotheses(patterns, target_properties)
        generated_hypotheses.extend(hypotheses)
        
        # Generate outlier-based hypotheses
        hypotheses = await self._generate_outlier_hypotheses(patterns, experiments, target_properties)
        generated_hypotheses.extend(hypotheses)
        
        # Generate trend-based hypotheses
        hypotheses = await self._generate_trend_hypotheses(patterns, target_properties)
        generated_hypotheses.extend(hypotheses)
        
        # Generate mechanistic hypotheses
        hypotheses = await self._generate_mechanistic_hypotheses(patterns, target_properties)
        generated_hypotheses.extend(hypotheses)
        
        # Filter and rank hypotheses
        filtered_hypotheses = self._filter_and_rank_hypotheses(generated_hypotheses)
        
        self.generated_hypotheses.extend(filtered_hypotheses)
        logger.info(f"Generated {len(filtered_hypotheses)} high-quality hypotheses")
        
        return filtered_hypotheses
    
    async def _generate_correlation_hypotheses(self, 
                                             patterns: Dict[str, Any],
                                             target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate hypotheses based on identified correlations."""
        hypotheses = []
        correlations = patterns.get("correlations", {})
        
        for correlation_key, correlation_data in correlations.items():
            param_name, prop_name = correlation_key.split("_", 1)
            
            if prop_name not in target_properties:
                continue
                
            correlation = correlation_data["correlation"]
            p_value = correlation_data["p_value"]
            strength = correlation_data["strength"]
            
            # Generate hypothesis text
            direction = "increases" if correlation > 0 else "decreases"
            hypothesis_text = (
                f"The parameter '{param_name}' has a {strength} {direction} "
                f"relationship with '{prop_name}' (r={correlation:.3f}, p={p_value:.4f}). "
                f"Modifying {param_name} should predictably affect {prop_name}."
            )
            
            # Determine confidence based on correlation strength and significance
            if abs(correlation) > 0.8 and p_value < 0.01:
                confidence = HypothesisConfidence.VERY_HIGH
            elif abs(correlation) > 0.6 and p_value < 0.05:
                confidence = HypothesisConfidence.HIGH
            elif abs(correlation) > 0.4:
                confidence = HypothesisConfidence.MEDIUM
            else:
                confidence = HypothesisConfidence.LOW
            
            hypothesis = ScientificHypothesis(
                hypothesis_text=hypothesis_text,
                hypothesis_type=HypothesisType.CORRELATIONAL,
                confidence=confidence,
                supporting_evidence=[{
                    "type": "correlation_analysis",
                    "correlation_coefficient": correlation,
                    "p_value": p_value,
                    "parameter": param_name,
                    "property": prop_name
                }],
                falsifiable_predictions=[
                    f"Systematically varying {param_name} should produce predictable changes in {prop_name}",
                    f"The relationship should hold across different material compositions"
                ],
                parameters_of_interest=[param_name],
                statistical_significance=p_value,
                validation_score=abs(correlation)
            )
            
            # Generate experimental tests
            if correlation > 0:
                test_direction = "increased"
                expected_outcome = "higher"
            else:
                test_direction = "decreased"
                expected_outcome = "lower"
                
            hypothesis.experimental_tests = [{
                "test_description": f"Controlled experiment with {test_direction} {param_name}",
                "expected_outcome": f"Should result in {expected_outcome} {prop_name}",
                "statistical_power": min(0.9, abs(correlation) * 1.2)
            }]
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_cluster_hypotheses(self, 
                                         patterns: Dict[str, Any],
                                         target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate hypotheses based on identified clusters."""
        hypotheses = []
        clusters = patterns.get("clusters", {})
        
        if len(clusters) < 2:
            return hypotheses
        
        # Analyze differences between clusters
        cluster_names = list(clusters.keys())
        
        for i, cluster1_name in enumerate(cluster_names):
            for cluster2_name in cluster_names[i+1:]:
                cluster1 = clusters[cluster1_name]
                cluster2 = clusters[cluster2_name]
                
                # Find parameters with significant differences
                significant_diffs = []
                
                for param in cluster1["mean_values"]:
                    if param.startswith("result_"):
                        prop_name = param[7:]  # Remove "result_" prefix
                        if prop_name not in target_properties:
                            continue
                            
                        mean1 = cluster1["mean_values"][param]
                        mean2 = cluster2["mean_values"][param]
                        std1 = cluster1["std_values"][param]
                        std2 = cluster2["std_values"][param]
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                        if pooled_std > 0:
                            effect_size = abs(mean1 - mean2) / pooled_std
                            
                            if effect_size > 0.5:  # Medium to large effect
                                significant_diffs.append({
                                    "property": prop_name,
                                    "cluster1_mean": mean1,
                                    "cluster2_mean": mean2,
                                    "effect_size": effect_size
                                })
                
                # Generate hypotheses for significant differences
                for diff in significant_diffs:
                    prop_name = diff["property"]
                    effect_size = diff["effect_size"]
                    
                    hypothesis_text = (
                        f"There exist distinct regimes in parameter space where {prop_name} "
                        f"behaves differently. Cluster analysis reveals {len(clusters)} distinct "
                        f"regions with significantly different {prop_name} values "
                        f"(effect size: {effect_size:.2f})."
                    )
                    
                    confidence = HypothesisConfidence.HIGH if effect_size > 0.8 else HypothesisConfidence.MEDIUM
                    
                    hypothesis = ScientificHypothesis(
                        hypothesis_text=hypothesis_text,
                        hypothesis_type=HypothesisType.COMPOSITIONAL,
                        confidence=confidence,
                        supporting_evidence=[{
                            "type": "cluster_analysis",
                            "num_clusters": len(clusters),
                            "effect_size": effect_size,
                            "property": prop_name
                        }],
                        falsifiable_predictions=[
                            f"New experiments in each regime should show consistent {prop_name} behavior",
                            f"Intermediate parameter values should show transitional behavior"
                        ],
                        parameters_of_interest=["all_parameters"],
                        validation_score=effect_size / 2.0  # Normalize to 0-1 range
                    )
                    
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_outlier_hypotheses(self, 
                                         patterns: Dict[str, Any],
                                         experiments: List[Dict[str, Any]],
                                         target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate hypotheses based on identified outliers."""
        hypotheses = []
        outliers = patterns.get("outliers", [])
        
        # Group outliers by property
        outlier_groups = {}
        for outlier in outliers:
            prop = outlier["property"]
            if prop not in target_properties:
                continue
                
            if prop not in outlier_groups:
                outlier_groups[prop] = []
            outlier_groups[prop].append(outlier)
        
        for prop_name, prop_outliers in outlier_groups.items():
            if len(prop_outliers) < 2:
                continue
                
            # Analyze outlier characteristics
            high_outliers = [o for o in prop_outliers if o["outlier_type"] == "high"]
            low_outliers = [o for o in prop_outliers if o["outlier_type"] == "low"]
            
            if high_outliers:
                # Analyze common characteristics of high-performing outliers
                outlier_experiments = [experiments[o["experiment_index"]] for o in high_outliers]
                common_params = self._find_common_parameters(outlier_experiments)
                
                if common_params:
                    hypothesis_text = (
                        f"Exceptional high {prop_name} values are associated with specific "
                        f"parameter combinations. Analysis of {len(high_outliers)} high-performing "
                        f"outliers reveals common characteristics that may indicate optimal conditions."
                    )
                    
                    hypothesis = ScientificHypothesis(
                        hypothesis_text=hypothesis_text,
                        hypothesis_type=HypothesisType.PREDICTIVE,
                        confidence=HypothesisConfidence.MEDIUM,
                        supporting_evidence=[{
                            "type": "outlier_analysis",
                            "num_outliers": len(high_outliers),
                            "property": prop_name,
                            "common_parameters": common_params
                        }],
                        falsifiable_predictions=[
                            f"Experiments with identified parameter combinations should consistently yield high {prop_name}",
                            f"Small variations around these conditions should maintain performance"
                        ],
                        parameters_of_interest=list(common_params.keys()),
                        validation_score=len(high_outliers) / len(experiments)  # Frequency of outliers
                    )
                    
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _find_common_parameters(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common parameter characteristics among experiments."""
        if not experiments:
            return {}
            
        common_params = {}
        
        # Get all parameter names
        all_params = set()
        for exp in experiments:
            all_params.update(exp.get("parameters", {}).keys())
        
        for param in all_params:
            values = []
            for exp in experiments:
                value = exp.get("parameters", {}).get(param)
                if isinstance(value, (int, float)):
                    values.append(value)
            
            if len(values) >= len(experiments) * 0.8:  # At least 80% have this parameter
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Consider it "common" if standard deviation is small relative to mean
                if std_val / abs(mean_val) < 0.3:  # Coefficient of variation < 30%
                    common_params[param] = {
                        "mean": mean_val,
                        "std": std_val,
                        "range": (min(values), max(values))
                    }
        
        return common_params
    
    async def _generate_trend_hypotheses(self, 
                                       patterns: Dict[str, Any],
                                       target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate hypotheses based on identified trends."""
        hypotheses = []
        trends = patterns.get("trends", {})
        
        for prop_name, trend_data in trends.items():
            if prop_name not in target_properties:
                continue
                
            slope = trend_data["slope"]
            r_squared = trend_data["r_squared"]
            p_value = trend_data["p_value"]
            direction = trend_data["trend_direction"]
            
            hypothesis_text = (
                f"There is a temporal {direction} trend in {prop_name} "
                f"(R² = {r_squared:.3f}, p = {p_value:.4f}). This suggests "
                f"either systematic improvement in experimental conditions "
                f"or underlying parameter space exploration effects."
            )
            
            confidence = HypothesisConfidence.HIGH if r_squared > 0.6 and p_value < 0.01 else HypothesisConfidence.MEDIUM
            
            hypothesis = ScientificHypothesis(
                hypothesis_text=hypothesis_text,
                hypothesis_type=HypothesisType.PREDICTIVE,
                confidence=confidence,
                supporting_evidence=[{
                    "type": "trend_analysis",
                    "slope": slope,
                    "r_squared": r_squared,
                    "p_value": p_value,
                    "property": prop_name
                }],
                falsifiable_predictions=[
                    f"Future experiments should continue the {direction} trend",
                    f"Controlling for temporal factors should reduce trend significance"
                ],
                parameters_of_interest=["temporal_factors"],
                statistical_significance=p_value,
                validation_score=r_squared
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_mechanistic_hypotheses(self, 
                                             patterns: Dict[str, Any],
                                             target_properties: List[str]) -> List[ScientificHypothesis]:
        """Generate mechanistic hypotheses based on scientific principles."""
        hypotheses = []
        correlations = patterns.get("correlations", {})
        phase_spaces = patterns.get("phase_spaces", {})
        
        # Generate mechanism-based hypotheses from strong correlations
        for correlation_key, correlation_data in correlations.items():
            param_name, prop_name = correlation_key.split("_", 1)
            
            if prop_name not in target_properties:
                continue
                
            correlation = correlation_data["correlation"]
            
            if abs(correlation) > 0.7:  # Strong correlation suggests mechanism
                mechanism_text = self._generate_mechanism_text(param_name, prop_name, correlation)
                
                if mechanism_text:
                    hypothesis_text = (
                        f"The strong correlation between {param_name} and {prop_name} "
                        f"suggests a mechanistic relationship: {mechanism_text}"
                    )
                    
                    hypothesis = ScientificHypothesis(
                        hypothesis_text=hypothesis_text,
                        hypothesis_type=HypothesisType.MECHANISTIC,
                        confidence=HypothesisConfidence.MEDIUM,
                        supporting_evidence=[{
                            "type": "mechanistic_reasoning",
                            "correlation": correlation,
                            "parameter": param_name,
                            "property": prop_name
                        }],
                        falsifiable_predictions=[
                            f"Mechanistic model should predict {prop_name} from {param_name}",
                            f"Physical constraints should limit the relationship"
                        ],
                        parameters_of_interest=[param_name],
                        validation_score=abs(correlation)
                    )
                    
                    hypotheses.append(hypothesis)
        
        # Generate phase-based mechanistic hypotheses
        for param, phase_data in phase_spaces.items():
            boundaries = phase_data["boundaries"]
            num_phases = phase_data["num_phases"]
            
            hypothesis_text = (
                f"The parameter {param} exhibits {num_phases} distinct behavioral regimes "
                f"with phase transitions at {boundaries}. This suggests underlying "
                f"physical or chemical mechanisms that change behavior at these boundaries."
            )
            
            hypothesis = ScientificHypothesis(
                hypothesis_text=hypothesis_text,
                hypothesis_type=HypothesisType.MECHANISTIC,
                confidence=HypothesisConfidence.MEDIUM,
                supporting_evidence=[{
                    "type": "phase_analysis",
                    "parameter": param,
                    "num_phases": num_phases,
                    "boundaries": boundaries
                }],
                falsifiable_predictions=[
                    f"Experiments near phase boundaries should show transitional behavior",
                    f"Each phase should have distinct parameter-property relationships"
                ],
                parameters_of_interest=[param],
                validation_score=min(1.0, num_phases / 5.0)  # More phases = higher complexity
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_mechanism_text(self, param_name: str, prop_name: str, correlation: float) -> str:
        """Generate mechanistic explanation text based on parameter-property relationships."""
        mechanisms = {
            ("temperature", "band_gap"): "Higher temperatures increase lattice vibrations, affecting electronic band structure",
            ("temperature", "efficiency"): "Temperature affects charge carrier mobility and recombination rates",
            ("concentration", "band_gap"): "Concentration changes affect quantum confinement and electronic structure",
            ("concentration", "efficiency"): "Optimal concentration balances charge injection and recombination",
            ("pressure", "band_gap"): "Pressure modifies lattice parameters and orbital overlap",
            ("time", "efficiency"): "Reaction time affects crystallinity and defect formation",
            ("ph", "stability"): "pH affects surface chemistry and degradation pathways"
        }
        
        key = (param_name.lower(), prop_name.lower())
        return mechanisms.get(key, f"changes in {param_name} directly affect {prop_name} through physical mechanisms")
    
    def _filter_and_rank_hypotheses(self, hypotheses: List[ScientificHypothesis]) -> List[ScientificHypothesis]:
        """Filter and rank hypotheses by quality and relevance."""
        # Remove low-confidence hypotheses
        filtered = [h for h in hypotheses if h.validation_score > 0.3]
        
        # Sort by combined score (confidence + validation + statistical significance)
        def hypothesis_score(h: ScientificHypothesis) -> float:
            confidence_score = {
                HypothesisConfidence.LOW: 0.25,
                HypothesisConfidence.MEDIUM: 0.5,
                HypothesisConfidence.HIGH: 0.75,
                HypothesisConfidence.VERY_HIGH: 1.0
            }[h.confidence]
            
            # Invert p-value for statistical significance (lower p-value = higher score)
            stat_score = max(0, 1 - h.statistical_significance) if h.statistical_significance > 0 else 0.5
            
            return (confidence_score * 0.4 + h.validation_score * 0.4 + stat_score * 0.2)
        
        filtered.sort(key=hypothesis_score, reverse=True)
        
        # Limit to max hypotheses per session
        return filtered[:self.max_hypotheses_per_session]
    
    async def validate_hypothesis(self, 
                                hypothesis: ScientificHypothesis,
                                new_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a hypothesis against new experimental data.
        
        Args:
            hypothesis: Hypothesis to validate
            new_experiments: New experimental data for validation
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "hypothesis_id": hypothesis.id,
            "validation_score": 0.0,
            "statistical_tests": [],
            "predictions_tested": [],
            "overall_support": "insufficient_data"
        }
        
        if len(new_experiments) < 3:
            return validation_results
        
        # Test specific predictions based on hypothesis type
        if hypothesis.hypothesis_type == HypothesisType.CORRELATIONAL:
            validation_results = await self._validate_correlation_hypothesis(hypothesis, new_experiments)
        elif hypothesis.hypothesis_type == HypothesisType.PREDICTIVE:
            validation_results = await self._validate_predictive_hypothesis(hypothesis, new_experiments)
        elif hypothesis.hypothesis_type == HypothesisType.MECHANISTIC:
            validation_results = await self._validate_mechanistic_hypothesis(hypothesis, new_experiments)
        
        # Update hypothesis with validation results
        hypothesis.validation_score = validation_results["validation_score"]
        hypothesis.last_updated = datetime.now()
        
        return validation_results
    
    async def _validate_correlation_hypothesis(self, 
                                             hypothesis: ScientificHypothesis,
                                             experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a correlation-based hypothesis."""
        # Extract the parameter-property pair from supporting evidence
        evidence = hypothesis.supporting_evidence[0]
        param_name = evidence["parameter"]
        prop_name = evidence["property"]
        expected_correlation = evidence["correlation_coefficient"]
        
        # Calculate correlation in new data
        param_values = []
        prop_values = []
        
        for exp in experiments:
            param_val = exp.get("parameters", {}).get(param_name)
            prop_val = exp.get("results", {}).get(prop_name)
            
            if isinstance(param_val, (int, float)) and isinstance(prop_val, (int, float)):
                param_values.append(param_val)
                prop_values.append(prop_val)
        
        if len(param_values) < 3:
            return {"validation_score": 0.0, "overall_support": "insufficient_data"}
        
        observed_correlation, p_value = stats.pearsonr(param_values, prop_values)
        
        # Compare with expected correlation
        correlation_agreement = 1.0 - abs(expected_correlation - observed_correlation) / 2.0
        significance_support = 1.0 if p_value < 0.05 else 0.5
        
        validation_score = correlation_agreement * significance_support
        
        support_level = "strong" if validation_score > 0.7 else "moderate" if validation_score > 0.4 else "weak"
        
        return {
            "validation_score": validation_score,
            "statistical_tests": [{
                "test_type": "correlation",
                "expected_correlation": expected_correlation,
                "observed_correlation": observed_correlation,
                "p_value": p_value
            }],
            "overall_support": support_level
        }
    
    async def _validate_predictive_hypothesis(self, 
                                            hypothesis: ScientificHypothesis,
                                            experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a predictive hypothesis."""
        # For trend-based predictions
        if "temporal" in hypothesis.hypothesis_text.lower():
            # Check if trend continues
            prop_values = []
            timestamps = []
            
            for exp in experiments:
                for prop_name in ["band_gap", "efficiency", "stability"]:
                    prop_val = exp.get("results", {}).get(prop_name)
                    if isinstance(prop_val, (int, float)):
                        prop_values.append(prop_val)
                        timestamps.append(len(prop_values))
                        break  # Use first available property
            
            if len(prop_values) >= 3:
                slope, _, r_value, p_value, _ = stats.linregress(timestamps, prop_values)
                
                # Check if trend direction matches prediction
                expected_direction = "increasing" if "increasing" in hypothesis.hypothesis_text else "decreasing"
                observed_direction = "increasing" if slope > 0 else "decreasing"
                
                direction_match = expected_direction == observed_direction
                trend_strength = abs(r_value)
                significance = 1.0 if p_value < 0.05 else 0.5
                
                validation_score = direction_match * trend_strength * significance
                support_level = "strong" if validation_score > 0.6 else "moderate" if validation_score > 0.3 else "weak"
                
                return {
                    "validation_score": validation_score,
                    "statistical_tests": [{
                        "test_type": "trend_analysis",
                        "expected_direction": expected_direction,
                        "observed_direction": observed_direction,
                        "slope": slope,
                        "r_value": r_value,
                        "p_value": p_value
                    }],
                    "overall_support": support_level
                }
        
        return {"validation_score": 0.5, "overall_support": "partial"}
    
    async def _validate_mechanistic_hypothesis(self, 
                                             hypothesis: ScientificHypothesis,
                                             experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a mechanistic hypothesis."""
        # Mechanistic hypotheses are harder to validate directly
        # Use correlation strength and consistency as proxies
        
        validation_score = 0.6  # Default moderate support for mechanistic hypotheses
        
        # Look for consistency in parameter-property relationships
        for evidence in hypothesis.supporting_evidence:
            if evidence["type"] == "mechanistic_reasoning":
                param_name = evidence["parameter"]
                prop_name = evidence["property"]
                
                # Check if relationship holds in new data
                param_values = []
                prop_values = []
                
                for exp in experiments:
                    param_val = exp.get("parameters", {}).get(param_name)
                    prop_val = exp.get("results", {}).get(prop_name)
                    
                    if isinstance(param_val, (int, float)) and isinstance(prop_val, (int, float)):
                        param_values.append(param_val)
                        prop_values.append(prop_val)
                
                if len(param_values) >= 3:
                    correlation, p_value = stats.pearsonr(param_values, prop_values)
                    expected_correlation = evidence["correlation"]
                    
                    consistency = 1.0 - abs(expected_correlation - correlation) / 2.0
                    validation_score = max(validation_score, consistency)
        
        support_level = "moderate" if validation_score > 0.5 else "weak"
        
        return {
            "validation_score": validation_score,
            "statistical_tests": [],
            "overall_support": support_level
        }
    
    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """Get summary of all generated hypotheses."""
        if not self.generated_hypotheses:
            return {"total_hypotheses": 0, "summary": "No hypotheses generated yet."}
        
        # Group by type and confidence
        by_type = {}
        by_confidence = {}
        
        for h in self.generated_hypotheses:
            type_key = h.hypothesis_type.value
            conf_key = h.confidence.value
            
            by_type[type_key] = by_type.get(type_key, 0) + 1
            by_confidence[conf_key] = by_confidence.get(conf_key, 0) + 1
        
        # Calculate average validation score
        avg_validation = np.mean([h.validation_score for h in self.generated_hypotheses])
        
        # Find highest confidence hypotheses
        top_hypotheses = sorted(
            self.generated_hypotheses,
            key=lambda x: x.validation_score,
            reverse=True
        )[:3]
        
        return {
            "total_hypotheses": len(self.generated_hypotheses),
            "by_type": by_type,
            "by_confidence": by_confidence,
            "average_validation_score": avg_validation,
            "top_hypotheses": [h.hypothesis_text for h in top_hypotheses],
            "summary": f"Generated {len(self.generated_hypotheses)} hypotheses with average validation score {avg_validation:.2f}"
        }


# Global instance for easy access
_global_hypothesis_generator: Optional[AutonomousHypothesisGenerator] = None


def get_global_hypothesis_generator() -> AutonomousHypothesisGenerator:
    """Get the global hypothesis generator instance."""
    global _global_hypothesis_generator
    if _global_hypothesis_generator is None:
        _global_hypothesis_generator = AutonomousHypothesisGenerator()
    return _global_hypothesis_generator


async def generate_scientific_hypotheses(experiments: List[Dict[str, Any]],
                                       target_properties: List[str] = None) -> List[ScientificHypothesis]:
    """Convenience function to generate hypotheses from experimental data.
    
    Args:
        experiments: List of experimental results
        target_properties: Properties to focus on (default: common materials properties)
        
    Returns:
        List of generated scientific hypotheses
    """
    if target_properties is None:
        target_properties = ["band_gap", "efficiency", "stability"]
    
    generator = get_global_hypothesis_generator()
    return await generator.generate_hypotheses(experiments, target_properties)