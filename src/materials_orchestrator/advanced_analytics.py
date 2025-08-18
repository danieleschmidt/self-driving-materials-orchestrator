"""Advanced analytics and insights for materials discovery campaigns."""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CampaignAnalytics:
    """Comprehensive analytics for discovery campaigns."""

    campaign_id: str
    start_time: datetime
    total_experiments: int
    successful_experiments: int
    target_property: str

    # Performance metrics
    success_rate: float = 0.0
    acceleration_factor: float = 0.0
    cost_efficiency: float = 0.0
    discovery_rate: float = 0.0

    # Statistical analysis
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    property_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Optimization insights
    convergence_analysis: Dict[str, Any] = field(default_factory=dict)
    exploration_coverage: float = 0.0
    exploitation_efficiency: float = 0.0

    # Quality metrics
    reproducibility_score: float = 0.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    outlier_detection: List[str] = field(default_factory=list)


class AdvancedAnalyzer:
    """Advanced analytics engine for materials discovery."""

    def __init__(self):
        """Initialize advanced analyzer."""
        self.analysis_cache = {}
        self.benchmark_data = {}

    def analyze_campaign(self, campaign_results) -> CampaignAnalytics:
        """Perform comprehensive campaign analysis."""
        logger.info(f"Analyzing campaign {campaign_results.campaign_id}")

        analytics = CampaignAnalytics(
            campaign_id=campaign_results.campaign_id,
            start_time=campaign_results.start_time,
            total_experiments=campaign_results.total_experiments,
            successful_experiments=campaign_results.successful_experiments,
            target_property=campaign_results.objective.target_property,
        )

        # Calculate performance metrics
        analytics.success_rate = self._calculate_success_rate(campaign_results)
        analytics.acceleration_factor = self._calculate_acceleration_factor(
            campaign_results
        )
        analytics.cost_efficiency = self._calculate_cost_efficiency(campaign_results)
        analytics.discovery_rate = self._calculate_discovery_rate(campaign_results)

        # Statistical analysis
        analytics.parameter_importance = self._analyze_parameter_importance(
            campaign_results
        )
        analytics.correlation_matrix = self._calculate_correlation_matrix(
            campaign_results
        )
        analytics.property_statistics = self._calculate_property_statistics(
            campaign_results
        )

        # Optimization insights
        analytics.convergence_analysis = self._analyze_convergence(campaign_results)
        analytics.exploration_coverage = self._calculate_exploration_coverage(
            campaign_results
        )
        analytics.exploitation_efficiency = self._calculate_exploitation_efficiency(
            campaign_results
        )

        # Quality metrics
        analytics.reproducibility_score = self._calculate_reproducibility(
            campaign_results
        )
        analytics.confidence_intervals = self._calculate_confidence_intervals(
            campaign_results
        )
        analytics.outlier_detection = self._detect_outliers(campaign_results)

        return analytics

    def _calculate_success_rate(self, campaign_results) -> float:
        """Calculate experiment success rate."""
        if campaign_results.total_experiments == 0:
            return 0.0
        return (
            campaign_results.successful_experiments / campaign_results.total_experiments
        )

    def _calculate_acceleration_factor(self, campaign_results) -> float:
        """Calculate acceleration compared to traditional methods."""
        # Estimate traditional method performance
        traditional_estimate = 200  # Typical grid/random search
        if campaign_results.total_experiments == 0:
            return 0.0
        return traditional_estimate / campaign_results.total_experiments

    def _calculate_cost_efficiency(self, campaign_results) -> float:
        """Calculate cost efficiency score."""
        # Cost per successful experiment
        if campaign_results.successful_experiments == 0:
            return 0.0

        # Assume base cost per experiment
        base_cost = 100  # arbitrary units
        total_cost = campaign_results.total_experiments * base_cost
        cost_per_success = total_cost / campaign_results.successful_experiments

        # Lower cost per success = higher efficiency
        return 1000 / cost_per_success if cost_per_success > 0 else 0.0

    def _calculate_discovery_rate(self, campaign_results) -> float:
        """Calculate discovery rate (successful experiments per hour)."""
        if not campaign_results.duration or campaign_results.duration == 0:
            return 0.0
        return campaign_results.successful_experiments / campaign_results.duration

    def _analyze_parameter_importance(self, campaign_results) -> Dict[str, float]:
        """Analyze relative importance of parameters using sensitivity analysis."""
        experiments = campaign_results.experiments
        if not experiments:
            return {}

        # Extract successful experiments
        successful_exp = [exp for exp in experiments if exp.status == "completed"]
        if len(successful_exp) < 3:
            return {}

        target_property = campaign_results.objective.target_property
        importance = {}

        # Get all parameter names
        if successful_exp:
            param_names = list(successful_exp[0].parameters.keys())
        else:
            return {}

        # Calculate variance-based importance for each parameter
        for param in param_names:
            try:
                param_values = [exp.parameters.get(param, 0) for exp in successful_exp]
                property_values = [
                    exp.results.get(target_property, 0) for exp in successful_exp
                ]

                if len(set(param_values)) > 1:  # Parameter has variation
                    # Calculate correlation coefficient
                    correlation = self._calculate_correlation(
                        param_values, property_values
                    )
                    importance[param] = abs(correlation)
                else:
                    importance[param] = 0.0

            except Exception as e:
                logger.warning(f"Error calculating importance for {param}: {e}")
                importance[param] = 0.0

        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        try:
            n = len(x_values)
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n

            numerator = sum(
                (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
            )
            x_var = sum((x - x_mean) ** 2 for x in x_values)
            y_var = sum((y - y_mean) ** 2 for y in y_values)

            denominator = math.sqrt(x_var * y_var)

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception:
            return 0.0

    def _calculate_correlation_matrix(
        self, campaign_results
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between all parameters and properties."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if len(experiments) < 3:
            return {}

        # Get all variables (parameters + properties)
        all_vars = set()
        for exp in experiments:
            all_vars.update(exp.parameters.keys())
            all_vars.update(exp.results.keys())

        all_vars = list(all_vars)
        correlation_matrix = {}

        for var1 in all_vars:
            correlation_matrix[var1] = {}
            for var2 in all_vars:
                # Get values for both variables
                values1 = []
                values2 = []

                for exp in experiments:
                    val1 = exp.parameters.get(var1) or exp.results.get(var1)
                    val2 = exp.parameters.get(var2) or exp.results.get(var2)

                    if val1 is not None and val2 is not None:
                        values1.append(float(val1))
                        values2.append(float(val2))

                correlation_matrix[var1][var2] = self._calculate_correlation(
                    values1, values2
                )

        return correlation_matrix

    def _calculate_property_statistics(
        self, campaign_results
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary for all measured properties."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if not experiments:
            return {}

        # Get all properties
        all_properties = set()
        for exp in experiments:
            all_properties.update(exp.results.keys())

        statistics_dict = {}

        for prop in all_properties:
            values = []
            for exp in experiments:
                if prop in exp.results:
                    values.append(float(exp.results[prop]))

            if values:
                statistics_dict[prop] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "variance": statistics.variance(values) if len(values) > 1 else 0.0,
                }

        return statistics_dict

    def _analyze_convergence(self, campaign_results) -> Dict[str, Any]:
        """Analyze optimization convergence behavior."""
        convergence_history = campaign_results.convergence_history
        if not convergence_history:
            return {}

        fitness_values = [entry["best_fitness"] for entry in convergence_history]
        experiments = [entry["experiment"] for entry in convergence_history]

        analysis = {
            "total_improvements": len(
                [
                    i
                    for i in range(1, len(fitness_values))
                    if fitness_values[i] > fitness_values[i - 1]
                ]
            ),
            "final_fitness": fitness_values[-1] if fitness_values else 0,
            "improvement_rate": 0.0,
            "convergence_point": None,
            "plateau_analysis": {},
        }

        # Calculate improvement rate
        if len(fitness_values) > 1:
            analysis["improvement_rate"] = (
                fitness_values[-1] - fitness_values[0]
            ) / len(fitness_values)

        # Find convergence point (where improvements become rare)
        window_size = min(10, len(fitness_values) // 4)
        if window_size >= 2:
            for i in range(window_size, len(fitness_values)):
                recent_window = fitness_values[i - window_size : i]
                if len(set(recent_window)) == 1:  # No improvement in window
                    analysis["convergence_point"] = experiments[i]
                    break

        # Plateau analysis
        if len(fitness_values) >= 5:
            last_quarter = fitness_values[-len(fitness_values) // 4 :]
            analysis["plateau_analysis"] = {
                "plateau_length": len(last_quarter),
                "plateau_variance": (
                    statistics.variance(last_quarter) if len(last_quarter) > 1 else 0
                ),
                "is_converged": (
                    statistics.variance(last_quarter) < 0.001
                    if len(last_quarter) > 1
                    else False
                ),
            }

        return analysis

    def _calculate_exploration_coverage(self, campaign_results) -> float:
        """Calculate how well the parameter space was explored."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if not experiments:
            return 0.0

        # Get parameter ranges from first experiment
        if not experiments:
            return 0.0

        param_names = list(experiments[0].parameters.keys())
        if not param_names:
            return 0.0

        # Calculate coverage for each parameter
        coverages = []

        for param in param_names:
            values = [exp.parameters.get(param, 0) for exp in experiments]
            if values and len(set(values)) > 1:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    # Simple coverage metric: range covered
                    coverage = (max_val - min_val) / max_val if max_val != 0 else 0
                    coverages.append(min(coverage, 1.0))

        return statistics.mean(coverages) if coverages else 0.0

    def _calculate_exploitation_efficiency(self, campaign_results) -> float:
        """Calculate how efficiently the algorithm exploited promising regions."""
        convergence_history = campaign_results.convergence_history
        if len(convergence_history) < 2:
            return 0.0

        # Calculate improvement per experiment in later stages
        later_half = convergence_history[len(convergence_history) // 2 :]
        if len(later_half) < 2:
            return 0.0

        improvements = 0
        for i in range(1, len(later_half)):
            if later_half[i]["best_fitness"] > later_half[i - 1]["best_fitness"]:
                improvements += 1

        return improvements / (len(later_half) - 1) if len(later_half) > 1 else 0.0

    def _calculate_reproducibility(self, campaign_results) -> float:
        """Calculate reproducibility score based on experiment consistency."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if len(experiments) < 5:
            return 0.0

        target_property = campaign_results.objective.target_property

        # Group experiments by similar parameters
        parameter_groups = self._group_similar_experiments(experiments, tolerance=0.1)

        reproducibility_scores = []
        for group in parameter_groups:
            if len(group) >= 2:
                property_values = [exp.results.get(target_property, 0) for exp in group]
                if property_values:
                    # Calculate coefficient of variation
                    mean_val = statistics.mean(property_values)
                    if mean_val != 0:
                        std_val = (
                            statistics.stdev(property_values)
                            if len(property_values) > 1
                            else 0
                        )
                        cv = std_val / abs(mean_val)
                        reproducibility_scores.append(
                            1.0 / (1.0 + cv)
                        )  # Higher score for lower variation

        return (
            statistics.mean(reproducibility_scores) if reproducibility_scores else 0.5
        )

    def _group_similar_experiments(
        self, experiments: List, tolerance: float = 0.1
    ) -> List[List]:
        """Group experiments with similar parameters."""
        groups = []
        used_experiments = set()

        for i, exp1 in enumerate(experiments):
            if i in used_experiments:
                continue

            group = [exp1]
            used_experiments.add(i)

            for j, exp2 in enumerate(experiments[i + 1 :], i + 1):
                if j in used_experiments:
                    continue

                # Check if parameters are similar
                if self._parameters_similar(
                    exp1.parameters, exp2.parameters, tolerance
                ):
                    group.append(exp2)
                    used_experiments.add(j)

            groups.append(group)

        return groups

    def _parameters_similar(
        self, params1: Dict, params2: Dict, tolerance: float
    ) -> bool:
        """Check if two parameter sets are similar within tolerance."""
        if set(params1.keys()) != set(params2.keys()):
            return False

        for key in params1:
            val1, val2 = params1[key], params2[key]
            if abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6) > tolerance:
                return False

        return True

    def _calculate_confidence_intervals(
        self, campaign_results
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate 95% confidence intervals for properties."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if len(experiments) < 3:
            return {}

        all_properties = set()
        for exp in experiments:
            all_properties.update(exp.results.keys())

        confidence_intervals = {}

        for prop in all_properties:
            values = [exp.results[prop] for exp in experiments if prop in exp.results]
            if len(values) >= 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                n = len(values)

                # 95% confidence interval (assuming t-distribution)
                t_critical = 1.96  # Approximation for large n
                margin_error = t_critical * std_val / math.sqrt(n)

                confidence_intervals[prop] = (
                    mean_val - margin_error,
                    mean_val + margin_error,
                )

        return confidence_intervals

    def _detect_outliers(self, campaign_results) -> List[str]:
        """Detect outlier experiments using statistical methods."""
        experiments = [
            exp for exp in campaign_results.experiments if exp.status == "completed"
        ]
        if len(experiments) < 5:
            return []

        target_property = campaign_results.objective.target_property
        values = [
            exp.results.get(target_property, 0)
            for exp in experiments
            if exp.results.get(target_property) is not None
        ]

        if len(values) < 5:
            return []

        # Calculate IQR-based outliers
        sorted_values = sorted(values)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_ids = []
        for exp in experiments:
            value = exp.results.get(target_property)
            if value is not None and (value < lower_bound or value > upper_bound):
                outlier_ids.append(exp.id)

        return outlier_ids

    def generate_insights_report(self, analytics: CampaignAnalytics) -> Dict[str, Any]:
        """Generate actionable insights from analytics."""
        insights = {
            "performance_summary": {
                "overall_rating": self._calculate_overall_rating(analytics),
                "key_strengths": [],
                "improvement_areas": [],
                "recommendations": [],
            },
            "parameter_insights": {
                "most_important_parameters": [],
                "parameter_optimization_suggestions": [],
                "correlation_insights": [],
            },
            "optimization_insights": {
                "convergence_assessment": "",
                "exploration_assessment": "",
                "next_campaign_suggestions": [],
            },
            "quality_assessment": {
                "reproducibility_rating": self._rate_reproducibility(
                    analytics.reproducibility_score
                ),
                "data_quality_issues": [],
                "confidence_assessment": {},
            },
        }

        # Performance insights
        if analytics.success_rate > 0.9:
            insights["performance_summary"]["key_strengths"].append(
                "Excellent experiment success rate"
            )
        elif analytics.success_rate < 0.7:
            insights["performance_summary"]["improvement_areas"].append(
                "Low experiment success rate"
            )
            insights["performance_summary"]["recommendations"].append(
                "Review experimental protocols"
            )

        if analytics.acceleration_factor > 3:
            insights["performance_summary"]["key_strengths"].append(
                f"Strong acceleration ({analytics.acceleration_factor:.1f}x)"
            )

        # Parameter insights
        if analytics.parameter_importance:
            sorted_params = sorted(
                analytics.parameter_importance.items(), key=lambda x: x[1], reverse=True
            )
            insights["parameter_insights"]["most_important_parameters"] = sorted_params[
                :3
            ]

            # Suggest focusing on most important parameters
            if sorted_params[0][1] > 0.4:
                insights["parameter_insights"][
                    "parameter_optimization_suggestions"
                ].append(
                    f"Focus optimization on {sorted_params[0][0]} (high importance: {sorted_params[0][1]:.2f})"
                )

        # Convergence insights
        if analytics.convergence_analysis.get("is_converged"):
            insights["optimization_insights"][
                "convergence_assessment"
            ] = "Algorithm converged successfully"
        else:
            insights["optimization_insights"][
                "convergence_assessment"
            ] = "Algorithm may benefit from more iterations"
            insights["optimization_insights"]["next_campaign_suggestions"].append(
                "Continue optimization with more experiments"
            )

        return insights

    def _calculate_overall_rating(self, analytics: CampaignAnalytics) -> str:
        """Calculate overall campaign rating."""
        score = 0

        # Success rate (40% weight)
        score += analytics.success_rate * 0.4

        # Acceleration factor (30% weight) - capped at 10x
        score += min(analytics.acceleration_factor / 10.0, 1.0) * 0.3

        # Reproducibility (20% weight)
        score += analytics.reproducibility_score * 0.2

        # Exploration coverage (10% weight)
        score += analytics.exploration_coverage * 0.1

        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"

    def _rate_reproducibility(self, score: float) -> str:
        """Rate reproducibility score."""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Moderate"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"


# Global analyzer instance
_global_analyzer = None


def get_global_analyzer() -> AdvancedAnalyzer:
    """Get global advanced analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = AdvancedAnalyzer()
    return _global_analyzer
