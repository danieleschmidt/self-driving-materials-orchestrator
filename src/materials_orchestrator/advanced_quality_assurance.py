"""Advanced Quality Assurance System for Materials Discovery.

Comprehensive quality control, validation, and assurance system for
autonomous materials discovery campaigns with statistical analysis.
"""

import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

# Graceful dependency handling
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    # Minimal numpy-like implementation
    class np:
        @staticmethod
        def array(data):
            return list(data)

        @staticmethod
        def mean(data):
            return statistics.mean(data) if data else 0

        @staticmethod
        def std(data):
            return statistics.stdev(data) if len(data) > 1 else 0

        @staticmethod
        def percentile(data, q):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * q / 100
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Quality metrics for materials discovery."""

    REPRODUCIBILITY = "reproducibility"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    EXPERIMENTAL_VALIDITY = "experimental_validity"
    DATA_INTEGRITY = "data_integrity"


class QualityLevel(Enum):
    """Quality levels for assessment."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityAssessment:
    """Represents a quality assessment result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric: QualityMetric = QualityMetric.REPRODUCIBILITY
    score: float = 0.0  # 0.0 to 1.0
    level: QualityLevel = QualityLevel.POOR
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "metric": self.metric.value,
            "score": self.score,
            "level": self.level.value,
            "details": self.details,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class AdvancedQualityAssurance:
    """Advanced quality assurance system for materials discovery."""

    def __init__(
        self,
        min_reproducibility_score: float = 0.8,
        min_statistical_significance: float = 0.05,
        min_data_completeness: float = 0.9,
        enable_automated_validation: bool = True,
    ):
        """Initialize quality assurance system.

        Args:
            min_reproducibility_score: Minimum reproducibility score (0.0-1.0)
            min_statistical_significance: Maximum p-value for significance
            min_data_completeness: Minimum data completeness ratio
            enable_automated_validation: Enable automated validation checks
        """
        self.min_reproducibility_score = min_reproducibility_score
        self.min_statistical_significance = min_statistical_significance
        self.min_data_completeness = min_data_completeness
        self.enable_automated_validation = enable_automated_validation

        # Quality tracking
        self.quality_assessments: List[QualityAssessment] = []
        self.quality_history: Dict[str, List[float]] = {}

        # Validation rules
        self.validation_rules = self._initialize_validation_rules()

        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.ACCEPTABLE: 0.7,
            QualityLevel.POOR: 0.5,
            QualityLevel.CRITICAL: 0.0,
        }

        logger.info("Advanced Quality Assurance system initialized")

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different aspects."""
        return {
            "parameter_ranges": {
                "temperature": (50, 500),  # Celsius
                "pressure": (0.1, 100),  # bar
                "concentration": (0.001, 10.0),  # M
                "time": (0.1, 48),  # hours
                "pH": (0, 14),
            },
            "result_ranges": {
                "band_gap": (0.1, 6.0),  # eV
                "efficiency": (0, 50),  # %
                "stability": (0, 1.0),  # fraction
                "conductivity": (1e-12, 1e6),  # S/cm
            },
            "physical_constraints": [
                "efficiency_bandgap_correlation",
                "stability_temperature_relationship",
                "concentration_solubility_limits",
            ],
        }

    async def assess_campaign_quality(
        self, experiments: List[Dict[str, Any]], campaign_id: str
    ) -> Dict[str, QualityAssessment]:
        """Assess overall quality of a materials discovery campaign.

        Args:
            experiments: List of experiment data
            campaign_id: Campaign identifier

        Returns:
            Dictionary of quality assessments by metric
        """
        logger.info(f"Assessing quality for campaign {campaign_id}")

        assessments = {}

        # Assess different quality metrics
        assessments[QualityMetric.REPRODUCIBILITY] = await self._assess_reproducibility(
            experiments
        )
        assessments[QualityMetric.ACCURACY] = await self._assess_accuracy(experiments)
        assessments[QualityMetric.PRECISION] = await self._assess_precision(experiments)
        assessments[QualityMetric.COMPLETENESS] = await self._assess_completeness(
            experiments
        )
        assessments[QualityMetric.CONSISTENCY] = await self._assess_consistency(
            experiments
        )
        assessments[QualityMetric.STATISTICAL_SIGNIFICANCE] = (
            await self._assess_statistical_significance(experiments)
        )
        assessments[QualityMetric.EXPERIMENTAL_VALIDITY] = (
            await self._assess_experimental_validity(experiments)
        )
        assessments[QualityMetric.DATA_INTEGRITY] = await self._assess_data_integrity(
            experiments
        )

        # Store assessments
        for assessment in assessments.values():
            self.quality_assessments.append(assessment)

        # Update quality history
        self._update_quality_history(campaign_id, assessments)

        return assessments

    async def _assess_reproducibility(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess reproducibility of experimental results."""

        # Group experiments by similar parameters
        parameter_groups = self._group_similar_experiments(experiments)

        reproducibility_scores = []

        for group in parameter_groups:
            if len(group) < 2:
                continue

            # Calculate coefficient of variation for each result property
            property_cvs = []

            for prop_name in ["band_gap", "efficiency", "stability"]:
                values = []
                for exp in group:
                    prop_value = exp.get("results", {}).get(prop_name)
                    if prop_value is not None:
                        values.append(float(prop_value))

                if len(values) >= 2:
                    cv = self._coefficient_of_variation(values)
                    property_cvs.append(cv)

            if property_cvs:
                # Good reproducibility has low CV
                group_reproducibility = 1.0 - min(np.mean(property_cvs), 1.0)
                reproducibility_scores.append(group_reproducibility)

        if not reproducibility_scores:
            overall_score = 0.0
            recommendations = [
                "Insufficient replicate experiments for reproducibility assessment"
            ]
        else:
            overall_score = np.mean(reproducibility_scores)
            recommendations = []

            if overall_score < self.min_reproducibility_score:
                recommendations.append(
                    f"Reproducibility below threshold ({overall_score:.3f} < {self.min_reproducibility_score})"
                )
                recommendations.append("Increase number of replicate experiments")
                recommendations.append("Review experimental protocols for consistency")

        return QualityAssessment(
            metric=QualityMetric.REPRODUCIBILITY,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "group_count": len(parameter_groups),
                "assessed_groups": len([g for g in parameter_groups if len(g) >= 2]),
                "group_scores": reproducibility_scores,
                "coefficient_of_variations": (
                    property_cvs if "property_cvs" in locals() else []
                ),
            },
            recommendations=recommendations,
        )

    async def _assess_accuracy(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess accuracy of experimental measurements."""

        accuracy_scores = []
        accuracy_details = {}

        # Check against known standards or theoretical values
        for exp in experiments:
            results = exp.get("results", {})
            parameters = exp.get("parameters", {})

            # Validate against physical constraints
            physical_validity = self._check_physical_validity(results, parameters)
            accuracy_scores.append(physical_validity)

        overall_score = np.mean(accuracy_scores) if accuracy_scores else 0.0

        recommendations = []
        if overall_score < 0.8:
            recommendations.append("Some results may be physically unrealistic")
            recommendations.append("Review measurement calibration and protocols")
            recommendations.append("Validate against known standards")

        return QualityAssessment(
            metric=QualityMetric.ACCURACY,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "individual_scores": accuracy_scores,
                "physical_validity_checks": len(accuracy_scores),
            },
            recommendations=recommendations,
        )

    async def _assess_precision(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess precision of experimental measurements."""

        # Calculate precision based on replicate measurements
        precision_scores = []

        # Group by similar conditions
        groups = self._group_similar_experiments(experiments, tolerance=0.05)

        for group in groups:
            if len(group) < 3:  # Need at least 3 for precision assessment
                continue

            # Calculate standard deviation for each property
            for prop_name in ["band_gap", "efficiency", "stability"]:
                values = []
                for exp in group:
                    prop_value = exp.get("results", {}).get(prop_name)
                    if prop_value is not None:
                        values.append(float(prop_value))

                if len(values) >= 3:
                    # Precision score based on relative standard deviation
                    rel_std = (
                        np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
                    )
                    precision_score = max(0.0, 1.0 - rel_std * 5)  # Scale factor
                    precision_scores.append(precision_score)

        overall_score = np.mean(precision_scores) if precision_scores else 0.5

        recommendations = []
        if overall_score < 0.7:
            recommendations.append("Low measurement precision detected")
            recommendations.append("Check instrument calibration and stability")
            recommendations.append("Increase measurement repeats")

        return QualityAssessment(
            metric=QualityMetric.PRECISION,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "precision_groups": len(groups),
                "precision_scores": precision_scores,
            },
            recommendations=recommendations,
        )

    async def _assess_completeness(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess completeness of experimental data."""

        total_experiments = len(experiments)
        complete_experiments = 0
        missing_data_summary = {}

        expected_parameters = ["temperature", "concentration", "time"]
        expected_results = ["band_gap", "efficiency", "stability"]

        for exp in experiments:
            parameters = exp.get("parameters", {})
            results = exp.get("results", {})

            # Check parameter completeness
            param_completeness = sum(
                1 for p in expected_parameters if p in parameters
            ) / len(expected_parameters)

            # Check result completeness
            result_completeness = sum(
                1 for r in expected_results if r in results
            ) / len(expected_results)

            # Overall experiment completeness
            exp_completeness = (param_completeness + result_completeness) / 2

            if exp_completeness >= self.min_data_completeness:
                complete_experiments += 1

            # Track missing data
            for param in expected_parameters:
                if param not in parameters:
                    missing_data_summary[f"missing_{param}"] = (
                        missing_data_summary.get(f"missing_{param}", 0) + 1
                    )

            for result in expected_results:
                if result not in results:
                    missing_data_summary[f"missing_{result}"] = (
                        missing_data_summary.get(f"missing_{result}", 0) + 1
                    )

        overall_score = (
            complete_experiments / total_experiments if total_experiments > 0 else 0.0
        )

        recommendations = []
        if overall_score < self.min_data_completeness:
            recommendations.append(
                f"Data completeness below threshold ({overall_score:.3f} < {self.min_data_completeness})"
            )
            recommendations.append(
                "Ensure all required parameters and results are recorded"
            )

            # Specific recommendations based on missing data
            for key, count in missing_data_summary.items():
                if count > total_experiments * 0.1:  # More than 10% missing
                    recommendations.append(
                        f"Frequently missing: {key.replace('missing_', '')} ({count}/{total_experiments} experiments)"
                    )

        return QualityAssessment(
            metric=QualityMetric.COMPLETENESS,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "total_experiments": total_experiments,
                "complete_experiments": complete_experiments,
                "missing_data_summary": missing_data_summary,
                "completeness_threshold": self.min_data_completeness,
            },
            recommendations=recommendations,
        )

    async def _assess_consistency(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess consistency of experimental protocols and data."""

        consistency_scores = []

        # Check parameter value consistency
        parameter_ranges = {}
        for exp in experiments:
            for param, value in exp.get("parameters", {}).items():
                if param not in parameter_ranges:
                    parameter_ranges[param] = []
                try:
                    parameter_ranges[param].append(float(value))
                except (ValueError, TypeError):
                    pass

        # Assess if parameter ranges are reasonable
        for param, values in parameter_ranges.items():
            if len(values) > 1:
                cv = self._coefficient_of_variation(values)
                # Good consistency means controlled variation
                consistency_score = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
                consistency_scores.append(consistency_score)

        # Check result consistency with physical laws
        physics_consistency = self._check_physics_consistency(experiments)
        consistency_scores.append(physics_consistency)

        overall_score = np.mean(consistency_scores) if consistency_scores else 0.5

        recommendations = []
        if overall_score < 0.7:
            recommendations.append("Inconsistencies detected in experimental data")
            recommendations.append("Review experimental protocols for standardization")
            recommendations.append("Check for systematic errors or outliers")

        return QualityAssessment(
            metric=QualityMetric.CONSISTENCY,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "parameter_consistency_scores": (
                    consistency_scores[:-1] if consistency_scores else []
                ),
                "physics_consistency_score": physics_consistency,
                "parameter_ranges": {
                    k: {"min": min(v), "max": max(v), "count": len(v)}
                    for k, v in parameter_ranges.items()
                },
            },
            recommendations=recommendations,
        )

    async def _assess_statistical_significance(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess statistical significance of results."""

        significance_tests = []

        # Group experiments for statistical testing
        groups = self._group_by_conditions(experiments)

        for condition, group_experiments in groups.items():
            if len(group_experiments) < 3:
                continue

            # Extract results for statistical testing
            for prop_name in ["band_gap", "efficiency", "stability"]:
                values = []
                for exp in group_experiments:
                    prop_value = exp.get("results", {}).get(prop_name)
                    if prop_value is not None:
                        values.append(float(prop_value))

                if len(values) >= 3:
                    # Simple statistical test (in practice, would use scipy.stats)
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    # Simplified significance test
                    if len(values) >= 5 and std_val > 0:
                        # Check if mean is significantly different from baseline
                        t_stat = abs(mean_val - 1.5) / (
                            std_val / (len(values) ** 0.5)
                        )  # Assuming baseline of 1.5

                        # Simplified p-value estimation
                        if t_stat > 2.0:  # Roughly p < 0.05 for small samples
                            p_value = 0.03
                        elif t_stat > 1.5:
                            p_value = 0.08
                        else:
                            p_value = 0.15

                        significance_tests.append(
                            {
                                "condition": condition,
                                "property": prop_name,
                                "p_value": p_value,
                                "significant": p_value
                                < self.min_statistical_significance,
                                "sample_size": len(values),
                            }
                        )

        # Calculate overall significance score
        significant_tests = [t for t in significance_tests if t["significant"]]
        overall_score = (
            len(significant_tests) / len(significance_tests)
            if significance_tests
            else 0.0
        )

        recommendations = []
        if overall_score < 0.5:
            recommendations.append("Low statistical significance in results")
            recommendations.append("Increase sample sizes for more robust statistics")
            recommendations.append("Consider experimental design improvements")

        return QualityAssessment(
            metric=QualityMetric.STATISTICAL_SIGNIFICANCE,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "total_tests": len(significance_tests),
                "significant_tests": len(significant_tests),
                "significance_threshold": self.min_statistical_significance,
                "test_results": significance_tests,
            },
            recommendations=recommendations,
        )

    async def _assess_experimental_validity(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess experimental validity and protocol adherence."""

        validity_scores = []
        validity_issues = []

        for exp in experiments:
            exp_validity = 1.0

            # Check parameter validity
            parameters = exp.get("parameters", {})
            for param, value in parameters.items():
                if not self._is_parameter_valid(param, value):
                    exp_validity *= 0.8
                    validity_issues.append(f"Invalid parameter {param}={value}")

            # Check result validity
            results = exp.get("results", {})
            for result, value in results.items():
                if not self._is_result_valid(result, value):
                    exp_validity *= 0.8
                    validity_issues.append(f"Invalid result {result}={value}")

            # Check experimental metadata
            if not exp.get("timestamp"):
                exp_validity *= 0.9
                validity_issues.append("Missing timestamp")

            validity_scores.append(exp_validity)

        overall_score = np.mean(validity_scores) if validity_scores else 0.0

        recommendations = []
        if overall_score < 0.8:
            recommendations.append("Experimental validity issues detected")
            recommendations.append("Review parameter and result validation rules")
            recommendations.append("Ensure complete experimental metadata")

        # Add specific recommendations based on common issues
        issue_counts = {}
        for issue in validity_issues:
            issue_type = issue.split()[0]  # First word
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        for issue_type, count in issue_counts.items():
            if count > len(experiments) * 0.1:
                recommendations.append(
                    f"Frequent {issue_type} issues: {count} occurrences"
                )

        return QualityAssessment(
            metric=QualityMetric.EXPERIMENTAL_VALIDITY,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "individual_validity_scores": validity_scores,
                "validity_issues": validity_issues[:10],  # First 10 issues
                "total_issues": len(validity_issues),
            },
            recommendations=recommendations,
        )

    async def _assess_data_integrity(
        self, experiments: List[Dict[str, Any]]
    ) -> QualityAssessment:
        """Assess data integrity and detect anomalies."""

        integrity_scores = []
        anomalies = []

        for i, exp in enumerate(experiments):
            integrity_score = 1.0

            # Check for missing critical data
            if not exp.get("parameters"):
                integrity_score *= 0.5
                anomalies.append(f"Experiment {i}: Missing parameters")

            if not exp.get("results"):
                integrity_score *= 0.5
                anomalies.append(f"Experiment {i}: Missing results")

            # Check for data type consistency
            try:
                for param, value in exp.get("parameters", {}).items():
                    float(value)  # Should be convertible to float
            except (ValueError, TypeError):
                integrity_score *= 0.8
                anomalies.append(f"Experiment {i}: Non-numeric parameter values")

            try:
                for result, value in exp.get("results", {}).items():
                    float(value)  # Should be convertible to float
            except (ValueError, TypeError):
                integrity_score *= 0.8
                anomalies.append(f"Experiment {i}: Non-numeric result values")

            # Check for outliers
            if self._is_outlier_experiment(exp, experiments):
                integrity_score *= 0.9
                anomalies.append(f"Experiment {i}: Statistical outlier")

            integrity_scores.append(integrity_score)

        overall_score = np.mean(integrity_scores) if integrity_scores else 0.0

        recommendations = []
        if overall_score < 0.9:
            recommendations.append("Data integrity issues detected")
            recommendations.append("Implement data validation at collection time")
            recommendations.append("Review data storage and processing pipelines")

        if len(anomalies) > len(experiments) * 0.05:  # More than 5% anomalies
            recommendations.append(
                f"High anomaly rate: {len(anomalies)} in {len(experiments)} experiments"
            )

        return QualityAssessment(
            metric=QualityMetric.DATA_INTEGRITY,
            score=overall_score,
            level=self._score_to_quality_level(overall_score),
            details={
                "individual_integrity_scores": integrity_scores,
                "anomalies": anomalies[:10],  # First 10 anomalies
                "total_anomalies": len(anomalies),
            },
            recommendations=recommendations,
        )

    def _group_similar_experiments(
        self, experiments: List[Dict[str, Any]], tolerance: float = 0.1
    ) -> List[List[Dict[str, Any]]]:
        """Group experiments with similar parameters."""

        groups = []

        for exp in experiments:
            placed = False

            for group in groups:
                if self._are_experiments_similar(exp, group[0], tolerance):
                    group.append(exp)
                    placed = True
                    break

            if not placed:
                groups.append([exp])

        return groups

    def _are_experiments_similar(
        self, exp1: Dict[str, Any], exp2: Dict[str, Any], tolerance: float
    ) -> bool:
        """Check if two experiments have similar parameters."""

        params1 = exp1.get("parameters", {})
        params2 = exp2.get("parameters", {})

        common_params = set(params1.keys()) & set(params2.keys())

        if len(common_params) == 0:
            return False

        for param in common_params:
            try:
                val1 = float(params1[param])
                val2 = float(params2[param])

                if abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6) > tolerance:
                    return False
            except (ValueError, TypeError):
                if params1[param] != params2[param]:
                    return False

        return True

    def _coefficient_of_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if not values or len(values) < 2:
            return 0.0

        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0

        return np.std(values) / abs(mean_val)

    def _check_physical_validity(
        self, results: Dict[str, Any], parameters: Dict[str, Any]
    ) -> float:
        """Check physical validity of results given parameters."""

        validity_score = 1.0

        # Check individual result ranges
        for result_name, value in results.items():
            if result_name in self.validation_rules["result_ranges"]:
                min_val, max_val = self.validation_rules["result_ranges"][result_name]
                try:
                    val = float(value)
                    if not (min_val <= val <= max_val):
                        validity_score *= 0.7
                except (ValueError, TypeError):
                    validity_score *= 0.5

        # Check parameter ranges
        for param_name, value in parameters.items():
            if param_name in self.validation_rules["parameter_ranges"]:
                min_val, max_val = self.validation_rules["parameter_ranges"][param_name]
                try:
                    val = float(value)
                    if not (min_val <= val <= max_val):
                        validity_score *= 0.8
                except (ValueError, TypeError):
                    validity_score *= 0.5

        return validity_score

    def _check_physics_consistency(self, experiments: List[Dict[str, Any]]) -> float:
        """Check consistency with physical laws and relationships."""

        consistency_score = 1.0

        # Check band gap vs efficiency relationship (should be correlated)
        band_gaps = []
        efficiencies = []

        for exp in experiments:
            results = exp.get("results", {})
            bg = results.get("band_gap")
            eff = results.get("efficiency")

            if bg is not None and eff is not None:
                try:
                    band_gaps.append(float(bg))
                    efficiencies.append(float(eff))
                except (ValueError, TypeError):
                    continue

        if len(band_gaps) > 5:
            # Simple correlation check
            correlation = self._simple_correlation(band_gaps, efficiencies)

            # Expect some correlation between band gap and efficiency
            if abs(correlation) < 0.2:  # Very weak correlation might indicate issues
                consistency_score *= 0.8

        return consistency_score

    def _simple_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i**2 for x_i in x)
        sum_y2 = sum(y_i**2 for y_i in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _group_by_conditions(
        self, experiments: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group experiments by experimental conditions."""

        groups = {}

        for exp in experiments:
            # Create condition key based on major parameters
            condition_key = self._create_condition_key(exp.get("parameters", {}))

            if condition_key not in groups:
                groups[condition_key] = []

            groups[condition_key].append(exp)

        return groups

    def _create_condition_key(self, parameters: Dict[str, Any]) -> str:
        """Create a condition key for grouping experiments."""

        # Focus on major experimental conditions
        key_params = ["temperature", "pH", "atmosphere"]

        key_parts = []
        for param in key_params:
            if param in parameters:
                try:
                    # Round to reasonable precision for grouping
                    val = float(parameters[param])
                    rounded_val = round(val, 1)
                    key_parts.append(f"{param}={rounded_val}")
                except (ValueError, TypeError):
                    key_parts.append(f"{param}={parameters[param]}")

        return "|".join(key_parts) if key_parts else "default"

    def _is_parameter_valid(self, param_name: str, value: Any) -> bool:
        """Check if a parameter value is valid."""

        if param_name in self.validation_rules["parameter_ranges"]:
            min_val, max_val = self.validation_rules["parameter_ranges"][param_name]
            try:
                val = float(value)
                return min_val <= val <= max_val
            except (ValueError, TypeError):
                return False

        return True  # No specific validation rule

    def _is_result_valid(self, result_name: str, value: Any) -> bool:
        """Check if a result value is valid."""

        if result_name in self.validation_rules["result_ranges"]:
            min_val, max_val = self.validation_rules["result_ranges"][result_name]
            try:
                val = float(value)
                return min_val <= val <= max_val
            except (ValueError, TypeError):
                return False

        return True  # No specific validation rule

    def _is_outlier_experiment(
        self, experiment: Dict[str, Any], all_experiments: List[Dict[str, Any]]
    ) -> bool:
        """Check if an experiment is a statistical outlier."""

        # Simple outlier detection based on result values
        results = experiment.get("results", {})

        for prop_name, prop_value in results.items():
            try:
                val = float(prop_value)

                # Get all values for this property
                all_values = []
                for exp in all_experiments:
                    other_val = exp.get("results", {}).get(prop_name)
                    if other_val is not None:
                        try:
                            all_values.append(float(other_val))
                        except (ValueError, TypeError):
                            continue

                if len(all_values) > 5:
                    # Use IQR method for outlier detection
                    q1 = np.percentile(all_values, 25)
                    q3 = np.percentile(all_values, 75)
                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    if val < lower_bound or val > upper_bound:
                        return True

            except (ValueError, TypeError):
                continue

        return False

    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert numerical score to quality level."""

        if score >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds[QualityLevel.ACCEPTABLE]:
            return QualityLevel.ACCEPTABLE
        elif score >= self.quality_thresholds[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _update_quality_history(
        self, campaign_id: str, assessments: Dict[QualityMetric, QualityAssessment]
    ):
        """Update quality history for trend analysis."""

        for metric, assessment in assessments.items():
            metric_key = f"{campaign_id}_{metric.value}"

            if metric_key not in self.quality_history:
                self.quality_history[metric_key] = []

            self.quality_history[metric_key].append(assessment.score)

    def generate_quality_report(
        self, assessments: Dict[QualityMetric, QualityAssessment]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""

        # Calculate overall quality score
        scores = [assessment.score for assessment in assessments.values()]
        overall_score = np.mean(scores) if scores else 0.0
        overall_level = self._score_to_quality_level(overall_score)

        # Collect all recommendations
        all_recommendations = []
        for assessment in assessments.values():
            all_recommendations.extend(assessment.recommendations)

        # Quality metrics summary
        metrics_summary = {}
        for metric, assessment in assessments.items():
            metrics_summary[metric.value] = {
                "score": assessment.score,
                "level": assessment.level.value,
                "recommendations_count": len(assessment.recommendations),
            }

        # Priority recommendations (most critical issues)
        priority_recommendations = []
        for assessment in assessments.values():
            if assessment.level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
                priority_recommendations.extend(assessment.recommendations)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": overall_score,
            "overall_quality_level": overall_level.value,
            "metrics_summary": metrics_summary,
            "total_recommendations": len(all_recommendations),
            "priority_recommendations": priority_recommendations[:10],  # Top 10
            "detailed_assessments": {
                metric.value: assessment.to_dict()
                for metric, assessment in assessments.items()
            },
        }


# Global instance
_global_quality_assurance = None


def get_global_quality_assurance() -> AdvancedQualityAssurance:
    """Get global quality assurance instance."""
    global _global_quality_assurance
    if _global_quality_assurance is None:
        _global_quality_assurance = AdvancedQualityAssurance()
    return _global_quality_assurance
