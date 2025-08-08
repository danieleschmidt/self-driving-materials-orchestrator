"""Data validation and quality assurance for experiment data."""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of data validation."""
    status: ValidationStatus
    message: str
    field: Optional[str] = None
    value: Any = None
    expected_range: Optional[Tuple[float, float]] = None
    confidence: float = 1.0


@dataclass
class QualityMetrics:
    """Quality metrics for experiment data."""
    completeness: float  # Fraction of required fields present
    consistency: float  # How consistent data is with expectations
    accuracy: float  # Estimated accuracy based on validation
    outlier_score: float  # How much data deviates from normal ranges
    timestamp: datetime = field(default_factory=datetime.now)


class ExperimentValidator:
    """Validates experiment data for quality and safety."""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.historical_data: List[Dict[str, Any]] = []
        self.statistics_cache: Dict[str, Dict[str, float]] = {}
        
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load validation rules for different data types."""
        return {
            "experiment_parameters": {
                "precursor_A_conc": {
                    "type": "float",
                    "range": (0.01, 5.0),
                    "units": "M",
                    "required": False  # Made optional for flexible parameter spaces
                },
                "precursor_B_conc": {
                    "type": "float",
                    "range": (0.01, 5.0),
                    "units": "M", 
                    "required": False
                },
                "temperature": {
                    "type": "float",
                    "range": (20, 1000),  # Expanded range for flexibility
                    "units": "°C",
                    "required": False
                },
                "reaction_time": {
                    "type": "float",
                    "range": (0.1, 168),  # Expanded to 1 week
                    "units": "hours",
                    "required": False
                },
                "time": {  # Alternative naming
                    "type": "float",
                    "range": (0.1, 168),
                    "units": "hours",
                    "required": False
                },
                "concentration": {  # Alternative naming
                    "type": "float",
                    "range": (0.01, 10.0),
                    "units": "M",
                    "required": False
                },
                "pH": {
                    "type": "float",
                    "range": (0, 15),  # Expanded range
                    "units": "pH",
                    "required": False
                },
                "solvent_ratio": {
                    "type": "float",
                    "range": (0, 1),
                    "units": "fraction",
                    "required": False
                }
            },
            "experiment_results": {
                "band_gap": {
                    "type": "float",
                    "range": (0.1, 4.0),
                    "units": "eV",
                    "required": False
                },
                "efficiency": {
                    "type": "float",
                    "range": (0, 50),
                    "units": "%",
                    "required": False
                },
                "stability": {
                    "type": "float",
                    "range": (0, 1),
                    "units": "fraction",
                    "required": False
                }
            }
        }
    
    def validate_experiment_data(
        self, 
        parameters: Dict[str, Any], 
        results: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate complete experiment data."""
        validation_results = []
        
        # Validate parameters
        param_results = self.validate_parameters(parameters)
        validation_results.extend(param_results)
        
        # Validate results if provided
        if results:
            result_validations = self.validate_results(results)
            validation_results.extend(result_validations)
            
            # Cross-validate parameters and results
            cross_validations = self.cross_validate(parameters, results)
            validation_results.extend(cross_validations)
        
        return validation_results
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Validate experiment parameters."""
        results = []
        rules = self.validation_rules["experiment_parameters"]
        
        # Check for required parameters
        for param_name, rule in rules.items():
            if rule.get("required", False) and param_name not in parameters:
                results.append(ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Required parameter '{param_name}' is missing",
                    field=param_name
                ))
        
        # Validate present parameters
        for param_name, value in parameters.items():
            if param_name in rules:
                param_results = self._validate_single_parameter(param_name, value, rules[param_name])
                results.extend(param_results)
            else:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Unknown parameter '{param_name}'",
                    field=param_name,
                    value=value
                ))
        
        return results
    
    def validate_results(self, results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate experiment results."""
        validation_results = []
        rules = self.validation_rules["experiment_results"]
        
        for result_name, value in results.items():
            if result_name in rules:
                result_validations = self._validate_single_parameter(result_name, value, rules[result_name])
                validation_results.extend(result_validations)
            else:
                validation_results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Unknown result field '{result_name}'",
                    field=result_name,
                    value=value
                ))
        
        return validation_results
    
    def _validate_single_parameter(
        self, 
        name: str, 
        value: Any, 
        rule: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate a single parameter against its rule."""
        results = []
        
        # Type validation
        expected_type = rule.get("type", "float")
        if expected_type == "float":
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                results.append(ValidationResult(
                    status=ValidationStatus.INVALID,
                    message=f"Parameter '{name}' must be a number, got {type(value).__name__}",
                    field=name,
                    value=value
                ))
                return results
            
            value = float_value
        
        # Range validation
        if "range" in rule:
            min_val, max_val = rule["range"]
            if not (min_val <= value <= max_val):
                if value < min_val:
                    results.append(ValidationResult(
                        status=ValidationStatus.INVALID,
                        message=f"Parameter '{name}' value {value} is below minimum {min_val}",
                        field=name,
                        value=value,
                        expected_range=(min_val, max_val)
                    ))
                else:
                    results.append(ValidationResult(
                        status=ValidationStatus.INVALID,
                        message=f"Parameter '{name}' value {value} is above maximum {max_val}",
                        field=name,
                        value=value,
                        expected_range=(min_val, max_val)
                    ))
        
        # Statistical validation (outlier detection)
        outlier_result = self._check_for_outliers(name, value)
        if outlier_result:
            results.append(outlier_result)
        
        # If no issues found, mark as valid
        if not results:
            results.append(ValidationResult(
                status=ValidationStatus.VALID,
                message=f"Parameter '{name}' is valid",
                field=name,
                value=value
            ))
        
        return results
    
    def cross_validate(
        self, 
        parameters: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Cross-validate parameters against results for consistency."""
        validation_results = []
        
        # Check for impossible combinations
        temp = parameters.get("temperature")
        band_gap = results.get("band_gap")
        
        if temp and band_gap:
            # Very high temperature with very high band gap is unusual
            if temp > 400 and band_gap > 3.0:
                validation_results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"High temperature ({temp}°C) with high band gap ({band_gap} eV) is unusual",
                    confidence=0.7
                ))
        
        # Check efficiency vs band gap relationship
        band_gap = results.get("band_gap")
        efficiency = results.get("efficiency")
        
        if band_gap and efficiency:
            # For photovoltaics, optimal band gap is around 1.3-1.4 eV
            optimal_gap = 1.35
            gap_deviation = abs(band_gap - optimal_gap)
            
            # High efficiency with non-optimal band gap is suspicious
            if efficiency > 25 and gap_deviation > 0.5:
                validation_results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"High efficiency ({efficiency}%) with non-optimal band gap ({band_gap} eV)",
                    confidence=0.6
                ))
        
        return validation_results
    
    def _check_for_outliers(self, parameter_name: str, value: float) -> Optional[ValidationResult]:
        """Check if a value is a statistical outlier."""
        if parameter_name not in self.statistics_cache:
            self._update_statistics_cache(parameter_name)
        
        stats = self.statistics_cache.get(parameter_name, {})
        if "mean" not in stats or "std" not in stats:
            return None  # Not enough data for outlier detection
        
        mean = stats["mean"]
        std = stats["std"]
        
        if std == 0:
            return None  # No variation in data
        
        # Calculate z-score
        z_score = abs(value - mean) / std
        
        if z_score > 3:  # More than 3 standard deviations
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Parameter '{parameter_name}' value {value} is a statistical outlier (z-score: {z_score:.2f})",
                field=parameter_name,
                value=value,
                confidence=min(1.0, z_score / 3)
            )
        
        return None
    
    def _update_statistics_cache(self, parameter_name: str) -> None:
        """Update statistical cache for a parameter."""
        values = []
        for data in self.historical_data:
            if parameter_name in data:
                try:
                    values.append(float(data[parameter_name]))
                except (ValueError, TypeError):
                    continue
        
        if len(values) >= 3:
            self.statistics_cache[parameter_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "count": len(values),
                "min": min(values),
                "max": max(values)
            }
    
    def add_historical_data(self, experiment_data: Dict[str, Any]) -> None:
        """Add experiment data to historical database for validation."""
        self.historical_data.append(experiment_data)
        
        # Limit historical data size
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-500:]
        
        # Clear cache to force recalculation
        self.statistics_cache.clear()
    
    def calculate_quality_metrics(
        self, 
        parameters: Dict[str, Any], 
        results: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Calculate overall quality metrics for experiment data."""
        validation_results = self.validate_experiment_data(parameters, results)
        
        # Calculate completeness
        required_params = [
            name for name, rule in self.validation_rules["experiment_parameters"].items()
            if rule.get("required", False)
        ]
        present_required = sum(1 for param in required_params if param in parameters)
        completeness = present_required / len(required_params) if required_params else 1.0
        
        # Calculate consistency (inverse of validation issues)
        total_validations = len(validation_results)
        issues = sum(
            1 for result in validation_results 
            if result.status in [ValidationStatus.WARNING, ValidationStatus.INVALID]
        )
        consistency = 1.0 - (issues / total_validations) if total_validations > 0 else 1.0
        
        # Calculate accuracy (based on confidence scores)
        confidences = [r.confidence for r in validation_results if r.confidence < 1.0]
        accuracy = 1.0 - (sum(1 - c for c in confidences) / len(confidences)) if confidences else 1.0
        
        # Calculate outlier score
        outlier_results = [
            r for r in validation_results 
            if "outlier" in r.message.lower()
        ]
        outlier_score = len(outlier_results) / len(validation_results) if validation_results else 0.0
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            outlier_score=outlier_score
        )
    
    def suggest_corrections(
        self, 
        validation_results: List[ValidationResult]
    ) -> List[Dict[str, Any]]:
        """Suggest corrections for validation issues."""
        suggestions = []
        
        for result in validation_results:
            if result.status == ValidationStatus.INVALID:
                suggestion = {
                    "field": result.field,
                    "issue": result.message,
                    "current_value": result.value,
                    "suggestion": ""
                }
                
                if result.expected_range:
                    min_val, max_val = result.expected_range
                    if result.value is not None:
                        # Suggest clamping to valid range
                        if result.value < min_val:
                            suggestion["suggestion"] = f"Use minimum value: {min_val}"
                        elif result.value > max_val:
                            suggestion["suggestion"] = f"Use maximum value: {max_val}"
                else:
                    suggestion["suggestion"] = "Please provide a valid value"
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def generate_quality_report(
        self, 
        parameters: Dict[str, Any], 
        results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report for experiment data."""
        validation_results = self.validate_experiment_data(parameters, results)
        quality_metrics = self.calculate_quality_metrics(parameters, results)
        suggestions = self.suggest_corrections(validation_results)
        
        # Categorize validation results
        valid_count = sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
        warning_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARNING)
        invalid_count = sum(1 for r in validation_results if r.status == ValidationStatus.INVALID)
        error_count = sum(1 for r in validation_results if r.status == ValidationStatus.ERROR)
        
        return {
            "overall_quality": {
                "completeness": quality_metrics.completeness,
                "consistency": quality_metrics.consistency,
                "accuracy": quality_metrics.accuracy,
                "outlier_score": quality_metrics.outlier_score,
                "overall_score": (
                    quality_metrics.completeness * 0.3 +
                    quality_metrics.consistency * 0.4 +
                    quality_metrics.accuracy * 0.3
                )
            },
            "validation_summary": {
                "total_checks": len(validation_results),
                "valid": valid_count,
                "warnings": warning_count,
                "invalid": invalid_count,
                "errors": error_count
            },
            "validation_details": [
                {
                    "field": r.field,
                    "status": r.status.value,
                    "message": r.message,
                    "value": r.value,
                    "confidence": r.confidence
                }
                for r in validation_results
            ],
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }


def create_validator() -> ExperimentValidator:
    """Create a configured experiment validator."""
    return ExperimentValidator()
