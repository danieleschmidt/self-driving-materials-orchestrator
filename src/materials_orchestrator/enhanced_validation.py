"""Enhanced validation system for materials orchestrator with comprehensive checks."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class ValidationCategory(Enum):
    """Categories of validation checks."""

    SAFETY = "safety"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    ENVIRONMENTAL = "environmental"


@dataclass
class ValidationResult:
    """Result of validation check."""

    is_valid: bool
    category: ValidationCategory
    severity: str  # "critical", "warning", "info"
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EnhancedValidator:
    """Advanced validation system with safety, chemistry, and physics checks."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.validation_history = []
        self.safety_rules = self._load_safety_rules()
        self.chemistry_rules = self._load_chemistry_rules()

        logger.info(
            f"Enhanced validator initialized with {validation_level.value} level"
        )

    def validate_experiment_parameters(
        self, parameters: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Comprehensive validation of experiment parameters."""
        results = []

        # Safety validation
        results.extend(self._validate_safety(parameters))

        # Chemistry validation
        results.extend(self._validate_chemistry(parameters))

        # Physics validation
        results.extend(self._validate_physics(parameters))

        # Economic validation
        results.extend(self._validate_economics(parameters))

        # Environmental validation
        results.extend(self._validate_environmental(parameters))

        # Store validation history
        self.validation_history.append(
            {
                "timestamp": datetime.now(),
                "parameters": parameters,
                "results": results,
                "total_checks": len(results),
                "critical_issues": len(
                    [r for r in results if r.severity == "critical"]
                ),
                "warnings": len([r for r in results if r.severity == "warning"]),
            }
        )

        return results

    def _validate_safety(self, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Safety validation checks."""
        results = []

        # Temperature safety
        temp = parameters.get("temperature", 0)
        if temp > 400:
            results.append(
                ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.SAFETY,
                    severity="critical",
                    message=f"Temperature {temp}°C exceeds safety limit (400°C)",
                    details={"parameter": "temperature", "value": temp, "limit": 400},
                )
            )
        elif temp > 300:
            results.append(
                ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.SAFETY,
                    severity="warning",
                    message=f"Temperature {temp}°C is high, ensure proper safety measures",
                    details={"parameter": "temperature", "value": temp},
                )
            )

        # pH safety
        pH = parameters.get("pH", 7)
        if pH < 1 or pH > 13:
            results.append(
                ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.SAFETY,
                    severity="critical",
                    message=f"pH {pH} is extremely dangerous",
                    details={"parameter": "pH", "value": pH, "safe_range": [1, 13]},
                )
            )
        elif pH < 2 or pH > 12:
            results.append(
                ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.SAFETY,
                    severity="warning",
                    message=f"pH {pH} requires special handling precautions",
                    details={"parameter": "pH", "value": pH},
                )
            )

        # Concentration safety
        for param, value in parameters.items():
            if "conc" in param.lower() and isinstance(value, (int, float)):
                if value > 10:
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.SAFETY,
                            severity="warning",
                            message=f"High concentration {value}M for {param} - verify safety protocols",
                            details={"parameter": param, "value": value},
                        )
                    )

        return results

    def _validate_chemistry(self, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Chemistry validation checks."""
        results = []

        # Reaction time validation
        reaction_time = parameters.get("reaction_time", 0)
        if reaction_time > 48:
            results.append(
                ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.CHEMISTRY,
                    severity="warning",
                    message=f"Long reaction time {reaction_time}h may indicate inefficient process",
                    details={"parameter": "reaction_time", "value": reaction_time},
                )
            )

        # Temperature vs reaction time correlation
        temp = parameters.get("temperature", 0)
        if temp > 200 and reaction_time > 12:
            results.append(
                ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.CHEMISTRY,
                    severity="warning",
                    message="High temperature with long reaction time may cause decomposition",
                    details={"temperature": temp, "reaction_time": reaction_time},
                )
            )

        # Solvent ratio validation
        solvent_ratio = parameters.get("solvent_ratio", 0.5)
        if not 0 <= solvent_ratio <= 1:
            results.append(
                ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CHEMISTRY,
                    severity="critical",
                    message=f"Solvent ratio {solvent_ratio} must be between 0 and 1",
                    details={"parameter": "solvent_ratio", "value": solvent_ratio},
                )
            )

        return results

    def _validate_physics(self, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Physics validation checks."""
        results = []

        # Thermodynamic feasibility
        temp = parameters.get("temperature", 0)
        if temp < -273.15:
            results.append(
                ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.PHYSICS,
                    severity="critical",
                    message=f"Temperature {temp}°C below absolute zero",
                    details={"parameter": "temperature", "value": temp},
                )
            )

        # Energy balance considerations
        if temp > 250:
            energy_estimate = temp * 4.18  # Rough energy estimate
            if energy_estimate > 2000:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        category=ValidationCategory.PHYSICS,
                        severity="info",
                        message=f"High energy requirement estimated: {energy_estimate:.0f} J",
                        details={
                            "temperature": temp,
                            "energy_estimate": energy_estimate,
                        },
                    )
                )

        return results

    def _validate_economics(self, parameters: Dict[str, Any]) -> List[ValidationResult]:
        """Economic feasibility validation."""
        results = []

        # Cost estimation based on parameters
        cost_factors = {
            "temperature": 0.01,  # $/°C
            "reaction_time": 0.5,  # $/hour
            "precursor_A_conc": 10,  # $/M
            "precursor_B_conc": 15,  # $/M
        }

        total_cost = 0
        for param, value in parameters.items():
            if param in cost_factors and isinstance(value, (int, float)):
                total_cost += cost_factors[param] * value

        if total_cost > 100:
            results.append(
                ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.ECONOMICS,
                    severity="warning",
                    message=f"High estimated cost ${total_cost:.2f} per experiment",
                    details={"total_cost": total_cost, "cost_breakdown": cost_factors},
                )
            )

        return results

    def _validate_environmental(
        self, parameters: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Environmental impact validation."""
        results = []

        # Waste generation assessment
        reaction_time = parameters.get("reaction_time", 0)
        concentrations = [
            v
            for k, v in parameters.items()
            if "conc" in k.lower() and isinstance(v, (int, float))
        ]

        if concentrations:
            waste_estimate = sum(concentrations) * reaction_time
            if waste_estimate > 50:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        category=ValidationCategory.ENVIRONMENTAL,
                        severity="info",
                        message=f"Consider waste reduction strategies (waste estimate: {waste_estimate:.1f})",
                        details={"waste_estimate": waste_estimate},
                    )
                )

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities."""
        if not self.validation_history:
            return {"total_validations": 0}

        total_validations = len(self.validation_history)
        total_checks = sum(h["total_checks"] for h in self.validation_history)
        total_critical = sum(h["critical_issues"] for h in self.validation_history)
        total_warnings = sum(h["warnings"] for h in self.validation_history)

        return {
            "total_validations": total_validations,
            "total_checks": total_checks,
            "total_critical_issues": total_critical,
            "total_warnings": total_warnings,
            "validation_level": self.validation_level.value,
            "categories_checked": [cat.value for cat in ValidationCategory],
            "recent_activity": (
                self.validation_history[-10:] if self.validation_history else []
            ),
        }

    def _load_safety_rules(self) -> Dict[str, Any]:
        """Load safety rules configuration."""
        return {
            "max_temperature": 400,
            "min_temperature": -100,
            "safe_ph_range": [1, 13],
            "max_concentration": 10,
            "max_reaction_time": 72,
        }

    def _load_chemistry_rules(self) -> Dict[str, Any]:
        """Load chemistry rules configuration."""
        return {
            "optimal_temp_range": [100, 300],
            "optimal_time_range": [1, 24],
            "solvent_ratio_range": [0, 1],
            "concentration_limits": {
                "precursor_A": [0.1, 5.0],
                "precursor_B": [0.1, 5.0],
            },
        }


class RobustErrorHandler:
    """Enhanced error handling with detailed logging and recovery strategies."""

    def __init__(self):
        self.error_count = 0
        self.error_history = []
        self.recovery_strategies = self._init_recovery_strategies()

        logger.info("Robust error handler initialized")

    def handle_experiment_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle experiment errors with recovery strategies."""
        self.error_count += 1

        error_info = {
            "timestamp": datetime.now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "error_id": f"ERR_{self.error_count:04d}",
        }

        self.error_history.append(error_info)

        # Determine recovery strategy
        recovery_action = self._determine_recovery_strategy(error, context)

        logger.error(f"Experiment error {error_info['error_id']}: {error}")
        logger.info(f"Applying recovery strategy: {recovery_action['strategy']}")

        return {
            "error_info": error_info,
            "recovery_action": recovery_action,
            "should_retry": recovery_action.get("retry", False),
            "modified_parameters": recovery_action.get("parameters", {}),
        }

    def _determine_recovery_strategy(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine appropriate recovery strategy based on error type."""
        error_type = type(error).__name__

        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]

        # Default strategy
        return {"strategy": "log_and_continue", "retry": False, "parameters": {}}

    def _init_recovery_strategies(self) -> Dict[str, Any]:
        """Initialize recovery strategies for different error types."""
        return {
            "TemperatureError": {
                "strategy": "reduce_temperature",
                "retry": True,
                "parameters": {"temperature_reduction": 50},
            },
            "ConcentrationError": {
                "strategy": "dilute_solution",
                "retry": True,
                "parameters": {"dilution_factor": 0.8},
            },
            "TimeoutError": {
                "strategy": "extend_time",
                "retry": True,
                "parameters": {"time_extension": 2.0},
            },
            "ValidationError": {
                "strategy": "use_safe_defaults",
                "retry": True,
                "parameters": {},
            },
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        if not self.error_history:
            return {"total_errors": 0}

        error_types = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        recent_errors = [
            e
            for e in self.error_history
            if (datetime.now() - e["timestamp"]).total_seconds() < 3600
        ]

        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "recent_errors_1h": len(recent_errors),
            "error_rate": len(recent_errors) / 60,  # per minute
            "most_common_error": (
                max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            ),
        }


def create_robust_validation_system() -> Tuple[EnhancedValidator, RobustErrorHandler]:
    """Factory function to create robust validation system."""
    validator = EnhancedValidator(ValidationLevel.NORMAL)
    error_handler = RobustErrorHandler()

    logger.info("Robust validation system created")
    return validator, error_handler
