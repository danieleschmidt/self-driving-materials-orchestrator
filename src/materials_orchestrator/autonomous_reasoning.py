"""Autonomous reasoning and decision-making for materials discovery."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .utils import np

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of autonomous decisions."""

    EXPERIMENT_DESIGN = "experiment_design"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STRATEGY_CHANGE = "strategy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    EARLY_STOPPING = "early_stopping"
    CAMPAIGN_EXTENSION = "campaign_extension"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""

    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class Decision:
    """Autonomous decision record."""

    timestamp: datetime
    decision_type: DecisionType
    description: str
    confidence: float
    reasoning: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    success_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningContext:
    """Context for autonomous reasoning."""

    current_state: Dict[str, Any]
    experiment_history: List[Any]
    objective: Any
    constraints: Dict[str, Any] = field(default_factory=dict)
    available_resources: Dict[str, Any] = field(default_factory=dict)


class AutonomousReasoner:
    """Autonomous reasoning engine for materials discovery."""

    def __init__(self):
        """Initialize autonomous reasoner."""
        self.decision_history = []
        self.knowledge_base = {}
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7

        # Initialize reasoning rules
        self._initialize_reasoning_rules()

    def _initialize_reasoning_rules(self):
        """Initialize base reasoning rules."""
        self.reasoning_rules = {
            "convergence_detection": self._check_convergence_status,
            "parameter_importance": self._assess_parameter_importance,
            "exploration_strategy": self._evaluate_exploration_strategy,
            "resource_optimization": self._optimize_resource_allocation,
            "quality_assessment": self._assess_experiment_quality,
            "anomaly_detection": self._detect_anomalies,
            "strategy_adaptation": self._adapt_strategy,
        }

    def _assess_parameter_importance(self, context: ReasoningContext) -> Dict[str, Any]:
        """Assess parameter importance."""
        return {"importance": "medium"}

    def _evaluate_exploration_strategy(
        self, context: ReasoningContext
    ) -> Dict[str, Any]:
        """Evaluate exploration strategy."""
        return {"strategy": "balanced"}

    def _optimize_resource_allocation(
        self, context: ReasoningContext
    ) -> Dict[str, Any]:
        """Optimize resource allocation."""
        return {"allocation": "optimal"}

    def _assess_experiment_quality(self, context: ReasoningContext) -> Dict[str, Any]:
        """Assess experiment quality."""
        return {"quality": "good"}

    def _detect_anomalies(self, context: ReasoningContext) -> List[Dict[str, Any]]:
        """Detect anomalies."""
        return []

    def _adapt_strategy(self, context: ReasoningContext) -> Dict[str, Any]:
        """Adapt strategy."""
        return {"adaptation": "none"}

    def make_decision(self, context: ReasoningContext) -> Optional[Decision]:
        """Make autonomous decision based on current context."""
        logger.info("Making autonomous decision based on current context")

        # Analyze current situation
        situation_analysis = self._analyze_situation(context)

        # Generate potential decisions
        potential_decisions = self._generate_potential_decisions(
            context, situation_analysis
        )

        # Evaluate and select best decision
        best_decision = self._select_best_decision(potential_decisions, context)

        if best_decision and best_decision.confidence >= self.confidence_threshold:
            self.decision_history.append(best_decision)
            logger.info(
                f"Decision made: {best_decision.description} (confidence: {best_decision.confidence:.2f})"
            )
            return best_decision

        return None

    def _analyze_situation(self, context: ReasoningContext) -> Dict[str, Any]:
        """Analyze current situation comprehensively."""
        analysis = {
            "performance_trends": {},
            "convergence_status": {},
            "resource_utilization": {},
            "quality_indicators": {},
            "anomalies": [],
            "opportunities": [],
        }

        # Analyze performance trends
        if context.experiment_history:
            analysis["performance_trends"] = self._analyze_performance_trends(
                context.experiment_history
            )

        # Check convergence status
        analysis["convergence_status"] = self._check_convergence_status(context)

        # Assess resource utilization
        analysis["resource_utilization"] = self._assess_resource_utilization(context)

        # Evaluate experiment quality
        analysis["quality_indicators"] = self._evaluate_experiment_quality(context)

        # Detect anomalies
        analysis["anomalies"] = self._detect_experiment_anomalies(context)

        # Identify opportunities
        analysis["opportunities"] = self._identify_opportunities(context)

        return analysis

    def _analyze_performance_trends(
        self, experiment_history: List[Any]
    ) -> Dict[str, Any]:
        """Analyze performance trends from experiment history."""
        if len(experiment_history) < 3:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # Extract success rate over time
        recent_experiments = experiment_history[-10:]
        success_rates = []

        for i in range(0, len(recent_experiments), 3):
            batch = recent_experiments[i : i + 3]
            successes = sum(1 for exp in batch if exp.status == "completed")
            success_rates.append(successes / len(batch))

        if len(success_rates) >= 2:
            # Calculate trend
            if success_rates[-1] > success_rates[0]:
                trend = "improving"
            elif success_rates[-1] < success_rates[0]:
                trend = "declining"
            else:
                trend = "stable"

            # Calculate confidence based on consistency
            variance = np.var(success_rates) if len(success_rates) > 1 else 0
            confidence = max(0.0, 1.0 - variance * 2)

            return {
                "trend": trend,
                "confidence": confidence,
                "recent_success_rate": success_rates[-1],
                "change_rate": success_rates[-1] - success_rates[0],
            }

        return {"trend": "unknown", "confidence": 0.0}

    def _check_convergence_status(self, context: ReasoningContext) -> Dict[str, Any]:
        """Check if optimization has converged."""
        experiments = context.experiment_history
        if len(experiments) < 10:
            return {
                "converged": False,
                "confidence": 0.0,
                "reason": "insufficient_data",
            }

        # Look at recent improvements
        recent_experiments = experiments[-10:]
        target_property = context.objective.target_property

        property_values = []
        for exp in recent_experiments:
            if exp.status == "completed" and target_property in exp.results:
                property_values.append(exp.results[target_property])

        if len(property_values) < 5:
            return {
                "converged": False,
                "confidence": 0.0,
                "reason": "insufficient_successful_experiments",
            }

        # Check for improvements in recent experiments
        recent_variance = np.var(property_values[-5:])
        overall_variance = np.var(property_values)

        # Low recent variance indicates convergence
        convergence_indicator = 1.0 - (recent_variance / (overall_variance + 1e-6))
        converged = convergence_indicator > 0.8

        return {
            "converged": converged,
            "confidence": convergence_indicator,
            "recent_variance": recent_variance,
            "improvement_potential": 1.0 - convergence_indicator,
        }

    def _assess_resource_utilization(self, context: ReasoningContext) -> Dict[str, Any]:
        """Assess current resource utilization efficiency."""
        utilization = {
            "experiment_efficiency": 0.8,  # Default assumption
            "time_efficiency": 0.7,
            "cost_efficiency": 0.75,
            "recommendations": [],
        }

        # Analyze experiment success rate as efficiency indicator
        if context.experiment_history:
            successful = sum(
                1 for exp in context.experiment_history if exp.status == "completed"
            )
            total = len(context.experiment_history)
            utilization["experiment_efficiency"] = (
                successful / total if total > 0 else 0
            )

        # Generate recommendations based on efficiency
        if utilization["experiment_efficiency"] < 0.7:
            utilization["recommendations"].append(
                "Improve experiment protocols to reduce failure rate"
            )

        if utilization["time_efficiency"] < 0.8:
            utilization["recommendations"].append(
                "Consider parallel experiment execution"
            )

        return utilization

    def _evaluate_experiment_quality(self, context: ReasoningContext) -> Dict[str, Any]:
        """Evaluate overall experiment quality."""
        if not context.experiment_history:
            return {"overall_quality": 0.5, "issues": [], "strengths": []}

        quality_indicators = {
            "reproducibility": 0.8,  # Assume good reproducibility
            "measurement_accuracy": 0.9,
            "parameter_coverage": 0.7,
            "data_completeness": 0.85,
        }

        # Calculate parameter space coverage
        successful_experiments = [
            exp for exp in context.experiment_history if exp.status == "completed"
        ]
        if successful_experiments:
            # Simple coverage metric based on parameter diversity
            param_ranges = {}
            for exp in successful_experiments:
                for param, value in exp.parameters.items():
                    if param not in param_ranges:
                        param_ranges[param] = [value, value]
                    else:
                        param_ranges[param][0] = min(param_ranges[param][0], value)
                        param_ranges[param][1] = max(param_ranges[param][1], value)

            # Coverage score based on range utilization
            coverage_scores = []
            for param, (min_val, max_val) in param_ranges.items():
                if max_val > min_val:
                    coverage_scores.append(1.0)  # Simplified: assume good coverage
                else:
                    coverage_scores.append(0.0)

            quality_indicators["parameter_coverage"] = (
                np.mean(coverage_scores) if coverage_scores else 0.5
            )

        # Overall quality score
        overall_quality = np.mean(list(quality_indicators.values()))

        # Identify issues and strengths
        issues = []
        strengths = []

        for indicator, score in quality_indicators.items():
            if score < 0.6:
                issues.append(f"Low {indicator} (score: {score:.2f})")
            elif score > 0.8:
                strengths.append(f"High {indicator} (score: {score:.2f})")

        return {
            "overall_quality": overall_quality,
            "quality_indicators": quality_indicators,
            "issues": issues,
            "strengths": strengths,
        }

    def _detect_experiment_anomalies(
        self, context: ReasoningContext
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in experiment data."""
        anomalies = []

        if not context.experiment_history or len(context.experiment_history) < 5:
            return anomalies

        target_property = context.objective.target_property
        successful_experiments = [
            exp for exp in context.experiment_history if exp.status == "completed"
        ]

        if len(successful_experiments) < 3:
            return anomalies

        # Detect outliers in target property
        property_values = [
            exp.results.get(target_property, 0) for exp in successful_experiments
        ]

        if len(property_values) >= 5:
            mean_val = np.mean(property_values)
            std_val = np.std(property_values)

            for exp in successful_experiments:
                value = exp.results.get(target_property, 0)
                z_score = abs(value - mean_val) / (std_val + 1e-6)

                if z_score > 3:  # Outlier detection
                    anomalies.append(
                        {
                            "type": "statistical_outlier",
                            "experiment_id": exp.id,
                            "property": target_property,
                            "value": value,
                            "z_score": z_score,
                            "severity": "high" if z_score > 4 else "medium",
                        }
                    )

        # Detect unusual failure patterns
        recent_failures = [
            exp for exp in context.experiment_history[-10:] if exp.status == "failed"
        ]
        if len(recent_failures) >= 3:
            anomalies.append(
                {
                    "type": "failure_pattern",
                    "count": len(recent_failures),
                    "severity": "high",
                    "description": f"{len(recent_failures)} failures in last 10 experiments",
                }
            )

        return anomalies

    def _identify_opportunities(
        self, context: ReasoningContext
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []

        # Convergence-based opportunities
        convergence_status = self._check_convergence_status(context)
        if (
            not convergence_status["converged"]
            and convergence_status["improvement_potential"] > 0.3
        ):
            opportunities.append(
                {
                    "type": "continued_optimization",
                    "potential": convergence_status["improvement_potential"],
                    "description": "Significant improvement potential remains",
                    "action": "continue_with_current_strategy",
                }
            )

        # Parameter exploration opportunities
        if context.experiment_history:
            successful_experiments = [
                exp for exp in context.experiment_history if exp.status == "completed"
            ]
            if len(successful_experiments) >= 5:
                # Simple parameter importance analysis
                target_property = context.objective.target_property
                param_correlations = {}

                for exp in successful_experiments:
                    for param, value in exp.parameters.items():
                        if param not in param_correlations:
                            param_correlations[param] = []

                        property_value = exp.results.get(target_property, 0)
                        param_correlations[param].append((value, property_value))

                # Find parameters with high correlation
                for param, correlations in param_correlations.items():
                    if len(correlations) >= 3:
                        param_values = [c[0] for c in correlations]
                        property_values = [c[1] for c in correlations]

                        # Simple correlation calculation
                        if len(set(param_values)) > 1:
                            corr = abs(np.corrcoef(param_values, property_values)[0, 1])
                            if not np.isnan(corr) and corr > 0.6:
                                opportunities.append(
                                    {
                                        "type": "parameter_focus",
                                        "parameter": param,
                                        "correlation": corr,
                                        "description": f"Strong correlation detected for {param}",
                                        "action": f"focus_optimization_on_{param}",
                                    }
                                )

        return opportunities

    def _generate_potential_decisions(
        self, context: ReasoningContext, situation_analysis: Dict[str, Any]
    ) -> List[Decision]:
        """Generate potential decisions based on situation analysis."""
        potential_decisions = []

        # Strategy change decisions
        if situation_analysis["convergence_status"]["converged"]:
            potential_decisions.append(
                Decision(
                    timestamp=datetime.now(),
                    decision_type=DecisionType.EARLY_STOPPING,
                    description="Stop optimization - convergence achieved",
                    confidence=situation_analysis["convergence_status"]["confidence"],
                    reasoning=[
                        "Optimization has converged",
                        "Further experiments unlikely to improve results",
                    ],
                    expected_outcome="Save resources while maintaining current best result",
                )
            )

        # Parameter adjustment decisions
        for opportunity in situation_analysis["opportunities"]:
            if opportunity["type"] == "parameter_focus":
                potential_decisions.append(
                    Decision(
                        timestamp=datetime.now(),
                        decision_type=DecisionType.PARAMETER_ADJUSTMENT,
                        description=f"Focus optimization on {opportunity['parameter']}",
                        confidence=min(opportunity["correlation"], 0.9),
                        reasoning=[
                            f"Strong correlation detected ({opportunity['correlation']:.2f})",
                            f"Parameter {opportunity['parameter']} shows high importance",
                        ],
                        parameters={"focus_parameter": opportunity["parameter"]},
                        expected_outcome="Improved convergence rate by focusing on important parameter",
                    )
                )

        # Resource allocation decisions
        utilization = situation_analysis["resource_utilization"]
        if utilization["experiment_efficiency"] < 0.7:
            potential_decisions.append(
                Decision(
                    timestamp=datetime.now(),
                    decision_type=DecisionType.RESOURCE_ALLOCATION,
                    description="Improve experiment protocols to reduce failure rate",
                    confidence=0.8,
                    reasoning=[
                        f"Low experiment success rate ({utilization['experiment_efficiency']:.2f})",
                        "Resource efficiency can be improved",
                    ],
                    expected_outcome="Higher experiment success rate and better resource utilization",
                )
            )

        # Anomaly response decisions
        for anomaly in situation_analysis["anomalies"]:
            if anomaly.get("severity") == "high":
                potential_decisions.append(
                    Decision(
                        timestamp=datetime.now(),
                        decision_type=DecisionType.STRATEGY_CHANGE,
                        description=f"Investigate {anomaly['type']} anomaly",
                        confidence=0.7,
                        reasoning=[
                            f"High severity anomaly detected: {anomaly.get('description', 'Unknown anomaly')}",
                            "Investigation needed to maintain data quality",
                        ],
                        expected_outcome="Improved data quality and experiment reliability",
                    )
                )

        return potential_decisions

    def _select_best_decision(
        self, potential_decisions: List[Decision], context: ReasoningContext
    ) -> Optional[Decision]:
        """Select the best decision from potential options."""
        if not potential_decisions:
            return None

        # Score decisions based on multiple criteria
        scored_decisions = []

        for decision in potential_decisions:
            score = self._score_decision(decision, context)
            scored_decisions.append((decision, score))

        # Sort by score and return best decision
        scored_decisions.sort(key=lambda x: x[1], reverse=True)
        best_decision, best_score = scored_decisions[0]

        # Update confidence based on scoring
        best_decision.confidence = min(best_decision.confidence * best_score, 1.0)

        return best_decision

    def _score_decision(self, decision: Decision, context: ReasoningContext) -> float:
        """Score a decision based on multiple criteria."""
        score = 0.0

        # Base score from confidence
        score += decision.confidence * 0.4

        # Priority boost based on decision type
        type_priorities = {
            DecisionType.EARLY_STOPPING: 0.9,
            DecisionType.PARAMETER_ADJUSTMENT: 0.8,
            DecisionType.STRATEGY_CHANGE: 0.7,
            DecisionType.RESOURCE_ALLOCATION: 0.6,
            DecisionType.EXPERIMENT_DESIGN: 0.5,
            DecisionType.CAMPAIGN_EXTENSION: 0.4,
        }
        score += type_priorities.get(decision.decision_type, 0.5) * 0.3

        # Context relevance boost
        if context.experiment_history and len(context.experiment_history) > 20:
            # Later stage - prefer convergence decisions
            if decision.decision_type in [
                DecisionType.EARLY_STOPPING,
                DecisionType.PARAMETER_ADJUSTMENT,
            ]:
                score += 0.2
        else:
            # Early stage - prefer exploration decisions
            if decision.decision_type in [
                DecisionType.EXPERIMENT_DESIGN,
                DecisionType.RESOURCE_ALLOCATION,
            ]:
                score += 0.2

        # Safety factor - penalize risky decisions
        if "stop" in decision.description.lower():
            score *= 0.9  # Slight penalty for stopping decisions

        return min(score, 1.0)

    def learn_from_outcome(self, decision: Decision, outcome: Dict[str, Any]):
        """Learn from decision outcomes to improve future reasoning."""
        # Update knowledge base with outcome
        decision_key = f"{decision.decision_type.value}_{hash(decision.description)}"

        if decision_key not in self.knowledge_base:
            self.knowledge_base[decision_key] = {
                "attempts": 0,
                "successes": 0,
                "average_confidence": 0.0,
                "outcomes": [],
            }

        entry = self.knowledge_base[decision_key]
        entry["attempts"] += 1
        entry["outcomes"].append(outcome)

        # Update success rate based on outcome
        if outcome.get("success", False):
            entry["successes"] += 1

        # Update average confidence
        entry["average_confidence"] = (
            entry["average_confidence"] * (entry["attempts"] - 1) + decision.confidence
        ) / entry["attempts"]

        # Adjust confidence threshold based on learning
        success_rate = entry["successes"] / entry["attempts"]
        if success_rate < 0.5 and entry["attempts"] >= 3:
            # Lower confidence threshold if this type of decision often fails
            self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.9)
        elif success_rate > 0.8 and entry["attempts"] >= 3:
            # Raise confidence threshold if this type of decision is very reliable
            self.confidence_threshold = max(self.confidence_threshold - 0.05, 0.5)

        logger.info(
            f"Learning from decision outcome: {decision_key}, success_rate: {success_rate:.2f}"
        )

    def get_reasoning_explanation(self, decision: Decision) -> str:
        """Generate human-readable explanation of reasoning."""
        explanation = f"Decision: {decision.description}\n"
        explanation += f"Confidence: {decision.confidence:.2f}\n"
        explanation += f"Type: {decision.decision_type.value}\n\n"

        explanation += "Reasoning:\n"
        for i, reason in enumerate(decision.reasoning, 1):
            explanation += f"{i}. {reason}\n"

        if decision.expected_outcome:
            explanation += f"\nExpected outcome: {decision.expected_outcome}\n"

        if decision.parameters:
            explanation += (
                f"\nParameters: {json.dumps(decision.parameters, indent=2)}\n"
            )

        return explanation


# Global reasoner instance
_global_reasoner = None


def get_global_reasoner() -> AutonomousReasoner:
    """Get global autonomous reasoner instance."""
    global _global_reasoner
    if _global_reasoner is None:
        _global_reasoner = AutonomousReasoner()
    return _global_reasoner
