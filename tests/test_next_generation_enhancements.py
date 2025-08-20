"""Comprehensive tests for next-generation AI enhancements.

Tests the newly implemented Generation 4+ capabilities including autonomous
hypothesis generation, quantum-hybrid optimization, federated learning,
and real-time adaptive protocols.
"""

import asyncio
import json
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import next-generation modules
from materials_orchestrator.autonomous_hypothesis_generator import (
    AutonomousHypothesisGenerator,
    ScientificHypothesis,
    HypothesisType,
    HypothesisConfidence,
    generate_scientific_hypotheses
)
from materials_orchestrator.quantum_hybrid_optimizer import (
    QuantumHybridOptimizer,
    QuantumOptimizationProblem,
    OptimizationStrategy,
    QuantumBackend,
    optimize_with_quantum_hybrid
)
from materials_orchestrator.federated_learning_coordinator import (
    FederatedLearningCoordinator,
    LabNode,
    FederatedModel,
    ModelUpdate,
    LabRole,
    PrivacyLevel,
    create_federated_materials_network
)
from materials_orchestrator.realtime_adaptive_protocols import (
    AdaptiveProtocolEngine,
    ExperimentalCondition,
    RealTimeResult,
    AdaptationStrategy,
    AdaptationTrigger,
    process_realtime_experiment_data
)


class TestAutonomousHypothesisGenerator:
    """Test suite for autonomous hypothesis generation."""
    
    @pytest.fixture
    def sample_experiments(self) -> List[Dict[str, Any]]:
        """Create sample experimental data for testing."""
        experiments = []
        
        # Generate realistic perovskite experiment data
        for i in range(50):
            temp = 100 + i * 2 + np.random.normal(0, 5)
            conc_a = 0.8 + i * 0.02 + np.random.normal(0, 0.1)
            conc_b = 0.6 + i * 0.015 + np.random.normal(0, 0.08)
            time = 2 + i * 0.05 + np.random.normal(0, 0.3)
            
            # Simulate realistic property relationships
            band_gap = 1.6 - 0.0001 * temp + 0.05 * np.exp(-abs(conc_a - 1.2)) + np.random.normal(0, 0.02)
            efficiency = 0.15 + 0.1 * np.exp(-abs(band_gap - 1.4)) + np.random.normal(0, 0.01)
            stability = 0.7 + 0.2 * np.exp(-0.001 * temp) + np.random.normal(0, 0.05)
            
            experiments.append({
                "id": f"exp_{i:03d}",
                "timestamp": datetime.now() - timedelta(days=i),
                "parameters": {
                    "temperature": temp,
                    "precursor_A_conc": conc_a,
                    "precursor_B_conc": conc_b,
                    "reaction_time": time,
                    "ph": 7 + np.random.normal(0, 0.5)
                },
                "results": {
                    "band_gap": band_gap,
                    "efficiency": efficiency,
                    "stability": stability
                }
            })
        
        return experiments
    
    @pytest.fixture
    def hypothesis_generator(self) -> AutonomousHypothesisGenerator:
        """Create hypothesis generator instance."""
        return AutonomousHypothesisGenerator(
            min_confidence_threshold=0.6,
            statistical_significance_threshold=0.05,
            max_hypotheses_per_session=8
        )
    
    def test_pattern_analysis(self, hypothesis_generator, sample_experiments):
        """Test experimental pattern analysis."""
        patterns = hypothesis_generator.analyze_experimental_patterns(sample_experiments)
        
        # Check that all expected pattern types are analyzed
        assert "correlations" in patterns
        assert "clusters" in patterns
        assert "outliers" in patterns
        assert "trends" in patterns
        assert "phase_spaces" in patterns
        
        # Verify correlations are detected
        correlations = patterns["correlations"]
        assert len(correlations) > 0
        
        # Check correlation structure
        for corr_key, corr_data in correlations.items():
            assert "correlation" in corr_data
            assert "p_value" in corr_data
            assert "strength" in corr_data
            assert abs(corr_data["correlation"]) <= 1.0
            assert 0 <= corr_data["p_value"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, hypothesis_generator, sample_experiments):
        """Test scientific hypothesis generation."""
        target_properties = ["band_gap", "efficiency", "stability"]
        
        hypotheses = await hypothesis_generator.generate_hypotheses(
            sample_experiments, target_properties
        )
        
        # Verify hypotheses were generated
        assert len(hypotheses) > 0
        assert len(hypotheses) <= hypothesis_generator.max_hypotheses_per_session
        
        # Check hypothesis structure
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, ScientificHypothesis)
            assert hypothesis.hypothesis_text != ""
            assert isinstance(hypothesis.hypothesis_type, HypothesisType)
            assert isinstance(hypothesis.confidence, HypothesisConfidence)
            assert len(hypothesis.supporting_evidence) > 0
            assert len(hypothesis.falsifiable_predictions) > 0
            assert hypothesis.validation_score >= 0
    
    @pytest.mark.asyncio
    async def test_hypothesis_validation(self, hypothesis_generator, sample_experiments):
        """Test hypothesis validation against new data."""
        # Generate initial hypotheses
        target_properties = ["band_gap", "efficiency"]
        hypotheses = await hypothesis_generator.generate_hypotheses(
            sample_experiments[:30], target_properties
        )
        
        assert len(hypotheses) > 0
        
        # Validate against new experiments
        new_experiments = sample_experiments[30:40]
        hypothesis = hypotheses[0]
        
        validation_results = await hypothesis_generator.validate_hypothesis(
            hypothesis, new_experiments
        )
        
        # Check validation structure
        assert "hypothesis_id" in validation_results
        assert "validation_score" in validation_results
        assert "statistical_tests" in validation_results
        assert "overall_support" in validation_results
        assert validation_results["hypothesis_id"] == hypothesis.id
        assert 0 <= validation_results["validation_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_experiments):
        """Test convenience function for hypothesis generation."""
        hypotheses = await generate_scientific_hypotheses(
            sample_experiments, ["band_gap", "efficiency"]
        )
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, ScientificHypothesis) for h in hypotheses)
    
    def test_hypothesis_summary(self, hypothesis_generator, sample_experiments):
        """Test hypothesis summary generation."""
        # Generate some hypotheses first
        asyncio.run(hypothesis_generator.generate_hypotheses(
            sample_experiments, ["band_gap", "efficiency"]
        ))
        
        summary = hypothesis_generator.get_hypothesis_summary()
        
        assert "total_hypotheses" in summary
        assert "by_type" in summary
        assert "by_confidence" in summary
        assert "average_validation_score" in summary
        assert "top_hypotheses" in summary
        assert summary["total_hypotheses"] > 0


class TestQuantumHybridOptimizer:
    """Test suite for quantum-hybrid optimization."""
    
    @pytest.fixture
    def quantum_optimizer(self) -> QuantumHybridOptimizer:
        """Create quantum optimizer instance."""
        return QuantumHybridOptimizer(
            default_strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            backend=QuantumBackend.LOCAL_SIMULATION
        )
    
    @pytest.fixture
    def sample_parameter_space(self) -> Dict[str, tuple]:
        """Create sample parameter space for optimization."""
        return {
            "temperature": (100, 300),
            "concentration_a": (0.5, 2.0),
            "concentration_b": (0.3, 1.5),
            "reaction_time": (1, 8),
            "ph": (4, 10)
        }
    
    @pytest.fixture
    def sample_objective_function(self):
        """Create sample objective function."""
        async def objective(params: Dict[str, float]) -> float:
            # Simulate band gap optimization
            temp = params.get("temperature", 150)
            conc_a = params.get("concentration_a", 1.0)
            conc_b = params.get("concentration_b", 0.8)
            
            # Target band gap of 1.4 eV
            band_gap = 1.6 - 0.0001 * temp + 0.1 * np.exp(-abs(conc_a - 1.2))
            return abs(band_gap - 1.4)  # Minimize error
        
        return objective
    
    @pytest.mark.asyncio
    async def test_quantum_annealing_optimization(self, quantum_optimizer, sample_parameter_space, sample_objective_function):
        """Test quantum annealing optimization."""
        result = await quantum_optimizer.optimize_materials_parameters(
            parameter_space=sample_parameter_space,
            objective_function=sample_objective_function,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=50
        )
        
        # Verify result structure
        assert result.optimal_parameters is not None
        assert len(result.optimal_parameters) == len(sample_parameter_space)
        assert result.optimal_value >= 0
        assert len(result.convergence_history) > 0
        assert result.quantum_advantage > 0
        assert 0 <= result.fidelity <= 1
        assert 0 <= result.success_probability <= 1
        assert result.optimization_strategy == OptimizationStrategy.QUANTUM_ANNEALING
        
        # Check parameter bounds
        for param, value in result.optimal_parameters.items():
            min_val, max_val = sample_parameter_space[param]
            assert min_val <= value <= max_val
    
    @pytest.mark.asyncio
    async def test_vqe_optimization(self, quantum_optimizer, sample_parameter_space, sample_objective_function):
        """Test Variational Quantum Eigensolver optimization."""
        result = await quantum_optimizer.optimize_materials_parameters(
            parameter_space=sample_parameter_space,
            objective_function=sample_objective_function,
            strategy=OptimizationStrategy.VARIATIONAL_QUANTUM_EIGENSOLVER,
            max_iterations=30
        )
        
        # Verify VQE-specific results
        assert result.optimization_strategy == OptimizationStrategy.VARIATIONAL_QUANTUM_EIGENSOLVER
        assert "num_layers" in result.metadata
        assert "final_theta_params" in result.metadata
        assert result.quantum_advantage > 1.5  # VQE typically shows good advantage
    
    @pytest.mark.asyncio
    async def test_benchmark_strategies(self, quantum_optimizer, sample_parameter_space, sample_objective_function):
        """Test benchmarking of quantum strategies."""
        benchmark_results = await quantum_optimizer.benchmark_quantum_strategies(
            parameter_space=sample_parameter_space,
            objective_function=sample_objective_function,
            num_runs=3
        )
        
        # Verify benchmark structure
        assert len(benchmark_results) >= 2  # At least two strategies
        
        for strategy, results in benchmark_results.items():
            assert len(results) <= 3  # Up to 3 runs
            for result in results:
                assert result.optimal_value >= 0
                assert len(result.optimal_parameters) == len(sample_parameter_space)
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, quantum_optimizer, sample_parameter_space):
        """Test adaptive strategy selection."""
        # Small problem
        small_space = {"temperature": (100, 200), "concentration": (0.5, 1.5)}
        strategy = await quantum_optimizer.adaptive_strategy_selection(small_space)
        assert strategy in [OptimizationStrategy.QUANTUM_ANNEALING, OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM]
        
        # Large problem
        large_space = {f"param_{i}": (0, 100) for i in range(8)}
        strategy = await quantum_optimizer.adaptive_strategy_selection(large_space)
        assert strategy in [OptimizationStrategy.VARIATIONAL_QUANTUM_EIGENSOLVER, OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM]
    
    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_parameter_space, sample_objective_function):
        """Test convenience function for quantum optimization."""
        result = await optimize_with_quantum_hybrid(
            parameter_space=sample_parameter_space,
            objective_function=sample_objective_function
        )
        
        assert result.optimal_parameters is not None
        assert result.optimal_value >= 0
        assert result.quantum_advantage > 0
    
    def test_optimization_summary(self, quantum_optimizer):
        """Test optimization summary generation."""
        summary = quantum_optimizer.get_optimization_summary()
        
        # Initially no optimizations
        assert summary["total_optimizations"] == 0
        assert "summary" in summary


class TestFederatedLearningCoordinator:
    """Test suite for federated learning coordination."""
    
    @pytest.fixture
    def coordinator(self) -> FederatedLearningCoordinator:
        """Create federated learning coordinator."""
        return FederatedLearningCoordinator(
            lab_name="Test Coordinator Lab",
            institution="Test University",
            role=LabRole.COORDINATOR
        )
    
    @pytest.fixture
    def sample_lab_info(self) -> List[Dict[str, Any]]:
        """Create sample lab information."""
        return [
            {
                "name": "MIT Materials Lab",
                "institution": "MIT",
                "role": "participant",
                "endpoint": "https://mit-lab.example.com",
                "capabilities": ["materials_synthesis", "characterization", "uv_vis_spectroscopy"]
            },
            {
                "name": "Stanford Nano Lab",
                "institution": "Stanford",
                "role": "participant",
                "endpoint": "https://stanford-lab.example.com",
                "capabilities": ["materials_synthesis", "advanced_characterization", "solar_cell_testing"]
            },
            {
                "name": "UC Berkeley Lab",
                "institution": "UC Berkeley",
                "role": "validator",
                "endpoint": "https://berkeley-lab.example.com",
                "capabilities": ["characterization", "long_term_testing"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_lab_registration(self, coordinator, sample_lab_info):
        """Test laboratory registration in federation."""
        lab_info = sample_lab_info[0]
        
        lab_node = await coordinator.register_lab(lab_info)
        
        # Verify lab node creation
        assert lab_node.name == lab_info["name"]
        assert lab_node.institution == lab_info["institution"]
        assert lab_node.role == LabRole.PARTICIPANT
        assert lab_node.capabilities == lab_info["capabilities"]
        assert lab_node.trust_score > 0
        
        # Verify lab is registered
        assert lab_node.id in coordinator.connected_labs
        assert len(coordinator.connected_labs) == 1
    
    @pytest.mark.asyncio
    async def test_federated_model_creation(self, coordinator):
        """Test federated model creation."""
        model_config = {
            "name": "Federated Perovskite Model",
            "model_type": "neural_network",
            "target_properties": ["band_gap", "efficiency"],
            "parameter_size": 100,
            "privacy_level": "differential_privacy"
        }
        
        model = await coordinator.create_federated_model(model_config)
        
        # Verify model creation
        assert model.name == model_config["name"]
        assert model.target_properties == model_config["target_properties"]
        assert model.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
        assert model.parameters is not None
        assert len(model.parameters) == 100
        
        # Verify model is stored
        assert model.id in coordinator.federated_models
        assert model.id in coordinator.model_updates
    
    @pytest.mark.asyncio
    async def test_training_round(self, coordinator, sample_lab_info):
        """Test federated training round."""
        # Register labs
        for lab_info in sample_lab_info:
            await coordinator.register_lab(lab_info)
        
        # Create model
        model_config = {
            "name": "Test Model",
            "target_properties": ["band_gap"],
            "parameter_size": 50
        }
        model = await coordinator.create_federated_model(model_config)
        
        # Start training round
        round_info = await coordinator.start_training_round(model.id)
        
        # Verify training round
        assert round_info["model_id"] == model.id
        assert round_info["round_number"] > 0
        assert round_info["participating_labs"] > 0
        assert "global_parameters" in round_info
        assert len(round_info["global_parameters"]) == 50
    
    @pytest.mark.asyncio
    async def test_model_update_processing(self, coordinator, sample_lab_info):
        """Test model update processing."""
        # Setup
        lab_info = sample_lab_info[0]
        lab_node = await coordinator.register_lab(lab_info)
        
        model_config = {"name": "Test Model", "parameter_size": 20}
        model = await coordinator.create_federated_model(model_config)
        
        await coordinator.start_training_round(model.id)
        
        # Create model update
        update_data = {
            "lab_id": lab_node.id,
            "model_id": model.id,
            "round_number": coordinator.current_round,
            "parameters": np.random.normal(0, 0.1, 20).tolist(),
            "local_performance": {"accuracy": 0.85, "loss": 0.15},
            "data_size": 100,
            "computation_time": 120.0
        }
        
        # Process update
        success = await coordinator.receive_model_update(update_data)
        
        assert success is True
        assert len(coordinator.model_updates[model.id]) == 1
        
        # Verify lab statistics updated
        assert lab_node.model_updates_contributed == 1
    
    @pytest.mark.asyncio
    async def test_model_aggregation(self, coordinator, sample_lab_info):
        """Test model aggregation."""
        # Setup with multiple labs
        lab_nodes = []
        for lab_info in sample_lab_info[:2]:
            lab_node = await coordinator.register_lab(lab_info)
            lab_nodes.append(lab_node)
        
        model_config = {"name": "Test Model", "parameter_size": 10}
        model = await coordinator.create_federated_model(model_config)
        
        await coordinator.start_training_round(model.id)
        
        # Submit updates from multiple labs
        for lab_node in lab_nodes:
            update_data = {
                "lab_id": lab_node.id,
                "model_id": model.id,
                "round_number": coordinator.current_round,
                "parameters": np.random.normal(0, 0.1, 10).tolist(),
                "local_performance": {"accuracy": np.random.uniform(0.7, 0.9)},
                "data_size": np.random.randint(50, 200)
            }
            await coordinator.receive_model_update(update_data)
        
        # Aggregate updates
        success = await coordinator.aggregate_model_updates(model.id)
        
        assert success is True
        assert model.training_rounds == 1
        assert len(coordinator.training_history) == 1
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, coordinator):
        """Test federated model evaluation."""
        # Setup
        model_config = {"name": "Test Model", "parameter_size": 15}
        model = await coordinator.create_federated_model(model_config)
        
        # Mock test data
        test_data = [{"features": [1, 2, 3], "target": 0.8} for _ in range(20)]
        
        # Evaluate model
        metrics = await coordinator.evaluate_federated_model(model.id, test_data)
        
        # Verify evaluation metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "total_training_rounds" in metrics
        assert "participating_labs" in metrics
        assert "data_privacy_level" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_federation_summary(self, coordinator):
        """Test federation summary generation."""
        summary = coordinator.get_federation_summary()
        
        # Verify summary structure
        assert "federation_status" in summary
        assert "total_labs" in summary
        assert "active_labs" in summary
        assert "total_models" in summary
        assert "privacy_level" in summary
        assert "aggregation_strategy" in summary
        
        # Initially empty federation
        assert summary["total_labs"] == 0
        assert summary["total_models"] == 0
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function for creating federated network."""
        config = {
            "lab_name": "Test Federation",
            "institution": "Test Org",
            "privacy_level": "differential_privacy",
            "aggregation_strategy": "fedavg",
            "trusted_institutions": ["MIT", "Stanford"]
        }
        
        coordinator = await create_federated_materials_network(config)
        
        assert coordinator.lab_name == config["lab_name"]
        assert coordinator.institution == config["institution"]
        assert coordinator.privacy_manager.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
        assert coordinator.aggregator.aggregation_strategy == config["aggregation_strategy"]


class TestRealtimeAdaptiveProtocols:
    """Test suite for real-time adaptive protocols."""
    
    @pytest.fixture
    def adaptive_engine(self) -> AdaptiveProtocolEngine:
        """Create adaptive protocol engine."""
        return AdaptiveProtocolEngine(
            adaptation_strategy=AdaptationStrategy.BALANCED,
            max_adaptation_rate=0.15
        )
    
    @pytest.fixture
    def sample_conditions(self) -> ExperimentalCondition:
        """Create sample experimental conditions."""
        return ExperimentalCondition(
            temperature=150.0,
            pressure=1.0,
            concentration_a=1.2,
            concentration_b=0.8,
            reaction_time=3.0,
            stirring_speed=500.0,
            ph=7.0
        )
    
    @pytest.fixture
    def sample_result(self, sample_conditions) -> RealTimeResult:
        """Create sample real-time result."""
        return RealTimeResult(
            conditions=sample_conditions,
            properties={
                "band_gap": 1.45,
                "efficiency": 0.23,
                "stability": 0.85
            },
            quality_indicators={
                "signal_noise_ratio": 15.2,
                "measurement_precision": 0.98
            },
            confidence_score=0.95
        )
    
    def test_performance_monitoring(self, adaptive_engine, sample_result):
        """Test performance monitoring capabilities."""
        # Add multiple results to build history
        results = []
        for i in range(10):
            result = RealTimeResult(
                conditions=sample_result.conditions,
                properties={
                    "band_gap": 1.4 + np.random.normal(0, 0.05),
                    "efficiency": 0.25 + np.random.normal(0, 0.02),
                    "stability": 0.9 + np.random.normal(0, 0.03)
                },
                confidence_score=0.9 + np.random.uniform(0, 0.1)
            )
            results.append(result)
            
            metrics = adaptive_engine.performance_monitor.update_performance(result)
            
            # Verify metrics structure
            assert "current_performance" in metrics
            assert "average_performance" in metrics
            assert "performance_trend" in metrics
            assert "performance_volatility" in metrics
            assert 0 <= metrics["current_performance"] <= 1
            assert -1 <= metrics["performance_trend"] <= 1
    
    def test_outlier_detection(self, adaptive_engine):
        """Test outlier detection capabilities."""
        # Add normal results
        for i in range(10):
            normal_result = RealTimeResult(
                conditions=ExperimentalCondition(),
                properties={
                    "band_gap": 1.4 + np.random.normal(0, 0.02),
                    "efficiency": 0.25 + np.random.normal(0, 0.01)
                }
            )
            adaptive_engine.outlier_detector.add_result(normal_result)
        
        # Add outlier result
        outlier_result = RealTimeResult(
            conditions=ExperimentalCondition(),
            properties={
                "band_gap": 2.1,  # Extreme outlier
                "efficiency": 0.45  # Extreme outlier
            }
        )
        
        outlier_analysis = adaptive_engine.outlier_detector.add_result(outlier_result)
        
        # Verify outlier detection
        assert outlier_analysis["is_outlier"] is True
        assert outlier_analysis["outlier_score"] > 0.5
        assert outlier_analysis["outlier_type"] in ["extreme_outlier", "moderate_outlier"]
        assert "property_analysis" in outlier_analysis
    
    @pytest.mark.asyncio
    async def test_realtime_processing(self, adaptive_engine, sample_result):
        """Test real-time result processing."""
        response = await adaptive_engine.process_realtime_result(sample_result)
        
        # Verify response structure
        assert "result_id" in response
        assert "performance_metrics" in response
        assert "outlier_analysis" in response
        assert "triggered_rules" in response
        assert "adaptations_made" in response
        assert "protocol_status" in response
        assert "current_conditions" in response
        
        assert response["result_id"] == sample_result.id
        assert isinstance(response["triggered_rules"], list)
        assert isinstance(response["adaptations_made"], list)
    
    @pytest.mark.asyncio
    async def test_performance_degradation_adaptation(self, adaptive_engine):
        """Test adaptation to performance degradation."""
        # Simulate declining performance
        for i in range(15):
            declining_result = RealTimeResult(
                conditions=ExperimentalCondition(temperature=150 + i),
                properties={
                    "band_gap": 1.4 + i * 0.05,  # Moving away from target
                    "efficiency": 0.25 - i * 0.01,  # Declining efficiency
                    "stability": 0.9 - i * 0.02  # Declining stability
                },
                confidence_score=0.9
            )
            
            response = await adaptive_engine.process_realtime_result(declining_result)
            
            # Check if adaptation was triggered
            if i > 10:  # After sufficient history
                if response["adaptations_made"]:
                    adaptation = response["adaptations_made"][0]
                    assert adaptation["rule_name"] == "performance_degradation_response"
                    assert "adaptation" in adaptation
                    break
    
    @pytest.mark.asyncio
    async def test_outlier_adaptation(self, adaptive_engine):
        """Test adaptation to outlier detection."""
        # Add normal results to establish baseline
        for i in range(8):
            normal_result = RealTimeResult(
                conditions=ExperimentalCondition(),
                properties={"band_gap": 1.4 + np.random.normal(0, 0.01)}
            )
            await adaptive_engine.process_realtime_result(normal_result)
        
        # Add extreme outlier
        outlier_result = RealTimeResult(
            conditions=ExperimentalCondition(),
            properties={"band_gap": 2.2}  # Extreme outlier
        )
        
        response = await adaptive_engine.process_realtime_result(outlier_result)
        
        # Should trigger outlier investigation
        assert any("outlier" in rule for rule in response["triggered_rules"])
    
    @pytest.mark.asyncio
    async def test_safety_adaptation(self, adaptive_engine):
        """Test safety-related adaptations."""
        # Create unsafe conditions
        unsafe_result = RealTimeResult(
            conditions=ExperimentalCondition(
                temperature=350,  # Dangerously high
                pressure=25,  # Dangerously high
                ph=1  # Dangerously low
            ),
            properties={"band_gap": 1.4},
            experimental_errors=["temperature_overshoot", "pressure_warning"]
        )
        
        response = await adaptive_engine.process_realtime_result(unsafe_result)
        
        # Should trigger safety response
        assert "safety_response" in response["triggered_rules"]
        assert len(response["adaptations_made"]) > 0
        
        # Verify safety adaptations were made
        if response["adaptations_made"]:
            safety_adaptation = response["adaptations_made"][0]
            assert safety_adaptation["adaptation"]["adaptation_type"] == "safety_response"
            
            # Check that conditions were made safer
            new_conditions = safety_adaptation["adaptation"]["new_conditions"]
            assert new_conditions["temperature"] < 350
            assert new_conditions["pressure"] < 25
            assert new_conditions["ph"] > 1
    
    def test_adaptation_learning(self, adaptive_engine):
        """Test learning from adaptation outcomes."""
        # Create mock adaptation
        adaptation_id = "test_adaptation"
        adaptive_engine.adaptation_history.append({
            "adaptation": {"adaptation_type": adaptation_id},
            "timestamp": datetime.now().isoformat()
        })
        
        # Create outcome result
        outcome_result = RealTimeResult(
            conditions=ExperimentalCondition(),
            properties={"band_gap": 1.39}  # Good result near target
        )
        
        # Learn from successful outcome
        adaptive_engine.learn_from_adaptation_outcomes(
            adaptation_id, outcome_result, success=True
        )
        
        assert len(adaptive_engine.successful_adaptations) == 1
        
        # Learn from failed outcome
        adaptive_engine.learn_from_adaptation_outcomes(
            adaptation_id, outcome_result, success=False
        )
        
        assert len(adaptive_engine.failed_adaptations) == 1
    
    def test_adaptation_summary(self, adaptive_engine):
        """Test adaptation summary generation."""
        summary = adaptive_engine.get_adaptation_summary()
        
        # Verify summary structure
        assert "protocol_status" in summary
        assert "adaptation_strategy" in summary
        assert "total_adaptations" in summary
        assert "success_rate" in summary
        assert "max_adaptation_rate" in summary
        assert "current_conditions" in summary
        assert "recent_performance" in summary
        assert "rule_statistics" in summary
        assert "performance_issues_detected" in summary
        
        # Initially no adaptations
        assert summary["total_adaptations"] == 0
        assert isinstance(summary["rule_statistics"], dict)
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function for processing experimental data."""
        experimental_data = {
            "conditions": {
                "temperature": 175.0,
                "concentration_a": 1.1,
                "reaction_time": 2.5
            },
            "properties": {
                "band_gap": 1.42,
                "efficiency": 0.24
            },
            "quality_indicators": {
                "signal_noise_ratio": 12.5
            },
            "confidence_score": 0.88
        }
        
        response = await process_realtime_experiment_data(experimental_data)
        
        # Verify response
        assert "performance_metrics" in response
        assert "outlier_analysis" in response
        assert "protocol_status" in response
        assert "current_conditions" in response


class TestIntegrationScenarios:
    """Integration tests combining multiple next-generation capabilities."""
    
    @pytest.mark.asyncio
    async def test_ai_enhanced_discovery_pipeline(self):
        """Test complete AI-enhanced discovery pipeline."""
        # 1. Generate sample experimental data
        experiments = []
        for i in range(30):
            exp = {
                "id": f"exp_{i}",
                "parameters": {
                    "temperature": 120 + i * 3,
                    "concentration_a": 0.8 + i * 0.03,
                    "concentration_b": 0.6 + i * 0.02
                },
                "results": {
                    "band_gap": 1.5 - i * 0.003 + np.random.normal(0, 0.01),
                    "efficiency": 0.2 + i * 0.004 + np.random.normal(0, 0.005)
                }
            }
            experiments.append(exp)
        
        # 2. Generate hypotheses from experimental data
        hypotheses = await generate_scientific_hypotheses(
            experiments, ["band_gap", "efficiency"]
        )
        
        assert len(hypotheses) > 0
        
        # 3. Use quantum optimization to test best hypothesis
        best_hypothesis = max(hypotheses, key=lambda h: h.validation_score)
        
        parameter_space = {
            "temperature": (100, 250),
            "concentration_a": (0.5, 1.5),
            "concentration_b": (0.3, 1.2)
        }
        
        quantum_result = await optimize_with_quantum_hybrid(
            parameter_space=parameter_space,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING
        )
        
        assert quantum_result.optimal_parameters is not None
        
        # 4. Process results through adaptive protocols
        experimental_data = {
            "conditions": quantum_result.optimal_parameters,
            "properties": {
                "band_gap": 1.38,  # Good result
                "efficiency": 0.26
            },
            "confidence_score": 0.95
        }
        
        adaptation_response = await process_realtime_experiment_data(experimental_data)
        
        assert "performance_metrics" in adaptation_response
        assert adaptation_response["protocol_status"] in ["stable", "optimizing"]
    
    @pytest.mark.asyncio
    async def test_federated_quantum_optimization(self):
        """Test integration of federated learning with quantum optimization."""
        # 1. Set up federated network
        coordinator_config = {
            "lab_name": "Central Quantum Lab",
            "privacy_level": "differential_privacy",
            "aggregation_strategy": "fedavg"
        }
        
        coordinator = await create_federated_materials_network(coordinator_config)
        
        # 2. Register virtual labs
        lab_configs = [
            {
                "name": "Quantum Lab A",
                "institution": "Quantum University",
                "capabilities": ["quantum_optimization", "materials_synthesis"]
            },
            {
                "name": "Quantum Lab B", 
                "institution": "Quantum Institute",
                "capabilities": ["quantum_optimization", "characterization"]
            }
        ]
        
        for lab_config in lab_configs:
            await coordinator.register_lab(lab_config)
        
        # 3. Create federated model for quantum-enhanced optimization
        model_config = {
            "name": "Quantum-Enhanced Materials Model",
            "target_properties": ["band_gap", "efficiency"],
            "parameter_size": 50
        }
        
        model = await coordinator.create_federated_model(model_config)
        
        # 4. Use quantum optimization in federated context
        parameter_space = {
            "temperature": (100, 300),
            "concentration": (0.5, 2.0)
        }
        
        quantum_result = await optimize_with_quantum_hybrid(
            parameter_space=parameter_space,
            strategy=OptimizationStrategy.VARIATIONAL_QUANTUM_EIGENSOLVER
        )
        
        # 5. Simulate federated training round
        round_info = await coordinator.start_training_round(model.id)
        
        assert round_info["participating_labs"] == len(lab_configs)
        assert model.id in coordinator.federated_models
        assert quantum_result.optimal_parameters is not None
    
    @pytest.mark.asyncio
    async def test_hypothesis_driven_adaptive_protocols(self):
        """Test hypothesis-driven adaptive protocols."""
        # 1. Generate hypotheses from initial data
        initial_experiments = []
        for i in range(20):
            exp = {
                "parameters": {"temperature": 150 + i * 2},
                "results": {"band_gap": 1.5 - i * 0.01}
            }
            initial_experiments.append(exp)
        
        hypotheses = await generate_scientific_hypotheses(
            initial_experiments, ["band_gap"]
        )
        
        # 2. Set up adaptive engine
        adaptive_engine = AdaptiveProtocolEngine(
            adaptation_strategy=AdaptationStrategy.EXPLORATORY
        )
        
        # 3. Test hypothesis-guided adaptation
        # Simulate discovering that temperature has strong correlation
        test_result = RealTimeResult(
            conditions=ExperimentalCondition(temperature=180),
            properties={"band_gap": 1.32},  # Better than expected
            confidence_score=0.95
        )
        
        response = await adaptive_engine.process_realtime_result(test_result)
        
        # Should lead to protocol optimization
        assert response["protocol_status"] in ["stable", "optimizing", "learning"]
        
        # 4. Validate hypothesis with new results
        if hypotheses:
            hypothesis = hypotheses[0]
            generator = AutonomousHypothesisGenerator()
            validation = await generator.validate_hypothesis(hypothesis, [
                {"parameters": {"temperature": 180}, "results": {"band_gap": 1.32}}
            ])
            
            assert "validation_score" in validation
            assert validation["validation_score"] >= 0


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for next-generation capabilities."""
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation_performance(self):
        """Benchmark hypothesis generation performance."""
        # Large dataset
        experiments = []
        for i in range(200):
            exp = {
                "parameters": {
                    "temperature": 100 + i,
                    "concentration_a": 0.5 + i * 0.01,
                    "concentration_b": 0.3 + i * 0.008
                },
                "results": {
                    "band_gap": 1.6 - i * 0.002 + np.random.normal(0, 0.01),
                    "efficiency": 0.15 + i * 0.002 + np.random.normal(0, 0.005)
                }
            }
            experiments.append(exp)
        
        start_time = datetime.now()
        
        generator = AutonomousHypothesisGenerator(max_hypotheses_per_session=5)
        hypotheses = await generator.generate_hypotheses(
            experiments, ["band_gap", "efficiency"]
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert duration < 10.0  # Should complete within 10 seconds
        assert len(hypotheses) > 0
        assert len(hypotheses) <= 5
        
        print(f"Generated {len(hypotheses)} hypotheses from {len(experiments)} experiments in {duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        parameter_space = {f"param_{i}": (0, 10) for i in range(6)}
        
        start_time = datetime.now()
        
        optimizer = QuantumHybridOptimizer()
        result = await optimizer.optimize_materials_parameters(
            parameter_space=parameter_space,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_iterations=100
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert duration < 30.0  # Should complete within 30 seconds
        assert result.optimal_parameters is not None
        assert result.quantum_advantage > 0
        
        print(f"Quantum optimization completed in {duration:.2f}s with {result.quantum_advantage:.1f}x advantage")
    
    @pytest.mark.asyncio
    async def test_realtime_processing_performance(self):
        """Benchmark real-time processing performance."""
        adaptive_engine = AdaptiveProtocolEngine()
        
        # Simulate burst of real-time results
        results = []
        for i in range(50):
            result = RealTimeResult(
                conditions=ExperimentalCondition(temperature=150 + i),
                properties={"band_gap": 1.4 + np.random.normal(0, 0.05)},
                confidence_score=0.9
            )
            results.append(result)
        
        start_time = datetime.now()
        
        responses = []
        for result in results:
            response = await adaptive_engine.process_realtime_result(result)
            responses.append(response)
        
        duration = (datetime.now() - start_time).total_seconds()
        avg_processing_time = duration / len(results)
        
        # Performance assertions
        assert avg_processing_time < 0.1  # Should process each result in < 100ms
        assert len(responses) == len(results)
        
        print(f"Processed {len(results)} real-time results in {duration:.2f}s (avg: {avg_processing_time*1000:.1f}ms per result)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])