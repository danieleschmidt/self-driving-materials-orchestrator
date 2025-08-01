"""End-to-end tests for complete autonomous campaigns."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from materials_orchestrator.core import AutonomousLab, MaterialsObjective
from materials_orchestrator.planners import BayesianPlanner
from tests.fixtures.mock_robots import create_mock_robot_orchestrator
from tests.fixtures.materials_data import PARAMETER_SPACES, generate_virtual_experiment_result


@pytest.mark.asyncio
@pytest.mark.slow
class TestFullCampaign:
    """Test complete autonomous materials discovery campaigns."""
    
    async def test_complete_perovskite_campaign(self):
        """Test a complete perovskite optimization campaign."""
        # Setup objective
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.4, 1.6),
            optimization_direction="minimize_variance",
            material_system="perovskites"
        )
        
        # Setup mock lab with virtual experiments
        with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = create_mock_robot_orchestrator()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Create lab
            lab = AutonomousLab(
                robots=["opentrons", "chemspeed"],
                instruments=["uv_vis", "xrd"],
                planner=BayesianPlanner(
                    acquisition_function="expected_improvement",
                    batch_size=3
                )
            )
            
            # Mock the experiment execution to return virtual results
            async def mock_run_experiment(parameters):
                # Simulate experiment time
                await asyncio.sleep(0.1)
                return generate_virtual_experiment_result(parameters, "perovskites")
            
            lab._run_single_experiment = mock_run_experiment
            
            # Run short campaign
            campaign_results = await lab.run_campaign(
                objective=objective,
                param_space=PARAMETER_SPACES["perovskite_synthesis"],
                initial_samples=5,
                max_experiments=20,
                convergence_threshold=0.1
            )
            
            # Verify campaign results
            assert campaign_results["status"] == "completed"
            assert campaign_results["total_experiments"] >= 5
            assert campaign_results["total_experiments"] <= 20
            assert "best_material" in campaign_results
            assert "convergence_history" in campaign_results
            
            # Check that we found materials in target range
            best_gap = campaign_results["best_material"]["properties"]["band_gap"]
            assert 1.0 <= best_gap <= 2.0  # Reasonable range
    
    async def test_multi_objective_optimization(self):
        """Test multi-objective optimization campaign."""
        objectives = [
            MaterialsObjective(
                target_property="band_gap",
                target_range=(1.4, 1.6),
                optimization_direction="target",
                weight=0.6
            ),
            MaterialsObjective(
                target_property="efficiency",
                target_range=(18, 25),
                optimization_direction="maximize",
                weight=0.4
            )
        ]
        
        with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = create_mock_robot_orchestrator()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            lab = AutonomousLab(
                robots=["opentrons"],
                instruments=["uv_vis"],
                planner=BayesianPlanner(multi_objective=True)
            )
            
            # Mock experiment execution
            async def mock_run_experiment(parameters):
                await asyncio.sleep(0.05)
                return generate_virtual_experiment_result(parameters)
            
            lab._run_single_experiment = mock_run_experiment
            
            # Run multi-objective campaign
            results = await lab.run_multi_objective_campaign(
                objectives=objectives,
                param_space=PARAMETER_SPACES["perovskite_synthesis"],
                initial_samples=8,
                max_experiments=25
            )
            
            assert results["status"] == "completed"
            assert "pareto_front" in results
            assert len(results["pareto_front"]) > 0
            assert "hypervolume" in results["metrics"]
    
    async def test_campaign_with_failures(self):
        """Test campaign resilience to experiment failures."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.5, 1.6),
            optimization_direction="target"
        )
        
        with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = create_mock_robot_orchestrator()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            lab = AutonomousLab(
                robots=["opentrons"],
                instruments=["uv_vis"],
                planner=BayesianPlanner()
            )
            
            # Mock experiment with 30% failure rate
            failure_count = 0
            async def mock_run_experiment_with_failures(parameters):
                nonlocal failure_count
                await asyncio.sleep(0.05)
                
                if failure_count % 3 == 0:  # Every 3rd experiment fails
                    failure_count += 1
                    raise RuntimeError("Simulated experiment failure")
                
                failure_count += 1
                return generate_virtual_experiment_result(parameters)
            
            lab._run_single_experiment = mock_run_experiment_with_failures
            
            # Campaign should handle failures gracefully
            results = await lab.run_campaign(
                objective=objective,
                param_space=PARAMETER_SPACES["perovskite_synthesis"],
                initial_samples=6,
                max_experiments=18,
                max_failures=10
            )
            
            assert results["status"] in ["completed", "stopped"]
            assert results["failed_experiments"] > 0
            assert results["successful_experiments"] > 0
            assert "failure_analysis" in results
    
    async def test_real_time_adaptation(self):
        """Test real-time adaptation to promising results."""
        objective = MaterialsObjective(
            target_property="efficiency",
            target_range=(20, 25),
            optimization_direction="maximize"
        )
        
        with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = create_mock_robot_orchestrator()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            lab = AutonomousLab(
                robots=["opentrons"],
                instruments=["uv_vis"],
                planner=BayesianPlanner(
                    adaptive_batch_size=True,
                    exploration_factor=0.1
                )
            )
            
            results_history = []
            
            async def mock_run_experiment_adaptive(parameters):
                await asyncio.sleep(0.05)
                result = generate_virtual_experiment_result(parameters)
                results_history.append((parameters, result))
                
                # Simulate finding a promising region
                if parameters.get("temperature", 100) > 150:
                    result["efficiency"] *= 1.5  # Boost efficiency in high-temp region
                
                return result
            
            lab._run_single_experiment = mock_run_experiment_adaptive
            
            results = await lab.run_adaptive_campaign(
                objective=objective,
                param_space=PARAMETER_SPACES["perovskite_synthesis"],
                initial_samples=5,
                max_experiments=20,
                adaptation_interval=5
            )
            
            assert results["status"] == "completed"
            assert len(results_history) > 5
            assert "adaptation_history" in results
            
            # Check that later experiments explore the promising region more
            later_experiments = results_history[-10:]
            high_temp_count = sum(1 for params, _ in later_experiments 
                                if params.get("temperature", 100) > 150)
            
            # Should have explored high-temperature region more in later experiments
            assert high_temp_count > 2


@pytest.mark.asyncio 
@pytest.mark.integration
class TestSystemIntegration:
    """Test integration between major system components."""
    
    async def test_database_experiment_logging(self):
        """Test that experiments are properly logged to database."""
        with patch('materials_orchestrator.database.ExperimentDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock database operations
            stored_experiments = []
            
            async def mock_store_experiment(experiment):
                stored_experiments.append(experiment)
                return f"exp_{len(stored_experiments):03d}"
            
            mock_db.store_experiment = mock_store_experiment
            mock_db.get_campaign_experiments.return_value = []
            
            # Run campaign with database logging
            objective = MaterialsObjective(
                target_property="band_gap",
                target_range=(1.5, 1.6)
            )
            
            with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = create_mock_robot_orchestrator()
                mock_orchestrator_class.return_value = mock_orchestrator
                
                lab = AutonomousLab(
                    robots=["opentrons"],
                    instruments=["uv_vis"],
                    planner=BayesianPlanner(),
                    database=mock_db
                )
                
                async def mock_run_experiment(parameters):
                    return generate_virtual_experiment_result(parameters)
                
                lab._run_single_experiment = mock_run_experiment
                
                await lab.run_campaign(
                    objective=objective,
                    param_space=PARAMETER_SPACES["perovskite_synthesis"],
                    initial_samples=3,
                    max_experiments=8
                )
                
                # Verify experiments were stored
                assert len(stored_experiments) >= 3
                
                for exp in stored_experiments:
                    assert "parameters" in exp
                    assert "results" in exp
                    assert "timestamp" in exp
                    assert "campaign_id" in exp
    
    async def test_safety_system_integration(self):
        """Test integration with safety monitoring system."""
        with patch('materials_orchestrator.safety.SafetyMonitor') as mock_safety_class:
            mock_safety = Mock()
            mock_safety_class.return_value = mock_safety
            
            # Mock safety checks
            safety_violations = []
            
            def mock_check_safety(parameters):
                if parameters.get("temperature", 0) > 250:
                    safety_violations.append("Temperature too high")
                    return False
                return True
            
            mock_safety.check_experiment_safety = mock_check_safety
            mock_safety.get_violations.return_value = safety_violations
            
            objective = MaterialsObjective(
                target_property="band_gap",
                target_range=(1.5, 1.6)
            )
            
            with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = create_mock_robot_orchestrator()
                mock_orchestrator_class.return_value = mock_orchestrator
                
                lab = AutonomousLab(
                    robots=["chemspeed"],
                    instruments=["uv_vis"],
                    planner=BayesianPlanner(),
                    safety_monitor=mock_safety
                )
                
                async def mock_run_experiment(parameters):
                    # Safety system should prevent dangerous experiments
                    if not mock_safety.check_experiment_safety(parameters):
                        raise RuntimeError("Safety violation prevented experiment")
                    return generate_virtual_experiment_result(parameters)
                
                lab._run_single_experiment = mock_run_experiment
                
                # Use parameter space that includes unsafe temperatures
                unsafe_param_space = PARAMETER_SPACES["perovskite_synthesis"].copy()
                unsafe_param_space["temperature"] = (100, 300)  # Includes unsafe temps
                
                results = await lab.run_campaign(
                    objective=objective,
                    param_space=unsafe_param_space,
                    initial_samples=5,
                    max_experiments=15,
                    safety_enabled=True
                )
                
                # Should have prevented some experiments
                assert len(safety_violations) > 0
                assert "safety_interventions" in results
                assert results["safety_interventions"] > 0