"""Test enhanced core implementation."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials_orchestrator import (
    AutonomousLab,
    BayesianPlanner,
    CampaignResult,
    Experiment,
    MaterialsObjective,
    RandomPlanner,
)


class TestMaterialsObjective:
    """Test MaterialsObjective functionality."""

    def test_creation(self):
        """Test objective creation."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        assert objective.target_property == "band_gap"
        assert objective.target_range == (1.2, 1.6)
        assert objective.optimization_direction == "target"

    def test_validation(self):
        """Test objective validation."""
        with pytest.raises(ValueError):
            MaterialsObjective(
                target_property="band_gap",
                target_range=(1.6, 1.2),  # Invalid range
                optimization_direction="target",
            )

        with pytest.raises(ValueError):
            MaterialsObjective(
                target_property="band_gap",
                target_range=(1.2, 1.6),
                optimization_direction="invalid",  # Invalid direction
            )

    def test_success_evaluation(self):
        """Test success evaluation."""
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6), success_threshold=1.4
        )

        assert objective.evaluate_success(1.4) == True
        assert objective.evaluate_success(1.0) == False
        assert objective.evaluate_success(2.0) == False

    def test_fitness_calculation(self):
        """Test fitness calculation."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="maximize",
        )

        assert objective.calculate_fitness(1.5) > objective.calculate_fitness(1.0)


class TestExperiment:
    """Test Experiment class."""

    def test_creation(self):
        """Test experiment creation."""
        exp = Experiment(
            parameters={"temp": 150, "time": 3}, results={"band_gap": 1.45}
        )

        assert exp.parameters["temp"] == 150
        assert exp.results["band_gap"] == 1.45
        assert exp.status == "pending"
        assert exp.id is not None

    def test_to_dict(self):
        """Test experiment serialization."""
        exp = Experiment(
            parameters={"temp": 150}, results={"band_gap": 1.45}, status="completed"
        )

        data = exp.to_dict()
        assert "id" in data
        assert "timestamp" in data
        assert data["parameters"]["temp"] == 150
        assert data["results"]["band_gap"] == 1.45
        assert data["status"] == "completed"


class TestPlanners:
    """Test experiment planners."""

    def test_random_planner(self):
        """Test random planner."""
        planner = RandomPlanner()
        param_space = {"temp": (100, 300), "time": (1, 24)}

        suggestions = planner.suggest_next(5, param_space, [])

        assert len(suggestions) == 5
        for suggestion in suggestions:
            assert "temp" in suggestion
            assert "time" in suggestion
            assert 100 <= suggestion["temp"] <= 300
            assert 1 <= suggestion["time"] <= 24

    def test_bayesian_planner_fallback(self):
        """Test Bayesian planner fallback to random."""
        planner = BayesianPlanner(target_property="band_gap")
        param_space = {"temp": (100, 300), "time": (1, 24)}

        # With no previous results, should fall back to random
        suggestions = planner.suggest_next(3, param_space, [])

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert "temp" in suggestion
            assert "time" in suggestion


class TestAutonomousLab:
    """Test AutonomousLab functionality."""

    def test_creation(self):
        """Test lab creation."""
        lab = AutonomousLab(robots=["robot1", "robot2"], instruments=["xrd", "uv_vis"])

        assert lab.robots == ["robot1", "robot2"]
        assert lab.instruments == ["xrd", "uv_vis"]
        assert lab.total_experiments == 0
        assert lab.success_rate == 0.0

    def test_single_experiment(self):
        """Test running a single experiment."""
        lab = AutonomousLab()

        experiment = lab.run_experiment(
            {
                "temperature": 150,
                "time": 3,
                "precursor_A_conc": 1.0,
                "precursor_B_conc": 1.0,
            }
        )

        assert experiment.status in ["completed", "failed"]
        if experiment.status == "completed":
            assert "band_gap" in experiment.results
            assert isinstance(experiment.results["band_gap"], float)
        assert lab.total_experiments == 1

    def test_campaign_execution(self):
        """Test campaign execution."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        param_space = {
            "temperature": (100, 300),
            "precursor_A_conc": (0.5, 2.0),
            "precursor_B_conc": (0.5, 2.0),
            "reaction_time": (1, 10),
        }

        planner = RandomPlanner()
        lab = AutonomousLab(planner=planner)

        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=5,
            max_experiments=10,
            stop_on_target=False,
        )

        assert isinstance(campaign, CampaignResult)
        assert campaign.total_experiments <= 10
        assert campaign.total_experiments >= 5
        assert campaign.objective == objective
        assert len(campaign.experiments) == campaign.total_experiments

        # Check that campaign has some successful experiments
        successful = [exp for exp in campaign.experiments if exp.status == "completed"]
        assert len(successful) > 0

    def test_campaign_with_bayesian_optimization(self):
        """Test campaign with Bayesian optimization."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.3, 1.5),
            optimization_direction="target",
        )

        param_space = {
            "temperature": (120, 200),
            "precursor_A_conc": (0.8, 1.5),
        }

        planner = BayesianPlanner(target_property="band_gap")
        lab = AutonomousLab(planner=planner)

        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=8,  # Enough for GP
            max_experiments=20,
            stop_on_target=False,
        )

        assert campaign.total_experiments <= 20
        assert campaign.total_experiments >= 8

        # Should have some successful experiments
        assert campaign.successful_experiments > 0
        assert campaign.success_rate > 0

    def test_convergence_tracking(self):
        """Test convergence tracking."""
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.3, 1.5), success_threshold=1.4
        )

        param_space = {"temperature": (140, 160)}

        lab = AutonomousLab()
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=10,
            max_experiments=15,
            convergence_patience=5,
        )

        # Should have convergence history
        assert len(campaign.convergence_history) > 0

        # Each entry should have required fields
        for entry in campaign.convergence_history:
            assert "experiment" in entry
            assert "best_fitness" in entry
            assert "current_value" in entry


class TestCampaignResult:
    """Test CampaignResult functionality."""

    def test_metrics_calculation(self):
        """Test campaign metrics."""
        from datetime import datetime, timedelta

        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )

        # Create mock experiments
        experiments = [
            Experiment(status="completed", results={"band_gap": 1.4}),
            Experiment(status="completed", results={"band_gap": 1.3}),
            Experiment(status="failed", results={}),
        ]

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)

        campaign = CampaignResult(
            campaign_id="test",
            objective=objective,
            best_material={"parameters": {}, "properties": {"band_gap": 1.4}},
            total_experiments=3,
            successful_experiments=2,
            best_properties={"band_gap": 1.4},
            convergence_history=[],
            experiments=experiments,
            start_time=start_time,
            end_time=end_time,
        )

        assert campaign.success_rate == 2 / 3
        assert abs(campaign.duration - 2.0) < 0.1
        assert campaign.get_best_fitness() > 0  # Should be positive for this objective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
