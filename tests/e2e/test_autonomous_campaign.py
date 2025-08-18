"""End-to-end tests for autonomous campaigns."""

from unittest.mock import patch

import pytest

from materials_orchestrator.core import AutonomousLab, MaterialsObjective
from materials_orchestrator.planners import BayesianPlanner
from tests.utils import MockDatabase, MockInstrument, MockRobot


class TestAutonomousCampaign:
    """Test complete autonomous campaign workflows."""

    @pytest.fixture
    def mock_lab_setup(self):
        """Set up a complete mock laboratory."""
        robot = MockRobot("test_robot_001", ["synthesis", "characterization"])
        instrument = MockInstrument("test_spec_001", ["band_gap", "efficiency"])
        database = MockDatabase()

        planner = BayesianPlanner(
            acquisition_function="expected_improvement", target_property="band_gap"
        )

        # Mock the database connection in the lab
        with patch("materials_orchestrator.core.get_database") as mock_get_db:
            mock_get_db.return_value = database

            lab = AutonomousLab(
                robots=[robot], instruments=[instrument], planner=planner
            )

            yield {
                "lab": lab,
                "robot": robot,
                "instrument": instrument,
                "database": database,
            }

    def test_simple_optimization_campaign(self, mock_lab_setup):
        """Test a simple optimization campaign from start to finish."""
        lab = mock_lab_setup["lab"]

        # Define optimization objective
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
            material_system="perovskites",
            success_threshold=1.4,
        )

        # Define parameter space
        param_space = {
            "temperature": (100, 300),
            "concentration": (0.1, 2.0),
            "time": (1, 24),
        }

        # Run a short campaign
        with patch.object(lab, "_execute_experiment") as mock_execute:
            # Mock successful experiment results
            mock_execute.side_effect = [
                {
                    "experiment_id": f"exp_{i:03d}",
                    "parameters": {
                        "temperature": 150 + i * 10,
                        "concentration": 1.0,
                        "time": 12,
                    },
                    "results": {"band_gap": 1.5 - i * 0.05, "efficiency": 15 + i * 2},
                    "status": "completed",
                }
                for i in range(5)
            ]

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=3,
                max_experiments=5,
                stop_on_target=True,
            )

        # Verify campaign results
        assert campaign is not None
        assert campaign.total_experiments >= 3
        assert campaign.total_experiments <= 5
        assert len(campaign.experiments) == campaign.total_experiments

        # Check that optimization is working (later experiments should be better)
        if len(campaign.experiments) > 1:
            first_result = campaign.experiments[0]["results"]["band_gap"]
            last_result = campaign.experiments[-1]["results"]["band_gap"]

            # Should be moving toward target (1.4)
            assert abs(last_result - 1.4) <= abs(first_result - 1.4)

    @pytest.mark.slow
    def test_multi_objective_campaign(self, mock_lab_setup):
        """Test campaign with multiple competing objectives."""
        lab = mock_lab_setup["lab"]

        objective = MaterialsObjective(
            target_property=["band_gap", "efficiency"],
            target_range=[(1.2, 1.6), (20, 30)],
            optimization_direction=["target", "maximize"],
            material_system="perovskites",
        )

        param_space = {
            "temperature": (100, 300),
            "concentration": (0.1, 2.0),
            "time": (1, 24),
            "pH": (3, 11),
        }

        with patch.object(lab, "_execute_experiment") as mock_execute:
            # Mock results with trade-offs between objectives
            mock_execute.side_effect = [
                {
                    "experiment_id": f"exp_{i:03d}",
                    "parameters": {
                        "temperature": 150 + i * 15,
                        "concentration": 1.0 + i * 0.1,
                        "time": 12 + i,
                        "pH": 7,
                    },
                    "results": {
                        "band_gap": 1.4
                        + (i - 5) * 0.02,  # Trade-off: efficiency vs band_gap
                        "efficiency": 25 - abs(i - 5) * 0.5,
                    },
                    "status": "completed",
                }
                for i in range(10)
            ]

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=5,
                max_experiments=10,
                convergence_patience=15,
            )

        # Verify multi-objective optimization
        assert campaign is not None
        assert campaign.total_experiments == 10

        # Should have found Pareto-optimal solutions
        pareto_front = campaign.get_pareto_front()
        assert len(pareto_front) >= 2  # Should have multiple non-dominated solutions

    def test_campaign_with_failures(self, mock_lab_setup):
        """Test campaign resilience to experiment failures."""
        lab = mock_lab_setup["lab"]

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        with patch.object(lab, "_execute_experiment") as mock_execute:
            # Mix of successful and failed experiments
            results = []
            for i in range(8):
                if i in [2, 5]:  # Simulate failures
                    results.append(
                        {
                            "experiment_id": f"exp_{i:03d}",
                            "parameters": {
                                "temperature": 150 + i * 10,
                                "concentration": 1.0,
                            },
                            "results": {},
                            "status": "failed",
                            "error": "Synthesis failure",
                        }
                    )
                else:
                    results.append(
                        {
                            "experiment_id": f"exp_{i:03d}",
                            "parameters": {
                                "temperature": 150 + i * 10,
                                "concentration": 1.0,
                            },
                            "results": {"band_gap": 1.5 - i * 0.03},
                            "status": "completed",
                        }
                    )

            mock_execute.side_effect = results

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=3,
                max_experiments=8,
                failure_tolerance=0.3,  # Allow 30% failure rate
            )

        # Verify campaign handled failures gracefully
        assert campaign is not None
        successful_experiments = [
            exp for exp in campaign.experiments if exp["status"] == "completed"
        ]
        failed_experiments = [
            exp for exp in campaign.experiments if exp["status"] == "failed"
        ]

        assert len(successful_experiments) >= 4  # At least some successful
        assert len(failed_experiments) == 2  # Expected failures
        assert campaign.success_rate >= 0.7  # Above tolerance threshold

    def test_early_convergence(self, mock_lab_setup):
        """Test early stopping when optimization converges."""
        lab = mock_lab_setup["lab"]

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
            success_threshold=1.41,  # Tight threshold for quick convergence
        )

        param_space = {"temperature": (100, 300)}

        with patch.object(lab, "_execute_experiment") as mock_execute:
            # Simulate quick convergence to optimal value
            mock_execute.side_effect = [
                {
                    "experiment_id": f"exp_{i:03d}",
                    "parameters": {"temperature": 150 + i * 5},
                    "results": {"band_gap": 1.5 - i * 0.1},  # Converges to ~1.4
                    "status": "completed",
                }
                for i in range(20)  # Would take 20 experiments if no early stopping
            ]

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=3,
                max_experiments=20,
                convergence_patience=3,  # Stop after 3 experiments without improvement
                stop_on_target=True,
            )

        # Should stop early when target is reached
        assert campaign is not None
        assert campaign.total_experiments < 15  # Should stop before max
        assert campaign.converged is True

        # Best result should be close to target
        best_result = campaign.best_result["band_gap"]
        assert abs(best_result - 1.41) < 0.1

    def test_resource_constraints(self, mock_lab_setup):
        """Test campaign with resource constraints."""
        lab = mock_lab_setup["lab"]
        robot = mock_lab_setup["robot"]

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        # Simulate robot becoming unavailable
        def mock_execute_with_constraints(experiment):
            if len(robot.protocols_executed) >= 3:
                robot.status = "maintenance"
                raise Exception("Robot unavailable for maintenance")

            return {
                "experiment_id": experiment["experiment_id"],
                "parameters": experiment["parameters"],
                "results": {"band_gap": 1.4},
                "status": "completed",
            }

        with patch.object(
            lab, "_execute_experiment", side_effect=mock_execute_with_constraints
        ):
            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=2,
                max_experiments=10,
                handle_resource_constraints=True,
            )

        # Should handle resource constraints gracefully
        assert campaign is not None
        assert campaign.total_experiments <= 3  # Limited by robot availability
        assert any(
            "resource" in exp.get("error", "").lower()
            for exp in campaign.experiments
            if exp.get("status") == "failed"
        )

    def test_campaign_persistence(self, mock_lab_setup):
        """Test that campaign state is properly persisted."""
        lab = mock_lab_setup["lab"]
        database = mock_lab_setup["database"]

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        param_space = {"temperature": (100, 300)}

        with patch.object(lab, "_execute_experiment") as mock_execute:
            mock_execute.side_effect = [
                {
                    "experiment_id": f"exp_{i:03d}",
                    "parameters": {"temperature": 150 + i * 10},
                    "results": {"band_gap": 1.5 - i * 0.05},
                    "status": "completed",
                }
                for i in range(5)
            ]

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=3,
                max_experiments=5,
                save_intermediate=True,
            )

        # Verify data was stored in database
        stored_experiments = database.get_experiments(campaign.campaign_id)
        assert len(stored_experiments) == 5

        stored_campaigns = database.campaigns
        assert len(stored_campaigns) >= 1

        # Verify campaign can be reconstructed from stored data
        campaign_data = stored_campaigns[-1]
        assert campaign_data["campaign_id"] == campaign.campaign_id
        assert campaign_data["objective"]["target_property"] == "band_gap"
