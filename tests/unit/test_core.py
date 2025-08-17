"""Unit tests for core functionality."""

import pytest
from materials_orchestrator.core import AutonomousLab, MaterialsObjective, LabStatus


class TestMaterialsObjective:
    """Test MaterialsObjective class."""

    def test_valid_objective_creation(self):
        """Test creating a valid materials objective."""
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="minimize_variance",
            material_system="perovskites",
        )

        assert objective.target_property == "band_gap"
        assert objective.target_range == (1.2, 1.6)
        assert objective.optimization_direction == "minimize_variance"
        assert objective.material_system == "perovskites"

    def test_invalid_optimization_direction(self):
        """Test that invalid optimization direction raises error."""
        with pytest.raises(ValueError, match="Invalid optimization direction"):
            MaterialsObjective(
                target_property="band_gap",
                target_range=(1.2, 1.6),
                optimization_direction="invalid_direction",
            )


class TestAutonomousLab:
    """Test AutonomousLab class."""

    def test_lab_initialization(self):
        """Test lab initializes with correct defaults."""
        lab = AutonomousLab()

        assert lab.robots == []
        assert lab.instruments == []
        assert lab.status == LabStatus.IDLE
        assert lab.total_experiments == 0
        assert lab.best_material is None

    def test_lab_initialization_with_config(self):
        """Test lab initialization with custom configuration."""
        robots = ["synthesis_robot", "characterization_robot"]
        instruments = ["xrd", "uv_vis"]

        lab = AutonomousLab(
            robots=robots, instruments=instruments, database_url="mongodb://test:27017/"
        )

        assert lab.robots == robots
        assert lab.instruments == instruments
        assert lab.database_url == "mongodb://test:27017/"

    def test_run_campaign(self):
        """Test running a basic campaign."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )

        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        result = lab.run_campaign(
            objective=objective, param_space=param_space, max_experiments=10
        )

        assert result.total_experiments > 0
        assert result.best_material is not None
        assert "band_gap" in result.best_properties
        assert lab.status == LabStatus.IDLE  # Should return to idle after campaign
