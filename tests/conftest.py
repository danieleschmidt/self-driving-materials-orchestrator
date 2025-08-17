"""Pytest configuration and shared fixtures."""

import pytest
from materials_orchestrator.core import AutonomousLab, MaterialsObjective
from materials_orchestrator.planners import BayesianPlanner


@pytest.fixture
def sample_objective():
    """Create a sample materials objective for testing."""
    return MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),
        optimization_direction="minimize_variance",
        material_system="perovskites",
    )


@pytest.fixture
def sample_lab():
    """Create a sample autonomous lab for testing."""
    return AutonomousLab(
        robots=["test_robot"],
        instruments=["test_instrument"],
        planner=BayesianPlanner(),
    )


@pytest.fixture
def sample_param_space():
    """Create a sample parameter space for testing."""
    return {
        "temperature": (100, 300),
        "concentration": (0.1, 2.0),
        "time": (1, 24),
        "pH": (3, 11),
    }


@pytest.fixture
def sample_results():
    """Create sample experiment results for testing."""
    return [
        {
            "parameters": {
                "temperature": 150,
                "concentration": 1.0,
                "time": 12,
                "pH": 7,
            },
            "results": {"band_gap": 1.45, "efficiency": 15.2},
        },
        {
            "parameters": {
                "temperature": 200,
                "concentration": 1.5,
                "time": 8,
                "pH": 6,
            },
            "results": {"band_gap": 1.52, "efficiency": 18.7},
        },
    ]
