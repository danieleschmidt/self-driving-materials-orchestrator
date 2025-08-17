"""Unit tests for experiment planners."""

import pytest
from materials_orchestrator.planners import BayesianPlanner, RandomPlanner, GridPlanner


class TestBayesianPlanner:
    """Test BayesianPlanner class."""

    def test_initialization(self):
        """Test planner initializes with correct defaults."""
        planner = BayesianPlanner()

        assert planner.acquisition_function == "expected_improvement"
        assert planner.batch_size == 1
        assert planner.kernel == "matern"
        assert planner.exploration_factor == 0.1

    def test_suggest_next(self):
        """Test suggestion generation."""
        planner = BayesianPlanner()
        param_space = {
            "temperature": (100, 300),
            "concentration": (0.1, 2.0),
            "time": (1, 24),
        }

        suggestions = planner.suggest_next(
            n_suggestions=5, param_space=param_space, previous_results=[]
        )

        assert len(suggestions) == 5
        for suggestion in suggestions:
            assert "temperature" in suggestion
            assert "concentration" in suggestion
            assert "time" in suggestion
            assert 100 <= suggestion["temperature"] <= 300
            assert 0.1 <= suggestion["concentration"] <= 2.0
            assert 1 <= suggestion["time"] <= 24


class TestRandomPlanner:
    """Test RandomPlanner class."""

    def test_suggest_next(self):
        """Test random suggestion generation."""
        planner = RandomPlanner()
        param_space = {"temperature": (100, 300), "pressure": (1, 10)}

        suggestions = planner.suggest_next(
            n_suggestions=3, param_space=param_space, previous_results=[]
        )

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert "temperature" in suggestion
            assert "pressure" in suggestion
            assert 100 <= suggestion["temperature"] <= 300
            assert 1 <= suggestion["pressure"] <= 10


class TestGridPlanner:
    """Test GridPlanner class."""

    def test_initialization(self):
        """Test grid planner initialization."""
        planner = GridPlanner(grid_density=20)
        assert planner.grid_density == 20

    def test_suggest_next(self):
        """Test grid-based suggestion generation."""
        planner = GridPlanner(grid_density=5)
        param_space = {"temperature": (100, 200), "time": (1, 2)}

        suggestions = planner.suggest_next(
            n_suggestions=10, param_space=param_space, previous_results=[]
        )

        assert len(suggestions) <= 10  # May be less if grid has fewer points
        for suggestion in suggestions:
            assert "temperature" in suggestion
            assert "time" in suggestion
            assert 100 <= suggestion["temperature"] <= 200
            assert 1 <= suggestion["time"] <= 2
