"""Enhanced pytest configuration with comprehensive fixtures."""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from tests.fixtures.mock_robots import create_mock_robot_orchestrator, MockInstrument


# Configure asyncio for tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Database fixtures
@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    db = Mock()
    
    # In-memory storage for testing
    experiments = []
    campaigns = []
    materials = []
    
    def store_experiment(experiment):
        experiment["_id"] = f"exp_{len(experiments):06d}"
        experiments.append(experiment)
        return experiment["_id"]
    
    def store_campaign(campaign):
        campaign["_id"] = f"camp_{len(campaigns):06d}"
        campaigns.append(campaign)
        return campaign["_id"]
    
    def get_experiments(campaign_id=None, filters=None):
        results = experiments.copy()
        if campaign_id:
            results = [exp for exp in results if exp.get("campaign_id") == campaign_id]
        if filters:
            # Simple filter implementation for testing
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range queries
                    if "$gte" in value and "$lte" in value:
                        results = [exp for exp in results 
                                 if value["$gte"] <= exp.get(key, 0) <= value["$lte"]]
                else:
                    results = [exp for exp in results if exp.get(key) == value]
        return results
    
    def get_campaign(campaign_id):
        for camp in campaigns:
            if camp["_id"] == campaign_id:
                return camp
        return None
    
    # Mock database methods
    db.store_experiment = store_experiment
    db.store_campaign = store_campaign
    db.get_experiments = get_experiments
    db.get_campaign = get_campaign
    db.experiments = experiments  # For direct access in tests
    db.campaigns = campaigns
    
    return db


@pytest.fixture
def mock_robot_orchestrator():
    """Create a mock robot orchestrator."""
    return create_mock_robot_orchestrator()


@pytest.fixture
def mock_instruments():
    """Create mock instruments for testing."""
    return {
        "uv_vis": MockInstrument("uv_vis"),
        "xrd": MockInstrument("xrd"),
        "pl_spectrometer": MockInstrument("photoluminescence"),
        "ftir": MockInstrument("ftir")
    }


# Environment fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "database": {
            "host": "localhost",
            "port": 27017,
            "name": "materials_test"
        },
        "robots": {
            "opentrons": {
                "ip": "127.0.0.1",
                "port": 31950,
                "simulation": True
            },
            "chemspeed": {
                "port": "/dev/null",
                "simulation": True
            }
        },
        "optimization": {
            "acquisition_function": "expected_improvement",
            "batch_size": 5,
            "convergence_threshold": 0.01
        },
        "safety": {
            "max_temperature": 300,
            "max_pressure": 5.0,
            "emergency_stop_enabled": True
        }
    }


# Enhanced fixtures for sample lab
@pytest.fixture
def enhanced_sample_lab():
    """Create an enhanced sample lab with full mocking."""
    with patch('materials_orchestrator.robots.RobotOrchestrator') as mock_orchestrator_class:
        mock_orchestrator = create_mock_robot_orchestrator()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        from materials_orchestrator.core import AutonomousLab
        from materials_orchestrator.planners import BayesianPlanner
        
        lab = AutonomousLab(
            robots=["opentrons", "chemspeed"],
            instruments=["uv_vis", "xrd"],
            planner=BayesianPlanner()
        )
        
        # Add mock experiment execution
        async def mock_run_experiment(parameters):
            from tests.fixtures.materials_data import generate_virtual_experiment_result
            await asyncio.sleep(0.01)  # Minimal delay for realism
            return generate_virtual_experiment_result(parameters)
        
        lab._run_single_experiment = mock_run_experiment
        
        yield lab


# Multi-objective fixture
@pytest.fixture
def multi_objective():
    """Create multiple objectives for multi-objective optimization tests."""
    from materials_orchestrator.core import MaterialsObjective
    return [
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


# Performance test configuration
@pytest.fixture(autouse=True)
def performance_test_config():
    """Configure performance test settings."""
    # Set shorter timeouts for tests
    os.environ["PYTEST_TIMEOUT"] = "30"
    os.environ["BENCHMARK_SKIP_SLOW"] = "true"


# Custom assertions
def assert_experiment_valid(experiment):
    """Assert that an experiment has valid structure."""
    assert "parameters" in experiment
    assert "results" in experiment
    assert "timestamp" in experiment
    assert isinstance(experiment["parameters"], dict)
    assert isinstance(experiment["results"], dict)


def assert_campaign_results_valid(results):
    """Assert that campaign results have valid structure."""
    required_fields = ["status", "total_experiments", "best_material", "convergence_history"]
    for field in required_fields:
        assert field in results, f"Missing required field: {field}"
    
    assert results["status"] in ["running", "completed", "failed", "stopped"]
    assert isinstance(results["total_experiments"], int)
    assert results["total_experiments"] >= 0


# Add custom assertions to pytest namespace
pytest.assert_experiment_valid = assert_experiment_valid
pytest.assert_campaign_results_valid = assert_campaign_results_valid