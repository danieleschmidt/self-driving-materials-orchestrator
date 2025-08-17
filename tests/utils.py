"""Testing utilities and helper functions."""

import asyncio
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock
import pytest


class MockRobot:
    """Mock robot for testing without hardware."""

    def __init__(self, robot_id: str = "test_robot", capabilities: list = None):
        self.robot_id = robot_id
        self.capabilities = capabilities or ["pipetting", "dispensing"]
        self.status = "available"
        self.connected = True
        self.protocols_executed = []
        self.current_protocol = None

    def connect(self) -> bool:
        """Mock connection."""
        self.connected = True
        return True

    def disconnect(self) -> bool:
        """Mock disconnection."""
        self.connected = False
        return True

    def execute_protocol(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Mock protocol execution."""
        self.current_protocol = protocol
        self.protocols_executed.append(protocol)

        # Simulate execution time
        import time

        time.sleep(0.1)  # Very short for testing

        return {"status": "completed", "duration": 0.1, "results": {"success": True}}

    def get_status(self) -> Dict[str, Any]:
        """Get mock robot status."""
        return {
            "robot_id": self.robot_id,
            "status": self.status,
            "connected": self.connected,
            "current_protocol": self.current_protocol,
            "capabilities": self.capabilities,
        }


class MockInstrument:
    """Mock analytical instrument for testing."""

    def __init__(
        self, instrument_id: str = "test_instrument", measurements: list = None
    ):
        self.instrument_id = instrument_id
        self.measurements = measurements or ["band_gap", "efficiency"]
        self.status = "available"
        self.connected = True
        self.measurement_history = []

    def connect(self) -> bool:
        """Mock connection."""
        self.connected = True
        return True

    def measure(self, sample_id: str, measurement_type: str) -> Dict[str, Any]:
        """Mock measurement."""
        import random

        # Generate realistic mock data based on measurement type
        if measurement_type == "band_gap":
            value = random.uniform(1.0, 2.0)
        elif measurement_type == "efficiency":
            value = random.uniform(0, 30)
        elif measurement_type == "stability":
            value = random.uniform(0, 1)
        else:
            value = random.uniform(0, 100)

        result = {
            "sample_id": sample_id,
            "measurement_type": measurement_type,
            "value": round(value, 3),
            "timestamp": "2025-01-01T12:00:00Z",
            "instrument_id": self.instrument_id,
        }

        self.measurement_history.append(result)
        return result


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.experiments = []
        self.campaigns = []
        self.models = []
        self.connected = False

    def connect(self) -> bool:
        """Mock database connection."""
        self.connected = True
        return True

    def store_experiment(self, experiment: Dict[str, Any]) -> str:
        """Store mock experiment."""
        experiment_id = f"exp_{len(self.experiments):04d}"
        experiment["experiment_id"] = experiment_id
        self.experiments.append(experiment)
        return experiment_id

    def get_experiments(self, campaign_id: str = None) -> list:
        """Get mock experiments."""
        if campaign_id:
            return [
                exp for exp in self.experiments if exp.get("campaign_id") == campaign_id
            ]
        return self.experiments

    def store_campaign(self, campaign: Dict[str, Any]) -> str:
        """Store mock campaign."""
        campaign_id = f"campaign_{len(self.campaigns):04d}"
        campaign["campaign_id"] = campaign_id
        self.campaigns.append(campaign)
        return campaign_id


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


@contextmanager
def mock_environment_variables(env_vars: Dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set environment variables for testing."""
    import os

    original_env = {}

    # Store original values
    for key in env_vars:
        original_env[key] = os.environ.get(key)

    # Set test values
    for key, value in env_vars.items():
        os.environ[key] = value

    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "database": {"url": "mongodb://localhost:27017/", "database": "materials_test"},
        "robots": [
            {
                "id": "test_robot_1",
                "type": "liquid_handler",
                "connection": {"type": "mock"},
            }
        ],
        "instruments": [
            {
                "id": "test_instrument_1",
                "type": "spectroscopy",
                "connection": {"type": "mock"},
            }
        ],
        "optimization": {
            "algorithm": "bayesian",
            "acquisition_function": "expected_improvement",
        },
    }


def assert_experiment_valid(experiment: Dict[str, Any]) -> None:
    """Assert that an experiment dictionary has required fields."""
    required_fields = ["experiment_id", "parameters", "results", "metadata"]
    for field in required_fields:
        assert field in experiment, f"Missing required field: {field}"

    assert isinstance(experiment["parameters"], dict), "Parameters must be a dictionary"
    assert isinstance(experiment["metadata"], dict), "Metadata must be a dictionary"


def assert_campaign_valid(campaign: Dict[str, Any]) -> None:
    """Assert that a campaign dictionary has required fields."""
    required_fields = ["campaign_id", "objective", "parameter_space", "status"]
    for field in required_fields:
        assert field in campaign, f"Missing required field: {field}"

    assert isinstance(campaign["objective"], dict), "Objective must be a dictionary"
    assert isinstance(
        campaign["parameter_space"], dict
    ), "Parameter space must be a dictionary"


class AsyncTestCase:
    """Base class for async testing."""

    @staticmethod
    def run_async(coro):
        """Run an async coroutine in tests."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def skip_if_no_hardware():
    """Skip test if hardware is not available."""
    return pytest.mark.skipif(
        True,  # Always skip hardware tests in CI
        reason="Hardware not available in test environment",
    )


def skip_if_no_mongodb():
    """Skip test if MongoDB is not available."""
    import pymongo

    try:
        client = pymongo.MongoClient(
            "mongodb://localhost:27017/", serverSelectionTimeoutMS=1000
        )
        client.server_info()
        return pytest.mark.skipif(False, reason="")
    except:
        return pytest.mark.skipif(True, reason="MongoDB not available")


def parametrize_optimization_algorithms():
    """Parametrize tests with different optimization algorithms."""
    return pytest.mark.parametrize(
        "algorithm,params",
        [
            ("random", {"seed": 42}),
            ("grid", {"resolution": 10}),
            ("bayesian", {"acquisition_function": "expected_improvement"}),
        ],
    )


# Fixtures for common test scenarios
@pytest.fixture
def mock_robot():
    """Provide a mock robot for testing."""
    return MockRobot()


@pytest.fixture
def mock_instrument():
    """Provide a mock instrument for testing."""
    return MockInstrument()


@pytest.fixture
def mock_database():
    """Provide a mock database for testing."""
    return MockDatabase()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return create_test_config()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    with temporary_directory() as temp_dir:
        yield temp_dir
