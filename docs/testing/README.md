# Testing Documentation

## Overview

This document describes the comprehensive testing strategy for the Self-Driving Materials Orchestrator platform. Our testing infrastructure is designed to ensure reliability, performance, and correctness across all system components.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Main pytest configuration and fixtures
├── conftest_enhanced.py     # Enhanced fixtures for complex tests
├── unit/                    # Fast, isolated unit tests
│   ├── test_core.py
│   └── test_planners.py
├── integration/             # Integration tests between components
│   ├── test_database_integration.py
│   └── test_robot_integration.py
├── e2e/                     # End-to-end system tests
│   └── test_full_campaign.py
├── robot/                   # Hardware-specific robot tests
│   └── test_robot_connections.py
├── performance/             # Performance and benchmark tests
│   └── test_optimization_performance.py
└── fixtures/                # Test data and mock objects
    ├── materials_data.py
    └── mock_robots.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Single functions, classes, or small modules
- **Dependencies**: No external dependencies (databases, robots, etc.)
- **Run Command**: `pytest tests/unit/`

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between components
- **Speed**: Medium (1-10 seconds each)
- **Scope**: Multiple components working together
- **Dependencies**: May use mock databases or simulated robots
- **Run Command**: `pytest tests/integration/`

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Speed**: Slow (10+ seconds each)
- **Scope**: Full system functionality
- **Dependencies**: All system components (with mocking for hardware)
- **Run Command**: `pytest tests/e2e/`

### Robot Tests (`tests/robot/`)
- **Purpose**: Test robot hardware integration
- **Speed**: Variable (depends on hardware)
- **Scope**: Actual robot communication and control
- **Dependencies**: Physical robot hardware
- **Run Command**: `pytest tests/robot/ -m robot`

### Performance Tests (`tests/performance/`)
- **Purpose**: Benchmark system performance
- **Speed**: Variable (can be very slow)
- **Scope**: Algorithm efficiency, scalability, resource usage
- **Dependencies**: Large datasets, timing requirements
- **Run Command**: `pytest tests/performance/ -m benchmark`

## Test Markers

We use pytest markers to categorize and selectively run tests:

```python
@pytest.mark.unit           # Fast unit tests
@pytest.mark.integration    # Integration between components
@pytest.mark.e2e           # End-to-end system tests
@pytest.mark.robot         # Requires robot hardware
@pytest.mark.slow          # Long-running tests
@pytest.mark.benchmark     # Performance benchmarks
@pytest.mark.database      # Requires database
@pytest.mark.simulation    # Uses simulation mode
```

### Running Specific Test Categories

```bash
# Run only fast unit tests
pytest -m "unit"

# Run integration and unit tests (skip slow tests)
pytest -m "unit or integration"

# Skip robot hardware tests
pytest -m "not robot"

# Run only performance benchmarks
pytest -m "benchmark"

# Skip slow tests for quick feedback
pytest -m "not slow"
```

## Test Fixtures

### Core Fixtures (`conftest.py`)

#### `sample_objective`
Creates a basic materials optimization objective for testing.

```python
def test_optimization(sample_objective):
    assert sample_objective.target_property == "band_gap"
    assert sample_objective.target_range == (1.2, 1.6)
```

#### `sample_lab`
Creates a fully mocked autonomous lab instance.

```python
def test_lab_creation(sample_lab):
    assert len(sample_lab.robots) > 0
    assert len(sample_lab.instruments) > 0
```

#### `sample_param_space`
Provides a standard parameter space for optimization tests.

#### `sample_results`
Contains realistic experiment results for testing data processing.

### Enhanced Fixtures (`conftest_enhanced.py`)

#### `mock_database`
In-memory database mock for testing data persistence.

```python
def test_experiment_storage(mock_database):
    exp_id = mock_database.store_experiment({"parameters": {}, "results": {}})
    assert exp_id.startswith("exp_")
```

#### `mock_robot_orchestrator`
Complete robot orchestrator mock with multiple robot types.

#### `mock_instruments`
Collection of analytical instrument mocks.

### Test Data Fixtures (`fixtures/materials_data.py`)

#### Sample Materials Data
- `PEROVSKITE_SAMPLES`: Real perovskite compositions and properties
- `PARAMETER_SPACES`: Realistic parameter ranges for different material systems
- `EXPERIMENT_SEQUENCES`: Complete experiment campaign data

#### Virtual Experiment Generation
```python
from tests.fixtures.materials_data import generate_virtual_experiment_result

result = generate_virtual_experiment_result(
    parameters={"temperature": 150, "concentration": 1.0},
    material_system="perovskites",
    noise_level=0.05
)
```

### Robot Mocks (`fixtures/mock_robots.py`)

#### MockOpentronsDriver
Simulates Opentrons liquid handling robot with realistic timing and failure modes.

#### MockChemspeedDriver  
Simulates Chemspeed synthesis robot with temperature control and stirring.

#### MockInstrument
Generic analytical instrument mock supporting various measurement types.

## Test Configuration

### Pytest Configuration (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --cov=materials_orchestrator"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "robot: marks tests that require robot hardware",
]
```

### Environment Variables
```bash
# Skip slow tests in CI
export PYTEST_TIMEOUT=30
export BENCHMARK_SKIP_SLOW=true

# Enable verbose output for debugging
export PYTEST_VERBOSE=true

# Use test database
export MONGODB_DATABASE=materials_test
```

## Writing New Tests

### Unit Test Example
```python
import pytest
from materials_orchestrator.planners import BayesianPlanner

@pytest.mark.unit
def test_bayesian_planner_initialization():
    """Test BayesianPlanner initializes with correct defaults."""
    planner = BayesianPlanner()
    assert planner.acquisition_function == "expected_improvement"
    assert planner.batch_size == 1
    
def test_bayesian_planner_with_custom_params():
    """Test BayesianPlanner with custom parameters."""
    planner = BayesianPlanner(
        acquisition_function="upper_confidence_bound",
        batch_size=5
    )
    assert planner.acquisition_function == "upper_confidence_bound"
    assert planner.batch_size == 5
```

### Integration Test Example
```python
import pytest
from unittest.mock import patch

@pytest.mark.integration
@pytest.mark.asyncio
async def test_lab_robot_coordination(sample_lab, mock_database):
    """Test coordination between lab and robot systems."""
    with patch.object(sample_lab, 'database', mock_database):
        # Test robot orchestration with database logging
        result = await sample_lab.run_single_experiment({
            "temperature": 150,
            "concentration": 1.0
        })
        
        assert result["synthesis_success"] is True
        assert len(mock_database.experiments) == 1
```

### End-to-End Test Example
```python
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_complete_optimization_campaign(sample_objective, mock_database):
    """Test a complete optimization campaign from start to finish."""
    # This test runs a full campaign with mocked components
    # and verifies the entire workflow
    pass
```

## Performance Testing

### Benchmark Tests
We use `pytest-benchmark` for performance testing:

```python
@pytest.mark.benchmark
def test_optimization_speed(benchmark):
    """Benchmark optimization algorithm performance."""
    
    def run_optimization():
        # Setup optimization problem
        optimizer = MaterialsOptimizer()
        return optimizer.optimize(parameters)
    
    result = benchmark(run_optimization)
    assert result is not None
```

### Performance Metrics
- Optimization algorithm convergence time
- Database query performance
- Robot command execution speed
- Memory usage patterns
- Scalability with dataset size

## Continuous Integration

### GitHub Actions
Our CI pipeline runs different test suites:

```yaml
# Unit and integration tests (fast)
- name: Run fast tests
  run: pytest -m "not slow and not robot" --cov

# Performance benchmarks (scheduled)
- name: Run benchmarks
  run: pytest -m benchmark --benchmark-json=results.json

# Robot simulation tests
- name: Run robot simulation tests  
  run: pytest tests/robot/ -m "simulation and not robot"
```

### Coverage Requirements
- Minimum 90% code coverage for core modules
- 100% coverage for critical safety systems
- Performance regression detection
- Documentation coverage for public APIs

## Test Data Management

### Sample Data
All test data is version controlled and documented:
- Realistic material compositions
- Experimental parameters from literature  
- Measurement data with appropriate noise
- Edge cases and error conditions

### Data Generation
Tests use deterministic data generation for reproducibility:
```python
# Use fixed random seeds
np.random.seed(42)

# Generate consistent test data
test_data = generate_virtual_experiment_result(
    parameters={"temperature": 150},
    material_system="perovskites",
    noise_level=0.05,
    random_seed=42
)
```

## Debugging Tests

### Common Issues and Solutions

#### Async Test Failures
```python
# Wrong - missing async/await
def test_async_function():
    result = async_function()  # This returns a coroutine, not the result

# Correct - use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
```

#### Mock Configuration
```python
# Wrong - mock not properly configured
with patch('module.function'):
    result = function()  # Returns MagicMock, not expected value

# Correct - configure mock return value
with patch('module.function', return_value="expected") as mock:
    result = function()
    assert result == "expected"
```

#### Fixture Scope Issues
```python
# Be careful with fixture scope
@pytest.fixture(scope="session")  # Shared across all tests
@pytest.fixture(scope="function")  # New instance per test (default)
```

### Debugging Commands
```bash
# Run single test with verbose output
pytest tests/unit/test_core.py::test_specific_function -v -s

# Debug test failures
pytest --pdb  # Drop into debugger on failure

# Show test output
pytest -s    # Don't capture stdout

# Run tests matching pattern
pytest -k "optimization" -v
```

## Test Maintenance

### Regular Tasks
- Update test data with new experimental results
- Review and update performance benchmarks
- Maintain mock objects to match real implementations
- Update integration tests when APIs change
- Monitor test execution time and optimize slow tests

### Best Practices
- Keep tests independent and idempotent
- Use descriptive test names that explain the scenario
- Test both success and failure paths
- Include edge cases and boundary conditions
- Mock external dependencies consistently
- Use appropriate assertions with clear error messages
- Document complex test scenarios
- Regular review and refactoring of test code