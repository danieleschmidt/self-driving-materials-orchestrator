# Testing Guide

This guide covers the comprehensive testing framework for the Self-Driving Materials Orchestrator.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── utils.py                 # Testing utilities and mock objects
├── pytest.ini              # Pytest configuration
├── fixtures/                # Test data and sample generators
│   ├── __init__.py
│   └── sample_data.py       # Sample experiment and campaign data
├── unit/                    # Unit tests for individual components
│   ├── test_core.py         # Core orchestration logic tests
│   └── test_planners.py     # Optimization algorithm tests
├── integration/             # Integration tests between components
│   └── test_database_integration.py
├── robot/                   # Hardware-specific tests
│   └── test_robot_connections.py
├── e2e/                     # End-to-end workflow tests
│   └── test_autonomous_campaign.py
└── performance/             # Performance and benchmark tests
    └── test_optimization_benchmarks.py
```

## Test Categories

### Unit Tests
Test individual components in isolation with mocked dependencies.

```bash
# Run all unit tests
pytest tests/unit/

# Run specific unit test file
pytest tests/unit/test_core.py -v

# Run with coverage
pytest tests/unit/ --cov=materials_orchestrator --cov-report=html
```

### Integration Tests
Test interactions between components with real or semi-real dependencies.

```bash
# Run integration tests (requires database)
pytest tests/integration/ -m database

# Skip integration tests that require external services
pytest tests/integration/ -m "not database"
```

### Robot Tests
Test hardware interactions (typically skipped in CI/CD).

```bash
# Run robot tests (requires hardware)
pytest tests/robot/ -m robot

# Skip robot tests
pytest tests/ -m "not robot"
```

### End-to-End Tests
Test complete workflows from start to finish.

```bash
# Run e2e tests
pytest tests/e2e/

# Run with increased verbosity for debugging
pytest tests/e2e/ -v -s
```

### Performance Tests
Benchmark performance and scalability.

```bash
# Run performance benchmarks
pytest tests/performance/ -m benchmark

# Generate benchmark report
pytest tests/performance/ --benchmark-json=benchmark.json
```

## Test Markers

Use pytest markers to categorize and selectively run tests:

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.robot` - Requires robot hardware
- `@pytest.mark.instrument` - Requires analytical instruments
- `@pytest.mark.database` - Requires database connection
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.simulation` - Uses simulation mode

### Example Usage

```python
import pytest

@pytest.mark.integration
@pytest.mark.database
def test_experiment_storage():
    """Test storing experiments in database."""
    pass

@pytest.mark.robot
@pytest.mark.slow
def test_robot_calibration():
    """Test robot calibration procedure."""
    pass
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=materials_orchestrator

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "integration and not slow"  # Fast integration tests
pytest -m "not robot"             # Skip hardware tests
```

### Development Workflow

```bash
# Quick test during development
pytest tests/unit/test_core.py::TestAutonomousLab::test_initialization -v

# Test with coverage and open report
pytest --cov=materials_orchestrator --cov-report=html && open htmlcov/index.html

# Run tests on file change (requires pytest-watch)
ptw
```

### Continuous Integration

```bash
# CI test command (in GitHub Actions)
pytest tests/ -m "not robot and not slow" --cov=materials_orchestrator --cov-report=xml
```

## Test Fixtures

### Available Fixtures

```python
# From conftest.py
def test_with_fixtures(sample_objective, sample_lab, sample_param_space):
    """Example test using shared fixtures."""
    campaign = sample_lab.run_campaign(
        objective=sample_objective,
        param_space=sample_param_space,
        max_experiments=5
    )
    assert campaign is not None

# From utils.py
def test_with_mocks(mock_robot, mock_instrument, mock_database):
    """Example test using mock objects."""
    assert mock_robot.connect() is True
    result = mock_instrument.measure("sample_001", "band_gap")
    assert "value" in result
```

### Custom Fixtures

```python
@pytest.fixture
def custom_campaign_config():
    """Create custom campaign configuration."""
    return {
        "max_experiments": 10,
        "convergence_patience": 5,
        "initial_samples": 3
    }

def test_custom_campaign(sample_lab, custom_campaign_config):
    """Test with custom configuration."""
    # Use custom_campaign_config in test
    pass
```

## Mock Objects

The testing framework provides comprehensive mock objects for hardware-free testing:

### MockRobot
```python
from tests.utils import MockRobot

robot = MockRobot("test_robot_001", capabilities=["synthesis", "characterization"])
robot.connect()
result = robot.execute_protocol({"action": "synthesis", "parameters": {...}})
```

### MockInstrument
```python
from tests.utils import MockInstrument

instrument = MockInstrument("test_spec_001", measurements=["band_gap", "efficiency"])
result = instrument.measure("sample_001", "band_gap")
```

### MockDatabase
```python
from tests.utils import MockDatabase

db = MockDatabase()
experiment_id = db.store_experiment({...})
experiments = db.get_experiments("campaign_001")
```

## Test Data Generation

Use the sample data generator for realistic test scenarios:

```python
from tests.fixtures.sample_data import SampleDataGenerator

# Generate experiment data
experiments = SampleDataGenerator.generate_experiment_data(
    n_experiments=100,
    noise_level=0.05,
    include_failures=True
)

# Generate campaign data
campaign = SampleDataGenerator.generate_campaign_data()

# Get standard parameter spaces
param_space = SampleDataGenerator.perovskite_parameters()
```

## Environment-Specific Testing

### Local Development
```bash
# Full test suite with hardware simulation
pytest tests/ --env=development

# Skip slow tests
pytest tests/ -m "not slow"
```

### CI/CD Environment
```bash
# Fast, hardware-free tests
pytest tests/ -m "not robot and not slow and not integration"

# With coverage for code quality
pytest tests/ --cov=materials_orchestrator --cov-fail-under=80
```

### Production Validation
```bash
# Integration tests against real services
pytest tests/integration/ --env=production

# Performance regression tests
pytest tests/performance/ --benchmark-json=benchmarks.json
```

## Coverage Requirements

- **Overall Coverage**: Minimum 80%
- **Unit Tests**: Minimum 90% 
- **Integration Tests**: Minimum 70%
- **Critical Paths**: 100% coverage required

### Coverage Exclusions
- Test files themselves
- Migration scripts
- Configuration files
- `__init__.py` files (unless they contain logic)
- Platform-specific code blocks

## Testing Best Practices

### 1. Test Organization
- One test class per module/component
- Descriptive test names that explain the scenario
- Group related tests using classes
- Use fixtures to reduce code duplication

### 2. Test Independence
- Each test should be independent and isolated
- Use fresh fixtures for each test
- Clean up resources after tests
- Don't rely on test execution order

### 3. Assertions
- Use specific assertions (`assert x == 5` not `assert x`)
- Include meaningful error messages
- Test both positive and negative cases
- Verify expected exceptions are raised

### 4. Performance Testing
- Set reasonable time limits for operations
- Test with realistic data volumes
- Monitor memory usage in long-running tests
- Use `@pytest.mark.benchmark` for performance tests

### 5. Hardware Simulation
- Default to simulation mode for most tests
- Use hardware markers for tests requiring real devices
- Provide fallback behavior when hardware unavailable
- Test error handling for hardware failures

## Debugging Tests

### Common Debugging Techniques

```bash
# Run single test with output
pytest tests/unit/test_core.py::test_specific_function -v -s

# Drop into debugger on failure
pytest tests/unit/test_core.py --pdb

# Show local variables on failure
pytest tests/unit/test_core.py --tb=long

# Increase logging level
pytest tests/unit/test_core.py --log-cli-level=DEBUG
```

### Test Debugging Tools

```python
import pytest

def test_with_debugging():
    """Example test with debugging aids."""
    # Use pytest.set_trace() for breakpoints
    pytest.set_trace()
    
    # Capture and examine logs
    with pytest.LoggingCapture() as log_capture:
        # Test code here
        pass
    
    # Check log messages
    assert "Expected message" in log_capture.messages
```

## Writing New Tests

### Template for Unit Tests
```python
import pytest
from unittest.mock import Mock, patch
from materials_orchestrator.core import YourClass

class TestYourClass:
    """Tests for YourClass functionality."""
    
    @pytest.fixture
    def test_instance(self):
        """Create test instance with mock dependencies."""
        return YourClass(config={"test": True})
    
    def test_basic_functionality(self, test_instance):
        """Test basic functionality works as expected."""
        result = test_instance.method_under_test()
        assert result is not None
        
    def test_error_handling(self, test_instance):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="Expected error message"):
            test_instance.method_with_validation("invalid_input")
            
    @pytest.mark.parametrize("input_value,expected", [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ])
    def test_parametrized_behavior(self, test_instance, input_value, expected):
        """Test behavior with different inputs."""
        result = test_instance.convert_number(input_value)
        assert result == expected
```

### Template for Integration Tests
```python
import pytest
from materials_orchestrator import AutonomousLab

@pytest.mark.integration
class TestLabIntegration:
    """Integration tests for lab components."""
    
    @pytest.fixture
    def integrated_lab(self):
        """Set up lab with real components."""
        return AutonomousLab.from_config("test_config.json")
    
    def test_end_to_end_workflow(self, integrated_lab):
        """Test complete workflow integration."""
        # Test implementation
        pass
```

## Continuous Testing

### Pre-commit Hooks
Tests run automatically before commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### GitHub Actions
Automated testing on:
- Pull requests
- Pushes to main branch
- Scheduled nightly runs

### Local Automation
```bash
# Watch for changes and run tests
ptw tests/ --runner "pytest tests/ -x"

# Run tests on specific file changes
watchmedo shell-command \
    --patterns="*.py" \
    --recursive \
    --command="pytest tests/unit/test_core.py" \
    src/
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="${PYTHONPATH}:./src"
   
   # Or install in development mode
   pip install -e .
   ```

2. **Database Connection Failures**
   ```bash
   # Start test database
   docker run -d -p 27017:27017 mongo:5.0
   
   # Or skip database tests
   pytest -m "not database"
   ```

3. **Hardware Test Failures**
   ```bash
   # Skip hardware tests in development
   pytest -m "not robot and not instrument"
   
   # Use simulation mode
   pytest --simulation-mode
   ```

4. **Slow Test Performance**
   ```bash
   # Profile test execution
   pytest --durations=10
   
   # Run only fast tests
   pytest -m "not slow"
   ```

For additional support, see the [Development Setup Guide](../DEVELOPMENT_SETUP.md) or open an issue on GitHub.