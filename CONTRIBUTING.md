# Contributing to Self-Driving Materials Orchestrator

Thank you for your interest in contributing to the Self-Driving Materials Orchestrator! This document provides guidelines and information to help you contribute effectively.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/self-driving-materials-orchestrator.git`
3. **Install** development dependencies: `pip install -e ".[dev]"`
4. **Set up** pre-commit hooks: `pre-commit install`
5. **Create** a feature branch: `git checkout -b feature/your-feature`
6. **Make** your changes and commit with descriptive messages
7. **Push** to your fork and **submit** a pull request

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- MongoDB (for database-dependent features)
- Docker (optional, for containerized development)

### Installation

```bash
# Clone the repository
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,robots,docs]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=materials_orchestrator

# Run specific test categories
pytest -m "not slow"  # Exclude slow tests
pytest -m integration  # Only integration tests
pytest -m robot  # Only robot hardware tests (requires equipment)
```

## Contributing Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Bandit**: Security analysis

Run all checks:
```bash
# Format code
black .

# Check linting
ruff check .

# Type checking
mypy src/

# Security analysis
bandit -r src/
```

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Example:
```
feat(optimization): add multi-objective Bayesian optimization

Implement Pareto-optimal experiment selection using NSGA-II
algorithm for simultaneous optimization of efficiency and stability.

Closes #123
```

### Pull Request Process

1. **Create Issue First**: For significant changes, create an issue to discuss the proposed changes
2. **Branch Naming**: Use descriptive names like `feature/bayesian-optimization` or `fix/robot-connection-timeout`
3. **Small PRs**: Keep pull requests focused and reasonably sized
4. **Tests**: Add tests for new functionality
5. **Documentation**: Update documentation for user-facing changes
6. **Review**: Respond to review feedback promptly

### Testing Guidelines

#### Test Structure

```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Integration tests with external systems
├── robot/            # Hardware-dependent tests
└── fixtures/         # Shared test data
```

#### Writing Tests

```python
import pytest
from materials_orchestrator import AutonomousLab

class TestAutonomousLab:
    def test_initialization(self):
        """Test that lab initializes with default settings."""
        lab = AutonomousLab()
        assert lab.status == "initialized"
    
    @pytest.mark.slow
    def test_full_campaign(self):
        """Test complete optimization campaign."""
        # Long-running test marked as slow
        pass
    
    @pytest.mark.robot
    def test_robot_connection(self):
        """Test robot connection (requires hardware)."""
        # Test requiring physical robot
        pass
```

## Areas for Contribution

### High Priority

- **Robot Drivers**: Support for additional laboratory equipment
- **Optimization Algorithms**: Advanced Bayesian optimization strategies
- **Material Models**: Domain-specific predictive models
- **Error Handling**: Robust error recovery and logging

### Medium Priority

- **Dashboard Features**: Enhanced visualization and monitoring
- **Data Pipeline**: Improved data processing and analysis
- **Documentation**: Tutorials and examples
- **Performance**: Optimization and profiling

### Getting Started Tasks

Look for issues labeled with:
- `good first issue`: Beginner-friendly tasks
- `help wanted`: Community input needed
- `documentation`: Documentation improvements

## Specialized Contributions

### Robot Integration

Contributing robot drivers requires:

1. **Hardware Access**: Physical or simulated robot system
2. **Driver Implementation**: Following the `RobotDriver` interface:

```python
from materials_orchestrator.robots.base import RobotDriver

class MyRobotDriver(RobotDriver):
    def connect(self) -> bool:
        """Establish connection to robot."""
        pass
    
    def execute_action(self, action: str, parameters: dict) -> bool:
        """Execute robot action with parameters."""
        pass
    
    def get_status(self) -> dict:
        """Return current robot status."""
        pass
```

3. **Testing**: Both simulated and hardware tests
4. **Documentation**: Setup and usage instructions

### Algorithm Development

Contributing optimization algorithms:

1. **Base Class**: Inherit from `BaseOptimizer`
2. **Research**: Cite relevant papers and benchmarks
3. **Validation**: Compare against existing methods
4. **Documentation**: Mathematical description and usage

### Bug Reports

When reporting bugs, include:

- **Environment**: OS, Python version, package versions
- **Reproduction**: Minimal code to reproduce the issue
- **Expected vs Actual**: What you expected vs what happened
- **Logs**: Relevant error messages and stack traces

### Feature Requests

For feature requests:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other solutions you've considered
4. **Impact**: Who would benefit from this feature

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: conduct@terragonlabs.com for code of conduct issues

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read and follow it in all interactions.

### Recognition

Contributors are recognized in:
- Release notes for significant contributions
- `CONTRIBUTORS.md` file (automatically updated)
- Project documentation and papers (with permission)

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask questions! Create an issue or start a discussion if you need help getting started or understanding any part of the codebase.