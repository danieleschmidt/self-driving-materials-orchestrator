# Development Setup Guide

## Quick Start

For immediate testing of the enhanced functionality:

```bash
# Clone repository
git clone https://github.com/danieleschmidt/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Test core functionality (no dependencies required)
python3 -c "
import sys
sys.path.insert(0, 'src')
from materials_orchestrator import AutonomousLab, MaterialsObjective, RandomPlanner
print('✅ Basic functionality works!')
"

# Run the example
python3 examples/perovskite_discovery_example.py
```

## Full Development Environment

### Prerequisites

- Python 3.9+ 
- Git
- (Optional) Docker for containerized development

### Installation Options

#### Option 1: Minimal Setup
```bash
# Works with just Python standard library
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Test basic functionality
python3 -c "from materials_orchestrator import AutonomousLab; print('✅ Installed')"
```

#### Option 2: Full Scientific Stack
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install scientific dependencies
pip install numpy scipy scikit-learn pandas matplotlib

# Install package with all features
pip install -e .[dev,robots,docs]

# Verify ML functionality
python3 -c "
from materials_orchestrator import BayesianPlanner
planner = BayesianPlanner(target_property='band_gap')
print('✅ Bayesian optimization available')
"
```

#### Option 3: Automated Setup
```bash
# Use provided setup script
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh

# Follow prompts for dependency installation
```

### Dependency Tiers

The implementation uses tiered dependencies for flexibility:

#### Tier 1: Core (Always Required)
- Python 3.9+ standard library
- Basic experiment simulation and planning
- Random and grid search planners

#### Tier 2: Scientific Computing (Recommended)
```bash
pip install numpy scipy pandas
```
- Enhanced numerical operations
- Better performance for large parameter spaces
- Improved data handling

#### Tier 3: Machine Learning (Optional)
```bash
pip install scikit-learn
```
- Gaussian Process Bayesian optimization
- Advanced acquisition functions
- Model-based experiment planning

#### Tier 4: Full Stack (Development)
```bash
pip install -e .[dev,robots,docs]
```
- Testing framework (pytest)
- Code quality tools (ruff, black, mypy)
- Documentation generation (mkdocs)
- Robot integration libraries

### Verifying Installation

#### Basic Functionality Test
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')

# Test imports
from materials_orchestrator import (
    AutonomousLab, MaterialsObjective, 
    RandomPlanner, BayesianPlanner
)
print('✅ All imports successful')

# Test basic workflow
objective = MaterialsObjective('band_gap', (1.2, 1.6))
lab = AutonomousLab()
experiment = lab.run_experiment({'temperature': 150})
print(f'✅ Experiment: {experiment.status}')
print('✅ Core functionality verified')
"
```

#### Advanced Features Test
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from materials_orchestrator import BayesianPlanner
    import numpy as np
    import sklearn
    print('✅ Advanced features available')
    print('   - Gaussian Process optimization')
    print('   - Scientific computing stack')
except ImportError as e:
    print('ℹ️  Advanced features not available:', e)
    print('   - Will use fallback implementations')
"
```

#### Full Example Test
```bash
# Run the complete example
python3 examples/perovskite_discovery_example.py

# Should output:
# ✅ Import successful
# ✅ Campaign completed successfully!
```

### Development Workflow

#### Running Tests
```bash
# If pytest is available
python3 -m pytest tests/ -v

# Basic test without pytest
python3 tests/test_enhanced_implementation.py

# Test specific functionality
python3 -c "
exec(open('tests/test_enhanced_implementation.py').read())
print('✅ Manual test execution completed')
"
```

#### Code Quality
```bash
# If ruff is available
ruff check src/ tests/

# If black is available  
black src/ tests/

# If mypy is available
mypy src/
```

#### Documentation
```bash
# Generate API documentation
python3 docs/gen_ref_pages.py

# Build documentation (if mkdocs available)
mkdocs serve
```

### IDE Setup

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.analysis.extraPaths": ["./src"],
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm
1. Open project directory
2. Configure Python interpreter to `./venv/bin/python`
3. Mark `src` directory as "Sources Root"
4. Enable code inspection tools

### Docker Development

#### Build Development Container
```bash
# Build development image
docker build -f Dockerfile -t materials-orchestrator:dev .

# Run interactive development
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  materials-orchestrator:dev bash

# Inside container
python3 examples/perovskite_discovery_example.py
```

#### Docker Compose Development
```bash
# Start development environment
docker-compose -f docker-compose.yml up -d

# Access development container
docker-compose exec app bash
```

### Troubleshooting

#### Common Issues

**Import Error: No module named 'materials_orchestrator'**
```bash
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use in scripts
import sys
sys.path.insert(0, 'src')
```

**Warning: numpy not available**
```bash
# This is expected and harmless
# The implementation will use fallback methods
# To enable full features:
pip install numpy scipy scikit-learn
```

**ModuleNotFoundError: sklearn**
```bash
# Bayesian optimization will fall back to random sampling
# To enable Gaussian Process optimization:
pip install scikit-learn
```

**Permission denied: scripts/setup-dev.sh**
```bash
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

#### Platform-Specific Issues

**Windows:**
```cmd
# Use Windows paths in examples
python examples\perovskite_discovery_example.py

# Virtual environment activation
venv\Scripts\activate
```

**macOS:**
```bash
# May need to install Xcode command line tools
xcode-select --install

# Use homebrew Python if system Python is restricted
brew install python@3.9
```

**Linux:**
```bash
# Install Python development headers if needed
sudo apt-get install python3-dev

# For scientific computing
sudo apt-get install build-essential
```

### Performance Optimization

#### For Large Parameter Spaces
```python
# Increase candidate generation for better exploration
planner = BayesianPlanner(
    acquisition_function="expected_improvement",
    target_property="band_gap"
)

# Use more efficient parameter space exploration
param_space = {
    "temperature": (120, 200),     # Narrow ranges
    "precursor_A_conc": (0.8, 1.2)  # Focus on promising regions
}
```

#### For Faster Development
```python
# Use fewer initial samples for testing
campaign = lab.run_campaign(
    objective=objective,
    param_space=param_space,
    initial_samples=5,        # Reduced from 20
    max_experiments=20,       # Quick tests
    convergence_patience=5    # Early stopping
)
```

### Contribution Workflow

1. **Setup development environment**
   ```bash
   git clone <fork-url>
   cd self-driving-materials-orchestrator
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   ```

2. **Make changes and test**
   ```bash
   # Edit code
   # Run tests
   python3 tests/test_enhanced_implementation.py
   python3 examples/perovskite_discovery_example.py
   ```

3. **Quality checks**
   ```bash
   # Format code (if available)
   black src/ tests/
   
   # Check style (if available)
   ruff check src/ tests/
   ```

4. **Submit changes**
   ```bash
   git add .
   git commit -m "feat: description of changes"
   git push origin feature-branch
   # Create pull request
   ```

This setup guide enables users to get started immediately with basic functionality while providing a path to unlock advanced features as needed.