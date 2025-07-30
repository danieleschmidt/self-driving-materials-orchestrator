# Development Guide

This guide covers the development workflow, architecture, and best practices for the Self-Driving Materials Orchestrator.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest

# Start development server
python -m materials_orchestrator.cli launch --debug
```

## Architecture Overview

### Core Components

```
src/materials_orchestrator/
├── core.py              # Main AutonomousLab and MaterialsObjective classes
├── planners.py          # Experiment planning algorithms (Bayesian, Random, Grid)
├── robots/              # Robot drivers and orchestration
├── optimization/        # Advanced optimization algorithms  
├── database/           # MongoDB integration and data management
├── analysis/           # Data analysis and ML models
├── dashboard/          # Streamlit web interface
└── cli.py              # Command-line interface
```

### Key Design Principles

1. **Modularity**: Each component is independently testable and replaceable
2. **Extensibility**: Easy to add new robots, algorithms, and materials models
3. **Safety**: Multiple layers of validation and emergency stops
4. **Reproducibility**: Complete experiment provenance and version tracking

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/bayesian-optimization

# Make changes with tests
# Commit with conventional format
git commit -m "feat(optimization): add multi-objective Bayesian optimization"

# Push and create PR
git push origin feature/bayesian-optimization
```

### 2. Testing Strategy

```bash
# Unit tests (fast, isolated)
pytest tests/unit/

# Integration tests (with external systems)  
pytest tests/integration/

# Robot tests (requires hardware)
pytest tests/robot/ -m robot

# All tests with coverage
pytest --cov=materials_orchestrator --cov-report=html
```

### 3. Code Quality

```bash
# Format code
black .

# Lint and fix issues
ruff check . --fix

# Type checking
mypy src/

# Security scan
bandit -r src/

# Run all checks
pre-commit run --all-files
```

## Adding New Components

### Robot Drivers

Create new robot driver by inheriting from `RobotDriver`:

```python
# src/materials_orchestrator/robots/my_robot.py
from .base import RobotDriver

class MyRobotDriver(RobotDriver):
    def connect(self) -> bool:
        # Establish connection
        return True
    
    def execute_action(self, action: str, parameters: dict) -> bool:
        # Execute robot action
        return True
    
    def get_status(self) -> dict:
        # Return current status
        return {"status": "ready"}
```

Register in `robots/__init__.py`:
```python
from .my_robot import MyRobotDriver

AVAILABLE_ROBOTS = {
    "my_robot": MyRobotDriver,
}
```

### Optimization Algorithms

Create new planner by inheriting from `BasePlanner`:

```python
# src/materials_orchestrator/planners.py
class MyCustomPlanner(BasePlanner):
    def suggest_next(self, n_suggestions, param_space, previous_results):
        # Your algorithm implementation
        return suggestions
```

### Material Models

Add domain-specific models:

```python
# src/materials_orchestrator/models/perovskites.py
from .base import MaterialModel

class PerovskiteModel(MaterialModel):
    def predict_properties(self, composition: dict) -> dict:
        # Predict material properties
        return {"band_gap": 1.5, "stability": 0.8}
```

## Database Schema

### Experiments Collection

```javascript
{
  "_id": ObjectId("..."),
  "campaign_id": "perovskite_optimization_001",
  "timestamp": ISODate("2025-01-15T10:30:00Z"),
  "parameters": {
    "temperature": 150,
    "precursor_A_conc": 1.0,
    "precursor_B_conc": 0.5,
    "reaction_time": 3.5
  },
  "results": {
    "band_gap": 1.55,
    "efficiency": 22.3,
    "stability": 0.89
  },
  "metadata": {
    "operator": "autonomous",
    "robot_id": "synthesis_bot_1",
    "software_version": "0.1.0",
    "experiment_type": "synthesis"
  },
  "status": "completed",
  "error_log": []
}
```

### Materials Collection

```javascript
{
  "_id": ObjectId("..."),
  "composition": {
    "Pb": 0.5,
    "Sn": 0.5,
    "I": 3.0
  },
  "properties": {
    "band_gap": 1.42,
    "formation_energy": -2.1,
    "stability": 0.92
  },
  "synthesis_conditions": {
    "temperature": 150,
    "solvent": "DMF",
    "atmosphere": "N2"
  },
  "characterization": {
    "xrd_pattern": "path/to/xrd.json",
    "uv_vis_spectrum": "path/to/spectrum.json"
  },
  "created_at": ISODate("2025-01-15T10:30:00Z")
}
```

## Configuration

### Environment Variables

```bash
# .env file
MONGODB_URL=mongodb://localhost:27017/materials_discovery
ROBOT_CONFIG_PATH=/path/to/robot/configs
LOG_LEVEL=INFO
STREAMLIT_PORT=8501
API_PORT=8000
```

### Robot Configuration

```yaml
# configs/robots.yaml
robots:
  synthesis_robot:
    driver: opentrons
    ip: "192.168.1.100"
    deck_config: "configs/ot2_synthesis_deck.json"
    
  characterization_robot:
    driver: custom
    port: "/dev/ttyUSB0"
    baud_rate: 9600
```

## Debugging

### Logging Configuration

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Logger for specific component
logger = logging.getLogger('materials_orchestrator.robots')
```

### Common Issues

1. **Robot Connection Failures**
   - Check IP addresses and network connectivity
   - Verify robot drivers are installed
   - Check firewall settings

2. **Database Connection Issues**
   - Ensure MongoDB is running
   - Check connection string format
   - Verify authentication credentials

3. **Optimization Not Converging**
   - Increase exploration factor
   - Check parameter space bounds
   - Verify acquisition function selection

## Performance Optimization

### Profiling

```bash
# Profile code execution
python -m cProfile -o profile.stats -m materials_orchestrator.cli campaign band_gap

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### Memory Usage

```bash
# Monitor memory usage
pip install memory-profiler
python -m memory_profiler materials_orchestrator/core.py
```

### Database Optimization

```javascript
// Create indexes for common queries
db.experiments.createIndex({"campaign_id": 1, "timestamp": -1})
db.experiments.createIndex({"results.band_gap": 1})
db.materials.createIndex({"composition": 1})
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t materials-orchestrator .

# Run with compose
docker-compose up -d
```

### Production Checklist

- [ ] Update configuration for production environment
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Enable SSL/TLS encryption
- [ ] Set up log aggregation
- [ ] Test emergency stop procedures
- [ ] Validate robot safety systems

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## Additional Resources

- [API Documentation](api.md)
- [Robot Integration Guide](robot_integration.md)
- [Optimization Algorithms](optimization.md)
- [Security Best Practices](security.md)