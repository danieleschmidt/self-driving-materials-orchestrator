# self-driving-materials-orchestrator

Autonomous lab framework for accelerated materials discovery using Bayesian optimization and active learning.

## Components

- **ExperimentDesigner** – Latin Hypercube Sampling for parameter space exploration
- **MaterialsSimulator** – Polynomial model for synthetic material property evaluation
- **BayesianOptimizer** – GP surrogate + Expected Improvement acquisition (numpy-based)
- **ExperimentLoop** – Closed-loop: design → simulate → update → repeat
- **ResultsTracker** – JSON experiment log + Pareto front tracking

## Usage

```python
from sdmo import ExperimentDesigner, MaterialsSimulator, BayesianOptimizer, ResultsTracker, ExperimentLoop

params = {"composition_x": (0.0, 1.0), "temperature": (300.0, 1000.0), "pressure": (0.1, 10.0)}

loop = ExperimentLoop(
    designer=ExperimentDesigner(params),
    simulator=MaterialsSimulator(),
    optimizer=BayesianOptimizer(params, objective="conductivity"),
    tracker=ResultsTracker("results.json"),
    n_initial=5,
)

results = loop.run(n_iterations=20)
best_params, best_val = loop.best()
print(f"Best conductivity: {best_val:.2f} at {best_params}")
```

## Development

```bash
pip install numpy
pytest tests/
```
