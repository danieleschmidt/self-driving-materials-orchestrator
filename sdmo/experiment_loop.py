"""ExperimentLoop: design → simulate → update → repeat."""

from typing import Callable, Dict, List, Optional

from .experiment_designer import ExperimentDesigner
from .simulator import MaterialsSimulator
from .bayesian_optimizer import BayesianOptimizer
from .results_tracker import ResultsTracker


class ExperimentLoop:
    """Autonomous experiment loop for materials discovery."""

    def __init__(
        self,
        designer: ExperimentDesigner,
        simulator: MaterialsSimulator,
        optimizer: BayesianOptimizer,
        tracker: ResultsTracker,
        n_initial: int = 5,
    ):
        self.designer = designer
        self.simulator = simulator
        self.optimizer = optimizer
        self.tracker = tracker
        self.n_initial = n_initial
        self.iteration = 0

    def initialize(self):
        """Run initial random/LHS experiments to seed the optimizer."""
        initial_params = self.designer.latin_hypercube_sample(self.n_initial)
        for params in initial_params:
            results = self.simulator.simulate(params)
            self.optimizer.update(params, results)
            self.tracker.log(params, results, iteration=self.iteration)
        self.iteration += 1
        return initial_params

    def step(self) -> Dict:
        """Run one iteration: suggest → simulate → update → log."""
        params = self.optimizer.suggest()
        results = self.simulator.simulate(params)
        self.optimizer.update(params, results)
        entry = self.tracker.log(params, results, iteration=self.iteration)
        self.iteration += 1
        return entry

    def run(self, n_iterations: int, callback: Optional[Callable[[int, Dict], None]] = None):
        """Run the full loop for n_iterations after initialization."""
        self.initialize()
        for i in range(n_iterations):
            entry = self.step()
            if callback:
                callback(i, entry)
        return self.tracker.get_all()

    def best(self):
        """Return the best experiment found so far."""
        return self.optimizer.best_observed()
