"""Tests for Self-Driving Materials Orchestrator."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from sdmo.experiment_designer import ExperimentDesigner
from sdmo.simulator import MaterialsSimulator
from sdmo.bayesian_optimizer import BayesianOptimizer, GaussianProcessSurrogate
from sdmo.results_tracker import ResultsTracker
from sdmo.experiment_loop import ExperimentLoop


PARAMS = {
    "composition_x": (0.0, 1.0),
    "temperature": (300.0, 1000.0),
    "pressure": (0.1, 10.0),
}


# ── ExperimentDesigner ─────────────────────────────────────────────────────────

class TestExperimentDesigner:
    def test_lhs_count(self):
        d = ExperimentDesigner(PARAMS)
        samples = d.latin_hypercube_sample(10)
        assert len(samples) == 10

    def test_lhs_all_params_present(self):
        d = ExperimentDesigner(PARAMS)
        samples = d.latin_hypercube_sample(5)
        for s in samples:
            assert set(s.keys()) == set(PARAMS.keys())

    def test_lhs_within_bounds(self):
        d = ExperimentDesigner(PARAMS)
        for sample in d.latin_hypercube_sample(20):
            for key, (lo, hi) in PARAMS.items():
                assert lo <= sample[key] <= hi

    def test_grid_sample(self):
        d = ExperimentDesigner({"x": (0, 1), "y": (0, 1)})
        samples = d.grid_sample(3)
        assert len(samples) == 9  # 3x3

    def test_suggest_next(self):
        d = ExperimentDesigner(PARAMS)
        existing = d.latin_hypercube_sample(5)
        next_pts = d.suggest_next(existing, n=2)
        assert len(next_pts) == 2


# ── MaterialsSimulator ─────────────────────────────────────────────────────────

class TestMaterialsSimulator:
    def test_simulate_returns_expected_keys(self):
        sim = MaterialsSimulator()
        result = sim.simulate({"composition_x": 0.5, "temperature": 600.0, "pressure": 5.0})
        assert "conductivity" in result
        assert "strength" in result
        assert "stability" in result

    def test_simulate_within_bounds(self):
        sim = MaterialsSimulator()
        for _ in range(10):
            params = {"composition_x": 0.5, "temperature": 500.0, "pressure": 3.0}
            r = sim.simulate(params)
            assert 0 <= r["conductivity"] <= 100
            assert 0 <= r["strength"] <= 200
            assert 0 <= r["stability"] <= 1

    def test_batch_simulate(self):
        sim = MaterialsSimulator()
        params = [{"composition_x": 0.3, "temperature": 400.0, "pressure": 2.0}] * 5
        results = sim.batch_simulate(params)
        assert len(results) == 5
        assert all("conductivity" in r for r in results)

    def test_deterministic_with_noise_zero(self):
        sim = MaterialsSimulator(noise_level=0.0)
        p = {"composition_x": 0.5, "temperature": 500.0, "pressure": 5.0}
        r1 = sim.simulate(p)
        r2 = sim.simulate(p)
        assert abs(r1["conductivity"] - r2["conductivity"]) < 1e-9


# ── GaussianProcessSurrogate ───────────────────────────────────────────────────

class TestGaussianProcess:
    def test_fit_predict_shape(self):
        gp = GaussianProcessSurrogate()
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        gp.fit(X, y)
        mu, std = gp.predict(np.random.rand(5, 2))
        assert mu.shape == (5,)
        assert std.shape == (5,)

    def test_predict_positive_std(self):
        gp = GaussianProcessSurrogate()
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        gp.fit(X, y)
        _, std = gp.predict(np.random.rand(3, 2))
        assert np.all(std > 0)


# ── BayesianOptimizer ──────────────────────────────────────────────────────────

class TestBayesianOptimizer:
    def test_suggest_within_bounds(self):
        opt = BayesianOptimizer({"x": (0, 1), "y": (0, 1)})
        suggestion = opt.suggest()
        assert 0 <= suggestion["x"] <= 1
        assert 0 <= suggestion["y"] <= 1

    def test_update_and_best(self):
        opt = BayesianOptimizer({"x": (0, 1)}, objective="conductivity")
        opt.update({"x": 0.3}, {"conductivity": 5.0})
        opt.update({"x": 0.7}, {"conductivity": 10.0})
        best_params, best_val = opt.best_observed()
        assert abs(best_val - 10.0) < 1e-9

    def test_suggest_after_observations(self):
        opt = BayesianOptimizer({"x": (0, 1), "y": (0, 1)})
        for _ in range(5):
            x, y = np.random.rand(), np.random.rand()
            opt.update({"x": x, "y": y}, {"conductivity": x + y})
        suggestion = opt.suggest()
        assert set(suggestion.keys()) == {"x", "y"}


# ── ResultsTracker ─────────────────────────────────────────────────────────────

class TestResultsTracker:
    def test_log_and_count(self):
        tracker = ResultsTracker()
        tracker.log({"x": 0.5}, {"conductivity": 10.0, "strength": 50.0})
        assert tracker.count() == 1

    def test_best_by_objective(self):
        tracker = ResultsTracker()
        tracker.log({"x": 0.1}, {"conductivity": 5.0, "strength": 40.0})
        tracker.log({"x": 0.9}, {"conductivity": 15.0, "strength": 30.0})
        best = tracker.best("conductivity")
        assert best["results"]["conductivity"] == 15.0

    def test_pareto_front(self):
        tracker = ResultsTracker(objectives=["conductivity", "strength"])
        tracker.log({"x": 0.1}, {"conductivity": 10.0, "strength": 80.0})
        tracker.log({"x": 0.5}, {"conductivity": 50.0, "strength": 50.0})
        tracker.log({"x": 0.9}, {"conductivity": 90.0, "strength": 20.0})
        # Middle one is dominated by neither; all may be pareto optimal
        pf = tracker.pareto_front()
        assert len(pf) >= 1

    def test_persistence(self, tmp_path):
        log_file = str(tmp_path / "results.json")
        tracker = ResultsTracker(log_path=log_file)
        tracker.log({"x": 0.5}, {"conductivity": 10.0, "strength": 50.0})
        # Reload
        tracker2 = ResultsTracker(log_path=log_file)
        assert tracker2.count() == 1

    def test_summary(self):
        tracker = ResultsTracker()
        tracker.log({"x": 0.1}, {"conductivity": 5.0, "strength": 40.0})
        tracker.log({"x": 0.9}, {"conductivity": 15.0, "strength": 60.0})
        s = tracker.summary()
        assert s["conductivity"]["max"] == 15.0


# ── ExperimentLoop ─────────────────────────────────────────────────────────────

class TestExperimentLoop:
    def _make_loop(self):
        designer = ExperimentDesigner({"x": (0, 1), "y": (0, 1)})
        simulator = MaterialsSimulator()
        optimizer = BayesianOptimizer({"x": (0, 1), "y": (0, 1)})
        tracker = ResultsTracker()
        return ExperimentLoop(designer, simulator, optimizer, tracker, n_initial=3)

    def test_initialize_runs_n_experiments(self):
        loop = self._make_loop()
        loop.initialize()
        assert loop.tracker.count() == 3

    def test_step_adds_experiment(self):
        loop = self._make_loop()
        loop.initialize()
        before = loop.tracker.count()
        loop.step()
        assert loop.tracker.count() == before + 1

    def test_run_full_loop(self):
        loop = self._make_loop()
        results = loop.run(n_iterations=5)
        assert len(results) == 3 + 5  # init + iterations

    def test_best_is_returned(self):
        loop = self._make_loop()
        loop.run(n_iterations=5)
        best = loop.best()
        assert best is not None
