"""Performance benchmarks for optimization algorithms."""

import time
from typing import Any, Dict

import numpy as np
import pytest

from materials_orchestrator.core import MaterialsObjective
from materials_orchestrator.planners import BayesianPlanner
from tests.fixtures.sample_data import SampleDataGenerator


class TestOptimizationPerformance:
    """Benchmark optimization algorithm performance."""

    @pytest.fixture
    def large_parameter_space(self):
        """Create a large parameter space for benchmarking."""
        return {f"param_{i}": (0.0, 10.0) for i in range(20)}  # 20-dimensional space

    @pytest.fixture
    def sample_data(self):
        """Generate sample experiment data for benchmarking."""
        return SampleDataGenerator.generate_experiment_data(1000, noise_level=0.1)

    @pytest.mark.benchmark
    def test_bayesian_planner_scaling(self, large_parameter_space, benchmark):
        """Benchmark Bayesian planner performance with increasing data size."""
        planner = BayesianPlanner(
            acquisition_function="expected_improvement", target_property="band_gap"
        )

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        # Generate increasing amounts of historical data
        def run_optimization(n_samples: int):
            # Generate synthetic data
            experiments = []
            for i in range(n_samples):
                params = {
                    param: np.random.uniform(low, high)
                    for param, (low, high) in large_parameter_space.items()
                }
                result = {"band_gap": np.random.uniform(1.0, 2.0)}
                experiments.append({"parameters": params, "results": result})

            # Benchmark suggestion generation
            start_time = time.time()
            suggestions = planner.suggest_next_experiments(
                objective=objective,
                parameter_space=large_parameter_space,
                previous_experiments=experiments,
                n_suggestions=5,
            )
            end_time = time.time()

            return end_time - start_time, len(suggestions)

        # Test with different data sizes
        results = benchmark(run_optimization, 100)

        execution_time, n_suggestions = results
        assert n_suggestions == 5
        assert execution_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.benchmark
    def test_acquisition_function_performance(self, benchmark):
        """Benchmark different acquisition functions."""
        acquisition_functions = [
            "expected_improvement",
            "upper_confidence_bound",
            "probability_of_improvement",
            "entropy_search",
        ]

        param_space = {"x": (0, 10), "y": (0, 10)}
        objective = MaterialsObjective(
            target_property="objective",
            target_range=(5, 6),
            optimization_direction="target",
        )

        # Generate sample data
        experiments = []
        for i in range(50):
            params = {"x": np.random.uniform(0, 10), "y": np.random.uniform(0, 10)}
            # Simple objective function: minimize distance from (5, 5)
            value = 10 - np.sqrt((params["x"] - 5) ** 2 + (params["y"] - 5) ** 2)
            experiments.append({"parameters": params, "results": {"objective": value}})

        def benchmark_acquisition_function(acq_func: str):
            planner = BayesianPlanner(
                acquisition_function=acq_func, target_property="objective"
            )

            start_time = time.time()
            suggestions = planner.suggest_next_experiments(
                objective=objective,
                parameter_space=param_space,
                previous_experiments=experiments,
                n_suggestions=10,
            )
            end_time = time.time()

            return end_time - start_time, len(suggestions)

        # Benchmark each acquisition function
        for acq_func in acquisition_functions:
            execution_time, n_suggestions = benchmark(
                benchmark_acquisition_function, acq_func
            )
            assert n_suggestions == 10
            assert execution_time < 5.0  # Each should complete within 5 seconds

    @pytest.mark.benchmark
    def test_parallel_optimization(self, benchmark):
        """Benchmark parallel experiment suggestion generation."""
        planner = BayesianPlanner(
            acquisition_function="expected_improvement",
            target_property="band_gap",
            enable_parallel=True,
        )

        param_space = {
            "temperature": (100, 300),
            "concentration": (0.1, 2.0),
            "time": (1, 24),
            "pH": (3, 11),
        }

        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
        )

        # Generate substantial historical data
        experiments = SampleDataGenerator.generate_experiment_data(
            200, noise_level=0.05
        )

        def run_parallel_optimization(n_parallel: int):
            start_time = time.time()
            suggestions = planner.suggest_next_experiments(
                objective=objective,
                parameter_space=param_space,
                previous_experiments=experiments,
                n_suggestions=n_parallel,
            )
            end_time = time.time()

            return end_time - start_time, len(suggestions)

        # Test with different batch sizes
        for batch_size in [1, 5, 10, 20]:
            execution_time, n_suggestions = benchmark(
                run_parallel_optimization, batch_size
            )
            assert n_suggestions == batch_size

            # Parallel optimization should scale reasonably
            if batch_size == 1:
                base_time = execution_time
            else:
                # Should not increase linearly with batch size
                assert execution_time < base_time * batch_size * 0.8

    @pytest.mark.benchmark
    def test_convergence_speed(self, benchmark):
        """Benchmark convergence speed for different optimization strategies."""
        strategies = [
            {"algorithm": "random", "params": {}},
            {
                "algorithm": "bayesian",
                "params": {"acquisition_function": "expected_improvement"},
            },
            {
                "algorithm": "bayesian",
                "params": {"acquisition_function": "upper_confidence_bound"},
            },
        ]

        param_space = {"x": (-5, 5), "y": (-5, 5)}

        # Define a test function to optimize (Himmelblau's function)
        def test_function(x: float, y: float) -> float:
            return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

        def benchmark_convergence(strategy: Dict[str, Any]):
            if strategy["algorithm"] == "random":
                # Random search
                best_value = float("inf")
                experiments = []

                start_time = time.time()
                for i in range(50):
                    x = np.random.uniform(-5, 5)
                    y = np.random.uniform(-5, 5)
                    value = test_function(x, y)

                    if value < best_value:
                        best_value = value

                    experiments.append(
                        {
                            "parameters": {"x": x, "y": y},
                            "results": {"objective": value},
                        }
                    )
                end_time = time.time()

            else:
                # Bayesian optimization
                planner = BayesianPlanner(
                    acquisition_function=strategy["params"]["acquisition_function"],
                    target_property="objective",
                )

                objective = MaterialsObjective(
                    target_property="objective",
                    target_range=(0, 1),
                    optimization_direction="minimize",
                )

                experiments = []
                best_value = float("inf")

                start_time = time.time()
                for i in range(50):
                    if i < 5:
                        # Random initialization
                        x = np.random.uniform(-5, 5)
                        y = np.random.uniform(-5, 5)
                    else:
                        # Bayesian suggestions
                        suggestions = planner.suggest_next_experiments(
                            objective=objective,
                            parameter_space=param_space,
                            previous_experiments=experiments,
                            n_suggestions=1,
                        )
                        x = suggestions[0]["x"]
                        y = suggestions[0]["y"]

                    value = test_function(x, y)
                    if value < best_value:
                        best_value = value

                    experiments.append(
                        {
                            "parameters": {"x": x, "y": y},
                            "results": {"objective": value},
                        }
                    )

                end_time = time.time()

            return end_time - start_time, best_value, len(experiments)

        # Benchmark each strategy
        results = {}
        for strategy in strategies:
            strategy_name = f"{strategy['algorithm']}_{strategy['params'].get('acquisition_function', 'default')}"
            execution_time, best_value, n_experiments = benchmark(
                benchmark_convergence, strategy
            )

            results[strategy_name] = {
                "time": execution_time,
                "best_value": best_value,
                "experiments": n_experiments,
            }

            # All strategies should complete within reasonable time
            assert execution_time < 30.0
            assert n_experiments == 50

        # Bayesian optimization should generally outperform random search
        bayesian_best = min(
            results["bayesian_expected_improvement"]["best_value"],
            results["bayesian_upper_confidence_bound"]["best_value"],
        )
        random_best = results["random_default"]["best_value"]

        # Bayesian should find better solutions (lower values for minimization)
        assert bayesian_best <= random_best * 1.2  # Allow some tolerance for randomness

    @pytest.mark.benchmark
    def test_memory_usage(self, benchmark):
        """Benchmark memory usage with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        def measure_memory_usage(n_experiments: int):
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Generate large dataset
            experiments = SampleDataGenerator.generate_experiment_data(
                n_experiments=n_experiments, noise_level=0.05
            )

            planner = BayesianPlanner(
                acquisition_function="expected_improvement", target_property="band_gap"
            )

            objective = MaterialsObjective(
                target_property="band_gap",
                target_range=(1.2, 1.6),
                optimization_direction="target",
            )

            param_space = SampleDataGenerator.perovskite_parameters()

            # Generate suggestions
            suggestions = planner.suggest_next_experiments(
                objective=objective,
                parameter_space=param_space,
                previous_experiments=experiments,
                n_suggestions=10,
            )

            # Measure peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory

            return memory_increase, len(suggestions)

        # Test with different dataset sizes
        for n_experiments in [100, 500, 1000]:
            memory_increase, n_suggestions = benchmark(
                measure_memory_usage, n_experiments
            )

            assert n_suggestions == 10
            assert (
                memory_increase < 500
            )  # Should not use more than 500MB additional memory

            # Memory usage should scale reasonably with data size
            memory_per_experiment = memory_increase / n_experiments
            assert memory_per_experiment < 1.0  # Less than 1MB per experiment
