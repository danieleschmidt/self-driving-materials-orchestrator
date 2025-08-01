"""Performance tests for optimization algorithms."""

import pytest
import time
import numpy as np
from typing import List, Dict, Any

from materials_orchestrator.planners import BayesianPlanner
from materials_orchestrator.optimization import MaterialsOptimizer
from tests.fixtures.materials_data import generate_large_dataset, PERFORMANCE_TEST_SIZES


@pytest.mark.benchmark
class TestOptimizationPerformance:
    """Benchmark optimization algorithm performance."""
    
    @pytest.mark.parametrize("dataset_size", PERFORMANCE_TEST_SIZES)
    def test_bayesian_optimization_scalability(self, benchmark, dataset_size):
        """Test Bayesian optimization performance with increasing dataset sizes."""
        
        def setup_and_optimize():
            # Generate synthetic data
            data = generate_large_dataset(dataset_size)
            
            # Extract parameters and results
            X = np.array([[d["parameters"]["temperature"], 
                          d["parameters"]["concentration"], 
                          d["parameters"]["time"]] for d in data])
            y = np.array([d["results"]["band_gap"] for d in data])
            
            # Initialize optimizer
            optimizer = MaterialsOptimizer(
                surrogate_model="gaussian_process",
                acquisition_function="expected_improvement"
            )
            
            # Fit model and suggest next experiments
            optimizer.fit(X, y)
            next_experiments = optimizer.suggest_next(
                n_suggestions=5,
                param_space={
                    "temperature": (80, 200),
                    "concentration": (0.5, 2.0),
                    "time": (5, 30)
                }
            )
            
            return len(next_experiments)
        
        # Benchmark the optimization process
        result = benchmark(setup_and_optimize)
        assert result == 5  # Should suggest 5 experiments
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_batch_optimization_performance(self, benchmark, batch_size):
        """Test performance of batch optimization with different batch sizes."""
        
        def run_batch_optimization():
            # Use medium-sized dataset
            data = generate_large_dataset(100)
            X = np.array([[d["parameters"]["temperature"], 
                          d["parameters"]["concentration"]] for d in data])
            y = np.array([d["results"]["efficiency"] for d in data])
            
            planner = BayesianPlanner(
                acquisition_function="upper_confidence_bound",
                batch_size=batch_size
            )
            
            planner.fit_surrogate_model(X, y)
            suggestions = planner.suggest_experiments(
                n_experiments=batch_size,
                param_space={
                    "temperature": (100, 200),
                    "concentration": (0.5, 2.0)
                }
            )
            
            return len(suggestions)
        
        result = benchmark(run_batch_optimization)
        assert result == batch_size
    
    def test_model_prediction_speed(self, benchmark):
        """Test speed of model predictions on large datasets."""
        
        # Generate large training dataset
        training_data = generate_large_dataset(1000)
        X_train = np.array([[d["parameters"]["temperature"], 
                           d["parameters"]["concentration"], 
                           d["parameters"]["time"]] for d in training_data])
        y_train = np.array([d["results"]["band_gap"] for d in training_data])
        
        # Generate prediction dataset
        test_data = generate_large_dataset(500)
        X_test = np.array([[d["parameters"]["temperature"], 
                          d["parameters"]["concentration"], 
                          d["parameters"]["time"]] for d in test_data])
        
        optimizer = MaterialsOptimizer()
        optimizer.fit(X_train, y_train)
        
        def make_predictions():
            predictions = optimizer.predict(X_test)
            return len(predictions)
        
        result = benchmark(make_predictions)
        assert result == 500
    
    def test_acquisition_function_computation_speed(self, benchmark):
        """Test speed of acquisition function computation."""
        
        data = generate_large_dataset(200)
        X = np.array([[d["parameters"]["temperature"], 
                      d["parameters"]["concentration"]] for d in data])
        y = np.array([d["results"]["band_gap"] for d in data])
        
        planner = BayesianPlanner(acquisition_function="expected_improvement")
        planner.fit_surrogate_model(X, y)
        
        # Generate candidate points
        n_candidates = 1000
        candidates = np.random.rand(n_candidates, 2)
        candidates[:, 0] = candidates[:, 0] * 120 + 80  # Temperature 80-200
        candidates[:, 1] = candidates[:, 1] * 1.5 + 0.5  # Concentration 0.5-2.0
        
        def compute_acquisition():
            scores = planner.compute_acquisition_scores(candidates)
            return len(scores)
        
        result = benchmark(compute_acquisition)
        assert result == n_candidates


@pytest.mark.benchmark
class TestDatabasePerformance:
    """Benchmark database operation performance."""
    
    @pytest.mark.parametrize("n_experiments", [100, 500, 1000, 5000])
    def test_experiment_storage_performance(self, benchmark, n_experiments, mock_database):
        """Test performance of storing large numbers of experiments."""
        
        experiments = generate_large_dataset(n_experiments)
        
        def store_experiments():
            stored_count = 0
            for exp in experiments:
                mock_database.store_experiment(exp)
                stored_count += 1
            return stored_count
        
        result = benchmark(store_experiments)
        assert result == n_experiments
    
    @pytest.mark.parametrize("query_size", [10, 50, 100, 500])
    def test_experiment_query_performance(self, benchmark, query_size, mock_database):
        """Test performance of querying experiments."""
        
        # Setup: store experiments in database
        all_experiments = generate_large_dataset(2000)
        for exp in all_experiments:
            mock_database.store_experiment(exp)
        
        def query_experiments():
            # Query for experiments with specific criteria
            results = mock_database.query_experiments(
                filters={
                    "parameters.temperature": {"$gte": 100, "$lte": 150}
                },
                limit=query_size
            )
            return len(results)
        
        result = benchmark(query_experiments)
        assert result <= query_size
    
    def test_aggregation_performance(self, benchmark, mock_database):
        """Test performance of database aggregation operations."""
        
        # Setup large dataset
        experiments = generate_large_dataset(1000)
        for exp in experiments:
            mock_database.store_experiment(exp)
        
        def run_aggregation():
            # Complex aggregation: group by temperature ranges and compute statistics
            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "$switch": {
                                "branches": [
                                    {"case": {"$lt": ["$parameters.temperature", 120]}, "then": "low"},
                                    {"case": {"$lt": ["$parameters.temperature", 160]}, "then": "medium"},
                                ],
                                "default": "high"
                            }
                        },
                        "avg_band_gap": {"$avg": "$results.band_gap"},
                        "count": {"$sum": 1}
                    }
                }
            ]
            results = mock_database.aggregate(pipeline)
            return len(results)
        
        result = benchmark(run_aggregation)
        assert result > 0


@pytest.mark.benchmark
class TestRobotOrchestrationPerformance:
    """Benchmark robot orchestration performance."""
    
    def test_parallel_robot_coordination(self, benchmark):
        """Test performance of coordinating multiple robots in parallel."""
        from tests.fixtures.mock_robots import MockOpentronsDriver, MockChemspeedDriver
        import asyncio
        
        async def coordinate_robots():
            # Create multiple mock robots
            robots = [
                MockOpentronsDriver(),
                MockOpentronsDriver(),
                MockChemspeedDriver(),
                MockChemspeedDriver()
            ]
            
            # Connect all robots
            await asyncio.gather(*[robot.connect() for robot in robots])
            
            # Execute actions in parallel
            actions = [
                ("pick_up_tip", {"tip_position": "A1"}),
                ("aspirate", {"volume": 100, "source": "B1"}),
                ("set_temperature", {"temperature": 150}),
                ("start_stirring", {"speed": 500})
            ]
            
            tasks = []
            for robot, (action, params) in zip(robots, actions):
                tasks.append(robot.execute_action(action, params))
            
            results = await asyncio.gather(*tasks)
            
            # Disconnect all robots
            await asyncio.gather(*[robot.disconnect() for robot in robots])
            
            return len(results)
        
        def run_coordination():
            return asyncio.run(coordinate_robots())
        
        result = benchmark(run_coordination)
        assert result == 4
    
    def test_protocol_execution_speed(self, benchmark):
        """Test speed of executing complex protocols."""
        from tests.fixtures.mock_robots import create_mock_robot_orchestrator
        import asyncio
        
        async def execute_complex_protocol():
            orchestrator = create_mock_robot_orchestrator()
            
            # Define complex protocol with multiple steps
            protocol = {
                "id": "complex_synthesis",
                "steps": [
                    {"robot": "opentrons", "action": "pick_up_tip", "parameters": {}},
                    {"robot": "opentrons", "action": "aspirate", "parameters": {"volume": 100}},
                    {"robot": "opentrons", "action": "dispense", "parameters": {"volume": 100}},
                    {"robot": "chemspeed", "action": "set_temperature", "parameters": {"temperature": 150}},
                    {"robot": "chemspeed", "action": "start_stirring", "parameters": {"speed": 500}},
                    {"robot": "chemspeed", "action": "wait", "parameters": {"duration": 0.1}},
                    {"robot": "chemspeed", "action": "stop_stirring", "parameters": {}},
                    {"robot": "opentrons", "action": "drop_tip", "parameters": {}}
                ]
            }
            
            result = await orchestrator.execute_protocol(protocol)
            return result["success"]
        
        def run_protocol():
            return asyncio.run(execute_complex_protocol())
        
        result = benchmark(run_protocol)
        assert result is True


@pytest.fixture
def mock_database():
    """Create a mock database for performance testing."""
    from unittest.mock import Mock
    
    db = Mock()
    stored_experiments = []
    
    def store_experiment(experiment):
        stored_experiments.append(experiment)
        return f"exp_{len(stored_experiments):06d}"
    
    def query_experiments(filters=None, limit=None):
        # Simple mock query implementation
        results = stored_experiments.copy()
        if filters and "parameters.temperature" in filters:
            temp_filter = filters["parameters.temperature"]
            if "$gte" in temp_filter and "$lte" in temp_filter:
                results = [exp for exp in results 
                          if temp_filter["$gte"] <= exp["parameters"]["temperature"] <= temp_filter["$lte"]]
        
        if limit:
            results = results[:limit]
        
        return results
    
    def aggregate(pipeline):
        # Mock aggregation - return dummy results
        return [
            {"_id": "low", "avg_band_gap": 1.45, "count": 100},
            {"_id": "medium", "avg_band_gap": 1.55, "count": 150},
            {"_id": "high", "avg_band_gap": 1.65, "count": 80}
        ]
    
    db.store_experiment = store_experiment
    db.query_experiments = query_experiments
    db.aggregate = aggregate
    
    return db