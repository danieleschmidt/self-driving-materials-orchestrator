"""Comprehensive tests for quantum-enhanced pipeline optimization."""

import asyncio
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from materials_orchestrator import (
    DistributedQuantumPipelineGuard,
    get_quantum_pipeline_guard,
    QuantumConfiguration,
    PipelineOptimizationProblem,
    create_quantum_optimization_problem
)
from materials_orchestrator.quantum_enhanced_pipeline_guard import (
    QuantumAnnealingOptimizer,
    HybridClassicalQuantumOptimizer,
    QuantumLoadBalancer,
    QuantumResultCache,
    OptimizationStrategy,
    QuantumState
)


class TestQuantumConfiguration:
    """Test suite for QuantumConfiguration."""
    
    def test_quantum_config_initialization(self):
        """Test quantum configuration initialization."""
        config = QuantumConfiguration()
        
        assert config.num_qubits == 10
        assert config.annealing_schedule == [0.0, 1.0]
        assert config.coupling_strength == 1.0
        assert config.coherence_time == 100.0
        assert config.error_rate == 0.001
        assert config.temperature == 0.01
    
    def test_quantum_config_custom_values(self):
        """Test quantum configuration with custom values."""
        config = QuantumConfiguration(
            num_qubits=16,
            annealing_schedule=[0.0, 0.5, 1.0],
            coupling_strength=2.0,
            coherence_time=200.0,
            error_rate=0.005,
            temperature=0.02
        )
        
        assert config.num_qubits == 16
        assert config.annealing_schedule == [0.0, 0.5, 1.0]
        assert config.coupling_strength == 2.0
        assert config.coherence_time == 200.0
        assert config.error_rate == 0.005
        assert config.temperature == 0.02
    
    def test_quantum_config_to_dict(self):
        """Test quantum configuration serialization."""
        config = QuantumConfiguration(num_qubits=8)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["num_qubits"] == 8
        assert "annealing_schedule" in config_dict
        assert "coupling_strength" in config_dict
        assert "coherence_time" in config_dict
        assert "error_rate" in config_dict
        assert "temperature" in config_dict


class TestPipelineOptimizationProblem:
    """Test suite for PipelineOptimizationProblem."""
    
    def test_optimization_problem_creation(self):
        """Test optimization problem creation."""
        variables = {
            "temperature": (100.0, 300.0),
            "pressure": (1.0, 10.0),
            "concentration": (0.1, 2.0)
        }
        
        problem = create_quantum_optimization_problem(
            objective="minimize_energy",
            variables=variables,
            constraints=["temperature < 250", "pressure > 2"]
        )
        
        assert problem.objective_function == "minimize_energy"
        assert len(problem.variables) == 3
        assert len(problem.constraints) == 2
        assert problem.optimization_type == "minimize"
        assert "temperature" in problem.variables
        assert problem.variables["temperature"] == (100.0, 300.0)
    
    def test_quantum_state_encoding(self):
        """Test quantum state encoding."""
        variables = {
            "var1": (0.0, 1.0),
            "var2": (0.0, 1.0)
        }
        
        problem = PipelineOptimizationProblem(
            objective_function="test_objective",
            constraints=[],
            variables=variables
        )
        
        values = {"var1": 0.5, "var2": 0.8}
        amplitudes = problem.encode_quantum_state(values)
        
        assert len(amplitudes) == 2 ** len(variables)  # 2^2 = 4 states
        assert all(isinstance(amp, complex) for amp in amplitudes)
        
        # Check normalization
        norm_squared = sum(abs(amp)**2 for amp in amplitudes)
        assert abs(norm_squared - 1.0) < 1e-6


class TestQuantumAnnealingOptimizer:
    """Test suite for QuantumAnnealingOptimizer."""
    
    @pytest.fixture
    def quantum_config(self):
        """Quantum configuration for testing."""
        return QuantumConfiguration(num_qubits=4, coherence_time=50.0)
    
    @pytest.fixture
    def optimizer(self, quantum_config):
        """Quantum annealing optimizer for testing."""
        return QuantumAnnealingOptimizer(quantum_config)
    
    @pytest.fixture
    def test_problem(self):
        """Test optimization problem."""
        return PipelineOptimizationProblem(
            objective_function="quadratic",
            constraints=[],
            variables={"x": (0.0, 1.0), "y": (0.0, 1.0)}
        )
    
    def test_optimizer_initialization(self, optimizer, quantum_config):
        """Test quantum optimizer initialization."""
        assert optimizer.config == quantum_config
        assert len(optimizer.quantum_state) == 2 ** quantum_config.num_qubits
        assert len(optimizer.optimization_history) == 0
    
    def test_quantum_state_reset(self, optimizer):
        """Test quantum state reset."""
        # Modify state
        optimizer.quantum_state[0] = complex(0.5, 0.5)
        
        # Reset
        optimizer.reset_quantum_state()
        
        # Check uniform superposition
        expected_amplitude = 1.0 / np.sqrt(len(optimizer.quantum_state))
        for amplitude in optimizer.quantum_state:
            assert abs(amplitude.real - expected_amplitude) < 1e-6
            assert abs(amplitude.imag) < 1e-6
    
    def test_hamiltonian_construction(self, optimizer, test_problem):
        """Test Hamiltonian construction."""
        H = optimizer.construct_hamiltonian(test_problem, 0.5)
        
        # Check properties
        assert H.shape == (16, 16)  # 2^4 x 2^4 for 4 qubits
        assert np.allclose(H, H.conj().T)  # Should be Hermitian
    
    def test_annealing_schedule(self, optimizer):
        """Test annealing schedule."""
        # Test points
        assert optimizer._annealing_schedule(0.0) == 0.0
        assert optimizer._annealing_schedule(1.0) == 1.0
        assert 0.0 <= optimizer._annealing_schedule(0.5) <= 1.0
    
    @pytest.mark.asyncio
    async def test_quantum_evolution(self, optimizer, test_problem):
        """Test quantum state evolution."""
        initial_state = optimizer.quantum_state.copy()
        
        # Evolve for short time
        await optimizer.evolve_quantum_state(test_problem, 0.1, dt=0.01)
        
        # State should have changed
        assert not np.allclose(optimizer.quantum_state, initial_state)
        
        # History should be recorded
        assert len(optimizer.optimization_history) > 0
    
    def test_quantum_measurement(self, optimizer):
        """Test quantum state measurement."""
        result = optimizer.measure_quantum_state()
        
        assert isinstance(result, dict)
        assert len(result) == optimizer.config.num_qubits
        
        # All values should be 0 or 1 (binary encoding)
        for key, value in result.items():
            assert value in [0.0, 1.0]
    
    @pytest.mark.asyncio
    async def test_full_optimization(self, optimizer, test_problem):
        """Test full quantum optimization."""
        result = await optimizer.optimize(test_problem, annealing_time=0.1)
        
        assert "result" in result
        assert "optimization_time" in result
        assert "final_energy" in result
        assert "quantum_state" in result
        assert "history" in result
        
        assert isinstance(result["result"], dict)
        assert result["optimization_time"] > 0
        assert isinstance(result["final_energy"], (int, float, complex))


class TestHybridClassicalQuantumOptimizer:
    """Test suite for HybridClassicalQuantumOptimizer."""
    
    @pytest.fixture
    def hybrid_optimizer(self):
        """Hybrid optimizer for testing."""
        config = QuantumConfiguration(num_qubits=6)
        return HybridClassicalQuantumOptimizer(config)
    
    @pytest.fixture
    def test_problem(self):
        """Test optimization problem."""
        return PipelineOptimizationProblem(
            objective_function="rosenbrock",
            constraints=[],
            variables={"x": (-2.0, 2.0), "y": (-2.0, 2.0)}
        )
    
    def test_hybrid_optimizer_initialization(self, hybrid_optimizer):
        """Test hybrid optimizer initialization."""
        assert hybrid_optimizer.quantum_optimizer is not None
        assert 0.0 <= hybrid_optimizer.quantum_classical_ratio <= 1.0
        assert hybrid_optimizer.iteration_limit > 0
        assert hybrid_optimizer.convergence_threshold > 0
    
    @pytest.mark.asyncio
    async def test_classical_optimization_step(self, hybrid_optimizer, test_problem):
        """Test classical optimization step."""
        result = await hybrid_optimizer._classical_optimization_step(test_problem)
        
        assert isinstance(result, dict)
        assert len(result) == len(test_problem.variables)
        
        # Check bounds
        for var_name, value in result.items():
            var_min, var_max = test_problem.variables[var_name]
            assert var_min <= value <= var_max
    
    def test_objective_evaluation(self, hybrid_optimizer, test_problem):
        """Test objective function evaluation."""
        values = {"x": 1.0, "y": 1.0}
        energy = hybrid_optimizer._evaluate_objective(test_problem, values)
        
        assert isinstance(energy, (int, float))
        assert energy >= 0  # Assuming non-negative objective
    
    @pytest.mark.asyncio
    async def test_hybrid_optimization(self, hybrid_optimizer, test_problem):
        """Test hybrid optimization."""
        # Use fewer iterations for testing
        hybrid_optimizer.iteration_limit = 10
        
        result = await hybrid_optimizer.optimize_hybrid(test_problem)
        
        assert "best_result" in result
        assert "best_energy" in result
        assert "optimization_time" in result
        assert "iterations" in result
        assert "method" in result
        
        assert result["method"] == "hybrid_classical_quantum"
        assert result["iterations"] <= 10
        assert result["optimization_time"] > 0


class TestQuantumLoadBalancer:
    """Test suite for QuantumLoadBalancer."""
    
    @pytest.fixture
    def load_balancer(self):
        """Load balancer for testing."""
        return QuantumLoadBalancer()
    
    @pytest.fixture
    def quantum_configs(self):
        """List of quantum configurations."""
        return [
            QuantumConfiguration(num_qubits=8, error_rate=0.001),
            QuantumConfiguration(num_qubits=12, error_rate=0.002),
            QuantumConfiguration(num_qubits=16, error_rate=0.003)
        ]
    
    @pytest.fixture
    def test_problem(self):
        """Test problem for load balancing."""
        return PipelineOptimizationProblem(
            objective_function="test",
            constraints=[],
            variables={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}
        )
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        assert len(load_balancer.node_loads) == 0
        assert len(load_balancer.node_performances) == 0
        assert len(load_balancer.selection_history) == 0
    
    def test_node_selection(self, load_balancer, quantum_configs, test_problem):
        """Test optimal node selection."""
        node_id = load_balancer.select_optimal_node(test_problem, quantum_configs)
        
        assert 0 <= node_id < len(quantum_configs)
        assert load_balancer.selection_history[node_id] == 1
        assert load_balancer.node_loads[node_id] > 0
    
    def test_performance_update(self, load_balancer):
        """Test performance update."""
        load_balancer.update_node_performance(0, 0.95)
        load_balancer.update_node_performance(0, 0.85)
        
        assert len(load_balancer.node_performances[0]) == 2
        assert 0.95 in load_balancer.node_performances[0]
        assert 0.85 in load_balancer.node_performances[0]
    
    def test_load_balancer_statistics(self, load_balancer, quantum_configs, test_problem):
        """Test load balancer statistics."""
        # Select some nodes
        for _ in range(5):
            load_balancer.select_optimal_node(test_problem, quantum_configs)
        
        # Update some performances
        load_balancer.update_node_performance(0, 0.9)
        load_balancer.update_node_performance(1, 0.8)
        
        stats = load_balancer.get_statistics()
        
        assert "current_loads" in stats
        assert "selection_history" in stats
        assert "avg_performances" in stats
        
        assert len(stats["selection_history"]) > 0
        assert sum(stats["selection_history"].values()) == 5


class TestQuantumResultCache:
    """Test suite for QuantumResultCache."""
    
    @pytest.fixture
    def cache(self):
        """Result cache for testing."""
        return QuantumResultCache(max_size=3)
    
    @pytest.fixture
    def test_problems(self):
        """List of test problems."""
        return [
            PipelineOptimizationProblem(
                objective_function=f"objective_{i}",
                constraints=[],
                variables={"x": (0.0, 1.0)}
            )
            for i in range(5)
        ]
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 3
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_cache_miss(self, cache, test_problems):
        """Test cache miss."""
        result = cache.get(test_problems[0])
        
        assert result is None
        assert cache.miss_count == 1
        assert cache.hit_count == 0
    
    def test_cache_store_and_hit(self, cache, test_problems):
        """Test cache store and hit."""
        test_result = {"energy": 1.5, "time": 10.0}
        
        # Store result
        cache.store(test_problems[0], test_result)
        
        # Retrieve result
        retrieved = cache.get(test_problems[0])
        
        assert retrieved is not None
        assert retrieved == test_result
        assert cache.hit_count == 1
        assert cache.miss_count == 0
    
    def test_cache_eviction(self, cache, test_problems):
        """Test LRU cache eviction."""
        # Fill cache beyond capacity
        for i in range(4):  # More than max_size (3)
            cache.store(test_problems[i], {"result": i})
        
        # First item should be evicted
        result = cache.get(test_problems[0])
        assert result is None  # Should be evicted
        
        # Last items should still be cached
        result = cache.get(test_problems[3])
        assert result is not None
    
    def test_cache_hit_rate(self, cache, test_problems):
        """Test cache hit rate calculation."""
        # Store some results
        for i in range(3):
            cache.store(test_problems[i], {"result": i})
        
        # Access cached items (hits)
        for i in range(3):
            cache.get(test_problems[i])
        
        # Access non-cached items (misses)
        cache.get(test_problems[3])
        cache.get(test_problems[4])
        
        hit_rate = cache.get_hit_rate()
        assert hit_rate == 3 / 5  # 3 hits out of 5 requests


class TestDistributedQuantumPipelineGuard:
    """Test suite for DistributedQuantumPipelineGuard."""
    
    @pytest.fixture
    def quantum_guard(self):
        """Distributed quantum pipeline guard for testing."""
        return DistributedQuantumPipelineGuard(num_quantum_nodes=3)
    
    @pytest.fixture
    def test_problems(self):
        """List of optimization problems."""
        return [
            create_quantum_optimization_problem(
                objective=f"objective_{i}",
                variables={"x": (0.0, 1.0), "y": (0.0, 1.0)}
            )
            for i in range(4)
        ]
    
    def test_quantum_guard_initialization(self, quantum_guard):
        """Test quantum guard initialization."""
        assert quantum_guard.num_quantum_nodes == 3
        assert len(quantum_guard.quantum_configs) == 3
        assert quantum_guard.load_balancer is not None
        assert quantum_guard.result_cache is not None
        assert len(quantum_guard.quantum_entanglement_graph) == 3
    
    def test_entanglement_graph_creation(self, quantum_guard):
        """Test entanglement graph creation."""
        graph = quantum_guard.quantum_entanglement_graph
        
        # Should be fully connected
        for node_id in range(quantum_guard.num_quantum_nodes):
            assert node_id in graph
            # Each node should be connected to all others
            expected_connections = quantum_guard.num_quantum_nodes - 1
            assert len(graph[node_id]) == expected_connections
    
    def test_problem_distribution(self, quantum_guard, test_problems):
        """Test problem distribution across nodes."""
        problem_batches = quantum_guard._distribute_problems(test_problems)
        
        assert isinstance(problem_batches, dict)
        
        # All problems should be distributed
        total_distributed = sum(len(batch) for batch in problem_batches.values())
        assert total_distributed == len(test_problems)
        
        # All node IDs should be valid
        for node_id in problem_batches.keys():
            assert 0 <= node_id < quantum_guard.num_quantum_nodes
    
    @pytest.mark.asyncio
    async def test_single_node_optimization(self, quantum_guard, test_problems):
        """Test optimization on a single node."""
        # Test with one problem
        results = await quantum_guard._optimize_on_quantum_node(0, [test_problems[0]])
        
        assert len(results) == 1
        result = results[0]
        
        assert "best_result" in result
        assert "best_energy" in result
        assert "optimization_time" in result
        assert "node_id" in result
        assert "quantum_config" in result
        
        assert result["node_id"] == 0
    
    @pytest.mark.asyncio
    async def test_quantum_result_fusion(self, quantum_guard):
        """Test quantum result fusion."""
        # Create mock results
        mock_results = [
            {
                "best_result": {"x": 0.5, "y": 0.5},
                "best_energy": 1.0,
                "node_id": 0,
                "optimization_time": 1.0
            },
            {
                "best_result": {"x": 0.6, "y": 0.4},
                "best_energy": 0.8,
                "node_id": 1,
                "optimization_time": 1.2
            }
        ]
        
        fused_result = await quantum_guard._quantum_result_fusion(mock_results)
        
        assert "best_result" in fused_result
        assert "combined_energy" in fused_result
        assert "individual_energy" in fused_result
        assert "fusion_fidelity" in fused_result
        assert "entanglement_strength" in fused_result
        
        # Best individual result should be preserved
        assert fused_result["individual_energy"] == 0.8
    
    @pytest.mark.asyncio
    async def test_parallel_optimization(self, quantum_guard, test_problems):
        """Test parallel quantum optimization."""
        result = await quantum_guard.optimize_pipeline_configuration(
            test_problems[:2],  # Use fewer problems for testing
            parallel_execution=True
        )
        
        assert "combined_result" in result
        assert "individual_results" in result
        assert "optimization_time" in result
        assert "quantum_efficiency" in result
        assert "nodes_used" in result
        
        assert result["optimization_time"] > 0
        assert result["nodes_used"] > 0
        assert len(result["individual_results"]) <= len(test_problems[:2])
    
    @pytest.mark.asyncio
    async def test_sequential_optimization(self, quantum_guard, test_problems):
        """Test sequential quantum optimization."""
        result = await quantum_guard.optimize_pipeline_configuration(
            test_problems[:2],
            parallel_execution=False
        )
        
        assert "combined_result" in result
        assert "individual_results" in result
        assert result["optimization_time"] > 0
    
    def test_quantum_efficiency_calculation(self, quantum_guard):
        """Test quantum efficiency calculation."""
        mock_results = [
            {"iterations": 10, "optimization_time": 1.0},
            {"iterations": 15, "optimization_time": 1.5},
            {"iterations": 8, "optimization_time": 0.8}
        ]
        
        efficiency = quantum_guard._calculate_quantum_efficiency(mock_results)
        
        assert isinstance(efficiency, float)
        assert efficiency > 0
    
    def test_entanglement_fidelity_calculation(self, quantum_guard):
        """Test entanglement fidelity calculation."""
        mock_results = [
            {"best_energy": 1.0},
            {"best_energy": 1.1},
            {"best_energy": 0.9}
        ]
        
        fidelity = quantum_guard._calculate_entanglement_fidelity(mock_results)
        
        assert isinstance(fidelity, float)
        assert 0 <= fidelity <= 1
    
    def test_performance_metrics(self, quantum_guard):
        """Test performance metrics collection."""
        # Add some mock performance data
        quantum_guard.performance_metrics["optimization_time"].extend([1.0, 1.5, 2.0])
        quantum_guard.performance_metrics["problems_solved"].extend([2, 3, 1])
        quantum_guard.performance_metrics["quantum_efficiency"].extend([0.8, 0.9, 0.7])
        
        metrics = quantum_guard.get_quantum_performance_metrics()
        
        assert "quantum_nodes" in metrics
        assert "entanglement_graph" in metrics
        assert "performance_metrics" in metrics
        assert "quantum_configs" in metrics
        assert "cache_hit_rate" in metrics
        assert "load_balancer_stats" in metrics
        
        assert metrics["quantum_nodes"] == 3
        assert "avg_optimization_time" in metrics["performance_metrics"]
        assert metrics["performance_metrics"]["total_problems_solved"] == 6


class TestIntegration:
    """Integration tests for quantum-enhanced optimization."""
    
    @pytest.mark.asyncio
    async def test_full_quantum_optimization_pipeline(self):
        """Test full quantum optimization pipeline."""
        # Get global quantum guard
        quantum_guard = get_quantum_pipeline_guard()
        
        # Create test problems
        problems = [
            create_quantum_optimization_problem(
                objective="minimize_variance",
                variables={
                    "temperature": (100.0, 300.0),
                    "pressure": (1.0, 10.0),
                    "concentration": (0.1, 2.0)
                },
                constraints=["temperature < 250"]
            ),
            create_quantum_optimization_problem(
                objective="maximize_efficiency", 
                variables={
                    "flow_rate": (0.1, 5.0),
                    "reaction_time": (1.0, 24.0)
                }
            )
        ]
        
        # Run optimization
        result = await quantum_guard.optimize_pipeline_configuration(
            problems,
            parallel_execution=True
        )
        
        # Verify results
        assert result["optimization_time"] > 0
        assert result["quantum_efficiency"] > 0
        assert len(result["individual_results"]) <= len(problems)
        assert "combined_result" in result
        
        # Check performance metrics
        metrics = quantum_guard.get_quantum_performance_metrics()
        assert metrics["performance_metrics"]["optimization_count"] > 0
    
    @pytest.mark.asyncio 
    async def test_quantum_cache_performance(self):
        """Test quantum result cache performance."""
        quantum_guard = get_quantum_pipeline_guard()
        
        # Create identical problems for cache testing
        problem = create_quantum_optimization_problem(
            objective="cache_test",
            variables={"x": (0.0, 1.0)}
        )
        
        # First optimization (cache miss)
        start_time = time.time()
        result1 = await quantum_guard.optimize_pipeline_configuration([problem])
        first_time = time.time() - start_time
        
        # Second optimization (cache hit)
        start_time = time.time()
        result2 = await quantum_guard.optimize_pipeline_configuration([problem])
        second_time = time.time() - start_time
        
        # Cache should improve performance
        cache_hit_rate = quantum_guard.result_cache.get_hit_rate()
        assert cache_hit_rate > 0  # Should have cache hits
        
        # Results should be consistent
        assert "combined_result" in result1
        assert "combined_result" in result2


# Performance benchmarks
class TestPerformance:
    """Performance tests for quantum optimization."""
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Test quantum optimization performance."""
        config = QuantumConfiguration(num_qubits=8)
        optimizer = QuantumAnnealingOptimizer(config)
        
        problem = create_quantum_optimization_problem(
            objective="performance_test",
            variables={"x": (0.0, 1.0), "y": (0.0, 1.0)}
        )
        
        start_time = time.time()
        result = await optimizer.optimize(problem, annealing_time=0.1)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should complete in reasonable time
        assert optimization_time < 5.0
        assert result["optimization_time"] > 0
    
    @pytest.mark.asyncio
    async def test_distributed_optimization_scalability(self):
        """Test distributed optimization scalability."""
        # Test with different numbers of nodes
        node_counts = [2, 4, 6]
        
        for num_nodes in node_counts:
            quantum_guard = DistributedQuantumPipelineGuard(num_quantum_nodes=num_nodes)
            
            problems = [
                create_quantum_optimization_problem(
                    objective=f"scalability_test_{i}",
                    variables={"x": (0.0, 1.0)}
                )
                for i in range(num_nodes)
            ]
            
            start_time = time.time()
            result = await quantum_guard.optimize_pipeline_configuration(problems)
            end_time = time.time()
            
            optimization_time = end_time - start_time
            
            # More nodes should handle more problems efficiently
            assert optimization_time < 10.0
            assert result["nodes_used"] <= num_nodes
    
    def test_load_balancer_efficiency(self):
        """Test load balancer efficiency."""
        load_balancer = QuantumLoadBalancer()
        
        configs = [
            QuantumConfiguration(num_qubits=8, error_rate=0.001),
            QuantumConfiguration(num_qubits=12, error_rate=0.002),
            QuantumConfiguration(num_qubits=16, error_rate=0.003)
        ]
        
        problems = [
            create_quantum_optimization_problem(
                objective=f"load_test_{i}",
                variables={"x": (0.0, 1.0)}
            )
            for i in range(20)
        ]
        
        # Distribute problems
        node_selections = []
        for problem in problems:
            node_id = load_balancer.select_optimal_node(problem, configs)
            node_selections.append(node_id)
        
        # Check distribution is somewhat balanced
        from collections import Counter
        distribution = Counter(node_selections)
        
        # No node should be overloaded (more than 80% of problems)
        max_load = max(distribution.values())
        assert max_load <= len(problems) * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])