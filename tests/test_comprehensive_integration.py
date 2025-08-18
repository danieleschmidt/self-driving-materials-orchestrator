"""Comprehensive integration tests for Materials Orchestrator."""

import asyncio
import os
import time
from datetime import datetime

import pytest

from materials_orchestrator import (
    AutonomousLab,
    BayesianPlanner,
    ExperimentDatabase,
    MaterialsObjective,
    RobotOrchestrator,
)
from materials_orchestrator.error_handling import ErrorHandler
from materials_orchestrator.monitoring import SystemMonitor
from materials_orchestrator.optimization import ConcurrentExperimentRunner, LRUCache
from materials_orchestrator.scalability import DistributedJobManager, WorkerNode
from materials_orchestrator.security import SecurityConfig, SecurityValidator


class TestFullSystemIntegration:
    """Test complete system integration."""

    def test_end_to_end_discovery_campaign(self):
        """Test complete materials discovery campaign."""
        # Initialize components
        db = ExperimentDatabase()
        planner = BayesianPlanner(target_property="band_gap")
        lab = AutonomousLab(planner=planner)

        # Define objective
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.2, 1.6),
            optimization_direction="target",
            material_system="perovskites",
        )

        # Define parameter space
        param_space = {
            "temperature": (100, 300),
            "concentration": (0.1, 2.0),
            "time": (1, 24),
        }

        # Run campaign
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=5,
            max_experiments=20,
            stop_on_target=True,
        )

        # Validate results
        assert campaign is not None
        assert campaign.total_experiments > 0
        assert campaign.successful_experiments > 0
        assert 0 <= campaign.success_rate <= 1
        assert campaign.best_material is not None

        # Store in database
        campaign_id = db.store_campaign(campaign)
        assert campaign_id is not None

        # Retrieve from database
        summary = db.get_campaign_summary(campaign_id)
        assert summary is not None
        assert summary["total_experiments"] == campaign.total_experiments

    def test_security_validation_integration(self):
        """Test security validation integration."""
        validator = SecurityValidator(SecurityConfig(max_parameter_value=100.0))

        # Test valid parameters
        valid_params = {"temperature": 150, "concentration": 1.0}
        validated = validator.validate_parameters(valid_params)
        assert validated == valid_params

        # Test invalid parameters
        invalid_params = {"temperature": 1000, "bad_name!!!": 1.0}

        with pytest.raises(ValueError):
            validator.validate_parameters(invalid_params)

        # Test rate limiting
        client_id = "test_client"

        # Should allow initial requests
        assert validator.check_rate_limit(client_id) == True

        # Simulate rate limit breach
        for _ in range(200):  # Exceed rate limit
            validator.check_rate_limit(client_id)

        # Should be rate limited now
        assert validator.check_rate_limit(client_id) == False

    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        monitor = SystemMonitor()

        # Test metrics collection
        monitor.metrics.record_metric("test_metric", 42.0)
        monitor.metrics.increment_counter("test_counter")
        monitor.metrics.record_histogram("test_histogram", 1.5)

        # Get system status
        status = monitor.get_system_status()
        assert "health" in status
        assert "metrics" in status
        assert "experiments" in status

        # Test health checks
        health_status = monitor.health.get_health_status()
        assert "overall_healthy" in health_status
        assert "checks" in health_status

        monitor.shutdown()

    def test_error_handling_integration(self):
        """Test error handling and recovery."""
        from materials_orchestrator.error_handling import (
            ErrorCategory,
            ErrorSeverity,
            handle_errors,
        )

        error_handler = ErrorHandler()

        # Test automatic error categorization and handling
        def failing_function():
            raise ValueError("Invalid parameter value")

        @handle_errors(category=ErrorCategory.VALIDATION, severity=ErrorSeverity.MEDIUM)
        def wrapped_function():
            return failing_function()

        # Should handle the error
        with pytest.raises(ValueError):
            wrapped_function()

        # Check error was logged
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] > 0
        assert "validation" in stats["category_counts"]

    def test_performance_optimization_integration(self):
        """Test performance optimization features."""
        # Test caching
        cache = LRUCache(max_size=100, ttl_seconds=60)

        cache.put("test_key", {"result": "cached_value"})
        assert cache.get("test_key") == {"result": "cached_value"}

        # Test cache stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1

        # Test concurrent experiment runner
        runner = ConcurrentExperimentRunner(max_workers=2, cache_enabled=True)

        lab = AutonomousLab()
        parameter_sets = [{"temp": 100 + i, "conc": 1.0} for i in range(5)]

        # Run batch
        results = runner.run_experiments_batch(lab, parameter_sets)
        assert len(results) == len(parameter_sets)

        # Check performance stats
        perf_stats = runner.get_performance_stats()
        assert "max_workers" in perf_stats
        assert "cache_enabled" in perf_stats

    def test_scalability_integration(self):
        """Test distributed processing capabilities."""
        # Create job manager
        job_manager = DistributedJobManager(enable_auto_scaling=False)
        job_manager.start()

        # Register a simulated worker
        worker = WorkerNode(
            node_id="worker_1",
            host="localhost",
            port=8080,
            capabilities=["experiment"],
            max_concurrent_jobs=2,
        )
        job_manager.register_worker(worker)

        # Submit test jobs
        from materials_orchestrator.scalability import DistributedJob

        job = DistributedJob(
            job_id="test_job_1",
            job_type="experiment",
            payload={"parameters": {"temp": 150}},
        )

        job_id = job_manager.submit_job(job)
        assert job_id == "test_job_1"

        # Wait for job completion
        timeout = time.time() + 10  # 10 second timeout
        while time.time() < timeout:
            status = job_manager.get_job_status(job_id)
            if status in ["completed", "failed"]:
                break
            time.sleep(0.1)

        assert status == "completed"

        # Get system status
        system_status = job_manager.get_system_status()
        assert system_status["workers"]["total"] == 1
        assert system_status["workers"]["active"] == 1

        job_manager.stop()

    def test_robot_orchestration_integration(self):
        """Test robot orchestration system."""
        orchestrator = RobotOrchestrator()

        # Create simulated robots
        orchestrator.create_robot("robot_1", "simulated", {"simulation_delay": 0.1})
        orchestrator.create_robot("robot_2", "simulated", {"simulation_delay": 0.1})
        orchestrator.create_instrument("xrd_1", "xrd", {})

        # Test connection
        async def test_connections():
            results = await orchestrator.connect_all()
            assert len(results) == 3
            assert all(results.values())  # All should connect successfully

            # Test system status
            status = await orchestrator.get_system_status()
            assert len(status["robots"]) == 2
            assert len(status["instruments"]) == 1

            # Test parallel action execution
            actions = [
                {
                    "robot": "robot_1",
                    "action": "move",
                    "parameters": {"x": 10, "y": 20},
                },
                {
                    "robot": "robot_2",
                    "action": "heat",
                    "parameters": {"temperature": 150},
                },
                {
                    "instrument": "xrd_1",
                    "action": "measure",
                    "parameters": {"sample_id": "test"},
                },
            ]

            results = await orchestrator.execute_parallel_actions(actions)
            assert len(results) == 3
            assert all(
                result.success for result in results if hasattr(result, "success")
            )

            await orchestrator.disconnect_all()

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_connections())
        finally:
            loop.close()

    def test_data_persistence_and_retrieval(self):
        """Test data persistence and retrieval capabilities."""
        db = ExperimentDatabase()

        # Store test experiments
        test_experiments = []
        for i in range(10):
            experiment = {
                "id": f"exp_{i:03d}",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": {
                    "temperature": 100 + i * 10,
                    "concentration": 1.0 + i * 0.1,
                },
                "results": {
                    "band_gap": 1.4 + i * 0.01,
                    "efficiency": 20 + i,
                },
                "campaign_id": "test_campaign",
            }

            exp_id = db.store_experiment(experiment)
            assert exp_id == experiment["id"]
            test_experiments.append(experiment)

        # Test queries
        all_experiments = db.query_experiments(limit=20)
        assert len(all_experiments) >= 10

        # Test filtered queries
        high_temp_experiments = db.query_experiments(
            filter_criteria={"parameters.temperature": {"$gte": 150}}, limit=10
        )
        assert len(high_temp_experiments) > 0

        # Test best materials query
        best_materials = db.get_best_materials("band_gap", limit=5)
        assert len(best_materials) > 0
        assert all("results" in exp for exp in best_materials)

    def test_dashboard_data_integration(self):
        """Test dashboard data integration."""
        from materials_orchestrator.dashboard.utils import (
            calculate_success_metrics,
            create_property_evolution_plot,
            generate_experiment_summary,
        )

        # Create test experiment data
        experiments = []
        for i in range(20):
            experiments.append(
                {
                    "id": f"exp_{i}",
                    "status": "completed" if i < 18 else "failed",
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {"temp": 100 + i * 5},
                    "results": {"band_gap": 1.3 + i * 0.01, "efficiency": 15 + i},
                }
            )

        # Test success metrics
        success_criteria = {
            "band_gap": {"range": (1.2, 1.6)},
            "efficiency": {"min": 20},
        }

        metrics = calculate_success_metrics(experiments, success_criteria)
        assert metrics["total_experiments"] == 20
        assert metrics["completed_experiments"] == 18
        assert metrics["success_rate"] > 0

        # Test summary generation
        summary = generate_experiment_summary(experiments)
        assert "Total experiments: 20" in summary
        assert "Completed: 18" in summary

        # Test plot creation
        plot = create_property_evolution_plot(
            experiments, property_name="band_gap", target_range=(1.2, 1.6)
        )
        assert plot is not None

    def test_cli_integration(self):
        """Test CLI integration (basic smoke test)."""
        from materials_orchestrator.cli import setup_logging

        # Test logging setup
        setup_logging(debug=True)

        # Test that imports work (CLI commands would be tested separately)
        import materials_orchestrator.cli as cli_module

        assert hasattr(cli_module, "app")
        assert hasattr(cli_module, "main")


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_experiment_throughput(self):
        """Test experiment execution throughput."""
        lab = AutonomousLab()

        # Run batch of experiments and measure time
        start_time = time.time()

        parameter_sets = [{"temp": 100 + i, "conc": 1.0 + i * 0.1} for i in range(50)]

        results = []
        for params in parameter_sets:
            result = lab.run_experiment(params)
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 50 experiments in reasonable time
        assert duration < 10.0  # Less than 10 seconds
        assert len(results) == 50

        throughput = len(results) / duration
        assert throughput > 5  # At least 5 experiments per second

    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        runner = ConcurrentExperimentRunner(max_workers=4, cache_enabled=True)
        lab = AutonomousLab()

        # Create larger batch
        parameter_sets = [{"temp": 100 + i, "conc": 1.0 + i * 0.05} for i in range(100)]

        start_time = time.time()
        results = runner.run_experiments_batch(lab, parameter_sets)
        end_time = time.time()

        duration = end_time - start_time
        throughput = len(results) / duration

        # Concurrent processing should be faster
        assert throughput > 10  # At least 10 experiments per second
        assert len(results) == 100
        assert all(r is not None for r in results)

    def test_database_query_performance(self):
        """Test database query performance."""
        db = ExperimentDatabase()

        # Insert test data
        experiments = []
        for i in range(1000):
            exp = {
                "id": f"perf_test_{i:04d}",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": {"temp": 100 + (i % 200)},
                "results": {"band_gap": 1.2 + (i % 100) * 0.004},
            }
            db.store_experiment(exp)
            experiments.append(exp)

        # Test query performance
        start_time = time.time()
        results = db.query_experiments(limit=100)
        query_time = time.time() - start_time

        assert query_time < 1.0  # Should complete in less than 1 second
        assert len(results) == 100

        # Test filtered query performance
        start_time = time.time()
        filtered_results = db.query_experiments(
            filter_criteria={"parameters.temp": {"$gte": 200}}, limit=50
        )
        filtered_query_time = time.time() - start_time

        assert filtered_query_time < 2.0  # Should complete in less than 2 seconds
        assert len(filtered_results) > 0

    def test_memory_usage(self):
        """Test memory usage stays reasonable."""
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large campaign
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap",
            target_range=(1.0, 2.0),
            optimization_direction="target",
        )

        param_space = {
            "temp": (50, 500),
            "conc": (0.1, 5.0),
            "time": (1, 48),
        }

        # Run campaign
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=20,
            max_experiments=100,
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100 MB)
        assert memory_increase < 100
        assert campaign.total_experiments > 0


class TestSecurityValidation:
    """Security-focused tests."""

    def test_input_sanitization(self):
        """Test input sanitization."""
        validator = SecurityValidator()

        # Test malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "'; DROP TABLE experiments; --",
            "__import__('os').system('rm -rf /')",
        ]

        for malicious_input in malicious_inputs:
            sanitized = validator.sanitize_input(malicious_input)
            assert malicious_input not in str(sanitized)
            assert "<script" not in str(sanitized).lower()
            assert "javascript:" not in str(sanitized).lower()

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation edge cases."""
        validator = SecurityValidator(
            SecurityConfig(
                max_parameter_value=1000,
                min_parameter_value=-100,
                max_parameter_count=10,
            )
        )

        # Test boundary values
        edge_params = {
            "max_value": 1000,  # At limit
            "min_value": -100,  # At limit
            "normal_value": 50,
        }

        validated = validator.validate_parameters(edge_params)
        assert validated["max_value"] == 1000
        assert validated["min_value"] == -100

        # Test exceeding limits
        invalid_params = {
            "too_high": 1001,
            "too_low": -101,
        }

        with pytest.raises(ValueError):
            validator.validate_parameters(invalid_params)

        # Test too many parameters
        too_many_params = {f"param_{i}": i for i in range(15)}

        with pytest.raises(ValueError):
            validator.validate_parameters(too_many_params)

    def test_file_path_validation(self):
        """Test file path validation."""
        validator = SecurityValidator()

        # Valid paths
        valid_paths = [
            "results.json",
            "data/experiment_001.csv",
            "campaign_results.yaml",
        ]

        for path in valid_paths:
            validated = validator.validate_file_path(path)
            assert validated == path

        # Invalid paths
        invalid_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "data/../../../sensitive.txt",
            "results.exe",  # Invalid extension
        ]

        for path in invalid_paths:
            with pytest.raises(ValueError):
                validator.validate_file_path(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
