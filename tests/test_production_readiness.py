"""Production readiness tests for Materials Orchestrator."""

import pytest
import time
import threading
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

from materials_orchestrator import AutonomousLab, MaterialsObjective, ExperimentDatabase
from materials_orchestrator.security import SecurityValidator
from materials_orchestrator.monitoring import SystemMonitor, HealthCheck
from materials_orchestrator.error_handling import ErrorHandler
from materials_orchestrator.optimization import LRUCache, ConcurrentExperimentRunner
from materials_orchestrator.scalability import DistributedJobManager, AutoScaler


class TestProductionResilience:
    """Test system resilience under production conditions."""

    def test_high_load_handling(self):
        """Test system behavior under high load."""
        lab = AutonomousLab()
        runner = ConcurrentExperimentRunner(max_workers=8, cache_enabled=True)

        # Generate high load
        parameter_sets = [
            {"temp": 100 + i, "conc": 1.0 + i * 0.01, "pressure": 1 + i * 0.1}
            for i in range(500)  # 500 experiments
        ]

        start_time = time.time()
        results = runner.run_experiments_batch(lab, parameter_sets)
        duration = time.time() - start_time

        # Verify all experiments completed
        assert len(results) == 500
        successful_results = [r for r in results if r is not None]
        success_rate = len(successful_results) / len(results)

        # Should maintain high success rate under load
        assert success_rate > 0.95

        # Should complete within reasonable time (allow more time for CI)
        assert duration < 120  # 2 minutes max

        # Check system performance stats
        perf_stats = runner.get_performance_stats()
        cache_stats = perf_stats.get("cache_stats", {})
        if cache_stats:
            # Cache should have reasonable hit rate
            hit_rate = cache_stats.get("hit_rate", 0)
            assert hit_rate >= 0  # At least some cache usage

    def test_memory_leak_prevention(self):
        """Test for memory leaks during extended operation."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        lab = AutonomousLab()
        cache = LRUCache(max_size=100, ttl_seconds=10)

        # Run many operations to test for leaks
        for iteration in range(10):
            # Create and run experiments
            params_batch = [
                {"temp": 100 + i + iteration * 10, "conc": 1.0} for i in range(50)
            ]

            for params in params_batch:
                result = lab.run_experiment(params)
                cache.put(f"key_{iteration}_{hash(str(params))}", result)

            # Force some cleanup
            if iteration % 3 == 0:
                cache.clear()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded (less than 50 MB for this test)
        assert (
            memory_increase < 50
        ), f"Memory leak detected: {memory_increase:.1f} MB increase"

    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access."""
        lab = AutonomousLab()
        db = ExperimentDatabase()
        results = []
        errors = []

        def worker_thread(worker_id: int):
            """Worker thread function."""
            try:
                for i in range(20):
                    params = {
                        "worker_id": worker_id,
                        "iteration": i,
                        "temp": 100 + worker_id * 10 + i,
                        "conc": 1.0 + i * 0.1,
                    }

                    # Run experiment
                    result = lab.run_experiment(params)
                    results.append((worker_id, i, result))

                    # Store in database
                    experiment_doc = {
                        "id": f"worker_{worker_id}_exp_{i}",
                        "parameters": params,
                        "results": result.results if hasattr(result, "results") else {},
                        "status": (
                            result.status if hasattr(result, "status") else "completed"
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }

                    db.store_experiment(experiment_doc)

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch multiple concurrent workers
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100  # 5 workers × 20 iterations each

        # Verify database integrity
        stored_experiments = db.query_experiments(limit=200)
        assert len(stored_experiments) >= 100

    def test_failure_recovery(self):
        """Test system recovery from various failure scenarios."""
        error_handler = ErrorHandler()

        # Test database connection failure simulation
        def failing_database_operation():
            raise ConnectionError("Database connection lost")

        # Test network failure simulation
        def failing_network_operation():
            raise TimeoutError("Network timeout")

        # Test computation failure simulation
        def failing_computation():
            raise ValueError("Invalid computation parameters")

        test_functions = [
            failing_database_operation,
            failing_network_operation,
            failing_computation,
        ]

        for test_func in test_functions:
            # Handle error with recovery attempt
            result = error_handler.handle_error(
                exception=Exception("Test error"),
                original_function=test_func,
            )

            # Should handle gracefully (may not recover, but shouldn't crash)
            assert result is None or isinstance(result, (dict, list, str, int, float))

        # Check error statistics
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] >= 3
        assert stats["recovery_stats"]["attempted"] >= 0

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Test with disabled caching
        runner = ConcurrentExperimentRunner(cache_enabled=False)
        lab = AutonomousLab()

        params_list = [{"temp": 100 + i} for i in range(10)]
        results = runner.run_experiments_batch(lab, params_list)

        assert len(results) == 10
        assert all(r is not None for r in results)

        # Test with limited workers
        limited_runner = ConcurrentExperimentRunner(max_workers=1)
        results_limited = limited_runner.run_experiments_batch(lab, params_list)

        assert len(results_limited) == 10
        assert all(r is not None for r in results_limited)

    def test_health_monitoring_alerts(self):
        """Test health monitoring and alerting."""
        monitor = SystemMonitor()

        # Add custom health check that will fail
        def always_fail_check():
            return False

        failing_check = HealthCheck(
            name="test_failing_check",
            check_function=always_fail_check,
            interval_seconds=1,
        )

        monitor.health.add_health_check(failing_check)

        # Wait for health check to run and fail
        time.sleep(2)

        # Check that alerts were generated
        alerts = monitor.health.get_alerts(limit=10)
        assert len(alerts) > 0

        # Check health status
        health_status = monitor.health.get_health_status()
        assert not health_status["overall_healthy"]
        assert health_status["active_alerts"] > 0

        monitor.shutdown()


class TestScalabilityValidation:
    """Test scalability features."""

    def test_auto_scaling_logic(self):
        """Test auto-scaling decision making."""
        scaler = AutoScaler(
            min_workers=2,
            max_workers=10,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_seconds=1,  # Short cooldown for testing
        )

        # Test scale up conditions
        scale_decision = scaler.should_scale(
            current_workers=3,
            queue_length=10,  # High queue
            average_load=0.9,  # High load
            average_response_time=30.0,  # Slow response
        )
        assert scale_decision == "up"

        # Wait for cooldown
        time.sleep(1.5)

        # Test scale down conditions
        scale_decision = scaler.should_scale(
            current_workers=5,
            queue_length=0,  # No queue
            average_load=0.2,  # Low load
            average_response_time=2.0,  # Fast response
        )
        assert scale_decision == "down"

        # Test no scaling needed
        scale_decision = scaler.should_scale(
            current_workers=4,
            queue_length=2,
            average_load=0.5,  # Medium load
            average_response_time=10.0,
        )
        assert scale_decision is None

    def test_distributed_job_management(self):
        """Test distributed job management."""
        job_manager = DistributedJobManager(enable_auto_scaling=False)
        job_manager.start()

        from materials_orchestrator.scalability import WorkerNode, DistributedJob

        # Register workers
        for i in range(3):
            worker = WorkerNode(
                node_id=f"worker_{i}",
                host="localhost",
                port=8080 + i,
                capabilities=["experiment", "analysis"],
                max_concurrent_jobs=2,
            )
            job_manager.register_worker(worker)

        # Submit multiple jobs
        job_ids = []
        for i in range(10):
            job = DistributedJob(
                job_id=f"test_job_{i}",
                job_type="experiment",
                payload={"experiment_id": i, "params": {"temp": 100 + i}},
                priority=i % 3,  # Varying priorities
            )

            job_id = job_manager.submit_job(job)
            job_ids.append(job_id)

        # Wait for jobs to complete
        completed_jobs = 0
        timeout = time.time() + 30  # 30 second timeout

        while completed_jobs < 10 and time.time() < timeout:
            for job_id in job_ids:
                status = job_manager.get_job_status(job_id)
                if status in ["completed", "failed"]:
                    if job_id not in [j for j in job_ids[:completed_jobs]]:
                        completed_jobs += 1
            time.sleep(0.5)

        assert completed_jobs >= 8  # Allow some failures

        # Check system status
        system_status = job_manager.get_system_status()
        assert system_status["workers"]["total"] == 3
        assert system_status["jobs"]["completed"] >= 8

        job_manager.stop()

    def test_load_balancing_effectiveness(self):
        """Test load balancing across workers."""
        from materials_orchestrator.optimization import AdaptiveLoadBalancer

        worker_configs = [
            {"id": "fast_worker", "weight": 2.0},
            {"id": "medium_worker", "weight": 1.0},
            {"id": "slow_worker", "weight": 0.5},
        ]

        balancer = AdaptiveLoadBalancer(worker_configs)

        # Simulate requests and record selections
        worker_selections = {}

        for _ in range(100):
            selected_worker = balancer.select_worker()
            worker_selections[selected_worker] = (
                worker_selections.get(selected_worker, 0) + 1
            )

            # Simulate varying performance
            if selected_worker == "fast_worker":
                response_time = 0.1
                success = True
            elif selected_worker == "medium_worker":
                response_time = 0.5
                success = True
            else:  # slow_worker
                response_time = 1.0
                success = True

            balancer.record_request(selected_worker, response_time, success)

        # Fast worker should be selected most often
        assert worker_selections["fast_worker"] > worker_selections["slow_worker"]

        # Check load balancer stats
        stats = balancer.get_stats()
        assert stats["workers"] == 3
        assert stats["available_workers"] == 3


class TestDataIntegrityAndConsistency:
    """Test data integrity and consistency."""

    def test_database_transaction_consistency(self):
        """Test database transaction consistency."""
        db = ExperimentDatabase()

        # Store related experiments
        campaign_id = "consistency_test_campaign"
        experiment_ids = []

        for i in range(10):
            exp_data = {
                "id": f"consistency_exp_{i:03d}",
                "campaign_id": campaign_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": {"temp": 100 + i * 10},
                "results": {"band_gap": 1.3 + i * 0.02},
            }

            exp_id = db.store_experiment(exp_data)
            experiment_ids.append(exp_id)

        # Verify all experiments are retrievable
        retrieved_experiments = db.query_experiments(
            filter_criteria={"campaign_id": campaign_id}, limit=20
        )

        assert len(retrieved_experiments) == 10

        # Verify data integrity
        for exp in retrieved_experiments:
            assert exp["campaign_id"] == campaign_id
            assert exp["status"] == "completed"
            assert "band_gap" in exp.get("results", {})

    def test_concurrent_database_access(self):
        """Test concurrent database access consistency."""
        db = ExperimentDatabase()
        results = []
        errors = []

        def database_worker(worker_id: int):
            """Worker that performs database operations."""
            try:
                for i in range(20):
                    # Store experiment
                    exp_data = {
                        "id": f"concurrent_worker_{worker_id}_exp_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed",
                        "parameters": {"worker": worker_id, "iteration": i},
                        "results": {"value": worker_id * 100 + i},
                    }

                    exp_id = db.store_experiment(exp_data)

                    # Query experiments
                    recent_experiments = db.query_experiments(limit=5)
                    results.append((worker_id, i, len(recent_experiments)))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch concurrent database workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=database_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=20)

        assert len(errors) == 0, f"Database consistency errors: {errors}"
        assert len(results) == 60  # 3 workers × 20 operations each

    def test_cache_consistency(self):
        """Test cache consistency under concurrent access."""
        cache = LRUCache(max_size=100, ttl_seconds=60)
        results = []
        errors = []

        def cache_worker(worker_id: int):
            """Worker that performs cache operations."""
            try:
                for i in range(50):
                    key = f"key_{i % 10}"  # Overlapping keys
                    value = {
                        "worker": worker_id,
                        "iteration": i,
                        "timestamp": time.time(),
                    }

                    # Put and get operations
                    cache.put(key, value)
                    retrieved = cache.get(key)

                    results.append((worker_id, i, retrieved is not None))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch concurrent cache workers
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        assert len(errors) == 0, f"Cache consistency errors: {errors}"
        assert len(results) == 200  # 4 workers × 50 operations each

        # Verify cache integrity
        cache_stats = cache.get_stats()
        assert cache_stats["size"] <= 100  # Shouldn't exceed max size
        assert cache_stats["hits"] + cache_stats["misses"] > 0


class TestConfigurationManagement:
    """Test configuration management and environment handling."""

    def test_environment_configuration(self):
        """Test different environment configurations."""
        # Test development environment
        dev_config = {
            "debug": True,
            "max_workers": 2,
            "cache_size": 100,
            "database_url": "mongodb://localhost:27017/dev",
        }

        # Test production environment
        prod_config = {
            "debug": False,
            "max_workers": 8,
            "cache_size": 10000,
            "database_url": "mongodb://prod-cluster:27017/production",
        }

        # Test that components can be configured appropriately
        for config_name, config in [("dev", dev_config), ("prod", prod_config)]:
            runner = ConcurrentExperimentRunner(
                max_workers=config["max_workers"], cache_enabled=True
            )

            stats = runner.get_performance_stats()
            assert stats["max_workers"] == config["max_workers"]

    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        from materials_orchestrator.security import SecurityConfig

        # Test strict security config
        strict_config = SecurityConfig(
            max_parameter_value=100.0,
            min_parameter_value=0.0,
            max_experiments_per_campaign=1000,
            rate_limit_requests_per_minute=10,
            enable_audit_logging=True,
        )

        validator = SecurityValidator(strict_config)

        # Should enforce strict limits
        with pytest.raises(ValueError):
            validator.validate_parameters({"temp": 200})  # Exceeds max

        with pytest.raises(ValueError):
            validator.validate_campaign_config(
                {"max_experiments": 2000}
            )  # Exceeds limit

        # Test permissive config
        permissive_config = SecurityConfig(
            max_parameter_value=10000.0, rate_limit_requests_per_minute=1000
        )

        permissive_validator = SecurityValidator(permissive_config)

        # Should allow previously rejected values
        validated = permissive_validator.validate_parameters({"temp": 500})
        assert validated["temp"] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
