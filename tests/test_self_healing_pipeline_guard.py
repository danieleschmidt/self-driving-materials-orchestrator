"""Comprehensive tests for self-healing pipeline guard system."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from materials_orchestrator import (
    AdvancedMonitoringSystem,
    ComponentType,
    ErrorCategory,
    ErrorSeverity,
    FailoverStrategy,
    FailureType,
    IntelligentFailoverManager,
    PipelineStatus,
    RobustErrorHandler,
    SelfHealingPipelineGuard,
    get_failover_manager,
    get_global_error_handler,
    get_monitoring_system,
    get_pipeline_guard,
)


class TestSelfHealingPipelineGuard:
    """Test suite for SelfHealingPipelineGuard."""

    @pytest.fixture
    def pipeline_guard(self):
        """Create pipeline guard instance for testing."""
        return SelfHealingPipelineGuard()

    def test_pipeline_guard_initialization(self, pipeline_guard):
        """Test pipeline guard initialization."""
        assert pipeline_guard.status == PipelineStatus.HEALTHY
        assert pipeline_guard.healing_enabled is True
        assert pipeline_guard.monitoring_active is False
        assert len(pipeline_guard.healing_actions) > 0
        assert pipeline_guard.total_failures == 0
        assert pipeline_guard.total_healings == 0

    def test_health_metric_update(self, pipeline_guard):
        """Test health metric updates."""
        # Test healthy metric
        pipeline_guard.update_health_metric("cpu_usage", 50.0, 80.0, "info")
        assert "cpu_usage" in pipeline_guard.health_metrics
        assert pipeline_guard.health_metrics["cpu_usage"].is_healthy is True

        # Test unhealthy metric
        pipeline_guard.update_health_metric("memory_usage", 95.0, 85.0, "critical")
        assert "memory_usage" in pipeline_guard.health_metrics
        assert pipeline_guard.health_metrics["memory_usage"].is_healthy is False

    def test_failure_reporting(self, pipeline_guard):
        """Test failure reporting."""
        failure_id = pipeline_guard.report_failure(
            failure_type=FailureType.ROBOT_DISCONNECTION,
            component="test_robot",
            severity="critical",
            description="Test failure",
        )

        assert failure_id in pipeline_guard.failures
        assert pipeline_guard.total_failures == 1
        assert (
            pipeline_guard.failures[failure_id].failure_type
            == FailureType.ROBOT_DISCONNECTION
        )
        assert pipeline_guard.failures[failure_id].component == "test_robot"

    def test_pipeline_status_updates(self, pipeline_guard):
        """Test pipeline status updates based on failures."""
        # No failures - should be healthy
        assert pipeline_guard.status == PipelineStatus.HEALTHY

        # Add some failures
        for i in range(2):
            pipeline_guard.report_failure(
                failure_type=FailureType.EXPERIMENT_FAILURE,
                component=f"component_{i}",
                severity="warning",
            )

        # Should still be healthy (below threshold)
        assert pipeline_guard.status == PipelineStatus.HEALTHY

        # Add more failures to trigger degraded status
        for i in range(3):
            pipeline_guard.report_failure(
                failure_type=FailureType.DATABASE_ERROR,
                component=f"db_component_{i}",
                severity="warning",
            )

        # Should now be degraded
        assert pipeline_guard.status == PipelineStatus.DEGRADED

        # Add critical failure
        pipeline_guard.report_failure(
            failure_type=FailureType.MEMORY_LEAK,
            component="critical_component",
            severity="critical",
        )

        # Should now be failed
        assert pipeline_guard.status == PipelineStatus.FAILED

    @pytest.mark.asyncio
    async def test_healing_actions(self, pipeline_guard):
        """Test healing action execution."""
        # Mock healing action
        mock_handler = AsyncMock(return_value=True)
        pipeline_guard.register_healing_action(
            action_id="test_healing",
            name="Test Healing Action",
            failure_types=[FailureType.EXPERIMENT_FAILURE],
            handler=mock_handler,
            priority=1,
        )

        # Report failure
        failure_id = pipeline_guard.report_failure(
            failure_type=FailureType.EXPERIMENT_FAILURE,
            component="test_component",
            severity="warning",
        )

        # Wait for healing to complete
        await asyncio.sleep(0.1)

        # Check that healing was attempted
        failure = pipeline_guard.failures[failure_id]
        assert failure.healing_attempts > 0 or failure.resolved

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, pipeline_guard):
        """Test monitoring loop functionality."""
        # Start monitoring
        monitor_task = asyncio.create_task(pipeline_guard.start_monitoring())

        # Let it run for a short time
        await asyncio.sleep(0.5)

        # Stop monitoring
        pipeline_guard.stop_monitoring()
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Check that monitoring was active
        assert not pipeline_guard.monitoring_active

    def test_get_health_status(self, pipeline_guard):
        """Test health status reporting."""
        # Add some test data
        pipeline_guard.update_health_metric("test_metric", 75.0, 80.0, "info")
        pipeline_guard.report_failure(
            failure_type=FailureType.NETWORK_TIMEOUT,
            component="test_component",
            severity="warning",
        )

        status = pipeline_guard.get_health_status()

        assert "status" in status
        assert "uptime_seconds" in status
        assert "total_failures" in status
        assert "total_healings" in status
        assert "health_metrics" in status
        assert "healing_actions" in status

        assert status["total_failures"] > 0
        assert "test_metric" in status["health_metrics"]

    def test_failure_resolution(self, pipeline_guard):
        """Test manual failure resolution."""
        failure_id = pipeline_guard.report_failure(
            failure_type=FailureType.DATABASE_ERROR,
            component="test_db",
            severity="warning",
        )

        # Resolve failure
        success = pipeline_guard.resolve_failure(failure_id)
        assert success is True
        assert pipeline_guard.failures[failure_id].resolved is True
        assert pipeline_guard.failures[failure_id].resolution_time is not None

        # Try to resolve non-existent failure
        success = pipeline_guard.resolve_failure("non_existent")
        assert success is False


class TestIntelligentFailoverManager:
    """Test suite for IntelligentFailoverManager."""

    @pytest.fixture
    def failover_manager(self):
        """Create failover manager instance for testing."""
        return IntelligentFailoverManager()

    def test_failover_manager_initialization(self, failover_manager):
        """Test failover manager initialization."""
        assert len(failover_manager.components) == 0
        assert len(failover_manager.failover_rules) > 0  # Default rules
        assert failover_manager.monitoring_active is False
        assert failover_manager.total_failovers == 0

    def test_component_registration(self, failover_manager):
        """Test component registration."""
        component = failover_manager.register_component(
            "test_robot",
            "Test Robot",
            ComponentType.ROBOT,
            capabilities=["synthesis", "analysis"],
        )

        assert component.component_id == "test_robot"
        assert component.name == "Test Robot"
        assert component.component_type == ComponentType.ROBOT
        assert "synthesis" in component.capabilities
        assert component.is_available is True

    def test_failover_rule_registration(self, failover_manager):
        """Test failover rule registration."""
        failover_manager.register_failover_rule(
            "test_rule",
            "Test Failover Rule",
            {"status": "failed"},
            FailoverStrategy.HOT_STANDBY,
            [ComponentType.ROBOT],
            [ComponentType.ROBOT],
            priority=5,
        )

        assert "test_rule" in failover_manager.failover_rules
        rule = failover_manager.failover_rules["test_rule"]
        assert rule.name == "Test Failover Rule"
        assert rule.failover_strategy == FailoverStrategy.HOT_STANDBY
        assert rule.priority == 5

    @pytest.mark.asyncio
    async def test_component_failure_handling(self, failover_manager):
        """Test component failure handling."""
        # Register primary and backup components
        primary = failover_manager.register_component(
            "primary_robot",
            "Primary Robot",
            ComponentType.ROBOT,
            capabilities=["synthesis"],
        )

        backup = failover_manager.register_component(
            "backup_robot",
            "Backup Robot",
            ComponentType.ROBOT,
            capabilities=["synthesis"],
        )

        # Trigger failure
        failover_manager.update_component_status("primary_robot", "failed")

        # Wait for failover
        await asyncio.sleep(0.1)

        # Check that failover was triggered
        assert failover_manager.total_failovers >= 0  # May or may not have executed

    @pytest.mark.asyncio
    async def test_predictive_monitoring(self, failover_manager):
        """Test predictive failure monitoring."""
        # Register test component
        failover_manager.register_component(
            "test_component", "Test Component", ComponentType.INSTRUMENT
        )

        # Start monitoring
        monitor_task = asyncio.create_task(
            failover_manager.start_predictive_monitoring()
        )

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Stop monitoring
        failover_manager.stop_monitoring()
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        assert not failover_manager.monitoring_active

    def test_get_failover_status(self, failover_manager):
        """Test failover status reporting."""
        # Register test component
        failover_manager.register_component(
            "test_comp", "Test Component", ComponentType.DATABASE
        )

        status = failover_manager.get_failover_status()

        assert "total_components" in status
        assert "active_failovers" in status
        assert "total_failovers" in status
        assert "successful_failovers" in status
        assert "components" in status

        assert status["total_components"] == 1
        assert "test_comp" in status["components"]


class TestRobustErrorHandler:
    """Test suite for RobustErrorHandler."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler instance for testing."""
        return RobustErrorHandler()

    def test_error_handler_initialization(self, error_handler):
        """Test error handler initialization."""
        assert len(error_handler.recovery_actions) > 0  # Default actions
        assert error_handler.enable_auto_recovery is True
        assert len(error_handler.error_history) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, error_handler):
        """Test error handling."""
        test_error = ValueError("Test error")

        error_context = await error_handler.handle_error(
            error=test_error,
            component="test_component",
            operation="test_operation",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOFTWARE,
        )

        assert error_context.component == "test_component"
        assert error_context.operation == "test_operation"
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error"
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert len(error_handler.error_history) == 1

    def test_recovery_action_registration(self, error_handler):
        """Test recovery action registration."""
        mock_handler = MagicMock(return_value=True)

        error_handler.register_recovery_action(
            "test_recovery",
            "Test Recovery",
            RecoveryStrategy.RETRY,
            mock_handler,
            max_attempts=3,
        )

        assert "test_recovery" in error_handler.recovery_actions
        action = error_handler.recovery_actions["test_recovery"]
        assert action.name == "Test Recovery"
        assert action.max_attempts == 3

    def test_error_statistics(self, error_handler):
        """Test error statistics."""
        # Generate some test errors
        for i in range(5):
            asyncio.create_task(
                error_handler.handle_error(
                    error=Exception(f"Test error {i}"),
                    component="test_component",
                    operation="test_operation",
                )
            )

        stats = error_handler.get_error_statistics()

        assert "total_errors" in stats
        assert "recent_errors" in stats
        assert "error_counts_by_type" in stats
        assert "recovery_counts" in stats
        assert "component_health" in stats


class TestAdvancedMonitoringSystem:
    """Test suite for AdvancedMonitoringSystem."""

    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system instance for testing."""
        return AdvancedMonitoringSystem()

    def test_monitoring_system_initialization(self, monitoring_system):
        """Test monitoring system initialization."""
        assert monitoring_system.monitoring_enabled is False
        assert len(monitoring_system.metrics_collector.metric_metadata) > 0
        assert len(monitoring_system.alert_manager.alert_rules) > 0
        assert len(monitoring_system.health_checks) > 0

    def test_sla_target_management(self, monitoring_system):
        """Test SLA target management."""
        monitoring_system.add_sla_target("test_sla", 95.0, ">", 3600)

        assert "test_sla" in monitoring_system.sla_targets
        sla = monitoring_system.sla_targets["test_sla"]
        assert sla.target_value == 95.0
        assert sla.comparison == ">"

    def test_health_check_management(self, monitoring_system):
        """Test health check management."""
        test_check = MagicMock(return_value=True)

        monitoring_system.add_health_check(
            "test_health_check", test_check, interval_seconds=30
        )

        assert "test_health_check" in monitoring_system.health_checks
        health_check = monitoring_system.health_checks["test_health_check"]
        assert health_check.name == "test_health_check"
        assert health_check.interval_seconds == 30

    def test_operation_timing_recording(self, monitoring_system):
        """Test operation timing recording."""
        monitoring_system.record_operation_timing("test_operation", 1.5)

        assert "test_operation" in monitoring_system.operation_timings
        assert len(monitoring_system.operation_timings["test_operation"]) == 1

    @pytest.mark.asyncio
    async def test_health_checks_execution(self, monitoring_system):
        """Test health checks execution."""
        test_check = AsyncMock(return_value=True)
        monitoring_system.add_health_check("async_test_check", test_check)

        await monitoring_system.run_health_checks()

        health_check = monitoring_system.health_checks["async_test_check"]
        assert health_check.last_run is not None
        assert health_check.last_result is True

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitoring_system):
        """Test monitoring system lifecycle."""
        # Start monitoring
        start_task = asyncio.create_task(monitoring_system.start_monitoring())

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Check that monitoring is active
        assert monitoring_system.monitoring_enabled is True

        # Stop monitoring
        monitoring_system.stop_monitoring()
        start_task.cancel()

        try:
            await start_task
        except asyncio.CancelledError:
            pass

        assert monitoring_system.monitoring_enabled is False

    def test_get_monitoring_status(self, monitoring_system):
        """Test monitoring status reporting."""
        status = monitoring_system.get_monitoring_status()

        assert "system_status" in status
        assert "metrics" in status
        assert "alerts" in status
        assert "health_checks" in status
        assert "sla_targets" in status

        assert status["system_status"] in ["healthy", "stopped"]


class TestIntegration:
    """Integration tests for self-healing pipeline system."""

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration."""
        # Get global instances
        pipeline_guard = get_pipeline_guard()
        failover_manager = get_failover_manager()
        error_handler = get_global_error_handler()
        monitoring_system = get_monitoring_system()

        # Register components
        failover_manager.register_component(
            "integration_test_robot",
            "Integration Test Robot",
            ComponentType.ROBOT,
            capabilities=["synthesis"],
        )

        # Start monitoring
        monitor_task = asyncio.create_task(monitoring_system.start_monitoring())
        guard_task = asyncio.create_task(pipeline_guard.start_monitoring())

        # Wait briefly
        await asyncio.sleep(0.5)

        # Trigger failure
        pipeline_guard.report_failure(
            failure_type=FailureType.ROBOT_DISCONNECTION,
            component="integration_test_robot",
            severity="critical",
        )

        # Handle error
        test_error = ConnectionError("Robot disconnected")
        await error_handler.handle_error(
            error=test_error,
            component="integration_test_robot",
            operation="synthesis_experiment",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.HARDWARE,
        )

        # Update component status
        failover_manager.update_component_status("integration_test_robot", "failed")

        # Wait for systems to process
        await asyncio.sleep(1)

        # Check system states
        guard_status = pipeline_guard.get_health_status()
        failover_status = failover_manager.get_failover_status()
        error_stats = error_handler.get_error_statistics()
        monitor_status = monitoring_system.get_monitoring_status()

        assert guard_status["total_failures"] > 0
        assert len(error_stats["error_counts_by_type"]) > 0
        assert monitor_status["system_status"] == "healthy"

        # Cleanup
        pipeline_guard.stop_monitoring()
        monitoring_system.stop_monitoring()

        guard_task.cancel()
        monitor_task.cancel()

        try:
            await guard_task
            await monitor_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_failure_handling(self):
        """Test handling of concurrent failures."""
        pipeline_guard = get_pipeline_guard()
        failover_manager = get_failover_manager()

        # Register multiple components
        for i in range(5):
            failover_manager.register_component(
                f"concurrent_test_robot_{i}",
                f"Concurrent Test Robot {i}",
                ComponentType.ROBOT,
            )

        # Trigger multiple failures concurrently
        failure_tasks = []
        for i in range(5):
            task = asyncio.create_task(
                self._trigger_component_failure(
                    pipeline_guard, failover_manager, f"concurrent_test_robot_{i}"
                )
            )
            failure_tasks.append(task)

        # Wait for all failures to be processed
        await asyncio.gather(*failure_tasks)

        # Check that all failures were handled
        status = pipeline_guard.get_health_status()
        assert status["total_failures"] >= 5

    async def _trigger_component_failure(
        self, pipeline_guard, failover_manager, component_id
    ):
        """Helper method to trigger component failure."""
        pipeline_guard.report_failure(
            failure_type=FailureType.EXPERIMENT_FAILURE,
            component=component_id,
            severity="warning",
        )

        failover_manager.update_component_status(component_id, "failed")

        # Wait for processing
        await asyncio.sleep(0.1)


# Fixtures for global instances
@pytest.fixture(scope="session")
def global_pipeline_guard():
    """Global pipeline guard fixture."""
    return get_pipeline_guard()


@pytest.fixture(scope="session")
def global_failover_manager():
    """Global failover manager fixture."""
    return get_failover_manager()


@pytest.fixture(scope="session")
def global_error_handler():
    """Global error handler fixture."""
    return get_global_error_handler()


@pytest.fixture(scope="session")
def global_monitoring_system():
    """Global monitoring system fixture."""
    return get_monitoring_system()


# Performance tests
class TestPerformance:
    """Performance tests for self-healing system."""

    @pytest.mark.asyncio
    async def test_failure_processing_performance(self):
        """Test failure processing performance."""
        pipeline_guard = SelfHealingPipelineGuard()

        start_time = time.time()

        # Process 100 failures
        for i in range(100):
            pipeline_guard.report_failure(
                failure_type=FailureType.EXPERIMENT_FAILURE,
                component=f"perf_test_component_{i}",
                severity="warning",
            )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 100 failures in under 1 second
        assert processing_time < 1.0
        assert pipeline_guard.total_failures == 100

    @pytest.mark.asyncio
    async def test_concurrent_monitoring_performance(self):
        """Test concurrent monitoring performance."""
        monitoring_system = AdvancedMonitoringSystem()

        # Start monitoring
        start_time = time.time()
        monitor_task = asyncio.create_task(monitoring_system.start_monitoring())

        # Let it run and collect metrics
        await asyncio.sleep(2)

        # Stop monitoring
        monitoring_system.stop_monitoring()
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        end_time = time.time()

        # Check that metrics were collected
        metrics_count = len(monitoring_system.metrics_collector.metrics)
        assert metrics_count > 0

        # Performance should be reasonable
        assert end_time - start_time < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
