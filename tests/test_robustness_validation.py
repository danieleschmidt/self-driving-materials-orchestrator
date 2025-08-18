"""Generation 2 - Robustness and reliability validation tests."""

import threading
import time
from unittest.mock import patch

import pytest

from materials_orchestrator import AutonomousLab, MaterialsObjective
from materials_orchestrator.error_handling import ValidationError


class TestErrorHandling:
    """Test comprehensive error handling and recovery."""

    def test_experiment_failure_recovery(self):
        """Test system recovers from experiment failures."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )
        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        # Mock experiment failure
        with patch(
            "materials_orchestrator.core.simulate_perovskite_experiment"
        ) as mock_exp:
            mock_exp.side_effect = [
                Exception("Equipment failure"),
                {"band_gap": 1.4, "efficiency": 20, "stability": 0.8},
            ]

            campaign = lab.run_campaign(objective, param_space, max_experiments=5)

            # Should still produce results despite failures
            assert campaign.total_experiments > 0
            assert campaign.success_rate < 1.0  # Some failures expected


class TestValidationRobustness:
    """Test input validation and data integrity."""

    def test_invalid_parameter_space(self):
        """Test handling of invalid parameter spaces."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )

        # Invalid parameter space (empty)
        with pytest.raises(ValueError):
            lab.run_campaign(objective, {}, max_experiments=5)

        # Invalid parameter ranges
        with pytest.raises(ValueError):
            lab.run_campaign(objective, {"temp": (300, 100)}, max_experiments=5)

    def test_data_integrity_validation(self):
        """Test experiment data validation."""
        lab = AutonomousLab()

        # Test parameter validation
        params = {"temperature": 150, "concentration": 1.0}
        result = lab._validate_experiment_parameters(params)
        assert result is True

        # Test invalid parameters
        invalid_params = {"temperature": -100}  # Invalid temperature
        with pytest.raises(ValidationError):
            lab._validate_experiment_parameters(invalid_params)


class TestConcurrencyRobustness:
    """Test concurrent execution reliability."""

    def test_concurrent_experiments(self):
        """Test system handles concurrent experiments safely."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )
        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        # Run multiple campaigns concurrently
        results = []
        threads = []

        def run_campaign():
            result = lab.run_campaign(objective, param_space, max_experiments=3)
            results.append(result)

        for _ in range(3):
            thread = threading.Thread(target=run_campaign)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All campaigns should complete successfully
        assert len(results) == 3
        for result in results:
            assert result.total_experiments > 0


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_memory_management(self):
        """Test system properly manages memory during experiments."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )
        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        # Run multiple campaigns to test memory cleanup
        for i in range(5):
            campaign = lab.run_campaign(objective, param_space, max_experiments=2)
            assert campaign.total_experiments > 0

            # Force cleanup
            del campaign

    def test_database_connection_resilience(self):
        """Test database connection handling."""
        lab = AutonomousLab(database_url="mongodb://invalid:27017/")

        # Should handle invalid database gracefully
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )
        param_space = {"temperature": (100, 300)}

        # Should still work without database
        campaign = lab.run_campaign(objective, param_space, max_experiments=2)
        assert campaign.total_experiments > 0


class TestPerformanceRobustness:
    """Test system performance under stress."""

    @pytest.mark.slow
    def test_large_parameter_space(self):
        """Test handling of large parameter spaces."""
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )

        # Large parameter space
        param_space = {f"param_{i}": (0, 100) for i in range(20)}

        start_time = time.time()
        campaign = lab.run_campaign(objective, param_space, max_experiments=10)
        duration = time.time() - start_time

        # Should complete within reasonable time
        assert campaign.total_experiments > 0
        assert duration < 60  # Should finish within 1 minute

    def test_monitoring_under_load(self):
        """Test health monitoring during heavy load."""
        lab = AutonomousLab()

        # Ensure monitoring is active
        assert hasattr(lab, "health_monitor")
        if lab.health_monitor:
            assert lab.health_monitor.monitoring_active

        objective = MaterialsObjective(
            target_property="band_gap", target_range=(1.2, 1.6)
        )
        param_space = {"temperature": (100, 300), "concentration": (0.1, 2.0)}

        # Run campaign while monitoring
        campaign = lab.run_campaign(objective, param_space, max_experiments=5)

        # Monitoring should remain stable
        if lab.health_monitor:
            health_status = lab.health_monitor.get_overall_health()
            assert health_status is not None
