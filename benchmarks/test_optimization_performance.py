"""Performance benchmarks for optimization algorithms."""

import pytest
import time
from materials_orchestrator.core import AutonomousLab, MaterialsObjective
from materials_orchestrator.planners import BayesianPlanner


class TestOptimizationPerformance:
    """Benchmark optimization algorithm performance."""
    
    def test_bayesian_planner_speed(self, benchmark):
        """Benchmark Bayesian planner initialization speed."""
        def create_planner():
            return BayesianPlanner(
                acquisition_function="expected_improvement",
                batch_size=5
            )
        
        planner = benchmark(create_planner)
        assert planner is not None
    
    def test_campaign_simulation_speed(self, benchmark):
        """Benchmark full campaign simulation speed."""
        def run_small_campaign():
            objective = MaterialsObjective(
                target_property="band_gap",
                target_range=(1.2, 1.6),
                material_system="perovskites"
            )
            
            lab = AutonomousLab()
            return lab.run_campaign(
                objective, 
                initial_samples=5, 
                max_experiments=10
            )
        
        result = benchmark(run_small_campaign)
        assert result.total_experiments > 0
    
    @pytest.mark.slow
    def test_large_parameter_space_performance(self, benchmark):
        """Benchmark performance with large parameter spaces."""
        def create_large_lab():
            return AutonomousLab(
                robots=["robot_" + str(i) for i in range(10)],
                instruments=["instrument_" + str(i) for i in range(20)]
            )
        
        lab = benchmark(create_large_lab)
        assert len(lab.robots) == 10
        assert len(lab.instruments) == 20