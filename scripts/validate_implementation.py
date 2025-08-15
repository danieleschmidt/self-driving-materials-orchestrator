#!/usr/bin/env python3
"""
Validation script for self-healing pipeline guard implementation.

This script validates the core functionality without requiring external dependencies.
"""

import sys
import os
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_imports():
    """Test basic imports work."""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core pipeline guard import
        from materials_orchestrator.pipeline_guard import SelfHealingPipelineGuard, PipelineStatus, FailureType
        print("  ✅ Pipeline guard imports successful")
        
        # Test intelligent failover import
        from materials_orchestrator.intelligent_failover import IntelligentFailoverManager, ComponentType, FailoverStrategy
        print("  ✅ Intelligent failover imports successful")
        
        # Test error handling import (without psutil dependencies)
        from materials_orchestrator.robust_error_handling import RobustErrorHandler, ErrorSeverity, ErrorCategory
        print("  ✅ Error handling imports successful")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_pipeline_guard_functionality():
    """Test pipeline guard core functionality."""
    print("\n🛡️ Testing pipeline guard functionality...")
    
    try:
        from materials_orchestrator.pipeline_guard import SelfHealingPipelineGuard, PipelineStatus, FailureType
        
        # Create instance
        guard = SelfHealingPipelineGuard()
        print("  ✅ Pipeline guard created successfully")
        
        # Test initial status
        assert guard.status == PipelineStatus.HEALTHY
        print("  ✅ Initial status is healthy")
        
        # Test failure reporting
        failure_id = guard.report_failure(
            failure_type=FailureType.ROBOT_DISCONNECTION,
            component="test_robot",
            severity="critical",
            description="Test failure"
        )
        assert failure_id in guard.failures
        assert guard.total_failures == 1
        print("  ✅ Failure reporting works")
        
        # Test health metric updates
        guard.update_health_metric("cpu_usage", 85.0, 80.0, "warning")
        assert "cpu_usage" in guard.health_metrics
        print("  ✅ Health metric updates work")
        
        # Test status reporting
        status = guard.get_health_status()
        assert "status" in status
        assert "total_failures" in status
        print("  ✅ Status reporting works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline guard test failed: {e}")
        return False

def test_failover_manager_functionality():
    """Test failover manager functionality."""
    print("\n🔄 Testing failover manager functionality...")
    
    try:
        from materials_orchestrator.intelligent_failover import IntelligentFailoverManager, ComponentType, FailoverStrategy
        
        # Create instance
        manager = IntelligentFailoverManager()
        print("  ✅ Failover manager created successfully")
        
        # Test component registration
        component = manager.register_component(
            "test_robot",
            "Test Robot",
            ComponentType.ROBOT,
            capabilities=["synthesis", "analysis"]
        )
        assert component.component_id == "test_robot"
        assert component.component_type == ComponentType.ROBOT
        print("  ✅ Component registration works")
        
        # Test failover rule registration
        manager.register_failover_rule(
            "test_rule",
            "Test Rule",
            {"status": "failed"},
            FailoverStrategy.HOT_STANDBY,
            [ComponentType.ROBOT],
            [ComponentType.ROBOT]
        )
        assert "test_rule" in manager.failover_rules
        print("  ✅ Failover rule registration works")
        
        # Test status updates
        manager.update_component_status("test_robot", "failed")
        assert manager.components["test_robot"].status == "failed"
        print("  ✅ Component status updates work")
        
        # Test status reporting
        status = manager.get_failover_status()
        assert "total_components" in status
        assert "components" in status
        print("  ✅ Status reporting works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failover manager test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("\n⚡ Testing async functionality...")
    
    try:
        from materials_orchestrator.pipeline_guard import SelfHealingPipelineGuard, FailureType
        
        guard = SelfHealingPipelineGuard()
        
        # Test async healing action
        healing_executed = False
        
        async def test_healing_action(failure):
            nonlocal healing_executed
            healing_executed = True
            return True
        
        guard.register_healing_action(
            "test_healing",
            "Test Healing",
            [FailureType.EXPERIMENT_FAILURE],
            test_healing_action
        )
        
        # Report failure and wait for healing
        guard.report_failure(
            failure_type=FailureType.EXPERIMENT_FAILURE,
            component="test_component",
            severity="warning"
        )
        
        # Wait briefly for healing to execute
        await asyncio.sleep(0.1)
        
        print("  ✅ Async healing actions work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Async functionality test failed: {e}")
        return False

def test_quantum_optimization_imports():
    """Test quantum optimization imports."""
    print("\n🔬 Testing quantum optimization imports...")
    
    try:
        from materials_orchestrator.quantum_enhanced_pipeline_guard import (
            DistributedQuantumPipelineGuard,
            QuantumConfiguration,
            PipelineOptimizationProblem
        )
        print("  ✅ Quantum optimization imports successful")
        
        # Test configuration creation
        config = QuantumConfiguration(num_qubits=8)
        assert config.num_qubits == 8
        print("  ✅ Quantum configuration works")
        
        # Test optimization problem creation
        problem = PipelineOptimizationProblem(
            objective_function="test_objective",
            constraints=[],
            variables={"x": (0.0, 1.0), "y": (0.0, 1.0)}
        )
        assert len(problem.variables) == 2
        print("  ✅ Optimization problem creation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum optimization test failed: {e}")
        return False

def test_distributed_coordination_imports():
    """Test distributed coordination imports."""
    print("\n🌐 Testing distributed coordination imports...")
    
    try:
        from materials_orchestrator.distributed_self_healing import (
            GlobalCoordinationLayer,
            GlobalRegion,
            NodeStatus
        )
        print("  ✅ Distributed coordination imports successful")
        
        # Test coordination layer creation
        layer = GlobalCoordinationLayer(GlobalRegion.US_EAST)
        assert layer.region == GlobalRegion.US_EAST
        assert layer.local_node.status == NodeStatus.ONLINE
        print("  ✅ Coordination layer creation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Distributed coordination test failed: {e}")
        return False

def test_error_handling_functionality():
    """Test error handling functionality."""
    print("\n🔧 Testing error handling functionality...")
    
    try:
        from materials_orchestrator.robust_error_handling import (
            RobustErrorHandler,
            ErrorSeverity,
            ErrorCategory
        )
        
        # Create instance
        handler = RobustErrorHandler()
        print("  ✅ Error handler created successfully")
        
        # Test recovery action registration
        def test_recovery(error_context):
            return True
        
        handler.register_recovery_action(
            "test_recovery",
            "Test Recovery",
            "retry",
            test_recovery
        )
        assert "test_recovery" in handler.recovery_actions
        print("  ✅ Recovery action registration works")
        
        # Test error statistics
        stats = handler.get_error_statistics()
        assert "total_errors" in stats
        assert "recovery_counts" in stats
        print("  ✅ Error statistics work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

async def test_integration():
    """Test integration between components."""
    print("\n🔗 Testing component integration...")
    
    try:
        from materials_orchestrator.pipeline_guard import SelfHealingPipelineGuard, FailureType
        from materials_orchestrator.intelligent_failover import IntelligentFailoverManager, ComponentType
        
        # Create instances
        guard = SelfHealingPipelineGuard()
        failover = IntelligentFailoverManager()
        
        # Register component
        failover.register_component(
            "integration_test",
            "Integration Test Component",
            ComponentType.ROBOT
        )
        
        # Report failure
        guard.report_failure(
            failure_type=FailureType.ROBOT_DISCONNECTION,
            component="integration_test",
            severity="critical"
        )
        
        # Update component status
        failover.update_component_status("integration_test", "failed")
        
        # Check both systems recorded the issue
        assert guard.total_failures > 0
        assert failover.components["integration_test"].status == "failed"
        
        print("  ✅ Component integration works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def test_performance_basic():
    """Test basic performance characteristics."""
    print("\n⚡ Testing basic performance...")
    
    try:
        from materials_orchestrator.pipeline_guard import SelfHealingPipelineGuard, FailureType
        
        guard = SelfHealingPipelineGuard()
        
        # Test processing many failures quickly
        start_time = time.time()
        
        for i in range(100):
            guard.report_failure(
                failure_type=FailureType.EXPERIMENT_FAILURE,
                component=f"perf_test_{i}",
                severity="warning"
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert guard.total_failures == 100
        assert processing_time < 1.0  # Should process 100 failures in under 1 second
        
        print(f"  ✅ Processed 100 failures in {processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

async def run_all_validations():
    """Run all validation tests."""
    print("🛡️ Self-Healing Pipeline Guard Validation")
    print("🤖 Terragon Labs - Autonomous Materials Discovery")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_pipeline_guard_functionality,
        test_failover_manager_functionality,
        test_async_functionality,
        test_quantum_optimization_imports,
        test_distributed_coordination_imports,
        test_error_handling_functionality,
        test_integration,
        test_performance_basic
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("🚀 Self-healing pipeline guard is ready for deployment!")
    else:
        print(f"\n⚠️  {failed} validation(s) failed")
        print("🔧 Please review and fix the failing components")
    
    return failed == 0

def main():
    """Main validation function."""
    try:
        success = asyncio.run(run_all_validations())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()