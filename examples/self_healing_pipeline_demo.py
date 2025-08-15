#!/usr/bin/env python3
"""
Self-Healing Pipeline Guard Demo

Demonstrates the self-healing pipeline guard capabilities with intelligent
failover and predictive failure detection for materials discovery pipelines.
"""

import asyncio
import logging
import time
import random
from datetime import datetime

from materials_orchestrator import (
    AutonomousLab,
    MaterialsObjective,
    get_pipeline_guard,
    get_failover_manager,
    SelfHealingPipelineGuard,
    IntelligentFailoverManager,
    ComponentType,
    FailoverStrategy,
    PipelineStatus,
    FailureType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_pipeline_guard():
    """Demonstrate self-healing pipeline guard capabilities."""
    print("🛡️ Self-Healing Pipeline Guard Demo")
    print("=" * 50)
    
    # Get pipeline guard instance
    guard = get_pipeline_guard()
    
    print(f"Initial pipeline status: {guard.status.value}")
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(guard.start_monitoring())
    
    print("\n⚡ Simulating pipeline failures and self-healing...")
    
    # Simulate robot disconnection
    print("\n1. Simulating robot disconnection...")
    failure_id = guard.report_failure(
        failure_type=FailureType.ROBOT_DISCONNECTION,
        component="synthesis_robot_1",
        severity="critical",
        description="Robot lost network connection during experiment"
    )
    
    # Wait for healing
    await asyncio.sleep(3)
    
    # Simulate database error
    print("\n2. Simulating database connection error...")
    guard.report_failure(
        failure_type=FailureType.DATABASE_ERROR,
        component="experiment_db",
        severity="warning",
        description="Connection timeout to MongoDB"
    )
    
    await asyncio.sleep(2)
    
    # Simulate memory leak
    print("\n3. Simulating memory leak detection...")
    guard.update_health_metric("memory_usage", 92.5, 85.0, "critical")
    
    await asyncio.sleep(4)
    
    # Simulate performance degradation
    print("\n4. Simulating performance degradation...")
    guard.update_health_metric("experiment_success_rate", 25.0, 15.0, "warning")
    
    await asyncio.sleep(3)
    
    # Get health status
    print("\n📊 Pipeline Health Status:")
    status = guard.get_health_status()
    print(f"Status: {status['status']}")
    print(f"Total failures: {status['total_failures']}")
    print(f"Total healings: {status['total_healings']}")
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"Active failures: {status['active_failures']}")
    
    print("\n🏥 Health Metrics:")
    for name, metric in status['health_metrics'].items():
        health_icon = "✅" if metric['healthy'] else "❌"
        print(f"  {health_icon} {name}: {metric['value']:.1f} (threshold: {metric['threshold']})")
    
    print("\n🔧 Healing Actions:")
    for action_id, action in status['healing_actions'].items():
        print(f"  {action['name']}: {action['success_count']} successes, {action['failure_count']} failures")
    
    # Stop monitoring
    guard.stop_monitoring()
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass


async def demo_intelligent_failover():
    """Demonstrate intelligent failover capabilities."""
    print("\n\n🔄 Intelligent Failover Demo")
    print("=" * 50)
    
    # Get failover manager
    failover_manager = get_failover_manager()
    
    # Register components
    print("\n📝 Registering pipeline components...")
    
    # Register robots
    robot1 = failover_manager.register_component(
        "robot_synthesis_1",
        "Primary Synthesis Robot",
        ComponentType.ROBOT,
        capabilities=["liquid_handling", "heating", "stirring"]
    )
    
    robot2 = failover_manager.register_component(
        "robot_synthesis_2", 
        "Backup Synthesis Robot",
        ComponentType.ROBOT,
        capabilities=["liquid_handling", "heating", "stirring"]
    )
    
    # Register databases
    db1 = failover_manager.register_component(
        "mongodb_primary",
        "Primary MongoDB",
        ComponentType.DATABASE,
        capabilities=["read", "write", "indexing"]
    )
    
    db2 = failover_manager.register_component(
        "mongodb_replica",
        "MongoDB Replica",
        ComponentType.DATABASE,
        capabilities=["read", "write", "indexing"]
    )
    
    # Register instruments
    xrd1 = failover_manager.register_component(
        "xrd_instrument_1",
        "XRD Analyzer 1",
        ComponentType.INSTRUMENT,
        capabilities=["xrd_analysis", "phase_identification"]
    )
    
    xrd2 = failover_manager.register_component(
        "xrd_instrument_2",
        "XRD Analyzer 2", 
        ComponentType.INSTRUMENT,
        capabilities=["xrd_analysis", "phase_identification"]
    )
    
    print(f"Registered {len(failover_manager.components)} components")
    
    # Start predictive monitoring
    print("\n🔮 Starting predictive failure monitoring...")
    monitoring_task = asyncio.create_task(failover_manager.start_predictive_monitoring())
    
    # Simulate component failures
    print("\n⚠️ Simulating component failures...")
    
    # Simulate robot failure
    print("\n1. Robot failure simulation...")
    failover_manager.update_component_status("robot_synthesis_1", "failed")
    await asyncio.sleep(2)
    
    # Simulate database degradation
    print("\n2. Database degradation simulation...")
    failover_manager.update_component_status("mongodb_primary", "degraded", load=0.95)
    await asyncio.sleep(2)
    
    # Simulate instrument overload
    print("\n3. Instrument overload simulation...")
    failover_manager.update_component_status("xrd_instrument_1", "degraded", load=0.85)
    await asyncio.sleep(3)
    
    # Get failover status
    print("\n📊 Failover Status:")
    status = failover_manager.get_failover_status()
    print(f"Total components: {status['total_components']}")
    print(f"Active failovers: {status['active_failovers']}")
    print(f"Total failovers: {status['total_failovers']}")
    print(f"Success rate: {status['success_rate']:.1%}")
    print(f"Experiments saved: {status['experiments_saved']}")
    
    print("\n🖥️ Component Status:")
    for comp_id, comp in status['components'].items():
        status_icon = "✅" if comp['status'] == "healthy" else "⚠️" if comp['status'] == "degraded" else "❌"
        print(f"  {status_icon} {comp['name']}: {comp['status']} (load: {comp['load']:.1%})")
    
    print("\n📋 Recent Failover Events:")
    for event in status['recent_events'][-5:]:  # Last 5 events
        success_icon = "✅" if event['success'] else "❌"
        print(f"  {success_icon} {event['strategy']} failover: {event['failed_component']} -> {event['backup_component']} ({event['duration']:.2f}s)")
    
    # Stop monitoring
    failover_manager.stop_monitoring()
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass


async def demo_materials_discovery_with_self_healing():
    """Demonstrate materials discovery with self-healing pipeline."""
    print("\n\n🧪 Materials Discovery with Self-Healing Demo")
    print("=" * 60)
    
    # Create autonomous lab with self-healing
    lab = AutonomousLab()
    
    # Get pipeline guard and failover manager
    guard = get_pipeline_guard()
    failover_manager = get_failover_manager()
    
    # Define materials objective
    objective = MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),
        optimization_direction="target",
        material_system="perovskites"
    )
    
    # Define parameter space
    param_space = {
        "precursor_A_conc": (0.1, 2.0),
        "precursor_B_conc": (0.1, 2.0), 
        "temperature": (100, 300),
        "reaction_time": (1, 24)
    }
    
    print("🎯 Objective: Optimize perovskite band gap to 1.2-1.6 eV")
    print("🔬 Parameter space: 4 continuous variables")
    
    # Start self-healing monitoring
    guard_task = asyncio.create_task(guard.start_monitoring())
    failover_task = asyncio.create_task(failover_manager.start_predictive_monitoring())
    
    # Simulate discovery campaign with failures
    print("\n🚀 Starting discovery campaign with simulated failures...")
    
    experiments_completed = 0
    target_experiments = 20
    
    for i in range(target_experiments):
        # Simulate experiment
        experiment_params = {
            param: random.uniform(bounds[0], bounds[1])
            for param, bounds in param_space.items()
        }
        
        print(f"\n🧪 Experiment {i+1}/{target_experiments}")
        
        # Simulate random failures during experiments
        if random.random() < 0.15:  # 15% chance of failure
            failure_types = [
                FailureType.ROBOT_DISCONNECTION,
                FailureType.DATABASE_ERROR,
                FailureType.NETWORK_TIMEOUT,
                FailureType.EXPERIMENT_FAILURE
            ]
            
            failure_type = random.choice(failure_types)
            component = f"component_{random.randint(1, 5)}"
            
            print(f"  ⚠️ Failure detected: {failure_type.value} in {component}")
            
            guard.report_failure(
                failure_type=failure_type,
                component=component,
                severity="warning",
                description=f"Failure during experiment {i+1}"
            )
            
            # Wait for healing
            await asyncio.sleep(1)
            print("  🔧 Self-healing completed, continuing...")
        
        # Simulate experiment execution time
        await asyncio.sleep(0.2)
        
        # Simulate results
        band_gap = random.uniform(1.0, 2.0)
        success = objective.target_range[0] <= band_gap <= objective.target_range[1]
        
        if success:
            print(f"  ✅ Success! Band gap: {band_gap:.3f} eV")
        else:
            print(f"  📊 Band gap: {band_gap:.3f} eV")
        
        experiments_completed += 1
    
    print(f"\n🏆 Campaign completed! {experiments_completed}/{target_experiments} experiments")
    
    # Final status
    guard_status = guard.get_health_status()
    failover_status = failover_manager.get_failover_status()
    
    print(f"\n📊 Final Pipeline Health:")
    print(f"  Pipeline status: {guard_status['status']}")
    print(f"  Total failures handled: {guard_status['total_failures']}")
    print(f"  Self-healing actions: {guard_status['total_healings']}")
    print(f"  Failovers executed: {failover_status['total_failovers']}")
    print(f"  System uptime: {guard_status['uptime_seconds']:.1f} seconds")
    print(f"  Overall reliability: {(1 - guard_status['total_failures'] / max(target_experiments, 1)):.1%}")
    
    # Stop monitoring
    guard.stop_monitoring()
    failover_manager.stop_monitoring()
    
    guard_task.cancel()
    failover_task.cancel()
    
    try:
        await guard_task
        await failover_task
    except asyncio.CancelledError:
        pass


async def main():
    """Run all self-healing pipeline demos."""
    print("🛡️ Self-Healing Pipeline Guard System Demo")
    print("🤖 Terragon Labs - Autonomous Materials Discovery")
    print("=" * 60)
    
    try:
        # Run pipeline guard demo
        await demo_pipeline_guard()
        
        # Run intelligent failover demo  
        await demo_intelligent_failover()
        
        # Run materials discovery with self-healing
        await demo_materials_discovery_with_self_healing()
        
        print("\n\n🎉 All demos completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("  ✅ Self-healing pipeline guard with automatic failure detection")
        print("  ✅ Intelligent failover with predictive failure analysis") 
        print("  ✅ Multiple failover strategies (hot standby, load balancing, etc.)")
        print("  ✅ Real-time health monitoring and metrics")
        print("  ✅ Integration with materials discovery workflows")
        print("  ✅ Comprehensive logging and status reporting")
        
        print("\n🚀 Ready for production deployment!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())