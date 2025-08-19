#!/usr/bin/env python3
"""
Ultra-High Performance Materials Discovery - Generation 3

Demonstrates the complete Generation 3 optimized implementation with:
- Ultra-high performance caching with intelligent eviction
- Distributed load balancing and auto-scaling
- Concurrent experiment processing
- Performance optimization and resource pooling
- Production-scale throughput capabilities
"""

import asyncio
import logging
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials_orchestrator import AutonomousLab, MaterialsObjective, BayesianPlanner
from materials_orchestrator.enhanced_validation import create_robust_validation_system
from materials_orchestrator.advanced_security import create_advanced_security_system, SecurityLevel
from materials_orchestrator.comprehensive_monitoring import create_comprehensive_monitoring
from materials_orchestrator.ultra_high_performance import create_ultra_high_performance_system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_experiment(experiment_data: dict) -> dict:
    """Simulate a materials experiment with realistic processing time."""
    # Simulate variable processing time based on complexity
    complexity = experiment_data.get('temperature', 100) / 100
    base_time = random.uniform(0.1, 0.3)  # Base simulation time
    processing_time = base_time * complexity
    
    time.sleep(processing_time)  # Simulate processing
    
    # Generate synthetic results
    temp = experiment_data.get('temperature', 150)
    conc_a = experiment_data.get('precursor_A_conc', 1.0)
    conc_b = experiment_data.get('precursor_B_conc', 1.0)
    pH = experiment_data.get('pH', 7)
    
    # Synthetic band gap calculation with some randomness
    band_gap = 1.4 + 0.3 * (temp - 150) / 100 + 0.2 * (conc_a - 1.0) - 0.1 * (pH - 7) + random.uniform(-0.1, 0.1)
    efficiency = max(0, min(35, 25 + (1.5 - abs(band_gap - 1.4)) * 20 + random.uniform(-2, 2)))
    stability = max(0, min(1, 0.8 + random.uniform(-0.1, 0.1)))
    
    return {
        'band_gap': band_gap,
        'efficiency': efficiency,
        'stability': stability,
        'processing_time': processing_time,
        'complexity_score': complexity
    }


def demonstrate_ultra_cache_system(cache):
    """Demonstrate ultra-high performance caching capabilities."""
    print("\n⚡ Demonstrating Ultra-High Performance Cache")
    print("-" * 60)
    
    # Test cache with various data types
    test_data = [
        ('experiment_123', {'temp': 150, 'results': [1.2, 1.3, 1.4]}),
        ('large_dataset', list(range(10000))),  # Large data
        ('simulation_result', {'matrix': [[i*j for j in range(100)] for i in range(100)]}),
        ('frequent_access', "This will be accessed frequently"),
        ('ttl_test', {'temp_data': True}),
    ]
    
    print("📥 Populating cache with test data...")
    for key, value in test_data:
        ttl = 300 if key == 'ttl_test' else None  # Special TTL for one entry
        success = cache.put(key, value, ttl=ttl)
        print(f"   {'✅' if success else '❌'} {key}: {len(str(value))} chars")
    
    # Test cache performance with rapid access
    print("\n🔥 Testing cache performance (1000 rapid accesses)...")
    start_time = time.time()
    
    for i in range(1000):
        # Mix of hits and misses
        if i % 3 == 0:
            cache.get('frequent_access')  # Frequent hits
        elif i % 5 == 0:
            cache.get('experiment_123')  # Regular hits
        else:
            cache.get(f'nonexistent_{i}')  # Misses
    
    access_time = time.time() - start_time
    print(f"   Completed 1000 accesses in {access_time:.3f}s ({1000/access_time:.0f} ops/sec)")
    
    # Show cache statistics
    stats = cache.get_statistics()
    print(f"\n📊 Cache Performance:")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Memory usage: {stats['memory_usage_mb']:.1f}MB / {stats['max_memory_mb']:.1f}MB")
    print(f"   Entries: {stats['size']} / {stats['max_size']}")
    print(f"   Evictions: {stats['evictions']}")
    
    # Test memory pressure and eviction
    print("\n🧠 Testing intelligent eviction under memory pressure...")
    large_data_count = 0
    while stats['memory_usage_mb'] < stats['max_memory_mb'] * 0.8:
        large_key = f"large_data_{large_data_count}"
        large_value = [random.random() for _ in range(10000)]  # ~80KB
        if not cache.put(large_key, large_value):
            break
        large_data_count += 1
        stats = cache.get_statistics()
    
    print(f"   Added {large_data_count} large entries")
    print(f"   Memory usage: {stats['memory_usage_mb']:.1f}MB")
    print(f"   Total evictions: {stats['evictions']}")
    
    return stats


def demonstrate_load_balancer(load_balancer):
    """Demonstrate distributed load balancing capabilities."""
    print("\n⚖️  Demonstrating Distributed Load Balancer")
    print("-" * 60)
    
    # Simulate task distribution
    print("📋 Distributing tasks across workers...")
    task_assignments = []
    
    for i in range(50):
        task_weight = random.randint(1, 3)
        worker = load_balancer.select_worker(task_weight)
        
        if worker:
            load_balancer.start_task(worker, task_weight)
            task_assignments.append((worker, task_weight, time.time()))
            print(f"   Task {i+1:2d}: {worker} (weight: {task_weight})")
        else:
            print(f"   Task {i+1:2d}: No available workers!")
        
        if i % 10 == 9:  # Status check every 10 tasks
            status = load_balancer.get_status()
            print(f"   Status: {status['current_load']}/{status['total_capacity']} load ({status['utilization']:.1%})")
    
    # Simulate task completions with performance tracking
    print("\n✅ Simulating task completions...")
    for worker, weight, start_time in task_assignments[:20]:  # Complete first 20 tasks
        duration = random.uniform(0.5, 2.0)  # Simulate task duration
        load_balancer.complete_task(worker, weight, duration)
    
    # Show final status
    final_status = load_balancer.get_status()
    print(f"\n📊 Load Balancer Status:")
    print(f"   Strategy: {final_status['strategy']}")
    print(f"   Workers: {final_status['healthy_workers']}/{final_status['total_workers']} healthy")
    print(f"   Utilization: {final_status['utilization']:.1%}")
    
    for worker_id, worker_info in final_status['workers'].items():
        print(f"   {worker_id}: {worker_info['load']}/{worker_info['capacity']} "
              f"(perf: {worker_info['performance']:.2f}, {'✅' if worker_info['healthy'] else '❌'})")
    
    return final_status


async def demonstrate_concurrent_processing(processor):
    """Demonstrate concurrent experiment processing."""
    print("\n🚀 Demonstrating Concurrent Experiment Processing")
    print("-" * 60)
    
    # Generate batch of experiments
    experiments = []
    for i in range(100):
        exp = {
            'id': f'exp_{i:03d}',
            'temperature': random.uniform(100, 300),
            'precursor_A_conc': random.uniform(0.5, 2.0),
            'precursor_B_conc': random.uniform(0.5, 2.0),
            'pH': random.uniform(5, 9),
            'reaction_time': random.uniform(1, 12)
        }
        experiments.append(exp)
    
    print(f"🧪 Processing batch of {len(experiments)} experiments...")
    
    # Process experiments concurrently
    start_time = time.time()
    results = await processor.process_experiments_batch(experiments, simulate_experiment)
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', True)]
    
    print(f"\n📊 Concurrent Processing Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Throughput: {len(experiments)/total_time:.1f} experiments/second")
    print(f"   Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results):.1%})")
    print(f"   Average experiment time: {total_time/len(experiments):.3f}s")
    
    if successful_results:
        band_gaps = [r.get('band_gap', 0) for r in successful_results]
        efficiencies = [r.get('efficiency', 0) for r in successful_results]
        
        print(f"   Band gap range: {min(band_gaps):.3f} - {max(band_gaps):.3f} eV")
        print(f"   Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")
        
        # Find best materials
        target_range = (1.2, 1.6)
        good_materials = [r for r in successful_results 
                         if target_range[0] <= r.get('band_gap', 0) <= target_range[1]]
        
        if good_materials:
            best_material = max(good_materials, key=lambda x: x.get('efficiency', 0))
            print(f"   🏆 Best material: {best_material['band_gap']:.3f} eV, {best_material['efficiency']:.1f}% efficiency")
    
    # Get performance statistics
    perf_stats = processor.get_performance_stats()
    if perf_stats.get('completed_experiments', 0) > 0:
        print(f"\n⚡ Performance Statistics:")
        print(f"   Experiments per second: {perf_stats.get('experiments_per_second', 0):.1f}")
        print(f"   Average duration: {perf_stats.get('avg_duration', 0):.3f}s")
        print(f"   Duration range: {perf_stats.get('min_duration', 0):.3f}s - {perf_stats.get('max_duration', 0):.3f}s")
    
    return results


def demonstrate_auto_scaling(auto_scaler):
    """Demonstrate intelligent auto-scaling capabilities."""
    print("\n📈 Demonstrating Intelligent Auto-Scaling")
    print("-" * 60)
    
    # Simulate varying load patterns
    print("📊 Simulating dynamic load patterns...")
    
    # Phase 1: Gradual load increase
    print("\n   Phase 1: Gradual load increase")
    for i in range(15):
        load = min(1.0, 0.1 + i * 0.06)  # Gradually increase to 100%
        queue_size = max(0, int((load - 0.5) * 40))  # Queue builds up at high load
        auto_scaler.report_load(load, queue_size)
        
        status = auto_scaler.get_scaling_status()
        print(f"      Step {i+1:2d}: Load {load:.2f}, Queue {queue_size:2d}, Workers {status['current_workers']}")
        
        if i % 5 == 4:  # Short pause every 5 steps
            time.sleep(0.1)
    
    # Phase 2: High sustained load
    print("\n   Phase 2: High sustained load")
    for i in range(10):
        load = random.uniform(0.85, 0.95)  # High sustained load
        queue_size = random.randint(15, 35)
        auto_scaler.report_load(load, queue_size)
        
        status = auto_scaler.get_scaling_status()
        print(f"      Step {i+1:2d}: Load {load:.2f}, Queue {queue_size:2d}, Workers {status['current_workers']}")
    
    # Phase 3: Load decrease
    print("\n   Phase 3: Load decrease") 
    for i in range(15):
        load = max(0.1, 0.9 - i * 0.05)  # Gradually decrease
        queue_size = max(0, int((load - 0.3) * 20))
        auto_scaler.report_load(load, queue_size)
        
        status = auto_scaler.get_scaling_status()
        print(f"      Step {i+1:2d}: Load {load:.2f}, Queue {queue_size:2d}, Workers {status['current_workers']}")
        
        if i % 5 == 4:
            time.sleep(0.1)
    
    # Show final scaling status
    final_status = auto_scaler.get_scaling_status()
    print(f"\n📊 Auto-Scaling Summary:")
    print(f"   Worker range: {final_status['min_workers']}-{final_status['max_workers']}")
    print(f"   Final workers: {final_status['current_workers']}")
    print(f"   Target utilization: {final_status['target_utilization']:.1%}")
    print(f"   Final load: {final_status['current_load']:.2f}")
    print(f"   Scaling decisions: {final_status['recent_decisions']}")
    
    if final_status.get('scaling_history'):
        print(f"   Recent scaling actions:")
        for decision in final_status['scaling_history'][-3:]:
            print(f"     {decision['action']}: {decision['old_workers']} → {decision['new_workers']} workers")
    
    return final_status


async def run_ultra_performance_campaign():
    """Run a complete ultra-performance materials discovery campaign."""
    print("\n🎯 Ultra-Performance Discovery Campaign")
    print("=" * 70)
    
    # Setup systems
    print("⚡ Initializing ultra-performance systems...")
    cache, load_balancer, processor, auto_scaler = create_ultra_high_performance_system()
    validator, error_handler = create_robust_validation_system()
    security_manager = create_advanced_security_system(SecurityLevel.STANDARD)  # Balanced performance
    monitoring, alert_manager = create_comprehensive_monitoring()
    
    print("✅ All systems initialized")
    
    # Define ultra-scale campaign parameters
    objective = MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),
        optimization_direction="target",
        material_system="perovskites_ultra_scale",
        success_threshold=1.4,
    )
    
    param_space = {
        "precursor_A_conc": (0.1, 2.0),
        "precursor_B_conc": (0.1, 2.0),
        "temperature": (100, 300),
        "reaction_time": (1, 24),
        "pH": (3, 11),
        "solvent_ratio": (0, 1),
    }
    
    print(f"🎯 Campaign objective: {objective.target_property} in range {objective.target_range}")
    
    # Initialize lab with ultra-performance configuration
    lab = AutonomousLab(
        robots=["synthesis_robot_1", "synthesis_robot_2", "characterization_robot_1", "characterization_robot_2"],
        instruments=["xrd_1", "xrd_2", "uv_vis_1", "uv_vis_2", "pl_spectrometer_1", "pl_spectrometer_2"],
        planner=BayesianPlanner(target_property="band_gap", exploration_factor=0.05)  # Optimized exploration
    )
    
    # Performance monitoring setup
    campaign_start = time.time()
    total_experiments = 0
    successful_experiments = 0
    
    # Generate and cache experiment parameters
    print("📋 Pre-generating experiment parameters...")
    experiment_batches = []
    
    for batch_num in range(10):  # 10 batches of 50 experiments each
        batch_key = f"experiment_batch_{batch_num}"
        
        # Check cache first
        cached_batch = cache.get(batch_key)
        if cached_batch:
            print(f"   📥 Using cached batch {batch_num}")
            experiment_batches.append(cached_batch)
        else:
            # Generate new batch
            batch_experiments = []
            for i in range(50):
                exp_params = {}
                for param, (low, high) in param_space.items():
                    exp_params[param] = random.uniform(low, high)
                exp_params['batch'] = batch_num
                exp_params['id'] = f"batch_{batch_num}_exp_{i}"
                batch_experiments.append(exp_params)
            
            # Cache the batch
            cache.put(batch_key, batch_experiments, ttl=1800)  # 30 min TTL
            experiment_batches.append(batch_experiments)
            print(f"   📤 Generated and cached batch {batch_num}")
    
    # Process experiments in batches with load balancing
    print("\n🚀 Starting ultra-performance campaign execution...")
    all_results = []
    
    for batch_num, batch in enumerate(experiment_batches):
        print(f"\n   Processing batch {batch_num + 1}/10 ({len(batch)} experiments)...")
        
        # Update auto-scaler with current load
        current_load = len(batch) / 100  # Normalize to 0-1
        auto_scaler.report_load(current_load, queue_size=sum(len(b) for b in experiment_batches[batch_num:]))
        
        # Process batch concurrently
        batch_start = time.time()
        batch_results = await processor.process_experiments_batch(batch, simulate_experiment)
        batch_time = time.time() - batch_start
        
        # Validate results
        validated_results = []
        for result in batch_results:
            if result.get('success', False):
                # Quick validation check
                validation_results = validator.validate_experiment_parameters(result.get('experiment', {}))
                critical_issues = [v for v in validation_results if v.severity == 'critical']
                
                if not critical_issues:
                    validated_results.append(result)
                    successful_experiments += 1
                else:
                    logger.warning(f"Experiment {result.get('experiment_id')} failed validation")
            
            total_experiments += 1
        
        all_results.extend(validated_results)
        
        # Update monitoring
        monitoring.increment_counter('ultra_experiments_total', len(batch))
        monitoring.increment_counter('ultra_experiments_successful', len(validated_results))
        monitoring.record_metric('batch_processing_time', batch_time)
        monitoring.record_metric('batch_throughput', len(batch) / batch_time)
        
        # Progress update
        throughput = len(batch) / batch_time
        print(f"     ✅ Completed: {len(validated_results)}/{len(batch)} successful ({throughput:.1f} exp/sec)")
        
        # Brief pause for system stability
        await asyncio.sleep(0.1)
    
    campaign_duration = time.time() - campaign_start
    
    # Analyze results
    print("\n" + "=" * 70)
    print("🏆 ULTRA-PERFORMANCE CAMPAIGN RESULTS")
    print("=" * 70)
    
    print(f"Total experiments: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments:.1%}")
    print(f"Campaign duration: {campaign_duration:.2f}s")
    print(f"Overall throughput: {total_experiments/campaign_duration:.1f} experiments/second")
    
    if all_results:
        # Find best materials
        band_gaps = [r.get('band_gap', 0) for r in all_results]
        efficiencies = [r.get('efficiency', 0) for r in all_results]
        
        target_materials = [r for r in all_results 
                          if objective.target_range[0] <= r.get('band_gap', 0) <= objective.target_range[1]]
        
        print(f"\n🔬 Results Analysis:")
        print(f"   Band gap range: {min(band_gaps):.3f} - {max(band_gaps):.3f} eV")
        print(f"   Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")
        print(f"   Target materials: {len(target_materials)}")
        
        if target_materials:
            best_material = max(target_materials, key=lambda x: x.get('efficiency', 0))
            print(f"\n🥇 Best Material:")
            print(f"   Band gap: {best_material['band_gap']:.3f} eV")
            print(f"   Efficiency: {best_material['efficiency']:.1f}%")
            print(f"   Stability: {best_material['stability']:.3f}")
    
    # Performance system analysis
    print(f"\n⚡ Performance Systems Analysis:")
    
    cache_stats = cache.get_statistics()
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"   Cache utilization: {cache_stats['memory_usage_mb']:.1f}MB")
    
    lb_status = load_balancer.get_status()
    print(f"   Load balancer efficiency: {lb_status['utilization']:.1%}")
    
    scaling_status = auto_scaler.get_scaling_status()
    print(f"   Auto-scaling decisions: {scaling_status['recent_decisions']}")
    print(f"   Final worker count: {scaling_status['current_workers']}")
    
    proc_stats = processor.get_performance_stats()
    print(f"   Concurrent processing efficiency: {proc_stats.get('success_rate', 0):.1%}")
    
    # Cleanup
    processor.shutdown()
    monitoring.stop_monitoring()
    
    return {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'duration': campaign_duration,
        'throughput': total_experiments / campaign_duration,
        'best_material': best_material if 'best_material' in locals() else None
    }


async def main():
    """Main execution function."""
    print("⚡ Ultra-High Performance Materials Discovery - Generation 3")
    print("=" * 80)
    print("Demonstrating optimized caching, load balancing, concurrent processing, and auto-scaling")
    
    try:
        # Create ultra-performance systems
        cache, load_balancer, processor, auto_scaler = create_ultra_high_performance_system()
        
        # Demonstrate each system
        demonstrate_ultra_cache_system(cache)
        demonstrate_load_balancer(load_balancer)
        await demonstrate_concurrent_processing(processor)
        demonstrate_auto_scaling(auto_scaler)
        
        # Run complete ultra-performance campaign
        campaign_results = await run_ultra_performance_campaign()
        
        print(f"\n🎉 Ultra-Performance Discovery Complete!")
        print(f"   Achieved {campaign_results['throughput']:.1f} experiments/second")
        print(f"   Success rate: {campaign_results['successful_experiments']}/{campaign_results['total_experiments']}")
        
        if campaign_results['best_material']:
            print(f"   Best material: {campaign_results['best_material']['band_gap']:.3f} eV")
        
        # Cleanup
        processor.shutdown()
        
    except KeyboardInterrupt:
        print("\n⏹️  Ultra-performance demonstration interrupted")
    except Exception as e:
        logger.error(f"Ultra-performance demonstration failed: {e}", exc_info=True)
        return 1
    
    print("\n⚡ Ultra-performance systems demonstration completed!")
    return 0


if __name__ == "__main__":
    # Handle missing os import
    import os
    
    # Run async main
    sys.exit(asyncio.run(main()))