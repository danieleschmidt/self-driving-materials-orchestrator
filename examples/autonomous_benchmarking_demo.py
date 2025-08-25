#!/usr/bin/env python3
"""Autonomous Benchmarking and Validation Demo.

Demonstrates comprehensive benchmarking capabilities for materials discovery algorithms.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from materials_orchestrator.autonomous_benchmarking import (
    AutonomousBenchmarkSuite,
    BenchmarkConfiguration,
    BenchmarkType,
    MetricType,
    benchmark_algorithms,
    create_performance_benchmark,
    create_accuracy_benchmark,
    get_benchmark_suite
)


class MockOptimizationResult:
    """Mock optimization result for testing."""
    def __init__(self, best_value: float, convergence_history: list = None):
        self.best_value = best_value
        self.convergence_history = convergence_history or []
        self.confidence = random.uniform(0.7, 0.95)


def random_search_algorithm(test_data: Dict[str, Any]) -> MockOptimizationResult:
    """Mock random search algorithm."""
    sample_size = test_data.get("sample_size", 100)
    
    # Simulate random search performance
    time.sleep(0.001 * sample_size)  # Simulate computation
    
    # Generate mock result
    best_value = random.uniform(-50, 0)  # Random performance
    convergence = [random.uniform(-100, best_value) for _ in range(10)]
    
    return MockOptimizationResult(best_value, convergence)


def bayesian_optimization_algorithm(test_data: Dict[str, Any]) -> MockOptimizationResult:
    """Mock Bayesian optimization algorithm."""
    sample_size = test_data.get("sample_size", 100)
    
    # Simulate more computation time but better results
    time.sleep(0.005 * sample_size)
    
    # Generate better mock result
    best_value = random.uniform(-10, 0)  # Better performance
    convergence = [max(-100, random.uniform(-20, best_value) - i) for i in range(15)]
    
    return MockOptimizationResult(best_value, convergence)


def gradient_descent_algorithm(test_data: Dict[str, Any]) -> MockOptimizationResult:
    """Mock gradient descent algorithm."""
    sample_size = test_data.get("sample_size", 100)
    
    # Fast but not always optimal
    time.sleep(0.002 * sample_size)
    
    # Deterministic-ish performance
    best_value = -sample_size / 100.0 + random.uniform(-5, 5)
    convergence = [best_value + 10 * (0.8 ** i) for i in range(12)]
    
    return MockOptimizationResult(best_value, convergence)


def evolutionary_algorithm(test_data: Dict[str, Any]) -> MockOptimizationResult:
    """Mock evolutionary algorithm."""
    sample_size = test_data.get("sample_size", 100)
    
    # Slower but robust
    time.sleep(0.008 * sample_size)
    
    # Good performance with high reliability
    best_value = random.uniform(-15, -2)
    convergence = [random.uniform(-50, best_value) for _ in range(20)]
    
    return MockOptimizationResult(best_value, convergence)


async def demo_basic_benchmarking():
    """Demonstrate basic benchmarking capabilities."""
    print("🔬 Autonomous Benchmarking Demo")
    print("=" * 40)
    
    # Define algorithms to benchmark
    algorithms = {
        "Random Search": random_search_algorithm,
        "Bayesian Optimization": bayesian_optimization_algorithm,
        "Gradient Descent": gradient_descent_algorithm,
        "Evolutionary Algorithm": evolutionary_algorithm
    }
    
    print(f"📊 Benchmarking {len(algorithms)} algorithms...")
    
    # Create custom benchmark
    custom_benchmark = BenchmarkConfiguration(
        name="Materials Discovery Algorithm Comparison",
        benchmark_type=BenchmarkType.ALGORITHM_COMPARISON,
        metrics=[MetricType.ACCURACY, MetricType.SPEED, MetricType.CONVERGENCE],
        sample_sizes=[25, 50, 100],
        iterations=5,  # Reduced for demo
        timeout_seconds=60
    )
    
    # Run benchmark
    start_time = time.time()
    results = await benchmark_algorithms(
        algorithms=algorithms,
        benchmark_config=custom_benchmark,
        export_results=True
    )
    duration = time.time() - start_time
    
    print(f"✅ Benchmarking completed in {duration:.2f} seconds")
    print(f"📈 Results Summary:")
    print(f"   Total runs: {results['analysis']['total_results']}")
    print(f"   Success rate: {results['analysis']['success_rate']:.1%}")
    
    # Show algorithm performance
    print(f"\n🏆 Algorithm Performance Rankings:")
    algo_performance = results['analysis']['algorithm_performance']
    
    # Sort by accuracy
    sorted_algos = sorted(
        algo_performance.items(),
        key=lambda x: x[1]['metric_averages'].get('accuracy', {}).get('mean', 0),
        reverse=True
    )
    
    for i, (algo_name, perf) in enumerate(sorted_algos, 1):
        accuracy = perf['metric_averages'].get('accuracy', {}).get('mean', 0)
        speed = perf['metric_averages'].get('speed', {}).get('mean', 0)
        print(f"   {i}. {algo_name}")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      Speed: {speed:.1f} ops/sec")
        print(f"      Runs: {perf['total_runs']}")
    
    return results


async def demo_performance_scaling():
    """Demonstrate performance scaling analysis."""
    print(f"\n⚡ Performance Scaling Analysis")
    print("-" * 35)
    
    # Create performance benchmark
    perf_benchmark = create_performance_benchmark(
        name="Scalability Analysis",
        sample_sizes=[10, 25, 50, 100, 200],
        iterations=8,
        timeout=120
    )
    
    # Test subset of algorithms for scaling
    scaling_algorithms = {
        "Bayesian Optimization": bayesian_optimization_algorithm,
        "Gradient Descent": gradient_descent_algorithm
    }
    
    print(f"📊 Testing scalability of {len(scaling_algorithms)} algorithms...")
    
    suite = get_benchmark_suite()
    suite.register_benchmark(perf_benchmark)
    
    scaling_results = await suite.run_benchmark_suite(
        algorithms=scaling_algorithms,
        benchmark_names=[perf_benchmark.name]
    )
    
    print(f"✅ Scaling analysis completed")
    print(f"📈 Scaling Results:")
    
    for algo_name, perf in scaling_results['analysis']['algorithm_performance'].items():
        print(f"\n   {algo_name}:")
        print(f"     Average time: {perf['avg_duration']:.3f}s")
        
        if 'scalability' in perf['metric_averages']:
            scalability = perf['metric_averages']['scalability']['mean']
            print(f"     Scalability score: {scalability:.1f}")
        
        if 'efficiency' in perf['metric_averages']:
            efficiency = perf['metric_averages']['efficiency']['mean']
            print(f"     Efficiency score: {efficiency:.3f}")


async def demo_accuracy_validation():
    """Demonstrate accuracy validation benchmarking."""
    print(f"\n🎯 Accuracy Validation Analysis")
    print("-" * 32)
    
    # Create accuracy benchmark
    accuracy_benchmark = create_accuracy_benchmark(
        name="Algorithm Accuracy Validation",
        sample_sizes=[50, 100],
        iterations=12,
        significance_level=0.05
    )
    
    # Test all algorithms for accuracy
    accuracy_algorithms = {
        "Random Search": random_search_algorithm,
        "Bayesian Optimization": bayesian_optimization_algorithm,
        "Evolutionary Algorithm": evolutionary_algorithm
    }
    
    print(f"🔬 Validating accuracy of {len(accuracy_algorithms)} algorithms...")
    
    accuracy_results = await benchmark_algorithms(
        algorithms=accuracy_algorithms,
        benchmark_config=accuracy_benchmark,
        export_results=False
    )
    
    print(f"✅ Accuracy validation completed")
    
    # Show statistical significance
    significance = accuracy_results['analysis'].get('statistical_significance', {})
    if significance.get('significant_differences'):
        print(f"\n📊 Statistical Significance Found:")
        for diff in significance['significant_differences']:
            better_algo = diff['algorithm1'] if diff['mean_difference'] > 0 else diff['algorithm2']
            other_algo = diff['algorithm2'] if diff['mean_difference'] > 0 else diff['algorithm1']
            print(f"   • {better_algo} significantly outperforms {other_algo}")
            print(f"     (p-value: {diff['p_value']:.3f})")
    else:
        print(f"\n📊 No statistically significant differences found")
    
    # Show accuracy rankings
    print(f"\n🏅 Accuracy Rankings:")
    algo_performance = accuracy_results['analysis']['algorithm_performance']
    
    sorted_by_accuracy = sorted(
        algo_performance.items(),
        key=lambda x: x[1]['metric_averages'].get('accuracy', {}).get('mean', 0),
        reverse=True
    )
    
    for i, (algo_name, perf) in enumerate(sorted_by_accuracy, 1):
        metrics = perf['metric_averages']
        accuracy = metrics.get('accuracy', {}).get('mean', 0)
        reliability = metrics.get('reliability', {}).get('mean', 0)
        convergence = metrics.get('convergence', {}).get('mean', 0)
        
        print(f"   {i}. {algo_name}")
        print(f"      Accuracy: {accuracy:.3f} ± {metrics.get('accuracy', {}).get('std', 0):.3f}")
        print(f"      Reliability: {reliability:.3f}")
        print(f"      Convergence: {convergence:.3f}")


async def demo_comprehensive_analysis():
    """Demonstrate comprehensive benchmark analysis."""
    print(f"\n📋 Comprehensive Benchmark Analysis")
    print("-" * 37)
    
    suite = get_benchmark_suite()
    
    # Run all standard benchmarks
    all_algorithms = {
        "Random Search": random_search_algorithm,
        "Bayesian Optimization": bayesian_optimization_algorithm,
        "Gradient Descent": gradient_descent_algorithm,
        "Evolutionary Algorithm": evolutionary_algorithm
    }
    
    print(f"🔄 Running comprehensive benchmark suite...")
    print(f"   Algorithms: {len(all_algorithms)}")
    print(f"   Benchmark types: {len(suite.benchmarks)}")
    
    comprehensive_results = await suite.run_benchmark_suite(all_algorithms)
    
    print(f"✅ Comprehensive analysis completed")
    print(f"⏱️  Total duration: {comprehensive_results['duration']:.1f} seconds")
    print(f"📊 Total benchmark runs: {comprehensive_results['total_results']}")
    
    # Generate and display report excerpt
    report = comprehensive_results['report']
    print(f"\n📄 Benchmark Report (Excerpt):")
    print("-" * 30)
    
    # Show first few lines of report
    report_lines = report.split('\n')
    for line in report_lines[:15]:
        print(f"   {line}")
    
    if len(report_lines) > 15:
        print(f"   ... (report continues for {len(report_lines) - 15} more lines)")
    
    # Export comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive results exported to: {results_file}")
    
    # Show recommendations
    print(f"\n💡 Algorithm Recommendations:")
    algo_performance = comprehensive_results['analysis']['algorithm_performance']
    
    # Find best overall algorithm (balanced scoring)
    best_algo = None
    best_score = -float('inf')
    
    for algo_name, perf in algo_performance.items():
        metrics = perf['metric_averages']
        
        # Calculate balanced score
        accuracy = metrics.get('accuracy', {}).get('mean', 0)
        speed = metrics.get('speed', {}).get('mean', 0)
        reliability = metrics.get('reliability', {}).get('mean', 0)
        
        # Normalize speed (log scale)
        normalized_speed = min(1.0, speed / 100.0) if speed > 0 else 0
        
        # Balanced score (accuracy 40%, speed 30%, reliability 30%)
        balanced_score = accuracy * 0.4 + normalized_speed * 0.3 + reliability * 0.3
        
        if balanced_score > best_score:
            best_score = balanced_score
            best_algo = algo_name
    
    if best_algo:
        print(f"   🏆 Best Overall: {best_algo} (score: {best_score:.3f})")
        
        # Show specific recommendations
        print(f"   📌 Recommendations:")
        print(f"      • For accuracy-critical tasks: Use highest accuracy algorithm")
        print(f"      • For real-time applications: Use highest speed algorithm") 
        print(f"      • For balanced performance: Use {best_algo}")
        print(f"      • For robust production: Consider reliability scores")


async def run_benchmarking_demo():
    """Run complete benchmarking demonstration."""
    print("🌟 Autonomous Benchmarking Framework - Complete Demo")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    try:
        # Run all demo sections
        await demo_basic_benchmarking()
        await demo_performance_scaling()
        await demo_accuracy_validation()
        await demo_comprehensive_analysis()
        
        total_duration = time.time() - total_start
        
        print(f"\n🎉 Benchmarking Demo Complete!")
        print(f"⏱️  Total demo duration: {total_duration:.2f} seconds")
        print(f"🔬 Autonomous benchmarking system fully operational")
        print(f"📊 Ready for comprehensive algorithm evaluation!")
        print(f"✨ Statistical validation and performance analysis available")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive benchmarking demo
    asyncio.run(run_benchmarking_demo())