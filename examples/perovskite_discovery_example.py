#!/usr/bin/env python3
"""
Perovskite Band Gap Optimization Example

Demonstrates autonomous materials discovery using the self-driving-materials-orchestrator
platform. This example shows how to set up and run a realistic campaign to discover
perovskite materials with optimal band gap properties.
"""

import logging
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials_orchestrator import (
    AutonomousLab, MaterialsObjective, BayesianPlanner, RandomPlanner
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_perovskite_optimization():
    """Run autonomous optimization of perovskite band gap."""
    
    print("🔬 Self-Driving Materials Discovery - Perovskite Example")
    print("=" * 60)
    
    # Define optimization objective
    objective = MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),  # Optimal for photovoltaics
        optimization_direction="target",
        material_system="perovskites",
        success_threshold=1.4,  # Target value
    )
    
    print(f"🎯 Objective: Optimize {objective.target_property}")
    print(f"   Target range: {objective.target_range} eV")
    print(f"   Material system: {objective.material_system}")
    
    # Define parameter space for perovskite synthesis
    param_space = {
        "precursor_A_conc": (0.1, 2.0),    # Molar concentration of precursor A
        "precursor_B_conc": (0.1, 2.0),    # Molar concentration of precursor B  
        "temperature": (100, 300),          # Synthesis temperature (°C)
        "reaction_time": (1, 24),           # Reaction time (hours)
        "pH": (3, 11),                      # Solution pH
        "solvent_ratio": (0, 1),            # DMF:DMSO ratio
    }
    
    print(f"\n📊 Parameter Space:")
    for param, (low, high) in param_space.items():
        print(f"   {param}: [{low}, {high}]")
    
    # Initialize autonomous laboratory with Bayesian planner
    bayesian_planner = BayesianPlanner(
        acquisition_function="expected_improvement",
        target_property="band_gap",
        exploration_factor=0.1
    )
    
    lab = AutonomousLab(
        robots=["synthesis_robot", "characterization_robot"],
        instruments=["xrd", "uv_vis", "pl_spectrometer"],
        planner=bayesian_planner
    )
    
    print(f"\n🤖 Laboratory Setup:")
    print(f"   Robots: {', '.join(lab.robots)}")
    print(f"   Instruments: {', '.join(lab.instruments)}")
    print(f"   Planner: Bayesian Optimization")
    
    # Run autonomous campaign
    print(f"\n🚀 Starting Discovery Campaign...")
    print("-" * 40)
    
    campaign = lab.run_campaign(
        objective=objective,
        param_space=param_space,
        initial_samples=15,       # Start with random exploration
        max_experiments=100,      # Budget constraint
        stop_on_target=True,      # Stop when target reached
        convergence_patience=20   # Stop if no improvement
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("🏆 CAMPAIGN RESULTS")
    print("=" * 60)
    
    print(f"Campaign ID: {campaign.campaign_id}")
    print(f"Total experiments: {campaign.total_experiments}")
    print(f"Successful experiments: {campaign.successful_experiments}")
    print(f"Success rate: {campaign.success_rate:.1%}")
    print(f"Duration: {campaign.duration:.2f} hours" if campaign.duration else "Duration: Ongoing")
    
    if campaign.best_material:
        print(f"\n🥇 Best Material Found:")
        print(f"   Band gap: {campaign.best_properties.get('band_gap', 'N/A'):.3f} eV")
        print(f"   Efficiency: {campaign.best_properties.get('efficiency', 'N/A'):.1f}%")
        print(f"   Stability: {campaign.best_properties.get('stability', 'N/A'):.3f}")
        
        print(f"\n🔬 Optimal Parameters:")
        for param, value in campaign.best_material['parameters'].items():
            print(f"   {param}: {value:.3f}")
    
    # Performance metrics
    print(f"\n📈 Performance Metrics:")
    if campaign.convergence_history:
        best_fitness = campaign.get_best_fitness()
        print(f"   Best fitness score: {best_fitness:.3f}")
        print(f"   Convergence points: {len(campaign.convergence_history)}")
        
        # Show convergence trend
        history = campaign.convergence_history
        if len(history) >= 5:
            improvement = history[-1]['best_fitness'] - history[4]['best_fitness']
            print(f"   Recent improvement: {improvement:.3f}")
    
    # Analysis of parameter importance (simplified)
    print(f"\n🔍 Parameter Analysis:")
    successful_experiments = [exp for exp in campaign.experiments if exp.status == "completed"]
    
    if successful_experiments:
        # Find experiments that achieved target
        target_achieved = [
            exp for exp in successful_experiments 
            if objective.evaluate_success(exp.results.get(objective.target_property, 0))
        ]
        
        print(f"   Experiments reaching target: {len(target_achieved)}")
        
        if target_achieved:
            # Calculate average parameter values for successful experiments
            param_averages = {}
            for param in param_space.keys():
                values = [exp.parameters.get(param, 0) for exp in target_achieved]
                param_averages[param] = sum(values) / len(values)
            
            print(f"   Average parameters for target materials:")
            for param, avg_value in param_averages.items():
                print(f"     {param}: {avg_value:.3f}")
    
    # Comparison with traditional methods
    print(f"\n⚡ Acceleration Analysis:")
    experiments_to_target = campaign.total_experiments
    
    # Estimate traditional method performance
    traditional_estimate = 200  # Typical grid/random search
    acceleration = traditional_estimate / max(experiments_to_target, 1)
    
    print(f"   Experiments to target: {experiments_to_target}")
    print(f"   Traditional estimate: {traditional_estimate}")
    print(f"   Acceleration factor: {acceleration:.1f}x")
    
    # Save results summary
    results_summary = {
        "campaign_id": campaign.campaign_id,
        "objective": {
            "target_property": objective.target_property,
            "target_range": objective.target_range,
            "material_system": objective.material_system,
        },
        "results": {
            "total_experiments": campaign.total_experiments,
            "successful_experiments": campaign.successful_experiments,
            "success_rate": campaign.success_rate,
            "best_properties": campaign.best_properties,
            "acceleration_factor": acceleration,
        },
        "best_material": campaign.best_material,
    }
    
    import json
    output_file = Path("perovskite_campaign_results.json")
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n✅ Campaign completed successfully!")
    
    return campaign


def compare_optimization_strategies():
    """Compare different optimization strategies."""
    
    print("\n" + "=" * 60)
    print("📊 STRATEGY COMPARISON")
    print("=" * 60)
    
    objective = MaterialsObjective(
        target_property="band_gap",
        target_range=(1.2, 1.6),
        optimization_direction="target",
        success_threshold=1.4,
    )
    
    param_space = {
        "precursor_A_conc": (0.1, 2.0),
        "precursor_B_conc": (0.1, 2.0),
        "temperature": (100, 300),
        "reaction_time": (1, 24),
    }
    
    strategies = [
        ("Random Search", RandomPlanner()),
        ("Bayesian Optimization", BayesianPlanner(target_property="band_gap")),
    ]
    
    results = {}
    
    for strategy_name, planner in strategies:
        print(f"\n🔄 Testing {strategy_name}...")
        
        lab = AutonomousLab(planner=planner)
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=5 if strategy_name == "Bayesian Optimization" else 0,
            max_experiments=50,
            stop_on_target=True,
        )
        
        results[strategy_name] = {
            "experiments": campaign.total_experiments,
            "success_rate": campaign.success_rate,
            "best_value": campaign.best_properties.get("band_gap", 0),
            "target_achieved": objective.evaluate_success(
                campaign.best_properties.get("band_gap", 0)
            ),
        }
    
    print(f"\n📈 Comparison Results:")
    print("-" * 40)
    
    for strategy, result in results.items():
        print(f"{strategy}:")
        print(f"  Experiments: {result['experiments']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Best band gap: {result['best_value']:.3f} eV")
        print(f"  Target achieved: {'✅' if result['target_achieved'] else '❌'}")
        print()
    
    return results


if __name__ == "__main__":
    print("Starting Perovskite Discovery Example...")
    
    try:
        # Run main optimization example
        campaign = run_perovskite_optimization()
        
        # Compare strategies
        comparison = compare_optimization_strategies()
        
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        sys.exit(1)
    
    print("\n🎉 Example completed!")