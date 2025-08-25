#!/usr/bin/env python3
"""Enhanced Virtual Laboratory Demonstration.

Shows advanced virtual experimentation capabilities for materials discovery.
"""

import asyncio
import json
import time
from datetime import datetime

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from materials_orchestrator.virtual_laboratory import (
    VirtualLaboratory,
    SimulationParameters,
    PhysicsModel,
    run_virtual_campaign,
    create_research_virtual_lab,
    ResearchSimulationFramework
)


async def demo_basic_virtual_experiments():
    """Demonstrate basic virtual laboratory capabilities."""
    print("🔬 Enhanced Virtual Laboratory Demo")
    print("=" * 50)
    
    # Create virtual laboratory
    lab = VirtualLaboratory()
    
    print(f"✅ Virtual laboratory initialized")
    print(f"📊 Status: {lab.get_laboratory_status()}")
    
    # Define experiment parameters
    experiment_params = {
        "precursor_A_conc": 1.5,
        "precursor_B_conc": 0.8,
        "temperature": 150,
        "reaction_time": 4.0,
        "pH": 6.5,
        "solvent_ratio": 0.3
    }
    
    print(f"\n🧪 Running single virtual experiment...")
    start_time = time.time()
    
    try:
        result = await lab.run_experiment(experiment_params, "perovskite")
        duration = time.time() - start_time
        
        print(f"✅ Experiment completed in {duration:.3f}s")
        print(f"🎯 Results:")
        for prop, value in result["results"].items():
            if isinstance(value, float):
                print(f"   {prop}: {value:.3f}")
            else:
                print(f"   {prop}: {value}")
                
    except Exception as e:
        print(f"❌ Experiment failed: {e}")


async def demo_virtual_campaign():
    """Demonstrate virtual experiment campaign."""
    print(f"\n🚀 Virtual Campaign Demo")
    print("-" * 30)
    
    # Generate parameter sets for optimization
    parameter_sets = []
    for i in range(20):
        params = {
            "precursor_A_conc": 0.5 + i * 0.1,
            "precursor_B_conc": 0.3 + i * 0.05,
            "temperature": 120 + i * 5,
            "reaction_time": 2.0 + i * 0.2,
            "pH": 5.0 + i * 0.2,
            "solvent_ratio": 0.1 + i * 0.04
        }
        parameter_sets.append(params)
    
    print(f"🔄 Running campaign with {len(parameter_sets)} experiments...")
    start_time = time.time()
    
    campaign_results = await run_virtual_campaign(parameter_sets, "perovskite")
    duration = time.time() - start_time
    
    print(f"✅ Campaign completed in {duration:.2f}s")
    print(f"📈 Campaign Results:")
    print(f"   Total experiments: {campaign_results['total_experiments']}")
    print(f"   Successful: {campaign_results['successful']}")
    print(f"   Success rate: {campaign_results['success_rate']:.1%}")
    
    if campaign_results["best_material"]:
        best = campaign_results["best_material"]
        print(f"\n🏆 Best Material Found:")
        print(f"   Parameters: {best['parameters']}")
        print(f"   Results: {best['results']}")


async def demo_research_simulation():
    """Demonstrate research-grade simulation capabilities."""
    print(f"\n🔬 Research-Grade Simulation Demo")  
    print("-" * 35)
    
    # Create research laboratory
    research_lab = create_research_virtual_lab(
        physics_model=PhysicsModel.QUANTUM_MECHANICAL,
        precision="research",
        enable_advanced_features=True
    )
    
    print(f"✅ Research laboratory created")
    
    # Run advanced experiment
    research_params = {
        "precursor_A_conc": 1.2,
        "precursor_B_conc": 0.6,
        "temperature": 135,
        "reaction_time": 3.5,
        "pH": 6.2,
        "solvent_ratio": 0.25
    }
    
    print(f"🧪 Running research-grade experiment...")
    result = await research_lab.run_experiment(research_params)
    
    print(f"✅ Research experiment completed")
    print(f"📊 Advanced Results:")
    for prop, value in result["results"].items():
        if isinstance(value, float):
            print(f"   {prop}: {value:.4f}")
    
    # Demonstrate physics simulations
    print(f"\n⚛️  Advanced Physics Simulation Demo")
    framework = ResearchSimulationFramework()
    
    # Get a material from the experiment
    material = research_lab.materials_database[-1] if research_lab.materials_database else None
    
    if material:
        print(f"🔬 Running quantum mechanical simulation...")
        quantum_results = await framework.run_physics_simulation(
            material, PhysicsModel.QUANTUM_MECHANICAL
        )
        print(f"   Quantum results: {quantum_results}")
        
        print(f"🔬 Running molecular dynamics simulation...")
        md_results = await framework.run_physics_simulation(
            material, PhysicsModel.MOLECULAR_DYNAMICS
        )
        print(f"   MD results: {md_results}")


async def demo_laboratory_management():
    """Demonstrate laboratory management capabilities."""
    print(f"\n🏭 Laboratory Management Demo")
    print("-" * 30)
    
    lab = VirtualLaboratory()
    
    # Check laboratory status
    status = lab.get_laboratory_status()
    print(f"🔍 Laboratory Status:")
    print(f"   Experiments run: {status['total_experiments']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Materials discovered: {status['materials_discovered']}")
    
    print(f"\n🤖 Robot Status:")
    for robot_id, robot_info in status["robots"].items():
        print(f"   {robot_id}: {'Busy' if robot_info['busy'] else 'Available'}")
        if robot_info["needs_maintenance"]:
            print(f"      ⚠️  Needs maintenance")
    
    print(f"\n📡 Instrument Status:")
    for inst_id, inst_info in status["instruments"].items():
        print(f"   {inst_id} ({inst_info['type']}): Precision {inst_info['precision']:.1%}")
        if inst_info["needs_calibration"]:
            print(f"      🔧 Needs calibration")
    
    # Run some experiments to generate data
    print(f"\n🔄 Running experiments for data generation...")
    for i in range(5):
        params = {
            "precursor_A_conc": 1.0 + i * 0.2,
            "precursor_B_conc": 0.5 + i * 0.1,
            "temperature": 130 + i * 10,
            "reaction_time": 3.0 + i * 0.5,
            "pH": 6.0 + i * 0.3,
            "solvent_ratio": 0.2 + i * 0.1
        }
        try:
            await lab.run_experiment(params)
            print(f"   ✅ Experiment {i+1} completed")
        except Exception as e:
            print(f"   ❌ Experiment {i+1} failed: {e}")
    
    # Export results
    print(f"\n💾 Exporting laboratory results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"virtual_lab_results_{timestamp}.json"
    export_path = lab.export_results(filename)
    
    print(f"✅ Results exported to: {export_path}")
    
    # Show final statistics
    final_status = lab.get_laboratory_status()
    print(f"\n📈 Final Laboratory Statistics:")
    print(f"   Total experiments: {final_status['total_experiments']}")
    print(f"   Successful experiments: {final_status['successful_experiments']}")
    print(f"   Success rate: {final_status['success_rate']:.1%}")
    print(f"   Materials in database: {final_status['materials_discovered']}")


async def run_comprehensive_demo():
    """Run comprehensive virtual laboratory demonstration."""
    print("🌟 Enhanced Virtual Laboratory - Comprehensive Demo")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    try:
        # Run all demos
        await demo_basic_virtual_experiments()
        await demo_virtual_campaign()
        await demo_research_simulation()
        await demo_laboratory_management()
        
        total_duration = time.time() - total_start
        
        print(f"\n🎉 Demo Complete!")
        print(f"⏱️  Total duration: {total_duration:.2f} seconds")
        print(f"🚀 Virtual laboratory system fully operational")
        print(f"✨ Ready for autonomous materials discovery!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo())