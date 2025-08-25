#!/usr/bin/env python3
"""Autonomous Research Coordination Demo.

Demonstrates end-to-end autonomous research coordination from hypothesis
generation through experimental design, execution, analysis, and publication.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from materials_orchestrator.autonomous_research_coordinator import (
    AutonomousResearchCoordinator,
    AutonomousHypothesisGenerator,
    AutonomousExperimentalDesigner,
    ResearchProject,
    ResearchPhase,
    ResearchPriority,
    initiate_autonomous_research,
    get_research_coordinator
)


async def demo_hypothesis_generation():
    """Demonstrate autonomous hypothesis generation."""
    print("🧠 Autonomous Hypothesis Generation Demo")
    print("=" * 45)
    
    generator = AutonomousHypothesisGenerator()
    
    # Define research parameters
    research_domain = "perovskite_solar_cells"
    target_properties = ["band_gap", "efficiency", "stability"]
    existing_knowledge = {
        "previous_studies": 50,
        "known_compositions": ["MAPbI3", "FAPbI3", "CsPbI3"],
        "baseline_efficiency": 20.1
    }
    
    print(f"🎯 Research Domain: {research_domain}")
    print(f"🎯 Target Properties: {', '.join(target_properties)}")
    print(f"🎯 Generating hypotheses...")
    
    start_time = time.time()
    hypotheses = await generator.generate_hypotheses(
        research_domain=research_domain,
        target_properties=target_properties,
        existing_knowledge=existing_knowledge,
        num_hypotheses=6
    )
    generation_time = time.time() - start_time
    
    print(f"✅ Generated {len(hypotheses)} hypotheses in {generation_time:.2f}s")
    
    # Display hypotheses
    print(f"\n📋 Generated Research Hypotheses:")
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n{i}. {hypothesis.title}")
        print(f"   Priority: {hypothesis.priority.value.upper()}")
        print(f"   Confidence: {hypothesis.confidence_score:.3f}")
        print(f"   Novelty: {hypothesis.novelty_score:.3f}")
        print(f"   Impact: {hypothesis.impact_score:.3f}")
        print(f"   Research Question: {hypothesis.research_question}")
        
        if hypothesis.testable_predictions:
            print(f"   Predictions: {hypothesis.testable_predictions[0]}")
        
        if hypothesis.proposed_experiments:
            exp_count = len(hypothesis.proposed_experiments)
            print(f"   Proposed Experiments: {exp_count}")
    
    return hypotheses


async def demo_experimental_design():
    """Demonstrate autonomous experimental design."""
    print(f"\n🔬 Autonomous Experimental Design Demo")
    print("-" * 40)
    
    # Use hypothesis from previous demo
    generator = AutonomousHypothesisGenerator()
    hypotheses = await generator.generate_hypotheses(
        "perovskite_solar_cells", 
        ["efficiency"], 
        {}, 
        num_hypotheses=1
    )
    
    hypothesis = hypotheses[0]
    designer = AutonomousExperimentalDesigner()
    
    print(f"🎯 Designing experiments for: {hypothesis.title}")
    
    # Set resource constraints
    resource_constraints = {
        "max_budget": 25000,  # USD
        "max_duration_days": 30,
        "max_samples": 200,
        "max_replicates": 3
    }
    
    print(f"💰 Resource Constraints:")
    for key, value in resource_constraints.items():
        print(f"   {key}: {value}")
    
    start_time = time.time()
    designs = await designer.design_experiments(hypothesis, resource_constraints)
    design_time = time.time() - start_time
    
    print(f"✅ Generated {len(designs)} experimental designs in {design_time:.2f}s")
    
    # Display experimental designs
    print(f"\n📊 Experimental Designs:")
    for i, design in enumerate(designs, 1):
        print(f"\n{i}. Design Type: {design.design_type}")
        print(f"   Sample Size: {design.sample_size}")
        print(f"   Replicates: {design.replicates}")
        print(f"   Variables: {len(design.variables)}")
        print(f"   Response Variables: {', '.join(design.response_variables[:3])}...")
        print(f"   Estimated Duration: {design.estimated_duration.days} days")
        
        resources = design.resource_requirements
        print(f"   Estimated Cost: ${resources.get('estimated_cost', 0):,.0f}")
        print(f"   Required Time: {resources.get('estimated_time_hours', 0):.1f} hours")
        
        if design.variables:
            print(f"   Key Variables:")
            for var_name, var_info in list(design.variables.items())[:3]:
                var_range = var_info.get('range', [0, 1])
                var_unit = var_info.get('unit', '')
                print(f"     {var_name}: {var_range[0]}-{var_range[1]} {var_unit}")
    
    return designs


async def demo_research_project_initiation():
    """Demonstrate full research project initiation."""
    print(f"\n🚀 Research Project Initiation Demo")
    print("-" * 38)
    
    coordinator = AutonomousResearchCoordinator()
    
    # Define research parameters
    domain = "perovskite_solar_cells"
    target_properties = ["band_gap", "efficiency", "stability"]
    research_goals = {
        "num_hypotheses": 5,
        "priority_focus": "efficiency",
        "innovation_level": "high"
    }
    
    resource_constraints = {
        "budget": 75000,
        "timeline_months": 6,
        "max_experiments": 150,
        "collaboration_allowed": True
    }
    
    print(f"🎯 Initiating research project:")
    print(f"   Domain: {domain}")
    print(f"   Target Properties: {', '.join(target_properties)}")
    print(f"   Budget: ${resource_constraints['budget']:,}")
    print(f"   Timeline: {resource_constraints['timeline_months']} months")
    
    start_time = time.time()
    project = await coordinator.initiate_research_project(
        domain=domain,
        target_properties=target_properties,
        research_goals=research_goals,
        resource_constraints=resource_constraints
    )
    initiation_time = time.time() - start_time
    
    print(f"✅ Research project initiated in {initiation_time:.2f}s")
    print(f"📋 Project Details:")
    print(f"   Project ID: {project.project_id}")
    print(f"   Title: {project.title}")
    print(f"   Phase: {project.phase.value}")
    print(f"   Priority: {project.priority.value}")
    print(f"   Hypotheses Generated: {len(project.hypotheses)}")
    print(f"   Experimental Designs: {len(project.experimental_designs)}")
    
    # Show hypothesis summary
    if project.hypotheses:
        print(f"\n🧠 Top Hypotheses:")
        for i, hypothesis in enumerate(project.hypotheses[:3], 1):
            print(f"   {i}. {hypothesis.title}")
            print(f"      Priority: {hypothesis.priority.value}")
            print(f"      Confidence: {hypothesis.confidence_score:.3f}")
    
    # Show experimental design summary
    if project.experimental_designs:
        print(f"\n🔬 Experimental Designs:")
        total_samples = sum(design.sample_size for design in project.experimental_designs)
        total_cost = sum(design.resource_requirements.get('estimated_cost', 0) 
                        for design in project.experimental_designs)
        print(f"   Total Samples: {total_samples}")
        print(f"   Total Estimated Cost: ${total_cost:,.0f}")
        print(f"   Budget Utilization: {(total_cost / resource_constraints['budget']) * 100:.1f}%")
    
    return project


async def demo_research_execution():
    """Demonstrate autonomous research project execution."""
    print(f"\n⚡ Research Project Execution Demo")
    print("-" * 36)
    
    coordinator = get_research_coordinator()
    
    # Create a research project
    print(f"🚀 Creating research project for execution...")
    project = await initiate_autonomous_research(
        domain="perovskite_solar_cells",
        target_properties=["efficiency"],
        research_goals={"num_hypotheses": 3},
        resource_constraints={"budget": 50000, "timeline_months": 3}
    )
    
    print(f"✅ Project created: {project.project_id}")
    print(f"📊 Starting autonomous execution...")
    
    start_time = time.time()
    execution_results = await coordinator.execute_research_project(
        project.project_id,
        simulation_mode=True  # Use simulation for demo
    )
    execution_time = time.time() - start_time
    
    print(f"✅ Research execution completed in {execution_time:.2f}s")
    print(f"📈 Execution Results:")
    print(f"   Final Status: {execution_results['final_status'].upper()}")
    print(f"   Execution Steps: {len(execution_results['execution_log'])}")
    print(f"   Discoveries: {len(execution_results['discoveries'])}")
    print(f"   Publications Generated: {len(execution_results['publications'])}")
    
    # Show discoveries
    if execution_results['discoveries']:
        print(f"\n🔍 Key Discoveries:")
        for i, discovery in enumerate(execution_results['discoveries'][:3], 1):
            disc_type = discovery.get('type', 'unknown').title()
            prop = discovery.get('property', 'unknown')
            value = discovery.get('value', 0)
            confidence = discovery.get('confidence', 0)
            print(f"   {i}. {disc_type}: {prop} = {value:.3f}")
            print(f"      Confidence: {confidence:.3f}")
            print(f"      Significance: {discovery.get('significance', 'N/A')}")
    
    # Show publications
    if execution_results['publications']:
        print(f"\n📚 Generated Publications:")
        for i, pub in enumerate(execution_results['publications'], 1):
            title = pub.get('title', 'Untitled')
            journal = pub.get('target_journal', 'Unknown Journal')
            impact = pub.get('estimated_impact_factor', 0)
            significance = pub.get('significance_level', 'unknown')
            print(f"   {i}. {title}")
            print(f"      Target Journal: {journal}")
            print(f"      Impact Factor: {impact}")
            print(f"      Significance: {significance.upper()}")
    
    return execution_results


async def demo_research_coordination_status():
    """Demonstrate research coordination status and reporting."""
    print(f"\n📊 Research Coordination Status Demo")
    print("-" * 38)
    
    coordinator = get_research_coordinator()
    
    # Check current status
    status = coordinator.get_research_status()
    
    print(f"🔍 Current Research Status:")
    print(f"   Active Projects: {status['active_projects']}")
    print(f"   Completed Projects: {status['completed_projects']}")
    print(f"   Total Discoveries: {status['total_discoveries']}")
    print(f"   Total Publications: {status['total_publications']}")
    print(f"   Success Rate: {status['success_rate']:.1%}")
    
    if status['research_domains']:
        print(f"\n🔬 Active Research Domains:")
        for domain in status['research_domains']:
            print(f"   • {domain.replace('_', ' ').title()}")
    
    # Generate comprehensive report
    print(f"\n📄 Generating comprehensive research report...")
    report = await coordinator.generate_research_report()
    
    print(f"✅ Research report generated")
    print(f"📋 Report Excerpt:")
    print("-" * 20)
    
    # Show first part of report
    report_lines = report.split('\n')
    for line in report_lines[:15]:
        print(f"   {line}")
    
    if len(report_lines) > 15:
        print(f"   ... (report continues for {len(report_lines) - 15} more lines)")
    
    return report


async def demo_multi_project_coordination():
    """Demonstrate coordination of multiple research projects."""
    print(f"\n🌐 Multi-Project Coordination Demo")
    print("-" * 35)
    
    coordinator = get_research_coordinator()
    
    # Launch multiple research projects
    research_domains = [
        ("perovskite_solar_cells", ["efficiency", "stability"]),
        ("battery_materials", ["capacity", "cycle_life"]),
        ("catalysts", ["activity", "selectivity"])
    ]
    
    print(f"🚀 Launching {len(research_domains)} research projects...")
    
    projects = []
    for domain, properties in research_domains:
        project = await initiate_autonomous_research(
            domain=domain,
            target_properties=properties,
            research_goals={"num_hypotheses": 3},
            resource_constraints={"budget": 30000, "timeline_months": 4}
        )
        projects.append(project)
        print(f"   ✅ {domain} project initiated")
    
    # Execute all projects concurrently (limited concurrency)
    print(f"\n⚡ Executing projects concurrently...")
    
    start_time = time.time()
    execution_tasks = [
        coordinator.execute_research_project(project.project_id, simulation_mode=True)
        for project in projects[:2]  # Limit for demo
    ]
    
    results = await asyncio.gather(*execution_tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    print(f"✅ Multi-project execution completed in {execution_time:.2f}s")
    
    # Summarize results
    successful_projects = [r for r in results if not isinstance(r, Exception)]
    failed_projects = [r for r in results if isinstance(r, Exception)]
    
    print(f"📈 Multi-Project Summary:")
    print(f"   Successful Projects: {len(successful_projects)}")
    print(f"   Failed Projects: {len(failed_projects)}")
    
    if successful_projects:
        total_discoveries = sum(len(r['discoveries']) for r in successful_projects)
        total_publications = sum(len(r['publications']) for r in successful_projects)
        
        print(f"   Total Discoveries: {total_discoveries}")
        print(f"   Total Publications: {total_publications}")
        
        # Show cross-project insights
        print(f"\n🔍 Cross-Project Insights:")
        all_discoveries = []
        for result in successful_projects:
            all_discoveries.extend(result['discoveries'])
        
        discovery_types = {}
        for discovery in all_discoveries:
            disc_type = discovery.get('type', 'unknown')
            discovery_types[disc_type] = discovery_types.get(disc_type, 0) + 1
        
        for disc_type, count in discovery_types.items():
            print(f"   {disc_type.title()}: {count} discoveries")
    
    # Final coordination status
    final_status = coordinator.get_research_status()
    print(f"\n📊 Final Coordination Status:")
    print(f"   Total Projects Managed: {final_status['total_projects']}")
    print(f"   Overall Success Rate: {final_status['success_rate']:.1%}")
    print(f"   Research Domains: {len(final_status['research_domains'])}")


async def run_autonomous_research_demo():
    """Run comprehensive autonomous research coordination demo."""
    print("🌟 Autonomous Research Coordination - Complete Demo")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    try:
        # Run all demo sections
        await demo_hypothesis_generation()
        await demo_experimental_design()
        await demo_research_project_initiation()
        await demo_research_execution()
        await demo_research_coordination_status()
        await demo_multi_project_coordination()
        
        total_duration = time.time() - total_start
        
        print(f"\n🎉 Autonomous Research Coordination Demo Complete!")
        print(f"⏱️  Total demo duration: {total_duration:.2f} seconds")
        print(f"🧠 Autonomous hypothesis generation ✅")
        print(f"🔬 Intelligent experimental design ✅") 
        print(f"⚡ Automated research execution ✅")
        print(f"🔍 Discovery identification & validation ✅")
        print(f"📚 Publication generation ✅")
        print(f"🌐 Multi-project coordination ✅")
        print(f"✨ Full autonomous research lifecycle operational!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive autonomous research demo
    asyncio.run(run_autonomous_research_demo())