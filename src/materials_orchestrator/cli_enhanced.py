"""Enhanced CLI with comprehensive commands for materials discovery."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import typer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_orchestrator import (
    AutonomousLab,
    BayesianPlanner,
    MaterialsObjective,
    RandomPlanner,
    create_database,
)

app = typer.Typer(help="🔬 Self-Driving Materials Orchestrator CLI")

# Global configuration
config = {
    "database_url": "mongodb://localhost:27017/",
    "log_level": "INFO",
    "output_dir": Path("./experiments"),
}


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def run_campaign(
    target_property: str = typer.Argument(..., help="Target property to optimize"),
    target_min: float = typer.Argument(..., help="Minimum target value"),
    target_max: float = typer.Argument(..., help="Maximum target value"),
    param_space_file: str = typer.Option(
        None, "--params", "-p", help="JSON file with parameter space"
    ),
    max_experiments: int = typer.Option(
        100, "--max-exp", "-m", help="Maximum experiments"
    ),
    initial_samples: int = typer.Option(
        20, "--init", "-i", help="Initial random samples"
    ),
    planner: str = typer.Option("bayesian", "--planner", help="Optimization planner"),
    material_system: str = typer.Option(
        "general", "--system", "-s", help="Material system"
    ),
    output_file: str = typer.Option(None, "--output", "-o", help="Output results file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """🚀 Run autonomous materials discovery campaign."""

    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")

    typer.echo("🔬 Starting materials discovery campaign")
    typer.echo(f"   Target: {target_property} in [{target_min}, {target_max}]")
    typer.echo(f"   System: {material_system}")
    typer.echo(f"   Planner: {planner}")

    # Define objective
    objective = MaterialsObjective(
        target_property=target_property,
        target_range=(target_min, target_max),
        optimization_direction="target",
        material_system=material_system,
        success_threshold=(target_min + target_max) / 2,
    )

    # Load parameter space
    if param_space_file:
        with open(param_space_file) as f:
            param_space = json.load(f)
    else:
        # Default parameter space for common materials
        param_space = _get_default_param_space(material_system)

    typer.echo(f"📊 Parameter space: {len(param_space)} parameters")

    # Setup planner
    if planner == "bayesian":
        exp_planner = BayesianPlanner(
            target_property=target_property, acquisition_function="expected_improvement"
        )
    elif planner == "random":
        exp_planner = RandomPlanner()
    else:
        exp_planner = BayesianPlanner(target_property=target_property)

    # Initialize lab
    lab = AutonomousLab(
        robots=["synthesis_robot", "characterization_robot"],
        instruments=["xrd", "uv_vis", "spectrometer"],
        planner=exp_planner,
    )

    # Run campaign
    with typer.progressbar(
        length=max_experiments, label="Running experiments"
    ) as progress:
        campaign = lab.run_campaign(
            objective=objective,
            param_space=param_space,
            initial_samples=initial_samples,
            max_experiments=max_experiments,
            stop_on_target=True,
            convergence_patience=20,
        )
        progress.update(campaign.total_experiments)

    # Display results
    typer.echo("\n" + "=" * 50)
    typer.echo("🏆 CAMPAIGN RESULTS")
    typer.echo("=" * 50)

    typer.echo(f"Campaign ID: {campaign.campaign_id}")
    typer.echo(f"Total experiments: {campaign.total_experiments}")
    typer.echo(f"Success rate: {campaign.success_rate:.1%}")
    typer.echo(
        f"Duration: {campaign.duration:.2f} hours"
        if campaign.duration
        else "Duration: < 0.01 hours"
    )

    if campaign.best_material:
        typer.echo("\n🥇 Best Material:")
        for prop, value in campaign.best_properties.items():
            typer.echo(f"   {prop}: {value}")

        typer.echo("\n🔬 Optimal Parameters:")
        for param, value in campaign.best_material["parameters"].items():
            typer.echo(f"   {param}: {value:.3f}")

    # Save results
    output_path = output_file or f"campaign_{campaign.campaign_id[:8]}.json"
    results_data = {
        "campaign_id": campaign.campaign_id,
        "objective": {
            "target_property": objective.target_property,
            "target_range": objective.target_range,
            "material_system": objective.material_system,
        },
        "configuration": {
            "max_experiments": max_experiments,
            "initial_samples": initial_samples,
            "planner": planner,
        },
        "results": {
            "total_experiments": campaign.total_experiments,
            "success_rate": campaign.success_rate,
            "best_properties": campaign.best_properties,
            "convergence_history": campaign.convergence_history,
        },
        "best_material": campaign.best_material,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    typer.echo(f"\n💾 Results saved to: {output_path}")
    typer.echo("✅ Campaign completed successfully!")


@app.command()
def analyze_results(
    results_file: str = typer.Argument(..., help="Results JSON file to analyze"),
    show_plots: bool = typer.Option(False, "--plots", help="Show analysis plots"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose analysis"),
) -> None:
    """📊 Analyze campaign results and generate insights."""

    typer.echo(f"📊 Analyzing results from: {results_file}")

    # Load results
    try:
        with open(results_file) as f:
            data = json.load(f)
    except Exception as e:
        typer.echo(f"❌ Error loading results: {e}")
        return

    # Basic statistics
    typer.echo("\n📈 Campaign Statistics:")
    typer.echo(f"   Campaign ID: {data.get('campaign_id', 'Unknown')}")
    typer.echo(f"   Total experiments: {data['results']['total_experiments']}")
    typer.echo(f"   Success rate: {data['results']['success_rate']:.1%}")

    # Best material analysis
    if "best_material" in data and data["best_material"]:
        typer.echo("\n🥇 Best Material Analysis:")
        properties = data["best_material"]["properties"]
        parameters = data["best_material"]["parameters"]

        for prop, value in properties.items():
            typer.echo(f"   {prop}: {value}")

        if verbose:
            typer.echo("\n🔬 Parameter Analysis:")
            for param, value in parameters.items():
                typer.echo(f"   {param}: {value}")

    # Convergence analysis
    convergence = data["results"].get("convergence_history", [])
    if convergence and len(convergence) > 5:
        typer.echo("\n🎯 Convergence Analysis:")
        initial_fitness = convergence[0]["best_fitness"]
        final_fitness = convergence[-1]["best_fitness"]
        improvement = final_fitness - initial_fitness

        typer.echo(f"   Initial fitness: {initial_fitness:.3f}")
        typer.echo(f"   Final fitness: {final_fitness:.3f}")
        typer.echo(f"   Total improvement: {improvement:.3f}")

        # Find convergence point
        best_fitness = final_fitness
        convergence_point = len(convergence)
        for i, point in enumerate(convergence):
            if abs(point["best_fitness"] - best_fitness) < 0.01:
                convergence_point = i + 1
                break

        typer.echo(f"   Experiments to convergence: {convergence_point}")

        if show_plots:
            _show_convergence_plot(convergence)

    # Performance metrics
    typer.echo("\n⚡ Performance Metrics:")
    traditional_estimate = 200  # Typical for grid search
    acceleration = traditional_estimate / max(data["results"]["total_experiments"], 1)
    typer.echo(f"   Acceleration factor: {acceleration:.1f}x")

    cost_savings = (
        1 - data["results"]["total_experiments"] / traditional_estimate
    ) * 100
    typer.echo(f"   Estimated cost savings: {cost_savings:.0f}%")


def _get_default_param_space(material_system: str) -> Dict[str, List[float]]:
    """Get default parameter space for material system."""
    if material_system == "perovskites":
        return {
            "precursor_A_conc": [0.1, 2.0],
            "precursor_B_conc": [0.1, 2.0],
            "temperature": [100, 300],
            "reaction_time": [1, 24],
            "pH": [3, 11],
            "solvent_ratio": [0, 1],
        }
    elif material_system == "catalysts":
        return {
            "metal_loading": [0.1, 5.0],
            "support_ratio": [0.1, 1.0],
            "calcination_temp": [200, 600],
            "calcination_time": [1, 8],
            "pH": [2, 12],
        }
    else:  # general
        return {
            "temperature": [100, 500],
            "pressure": [1, 10],
            "concentration": [0.1, 5.0],
            "reaction_time": [0.5, 24],
            "pH": [1, 14],
        }


def _show_convergence_plot(convergence: List[Dict]) -> None:
    """Show convergence plot (simplified text version)."""
    typer.echo("\n📈 Convergence Plot (simplified):")

    fitness_values = [point["best_fitness"] for point in convergence]
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)

    if max_fitness == min_fitness:
        typer.echo("   No variation in fitness values")
        return

    # Normalize values for display
    normalized = [
        (f - min_fitness) / (max_fitness - min_fitness) for f in fitness_values
    ]

    # Simple text plot
    width = 50
    for i, norm_val in enumerate(
        normalized[:: max(1, len(normalized) // 20)]
    ):  # Sample points
        bar_length = int(norm_val * width)
        bar = "█" * bar_length + "░" * (width - bar_length)
        typer.echo(
            f"   {i*max(1, len(normalized)//20):3d}: {bar} {fitness_values[i*max(1, len(normalized)//20)]:.3f}"
        )


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Dashboard port"),
    host: str = typer.Option("localhost", "--host", help="Dashboard host"),
) -> None:
    """🖥️ Launch interactive dashboard."""

    typer.echo(f"🖥️ Starting dashboard on http://{host}:{port}")

    try:
        import subprocess
        import sys

        import streamlit

        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]

        subprocess.run(cmd)

    except ImportError:
        typer.echo("❌ Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        typer.echo(f"❌ Dashboard failed to start: {e}")


@app.command()
def health_check() -> None:
    """🩺 Perform system health check."""

    typer.echo("🩺 Performing system health check...")

    # Check core imports
    try:
        from materials_orchestrator import AutonomousLab

        typer.echo("✅ Core modules: OK")
    except Exception as e:
        typer.echo(f"❌ Core modules: {e}")
        return

    # Check dependencies
    deps = {
        "numpy": "Scientific computing",
        "scipy": "Scientific algorithms",
        "scikit-learn": "Machine learning",
        "pymongo": "Database connectivity",
        "streamlit": "Dashboard interface",
    }

    for dep, desc in deps.items():
        try:
            __import__(dep)
            typer.echo(f"✅ {dep}: OK ({desc})")
        except ImportError:
            typer.echo(f"⚠️  {dep}: Missing ({desc}) - using fallback")

    # Test basic functionality
    try:
        lab = AutonomousLab()
        objective = MaterialsObjective(
            target_property="test_prop", target_range=(1.0, 2.0)
        )
        typer.echo("✅ Basic functionality: OK")
    except Exception as e:
        typer.echo(f"❌ Basic functionality: {e}")

    # Check database connectivity
    try:
        db = create_database()
        typer.echo("✅ Database: OK (using fallback)")
    except Exception as e:
        typer.echo(f"⚠️  Database: Using file storage ({e})")

    typer.echo("\n🎉 Health check completed!")


@app.command()
def create_param_space(
    material_system: str = typer.Argument(..., help="Material system type"),
    output_file: str = typer.Option(
        "param_space.json", "--output", "-o", help="Output file"
    ),
) -> None:
    """📋 Create parameter space template for material system."""

    typer.echo(f"📋 Creating parameter space for: {material_system}")

    param_space = _get_default_param_space(material_system)

    # Convert to proper format
    formatted_space = {}
    for param, (low, high) in param_space.items():
        formatted_space[param] = [low, high]

    template = {
        "material_system": material_system,
        "parameter_space": formatted_space,
        "description": f"Parameter space template for {material_system}",
        "created": datetime.now().isoformat(),
    }

    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)

    typer.echo(f"✅ Parameter space saved to: {output_file}")
    typer.echo(f"   Parameters: {list(param_space.keys())}")


@app.command()
def list_examples() -> None:
    """📚 List available examples and templates."""

    typer.echo("📚 Available Examples:")

    examples = [
        {
            "name": "perovskite_discovery",
            "description": "Autonomous perovskite band gap optimization",
            "material_system": "perovskites",
            "target_property": "band_gap",
        },
        {
            "name": "catalyst_optimization",
            "description": "Catalyst activity optimization",
            "material_system": "catalysts",
            "target_property": "activity",
        },
        {
            "name": "battery_materials",
            "description": "Battery electrode material discovery",
            "material_system": "battery_materials",
            "target_property": "capacity",
        },
    ]

    for example in examples:
        typer.echo(f"\n🧪 {example['name']}")
        typer.echo(f"   Description: {example['description']}")
        typer.echo(f"   System: {example['material_system']}")
        typer.echo(f"   Target: {example['target_property']}")
        typer.echo(
            f"   Command: materials-orchestrator run-campaign {example['target_property']} 0.0 2.0 --system {example['material_system']}"
        )


if __name__ == "__main__":
    app()
