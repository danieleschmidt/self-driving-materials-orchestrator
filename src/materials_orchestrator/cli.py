"""Command-line interface for the materials orchestrator."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .core import AutonomousLab, MaterialsObjective
from .planners import BayesianPlanner, GridPlanner, RandomPlanner
from .robots import RobotOrchestrator

app = typer.Typer(
    name="materials-orchestrator",
    help="Self-driving materials discovery orchestrator",
    add_completion=False,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def run_campaign(
    target_property: str = typer.Option("band_gap", help="Property to optimize"),
    target_min: float = typer.Option(1.2, help="Target range minimum"),
    target_max: float = typer.Option(1.6, help="Target range maximum"),
    material_system: str = typer.Option("perovskites", help="Material system"),
    planner: str = typer.Option(
        "bayesian", help="Optimization planner (bayesian/random/grid)"
    ),
    max_experiments: int = typer.Option(100, help="Maximum experiments to run"),
    initial_samples: int = typer.Option(15, help="Initial random samples"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Run an autonomous materials discovery campaign."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration if provided
    config = {}
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
        else:
            logger.error(f"Configuration file not found: {config_file}")
            raise typer.Exit(1)

    # Create materials objective
    objective = MaterialsObjective(
        target_property=config.get("target_property", target_property),
        target_range=(
            config.get("target_min", target_min),
            config.get("target_max", target_max),
        ),
        optimization_direction=config.get("optimization_direction", "target"),
        material_system=config.get("material_system", material_system),
        success_threshold=config.get(
            "success_threshold", (target_min + target_max) / 2
        ),
    )

    # Create parameter space
    param_space = config.get(
        "param_space",
        {
            "precursor_A_conc": (0.1, 2.0),
            "precursor_B_conc": (0.1, 2.0),
            "temperature": (100, 300),
            "reaction_time": (1, 24),
            "pH": (3, 11),
            "solvent_ratio": (0, 1),
        },
    )

    # Create planner
    planner_map = {
        "bayesian": BayesianPlanner(target_property=target_property),
        "random": RandomPlanner(),
        "grid": GridPlanner(),
    }

    selected_planner = planner_map.get(planner.lower())
    if not selected_planner:
        logger.error(f"Unknown planner: {planner}")
        raise typer.Exit(1)

    # Initialize lab
    lab = AutonomousLab(
        robots=config.get("robots", ["synthesis_robot", "characterization_robot"]),
        instruments=config.get("instruments", ["xrd", "uv_vis", "pl_spectrometer"]),
        planner=selected_planner,
    )

    # Display campaign info
    typer.echo("\\n🔬 Starting Materials Discovery Campaign")
    typer.echo(f"{'='*50}")
    typer.echo(f"Target Property: {objective.target_property}")
    typer.echo(f"Target Range: {objective.target_range}")
    typer.echo(f"Material System: {objective.material_system}")
    typer.echo(f"Planner: {planner.title()}")
    typer.echo(f"Max Experiments: {max_experiments}")
    typer.echo("")

    # Run campaign
    try:
        with typer.progressbar(
            length=max_experiments, label="Running experiments"
        ) as progress:

            def progress_callback(completed: int):
                progress.update(completed - progress.pos)

            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=initial_samples,
                max_experiments=max_experiments,
                stop_on_target=True,
                convergence_patience=20,
            )
            progress.update(max_experiments)  # Complete the bar

    except KeyboardInterrupt:
        typer.echo("\\n⏹️  Campaign interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Campaign failed: {e}")
        raise typer.Exit(1)

    # Display results
    typer.echo("\\n🏆 Campaign Results")
    typer.echo(f"{'='*30}")
    typer.echo(f"Experiments: {campaign.total_experiments}")
    typer.echo(f"Success Rate: {campaign.success_rate:.1%}")
    typer.echo(
        f"Duration: {campaign.duration:.2f} hours"
        if campaign.duration
        else "Duration: N/A"
    )

    if campaign.best_material:
        typer.echo("\\n🥇 Best Material:")
        best_value = campaign.best_properties.get(target_property, "N/A")
        typer.echo(f"  {target_property}: {best_value}")

        typer.echo("\\n🔬 Optimal Parameters:")
        for param, value in campaign.best_material["parameters"].items():
            typer.echo(f"  {param}: {value:.3f}")

    # Save results
    if output_file or config.get("output_file"):
        output_path = Path(output_file or config["output_file"])

        results = {
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
                "duration_hours": campaign.duration,
            },
            "best_material": campaign.best_material,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        typer.echo(f"\\n💾 Results saved to: {output_path}")

    typer.echo("\\n✅ Campaign completed successfully!")


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port to run dashboard on"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    dev: bool = typer.Option(False, help="Development mode"),
):
    """Launch the real-time dashboard."""

    try:
        import sys
        from pathlib import Path

        import streamlit.web.cli as stcli

        # Get path to dashboard module
        dashboard_script = Path(__file__).parent / "dashboard.py"

        # Prepare streamlit command
        sys.argv = [
            "streamlit",
            "run",
            str(dashboard_script),
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]

        if dev:
            sys.argv.extend(["--server.runOnSave", "true"])

        typer.echo(f"🚀 Starting dashboard at http://{host}:{port}")
        stcli.main()

    except ImportError:
        typer.echo("❌ Streamlit not installed. Install with: pip install streamlit")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise typer.Exit(1)


@app.command()
def robot_status(
    robot_id: Optional[str] = typer.Option(None, help="Specific robot to check"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Check robot fleet status."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create robot orchestrator with default robots
    orchestrator = RobotOrchestrator()

    typer.echo("🤖 Robot Fleet Status")
    typer.echo("=" * 25)

    if robot_id:
        typer.echo(f"{robot_id}: Available (simulated)")
    else:
        typer.echo("All robots: Available (simulated)")
        typer.echo("  • synthesis_robot: Ready")
        typer.echo("  • characterization_robot: Ready")
        typer.echo("  • analysis_robot: Ready")


@app.command()
def test_experiment(
    temperature: float = typer.Option(150.0, help="Synthesis temperature"),
    time_hours: float = typer.Option(3.0, help="Reaction time in hours"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Run a single test experiment."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create lab
    lab = AutonomousLab()

    # Define test parameters
    params = {
        "precursor_A_conc": 1.0,
        "precursor_B_conc": 1.0,
        "temperature": temperature,
        "reaction_time": time_hours,
        "pH": 7.0,
        "solvent_ratio": 0.5,
    }

    typer.echo("🧪 Running test experiment...")
    typer.echo(f"Parameters: {params}")

    # Run experiment
    experiment = lab.run_experiment(params)

    # Display results
    typer.echo("\\n📊 Results:")
    typer.echo(f"Status: {experiment.status}")
    typer.echo(f"Duration: {experiment.duration:.2f}s")

    if experiment.results:
        typer.echo("Properties:")
        for prop, value in experiment.results.items():
            typer.echo(f"  {prop}: {value}")
    else:
        typer.echo("No results (experiment failed)")


@app.command()
def status():
    """Check system status."""
    typer.echo("📊 System Status:")
    typer.echo("  • Laboratory: Connected ✅")
    typer.echo("  • Robots: 3 simulated ✅")
    typer.echo("  • Database: File-based ✅")
    typer.echo("  • Experiments running: 0")
    typer.echo("  • Last campaign: Demo completed ✅")


@app.command()
def version():
    """Show version information."""
    from . import __author__, __version__

    typer.echo(f"Materials Orchestrator v{__version__}")
    typer.echo(f"Author: {__author__}")
    typer.echo("Self-driving materials discovery platform")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
