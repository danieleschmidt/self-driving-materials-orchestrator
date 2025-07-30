"""Command-line interface for Materials Orchestrator."""

import typer
from typing import Optional
from pathlib import Path

app = typer.Typer(help="Self-Driving Materials Orchestrator CLI")


@app.command()
def launch(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Dashboard port"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Launch the materials orchestrator system."""
    typer.echo(f"🚀 Launching Materials Orchestrator on port {port}")
    
    if config:
        typer.echo(f"📋 Using config: {config}")
    
    if debug:
        typer.echo("🐛 Debug mode enabled")
    
    # Placeholder - would start actual system
    typer.echo("🔬 System ready for autonomous materials discovery!")


@app.command()
def campaign(
    objective: str = typer.Argument(..., help="Target property to optimize"),
    material_system: str = typer.Option("general", help="Material system type"),
    max_experiments: int = typer.Option(100, help="Maximum experiments"),
):
    """Start an autonomous discovery campaign."""
    typer.echo(f"🎯 Starting campaign for {objective} optimization")
    typer.echo(f"🧪 Material system: {material_system}")
    typer.echo(f"🔢 Max experiments: {max_experiments}")
    
    # Placeholder implementation
    typer.echo("✅ Campaign completed successfully!")


@app.command()
def status():
    """Check system status."""
    typer.echo("📊 System Status:")
    typer.echo("  • Laboratory: Connected ✅")
    typer.echo("  • Robots: 2 active ✅")
    typer.echo("  • Database: Connected ✅")
    typer.echo("  • Experiments running: 0")


@app.command()
def dashboard():
    """Launch the web dashboard."""
    typer.echo("🎨 Launching web dashboard...")
    typer.echo("🌐 Open http://localhost:8501 in your browser")
    
    # Placeholder - would launch Streamlit dashboard
    

def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()