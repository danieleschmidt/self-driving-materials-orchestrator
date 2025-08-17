"""Utility functions for dashboard components."""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_campaign_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load campaign data from JSON file.

    Args:
        file_path: Path to campaign results file

    Returns:
        Campaign data dictionary or None if loading fails
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load campaign data from {file_path}: {e}")
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_property_evolution_plot(
    experiments: List[Dict[str, Any]],
    property_name: str = "band_gap",
    target_value: Optional[float] = None,
    target_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Create property evolution plot over experiments.

    Args:
        experiments: List of experiment data
        property_name: Property to plot
        target_value: Target value to show as horizontal line
        target_range: Target range to show as shaded area

    Returns:
        Plotly figure
    """
    # Filter completed experiments with results
    completed_experiments = [
        exp
        for exp in experiments
        if exp.get("status") == "completed"
        and exp.get("results", {}).get(property_name) is not None
    ]

    if not completed_experiments:
        # Return empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Extract data
    x_values = list(range(1, len(completed_experiments) + 1))
    y_values = [exp["results"][property_name] for exp in completed_experiments]

    # Create plot
    fig = go.Figure()

    # Add target range as shaded area
    if target_range:
        fig.add_hrect(
            y0=target_range[0],
            y1=target_range[1],
            fillcolor="lightgreen",
            opacity=0.2,
            annotation_text="Target Range",
            annotation_position="top left",
        )

    # Add main line plot
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name=property_name.replace("_", " ").title(),
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    # Add target value line
    if target_value:
        fig.add_hline(
            y=target_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {target_value}",
        )

    # Update layout
    fig.update_layout(
        title=f"{property_name.replace('_', ' ').title()} Evolution",
        xaxis_title="Experiment Number",
        yaxis_title=property_name.replace("_", " ").title(),
        height=400,
        showlegend=True,
    )

    return fig


def create_parameter_space_plot(
    experiments: List[Dict[str, Any]],
    x_param: str,
    y_param: str,
    color_property: str = "band_gap",
    size_property: Optional[str] = None,
) -> go.Figure:
    """Create parameter space exploration plot.

    Args:
        experiments: List of experiment data
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        color_property: Property to use for color mapping
        size_property: Property to use for marker size

    Returns:
        Plotly figure
    """
    # Filter valid experiments
    valid_experiments = []
    for exp in experiments:
        if (
            exp.get("status") == "completed"
            and exp.get("parameters", {}).get(x_param) is not None
            and exp.get("parameters", {}).get(y_param) is not None
            and exp.get("results", {}).get(color_property) is not None
        ):

            valid_experiments.append(exp)

    if not valid_experiments:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for parameter space plot",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    # Extract data
    x_values = [exp["parameters"][x_param] for exp in valid_experiments]
    y_values = [exp["parameters"][y_param] for exp in valid_experiments]
    color_values = [exp["results"][color_property] for exp in valid_experiments]

    # Size values (optional)
    size_values = None
    if size_property:
        size_values = [
            exp["results"].get(size_property, 10) for exp in valid_experiments
        ]
        # Normalize size values
        if size_values:
            min_size, max_size = min(size_values), max(size_values)
            if max_size > min_size:
                size_values = [
                    10 + 20 * (val - min_size) / (max_size - min_size)
                    for val in size_values
                ]
            else:
                size_values = [15] * len(size_values)

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                size=size_values if size_values else 8,
                color=color_values,
                colorscale="viridis",
                showscale=True,
                colorbar=dict(title=color_property.replace("_", " ").title()),
            ),
            name="Experiments",
            text=[f"Exp {i+1}" for i in range(len(valid_experiments))],
            hovertemplate=(
                f"{x_param}: %{{x:.3f}}<br>"
                f"{y_param}: %{{y:.3f}}<br>"
                f"{color_property}: %{{marker.color:.3f}}"
                + (
                    f"<br>{size_property}: %{{marker.size:.1f}}"
                    if size_property
                    else ""
                )
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Parameter Space: {x_param} vs {y_param}",
        xaxis_title=x_param.replace("_", " ").title(),
        yaxis_title=y_param.replace("_", " ").title(),
        height=500,
    )

    return fig


def calculate_success_metrics(
    experiments: List[Dict[str, Any]], success_criteria: Dict[str, Any]
) -> Dict[str, float]:
    """Calculate success metrics for experiments.

    Args:
        experiments: List of experiment data
        success_criteria: Dictionary defining success criteria

    Returns:
        Dictionary of success metrics
    """
    completed_experiments = [
        exp for exp in experiments if exp.get("status") == "completed"
    ]

    if not completed_experiments:
        return {
            "total_experiments": len(experiments),
            "completed_experiments": 0,
            "success_rate": 0.0,
            "target_hit_rate": 0.0,
        }

    # Count successful experiments based on criteria
    successful_count = 0
    target_hits = 0

    for exp in completed_experiments:
        results = exp.get("results", {})
        is_successful = True
        hits_target = True

        for property_name, criteria in success_criteria.items():
            value = results.get(property_name)
            if value is None:
                is_successful = False
                hits_target = False
                continue

            # Check range criteria
            if "range" in criteria:
                min_val, max_val = criteria["range"]
                if not (min_val <= value <= max_val):
                    is_successful = False

            # Check target criteria (exact match within tolerance)
            if "target" in criteria:
                target = criteria["target"]
                tolerance = criteria.get("tolerance", 0.1)
                if abs(value - target) > tolerance:
                    hits_target = False

            # Check minimum/maximum criteria
            if "min" in criteria and value < criteria["min"]:
                is_successful = False
            if "max" in criteria and value > criteria["max"]:
                is_successful = False

        if is_successful:
            successful_count += 1
        if hits_target:
            target_hits += 1

    return {
        "total_experiments": len(experiments),
        "completed_experiments": len(completed_experiments),
        "successful_experiments": successful_count,
        "success_rate": successful_count / len(completed_experiments),
        "target_hits": target_hits,
        "target_hit_rate": target_hits / len(completed_experiments),
        "completion_rate": len(completed_experiments) / len(experiments),
    }


def generate_experiment_summary(experiments: List[Dict[str, Any]]) -> str:
    """Generate text summary of experiments.

    Args:
        experiments: List of experiment data

    Returns:
        Formatted summary string
    """
    if not experiments:
        return "No experiments found."

    total = len(experiments)
    completed = len([exp for exp in experiments if exp.get("status") == "completed"])
    failed = len([exp for exp in experiments if exp.get("status") == "failed"])
    running = len([exp for exp in experiments if exp.get("status") == "running"])

    # Find best results
    best_results = {}
    for exp in experiments:
        if exp.get("status") == "completed":
            results = exp.get("results", {})
            for prop, value in results.items():
                if isinstance(value, (int, float)):
                    if prop not in best_results or value > best_results[prop]:
                        best_results[prop] = value

    summary_lines = [
        f"📊 **Experiment Summary**",
        f"• Total experiments: {total}",
        f"• Completed: {completed} ({completed/total:.1%})",
        f"• Failed: {failed}",
        f"• Running: {running}",
        "",
        f"🏆 **Best Results**",
    ]

    for prop, value in best_results.items():
        summary_lines.append(f"• Best {prop.replace('_', ' ')}: {value:.3f}")

    return "\n".join(summary_lines)


def export_data_to_csv(
    experiments: List[Dict[str, Any]], filename: str = "experiments.csv"
) -> bool:
    """Export experiment data to CSV file.

    Args:
        experiments: List of experiment data
        filename: Output filename

    Returns:
        True if export successful, False otherwise
    """
    try:
        # Flatten experiment data
        flattened_data = []
        for exp in experiments:
            row = {
                "id": exp.get("id", ""),
                "timestamp": exp.get("timestamp", ""),
                "status": exp.get("status", ""),
                "duration": exp.get("duration", 0),
            }

            # Add parameters
            parameters = exp.get("parameters", {})
            for param, value in parameters.items():
                row[f"param_{param}"] = value

            # Add results
            results = exp.get("results", {})
            for result, value in results.items():
                row[f"result_{result}"] = value

            flattened_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)

        logger.info(f"Exported {len(experiments)} experiments to {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return False
