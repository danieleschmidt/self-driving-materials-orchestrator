"""Reusable dashboard components."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_campaign_overview(
    experiments: List[Dict[str, Any]], campaign_id: Optional[str] = None
):
    """Render campaign overview section."""
    st.subheader("📊 Campaign Overview")

    if not experiments:
        st.info("No experiments found for this campaign")
        return

    # Campaign summary statistics
    total_experiments = len(experiments)
    successful_experiments = len(
        [exp for exp in experiments if exp.get("status") == "completed"]
    )
    failed_experiments = len(
        [exp for exp in experiments if exp.get("status") == "failed"]
    )
    running_experiments = len(
        [exp for exp in experiments if exp.get("status") == "running"]
    )

    # Display summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", total_experiments)
    with col2:
        st.metric(
            "Successful",
            successful_experiments,
            delta=(
                f"{successful_experiments/total_experiments:.1%}"
                if total_experiments > 0
                else "0%"
            ),
        )
    with col3:
        st.metric("Failed", failed_experiments)
    with col4:
        st.metric("Running", running_experiments)

    # Progress visualization
    if total_experiments > 0:
        progress_data = {
            "Status": ["Completed", "Failed", "Running", "Pending"],
            "Count": [
                successful_experiments,
                failed_experiments,
                running_experiments,
                max(
                    0,
                    total_experiments
                    - successful_experiments
                    - failed_experiments
                    - running_experiments,
                ),
            ],
        }

        fig = px.pie(
            values=progress_data["Count"],
            names=progress_data["Status"],
            title="Experiment Status Distribution",
            color_discrete_map={
                "Completed": "#28a745",
                "Failed": "#dc3545",
                "Running": "#ffc107",
                "Pending": "#6c757d",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    # Best materials found
    st.write("### 🏆 Best Materials Discovered")

    completed_experiments = [
        exp for exp in experiments if exp.get("status") == "completed"
    ]

    if completed_experiments:
        # Sort by band gap (closest to 1.4 eV target)
        target_bandgap = 1.4
        best_materials = []

        for exp in completed_experiments:
            results = exp.get("results", {})
            bandgap = results.get("band_gap")
            if bandgap is not None:
                score = -abs(
                    bandgap - target_bandgap
                )  # Negative for sorting (higher is better)
                best_materials.append(
                    {
                        "experiment_id": exp.get("id", "unknown")[:8],
                        "band_gap": bandgap,
                        "efficiency": results.get("efficiency", 0),
                        "stability": results.get("stability", 0),
                        "score": score,
                        "parameters": exp.get("parameters", {}),
                    }
                )

        best_materials.sort(key=lambda x: x["score"], reverse=True)

        # Display top 5 materials
        for i, material in enumerate(best_materials[:5]):
            with st.expander(
                f"#{i+1} - Experiment {material['experiment_id']} (Band gap: {material['band_gap']:.3f} eV)"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Properties:**")
                    st.write(f"Band gap: {material['band_gap']:.3f} eV")
                    st.write(f"Efficiency: {material['efficiency']:.2f}%")
                    st.write(f"Stability: {material['stability']:.3f}")

                with col2:
                    st.write("**Parameters:**")
                    for param, value in material["parameters"].items():
                        if isinstance(value, (int, float)):
                            st.write(f"{param}: {value:.3f}")
                        else:
                            st.write(f"{param}: {value}")
    else:
        st.info("No completed experiments with results found")


class CampaignMonitor:
    """Real-time campaign monitoring component."""

    def __init__(self, campaign_id: str):
        """Initialize campaign monitor.

        Args:
            campaign_id: Campaign to monitor
        """
        self.campaign_id = campaign_id
        self._last_update = None

    def render(self, experiments: List[Dict[str, Any]]):
        """Render campaign monitoring interface."""
        st.subheader(f"📈 Campaign Monitor: {self.campaign_id}")

        # Real-time metrics
        self._render_realtime_metrics(experiments)

        # Property evolution chart
        self._render_property_evolution(experiments)

        # Parameter optimization heatmap
        self._render_parameter_heatmap(experiments)

    def _render_realtime_metrics(self, experiments: List[Dict[str, Any]]):
        """Render real-time metrics display."""
        if not experiments:
            return

        # Calculate recent performance
        recent_experiments = (
            experiments[-10:] if len(experiments) >= 10 else experiments
        )
        recent_success_rate = len(
            [exp for exp in recent_experiments if exp.get("status") == "completed"]
        ) / len(recent_experiments)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Recent Success Rate", f"{recent_success_rate:.1%}")

        with col2:
            if recent_experiments:
                latest_result = recent_experiments[-1].get("results", {})
                latest_bandgap = latest_result.get("band_gap", 0)
                st.metric("Latest Band Gap", f"{latest_bandgap:.3f} eV")

        with col3:
            st.metric(
                "Experiments Today",
                len(
                    [
                        exp
                        for exp in experiments
                        if self._is_today(exp.get("timestamp", ""))
                    ]
                ),
            )

    def _render_property_evolution(self, experiments: List[Dict[str, Any]]):
        """Render property evolution over time."""
        completed_experiments = [
            exp for exp in experiments if exp.get("status") == "completed"
        ]

        if not completed_experiments:
            return

        # Prepare data
        data = []
        for i, exp in enumerate(completed_experiments):
            results = exp.get("results", {})
            if "band_gap" in results:
                data.append(
                    {
                        "experiment": i + 1,
                        "band_gap": results["band_gap"],
                        "efficiency": results.get("efficiency", 0),
                        "timestamp": exp.get("timestamp", ""),
                    }
                )

        if data:
            df = pd.DataFrame(data)

            fig = go.Figure()

            # Band gap line
            fig.add_trace(
                go.Scatter(
                    x=df["experiment"],
                    y=df["band_gap"],
                    mode="lines+markers",
                    name="Band Gap (eV)",
                    line=dict(color="blue", width=2),
                )
            )

            # Add target range
            fig.add_hline(
                y=1.2,
                line_dash="dash",
                line_color="green",
                annotation_text="Target Min",
            )
            fig.add_hline(
                y=1.6,
                line_dash="dash",
                line_color="green",
                annotation_text="Target Max",
            )
            fig.add_hline(
                y=1.4, line_dash="dot", line_color="red", annotation_text="Optimal"
            )

            fig.update_layout(
                title="Band Gap Evolution",
                xaxis_title="Experiment Number",
                yaxis_title="Band Gap (eV)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_parameter_heatmap(self, experiments: List[Dict[str, Any]]):
        """Render parameter optimization heatmap."""
        completed_experiments = [
            exp for exp in experiments if exp.get("status") == "completed"
        ]

        if len(completed_experiments) < 5:
            return

        # Extract parameter data
        data = []
        for exp in completed_experiments:
            parameters = exp.get("parameters", {})
            results = exp.get("results", {})

            if "band_gap" in results:
                row = parameters.copy()
                row["band_gap"] = results["band_gap"]
                data.append(row)

        if data:
            df = pd.DataFrame(data)

            # Get numeric parameters
            numeric_params = [
                col
                for col in df.columns
                if col != "band_gap" and pd.api.types.is_numeric_dtype(df[col])
            ]

            if len(numeric_params) >= 2:
                # Create correlation heatmap
                correlation_data = df[numeric_params + ["band_gap"]].corr()

                fig = px.imshow(
                    correlation_data,
                    title="Parameter Correlation with Band Gap",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                )

                st.plotly_chart(fig, use_container_width=True)

    def _is_today(self, timestamp: str) -> bool:
        """Check if timestamp is from today."""
        if not timestamp:
            return False

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.date() == datetime.now().date()
        except:
            return False


class RobotStatus:
    """Robot status monitoring component."""

    @staticmethod
    def render(robot_data: List[Dict[str, Any]]):
        """Render robot status display."""
        st.subheader("🤖 Robot Status")

        for robot in robot_data:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

                with col1:
                    st.write(f"**{robot['id']}**")
                    st.write(f"*{robot.get('type', 'Unknown')}*")

                with col2:
                    status = robot.get("status", "unknown")
                    if status == "idle":
                        st.success("🟢 Idle")
                    elif status == "busy":
                        st.warning("🟡 Busy")
                    else:
                        st.error("🔴 Error")

                with col3:
                    uptime = robot.get("uptime", "0 hrs")
                    st.metric("Uptime", uptime)

                with col4:
                    last_action = robot.get("last_action", "None")
                    st.write(f"Last: {last_action}")

                st.markdown("---")


class ExperimentViewer:
    """Detailed experiment viewer component."""

    @staticmethod
    def render(experiment: Dict[str, Any]):
        """Render detailed experiment view."""
        exp_id = experiment.get("id", "unknown")
        status = experiment.get("status", "unknown")
        timestamp = experiment.get("timestamp", "")

        st.subheader(f"🔬 Experiment {exp_id[:8]}")

        # Basic info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Status:**", status)
        with col2:
            st.write("**Timestamp:**", timestamp[:19] if timestamp else "Unknown")
        with col3:
            duration = experiment.get("duration")
            if duration:
                st.write("**Duration:**", f"{duration:.2f}s")

        # Parameters
        parameters = experiment.get("parameters", {})
        if parameters:
            st.write("### Parameters")
            param_df = pd.DataFrame([parameters])
            st.dataframe(param_df, use_container_width=True)

        # Results
        results = experiment.get("results", {})
        if results:
            st.write("### Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if "band_gap" in results:
                    st.metric("Band Gap", f"{results['band_gap']:.3f} eV")

            with col2:
                if "efficiency" in results:
                    st.metric("Efficiency", f"{results['efficiency']:.2f}%")

            with col3:
                if "stability" in results:
                    st.metric("Stability", f"{results['stability']:.3f}")

        # Metadata
        metadata = experiment.get("metadata", {})
        if metadata:
            with st.expander("Metadata"):
                st.json(metadata)
