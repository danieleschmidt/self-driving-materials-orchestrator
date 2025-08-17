"""Main Streamlit dashboard application."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Optional, Any

from ..database import ExperimentDatabase
from .utils import load_campaign_data, format_duration, create_property_evolution_plot
from .components import (
    render_campaign_overview,
    render_robot_status,
    render_experiment_feed,
)


def create_dashboard_app():
    """Create and configure the main dashboard application."""

    st.set_page_config(
        page_title="🔬 Self-Driving Materials Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .robot-status-online {
        color: #28a745;
        font-weight: bold;
    }
    .robot-status-offline {
        color: #dc3545; 
        font-weight: bold;
    }
    .experiment-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    .experiment-failed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = ExperimentDatabase()

    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

    # Sidebar configuration
    st.sidebar.title("🎛️ Dashboard Controls")

    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto-refresh", value=st.session_state.auto_refresh
    )

    refresh_interval = st.sidebar.selectbox(
        "Refresh interval (seconds)", [5, 10, 30, 60], index=1
    )

    # Campaign selection
    available_campaigns = get_available_campaigns(st.session_state.db)

    if available_campaigns:
        selected_campaign = st.sidebar.selectbox(
            "Select Campaign", ["All Campaigns"] + available_campaigns, index=0
        )
    else:
        selected_campaign = "No campaigns found"
        st.sidebar.info("No campaigns found. Run an experiment to see data.")

    # Main dashboard content
    st.title("🔬 Self-Driving Materials Discovery Lab")
    st.markdown("---")

    # Auto-refresh mechanism
    placeholder = st.empty()

    if st.session_state.auto_refresh:
        # Create auto-refresh loop
        with placeholder.container():
            render_dashboard_content(st.session_state.db, selected_campaign)

        # Add refresh timer
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()

        # JavaScript auto-refresh (fallback)
        st.markdown(
            f"""
        <script>
        setTimeout(function(){{
            window.location.reload();
        }}, {refresh_interval * 1000});
        </script>
        """,
            unsafe_allow_html=True,
        )
    else:
        render_dashboard_content(st.session_state.db, selected_campaign)


def render_dashboard_content(db: ExperimentDatabase, selected_campaign: str):
    """Render the main dashboard content."""

    # Load data
    if (
        selected_campaign == "All Campaigns"
        or selected_campaign == "No campaigns found"
    ):
        experiments = db.query_experiments(limit=1000)
        campaign_filter = None
    else:
        experiments = db.query_experiments(
            filter_criteria={"campaign_id": selected_campaign}, limit=1000
        )
        campaign_filter = selected_campaign

    if not experiments:
        st.info("No experiments found. Start a campaign to see real-time data!")
        return

    # Overview metrics
    render_overview_metrics(experiments)

    st.markdown("---")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Campaign Overview",
            "🤖 Robot Status",
            "🔴 Live Feed",
            "📈 Analytics",
            "⚙️ Settings",
        ]
    )

    with tab1:
        render_campaign_overview(experiments, campaign_filter)

    with tab2:
        render_robot_status_tab()

    with tab3:
        render_live_experiment_feed(experiments[-20:])  # Last 20 experiments

    with tab4:
        render_analytics_tab(experiments)

    with tab5:
        render_settings_tab()


def render_overview_metrics(experiments: List[Dict[str, Any]]):
    """Render top-level overview metrics."""

    # Calculate metrics
    total_experiments = len(experiments)
    successful_experiments = len(
        [exp for exp in experiments if exp.get("status") == "completed"]
    )
    success_rate = (
        successful_experiments / total_experiments if total_experiments > 0 else 0
    )

    # Get time span
    if experiments:
        timestamps = [exp.get("timestamp", "") for exp in experiments]
        timestamps = [t for t in timestamps if t]
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            time_span = f"{earliest[:10]} to {latest[:10]}"
        else:
            time_span = "Unknown"
    else:
        time_span = "No data"

    # Best material found
    best_bandgap = 0
    best_efficiency = 0

    for exp in experiments:
        if exp.get("status") == "completed" and exp.get("results"):
            bg = exp["results"].get("band_gap", 0)
            eff = exp["results"].get("efficiency", 0)
            if bg > best_bandgap:
                best_bandgap = bg
            if eff > best_efficiency:
                best_efficiency = eff

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Experiments",
            f"{total_experiments:,}",
            delta=f"+{min(5, total_experiments)}" if total_experiments > 0 else None,
        )

    with col2:
        st.metric(
            "Success Rate",
            f"{success_rate:.1%}",
            delta=f"+{success_rate-0.8:.1%}" if success_rate > 0.8 else None,
        )

    with col3:
        st.metric(
            "Best Band Gap",
            f"{best_bandgap:.3f} eV" if best_bandgap > 0 else "N/A",
            delta="Target range" if 1.2 <= best_bandgap <= 1.6 else None,
        )

    with col4:
        st.metric(
            "Best Efficiency",
            f"{best_efficiency:.1f}%" if best_efficiency > 0 else "N/A",
            delta=f"+{best_efficiency-20:.1f}%" if best_efficiency > 20 else None,
        )


def render_robot_status_tab():
    """Render robot status monitoring tab."""
    st.subheader("🤖 Robot & Instrument Status")

    # Simulated robot status data
    robot_data = [
        {
            "id": "synthesis_robot",
            "type": "Chemspeed",
            "status": "idle",
            "uptime": "12.5 hrs",
            "last_action": "heat",
        },
        {
            "id": "liquid_handler",
            "type": "Opentrons",
            "status": "busy",
            "uptime": "8.2 hrs",
            "last_action": "dispense",
        },
        {
            "id": "characterization_robot",
            "type": "Custom",
            "status": "idle",
            "uptime": "15.1 hrs",
            "last_action": "move",
        },
    ]

    instrument_data = [
        {
            "id": "xrd",
            "type": "Bruker D8",
            "status": "idle",
            "last_measurement": "5 min ago",
        },
        {
            "id": "uv_vis",
            "type": "Shimadzu",
            "status": "measuring",
            "last_measurement": "Running",
        },
        {
            "id": "pl_spectrometer",
            "type": "Horiba",
            "status": "idle",
            "last_measurement": "1 hr ago",
        },
    ]

    # Robot status
    st.write("### Robots")
    for robot in robot_data:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 2])

        with col1:
            st.write(f"**{robot['id']}**")
        with col2:
            st.write(robot["type"])
        with col3:
            status_class = (
                "robot-status-online"
                if robot["status"] == "idle"
                else "robot-status-offline"
            )
            st.markdown(
                f"<span class='{status_class}'>{robot['status']}</span>",
                unsafe_allow_html=True,
            )
        with col4:
            st.write(robot["uptime"])
        with col5:
            st.write(robot["last_action"])

    st.write("### Instruments")
    for instrument in instrument_data:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 2])

        with col1:
            st.write(f"**{instrument['id']}**")
        with col2:
            st.write(instrument["type"])
        with col3:
            status_class = (
                "robot-status-online"
                if instrument["status"] == "idle"
                else "robot-status-offline"
            )
            st.markdown(
                f"<span class='{status_class}'>{instrument['status']}</span>",
                unsafe_allow_html=True,
            )
        with col4:
            st.write(instrument["last_measurement"])


def render_live_experiment_feed(recent_experiments: List[Dict[str, Any]]):
    """Render live experiment feed."""
    st.subheader("🔴 Live Experiment Feed")

    if not recent_experiments:
        st.info("No recent experiments to display")
        return

    # Sort by timestamp (most recent first)
    sorted_experiments = sorted(
        recent_experiments, key=lambda x: x.get("timestamp", ""), reverse=True
    )

    for exp in sorted_experiments[:10]:  # Show last 10
        exp_id = exp.get("id", "unknown")[:8]
        status = exp.get("status", "unknown")
        timestamp = exp.get("timestamp", "")

        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
        else:
            time_str = "unknown"

        # Experiment card
        if status == "completed":
            card_class = "experiment-success"
            status_emoji = "✅"
        elif status == "failed":
            card_class = "experiment-failed"
            status_emoji = "❌"
        else:
            card_class = ""
            status_emoji = "🔄"

        results = exp.get("results", {})
        bandgap = results.get("band_gap", "N/A")
        efficiency = results.get("efficiency", "N/A")

        st.markdown(
            f"""
        <div class="{card_class}">
        <strong>{status_emoji} Experiment {exp_id}</strong> - {time_str}<br>
        Status: {status}<br>
        Band gap: {bandgap} eV | Efficiency: {efficiency}%
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_analytics_tab(experiments: List[Dict[str, Any]]):
    """Render advanced analytics and visualizations."""
    st.subheader("📈 Advanced Analytics")

    if not experiments:
        st.info("No experiment data available for analysis")
        return

    # Prepare data for analysis
    df_data = []
    for i, exp in enumerate(experiments):
        if exp.get("status") == "completed" and exp.get("results"):
            row = {
                "experiment_number": i + 1,
                "timestamp": exp.get("timestamp", ""),
                **exp.get("parameters", {}),
                **exp.get("results", {}),
            }
            df_data.append(row)

    if not df_data:
        st.info("No completed experiments with results")
        return

    df = pd.DataFrame(df_data)

    # Property evolution over time
    st.write("### Property Evolution")

    col1, col2 = st.columns(2)

    with col1:
        if "band_gap" in df.columns:
            fig = px.line(
                df,
                x="experiment_number",
                y="band_gap",
                title="Band Gap Evolution",
                labels={
                    "experiment_number": "Experiment #",
                    "band_gap": "Band Gap (eV)",
                },
            )
            fig.add_hline(y=1.4, line_dash="dash", annotation_text="Target")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "efficiency" in df.columns:
            fig = px.line(
                df,
                x="experiment_number",
                y="efficiency",
                title="Efficiency Evolution",
                labels={
                    "experiment_number": "Experiment #",
                    "efficiency": "Efficiency (%)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    # Parameter space exploration
    st.write("### Parameter Space Exploration")

    numeric_params = [
        col
        for col in df.columns
        if col.startswith(("temperature", "precursor", "reaction", "pH"))
    ]

    if len(numeric_params) >= 2:
        x_param = st.selectbox("X-axis parameter", numeric_params, index=0)
        y_param = st.selectbox("Y-axis parameter", numeric_params, index=1)

        fig = px.scatter(
            df,
            x=x_param,
            y=y_param,
            color="band_gap" if "band_gap" in df.columns else None,
            size="efficiency" if "efficiency" in df.columns else None,
            title=f"Parameter Space: {x_param} vs {y_param}",
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Success rate analysis
    st.write("### Success Rate Analysis")

    if "band_gap" in df.columns:
        # Define success as being within target range
        df["in_target_range"] = (df["band_gap"] >= 1.2) & (df["band_gap"] <= 1.6)
        success_rate = df["in_target_range"].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Success Rate", f"{success_rate:.1%}")

        with col2:
            # Success rate over time (rolling average)
            window_size = min(10, len(df))
            rolling_success = df["in_target_range"].rolling(window=window_size).mean()

            fig = px.line(
                x=df["experiment_number"],
                y=rolling_success,
                title=f"Success Rate (Rolling {window_size}-experiment average)",
                labels={"x": "Experiment #", "y": "Success Rate"},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_settings_tab():
    """Render dashboard settings and configuration."""
    st.subheader("⚙️ Dashboard Settings")

    # Database settings
    st.write("### Database Configuration")

    # Display current database info
    st.info("Using file-based storage (fallback mode)")

    # Export data
    st.write("### Data Export")

    if st.button("📥 Export All Data"):
        # This would export data in real implementation
        st.success("Data export would be triggered here")

    # Visualization settings
    st.write("### Visualization Preferences")

    theme = st.selectbox("Dashboard Theme", ["Default", "Dark", "Light"])
    chart_type = st.selectbox("Default Chart Type", ["Line", "Scatter", "Bar"])

    # System information
    st.write("### System Information")

    info_data = {
        "Dashboard Version": "1.0.0",
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Sessions": "142",
        "Active Users": "3",
    }

    for key, value in info_data.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(value)


def get_available_campaigns(db: ExperimentDatabase) -> List[str]:
    """Get list of available campaign IDs."""
    try:
        experiments = db.query_experiments(limit=100)
        campaign_ids = list(
            set(exp.get("campaign_id") for exp in experiments if exp.get("campaign_id"))
        )
        return sorted(campaign_ids)
    except Exception as e:
        st.error(f"Error loading campaigns: {e}")
        return []


if __name__ == "__main__":
    create_dashboard_app()
