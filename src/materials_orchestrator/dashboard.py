"""Real-time dashboard for monitoring autonomous lab operations."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .core import AutonomousLab, MaterialsObjective
    from .database import ExperimentTracker, create_database
except ImportError:
    # Handle imports when running as standalone script
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from core import AutonomousLab
    from database import ExperimentTracker

logger = logging.getLogger(__name__)


class LabDashboard:
    """Dashboard controller for autonomous lab monitoring."""

    def __init__(
        self,
        lab: Optional[AutonomousLab] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.lab = lab
        self.tracker = tracker or ExperimentTracker()
        self.current_campaign = None

    def render_main_dashboard(self):
        """Render the main dashboard interface."""
        st.set_page_config(
            page_title="🔬 Autonomous Materials Lab",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🔬 Self-Driving Materials Discovery Lab")
        st.markdown("*Real-time monitoring and control interface*")

        # Sidebar for navigation and controls
        self._render_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Dashboard", "🧪 Experiments", "🤖 Robots", "📈 Analytics"]
        )

        with tab1:
            self._render_overview_dashboard()

        with tab2:
            self._render_experiments_tab()

        with tab3:
            self._render_robots_tab()

        with tab4:
            self._render_analytics_tab()

    def _render_sidebar(self):
        """Render the sidebar with controls and status."""
        st.sidebar.header("🎛️ Control Panel")

        # Campaign selection
        st.sidebar.subheader("Campaign")
        campaign_id = st.sidebar.text_input("Campaign ID", value="demo_campaign_001")

        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

        # Lab status (simulated)
        st.sidebar.subheader("🔋 Lab Status")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.metric("Status", "🟢 Active")
            st.metric("Uptime", "2d 4h")

        with col2:
            st.metric("Queue", "3 exp")
            st.metric("Robots", "2/3 online")

        # Quick actions
        st.sidebar.subheader("⚡ Quick Actions")

        if st.sidebar.button("🚨 Emergency Stop", type="secondary"):
            st.sidebar.error("Emergency stop activated!")

        if st.sidebar.button("▶️ Resume Operations", type="primary"):
            st.sidebar.success("Operations resumed")

    def _render_overview_dashboard(self):
        """Render the main overview dashboard."""
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        # Simulate real-time metrics
        current_time = datetime.now()

        with col1:
            st.metric(
                "Total Experiments",
                "342",
                delta="+12",
                help="Total experiments run today",
            )

        with col2:
            st.metric(
                "Success Rate",
                "94.2%",
                delta="+1.3%",
                help="Successful experiment completion rate",
            )

        with col3:
            st.metric(
                "Best Band Gap",
                "1.413 eV",
                delta="-0.02 eV",
                delta_color="inverse",
                help="Best band gap achieved (closer to 1.4 eV target)",
            )

        with col4:
            st.metric(
                "Efficiency",
                "27.1%",
                delta="+2.1%",
                help="Best photovoltaic efficiency achieved",
            )

        with col5:
            st.metric(
                "Time Saved",
                "8.5 days",
                delta="+1.2 days",
                help="Time saved vs traditional methods",
            )

        st.divider()

        # Real-time experiment feed and progress charts
        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_property_evolution_chart()

        with col2:
            self._render_live_experiments_feed()

    def _render_property_evolution_chart(self):
        """Render property evolution over time chart."""
        st.subheader("📊 Property Evolution")

        # Generate sample data for demonstration
        import numpy as np

        np.random.seed(42)

        n_points = 50
        experiments = list(range(1, n_points + 1))

        # Simulate band gap convergence to target
        target = 1.4
        band_gaps = []
        current_best = 2.0

        for i in range(n_points):
            # Simulate Bayesian optimization convergence
            improvement_rate = 0.95 ** (i / 10)
            noise = np.random.normal(0, 0.1)
            new_value = target + (current_best - target) * improvement_rate + noise

            if new_value < current_best:
                current_best = new_value

            band_gaps.append(current_best)

        # Create plotly chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=experiments,
                y=band_gaps,
                mode="lines+markers",
                name="Best Band Gap",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )

        # Add target line
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 1.4 eV",
        )

        # Add target range
        fig.add_hrect(
            y0=1.2,
            y1=1.6,
            fillcolor="green",
            opacity=0.1,
            line_width=0,
            annotation_text="Target Range",
        )

        fig.update_layout(
            title="Band Gap Optimization Progress",
            xaxis_title="Experiment Number",
            yaxis_title="Band Gap (eV)",
            height=400,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_live_experiments_feed(self):
        """Render live experiments feed."""
        st.subheader("🔴 Live Experiments")

        # Simulate running experiments
        experiments = [
            {"id": "EXP-001", "status": "⚗️ Synthesis", "progress": 85, "eta": "2 min"},
            {"id": "EXP-002", "status": "🔬 Analysis", "progress": 45, "eta": "8 min"},
            {"id": "EXP-003", "status": "🌡️ Heating", "progress": 20, "eta": "15 min"},
        ]

        for exp in experiments:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text(f"**{exp['id']}**: {exp['status']}")
                    st.progress(exp["progress"] / 100)

                with col2:
                    st.text(f"ETA: {exp['eta']}")

        st.divider()

        # Queue status
        st.subheader("📋 Experiment Queue")
        st.text("Queue: 3 experiments pending")
        st.text("Next up: Perovskite synthesis batch #42")

        if st.button("▶️ Process Queue", key="process_queue"):
            st.success("Queue processing started")

    def _render_experiments_tab(self):
        """Render experiments management tab."""
        st.header("🧪 Experiment Management")

        # Experiment submission form
        with st.expander("➕ Submit New Experiment", expanded=False):
            self._render_experiment_form()

        # Recent experiments table
        st.subheader("📋 Recent Experiments")

        # Generate sample experiment data
        sample_data = [
            {
                "ID": "EXP-340",
                "Timestamp": "2024-12-07 14:23:15",
                "Status": "✅ Completed",
                "Band Gap (eV)": 1.423,
                "Efficiency (%)": 25.6,
                "Temperature (°C)": 145,
                "Time (h)": 3.2,
            },
            {
                "ID": "EXP-339",
                "Timestamp": "2024-12-07 14:15:42",
                "Status": "✅ Completed",
                "Band Gap (eV)": 1.398,
                "Efficiency (%)": 27.1,
                "Temperature (°C)": 152,
                "Time (h)": 2.8,
            },
            {
                "ID": "EXP-338",
                "Timestamp": "2024-12-07 14:08:11",
                "Status": "❌ Failed",
                "Band Gap (eV)": None,
                "Efficiency (%)": None,
                "Temperature (°C)": 180,
                "Time (h)": 1.5,
            },
        ]

        df = pd.DataFrame(sample_data)
        st.dataframe(df, use_container_width=True)

        # Experiment details
        selected_exp = st.selectbox("Select experiment for details:", df["ID"].tolist())

        if selected_exp:
            exp_data = df[df["ID"] == selected_exp].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📊 {selected_exp} Details")
                st.json(
                    {
                        "parameters": {
                            "precursor_A_conc": 1.2,
                            "precursor_B_conc": 0.8,
                            "temperature": (
                                float(exp_data["Temperature (°C)"])
                                if pd.notna(exp_data["Temperature (°C)"])
                                else None
                            ),
                            "reaction_time": (
                                float(exp_data["Time (h)"])
                                if pd.notna(exp_data["Time (h)"])
                                else None
                            ),
                            "pH": 7.2,
                            "solvent_ratio": 0.6,
                        },
                        "results": {
                            "band_gap": (
                                float(exp_data["Band Gap (eV)"])
                                if pd.notna(exp_data["Band Gap (eV)"])
                                else None
                            ),
                            "efficiency": (
                                float(exp_data["Efficiency (%)"])
                                if pd.notna(exp_data["Efficiency (%)"])
                                else None
                            ),
                            "stability": (
                                0.89 if exp_data["Status"] == "✅ Completed" else None
                            ),
                        },
                    }
                )

            with col2:
                st.subheader("📈 Parameter Visualization")
                # Simple bar chart of parameters
                if exp_data["Status"] == "✅ Completed":
                    params = ["Temperature", "Time", "Efficiency"]
                    values = [
                        float(exp_data["Temperature (°C)"]),
                        float(exp_data["Time (h)"]) * 10,  # Scale for visibility
                        float(exp_data["Efficiency (%)"]),
                    ]

                    fig = go.Figure(data=go.Bar(x=params, y=values))
                    fig.update_layout(title="Experiment Parameters", height=300)
                    st.plotly_chart(fig, use_container_width=True)

    def _render_experiment_form(self):
        """Render experiment submission form."""
        st.subheader("🧪 New Experiment Parameters")

        col1, col2 = st.columns(2)

        with col1:
            precursor_a = st.slider("Precursor A Concentration (M)", 0.1, 2.0, 1.0, 0.1)
            precursor_b = st.slider("Precursor B Concentration (M)", 0.1, 2.0, 1.0, 0.1)
            temperature = st.slider("Temperature (°C)", 100, 300, 150, 5)

        with col2:
            reaction_time = st.slider("Reaction Time (hours)", 1, 24, 3, 1)
            ph = st.slider("pH", 3.0, 11.0, 7.0, 0.1)
            solvent_ratio = st.slider("Solvent Ratio (DMF:DMSO)", 0.0, 1.0, 0.5, 0.1)

        priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)

        if st.button("🚀 Submit Experiment", type="primary"):
            # Simulate experiment submission
            exp_params = {
                "precursor_A_conc": precursor_a,
                "precursor_B_conc": precursor_b,
                "temperature": temperature,
                "reaction_time": reaction_time,
                "pH": ph,
                "solvent_ratio": solvent_ratio,
                "priority": priority,
            }

            st.success(f"Experiment submitted with parameters: {exp_params}")
            st.info("Experiment added to queue. Estimated start time: 15 minutes")

    def _render_robots_tab(self):
        """Render robot monitoring tab."""
        st.header("🤖 Robot Fleet Status")

        # Robot status cards
        robots = [
            {
                "name": "Liquid Handler",
                "id": "LH-001",
                "status": "🟢 Active",
                "current_task": "Dispensing reagents",
                "utilization": 78,
                "next_maintenance": "2024-12-15",
            },
            {
                "name": "Synthesizer",
                "id": "SYN-002",
                "status": "🟡 Busy",
                "current_task": "Heating reaction vessel",
                "utilization": 92,
                "next_maintenance": "2024-12-10",
            },
            {
                "name": "Analyzer",
                "id": "ANA-003",
                "status": "🔴 Offline",
                "current_task": "Maintenance mode",
                "utilization": 0,
                "next_maintenance": "2024-12-08",
            },
        ]

        for robot in robots:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.subheader(f"🤖 {robot['name']}")
                    st.text(f"ID: {robot['id']}")
                    st.text(f"Status: {robot['status']}")

                with col2:
                    st.text(f"Task: {robot['current_task']}")
                    st.progress(robot["utilization"] / 100)
                    st.text(f"Utilization: {robot['utilization']}%")

                with col3:
                    st.text(f"Next Maintenance: {robot['next_maintenance']}")
                    if robot["status"] == "🔴 Offline":
                        if st.button(
                            f"🔄 Restart {robot['id']}", key=f"restart_{robot['id']}"
                        ):
                            st.success(f"Restart command sent to {robot['name']}")

                with col4:
                    with st.expander("⚙️"):
                        st.text("Robot Controls:")
                        st.button("⏸️ Pause", key=f"pause_{robot['id']}")
                        st.button("🛑 Stop", key=f"stop_{robot['id']}")
                        st.button("🔧 Maintenance", key=f"maint_{robot['id']}")

                st.divider()

        # Robot utilization chart
        st.subheader("📊 Robot Utilization Over Time")

        # Generate sample utilization data
        hours = list(range(24))
        lh_util = [
            50 + 30 * abs(h - 12) / 12 + np.random.randint(-10, 10) for h in hours
        ]
        syn_util = [
            60 + 20 * abs(h - 14) / 14 + np.random.randint(-5, 15) for h in hours
        ]
        ana_util = [
            30 + 40 * abs(h - 10) / 10 + np.random.randint(-15, 5) for h in hours
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=hours, y=lh_util, name="Liquid Handler", mode="lines+markers")
        )
        fig.add_trace(
            go.Scatter(x=hours, y=syn_util, name="Synthesizer", mode="lines+markers")
        )
        fig.add_trace(
            go.Scatter(x=hours, y=ana_util, name="Analyzer", mode="lines+markers")
        )

        fig.update_layout(
            title="24-Hour Robot Utilization",
            xaxis_title="Hour of Day",
            yaxis_title="Utilization (%)",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_analytics_tab(self):
        """Render analytics and insights tab."""
        st.header("📈 Discovery Analytics")

        # Parameter space exploration
        st.subheader("🎯 Parameter Space Exploration")

        # 3D parameter space plot
        import numpy as np

        np.random.seed(42)

        n_points = 100
        temp = np.random.uniform(100, 300, n_points)
        time = np.random.uniform(1, 24, n_points)
        conc = np.random.uniform(0.1, 2.0, n_points)

        # Simulate band gap based on parameters
        band_gap = (
            1.5
            + (temp - 200) / 1000
            + (time - 12) / 100
            + (conc - 1) / 10
            + np.random.normal(0, 0.1, n_points)
        )

        fig = go.Figure(
            data=go.Scatter3d(
                x=temp,
                y=time,
                z=conc,
                mode="markers",
                marker=dict(
                    size=5,
                    color=band_gap,
                    colorscale="Viridis",
                    colorbar=dict(title="Band Gap (eV)"),
                    showscale=True,
                ),
                text=[f"Band Gap: {bg:.3f} eV" for bg in band_gap],
                textposition="top center",
            )
        )

        fig.update_layout(
            title="3D Parameter Space (Temperature vs Time vs Concentration)",
            scene=dict(
                xaxis_title="Temperature (°C)",
                yaxis_title="Reaction Time (h)",
                zaxis_title="Concentration (M)",
            ),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Discovery Performance")

            # Comparison chart
            methods = ["Random", "Grid Search", "Bayesian", "Our Method"]
            experiments_needed = [350, 200, 67, 52]
            success_rates = [0.15, 0.35, 0.85, 0.92]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Experiments to Target",
                    x=methods,
                    y=experiments_needed,
                    yaxis="y",
                    offsetgroup=1,
                    marker_color="lightblue",
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Success Rate",
                    x=methods,
                    y=[sr * 400 for sr in success_rates],  # Scale for dual axis
                    yaxis="y2",
                    mode="lines+markers",
                    marker_color="red",
                    line=dict(width=3),
                )
            )

            fig.update_layout(
                title="Method Comparison",
                xaxis_title="Optimization Method",
                yaxis=dict(title="Experiments Needed", side="left"),
                yaxis2=dict(
                    title="Success Rate",
                    side="right",
                    overlaying="y",
                    tickmode="array",
                    tickvals=[0, 100, 200, 300, 400],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                ),
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Target Achievement")

            # Target achievement timeline
            days = list(range(1, 11))
            cumulative_targets = [0, 1, 1, 3, 5, 8, 12, 15, 18, 22]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=cumulative_targets,
                    mode="lines+markers",
                    name="Materials in Target Range",
                    line=dict(color="green", width=3),
                    marker=dict(size=8),
                )
            )

            fig.update_layout(
                title="Cumulative Target Achievements",
                xaxis_title="Days",
                yaxis_title="Materials in Target Range",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

        # Cost analysis
        st.subheader("💰 Cost Analysis")

        cost_data = {
            "Category": ["Materials", "Labor", "Equipment", "Energy", "Total"],
            "Traditional ($/day)": [500, 800, 200, 100, 1600],
            "Autonomous ($/day)": [600, 100, 300, 150, 1150],
            "Savings ($/day)": [0, 700, 0, 0, 450],
        }

        cost_df = pd.DataFrame(cost_data)
        cost_df["Savings ($/day)"] = (
            cost_df["Traditional ($/day)"] - cost_df["Autonomous ($/day)"]
        )

        st.dataframe(cost_df, use_container_width=True)

        st.success(
            f"💰 Daily savings: ${cost_df.loc[cost_df['Category'] == 'Total', 'Savings ($/day)'].iloc[0]}"
        )
        st.info(
            f"📅 Monthly savings: ${cost_df.loc[cost_df['Category'] == 'Total', 'Savings ($/day)'].iloc[0] * 30:,}"
        )


def main():
    """Main dashboard application entry point."""
    dashboard = LabDashboard()
    dashboard.render_main_dashboard()


if __name__ == "__main__":
    main()
