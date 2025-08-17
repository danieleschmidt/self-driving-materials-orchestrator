"""Production API for Materials Orchestrator."""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# FastAPI imports with fallbacks
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback classes for basic API structure
    class FastAPI:
        def __init__(self, **kwargs):
            pass

        def get(self, path):
            return lambda f: f

        def post(self, path):
            return lambda f: f

        def middleware(self, middleware_type):
            return lambda f: f

    class HTTPException(Exception):
        pass

    class BackgroundTasks:
        pass

    class CORSMiddleware:
        pass

    class BaseModel:
        pass

    def Field(**kwargs):
        return None

    FASTAPI_AVAILABLE = False

from .core import AutonomousLab, MaterialsObjective, Experiment
from .planners import BayesianPlanner, RandomPlanner
from .security_enhanced import get_global_security_manager
from .health_monitoring import get_global_health_monitor
from .performance_optimizer import get_global_performance_optimizer
from .quality_gates import get_global_quality_runner

logger = logging.getLogger(__name__)


# Pydantic models for API
class CampaignRequest(BaseModel):
    """Request model for starting a campaign."""

    target_property: str = Field(..., description="Target property to optimize")
    target_min: float = Field(..., description="Target range minimum")
    target_max: float = Field(..., description="Target range maximum")
    material_system: str = Field(default="general", description="Material system type")
    param_space: Dict[str, List[float]] = Field(..., description="Parameter space")
    max_experiments: int = Field(default=100, description="Maximum experiments")
    initial_samples: int = Field(default=20, description="Initial random samples")
    planner_type: str = Field(
        default="bayesian", description="Optimization planner type"
    )


class ExperimentRequest(BaseModel):
    """Request model for running an experiment."""

    parameters: Dict[str, float] = Field(..., description="Experiment parameters")


class HealthResponse(BaseModel):
    """Response model for health status."""

    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Check timestamp")
    details: Dict[str, Any] = Field(..., description="Detailed health information")


# Global lab instance
_global_lab: Optional[AutonomousLab] = None


def get_global_lab() -> AutonomousLab:
    """Get or create global lab instance."""
    global _global_lab
    if _global_lab is None:
        _global_lab = AutonomousLab(
            planner=BayesianPlanner("band_gap"),  # Default planner
            enable_monitoring=True,
        )
    return _global_lab


def create_production_api() -> FastAPI:
    """Create production-ready FastAPI application."""

    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available - cannot create production API")
        raise RuntimeError("FastAPI required for production API")

    app = FastAPI(
        title="Materials Orchestrator API",
        description="Autonomous materials discovery platform API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Materials Orchestrator API",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "/health",
                "experiment": "/experiment",
                "campaign": "/campaign",
                "status": "/status",
                "metrics": "/metrics",
                "docs": "/docs",
            },
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Comprehensive health check endpoint."""
        try:
            # Get health monitor
            health_monitor = get_global_health_monitor()

            # Run health checks
            overall_status, summary = health_monitor.get_overall_health()

            # Get performance metrics
            perf_optimizer = get_global_performance_optimizer()
            perf_report = perf_optimizer.get_performance_report()

            # Get security status
            security_manager = get_global_security_manager()
            security_status = security_manager.get_security_status()

            health_details = {
                "overall_status": overall_status.value,
                "system_summary": summary,
                "performance": {
                    "cache_hit_rate": perf_report["cache_performance"]["hit_rate"],
                    "resource_utilization": perf_report["execution_performance"][
                        "resource_utilization"
                    ],
                    "average_response_time": perf_report["execution_performance"][
                        "average_response_time"
                    ],
                },
                "security": {
                    "security_level": security_status["security_level"],
                    "blocked_sources": security_status["blocked_sources_count"],
                    "recent_events": security_status["recent_events_1h"],
                },
            }

            return HealthResponse(
                status=overall_status.value,
                timestamp=datetime.now().isoformat(),
                details=health_details,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="error",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)},
            )

    @app.post("/experiment")
    async def run_single_experiment(request: ExperimentRequest):
        """Run a single experiment."""
        try:
            lab = get_global_lab()

            # Run experiment
            experiment = lab.run_experiment(request.parameters)

            return {
                "experiment_id": experiment.id,
                "status": experiment.status,
                "parameters": experiment.parameters,
                "results": experiment.results,
                "duration": experiment.duration,
                "timestamp": experiment.timestamp.isoformat(),
                "metadata": experiment.metadata,
            }

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/campaign")
    async def start_campaign(
        request: CampaignRequest, background_tasks: BackgroundTasks
    ):
        """Start an autonomous discovery campaign."""
        try:
            # Create materials objective
            objective = MaterialsObjective(
                target_property=request.target_property,
                target_range=(request.target_min, request.target_max),
                optimization_direction="target",
                material_system=request.material_system,
            )

            # Create planner
            if request.planner_type == "bayesian":
                planner = BayesianPlanner(request.target_property)
            else:
                planner = RandomPlanner()

            # Create lab with specified planner
            lab = AutonomousLab(planner=planner, enable_monitoring=True)

            # Convert parameter space format
            param_space = {}
            for param, bounds in request.param_space.items():
                if isinstance(bounds, list) and len(bounds) >= 2:
                    param_space[param] = (bounds[0], bounds[1])
                else:
                    param_space[param] = (0.1, 10.0)  # Default range

            # Run campaign
            campaign = lab.run_campaign(
                objective=objective,
                param_space=param_space,
                initial_samples=request.initial_samples,
                max_experiments=request.max_experiments,
                stop_on_target=True,
            )

            return {
                "campaign_id": campaign.campaign_id,
                "status": "completed",
                "total_experiments": campaign.total_experiments,
                "successful_experiments": campaign.successful_experiments,
                "success_rate": campaign.success_rate,
                "best_material": campaign.best_material,
                "best_properties": campaign.best_properties,
                "duration_hours": campaign.duration,
                "start_time": campaign.start_time.isoformat(),
                "end_time": (
                    campaign.end_time.isoformat() if campaign.end_time else None
                ),
            }

        except Exception as e:
            logger.error(f"Campaign execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/status")
    async def get_system_status():
        """Get comprehensive system status."""
        try:
            lab = get_global_lab()

            # Get performance metrics
            perf_optimizer = get_global_performance_optimizer()
            perf_report = perf_optimizer.get_performance_report()

            # Get quality metrics
            quality_runner = get_global_quality_runner()

            # System information
            import psutil

            system_info = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage(".").percent,
            }

            return {
                "timestamp": datetime.now().isoformat(),
                "lab_status": lab.status.value,
                "experiments_run": lab.total_experiments,
                "success_rate": lab.success_rate,
                "performance_metrics": {
                    "cache_hit_rate": perf_report["cache_performance"]["hit_rate"],
                    "average_response_time": perf_report["execution_performance"][
                        "average_response_time"
                    ],
                    "throughput": perf_report["execution_performance"]["throughput"],
                    "resource_utilization": perf_report["execution_performance"][
                        "resource_utilization"
                    ],
                },
                "system_resources": system_info,
                "optimization_recommendations": perf_report[
                    "optimization_recommendations"
                ],
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            # Fallback status without system metrics
            return {
                "timestamp": datetime.now().isoformat(),
                "lab_status": "operational",
                "experiments_run": 0,
                "success_rate": 1.0,
                "error": f"Detailed status unavailable: {e}",
            }

    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics in Prometheus format."""
        try:
            lab = get_global_lab()
            perf_optimizer = get_global_performance_optimizer()
            perf_report = perf_optimizer.get_performance_report()

            # Generate Prometheus-style metrics
            metrics = []
            metrics.append(
                f"materials_orchestrator_experiments_total {lab.total_experiments}"
            )
            metrics.append(f"materials_orchestrator_success_rate {lab.success_rate}")
            metrics.append(
                f"materials_orchestrator_cache_hit_rate {perf_report['cache_performance']['hit_rate']}"
            )
            metrics.append(
                f"materials_orchestrator_response_time_seconds {perf_report['execution_performance']['average_response_time']}"
            )
            metrics.append(
                f"materials_orchestrator_throughput_ops_per_second {perf_report['execution_performance']['throughput']}"
            )

            return {"metrics": "\\n".join(metrics)}

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {"error": str(e)}

    @app.get("/quality-gates")
    async def run_quality_gates():
        """Run quality gates and return results."""
        try:
            quality_runner = get_global_quality_runner()
            result = quality_runner.run_quality_gates()

            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": result.overall_status.value,
                "total_tests": result.total_tests,
                "passed_tests": result.passed_tests,
                "failed_tests": result.failed_tests,
                "success_rate": result.success_rate,
                "execution_time": result.execution_time,
                "test_summary": [
                    {
                        "name": test.name,
                        "status": test.status.value,
                        "message": test.message,
                        "severity": test.severity.value,
                    }
                    for test in result.test_results
                ],
            }

        except Exception as e:
            logger.error(f"Quality gates execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Simple development server
def run_development_server(host: str = "0.0.0.0", port: int = 8000):
    """Run development server."""
    try:
        import uvicorn

        app = create_production_api()
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        logger.error("uvicorn not available - cannot run development server")
        print("Install uvicorn to run development server: pip install uvicorn")


if __name__ == "__main__":
    run_development_server()
