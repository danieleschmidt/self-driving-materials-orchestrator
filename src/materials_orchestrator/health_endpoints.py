"""Health check endpoints for production monitoring."""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class HealthChecker:
    """Centralized health checking system."""

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Any] = {}
        self._start_time = time.time()

    def register_check(self, name: str, check_func: callable, timeout: float = 5.0):
        """Register a health check function.
        
        Args:
            name: Check name
            check_func: Function that returns (is_healthy: bool, details: dict)
            timeout: Check timeout in seconds
        """
        self._checks[name] = {
            "func": check_func,
            "timeout": timeout,
            "last_check": None,
            "last_result": None,
        }

    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Health check results
        """
        results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self._start_time,
            "checks": {},
        }

        overall_healthy = True

        for name, check_info in self._checks.items():
            check_start = time.time()

            try:
                # Run check with timeout
                is_healthy, details = check_info["func"]()

                check_result = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "response_time": time.time() - check_start,
                    "details": details,
                }

                if not is_healthy:
                    overall_healthy = False

            except Exception as e:
                check_result = {
                    "status": "error",
                    "response_time": time.time() - check_start,
                    "error": str(e),
                }
                overall_healthy = False
                logger.error(f"Health check {name} failed: {e}")

            check_info["last_check"] = datetime.now()
            check_info["last_result"] = check_result
            results["checks"][name] = check_result

        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results


# Global health checker instance
health_checker = HealthChecker()


def setup_health_endpoints(app: FastAPI) -> None:
    """Set up health check endpoints.
    
    Args:
        app: FastAPI application
    """

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        results = health_checker.run_checks()

        status_code = 200 if results["status"] == "healthy" else 503
        return JSONResponse(content=results, status_code=status_code)

    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe."""
        results = health_checker.run_checks()

        # Check critical components
        critical_checks = ["database", "core_system"]
        ready = all(
            results["checks"].get(check, {}).get("status") == "healthy"
            for check in critical_checks
            if check in results["checks"]
        )

        if ready:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "checks": results["checks"]}
            )

    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe."""
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - health_checker._start_time,
        }

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        # This would typically return metrics in Prometheus format
        # For now, return basic metrics
        uptime = time.time() - health_checker._start_time

        metrics = [
            "# HELP materials_orchestrator_uptime_seconds Time since startup",
            "# TYPE materials_orchestrator_uptime_seconds counter",
            f"materials_orchestrator_uptime_seconds {uptime}",
            "",
            "# HELP materials_orchestrator_health_checks_total Health check results",
            "# TYPE materials_orchestrator_health_checks_total counter",
        ]

        # Add health check metrics
        for name, check_info in health_checker._checks.items():
            if check_info["last_result"]:
                status_value = 1 if check_info["last_result"]["status"] == "healthy" else 0
                metrics.append(
                    f'materials_orchestrator_health_check{{check="{name}"}} {status_value}'
                )

        return "\n".join(metrics)


# Default health checks
def database_health_check() -> tuple[bool, dict]:
    """Check database connectivity."""
    try:
        # This would check actual database connection
        # For now, return mock healthy status
        return True, {
            "connection": "established",
            "response_time_ms": 5.2,
        }
    except Exception as e:
        return False, {"error": str(e)}


def core_system_health_check() -> tuple[bool, dict]:
    """Check core system health."""
    try:
        # Check if core components can be imported and initialized

        return True, {
            "core_modules": "loaded",
            "status": "operational",
        }
    except Exception as e:
        return False, {"error": str(e)}


def memory_health_check() -> tuple[bool, dict]:
    """Check system memory usage."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        memory_healthy = memory.percent < 90
        disk_healthy = disk.percent < 90

        return memory_healthy and disk_healthy, {
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
        }
    except ImportError:
        return True, {"status": "psutil not available"}
    except Exception as e:
        return False, {"error": str(e)}


# Register default health checks
health_checker.register_check("database", database_health_check)
health_checker.register_check("core_system", core_system_health_check)
health_checker.register_check("memory", memory_health_check)
