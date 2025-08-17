"""Distributed computing capabilities for large-scale materials discovery."""

import logging
import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import threading
import queue
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""

    node_id: str
    host: str
    port: int
    capabilities: List[str]
    status: str = "idle"  # idle, busy, offline
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    experiment_queue: int = 0


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class DistributedOrchestrator:
    """Orchestrates distributed materials discovery experiments."""

    def __init__(self, max_workers: int = None):
        """Initialize distributed orchestrator."""
        self.max_workers = max_workers or mp.cpu_count()
        self.compute_nodes = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}

        # Thread pools for different types of work
        self.experiment_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="experiment"
        )
        self.analysis_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, self.max_workers), thread_name_prefix="analysis"
        )

        # Process pool for CPU-intensive tasks
        self.compute_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )

        # Coordination
        self._shutdown_event = threading.Event()
        self._scheduler_thread = None
        self._monitor_thread = None

        # Performance metrics
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_compute_time": 0.0,
            "average_task_time": 0.0,
            "throughput": 0.0,  # tasks per second
        }

        self._start_scheduler()

    def _start_scheduler(self):
        """Start the task scheduler."""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._scheduler_thread.start()

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Distributed orchestrator started")

    def submit_experiment_batch(
        self, experiments: List[Dict[str, Any]], priority: int = 0
    ) -> List[str]:
        """Submit a batch of experiments for distributed execution."""
        task_ids = []

        for i, experiment_params in enumerate(experiments):
            task_id = f"exp_{int(time.time() * 1000)}_{i}"
            task = DistributedTask(
                task_id=task_id,
                task_type="experiment",
                payload=experiment_params,
                priority=priority,
            )

            self.task_queue.put((priority, time.time(), task))
            self.metrics["tasks_submitted"] += 1
            task_ids.append(task_id)

        logger.info(
            f"Submitted {len(experiments)} experiments for distributed execution"
        )
        return task_ids

    def submit_analysis_task(
        self, analysis_type: str, data: Dict[str, Any], dependencies: List[str] = None
    ) -> str:
        """Submit an analysis task."""
        task_id = f"analysis_{int(time.time() * 1000)}"
        task = DistributedTask(
            task_id=task_id,
            task_type="analysis",
            payload={"analysis_type": analysis_type, "data": data},
            priority=1,  # Higher priority for analysis
            dependencies=dependencies or [],
        )

        self.task_queue.put((1, time.time(), task))
        self.metrics["tasks_submitted"] += 1

        logger.info(f"Submitted analysis task {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get status of a specific task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None

    def wait_for_completion(
        self, task_ids: List[str], timeout: float = None
    ) -> Dict[str, Any]:
        """Wait for specified tasks to complete."""
        start_time = time.time()
        results = {}

        while task_ids and (timeout is None or time.time() - start_time < timeout):
            completed_ids = []

            for task_id in task_ids:
                task = self.get_task_status(task_id)
                if task and task.status in ["completed", "failed"]:
                    results[task_id] = {
                        "status": task.status,
                        "result": task.result,
                        "error": task.error,
                        "duration": (
                            (task.completed_at - task.started_at).total_seconds()
                            if task.completed_at and task.started_at
                            else 0
                        ),
                    }
                    completed_ids.append(task_id)

            # Remove completed tasks from waiting list
            for task_id in completed_ids:
                task_ids.remove(task_id)

            if task_ids:
                time.sleep(0.1)  # Short polling interval

        # Handle timeout
        for task_id in task_ids:
            results[task_id] = {
                "status": "timeout",
                "result": None,
                "error": "Timeout waiting for completion",
            }

        return results

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task (blocking with timeout)
                try:
                    priority, submission_time, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check dependencies
                if not self._dependencies_satisfied(task):
                    # Re-queue task for later
                    self.task_queue.put((priority, submission_time, task))
                    time.sleep(0.1)
                    continue

                # Execute task
                self._execute_task(task)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1.0)

    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_task = self.get_task_status(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True

    def _execute_task(self, task: DistributedTask):
        """Execute a distributed task."""
        task.status = "running"
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task

        # Choose appropriate executor based on task type
        if task.task_type == "experiment":
            future = self.experiment_executor.submit(self._run_experiment, task)
        elif task.task_type == "analysis":
            future = self.analysis_executor.submit(self._run_analysis, task)
        elif task.task_type == "compute":
            future = self.compute_executor.submit(self._run_computation, task)
        else:
            logger.error(f"Unknown task type: {task.task_type}")
            self._complete_task(task, success=False, error="Unknown task type")
            return

        # Add completion callback
        future.add_done_callback(lambda f: self._handle_task_completion(task, f))

    def _run_experiment(self, task: DistributedTask) -> Any:
        """Run a single experiment."""
        try:
            # Import here to avoid circular imports
            from .core import AutonomousLab

            # Create temporary lab instance
            lab = AutonomousLab()

            # Extract parameters
            parameters = task.payload

            # Run experiment
            experiment = lab.run_experiment(parameters)

            return {
                "experiment_id": experiment.id,
                "status": experiment.status,
                "results": experiment.results,
                "duration": experiment.duration,
                "metadata": experiment.metadata,
            }

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            raise

    def _run_analysis(self, task: DistributedTask) -> Any:
        """Run an analysis task."""
        try:
            analysis_type = task.payload["analysis_type"]
            data = task.payload["data"]

            if analysis_type == "parameter_correlation":
                return self._analyze_parameter_correlation(data)
            elif analysis_type == "optimization_progress":
                return self._analyze_optimization_progress(data)
            elif analysis_type == "pattern_detection":
                return self._detect_patterns(data)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

        except Exception as e:
            logger.error(f"Analysis execution failed: {e}")
            raise

    def _run_computation(self, task: DistributedTask) -> Any:
        """Run a CPU-intensive computation."""
        try:
            computation_type = task.payload["computation_type"]

            if computation_type == "bayesian_optimization":
                return self._run_bayesian_optimization(task.payload)
            elif computation_type == "ml_training":
                return self._run_ml_training(task.payload)
            else:
                raise ValueError(f"Unknown computation type: {computation_type}")

        except Exception as e:
            logger.error(f"Computation execution failed: {e}")
            raise

    def _analyze_parameter_correlation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter correlations."""
        import numpy as np

        experiments = data.get("experiments", [])
        if len(experiments) < 3:
            return {"correlation_matrix": {}, "significant_correlations": []}

        # Extract parameters and results
        param_names = set()
        for exp in experiments:
            param_names.update(exp.get("parameters", {}).keys())
            param_names.update(exp.get("results", {}).keys())

        param_names = list(param_names)
        correlation_matrix = {}

        # Calculate correlations
        for param1 in param_names:
            correlation_matrix[param1] = {}
            for param2 in param_names:
                values1 = []
                values2 = []

                for exp in experiments:
                    val1 = exp.get("parameters", {}).get(param1) or exp.get(
                        "results", {}
                    ).get(param1)
                    val2 = exp.get("parameters", {}).get(param2) or exp.get(
                        "results", {}
                    ).get(param2)

                    if val1 is not None and val2 is not None:
                        values1.append(float(val1))
                        values2.append(float(val2))

                if len(values1) >= 3:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlation_matrix[param1][param2] = (
                        float(correlation) if not np.isnan(correlation) else 0.0
                    )
                else:
                    correlation_matrix[param1][param2] = 0.0

        # Find significant correlations
        significant_correlations = []
        for param1 in param_names:
            for param2 in param_names:
                if param1 != param2:
                    corr = correlation_matrix[param1][param2]
                    if abs(corr) > 0.6:
                        significant_correlations.append(
                            {"param1": param1, "param2": param2, "correlation": corr}
                        )

        return {
            "correlation_matrix": correlation_matrix,
            "significant_correlations": significant_correlations,
        }

    def _analyze_optimization_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization progress."""
        convergence_history = data.get("convergence_history", [])
        if not convergence_history:
            return {"progress": "insufficient_data"}

        # Extract fitness values
        fitness_values = [entry.get("best_fitness", 0) for entry in convergence_history]

        # Calculate improvement rate
        if len(fitness_values) > 1:
            improvement_rate = (fitness_values[-1] - fitness_values[0]) / len(
                fitness_values
            )

            # Detect plateaus
            recent_variance = np.var(fitness_values[-min(10, len(fitness_values)) :])
            is_plateaued = recent_variance < 0.001

            return {
                "progress": "analyzed",
                "improvement_rate": improvement_rate,
                "current_fitness": fitness_values[-1],
                "is_plateaued": is_plateaued,
                "total_improvements": len(
                    [
                        i
                        for i in range(1, len(fitness_values))
                        if fitness_values[i] > fitness_values[i - 1]
                    ]
                ),
            }

        return {"progress": "insufficient_data"}

    def _detect_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in experimental data."""
        experiments = data.get("experiments", [])
        if len(experiments) < 5:
            return {"patterns": []}

        patterns = []

        # Pattern 1: Success parameter ranges
        successful_experiments = [
            exp for exp in experiments if exp.get("success", False)
        ]
        if len(successful_experiments) >= 3:
            for param in successful_experiments[0].get("parameters", {}).keys():
                values = [
                    exp["parameters"][param]
                    for exp in successful_experiments
                    if param in exp["parameters"]
                ]
                if values:
                    patterns.append(
                        {
                            "type": "success_range",
                            "parameter": param,
                            "range": [min(values), max(values)],
                            "mean": np.mean(values),
                            "confidence": len(values) / len(experiments),
                        }
                    )

        return {"patterns": patterns}

    def _run_bayesian_optimization(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bayesian optimization computation."""
        # Placeholder for actual Bayesian optimization
        param_space = payload.get("param_space", {})
        n_suggestions = payload.get("n_suggestions", 1)

        # Simple random suggestions for now
        suggestions = []
        for _ in range(n_suggestions):
            suggestion = {}
            for param, (low, high) in param_space.items():
                suggestion[param] = np.random.uniform(low, high)
            suggestions.append(suggestion)

        return {"suggestions": suggestions}

    def _run_ml_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML model training."""
        # Placeholder for ML training
        return {"model_trained": True, "accuracy": 0.85}

    def _handle_task_completion(
        self, task: DistributedTask, future: concurrent.futures.Future
    ):
        """Handle task completion."""
        try:
            result = future.result()
            self._complete_task(task, success=True, result=result)
        except Exception as e:
            self._complete_task(task, success=False, error=str(e))

    def _complete_task(
        self,
        task: DistributedTask,
        success: bool,
        result: Any = None,
        error: str = None,
    ):
        """Mark task as completed."""
        task.status = "completed" if success else "failed"
        task.completed_at = datetime.now()
        task.result = result
        task.error = error

        # Move from active to completed
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        self.completed_tasks[task.task_id] = task

        # Update metrics
        if success:
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1

        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            self.metrics["total_compute_time"] += duration

            # Update average task time
            total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            if total_tasks > 0:
                self.metrics["average_task_time"] = (
                    self.metrics["total_compute_time"] / total_tasks
                )

        logger.debug(f"Task {task.task_id} completed: {task.status}")

    def _monitor_loop(self):
        """Monitor system performance and health."""
        while not self._shutdown_event.is_set():
            try:
                # Calculate throughput
                current_time = time.time()
                if hasattr(self, "_last_throughput_time"):
                    time_delta = current_time - self._last_throughput_time
                    if time_delta > 0:
                        completed_delta = self.metrics["tasks_completed"] - getattr(
                            self, "_last_completed_count", 0
                        )
                        self.metrics["throughput"] = completed_delta / time_delta

                self._last_throughput_time = current_time
                self._last_completed_count = self.metrics["tasks_completed"]

                # Clean up old completed tasks (keep last 1000)
                if len(self.completed_tasks) > 1000:
                    # Remove oldest tasks
                    sorted_tasks = sorted(
                        self.completed_tasks.items(),
                        key=lambda x: x[1].completed_at or datetime.min,
                    )
                    tasks_to_remove = sorted_tasks[:-1000]
                    for task_id, _ in tasks_to_remove:
                        del self.completed_tasks[task_id]

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "compute_nodes": len(self.compute_nodes),
            "executor_status": {
                "experiment_workers": self.experiment_executor._threads,
                "analysis_workers": self.analysis_executor._threads,
                "compute_workers": getattr(self.compute_executor, "_processes", {}),
            },
        }

    def scale_workers(self, new_size: int):
        """Dynamically scale the number of workers."""
        if new_size <= 0:
            return

        logger.info(f"Scaling workers from {self.max_workers} to {new_size}")

        # For thread pools, we need to recreate them
        # This is a simplified approach - in production, you'd want more sophisticated scaling
        old_experiment_executor = self.experiment_executor
        old_analysis_executor = self.analysis_executor

        self.experiment_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=new_size, thread_name_prefix="experiment"
        )
        self.analysis_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, new_size), thread_name_prefix="analysis"
        )

        # Shutdown old executors
        old_experiment_executor.shutdown(wait=False)
        old_analysis_executor.shutdown(wait=False)

        self.max_workers = new_size

    def shutdown(self, wait: bool = True):
        """Shutdown the distributed orchestrator."""
        logger.info("Shutting down distributed orchestrator")

        self._shutdown_event.set()

        # Shutdown executors
        self.experiment_executor.shutdown(wait=wait)
        self.analysis_executor.shutdown(wait=wait)
        self.compute_executor.shutdown(wait=wait)

        # Wait for scheduler thread
        if self._scheduler_thread and wait:
            self._scheduler_thread.join(timeout=5.0)

        if self._monitor_thread and wait:
            self._monitor_thread.join(timeout=5.0)


class DistributedCampaignManager:
    """Manages distributed execution of entire campaigns."""

    def __init__(self, orchestrator: DistributedOrchestrator):
        """Initialize distributed campaign manager."""
        self.orchestrator = orchestrator

    def run_distributed_campaign(
        self,
        objective,
        param_space: Dict[str, tuple],
        initial_samples: int = 20,
        max_experiments: int = 500,
    ) -> Dict[str, Any]:
        """Run a campaign using distributed computing."""
        logger.info("Starting distributed campaign")

        campaign_id = f"dist_campaign_{int(time.time())}"
        start_time = datetime.now()

        # Phase 1: Initial random sampling
        initial_params = self._generate_random_parameters(param_space, initial_samples)
        initial_task_ids = self.orchestrator.submit_experiment_batch(
            initial_params, priority=1
        )

        # Wait for initial experiments
        initial_results = self.orchestrator.wait_for_completion(
            initial_task_ids, timeout=300
        )

        # Analyze initial results
        analysis_data = {
            "experiments": [
                result["result"]
                for result in initial_results.values()
                if result["status"] == "completed"
            ]
        }

        correlation_task_id = self.orchestrator.submit_analysis_task(
            "parameter_correlation", analysis_data
        )

        # Phase 2: Iterative optimization
        all_experiments = list(initial_results.values())
        iteration = 0
        max_iterations = (max_experiments - initial_samples) // 10  # Batch size of 10

        while iteration < max_iterations:
            # Wait for correlation analysis
            correlation_result = self.orchestrator.wait_for_completion(
                [correlation_task_id], timeout=60
            )

            # Generate next batch based on analysis
            next_params = self._generate_next_parameters(
                param_space, all_experiments, batch_size=10
            )

            # Submit next batch
            batch_task_ids = self.orchestrator.submit_experiment_batch(
                next_params, priority=0
            )

            # Wait for batch completion
            batch_results = self.orchestrator.wait_for_completion(
                batch_task_ids, timeout=300
            )
            all_experiments.extend(batch_results.values())

            # Submit new analysis
            updated_analysis_data = {
                "experiments": [
                    result["result"]
                    for result in all_experiments
                    if result.get("status") == "completed"
                ]
            }
            correlation_task_id = self.orchestrator.submit_analysis_task(
                "parameter_correlation", updated_analysis_data
            )

            iteration += 1
            logger.info(f"Completed iteration {iteration}/{max_iterations}")

        end_time = datetime.now()

        # Compile results
        successful_experiments = [
            exp for exp in all_experiments if exp.get("status") == "completed"
        ]

        campaign_result = {
            "campaign_id": campaign_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": (end_time - start_time).total_seconds(),
            "total_experiments": len(all_experiments),
            "successful_experiments": len(successful_experiments),
            "success_rate": (
                len(successful_experiments) / len(all_experiments)
                if all_experiments
                else 0
            ),
            "experiments": all_experiments,
            "objective": {
                "target_property": objective.target_property,
                "target_range": objective.target_range,
            },
        }

        logger.info(
            f"Distributed campaign completed: {len(successful_experiments)} successful experiments"
        )
        return campaign_result

    def _generate_random_parameters(
        self, param_space: Dict[str, tuple], count: int
    ) -> List[Dict[str, float]]:
        """Generate random parameter combinations."""
        import numpy as np

        parameters = []
        for _ in range(count):
            params = {}
            for param, (low, high) in param_space.items():
                params[param] = np.random.uniform(low, high)
            parameters.append(params)

        return parameters

    def _generate_next_parameters(
        self,
        param_space: Dict[str, tuple],
        previous_experiments: List[Dict],
        batch_size: int,
    ) -> List[Dict[str, float]]:
        """Generate next parameter combinations based on previous results."""
        # Simple strategy: random + some bias towards successful regions
        import numpy as np

        parameters = []

        # Find successful experiments
        successful_experiments = [
            exp
            for exp in previous_experiments
            if exp.get("status") == "completed" and exp.get("result")
        ]

        if successful_experiments:
            # 50% exploitation of successful regions
            exploit_count = batch_size // 2
            for _ in range(exploit_count):
                # Pick a random successful experiment
                base_exp = np.random.choice(successful_experiments)
                base_params = base_exp["result"].get("parameters", {})

                # Add small variations
                params = {}
                for param, (low, high) in param_space.items():
                    if param in base_params:
                        base_val = base_params[param]
                        variation = (high - low) * 0.1  # 10% of range
                        new_val = np.random.normal(base_val, variation / 3)
                        new_val = max(low, min(high, new_val))  # Clamp to bounds
                        params[param] = new_val
                    else:
                        params[param] = np.random.uniform(low, high)

                parameters.append(params)

        # Fill remaining with random exploration
        remaining_count = batch_size - len(parameters)
        random_params = self._generate_random_parameters(param_space, remaining_count)
        parameters.extend(random_params)

        return parameters


# Global distributed orchestrator
_global_orchestrator = None


def get_global_orchestrator() -> DistributedOrchestrator:
    """Get global distributed orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = DistributedOrchestrator()
    return _global_orchestrator
