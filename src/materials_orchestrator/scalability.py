"""Scalability and distributed processing for Materials Orchestrator."""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerNode:
    """Information about a worker node."""

    node_id: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_jobs: int = 4
    current_jobs: int = 0
    status: str = "active"  # active, inactive, error
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if worker is available for new jobs."""
        return (
            self.status == "active"
            and self.current_jobs < self.max_concurrent_jobs
            and (datetime.now() - self.last_heartbeat).total_seconds() < 60
        )

    @property
    def load_factor(self) -> float:
        """Get current load factor (0.0 to 1.0)."""
        if self.max_concurrent_jobs == 0:
            return 1.0
        return self.current_jobs / self.max_concurrent_jobs


@dataclass
class DistributedJob:
    """Distributed job definition."""

    job_id: str
    job_type: str
    payload: Dict[str, Any]
    priority: int = 0  # Higher = more priority
    created_at: datetime = field(default_factory=datetime.now)
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def status(self) -> str:
        """Get job status."""
        if self.error:
            return "failed"
        elif self.completed_at:
            return "completed"
        elif self.started_at:
            return "running"
        elif self.assigned_node:
            return "assigned"
        else:
            return "pending"

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: int = 300,
    ):
        """Initialize auto scaler.

        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Load threshold for scaling up
            scale_down_threshold: Load threshold for scaling down
            cooldown_seconds: Cooldown period between scaling actions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds

        self._last_scale_action = datetime.now() - timedelta(seconds=cooldown_seconds)
        self._scaling_history: List[Dict[str, Any]] = []
        self._metrics_history: List[Dict[str, Any]] = []

    def should_scale(
        self,
        current_workers: int,
        queue_length: int,
        average_load: float,
        average_response_time: float,
    ) -> Optional[str]:
        """Determine if scaling is needed.

        Args:
            current_workers: Current number of workers
            queue_length: Length of job queue
            average_load: Average worker load (0.0 to 1.0)
            average_response_time: Average response time in seconds

        Returns:
            'up', 'down', or None
        """
        # Check cooldown period
        if (
            datetime.now() - self._last_scale_action
        ).total_seconds() < self.cooldown_seconds:
            return None

        # Record current metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "workers": current_workers,
            "queue_length": queue_length,
            "average_load": average_load,
            "response_time": average_response_time,
        }
        self._metrics_history.append(metrics)

        # Keep only recent metrics
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

        # Scale up conditions
        scale_up_reasons = []

        if (
            average_load > self.scale_up_threshold
            and current_workers < self.max_workers
        ):
            scale_up_reasons.append(
                f"load ({average_load:.2f}) > threshold ({self.scale_up_threshold})"
            )

        if queue_length > current_workers * 2 and current_workers < self.max_workers:
            scale_up_reasons.append(
                f"queue ({queue_length}) > 2x workers ({current_workers})"
            )

        if average_response_time > 60.0 and current_workers < self.max_workers:
            scale_up_reasons.append(
                f"response time ({average_response_time:.1f}s) > 60s"
            )

        # Scale down conditions
        scale_down_reasons = []

        if (
            average_load < self.scale_down_threshold
            and queue_length == 0
            and current_workers > self.min_workers
        ):
            scale_down_reasons.append(
                f"load ({average_load:.2f}) < threshold ({self.scale_down_threshold})"
            )

        if (
            average_response_time < 5.0
            and queue_length == 0
            and current_workers > self.min_workers
        ):
            scale_down_reasons.append(
                f"response time ({average_response_time:.1f}s) < 5s"
            )

        # Make scaling decision
        if scale_up_reasons:
            logger.info(f"Scaling up: {', '.join(scale_up_reasons)}")
            self._record_scaling_action("up", scale_up_reasons)
            return "up"
        elif scale_down_reasons:
            logger.info(f"Scaling down: {', '.join(scale_down_reasons)}")
            self._record_scaling_action("down", scale_down_reasons)
            return "down"

        return None

    def _record_scaling_action(self, action: str, reasons: List[str]):
        """Record scaling action.

        Args:
            action: Scaling action (up/down)
            reasons: Reasons for scaling
        """
        self._last_scale_action = datetime.now()

        record = {
            "timestamp": self._last_scale_action.isoformat(),
            "action": action,
            "reasons": reasons,
        }

        self._scaling_history.append(record)

        # Keep only recent history
        if len(self._scaling_history) > 50:
            self._scaling_history = self._scaling_history[-50:]

    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling history.

        Returns:
            List of scaling actions
        """
        return self._scaling_history.copy()

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history.

        Returns:
            List of metrics snapshots
        """
        return self._metrics_history.copy()


class DistributedJobManager:
    """Manages distributed job execution across worker nodes."""

    def __init__(self, enable_auto_scaling: bool = True):
        """Initialize distributed job manager.

        Args:
            enable_auto_scaling: Enable automatic scaling
        """
        self.enable_auto_scaling = enable_auto_scaling

        self._workers: Dict[str, WorkerNode] = {}
        self._job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_jobs: Dict[str, DistributedJob] = {}
        self._completed_jobs: List[DistributedJob] = []
        self._job_stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retry_count": 0,
        }

        if enable_auto_scaling:
            self.auto_scaler = AutoScaler()

        self._lock = threading.Lock()
        self._dispatcher_thread = None
        self._heartbeat_thread = None
        self._running = False

    def start(self):
        """Start the job manager."""
        if self._running:
            return

        self._running = True

        # Start dispatcher thread
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True
        )
        self._dispatcher_thread.start()

        # Start heartbeat monitor
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        logger.info("Distributed job manager started")

    def stop(self):
        """Stop the job manager."""
        self._running = False

        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5)

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

        logger.info("Distributed job manager stopped")

    def register_worker(self, worker: WorkerNode):
        """Register a worker node.

        Args:
            worker: Worker node to register
        """
        with self._lock:
            self._workers[worker.node_id] = worker
            logger.info(
                f"Registered worker {worker.node_id} at {worker.host}:{worker.port}"
            )

    def unregister_worker(self, node_id: str):
        """Unregister a worker node.

        Args:
            node_id: Worker node ID
        """
        with self._lock:
            if node_id in self._workers:
                # Reassign active jobs from this worker
                jobs_to_reassign = [
                    job
                    for job in self._active_jobs.values()
                    if job.assigned_node == node_id
                    and job.status in ["assigned", "running"]
                ]

                for job in jobs_to_reassign:
                    logger.warning(
                        f"Reassigning job {job.job_id} from failed worker {node_id}"
                    )
                    job.assigned_node = None
                    job.started_at = None
                    job.retry_count += 1

                    if job.retry_count <= job.max_retries:
                        self._job_queue.put((-job.priority, job.created_at, job))
                        del self._active_jobs[job.job_id]
                    else:
                        job.error = f"Worker {node_id} failed, max retries exceeded"
                        self._completed_jobs.append(job)
                        del self._active_jobs[job.job_id]
                        self._job_stats["failed"] += 1

                del self._workers[node_id]
                logger.info(f"Unregistered worker {node_id}")

    def submit_job(self, job: DistributedJob) -> str:
        """Submit a job for execution.

        Args:
            job: Job to submit

        Returns:
            Job ID
        """
        with self._lock:
            self._job_queue.put((-job.priority, job.created_at, job))
            self._job_stats["submitted"] += 1

        logger.debug(f"Submitted job {job.job_id} (type: {job.job_type})")
        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status.

        Args:
            job_id: Job ID

        Returns:
            Job status or None if not found
        """
        with self._lock:
            # Check active jobs
            if job_id in self._active_jobs:
                return self._active_jobs[job_id].status

            # Check completed jobs
            for job in self._completed_jobs:
                if job.job_id == job_id:
                    return job.status

            # Check queue
            temp_jobs = []
            found_job = None

            while not self._job_queue.empty():
                try:
                    priority, created_at, job = self._job_queue.get_nowait()
                    if job.job_id == job_id:
                        found_job = job
                    temp_jobs.append((priority, created_at, job))
                except queue.Empty:
                    break

            # Put jobs back
            for item in temp_jobs:
                self._job_queue.put(item)

            return found_job.status if found_job else None

    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get job result.

        Args:
            job_id: Job ID

        Returns:
            Job result or None
        """
        with self._lock:
            # Check active jobs
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                return job.result if job.status == "completed" else None

            # Check completed jobs
            for job in self._completed_jobs:
                if job.job_id == job_id:
                    return job.result

            return None

    def _dispatcher_loop(self):
        """Main job dispatcher loop."""
        while self._running:
            try:
                # Get next job
                try:
                    priority, created_at, job = self._job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Find available worker
                worker = self._select_worker(job)
                if worker is None:
                    # No workers available, put job back
                    self._job_queue.put((priority, created_at, job))
                    time.sleep(1.0)
                    continue

                # Assign job to worker
                with self._lock:
                    job.assigned_node = worker.node_id
                    worker.current_jobs += 1
                    self._active_jobs[job.job_id] = job

                # Execute job asynchronously
                threading.Thread(
                    target=self._execute_job_on_worker, args=(job, worker), daemon=True
                ).start()

            except Exception as e:
                logger.error(f"Error in dispatcher loop: {e}")

    def _select_worker(self, job: DistributedJob) -> Optional[WorkerNode]:
        """Select best worker for job.

        Args:
            job: Job to assign

        Returns:
            Selected worker or None
        """
        with self._lock:
            available_workers = [
                worker for worker in self._workers.values() if worker.is_available
            ]

            if not available_workers:
                return None

            # Check capability requirements
            if job.job_type != "generic":
                capable_workers = [
                    worker
                    for worker in available_workers
                    if job.job_type in worker.capabilities
                ]
                if capable_workers:
                    available_workers = capable_workers

            # Select worker with lowest load
            best_worker = min(available_workers, key=lambda w: w.load_factor)
            return best_worker

    def _execute_job_on_worker(self, job: DistributedJob, worker: WorkerNode):
        """Execute job on worker.

        Args:
            job: Job to execute
            worker: Worker node
        """
        job.started_at = datetime.now()

        try:
            # Simulate job execution (in real implementation, this would send to worker)
            result = self._simulate_job_execution(job)

            job.result = result
            job.completed_at = datetime.now()

            with self._lock:
                worker.current_jobs = max(0, worker.current_jobs - 1)
                self._completed_jobs.append(job)
                del self._active_jobs[job.job_id]
                self._job_stats["completed"] += 1

            logger.debug(f"Job {job.job_id} completed on worker {worker.node_id}")

        except Exception as e:
            job.error = str(e)
            job.completed_at = datetime.now()

            with self._lock:
                worker.current_jobs = max(0, worker.current_jobs - 1)

                # Retry if possible
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.assigned_node = None
                    job.started_at = None
                    job.error = None

                    self._job_queue.put((-job.priority, job.created_at, job))
                    del self._active_jobs[job.job_id]
                    self._job_stats["retry_count"] += 1

                    logger.warning(
                        f"Retrying job {job.job_id} (attempt {job.retry_count})"
                    )
                else:
                    self._completed_jobs.append(job)
                    del self._active_jobs[job.job_id]
                    self._job_stats["failed"] += 1

                    logger.error(
                        f"Job {job.job_id} failed after {job.retry_count} retries: {e}"
                    )

    def _simulate_job_execution(self, job: DistributedJob) -> Any:
        """Simulate job execution (placeholder).

        Args:
            job: Job to execute

        Returns:
            Job result
        """
        # This would be replaced with actual worker communication
        import random

        time.sleep(random.uniform(0.1, 2.0))  # Simulate work

        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated job failure")

        return {
            "result": f"Processed {job.job_type}",
            "timestamp": datetime.now().isoformat(),
        }

    def _heartbeat_loop(self):
        """Monitor worker heartbeats."""
        while self._running:
            try:
                current_time = datetime.now()

                with self._lock:
                    for worker in self._workers.values():
                        # Check heartbeat timeout
                        time_since_heartbeat = (
                            current_time - worker.last_heartbeat
                        ).total_seconds()

                        if time_since_heartbeat > 120:  # 2 minutes timeout
                            if worker.status == "active":
                                worker.status = "inactive"
                                logger.warning(
                                    f"Worker {worker.node_id} marked as inactive (heartbeat timeout)"
                                )

                # Auto-scaling check
                if self.enable_auto_scaling and hasattr(self, "auto_scaler"):
                    self._check_auto_scaling()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        with self._lock:
            active_workers = sum(
                1 for w in self._workers.values() if w.status == "active"
            )
            queue_length = self._job_queue.qsize()

            if active_workers > 0:
                total_load = sum(
                    w.load_factor
                    for w in self._workers.values()
                    if w.status == "active"
                )
                average_load = total_load / active_workers
            else:
                average_load = 0.0

            # Calculate average response time
            recent_jobs = [j for j in self._completed_jobs[-50:] if j.duration]
            average_response_time = (
                sum(j.duration for j in recent_jobs) / len(recent_jobs)
                if recent_jobs
                else 0.0
            )

            scale_action = self.auto_scaler.should_scale(
                current_workers=active_workers,
                queue_length=queue_length,
                average_load=average_load,
                average_response_time=average_response_time,
            )

            if scale_action:
                logger.info(f"Auto-scaling recommendation: {scale_action}")
                # In a real implementation, this would trigger worker provisioning

    def update_worker_heartbeat(self, node_id: str):
        """Update worker heartbeat.

        Args:
            node_id: Worker node ID
        """
        with self._lock:
            if node_id in self._workers:
                self._workers[node_id].last_heartbeat = datetime.now()
                if self._workers[node_id].status == "inactive":
                    self._workers[node_id].status = "active"
                    logger.info(f"Worker {node_id} reactivated")

    def get_system_status(self) -> Dict[str, Any]:
        """Get distributed system status.

        Returns:
            System status information
        """
        with self._lock:
            active_workers = [w for w in self._workers.values() if w.status == "active"]

            return {
                "workers": {
                    "total": len(self._workers),
                    "active": len(active_workers),
                    "average_load": (
                        sum(w.load_factor for w in active_workers) / len(active_workers)
                        if active_workers
                        else 0
                    ),
                },
                "jobs": {
                    "queued": self._job_queue.qsize(),
                    "active": len(self._active_jobs),
                    "completed": len(self._completed_jobs),
                    "stats": self._job_stats.copy(),
                },
                "auto_scaling": {
                    "enabled": self.enable_auto_scaling,
                    "recent_actions": (
                        self.auto_scaler.get_scaling_history()[-5:]
                        if self.enable_auto_scaling
                        else []
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            }


class ExperimentBatchProcessor:
    """Process large batches of experiments efficiently."""

    def __init__(self, job_manager: DistributedJobManager):
        """Initialize batch processor.

        Args:
            job_manager: Distributed job manager
        """
        self.job_manager = job_manager
        self._batch_results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def process_experiment_batch(
        self,
        parameters_list: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        priority: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Process a batch of experiments.

        Args:
            parameters_list: List of parameter sets
            batch_id: Optional batch ID
            priority: Job priority
            progress_callback: Optional progress callback

        Returns:
            Batch ID
        """
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        # Create jobs for each parameter set
        job_ids = []

        for i, parameters in enumerate(parameters_list):
            job = DistributedJob(
                job_id=f"{batch_id}_{i}",
                job_type="experiment",
                payload={"parameters": parameters, "batch_id": batch_id},
                priority=priority,
            )

            job_id = self.job_manager.submit_job(job)
            job_ids.append(job_id)

        # Initialize batch tracking
        with self._lock:
            self._batch_results[batch_id] = {
                "job_ids": job_ids,
                "total_jobs": len(job_ids),
                "completed_jobs": 0,
                "results": {},
                "errors": {},
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "running",
            }

        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_batch, args=(batch_id, progress_callback), daemon=True
        )
        monitor_thread.start()

        logger.info(f"Started batch {batch_id} with {len(job_ids)} experiments")
        return batch_id

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status.

        Args:
            batch_id: Batch ID

        Returns:
            Batch status or None
        """
        with self._lock:
            return self._batch_results.get(batch_id, {}).copy()

    def get_batch_results(self, batch_id: str) -> Optional[List[Any]]:
        """Get batch results.

        Args:
            batch_id: Batch ID

        Returns:
            List of results or None
        """
        with self._lock:
            batch_info = self._batch_results.get(batch_id)
            if not batch_info or batch_info["status"] != "completed":
                return None

            # Return results in original order
            results = []
            for i in range(batch_info["total_jobs"]):
                job_id = f"{batch_id}_{i}"
                if job_id in batch_info["results"]:
                    results.append(batch_info["results"][job_id])
                elif job_id in batch_info["errors"]:
                    results.append({"error": batch_info["errors"][job_id]})
                else:
                    results.append(None)

            return results

    def _monitor_batch(self, batch_id: str, progress_callback: Optional[Callable]):
        """Monitor batch progress.

        Args:
            batch_id: Batch ID
            progress_callback: Progress callback function
        """
        while True:
            with self._lock:
                batch_info = self._batch_results.get(batch_id)
                if not batch_info:
                    break

                job_ids = batch_info["job_ids"]
                completed_count = 0

                # Check each job
                for job_id in job_ids:
                    status = self.job_manager.get_job_status(job_id)

                    if status == "completed":
                        if job_id not in batch_info["results"]:
                            result = self.job_manager.get_job_result(job_id)
                            batch_info["results"][job_id] = result
                            completed_count += 1
                    elif status == "failed":
                        if job_id not in batch_info["errors"]:
                            # Get error information
                            batch_info["errors"][job_id] = "Job failed"
                            completed_count += 1

                batch_info["completed_jobs"] = len(batch_info["results"]) + len(
                    batch_info["errors"]
                )

                # Check if batch is complete
                if batch_info["completed_jobs"] >= batch_info["total_jobs"]:
                    batch_info["status"] = "completed"
                    batch_info["end_time"] = datetime.now().isoformat()

                    logger.info(
                        f"Batch {batch_id} completed: {batch_info['completed_jobs']} jobs"
                    )

                    if progress_callback:
                        progress_callback(
                            batch_id,
                            batch_info["completed_jobs"],
                            batch_info["total_jobs"],
                        )

                    break

                # Call progress callback
                if progress_callback and completed_count > 0:
                    progress_callback(
                        batch_id, batch_info["completed_jobs"], batch_info["total_jobs"]
                    )

            time.sleep(5)  # Check every 5 seconds


# Global distributed system instances
_global_job_manager: Optional[DistributedJobManager] = None
_global_batch_processor: Optional[ExperimentBatchProcessor] = None


def get_job_manager() -> DistributedJobManager:
    """Get global distributed job manager.

    Returns:
        Distributed job manager
    """
    global _global_job_manager
    if _global_job_manager is None:
        _global_job_manager = DistributedJobManager()
        _global_job_manager.start()
    return _global_job_manager


def get_batch_processor() -> ExperimentBatchProcessor:
    """Get global batch processor.

    Returns:
        Experiment batch processor
    """
    global _global_batch_processor
    if _global_batch_processor is None:
        job_manager = get_job_manager()
        _global_batch_processor = ExperimentBatchProcessor(job_manager)
    return _global_batch_processor


def distributed_experiment(priority: int = 0):
    """Decorator for distributing experiment execution.

    Args:
        priority: Job priority

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract parameters
            parameters = kwargs.get("parameters", {})

            # Create distributed job
            job = DistributedJob(
                job_id=str(uuid.uuid4()),
                job_type="experiment",
                payload={"function": func.__name__, "parameters": parameters},
                priority=priority,
            )

            job_manager = get_job_manager()
            job_id = job_manager.submit_job(job)

            # Wait for completion (blocking)
            while True:
                status = job_manager.get_job_status(job_id)
                if status in ["completed", "failed"]:
                    break
                time.sleep(0.1)

            if status == "completed":
                return job_manager.get_job_result(job_id)
            else:
                raise RuntimeError(f"Distributed experiment failed: {job_id}")

        return wrapper

    return decorator
