"""Robot orchestration and coordination system."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .base import RobotDriver, InstrumentDriver, RobotStatus, ActionResult
from .drivers import SimulatedDriver, OpentronsDriver, ChemspeedDriver, XRDInstrument

logger = logging.getLogger(__name__)


class RobotOrchestrator:
    """Coordinates multiple robots and instruments for autonomous experiments."""

    def __init__(self, max_concurrent_actions: int = 5):
        """Initialize robot orchestrator.

        Args:
            max_concurrent_actions: Maximum number of concurrent robot actions
        """
        self._robots: Dict[str, RobotDriver] = {}
        self._instruments: Dict[str, InstrumentDriver] = {}
        self._action_queue: List[Dict[str, Any]] = []
        self._running_actions: Dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent_actions
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_actions)
        self._is_running = False

    def add_robot(self, robot_id: str, driver: RobotDriver):
        """Add robot to orchestrator.

        Args:
            robot_id: Unique robot identifier
            driver: Robot driver instance
        """
        self._robots[robot_id] = driver
        logger.info(f"Added robot {robot_id} to orchestrator")

    def add_instrument(self, instrument_id: str, driver: InstrumentDriver):
        """Add instrument to orchestrator.

        Args:
            instrument_id: Unique instrument identifier
            driver: Instrument driver instance
        """
        self._instruments[instrument_id] = driver
        logger.info(f"Added instrument {instrument_id} to orchestrator")

    def create_robot(
        self, robot_id: str, robot_type: str, config: Optional[Dict[str, Any]] = None
    ) -> RobotDriver:
        """Create and add robot by type.

        Args:
            robot_id: Unique robot identifier
            robot_type: Type of robot ('simulated', 'opentrons', 'chemspeed')
            config: Robot configuration

        Returns:
            Created robot driver
        """
        if robot_type == "simulated":
            driver = SimulatedDriver(robot_id, config)
        elif robot_type == "opentrons":
            driver = OpentronsDriver(robot_id, config)
        elif robot_type == "chemspeed":
            driver = ChemspeedDriver(robot_id, config)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

        self.add_robot(robot_id, driver)
        return driver

    def create_instrument(
        self,
        instrument_id: str,
        instrument_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> InstrumentDriver:
        """Create and add instrument by type.

        Args:
            instrument_id: Unique instrument identifier
            instrument_type: Type of instrument ('xrd', 'uv_vis', etc.)
            config: Instrument configuration

        Returns:
            Created instrument driver
        """
        if instrument_type == "xrd":
            driver = XRDInstrument(instrument_id, config)
        else:
            # For now, create a simulated instrument
            class SimulatedInstrument(InstrumentDriver):
                async def connect(self) -> bool:
                    self.status = RobotStatus.IDLE
                    return True

                async def disconnect(self) -> bool:
                    self.status = RobotStatus.DISCONNECTED
                    return True

                async def measure(
                    self, sample_id: str, parameters: Dict[str, Any]
                ) -> Dict[str, Any]:
                    await asyncio.sleep(1.0)
                    return {
                        "sample_id": sample_id,
                        "measurement_type": instrument_type,
                        "timestamp": datetime.now().isoformat(),
                        "data": {"simulated": True},
                    }

                async def calibrate(self) -> ActionResult:
                    return ActionResult(
                        success=True, action="calibrate", parameters={}, duration=1.0
                    )

            driver = SimulatedInstrument(instrument_id, config)

        self.add_instrument(instrument_id, driver)
        return driver

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all robots and instruments.

        Returns:
            Dictionary of connection results
        """
        logger.info("Connecting to all robots and instruments")
        results = {}

        # Connect robots
        for robot_id, robot in self._robots.items():
            try:
                success = await robot.connect()
                results[robot_id] = success
                if success:
                    logger.info(f"Successfully connected to robot {robot_id}")
                else:
                    logger.error(f"Failed to connect to robot {robot_id}")
            except Exception as e:
                logger.error(f"Error connecting to robot {robot_id}: {e}")
                results[robot_id] = False

        # Connect instruments
        for instrument_id, instrument in self._instruments.items():
            try:
                success = await instrument.connect()
                results[instrument_id] = success
                if success:
                    logger.info(f"Successfully connected to instrument {instrument_id}")
                else:
                    logger.error(f"Failed to connect to instrument {instrument_id}")
            except Exception as e:
                logger.error(f"Error connecting to instrument {instrument_id}: {e}")
                results[instrument_id] = False

        return results

    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all robots and instruments.

        Returns:
            Dictionary of disconnection results
        """
        logger.info("Disconnecting from all robots and instruments")
        results = {}

        # Cancel running actions
        for task in self._running_actions.values():
            task.cancel()

        # Disconnect robots
        for robot_id, robot in self._robots.items():
            try:
                success = await robot.disconnect()
                results[robot_id] = success
            except Exception as e:
                logger.error(f"Error disconnecting from robot {robot_id}: {e}")
                results[robot_id] = False

        # Disconnect instruments
        for instrument_id, instrument in self._instruments.items():
            try:
                success = await instrument.disconnect()
                results[instrument_id] = success
            except Exception as e:
                logger.error(
                    f"Error disconnecting from instrument {instrument_id}: {e}"
                )
                results[instrument_id] = False

        return results

    async def execute_protocol(self, protocol: Dict[str, Any]) -> List[ActionResult]:
        """Execute a complete experimental protocol.

        Args:
            protocol: Protocol definition with steps

        Returns:
            List of action results
        """
        steps = protocol.get("steps", [])
        results = []

        logger.info(f"Executing protocol with {len(steps)} steps")

        for i, step in enumerate(steps):
            logger.info(
                f"Executing step {i+1}/{len(steps)}: {step.get('action', 'unknown')}"
            )

            robot_id = step.get("robot")
            instrument_id = step.get("instrument")
            action = step.get("action")
            parameters = step.get("parameters", {})

            if robot_id and robot_id in self._robots:
                # Robot action
                robot = self._robots[robot_id]
                result = await robot.execute_action(action, parameters)
                results.append(result)

                if not result.success:
                    logger.error(f"Step {i+1} failed: {result.error}")
                    break

            elif instrument_id and instrument_id in self._instruments:
                # Instrument measurement
                instrument = self._instruments[instrument_id]
                sample_id = parameters.get("sample_id", f"sample_{i}")
                measurement_result = await instrument.measure(sample_id, parameters)

                # Convert to ActionResult format
                result = ActionResult(
                    success=True,
                    action=f"measure_{action}",
                    parameters=parameters,
                    duration=1.0,
                    data=measurement_result,
                )
                results.append(result)

            else:
                # Unknown robot/instrument
                result = ActionResult(
                    success=False,
                    action=action,
                    parameters=parameters,
                    duration=0.0,
                    error=f"Robot/instrument not found: {robot_id or instrument_id}",
                )
                results.append(result)
                break

        logger.info(f"Protocol execution completed with {len(results)} steps")
        return results

    async def execute_parallel_actions(
        self, actions: List[Dict[str, Any]]
    ) -> List[ActionResult]:
        """Execute multiple actions in parallel.

        Args:
            actions: List of action definitions

        Returns:
            List of action results
        """
        tasks = []

        for action_spec in actions:
            robot_id = action_spec.get("robot")
            instrument_id = action_spec.get("instrument")
            action = action_spec.get("action")
            parameters = action_spec.get("parameters", {})

            if robot_id and robot_id in self._robots:
                robot = self._robots[robot_id]
                task = asyncio.create_task(robot.execute_action(action, parameters))
                tasks.append(task)

            elif instrument_id and instrument_id in self._instruments:
                instrument = self._instruments[instrument_id]
                sample_id = parameters.get("sample_id", "sample")
                task = asyncio.create_task(instrument.measure(sample_id, parameters))
                tasks.append(task)

        if not tasks:
            return []

        logger.info(f"Executing {len(tasks)} actions in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to ActionResult format
        action_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                action_results.append(
                    ActionResult(
                        success=False,
                        action=actions[i].get("action", "unknown"),
                        parameters=actions[i].get("parameters", {}),
                        duration=0.0,
                        error=str(result),
                    )
                )
            elif isinstance(result, ActionResult):
                action_results.append(result)
            else:
                # Instrument measurement result
                action_results.append(
                    ActionResult(
                        success=True,
                        action=actions[i].get("action", "measure"),
                        parameters=actions[i].get("parameters", {}),
                        duration=1.0,
                        data=result,
                    )
                )

        return action_results

    def get_available_robots(self) -> List[str]:
        """Get list of available (idle) robots.

        Returns:
            List of robot IDs
        """
        available = []
        for robot_id, robot in self._robots.items():
            if robot.is_available:
                available.append(robot_id)
        return available

    def get_available_instruments(self) -> List[str]:
        """Get list of available instruments.

        Returns:
            List of instrument IDs
        """
        available = []
        for instrument_id, instrument in self._instruments.items():
            if instrument.status == RobotStatus.IDLE:
                available.append(instrument_id)
        return available

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            System status information
        """
        robot_statuses = {}
        for robot_id, robot in self._robots.items():
            robot_statuses[robot_id] = await robot.get_status()

        instrument_statuses = {}
        for instrument_id, instrument in self._instruments.items():
            instrument_statuses[instrument_id] = {
                "instrument_id": instrument_id,
                "status": instrument.status.value,
                "measurement_count": len(instrument.get_measurement_history()),
            }

        return {
            "robots": robot_statuses,
            "instruments": instrument_statuses,
            "available_robots": self.get_available_robots(),
            "available_instruments": self.get_available_instruments(),
            "running_actions": list(self._running_actions.keys()),
            "queue_length": len(self._action_queue),
        }

    async def emergency_stop_all(self) -> Dict[str, ActionResult]:
        """Emergency stop all robots and instruments.

        Returns:
            Dictionary of emergency stop results
        """
        logger.warning("EMERGENCY STOP activated for all robots")
        results = {}

        # Stop all robots
        for robot_id, robot in self._robots.items():
            try:
                result = await robot.emergency_stop()
                results[robot_id] = result
            except Exception as e:
                results[robot_id] = ActionResult(
                    success=False,
                    action="emergency_stop",
                    parameters={},
                    duration=0.0,
                    error=str(e),
                )

        # Cancel all running tasks
        for task in self._running_actions.values():
            task.cancel()
        self._running_actions.clear()

        return results
