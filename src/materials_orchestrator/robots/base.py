"""Base classes for robot drivers and protocols."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RobotStatus(Enum):
    """Robot operational status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ActionResult:
    """Result of a robot action."""

    success: bool
    action: str
    parameters: Dict[str, Any]
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RobotDriver(ABC):
    """Abstract base class for robot drivers."""

    def __init__(self, robot_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize robot driver.

        Args:
            robot_id: Unique identifier for this robot
            config: Robot-specific configuration
        """
        self.robot_id = robot_id
        self.config = config or {}
        self.status = RobotStatus.DISCONNECTED
        self._capabilities: List[str] = []
        self._current_action: Optional[str] = None
        self._action_history: List[ActionResult] = []

    @property
    def capabilities(self) -> List[str]:
        """Get list of robot capabilities."""
        return self._capabilities.copy()

    @property
    def is_available(self) -> bool:
        """Check if robot is available for new actions."""
        return self.status == RobotStatus.IDLE

    @property
    def current_action(self) -> Optional[str]:
        """Get currently executing action."""
        return self._current_action

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the robot.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the robot.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    async def execute_action(
        self, action: str, parameters: Dict[str, Any]
    ) -> ActionResult:
        """Execute a specific action.

        Args:
            action: Action name
            parameters: Action parameters

        Returns:
            Result of the action
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed robot status.

        Returns:
            Status information dictionary
        """
        pass

    @abstractmethod
    async def calibrate(self) -> ActionResult:
        """Perform robot calibration.

        Returns:
            Calibration result
        """
        pass

    async def home(self) -> ActionResult:
        """Move robot to home position.

        Returns:
            Homing result
        """
        return await self.execute_action("home", {})

    async def emergency_stop(self) -> ActionResult:
        """Emergency stop all robot operations.

        Returns:
            Emergency stop result
        """
        logger.warning(f"Emergency stop triggered for robot {self.robot_id}")
        self.status = RobotStatus.ERROR
        self._current_action = None

        return ActionResult(
            success=True,
            action="emergency_stop",
            parameters={},
            duration=0.0,
            message="Emergency stop executed",
        )

    def get_action_history(self, limit: Optional[int] = None) -> List[ActionResult]:
        """Get robot action history.

        Args:
            limit: Maximum number of actions to return

        Returns:
            List of action results
        """
        history = self._action_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def _record_action(self, result: ActionResult):
        """Record action result in history."""
        self._action_history.append(result)

        # Keep only last 1000 actions
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]


class InstrumentDriver(ABC):
    """Abstract base class for analytical instrument drivers."""

    def __init__(self, instrument_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize instrument driver.

        Args:
            instrument_id: Unique identifier
            config: Instrument configuration
        """
        self.instrument_id = instrument_id
        self.config = config or {}
        self.status = RobotStatus.DISCONNECTED
        self._measurement_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the instrument."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the instrument."""
        pass

    @abstractmethod
    async def measure(
        self, sample_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform measurement on sample.

        Args:
            sample_id: Sample identifier
            parameters: Measurement parameters

        Returns:
            Measurement results
        """
        pass

    @abstractmethod
    async def calibrate(self) -> ActionResult:
        """Calibrate the instrument."""
        pass

    def get_measurement_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get measurement history."""
        history = self._measurement_history.copy()
        if limit:
            history = history[-limit:]
        return history


@dataclass
class RobotCapability:
    """Description of a robot capability."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[float] = None  # seconds
