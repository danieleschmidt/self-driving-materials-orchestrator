"""Mock robot implementations for testing."""

import asyncio
import random
from typing import Dict, Any, List
from unittest.mock import Mock
from materials_orchestrator.robots.base import RobotDriver, RobotStatus

class MockOpentronsDriver(RobotDriver):
    """Mock Opentrons robot driver for testing."""
    
    def __init__(self, simulation_mode: bool = True):
        super().__init__()
        self.simulation_mode = simulation_mode
        self.connected = False
        self.position = {"x": 0, "y": 0, "z": 50}
        self.pipette_volume = 0
        self.tip_attached = False
        self.failure_rate = 0.05  # 5% chance of random failures
        
    async def connect(self) -> bool:
        """Simulate connection to Opentrons robot."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        return True
        
    async def disconnect(self) -> bool:
        """Simulate disconnection from robot."""
        await asyncio.sleep(0.05)
        self.connected = False
        return True
        
    async def get_status(self) -> RobotStatus:
        """Get simulated robot status."""
        if not self.connected:
            return RobotStatus.DISCONNECTED
            
        # Simulate random status changes
        statuses = [RobotStatus.IDLE, RobotStatus.RUNNING, RobotStatus.PAUSED]
        weights = [0.7, 0.2, 0.1]
        return random.choices(statuses, weights=weights)[0]
        
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulated robot action."""
        if not self.connected:
            raise ConnectionError("Robot not connected")
            
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated robot failure during {action}")
            
        # Simulate action execution time
        execution_times = {
            "pick_up_tip": 0.5,
            "drop_tip": 0.3,
            "aspirate": 0.8,
            "dispense": 0.6,
            "move_to": 0.4,
            "home": 1.0
        }
        
        await asyncio.sleep(execution_times.get(action, 0.5))
        
        # Update internal state based on action
        if action == "pick_up_tip":
            self.tip_attached = True
        elif action == "drop_tip":
            self.tip_attached = False
        elif action == "aspirate":
            self.pipette_volume = parameters.get("volume", 0)
        elif action == "dispense":
            self.pipette_volume = 0
        elif action == "move_to":
            self.position.update(parameters.get("position", {}))
            
        return {
            "success": True,
            "action": action,
            "parameters": parameters,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    async def emergency_stop(self) -> bool:
        """Simulate emergency stop."""
        await asyncio.sleep(0.1)
        return True


class MockChemspeedDriver(RobotDriver):
    """Mock Chemspeed robot driver for testing."""
    
    def __init__(self, simulation_mode: bool = True):
        super().__init__()
        self.simulation_mode = simulation_mode
        self.connected = False
        self.temperature = 25.0
        self.stirrer_speed = 0
        self.reactor_volumes = [0] * 4
        self.valve_positions = ["closed"] * 8
        self.failure_rate = 0.03
        
    async def connect(self) -> bool:
        """Simulate connection to Chemspeed robot."""
        await asyncio.sleep(0.2)
        self.connected = True
        return True
        
    async def disconnect(self) -> bool:
        """Simulate disconnection."""
        await asyncio.sleep(0.1)
        self.connected = False
        return True
        
    async def get_status(self) -> RobotStatus:
        """Get simulated status."""
        if not self.connected:
            return RobotStatus.DISCONNECTED
            
        # Simulate heating/cooling states
        if self.temperature > 30:
            return RobotStatus.RUNNING
        return RobotStatus.IDLE
        
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulated synthesis action."""
        if not self.connected:
            raise ConnectionError("Robot not connected")
            
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated failure during {action}")
            
        # Simulate different synthesis actions
        execution_times = {
            "set_temperature": 1.0,
            "start_stirring": 0.5,
            "stop_stirring": 0.3,
            "add_reagent": 0.8,
            "open_valve": 0.2,
            "close_valve": 0.2,
            "wait": parameters.get("duration", 1.0)
        }
        
        await asyncio.sleep(execution_times.get(action, 0.5))
        
        # Update internal state
        if action == "set_temperature":
            self.temperature = parameters.get("temperature", 25.0)
        elif action == "start_stirring":
            self.stirrer_speed = parameters.get("speed", 300)
        elif action == "stop_stirring":
            self.stirrer_speed = 0
        elif action == "add_reagent":
            reactor = parameters.get("reactor", 0)
            volume = parameters.get("volume", 0)
            if 0 <= reactor < len(self.reactor_volumes):
                self.reactor_volumes[reactor] += volume
                
        return {
            "success": True,
            "action": action,
            "parameters": parameters,
            "state": {
                "temperature": self.temperature,
                "stirrer_speed": self.stirrer_speed,
                "reactor_volumes": self.reactor_volumes.copy()
            }
        }
        
    async def emergency_stop(self) -> bool:
        """Simulate emergency stop."""
        self.temperature = 25.0
        self.stirrer_speed = 0
        await asyncio.sleep(0.1)
        return True


class MockInstrument:
    """Mock analytical instrument for testing."""
    
    def __init__(self, instrument_type: str = "uv_vis"):
        self.instrument_type = instrument_type
        self.connected = False
        self.calibrated = False
        self.measurement_count = 0
        
    async def connect(self) -> bool:
        """Connect to instrument."""
        await asyncio.sleep(0.2)
        self.connected = True
        return True
        
    async def calibrate(self) -> bool:
        """Perform instrument calibration."""
        if not self.connected:
            return False
        await asyncio.sleep(2.0)  # Calibration takes time
        self.calibrated = True
        return True
        
    async def measure(self, sample_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform measurement on sample."""
        if not self.connected or not self.calibrated:
            raise RuntimeError("Instrument not ready")
            
        # Simulate measurement time
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        self.measurement_count += 1
        
        # Generate realistic mock data based on instrument type
        if self.instrument_type == "uv_vis":
            wavelengths = list(range(300, 800, 10))
            absorbance = [random.uniform(0, 2) for _ in wavelengths]
            return {
                "sample_id": sample_id,
                "measurement_type": "uv_vis_spectrum",
                "wavelengths": wavelengths,
                "absorbance": absorbance,
                "band_gap": random.uniform(1.0, 2.5)  # Derived property
            }
            
        elif self.instrument_type == "xrd":
            angles = list(range(10, 80, 1))
            intensity = [random.uniform(0, 1000) for _ in angles]
            return {
                "sample_id": sample_id,
                "measurement_type": "xrd_pattern",
                "two_theta": angles,
                "intensity": intensity,
                "crystallinity": random.uniform(0.5, 1.0)
            }
            
        else:
            return {
                "sample_id": sample_id,
                "measurement_type": self.instrument_type,
                "value": random.uniform(0, 100),
                "units": "a.u."
            }


def create_mock_robot_orchestrator() -> Mock:
    """Create a mock robot orchestrator with pre-configured robots."""
    orchestrator = Mock()
    
    # Mock robots
    orchestrator.robots = {
        "opentrons": MockOpentronsDriver(),
        "chemspeed": MockChemspeedDriver()
    }
    
    # Mock instruments
    orchestrator.instruments = {
        "uv_vis": MockInstrument("uv_vis"),
        "xrd": MockInstrument("xrd"),
        "pl_spectrometer": MockInstrument("photoluminescence")
    }
    
    # Mock methods
    async def mock_execute_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Mock protocol execution."""
        await asyncio.sleep(0.5)  # Simulate execution time
        return {
            "success": True,
            "protocol_id": protocol.get("id", "test_protocol"),
            "execution_time": 0.5,
            "results": {"synthesis_success": True}
        }
    
    orchestrator.execute_protocol = mock_execute_protocol
    orchestrator.get_robot_status = lambda robot_id: RobotStatus.IDLE
    
    return orchestrator