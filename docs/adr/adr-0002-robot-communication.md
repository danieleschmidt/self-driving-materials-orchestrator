# ADR-0002: Robot Communication Architecture

**Status:** Accepted  
**Date:** 2025-08-01  
**Deciders:** Robotics Team, Systems Architecture Team

## Context

The orchestrator needs to communicate with various types of laboratory robots:
- Opentrons liquid handling robots (HTTP API)
- Chemspeed synthesis robots (Serial/TCP)
- Custom ROS2-based robots (ROS2 topics/services)
- Simulation robots for testing (In-process)

Requirements:
- Unified interface for different robot types
- Asynchronous operation support
- Real-time status monitoring
- Error handling and recovery
- Safety system integration

## Decision

We will implement a **layered robot communication architecture** with:

1. **Abstract Robot Driver Interface**: Common API for all robot types
2. **Hardware-Specific Drivers**: Concrete implementations for each robot platform
3. **Robot Orchestrator**: High-level coordination and resource management
4. **Safety Monitor**: Cross-cutting safety system integration

## Consequences

### Positive
- **Unified interface**: Single API for all robot operations regardless of hardware
- **Extensibility**: Easy to add new robot types without changing core logic
- **Testability**: Clear separation allows mocking and simulation
- **Safety integration**: Centralized safety monitoring across all robots
- **Async support**: Non-blocking operations for parallel experiment execution

### Negative
- **Abstraction overhead**: Additional layer may introduce latency
- **Complexity**: More complex than direct hardware communication
- **Driver maintenance**: Need to maintain multiple hardware-specific drivers

### Risks
- **Hardware compatibility**: Changes in robot firmware may break drivers
- **Performance bottlenecks**: Orchestrator could become a performance bottleneck
- **Safety system reliability**: Centralized safety monitoring is critical point of failure

## Architecture Components

### Abstract Robot Driver
```python
class RobotDriver:
    async def connect(self) -> bool
    async def execute_action(self, action: Action) -> Result
    async def get_status(self) -> Status
    async def emergency_stop(self) -> bool
    async def disconnect(self) -> bool
```

### Robot Orchestrator
- Resource allocation and scheduling
- Parallel execution management
- Error handling and retry logic
- Integration with experiment queue

### Safety Monitor
- Emergency stop propagation
- Environmental monitoring integration
- Automated safety protocol execution
- Real-time alert generation

## Alternatives Considered

### Direct Hardware Communication
- **Pros**: Lower latency, simpler initial implementation
- **Cons**: Tight coupling, difficult testing, no unified interface
- **Decision**: Rejected due to maintainability concerns

### ROS2 for All Robots
- **Pros**: Unified communication protocol, excellent tooling
- **Cons**: Overhead for simple robots, requires ROS2 bridge for non-ROS robots
- **Decision**: Rejected due to complexity for simple HTTP/Serial robots

### Message Queue Based (Redis/RabbitMQ)
- **Pros**: Excellent async support, good for distributed systems
- **Cons**: Additional infrastructure dependency, overkill for current scale
- **Decision**: Rejected as unnecessary complexity for current requirements

### gRPC Services
- **Pros**: Type-safe communication, good performance
- **Cons**: Additional complexity, not needed for internal communication
- **Decision**: Rejected as overhead for internal robot communication

## Implementation Notes

- Implement connection pooling for HTTP-based robots
- Use asyncio for non-blocking operations
- Implement exponential backoff for retry logic
- Add comprehensive logging for debugging robot communication
- Include hardware simulators for each robot type
- Implement health checks for all connected robots