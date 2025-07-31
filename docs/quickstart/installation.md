# Installation

This guide will help you install and set up the Self-Driving Materials Orchestrator.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized setup)
- MongoDB (local or Docker)
- Git

## Quick Installation

### Option 1: pip install (Recommended)

```bash
pip install self-driving-materials-orchestrator
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Install in development mode
pip install -e ".[dev]"
```

## Database Setup

### MongoDB with Docker

```bash
# Start MongoDB container
docker run -d -p 27017:27017 --name materials-db mongo:5.0
```

### MongoDB Local Installation

Follow the [MongoDB installation guide](https://docs.mongodb.com/manual/installation/) for your operating system.

## Verification

Test your installation:

```bash
# Check CLI is working
materials-orchestrator --help

# Test basic functionality
python -c "from materials_orchestrator import AutonomousLab; print('Installation successful!')"
```

## Optional Dependencies

### Robot Drivers

For physical robot integration:

```bash
pip install self-driving-materials-orchestrator[robots]
```

### Documentation

To build documentation locally:

```bash
pip install self-driving-materials-orchestrator[docs]
mkdocs serve
```

## Docker Setup

For a complete containerized environment:

```bash
# Clone repository
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Start all services
docker-compose up -d

# Access dashboard at http://localhost:8501
# Access API at http://localhost:8000
```

## Development Setup

For contributors and developers:

```bash
# Clone and install
git clone https://github.com/terragonlabs/self-driving-materials-orchestrator.git
cd self-driving-materials-orchestrator

# Install with all dependencies
pip install -e ".[dev,robots,docs]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Next Steps

- [Run your first campaign](first-campaign.md)
- [Set up robots](robot-setup.md)
- [Configure monitoring](../deployment/monitoring.md)