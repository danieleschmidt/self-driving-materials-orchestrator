#!/bin/bash

# Self-Driving Materials Orchestrator Development Environment Setup
echo "🤖 Setting up Self-Driving Materials Orchestrator development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    jq \
    tree \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
echo "📋 Installing project dependencies..."
pip install -e ".[dev,robots,docs]"

# Setup pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p data/experiments
mkdir -p data/models
mkdir -p data/exports
mkdir -p configs/robots
mkdir -p configs/instruments

# Set up git configuration
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspace
git config --global core.autocrlf false
git config --global core.editor "code --wait"

# Install additional development tools
echo "🛠️ Installing development tools..."
pip install \
    ipython \
    jupyter \
    notebook \
    jupyterlab \
    matplotlib \
    seaborn \
    plotly \
    dash

# Install MongoDB tools
echo "🗄️ Installing MongoDB tools..."
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-mongosh

# Set up shell aliases and functions
echo "🔧 Setting up shell configuration..."
cat >> ~/.bashrc << 'EOF'

# Self-Driving Materials Orchestrator aliases
alias mo="materials-orchestrator"
alias pytest-cov="pytest --cov=materials_orchestrator --cov-report=html --cov-report=term"
alias lint="ruff check src/ tests/"
alias format="black src/ tests/"
alias typecheck="mypy src/"
alias docs-serve="mkdocs serve -a 0.0.0.0:8080"
alias test-integration="pytest tests/integration/ -v"
alias test-unit="pytest tests/unit/ -v"

# Development shortcuts
alias logs="tail -f logs/orchestrator.log"
alias mongo-connect="mongosh mongodb://localhost:27017/materials_discovery"

# Git shortcuts
alias gst="git status"
alias gco="git checkout"
alias gcb="git checkout -b"
alias gp="git push"
alias gpl="git pull"
alias gd="git diff"
alias gl="git log --oneline -10"

# Docker shortcuts
alias dcu="docker-compose up -d"
alias dcd="docker-compose down"
alias dcl="docker-compose logs -f"
alias dps="docker ps"

# Quick project setup
function setup_experiment() {
    local name=${1:-"test_experiment"}
    mkdir -p "experiments/$name"
    echo "📊 Created experiment directory: experiments/$name"
}

function run_campaign() {
    local config=${1:-"examples/perovskite_discovery_example.py"}
    echo "🚀 Running campaign: $config"
    python "$config"
}

EOF

# Set up environment variables
echo "🌍 Setting up environment variables..."
cp .env.example .env
echo "ENVIRONMENT=development" >> .env
echo "DEBUG=true" >> .env
echo "SIMULATION_MODE=true" >> .env

# Initialize database (if MongoDB is running)
echo "🗄️ Initializing database..."
if command -v mongosh &> /dev/null; then
    echo "Creating database indexes..."
    cat > /tmp/init_db.js << 'EOF'
use materials_discovery;

// Create collections and indexes
db.experiments.createIndex({ "campaign_id": 1, "timestamp": -1 });
db.experiments.createIndex({ "parameters": 1 });
db.experiments.createIndex({ "results.band_gap": 1 });
db.experiments.createIndex({ "metadata.status": 1 });

db.campaigns.createIndex({ "objective.target_property": 1 });
db.campaigns.createIndex({ "created_at": -1 });
db.campaigns.createIndex({ "status": 1 });

db.models.createIndex({ "campaign_id": 1, "version": -1 });
db.models.createIndex({ "created_at": -1 });

print("Database initialized successfully!");
EOF

    # Try to run the initialization (will fail if MongoDB is not running, which is OK)
    mongosh mongodb://localhost:27017/ /tmp/init_db.js 2>/dev/null || echo "⚠️  MongoDB not running - database will be initialized on first use"
    rm /tmp/init_db.js
fi

# Generate sample configuration files
echo "📄 Generating sample configuration files..."

# Robot configuration
cat > configs/robots/opentrons_ot2.json << 'EOF'
{
  "name": "Opentrons OT-2",
  "type": "liquid_handler",
  "connection": {
    "type": "http",
    "host": "192.168.1.100",
    "port": 31950
  },
  "capabilities": [
    "pipetting",
    "dispensing",
    "mixing",
    "heating",
    "cooling"
  ],
  "deck_configuration": {
    "slots": 11,
    "labware": {
      "1": "96-well-plate",
      "2": "reagent-reservoir",
      "3": "tip-rack-20ul",
      "4": "tip-rack-200ul"
    }
  }
}
EOF

# Instrument configuration
cat > configs/instruments/uvvis_spectrometer.json << 'EOF'
{
  "name": "UV-Vis Spectrometer",
  "type": "analytical",
  "connection": {
    "type": "scpi",
    "host": "192.168.1.102",
    "port": 5025
  },
  "parameters": {
    "wavelength_range": [200, 800],
    "resolution": 1.0,
    "integration_time": 1000
  },
  "measurements": [
    "absorbance",
    "transmittance",
    "reflectance"
  ]
}
EOF

# Create development scripts
echo "📝 Creating development scripts..."
mkdir -p scripts/dev

cat > scripts/dev/reset-database.sh << 'EOF'
#!/bin/bash
echo "🗄️ Resetting development database..."
mongosh mongodb://localhost:27017/materials_discovery --eval "db.dropDatabase()"
echo "✅ Database reset complete"
EOF
chmod +x scripts/dev/reset-database.sh

cat > scripts/dev/run-tests.sh << 'EOF'
#!/bin/bash
echo "🧪 Running comprehensive test suite..."
pytest tests/ --cov=materials_orchestrator --cov-report=html --cov-report=term-missing -v
echo "📊 Coverage report generated in htmlcov/"
EOF
chmod +x scripts/dev/run-tests.sh

cat > scripts/dev/lint-and-format.sh << 'EOF'
#!/bin/bash
echo "🔍 Running code quality checks..."
ruff check src/ tests/ --fix
black src/ tests/
mypy src/
echo "✅ Code quality checks complete"
EOF
chmod +x scripts/dev/lint-and-format.sh

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name materials-orchestrator --display-name "Materials Orchestrator"

# Create development notebooks directory
mkdir -p notebooks/development
cat > notebooks/development/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Self-Driving Materials Orchestrator - Development Notebook\n",
    "\n",
    "This notebook provides a quick introduction to the development environment and core functionality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '/workspace/src')\n",
    "\n",
    "from materials_orchestrator import AutonomousLab, MaterialsObjective\n",
    "print(\"✅ Self-Driving Materials Orchestrator imported successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quick test of the core functionality\n",
    "objective = MaterialsObjective(\n",
    "    target_property=\"band_gap\",\n",
    "    target_range=(1.2, 1.6),\n",
    "    optimization_direction=\"target\"\n",
    ")\n",
    "print(f\"Created objective: {objective}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Materials Orchestrator",
   "language": "python",
   "name": "materials-orchestrator"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Final setup message
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Quick Start Commands:"
echo "  mo --help                    # CLI help"
echo "  pytest-cov                  # Run tests with coverage"
echo "  lint                         # Check code style"
echo "  format                       # Format code"
echo "  docs-serve                   # Serve documentation"
echo "  setup_experiment <name>     # Create experiment directory"
echo "  run_campaign                 # Run example campaign"
echo ""
echo "🌐 Available ports:"
echo "  8000  - FastAPI server"
echo "  8501  - Streamlit dashboard"
echo "  27017 - MongoDB"
echo "  9090  - Prometheus metrics"
echo "  8080  - Documentation server"
echo ""
echo "📖 Documentation: http://localhost:8080"
echo "🎮 Dashboard: http://localhost:8501"
echo ""
echo "Happy experimenting! 🔬✨"