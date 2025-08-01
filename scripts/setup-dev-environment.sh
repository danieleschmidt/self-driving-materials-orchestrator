#!/bin/bash
set -e

echo "🚀 Setting up Materials Orchestrator development environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is available
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python $python_version is compatible"
else
    print_error "Python 3.9+ is required. Found: $python_version"
    print_status "Please install Python 3.9 or later"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev,robots,docs]"

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data logs configs backups
mkdir -p data/experiments data/models data/cache
print_success "Project directories created"

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration"
else
    print_status ".env file already exists"
fi

# Test installation
print_status "Testing installation..."
if python -c "import materials_orchestrator; print('✅ Package import successful')"; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Run initial code quality checks
print_status "Running initial code quality checks..."
if make quality > /dev/null 2>&1; then
    print_success "Code quality checks passed"
else
    print_warning "Some code quality issues found. Run 'make quality' for details"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    print_status "Docker is available - you can use 'make docker-compose-up' for full stack"
else
    print_warning "Docker not found - some features may not be available"
fi

# Check if MongoDB is available
if command -v mongod &> /dev/null; then
    print_status "MongoDB is available locally"
elif docker --version &> /dev/null; then
    print_status "MongoDB can be run via Docker: 'make db-start'"
else
    print_warning "MongoDB not found - you'll need to install it or use Docker"
fi

print_success "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate  # Activate virtual environment"
echo "  2. make help                 # See available commands"
echo "  3. make db-start            # Start MongoDB (Docker)"
echo "  4. make dev                 # Start development server"
echo "  5. make dashboard           # Start dashboard"
echo ""
echo "📚 Documentation: https://self-driving-materials.readthedocs.io"
echo "🐛 Issues: https://github.com/danieleschmidt/self-driving-materials-orchestrator/issues"