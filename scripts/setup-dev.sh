#!/bin/bash
# Development environment setup script

set -e

echo "🔧 Setting up Materials Orchestrator development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is not compatible. Requires Python $required_version or higher"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📋 Installing package dependencies..."
pip install -e ".[dev,robots,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data logs configs notebooks

# Set up database
echo "🗄️ Setting up MongoDB..."
if command -v docker &> /dev/null; then
    if ! docker ps | grep -q materials-db; then
        echo "🐳 Starting MongoDB container..."
        docker run -d -p 27017:27017 --name materials-db \
            -v "$(pwd)/scripts/init-mongo.js:/docker-entrypoint-initdb.d/init-mongo.js:ro" \
            mongo:5.0
        
        # Wait for MongoDB to be ready
        echo "⏳ Waiting for MongoDB to be ready..."
        sleep 10
    else
        echo "✅ MongoDB container already running"
    fi
else
    echo "⚠️ Docker not found. Please install MongoDB manually or install Docker."
fi

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/unit/ -v

# Check code quality
echo "🔍 Running code quality checks..."
make quality || echo "⚠️ Code quality checks failed. Run 'make quality' to see details."

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Start development server: make dev"
echo "  4. View dashboard: make dashboard"
echo "  5. Build documentation: make docs-serve"
echo ""
echo "For VS Code users:"
echo "  - Install recommended extensions from .vscode/extensions.json"
echo "  - Use the integrated debugger with provided launch configurations"
echo ""