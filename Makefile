.PHONY: help install test lint format type-check security-check clean build docker-build docker-run

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Install development dependencies
	pip install -e ".[dev,robots,docs]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests
	pytest tests/integration/

test-robot: ## Run robot hardware tests
	pytest tests/robot/ -m robot

test-coverage: ## Run tests with coverage
	pytest --cov=materials_orchestrator --cov-report=html --cov-report=term

# Code quality
lint: ## Run linting
	ruff check .

lint-fix: ## Run linting with auto-fix
	ruff check . --fix

format: ## Format code with Black
	black .

format-check: ## Check code formatting
	black --check .

type-check: ## Run type checking
	mypy src/

security-check: ## Run security analysis
	bandit -r src/
	safety check

# Quality checks (combined)
quality: format-check lint type-check security-check ## Run all quality checks

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Documentation
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

# Cleaning
clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean ## Build package
	python -m build

upload-test: build ## Upload to TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload: build ## Upload to PyPI
	twine upload dist/*

# Docker
docker-build: ## Build Docker image
	docker build -t materials-orchestrator:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -p 8501:8501 materials-orchestrator:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# Database
db-start: ## Start MongoDB container
	docker run -d -p 27017:27017 --name materials-db mongo:5.0

db-stop: ## Stop MongoDB container
	docker stop materials-db && docker rm materials-db

# Development server
dev: ## Start development server
	python -m materials_orchestrator.cli launch --debug

dashboard: ## Start dashboard only
	python -m materials_orchestrator.cli dashboard

# Benchmarks
benchmark: ## Run performance benchmarks
	pytest benchmarks/ --benchmark-json=benchmark.json

# Setup for different environments
setup-dev: install ## Setup development environment
	pre-commit install
	mkdir -p data logs configs

setup-prod: ## Setup production environment
	pip install -e .
	mkdir -p data logs configs

# Monitoring and maintenance
health-check: ## Check system health
	python -c "from materials_orchestrator.cli import app; app()"

logs: ## View application logs
	tail -f logs/materials_orchestrator.log

# Robot-specific commands  
robot-test: ## Test robot connections
	python -m materials_orchestrator.robots.test_connections

robot-calibrate: ## Calibrate robot systems
	python -m materials_orchestrator.robots.calibrate

# Campaign management
campaign-start: ## Start a new campaign (requires parameters)
	@echo "Usage: make campaign-start OBJECTIVE=band_gap MATERIAL=perovskites"
	@echo "python -m materials_orchestrator.cli campaign $(OBJECTIVE) --material-system $(MATERIAL)"

campaign-status: ## Check campaign status  
	python -m materials_orchestrator.cli status

# Backup and restore
backup-db: ## Backup MongoDB database
	mkdir -p backups
	mongodump --host localhost:27017 --db materials_discovery --out backups/$(shell date +%Y%m%d_%H%M%S)

restore-db: ## Restore MongoDB database (specify BACKUP_DIR)
	@echo "Usage: make restore-db BACKUP_DIR=backups/20250130_120000"
	mongorestore --host localhost:27017 --db materials_discovery $(BACKUP_DIR)/materials_discovery

# Version management
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version  
	bump2version minor

version-major: ## Bump major version
	bump2version major

# Release process
release-check: ## Check if ready for release
	@echo "Running pre-release checks..."
	make quality
	make test
	make build
	@echo "✅ Ready for release"

release: release-check ## Create release
	@echo "Creating release..."
	make version-minor
	git push --tags
	make upload