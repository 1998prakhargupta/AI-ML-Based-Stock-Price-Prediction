# Makefile for Price Predictor Project - Enterprise Structure

.PHONY: help setup install test lint clean run-pipeline train-model docs

# Default target
help:
	@echo "🚀 Price Predictor Project - Enterprise Commands"
	@echo "================================================"
	@echo "🔧 Development Commands:"
	@echo "  dev-setup      - Complete development environment setup"
	@echo "  dev-install    - Install development dependencies"
	@echo "  dev-run        - Run application in development mode"
	@echo "  dev-test       - Run tests in development mode"
	@echo ""
	@echo "🏗️ Build & Installation:"
	@echo "  setup          - Set up project environment"
	@echo "  install        - Install dependencies"
	@echo "  build          - Build application"
	@echo "  clean          - Clean temporary files"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e       - Run end-to-end tests"
	@echo "  coverage       - Generate test coverage report"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code"
	@echo "  security       - Run security checks"
	@echo "  validate       - Validate configuration"
	@echo "  gitkeep        - Manage .gitkeep files"
	@echo "  structure      - Validate project structure"
	@echo ""
	@echo "🚀 Application Commands:"
	@echo "  run-app        - Run main application"
	@echo "  run-pipeline   - Run data pipeline"
	@echo "  train-model    - Train ML models"
	@echo "  run-predictions - Run prediction models"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-dev     - Run development environment"
	@echo "  docker-prod    - Run production environment"
	@echo "  docker-clean   - Clean Docker resources"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs           - Generate documentation"
	@echo "  docs-serve     - Serve documentation locally"
	@echo ""
	@echo "🔒 Compliance & Security:"
	@echo "  compliance     - Run compliance checks"
	@echo "  audit          - Security audit"

# Environment variables
export PYTHONPATH := $(shell pwd)
export CONFIG_DIR := config
export ENVIRONMENT ?= development

# Project setup
setup:
	@echo "🔧 Setting up project environment..."
	python scripts/setup.py
	@echo "📁 Creating required directories..."
	mkdir -p data/{raw,processed,cache,outputs,plots,reports,backups}
	mkdir -p logs models/{production,experiments,checkpoints,archived}
	mkdir -p external/{vendor,apis,plugins}
	@echo "✅ Project setup complete!"

# Development setup
dev-setup: setup dev-install
	@echo "🛠️ Setting up development environment..."
	cp .env.example .env
	pre-commit install
	python3 tools/gitkeep_manager.py create
	@echo "✅ Development setup complete!"

# Install dependencies
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt
	@echo "✅ Production dependencies installed!"

dev-install:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt -r requirements-dev.txt
	@echo "✅ Development dependencies installed!"

build:
	@echo "🏗️ Building application..."
	python setup.py build
	@echo "✅ Build complete!"

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@echo "✅ Cleanup complete!"

# Testing commands  
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ All tests completed!"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v
	@echo "✅ Unit tests completed!"

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v
	@echo "✅ Integration tests completed!"

test-e2e:
	@echo "🧪 Running end-to-end tests..."
	pytest tests/e2e/ -v
	@echo "✅ E2E tests completed!"

coverage:
	@echo "📊 Generating test coverage report..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generated in htmlcov/"

# Code quality
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ app/ tools/ --count --select=E9,F63,F7,F82 --show-source --statistics
	pylint src/ app/ tools/ --errors-only
	@echo "✅ Linting completed!"

format:
	@echo "✨ Formatting code..."
	black src/ app/ tools/ config/
	isort src/ app/ tools/ config/
	@echo "✅ Code formatting completed!"

security:
	@echo "🔒 Running security checks..."
	bandit -r src/ app/ tools/ -f json -o security-report.json
	safety check
	@echo "✅ Security checks completed!"

validate:
	@echo "🔍 Validating configuration..."
	python3 -c "from config import get_config; config = get_config(); issues = config.validate(); print('✅ Configuration is valid!' if not issues else f'❌ Issues found: {issues}')"
	@echo "✅ Configuration validation completed!"

gitkeep:
	@echo "📁 Managing .gitkeep files..."
	python3 tools/gitkeep_manager.py maintenance
	@echo "✅ .gitkeep management completed!"

structure:
	@echo "🏗️ Validating project structure..."
	python3 tools/gitkeep_manager.py validate
	tree -I '__pycache__|*.pyc|.git|.venv|node_modules' -L 3
	@echo "✅ Project structure validation completed!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Run tests
test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v --cov=src/

test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v

test-integration:
	@echo "🧪 Running integration tests..."
	python -m pytest tests/integration/ -v

test-compliance:
	@echo "🛡️ Running compliance tests..."
	python -m pytest tests/compliance/ -v

# Code quality
lint:
	@echo "🔍 Running linting..."
	flake8 src/ --max-line-length=120
	pylint src/

format:
	@echo "✨ Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Clean up
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/

# Run applications
run-pipeline:
	@echo "🚀 Running data pipeline..."
	python scripts/data_pipeline.py

train-model:
	@echo "🤖 Training models..."
	python scripts/train_model.py

run-predictions:
	@echo "📈 Running predictions..."
	python scripts/run_predictions.py

# Documentation
docs:
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/

# Compliance
compliance:
	@echo "🛡️ Running compliance demonstration..."
	python scripts/compliance_demo.py

# Development
dev-install:
	@echo "🛠️ Installing development dependencies..."
	pip install -r requirements-dev.txt

pre-commit:
	@echo "🔍 Running pre-commit checks..."
	pre-commit run --all-files

# Docker commands (if needed)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t price-predictor .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm price-predictor

# Environment management
create-env:
	@echo "🌐 Creating virtual environment..."
	python -m venv .venv

activate-env:
	@echo "🌐 Activate with: source .venv/bin/activate"

# Data management
fetch-data:
	@echo "📊 Fetching market data..."
	python -c "from scripts.data_pipeline import DataPipeline; p = DataPipeline(); p.fetch_market_data(['TCS', 'INFY'])"

# Monitoring
check-logs:
	@echo "📝 Checking application logs..."
	tail -f logs/application.log

check-compliance-logs:
	@echo "🛡️ Checking compliance logs..."
	tail -f logs/compliance.log

# All-in-one commands
dev-setup: setup dev-install pre-commit
	@echo "✅ Development environment ready!"

production-setup: setup install test
	@echo "✅ Production environment ready!"

quick-start: setup install run-pipeline
	@echo "🎉 Quick start completed!"
