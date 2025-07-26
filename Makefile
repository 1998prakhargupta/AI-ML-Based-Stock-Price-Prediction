# Makefile for Price Predictor Project

.PHONY: help setup install test lint clean run-pipeline train-model docs

# Default target
help:
	@echo "🚀 Price Predictor Project Commands"
	@echo "=================================="
	@echo "setup          - Set up project environment"
	@echo "install        - Install dependencies"
	@echo "test           - Run all tests"
	@echo "test-unit      - Run unit tests"
	@echo "test-integration - Run integration tests"
	@echo "lint           - Run code linting"
	@echo "format         - Format code"
	@echo "clean          - Clean temporary files"
	@echo "run-pipeline   - Run data pipeline"
	@echo "train-model    - Train ML models"
	@echo "docs           - Generate documentation"
	@echo "compliance     - Run compliance checks"

# Project setup
setup:
	@echo "🔧 Setting up project..."
	python scripts/setup.py

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
