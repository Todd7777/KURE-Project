.PHONY: help install install-dev clean test lint format type-check pre-commit docker-build docker-run train eval sample docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Nonlinear Rectified Flows - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation
install: ## Install package and dependencies
	@echo "$(BLUE)Installing package...$(NC)"
	pip install -e .

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing package with dev dependencies...$(NC)"
	pip install -e ".[dev,docs]"
	pre-commit install

# Cleaning
clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

clean-all: clean ## Clean everything including checkpoints and outputs
	@echo "$(YELLOW)Warning: This will delete checkpoints and outputs!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/ outputs/ results/ logs/; \
		echo "$(GREEN)Cleaned all generated files$(NC)"; \
	fi

# Testing
test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest tests/ -v -n auto --cov=src --cov-report=term

test-coverage: ## Run tests with HTML coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

# Code quality
lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 src/ scripts/ tests/
	@echo "$(GREEN)Linting passed!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/ --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)Type checking passed!$(NC)"

pre-commit: ## Run all pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# Docker
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build

docker-dev: ## Start development container
	@echo "$(BLUE)Starting development container...$(NC)"
	docker-compose run --rm dev

docker-jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter server...$(NC)"
	docker-compose up jupyter

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker-compose run --rm test

docker-clean: ## Clean Docker images and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down -v
	docker system prune -f

# Training
train-linear: ## Train with linear teacher
	@echo "$(BLUE)Training with linear teacher...$(NC)"
	python scripts/train.py --config configs/base.yaml

train-quadratic: ## Train with quadratic teacher
	@echo "$(BLUE)Training with quadratic teacher...$(NC)"
	python scripts/train.py --config configs/quadratic.yaml

train-spline: ## Train with cubic spline teacher
	@echo "$(BLUE)Training with cubic spline teacher...$(NC)"
	python scripts/train.py --config configs/spline.yaml

train-sb: ## Train with Schrödinger Bridge teacher
	@echo "$(BLUE)Training with Schrödinger Bridge teacher...$(NC)"
	python scripts/train.py --config configs/sb.yaml

# Evaluation
eval: ## Evaluate model on COCO
	@echo "$(BLUE)Evaluating model...$(NC)"
	python scripts/evaluate.py \
		--checkpoint checkpoints/nrf_spline_best.pt \
		--dataset coco \
		--steps "1 2 4 8"

eval-compositional: ## Evaluate on compositional suite
	@echo "$(BLUE)Evaluating compositional capabilities...$(NC)"
	python scripts/evaluate.py \
		--checkpoint checkpoints/nrf_spline_best.pt \
		--eval_type compositional \
		--compositional_steps 4

# Sampling
sample: ## Generate sample images
	@echo "$(BLUE)Generating samples...$(NC)"
	python scripts/sample.py \
		--checkpoint checkpoints/nrf_spline_best.pt \
		--prompt "A red cube on top of a blue sphere" \
		--steps 4 \
		--num_samples 16

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentation built in docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8080

# CI/CD
ci: format-check lint type-check test ## Run all CI checks
	@echo "$(GREEN)All CI checks passed!$(NC)"

# Git hooks
setup-hooks: ## Set up git hooks
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Git hooks installed!$(NC)"

# Release
release-patch: ## Create a patch release
	@echo "$(BLUE)Creating patch release...$(NC)"
	bump2version patch
	git push && git push --tags

release-minor: ## Create a minor release
	@echo "$(BLUE)Creating minor release...$(NC)"
	bump2version minor
	git push && git push --tags

release-major: ## Create a major release
	@echo "$(BLUE)Creating major release...$(NC)"
	bump2version major
	git push && git push --tags

# Utilities
check-gpu: ## Check GPU availability
	@echo "$(BLUE)Checking GPU...$(NC)"
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

download-data: ## Download datasets (placeholder)
	@echo "$(YELLOW)Data download not implemented yet$(NC)"
	@echo "Please manually download CC3M, LAION, and COCO datasets"

setup: install-dev setup-hooks ## Complete setup (install + hooks)
	@echo "$(GREEN)Setup complete!$(NC)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python -m pytest tests/benchmarks/ -v --benchmark-only

# Profiling
profile: ## Profile training performance
	@echo "$(BLUE)Profiling training...$(NC)"
	python -m cProfile -o profile.stats scripts/train.py --config configs/base.yaml
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# All-in-one commands
all: clean install test lint ## Clean, install, test, and lint
	@echo "$(GREEN)All tasks completed!$(NC)"

quick-check: format-check lint test-fast ## Quick validation before commit
	@echo "$(GREEN)Quick check passed!$(NC)"
