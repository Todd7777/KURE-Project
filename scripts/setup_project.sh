#!/bin/bash
# Setup script for Nonlinear Rectified Flows project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    echo -e "${2}${1}${NC}"
}

print_message "🚀 Setting up Nonlinear Rectified Flows project..." "$BLUE"
echo ""

# Check Python version
print_message "Checking Python version..." "$BLUE"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_message "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION" "$RED"
    exit 1
fi
print_message "✅ Python version: $PYTHON_VERSION" "$GREEN"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_message "⚠️  Warning: Not in a virtual environment" "$YELLOW"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_message "Setup cancelled. Please activate a virtual environment first." "$YELLOW"
        exit 1
    fi
fi

# Install package
print_message "📦 Installing package..." "$BLUE"
pip install -e ".[dev]"
print_message "✅ Package installed" "$GREEN"
echo ""

# Set up pre-commit hooks
print_message "🔧 Setting up pre-commit hooks..." "$BLUE"
pre-commit install
print_message "✅ Pre-commit hooks installed" "$GREEN"
echo ""

# Create necessary directories
print_message "📁 Creating project directories..." "$BLUE"
mkdir -p data
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs
mkdir -p results
print_message "✅ Directories created" "$GREEN"
echo ""

# Check GPU availability
print_message "🎮 Checking GPU availability..." "$BLUE"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || print_message "⚠️  No GPU detected" "$YELLOW"
echo ""

# Run tests
print_message "🧪 Running tests..." "$BLUE"
if pytest tests/ -v --tb=short -x; then
    print_message "✅ All tests passed!" "$GREEN"
else
    print_message "⚠️  Some tests failed. Please check the output above." "$YELLOW"
fi
echo ""

# Check code quality
print_message "🔍 Checking code quality..." "$BLUE"
if make format-check && make lint; then
    print_message "✅ Code quality checks passed!" "$GREEN"
else
    print_message "⚠️  Code quality issues detected. Run 'make format' to fix." "$YELLOW"
fi
echo ""

# Print summary
print_message "═══════════════════════════════════════════════════════" "$BLUE"
print_message "✨ Setup complete!" "$GREEN"
print_message "═══════════════════════════════════════════════════════" "$BLUE"
echo ""
print_message "Next steps:" "$BLUE"
echo "  1. Download datasets: make download-data"
echo "  2. Train a model: make train-spline"
echo "  3. Evaluate: make eval"
echo "  4. Generate samples: make sample"
echo ""
print_message "Useful commands:" "$BLUE"
echo "  make help          - Show all available commands"
echo "  make test          - Run tests"
echo "  make ci            - Run all CI checks"
echo "  make docker-build  - Build Docker images"
echo ""
print_message "Documentation:" "$BLUE"
echo "  README.md          - Project overview"
echo "  EXPERIMENTS.md     - Experiment guide"
echo "  CONTRIBUTING.md    - Contribution guidelines"
echo ""
print_message "Happy researching! 🎉" "$GREEN"
