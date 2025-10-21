# Contributing to Nonlinear Rectified Flows

Thank you for your interest in contributing to NRF! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Open an issue to discuss it
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve docs, add examples, or fix typos
- **Testing**: Add test cases or improve test coverage
- **Research**: Share experimental results or ablation studies

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/KURE-Project.git
cd KURE-Project/nrf_project

# Add upstream remote
git remote add upstream https://github.com/Todd7777/KURE-Project.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
conda create -n nrf-dev python=3.10
conda activate nrf-dev

# Install in development mode
make install-dev

# Set up pre-commit hooks
make setup-hooks
```

### 3. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

## 📝 Development Workflow

### Code Style

We follow strict code quality standards:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **Google-style docstrings**

Run all checks:

```bash
make ci
```

Or individually:

```bash
make format      # Format code
make lint        # Run linters
make type-check  # Type checking
make test        # Run tests
```

### Writing Code

1. **Follow existing patterns**: Look at similar code in the codebase
2. **Add type hints**: All functions should have type annotations
3. **Write docstrings**: Use Google-style docstrings
4. **Keep it simple**: Prefer clarity over cleverness
5. **Test your code**: Add unit tests for new features

Example:

```python
def compute_velocity(
    x_t: torch.Tensor,
    t: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute velocity at time t.
    
    Args:
        x_t: Current state, shape (B, C, H, W)
        t: Time, shape (B,)
        context: Optional conditioning, shape (B, D)
        
    Returns:
        Velocity field, shape (B, C, H, W)
        
    Raises:
        ValueError: If input shapes are incompatible
    """
    if x_t.shape[0] != t.shape[0]:
        raise ValueError(f"Batch size mismatch: {x_t.shape[0]} vs {t.shape[0]}")
    
    # Implementation here
    return velocity
```

### Writing Tests

All new code should have tests:

```python
import pytest
import torch

def test_compute_velocity_shape():
    """Test that velocity has correct shape."""
    x_t = torch.randn(4, 3, 64, 64)
    t = torch.rand(4)
    
    velocity = compute_velocity(x_t, t)
    
    assert velocity.shape == x_t.shape

def test_compute_velocity_invalid_input():
    """Test that invalid input raises error."""
    x_t = torch.randn(4, 3, 64, 64)
    t = torch.rand(2)  # Wrong batch size
    
    with pytest.raises(ValueError):
        compute_velocity(x_t, t)
```

Run tests:

```bash
make test           # Run all tests
make test-fast      # Run tests in parallel
make test-coverage  # Generate coverage report
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:

```
feat(teachers): add adaptive quadratic teacher

Implement context-dependent curvature parameter that adapts
based on text embeddings.

Closes #42
```

```
fix(vae): correct pullback metric computation

The Jacobian computation was using wrong dimensions.
Now correctly computes J^T @ J.

Fixes #56
```

## 🔍 Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   make ci
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

5. **Update CHANGELOG.md** with your changes

### Submitting PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub

3. **Fill out PR template** completely

4. **Link related issues** using keywords (Fixes #123, Closes #456)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
- [ ] Dependent changes merged

## Screenshots (if applicable)

## Additional Notes
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one approval** from maintainers required
3. **Address review comments** promptly
4. **Squash commits** if requested
5. **Rebase on main** before merge

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
├── benchmarks/        # Performance benchmarks
└── fixtures/          # Shared test fixtures
```

### Writing Good Tests

1. **Test one thing**: Each test should verify one behavior
2. **Use descriptive names**: `test_velocity_respects_endpoints`
3. **Arrange-Act-Assert**: Clear test structure
4. **Use fixtures**: Share setup code with pytest fixtures
5. **Parametrize**: Test multiple inputs with `@pytest.mark.parametrize`

Example:

```python
@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    return {
        'x_t': torch.randn(4, 3, 64, 64),
        't': torch.rand(4),
    }

@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_sampling_different_steps(model, sample_data, num_steps):
    """Test that sampling works with different step counts."""
    # Arrange
    batch_size = sample_data['x_t'].shape[0]
    
    # Act
    samples = model.sample(batch_size, num_steps=num_steps)
    
    # Assert
    assert samples.shape[0] == batch_size
```

## 📚 Documentation Guidelines

### Code Documentation

- **Module docstrings**: Describe module purpose
- **Class docstrings**: Explain class responsibility
- **Function docstrings**: Document args, returns, raises
- **Inline comments**: Explain complex logic

### User Documentation

Located in `docs/`:

```
docs/
├── getting_started.md
├── tutorials/
├── api/
└── examples/
```

Build docs:

```bash
make docs
make docs-serve  # View at http://localhost:8080
```

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try latest version** to see if bug is fixed
3. **Minimal reproduction** to isolate the issue

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. ...
2. ...
3. ...

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 12.1]
- GPU: [e.g., NVIDIA A100]

## Additional Context
- Error messages
- Stack traces
- Screenshots
- Relevant config files
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing issues** for similar requests
2. **Consider scope**: Does it fit the project goals?
3. **Think about implementation**: Is it feasible?

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should this work?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
- Use cases
- Examples from other projects
- Mockups or diagrams
```

## 🎯 Development Priorities

Current focus areas:

1. **Core functionality**: Teachers, training, evaluation
2. **Performance**: Optimization, memory efficiency
3. **Documentation**: Tutorials, examples, API docs
4. **Testing**: Increase coverage, add integration tests
5. **Usability**: Better error messages, logging

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Discord**: [Join our server](https://discord.gg/XXXXXXX)
- **Email**: todd.zhou@example.com

## 🏆 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release notes**: Significant contributions highlighted
- **Paper acknowledgments**: Major contributions acknowledged

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, helps make NRF better. Thank you for being part of this project!

---

**Questions?** Feel free to ask in [GitHub Discussions](https://github.com/Todd7777/KURE-Project/discussions) or reach out directly.
