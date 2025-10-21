---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

<!-- A clear and concise description of what the bug is -->

## To Reproduce

Steps to reproduce the behavior:

1. 
2. 
3. 
4. 

### Minimal Code Example

```python
# Minimal code to reproduce the issue
import torch
from src.models.nrf_base import NonlinearRectifiedFlow

# Your code here
```

## Expected Behavior

<!-- A clear and concise description of what you expected to happen -->

## Actual Behavior

<!-- What actually happened -->

### Error Message

```
# Paste the full error message and stack trace here
```

## Environment

**System Information:**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 12.1, N/A if CPU]
- GPU: [e.g., NVIDIA A100 80GB, N/A if CPU]

**Package Versions:**

```bash
# Output of: pip list | grep -E "torch|numpy|scipy"
```

**Installation Method:**
- [ ] pip install from PyPI
- [ ] pip install from source
- [ ] Docker
- [ ] Other (please specify):

## Additional Context

<!-- Add any other context about the problem here -->

### Configuration File

```yaml
# If relevant, paste your config file here
```

### Logs

```
# Paste relevant logs here
```

### Screenshots

<!-- If applicable, add screenshots to help explain your problem -->

## Possible Solution

<!-- If you have suggestions on how to fix the bug, please describe them here -->

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have tested with the latest version of the code
- [ ] I have provided a minimal reproducible example
- [ ] I have included all relevant error messages and logs
