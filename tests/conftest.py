"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_batch():
    """Generate a sample batch of data for testing."""
    batch_size = 4
    channels = 4
    height = 64
    width = 64
    
    return {
        "images": torch.randn(batch_size, channels, height, width),
        "latents": torch.randn(batch_size, channels, height, width),
        "time": torch.rand(batch_size),
        "context": torch.randn(batch_size, 768),
    }


@pytest.fixture
def small_batch():
    """Generate a smaller batch for faster tests."""
    batch_size = 2
    channels = 4
    height = 32
    width = 32
    
    return {
        "images": torch.randn(batch_size, channels, height, width),
        "latents": torch.randn(batch_size, channels, height, width),
        "time": torch.rand(batch_size),
        "context": torch.randn(batch_size, 768),
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
