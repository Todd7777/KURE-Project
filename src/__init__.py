"""Nonlinear Rectified Flows for AI Image Generation.

This package implements geometry-aware nonlinear teachers for rectified flow
models, enabling faster and more compositionally faithful image generation.
"""

__version__ = "0.1.0"
__author__ = "Todd Y. Zhou"
__email__ = "todd.zhou@example.com"
__license__ = "MIT"

from src.models.nrf_base import NonlinearRectifiedFlow, TimeScheduler
from src.models.teachers.linear import LinearTeacher
from src.models.teachers.quadratic import AdaptiveQuadraticTeacher, QuadraticTeacher
from src.models.teachers.cubic_spline import CubicSplineController, CubicSplineTeacher
from src.models.vae import PullbackMetricVAE, create_vae

__all__ = [
    "NonlinearRectifiedFlow",
    "TimeScheduler",
    "LinearTeacher",
    "QuadraticTeacher",
    "AdaptiveQuadraticTeacher",
    "CubicSplineController",
    "CubicSplineTeacher",
    "PullbackMetricVAE",
    "create_vae",
]
