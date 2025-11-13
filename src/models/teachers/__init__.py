"""
Teacher models for Nonlinear Rectified Flows.
"""

from .linear import LinearTeacher
from .quadratic import QuadraticTeacher, AdaptiveQuadraticTeacher
from .cubic_spline import CubicSplineTeacher, CubicSplineController
from .schrodinger_bridge import (
    SchrodingerBridgeTeacher,
    SchrodingerBridgeDriftNet,
    NystromSinkhornSolver,
)

__all__ = [
    "LinearTeacher",
    "QuadraticTeacher",
    "AdaptiveQuadraticTeacher",
    "CubicSplineTeacher",
    "CubicSplineController",
    "SchrodingerBridgeTeacher",
    "SchrodingerBridgeDriftNet",
    "NystromSinkhornSolver",
]


