"""Unit tests for teacher models."""

import pytest
import torch

from src.models.teachers.linear import LinearTeacher
from src.models.teachers.quadratic import AdaptiveQuadraticTeacher, QuadraticTeacher
from src.models.teachers.cubic_spline import CubicSplineController, CubicSplineTeacher


class TestLinearTeacher:
    """Test linear teacher implementation."""

    @pytest.fixture
    def teacher(self):
        return LinearTeacher()

    @pytest.fixture
    def sample_data(self):
        batch_size = 4
        channels = 4
        size = 64
        z_0 = torch.randn(batch_size, channels, size, size)
        x_1 = torch.randn(batch_size, channels, size, size)
        t = torch.rand(batch_size)
        return z_0, x_1, t

    def test_interpolate_shape(self, teacher, sample_data):
        """Test that interpolation produces correct shape."""
        z_0, x_1, t = sample_data
        x_t = teacher.interpolate(z_0, x_1, t)
        assert x_t.shape == z_0.shape

    def test_interpolate_endpoints(self, teacher, sample_data):
        """Test that interpolation respects endpoints."""
        z_0, x_1, _ = sample_data
        batch_size = z_0.shape[0]

        # At t=0, should be z_0
        t_0 = torch.zeros(batch_size)
        x_0 = teacher.interpolate(z_0, x_1, t_0)
        assert torch.allclose(x_0, z_0, atol=1e-5)

        # At t=1, should be x_1
        t_1 = torch.ones(batch_size)
        x_1_pred = teacher.interpolate(z_0, x_1, t_1)
        assert torch.allclose(x_1_pred, x_1, atol=1e-5)

    def test_velocity_constant(self, teacher, sample_data):
        """Test that velocity is constant for linear teacher."""
        z_0, x_1, t = sample_data

        v_t1 = teacher.velocity(z_0, x_1, t)
        v_t2 = teacher.velocity(z_0, x_1, torch.rand_like(t))

        # Velocity should be the same regardless of t
        assert torch.allclose(v_t1, v_t2, atol=1e-5)

    def test_velocity_equals_displacement(self, teacher, sample_data):
        """Test that velocity equals displacement."""
        z_0, x_1, t = sample_data
        v_t = teacher.velocity(z_0, x_1, t)
        expected = x_1 - z_0
        assert torch.allclose(v_t, expected, atol=1e-5)

    def test_regularization_zero(self, teacher, sample_data):
        """Test that regularization is zero for linear teacher."""
        z_0, x_1, _ = sample_data
        reg = teacher.regularization_loss(z_0, x_1)
        assert reg == 0.0


class TestQuadraticTeacher:
    """Test quadratic teacher implementation."""

    @pytest.fixture
    def teacher(self):
        return QuadraticTeacher(alpha=0.5, learnable=False)

    @pytest.fixture
    def sample_data(self):
        batch_size = 4
        channels = 4
        size = 64
        z_0 = torch.randn(batch_size, channels, size, size)
        x_1 = torch.randn(batch_size, channels, size, size)
        t = torch.rand(batch_size)
        return z_0, x_1, t

    def test_interpolate_endpoints(self, teacher, sample_data):
        """Test that quadratic interpolation respects endpoints."""
        z_0, x_1, _ = sample_data
        batch_size = z_0.shape[0]

        # At t=0, should be z_0
        t_0 = torch.zeros(batch_size)
        x_0 = teacher.interpolate(z_0, x_1, t_0)
        assert torch.allclose(x_0, z_0, atol=1e-5)

        # At t=1, should be x_1
        t_1 = torch.ones(batch_size)
        x_1_pred = teacher.interpolate(z_0, x_1, t_1)
        assert torch.allclose(x_1_pred, x_1, atol=1e-5)

    def test_velocity_time_dependent(self, teacher, sample_data):
        """Test that velocity is time-dependent for quadratic teacher."""
        z_0, x_1, _ = sample_data

        t1 = torch.tensor([0.25, 0.25, 0.25, 0.25])
        t2 = torch.tensor([0.75, 0.75, 0.75, 0.75])

        v_t1 = teacher.velocity(z_0, x_1, t1)
        v_t2 = teacher.velocity(z_0, x_1, t2)

        # Velocities should be different
        assert not torch.allclose(v_t1, v_t2, atol=1e-3)

    def test_alpha_zero_equals_linear(self, sample_data):
        """Test that alpha=0 reduces to linear interpolation."""
        z_0, x_1, t = sample_data

        linear_teacher = LinearTeacher()
        quad_teacher = QuadraticTeacher(alpha=0.0)

        x_t_linear = linear_teacher.interpolate(z_0, x_1, t)
        x_t_quad = quad_teacher.interpolate(z_0, x_1, t)

        assert torch.allclose(x_t_linear, x_t_quad, atol=1e-5)

    def test_learnable_alpha(self):
        """Test that learnable alpha has requires_grad=True."""
        teacher = QuadraticTeacher(alpha=0.5, learnable=True)
        assert teacher.alpha.requires_grad

    def test_regularization_nonzero(self, teacher, sample_data):
        """Test that regularization is non-zero."""
        z_0, x_1, _ = sample_data
        reg = teacher.regularization_loss(z_0, x_1)
        assert reg > 0.0


class TestAdaptiveQuadraticTeacher:
    """Test adaptive quadratic teacher implementation."""

    @pytest.fixture
    def teacher(self):
        return AdaptiveQuadraticTeacher(context_dim=768, alpha_min=-0.5, alpha_max=1.0)

    @pytest.fixture
    def sample_data(self):
        batch_size = 4
        channels = 4
        size = 64
        z_0 = torch.randn(batch_size, channels, size, size)
        x_1 = torch.randn(batch_size, channels, size, size)
        t = torch.rand(batch_size)
        context = torch.randn(batch_size, 768)
        return z_0, x_1, t, context

    def test_context_dependent_alpha(self, teacher, sample_data):
        """Test that alpha depends on context."""
        z_0, x_1, t, context = sample_data

        # Different contexts should produce different alphas
        context1 = torch.randn(4, 768)
        context2 = torch.randn(4, 768)

        x_t1 = teacher.interpolate(z_0, x_1, t, context1)
        x_t2 = teacher.interpolate(z_0, x_1, t, context2)

        # Results should be different
        assert not torch.allclose(x_t1, x_t2, atol=1e-3)

    def test_alpha_in_range(self, teacher, sample_data):
        """Test that predicted alpha is in valid range."""
        z_0, x_1, t, context = sample_data

        alpha = teacher.get_alpha(t, context)

        # Alpha should be in [alpha_min, alpha_max]
        assert torch.all(alpha >= teacher.alpha_min)
        assert torch.all(alpha <= teacher.alpha_max)


class TestCubicSplineTeacher:
    """Test cubic spline teacher implementation."""

    @pytest.fixture
    def controller(self):
        return CubicSplineController(
            context_dim=768, latent_channels=4, latent_size=64, num_control_points=3
        )

    @pytest.fixture
    def teacher(self, controller):
        return CubicSplineTeacher(controller=controller)

    @pytest.fixture
    def sample_data(self):
        batch_size = 2  # Smaller batch for spline tests
        channels = 4
        size = 64
        z_0 = torch.randn(batch_size, channels, size, size)
        x_1 = torch.randn(batch_size, channels, size, size)
        t = torch.rand(batch_size)
        context = torch.randn(batch_size, 768)
        return z_0, x_1, t, context

    def test_controller_output_shape(self, controller, sample_data):
        """Test that controller produces correct number of control points."""
        z_0, x_1, _, context = sample_data
        control_points = controller(z_0, x_1, context)

        assert len(control_points) == 3
        for cp in control_points:
            assert cp.shape == z_0.shape

    def test_interpolate_endpoints(self, teacher, sample_data):
        """Test that spline interpolation respects endpoints."""
        z_0, x_1, _, context = sample_data
        batch_size = z_0.shape[0]

        # At t=0, should be close to z_0
        t_0 = torch.zeros(batch_size)
        x_0 = teacher.interpolate(z_0, x_1, t_0, context)
        assert torch.allclose(x_0, z_0, atol=1e-2)

        # At t=1, should be close to x_1
        t_1 = torch.ones(batch_size)
        x_1_pred = teacher.interpolate(z_0, x_1, t_1, context)
        assert torch.allclose(x_1_pred, x_1, atol=1e-2)

    def test_regularization_components(self, teacher, sample_data):
        """Test that regularization includes all components."""
        z_0, x_1, _, context = sample_data

        reg = teacher.regularization_loss(z_0, x_1, context)

        # Should be positive
        assert reg > 0.0

        # Should be a scalar
        assert reg.dim() == 0

    def test_fallback_without_context(self, teacher, sample_data):
        """Test that teacher falls back to linear without context."""
        z_0, x_1, t, _ = sample_data

        # Without context, should use linear interpolation
        x_t = teacher.interpolate(z_0, x_1, t, context=None)

        # Should produce valid output
        assert x_t.shape == z_0.shape


@pytest.mark.parametrize(
    "teacher_class,kwargs",
    [
        (LinearTeacher, {}),
        (QuadraticTeacher, {"alpha": 0.5}),
    ],
)
def test_teacher_consistency(teacher_class, kwargs):
    """Test that teachers produce consistent results."""
    teacher = teacher_class(**kwargs)

    batch_size = 4
    channels = 4
    size = 64
    z_0 = torch.randn(batch_size, channels, size, size)
    x_1 = torch.randn(batch_size, channels, size, size)
    t = torch.rand(batch_size)

    # Multiple calls should produce same result
    x_t1 = teacher.interpolate(z_0, x_1, t)
    x_t2 = teacher.interpolate(z_0, x_1, t)

    assert torch.allclose(x_t1, x_t2, atol=1e-6)


def test_teacher_gradient_flow():
    """Test that gradients flow through teachers."""
    teacher = QuadraticTeacher(alpha=0.5, learnable=True)

    batch_size = 4
    channels = 4
    size = 64
    z_0 = torch.randn(batch_size, channels, size, size, requires_grad=True)
    x_1 = torch.randn(batch_size, channels, size, size, requires_grad=True)
    t = torch.rand(batch_size)

    x_t = teacher.interpolate(z_0, x_1, t)
    loss = x_t.sum()
    loss.backward()

    # Gradients should exist
    assert z_0.grad is not None
    assert x_1.grad is not None
    assert teacher.alpha.grad is not None
