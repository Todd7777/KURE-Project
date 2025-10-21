"""Unit tests for NRF base framework."""

import pytest
import torch
import torch.nn as nn

from src.models.nrf_base import NonlinearRectifiedFlow, TimeScheduler
from src.models.teachers.linear import LinearTeacher


class DummyVelocityNet(nn.Module):
    """Dummy velocity network for testing."""

    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_t, t, context=None):
        return self.conv(x_t)


class TestTimeScheduler:
    """Test time scheduling strategies."""

    @pytest.mark.parametrize(
        "schedule_type", ["linear", "cosine", "sigmoid", "exponential"]
    )
    def test_schedule_types(self, schedule_type):
        """Test that all schedule types work."""
        scheduler = TimeScheduler(schedule_type=schedule_type)

        t = torch.linspace(0, 1, 10)
        a = scheduler.a(t)
        b = scheduler.b(t)

        # Should produce valid outputs
        assert a.shape == t.shape
        assert b.shape == t.shape
        assert torch.all(torch.isfinite(a))
        assert torch.all(torch.isfinite(b))

    def test_linear_schedule_constant(self):
        """Test that linear schedule produces constants."""
        scheduler = TimeScheduler("linear")

        t = torch.linspace(0, 1, 10)
        a = scheduler.a(t)
        b = scheduler.b(t)

        assert torch.allclose(a, torch.ones_like(t))
        assert torch.allclose(b, torch.ones_like(t))

    def test_invalid_schedule_raises(self):
        """Test that invalid schedule type raises error."""
        scheduler = TimeScheduler("invalid")

        with pytest.raises(ValueError):
            scheduler.a(torch.tensor([0.5]))


class TestNonlinearRectifiedFlow:
    """Test NRF model."""

    @pytest.fixture
    def model(self):
        velocity_net = DummyVelocityNet(channels=4)
        teacher = LinearTeacher()
        scheduler = TimeScheduler("linear")
        return NonlinearRectifiedFlow(velocity_net, teacher, scheduler)

    @pytest.fixture
    def sample_data(self):
        batch_size = 4
        channels = 4
        size = 64
        x_1 = torch.randn(batch_size, channels, size, size)
        return x_1

    def test_forward_shape(self, model, sample_data):
        """Test that forward pass produces correct shape."""
        x_t = sample_data
        t = torch.rand(x_t.shape[0])

        v_pred = model(x_t, t)

        assert v_pred.shape == x_t.shape

    def test_compute_loss(self, model, sample_data):
        """Test loss computation."""
        x_1 = sample_data

        loss, metrics = model.compute_loss(x_1)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss > 0

        # Metrics should contain expected keys
        assert "loss" in metrics
        assert "velocity_loss" in metrics
        assert "reg_loss" in metrics

    def test_sample_shape(self, model):
        """Test that sampling produces correct shape."""
        batch_size = 4
        shape = (4, 64, 64)

        samples = model.sample(
            batch_size=batch_size, shape=shape, num_steps=4, use_ema=False
        )

        assert samples.shape == (batch_size, *shape)

    @pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
    def test_sample_different_steps(self, model, num_steps):
        """Test sampling with different step counts."""
        batch_size = 2
        shape = (4, 64, 64)

        samples = model.sample(
            batch_size=batch_size, shape=shape, num_steps=num_steps, use_ema=False
        )

        assert samples.shape == (batch_size, *shape)

    @pytest.mark.parametrize("solver", ["euler", "heun", "rk4"])
    def test_sample_different_solvers(self, model, solver):
        """Test sampling with different ODE solvers."""
        batch_size = 2
        shape = (4, 64, 64)

        samples = model.sample(
            batch_size=batch_size,
            shape=shape,
            num_steps=4,
            solver=solver,
            use_ema=False,
        )

        assert samples.shape == (batch_size, *shape)

    def test_ema_creation(self):
        """Test that EMA model is created."""
        velocity_net = DummyVelocityNet()
        teacher = LinearTeacher()
        model = NonlinearRectifiedFlow(velocity_net, teacher, ema_decay=0.9999)

        assert model.ema_velocity_net is not None

    def test_ema_update(self):
        """Test that EMA updates work."""
        velocity_net = DummyVelocityNet()
        teacher = LinearTeacher()
        model = NonlinearRectifiedFlow(velocity_net, teacher, ema_decay=0.999)

        # Get initial EMA parameters
        ema_params_before = [
            p.clone() for p in model.ema_velocity_net.parameters()
        ]

        # Update main model
        x_1 = torch.randn(2, 4, 64, 64)
        loss, _ = model.compute_loss(x_1)
        loss.backward()

        # Update EMA
        model.update_ema()

        # EMA parameters should have changed
        ema_params_after = list(model.ema_velocity_net.parameters())

        for p_before, p_after in zip(ema_params_before, ema_params_after):
            assert not torch.allclose(p_before, p_after, atol=1e-6)

    def test_get_trajectory(self, model, sample_data):
        """Test trajectory extraction."""
        x_1 = sample_data
        num_points = 10

        trajectory = model.get_trajectory(x_1, num_points=num_points)

        assert trajectory.shape == (num_points, *x_1.shape)

        # First point should be different from last
        assert not torch.allclose(trajectory[0], trajectory[-1], atol=1e-3)

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through the model."""
        x_1 = sample_data.requires_grad_(True)

        loss, _ = model.compute_loss(x_1)
        loss.backward()

        # Gradients should exist
        assert x_1.grad is not None

        # Model parameters should have gradients
        for param in model.velocity_net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_loss_types(self, model, sample_data):
        """Test different loss types."""
        x_1 = sample_data

        # MSE loss
        loss_mse, _ = model.compute_loss(x_1, loss_type="mse")
        assert loss_mse > 0

        # Huber loss
        loss_huber, _ = model.compute_loss(x_1, loss_type="huber")
        assert loss_huber > 0

    def test_invalid_loss_type_raises(self, model, sample_data):
        """Test that invalid loss type raises error."""
        x_1 = sample_data

        with pytest.raises(ValueError):
            model.compute_loss(x_1, loss_type="invalid")

    def test_invalid_solver_raises(self, model):
        """Test that invalid solver raises error."""
        with pytest.raises(ValueError):
            model.sample(
                batch_size=2,
                shape=(4, 64, 64),
                num_steps=4,
                solver="invalid",
                use_ema=False,
            )


def test_model_device_consistency():
    """Test that model handles device placement correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    velocity_net = DummyVelocityNet().to(device)
    teacher = LinearTeacher()
    model = NonlinearRectifiedFlow(velocity_net, teacher).to(device)

    x_1 = torch.randn(2, 4, 64, 64).to(device)

    loss, _ = model.compute_loss(x_1)

    assert loss.device.type == device


def test_model_deterministic_with_seed():
    """Test that model is deterministic with fixed seed."""
    torch.manual_seed(42)

    velocity_net1 = DummyVelocityNet()
    teacher1 = LinearTeacher()
    model1 = NonlinearRectifiedFlow(velocity_net1, teacher1)

    x_1 = torch.randn(2, 4, 64, 64)
    loss1, _ = model1.compute_loss(x_1)

    torch.manual_seed(42)

    velocity_net2 = DummyVelocityNet()
    teacher2 = LinearTeacher()
    model2 = NonlinearRectifiedFlow(velocity_net2, teacher2)

    loss2, _ = model2.compute_loss(x_1)

    # Losses should be identical with same seed
    assert torch.allclose(loss1, loss2, atol=1e-6)
