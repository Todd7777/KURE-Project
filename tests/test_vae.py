"""Unit tests for VAE with pullback metrics."""

import pytest
import torch

from src.models.vae import PullbackMetricVAE, VAEDecoder, VAEEncoder, create_vae


class TestVAEEncoder:
    """Test VAE encoder."""

    @pytest.fixture
    def encoder(self):
        return VAEEncoder(in_channels=3, latent_channels=4, base_channels=64)

    def test_forward_shape(self, encoder):
        """Test encoder output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 256, 256)

        mean, logvar = encoder(x)

        # Output should be downsampled
        assert mean.shape[0] == batch_size
        assert mean.shape[1] == 4  # latent_channels
        assert logvar.shape == mean.shape

    def test_output_finite(self, encoder):
        """Test that encoder produces finite outputs."""
        x = torch.randn(2, 3, 256, 256)

        mean, logvar = encoder(x)

        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(logvar))


class TestVAEDecoder:
    """Test VAE decoder."""

    @pytest.fixture
    def decoder(self):
        return VAEDecoder(latent_channels=4, out_channels=3, base_channels=64)

    def test_forward_shape(self, decoder):
        """Test decoder output shape."""
        batch_size = 4
        z = torch.randn(batch_size, 4, 64, 64)

        x = decoder(z)

        # Output should be upsampled
        assert x.shape[0] == batch_size
        assert x.shape[1] == 3  # out_channels
        assert x.shape[2] > z.shape[2]  # upsampled

    def test_output_finite(self, decoder):
        """Test that decoder produces finite outputs."""
        z = torch.randn(2, 4, 64, 64)

        x = decoder(z)

        assert torch.all(torch.isfinite(x))


class TestPullbackMetricVAE:
    """Test VAE with pullback metrics."""

    @pytest.fixture
    def vae(self):
        encoder = VAEEncoder(in_channels=3, latent_channels=4)
        decoder = VAEDecoder(latent_channels=4, out_channels=3)
        return PullbackMetricVAE(encoder, decoder, kl_weight=1e-6)

    @pytest.fixture
    def sample_images(self):
        return torch.randn(2, 3, 256, 256)

    def test_forward_shape(self, vae, sample_images):
        """Test VAE forward pass shape."""
        recon, mean, logvar = vae(sample_images)

        assert recon.shape == sample_images.shape
        assert mean.shape[0] == sample_images.shape[0]
        assert logvar.shape == mean.shape

    def test_encode_decode_cycle(self, vae, sample_images):
        """Test encode-decode cycle."""
        mean, logvar = vae.encode(sample_images)
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z)

        assert recon.shape == sample_images.shape

    def test_reparameterization_trick(self, vae):
        """Test reparameterization produces different samples."""
        mean = torch.zeros(2, 4, 64, 64)
        logvar = torch.zeros(2, 4, 64, 64)

        z1 = vae.reparameterize(mean, logvar)
        z2 = vae.reparameterize(mean, logvar)

        # Should produce different samples
        assert not torch.allclose(z1, z2, atol=1e-3)

    def test_compute_loss(self, vae, sample_images):
        """Test loss computation."""
        recon, mean, logvar = vae(sample_images)
        loss, metrics = vae.compute_loss(sample_images, recon, mean, logvar)

        # Loss should be positive scalar
        assert loss.dim() == 0
        assert loss > 0

        # Metrics should contain expected keys
        assert "vae_loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics

    def test_pullback_metric_shape(self, vae):
        """Test pullback metric computation shape."""
        z = torch.randn(2, 4, 64, 64, requires_grad=True)

        # Approximate metric (faster)
        G_approx = vae.compute_pullback_metric(z, approximate=True)

        batch_size = z.shape[0]
        latent_dim = z.numel() // batch_size

        assert G_approx.shape == (batch_size, latent_dim, latent_dim)

    def test_geodesic_distance_positive(self, vae):
        """Test that geodesic distance is positive."""
        z1 = torch.randn(2, 4, 64, 64)
        z2 = torch.randn(2, 4, 64, 64)

        distance = vae.geodesic_distance(z1, z2, num_steps=5)

        assert torch.all(distance >= 0)

    def test_geodesic_distance_zero_for_same_point(self, vae):
        """Test that geodesic distance is zero for same point."""
        z = torch.randn(2, 4, 64, 64)

        distance = vae.geodesic_distance(z, z, num_steps=5)

        assert torch.allclose(distance, torch.zeros_like(distance), atol=1e-2)

    def test_path_length_positive(self, vae):
        """Test that path length is positive."""
        num_points = 5
        trajectory = torch.randn(num_points, 2, 4, 64, 64)

        length = vae.path_length(trajectory)

        assert torch.all(length >= 0)

    def test_curvature_positive(self, vae):
        """Test that curvature is positive."""
        num_points = 5
        trajectory = torch.randn(num_points, 2, 4, 64, 64)

        curvature = vae.curvature(trajectory)

        assert torch.all(curvature >= 0)

    def test_gradient_flow(self, vae, sample_images):
        """Test that gradients flow through VAE."""
        sample_images.requires_grad_(True)

        recon, mean, logvar = vae(sample_images)
        loss, _ = vae.compute_loss(sample_images, recon, mean, logvar)
        loss.backward()

        # Gradients should exist
        assert sample_images.grad is not None

        # Model parameters should have gradients
        for param in vae.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCreateVAE:
    """Test VAE factory function."""

    @pytest.mark.parametrize(
        "image_size,latent_size",
        [(256, 64), (512, 128), (128, 32)],
    )
    def test_create_vae_different_sizes(self, image_size, latent_size):
        """Test creating VAE with different sizes."""
        vae = create_vae(
            image_size=image_size,
            latent_size=latent_size,
            in_channels=3,
            latent_channels=4,
        )

        # Test with appropriate input size
        x = torch.randn(2, 3, image_size, image_size)
        recon, mean, logvar = vae(x)

        assert recon.shape == x.shape

    def test_create_vae_default_params(self):
        """Test creating VAE with default parameters."""
        vae = create_vae()

        assert isinstance(vae, PullbackMetricVAE)
        assert vae.encoder is not None
        assert vae.decoder is not None


def test_vae_device_consistency():
    """Test VAE device consistency."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = create_vae().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)

    recon, mean, logvar = vae(x)

    assert recon.device.type == device
    assert mean.device.type == device
    assert logvar.device.type == device


def test_vae_deterministic_with_seed():
    """Test VAE determinism with seed."""
    torch.manual_seed(42)

    vae = create_vae()
    x = torch.randn(2, 3, 256, 256)

    mean1, logvar1 = vae.encode(x)

    torch.manual_seed(42)

    vae2 = create_vae()
    mean2, logvar2 = vae2.encode(x)

    # Encodings should be identical with same seed
    assert torch.allclose(mean1, mean2, atol=1e-6)
    assert torch.allclose(logvar1, logvar2, atol=1e-6)
