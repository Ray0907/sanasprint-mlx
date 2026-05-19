import numpy as np

from sanasprint_mlx.transformer.conditioning import conditioning_vector


def test_conditioning_combines_timestep_and_guidance():
    timestep = np.array([0.5], dtype=np.float32)
    guidance = np.array([4.5], dtype=np.float32)

    result = conditioning_vector(timestep, guidance, dim=8)

    assert result.shape == (1, 8)
    assert not np.allclose(np.array(result), 0)


def test_conditioning_preserves_batch_dimension():
    result = conditioning_vector(np.array([0.1, 0.2]), np.array([1.0, 2.0]), dim=6)

    assert result.shape == (2, 6)
