import numpy as np

from sanasprint_mlx.primitives.embeddings import guidance_embedding, sinusoidal_embedding


def test_sinusoidal_embedding_is_deterministic():
    timesteps = np.array([0.0, 1.0], dtype=np.float32)

    first = np.array(sinusoidal_embedding(timesteps, dim=6))
    second = np.array(sinusoidal_embedding(timesteps, dim=6))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 6)


def test_guidance_embedding_shape():
    guidance = np.array([4.5, 7.0], dtype=np.float32)

    result = guidance_embedding(guidance, dim=8)

    assert result.shape == (2, 8)
