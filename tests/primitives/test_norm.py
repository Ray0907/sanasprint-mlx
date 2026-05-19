import numpy as np

from sanasprint_mlx.primitives.norm import modulated_rms_norm, rms_norm


def reference_rms_norm(x, weight, eps=1e-6):
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def test_rms_norm_matches_numpy_reference():
    x = np.array([[[1.0, 2.0, 3.0], [2.0, 0.5, -1.0]]], dtype=np.float32)
    weight = np.array([1.0, 0.5, 2.0], dtype=np.float32)

    result = np.array(rms_norm(x, weight, eps=1e-6))
    expected = reference_rms_norm(x, weight)

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_modulated_rms_norm_matches_numpy_reference():
    x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    weight = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    shift = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    scale = np.array([0.5, 0.0, -0.25], dtype=np.float32)

    result = np.array(modulated_rms_norm(x, weight, shift, scale, eps=1e-6))
    expected = reference_rms_norm(x, weight) * (1 + scale) + shift

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_rms_norm_preserves_shape():
    x = np.ones((2, 3, 4), dtype=np.float32)
    weight = np.ones((4,), dtype=np.float32)

    assert rms_norm(x, weight).shape == x.shape
