import numpy as np

from sanasprint_mlx.primitives.feed_forward import geglu, glumbconv_like, linear, silu


def test_linear_matches_numpy_reference():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    weight = np.array([[1.0, 0.5], [-1.0, 2.0]], dtype=np.float32)
    bias = np.array([0.25, -0.5], dtype=np.float32)

    result = np.array(linear(x, weight, bias))
    expected = x @ weight.T + bias

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_silu_matches_numpy_reference():
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)

    result = np.array(silu(x))
    expected = x / (1 + np.exp(-x))

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_geglu_matches_numpy_reference():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gate = np.array([[0.5, -1.0]], dtype=np.float32)

    result = np.array(geglu(x, gate))
    expected = x * (gate / (1 + np.exp(-gate)))

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_glumbconv_like_matches_numpy_reference():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gate = np.array([[0.5, -1.0]], dtype=np.float32)
    up_weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    down_weight = np.array([[0.5, 1.0]], dtype=np.float32)

    result = np.array(glumbconv_like(x, gate, up_weight, down_weight))
    hidden = (x @ up_weight.T) * (gate / (1 + np.exp(-gate)))
    expected = hidden @ down_weight.T

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)
