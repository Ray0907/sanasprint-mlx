import numpy as np

from sanasprint_mlx.transformer.block import ToyTransformerBlock


def test_toy_block_matches_numpy_reference():
    block = ToyTransformerBlock(hidden_size=2)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    encoder = np.array([[[0.5, 1.0], [1.5, 2.0]]], dtype=np.float32)
    mask = np.array([[1, 1]], dtype=np.int32)

    result = np.array(block(x, encoder, mask))
    expected = x + np.mean(encoder, axis=1, keepdims=True)

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_toy_block_applies_encoder_mask():
    block = ToyTransformerBlock(hidden_size=2)
    x = np.zeros((1, 1, 2), dtype=np.float32)
    encoder = np.array([[[1.0, 1.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int32)

    result = np.array(block(x, encoder, mask))

    np.testing.assert_allclose(result, np.array([[[1.0, 1.0]]], dtype=np.float32), atol=1e-4, rtol=0)
