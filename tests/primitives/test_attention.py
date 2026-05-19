import numpy as np

from sanasprint_mlx.primitives.attention import sana_linear_attention, scaled_dot_product_attention


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def test_attention_matches_numpy_reference():
    q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    k = q.copy()
    v = np.array([[[[2.0, 0.0], [0.0, 4.0]]]], dtype=np.float32)

    result = np.array(scaled_dot_product_attention(q, k, v))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(2.0)
    expected = np.matmul(softmax(scores), v)

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_attention_mask_blocks_tokens():
    q = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[2.0, 0.0], [0.0, 4.0]]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int32)

    result = np.array(scaled_dot_product_attention(q, k, v, mask=mask))

    np.testing.assert_allclose(result, np.array([[[[2.0, 0.0]]]], dtype=np.float32), atol=1e-4, rtol=0)


def test_sana_linear_attention_matches_numpy_reference():
    q = np.array([[[[1.0, 2.0], [0.5, 1.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.5], [2.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[3.0, 1.0], [2.0, 4.0]]]], dtype=np.float32)

    result = np.array(sana_linear_attention(q, k, v, eps=1e-6))

    q_phi = np.maximum(q, 0) + 1
    k_phi = np.maximum(k, 0) + 1
    kv = np.matmul(np.swapaxes(k_phi, -1, -2), v)
    normalizer = 1.0 / (np.matmul(q_phi, np.sum(k_phi, axis=-2, keepdims=True).swapaxes(-1, -2)) + 1e-6)
    expected = np.matmul(q_phi, kv) * normalizer

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)


def test_sana_linear_attention_preserves_shape():
    q = np.ones((2, 3, 4, 5), dtype=np.float32)
    k = np.ones((2, 3, 4, 5), dtype=np.float32)
    v = np.ones((2, 3, 4, 6), dtype=np.float32)

    assert sana_linear_attention(q, k, v).shape == (2, 3, 4, 6)


def test_sana_linear_attention_mask_blocks_tokens_when_tokens_and_dim_differ():
    q = np.array([[[[1.0, 2.0], [0.5, 1.0], [2.0, 1.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.5], [2.0, 1.0], [4.0, 2.0]]]], dtype=np.float32)
    v = np.array([[[[3.0, 1.0, 0.5, 2.0], [2.0, 4.0, 1.0, 3.0], [100.0, 100.0, 100.0, 100.0]]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int32)

    result = np.array(sana_linear_attention(q, k, v, mask=mask, eps=1e-6))

    q_phi = np.maximum(q, 0) + 1
    k_phi = np.maximum(k, 0) + 1
    keep = mask[:, None, :, None].astype(np.float32)
    k_phi = k_phi * keep
    v = v * keep
    kv = np.matmul(np.swapaxes(k_phi, -1, -2), v)
    normalizer = 1.0 / (np.matmul(q_phi, np.sum(k_phi, axis=-2, keepdims=True).swapaxes(-1, -2)) + 1e-6)
    expected = np.matmul(q_phi, kv) * normalizer

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=0)
