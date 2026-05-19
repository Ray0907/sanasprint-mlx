from __future__ import annotations

import math

import mlx.core as mx


def scaled_dot_product_attention(q, k, v, mask=None):
    q = mx.array(q)
    k = mx.array(k)
    v = mx.array(v)
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        keep = mx.array(mask).astype(mx.bool_)
        while len(keep.shape) < len(scores.shape):
            keep = mx.expand_dims(keep, axis=1)
        scores = mx.where(keep, scores, mx.array(-1e9, dtype=scores.dtype))
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v)


def sana_linear_attention(q, k, v, mask=None, eps: float = 1e-6):
    q = mx.array(q)
    k = mx.array(k)
    v = mx.array(v)
    q_phi = mx.maximum(q, 0) + 1
    k_phi = mx.maximum(k, 0) + 1

    if mask is not None:
        keep = _token_mask(mask, like=k_phi).astype(k_phi.dtype)
        k_phi = k_phi * keep
        v = v * keep

    kv = mx.matmul(mx.swapaxes(k_phi, -1, -2), v)
    k_sum = mx.sum(k_phi, axis=-2, keepdims=True)
    normalizer = 1.0 / (mx.matmul(q_phi, mx.swapaxes(k_sum, -1, -2)) + eps)
    return mx.matmul(q_phi, kv) * normalizer


def _token_mask(mask, *, like):
    keep = mx.array(mask)
    if len(keep.shape) == 2:
        return keep[:, None, :, None]
    while len(keep.shape) < len(like.shape):
        keep = mx.expand_dims(keep, axis=-1)
    return keep
