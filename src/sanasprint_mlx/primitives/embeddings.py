from __future__ import annotations

import math

import mlx.core as mx


def sinusoidal_embedding(values, dim: int, max_period: int = 10_000):
    if dim <= 0:
        raise ValueError("dim must be positive")
    values = mx.array(values)
    half = dim // 2
    if half == 0:
        return mx.zeros((*values.shape, dim))

    frequencies = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = values[..., None].astype(mx.float32) * frequencies
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros((*values.shape, 1))], axis=-1)
    return embedding


def guidance_embedding(guidance, dim: int, max_period: int = 10_000):
    return sinusoidal_embedding(guidance, dim=dim, max_period=max_period)
