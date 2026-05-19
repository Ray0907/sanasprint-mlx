from __future__ import annotations

import mlx.core as mx


def rms_norm(x, weight, eps: float = 1e-6):
    x = mx.array(x)
    weight = mx.array(weight)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * weight


def modulated_rms_norm(x, weight, shift, scale, eps: float = 1e-6):
    normalized = rms_norm(x, weight, eps=eps)
    return normalized * (1 + mx.array(scale)) + mx.array(shift)
