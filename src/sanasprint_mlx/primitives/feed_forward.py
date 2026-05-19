from __future__ import annotations

import mlx.core as mx


def linear(x, weight, bias=None):
    y = mx.matmul(mx.array(x), mx.array(weight).T)
    if bias is not None:
        y = y + mx.array(bias)
    return y


def silu(x):
    x = mx.array(x)
    return x * mx.sigmoid(x)


def geglu(x, gate):
    return mx.array(x) * silu(gate)


def glumbconv_like(x, gate, up_weight, down_weight, up_bias=None, down_bias=None):
    hidden = linear(x, up_weight, up_bias)
    gated = geglu(hidden, gate)
    return linear(gated, down_weight, down_bias)
