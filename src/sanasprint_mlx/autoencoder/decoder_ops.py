from __future__ import annotations

import mlx.core as mx


def conv2d_nchw(x, weight, bias=None, *, padding: int = 0, stride: int = 1, groups: int = 1):
    x = mx.array(x).transpose(0, 2, 3, 1)
    weight = mx.array(weight).transpose(0, 2, 3, 1)
    output = mx.conv2d(x, weight, stride=stride, padding=padding, groups=groups)
    if bias is not None:
        output = output + mx.array(bias)
    return output.transpose(0, 3, 1, 2)


def rms_norm_nchw(x, weight, bias=None, *, eps: float = 1e-5):
    x = mx.array(x).transpose(0, 2, 3, 1)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    output = x * mx.rsqrt(variance + eps) * mx.array(weight)
    if bias is not None:
        output = output + mx.array(bias)
    return output.transpose(0, 3, 1, 2)


def res_block(x, *, conv1_weight, conv1_bias, conv2_weight, norm_weight, norm_bias):
    residual = mx.array(x)
    hidden = conv2d_nchw(residual, conv1_weight, conv1_bias, padding=1)
    hidden = silu(hidden)
    hidden = conv2d_nchw(hidden, conv2_weight, padding=1)
    hidden = rms_norm_nchw(hidden, norm_weight, norm_bias, eps=1e-5)
    return hidden + residual


def dc_up_block_interpolate(x, *, conv_weight, conv_bias):
    x = mx.array(x)
    hidden = nearest_upsample_2x(x)
    hidden = conv2d_nchw(hidden, conv_weight, conv_bias, padding=1)
    out_channels = mx.array(conv_weight).shape[0]
    repeats = out_channels * 4 // x.shape[1]
    shortcut = mx.repeat(x, repeats, axis=1)
    shortcut = pixel_shuffle_nchw(shortcut, 2)
    return hidden + shortcut


def nearest_upsample_2x(x):
    x = mx.array(x)
    x = mx.repeat(x, 2, axis=2)
    return mx.repeat(x, 2, axis=3)


def pixel_shuffle_nchw(x, factor: int):
    x = mx.array(x)
    batch, channels, height, width = x.shape
    out_channels = channels // (factor * factor)
    x = x.reshape(batch, out_channels, factor, factor, height, width)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(batch, out_channels, height * factor, width * factor)


def silu(x):
    x = mx.array(x)
    return x * mx.sigmoid(x)
