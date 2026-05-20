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


def glumb_conv(
    x,
    *,
    conv_inverted_weight,
    conv_inverted_bias,
    conv_depth_weight,
    conv_depth_bias,
    conv_point_weight,
    norm_weight,
    norm_bias,
    residual_connection: bool = True,
):
    residual = mx.array(x)
    hidden = conv2d_nchw(residual, conv_inverted_weight, conv_inverted_bias)
    hidden = silu(hidden)
    hidden = conv2d_nchw(
        hidden,
        conv_depth_weight,
        conv_depth_bias,
        padding=1,
        groups=mx.array(conv_depth_weight).shape[0],
    )
    split = hidden.shape[1] // 2
    hidden, gate = hidden[:, :split], hidden[:, split:]
    hidden = hidden * silu(gate)
    hidden = conv2d_nchw(hidden, conv_point_weight)
    hidden = rms_norm_nchw(hidden, norm_weight, norm_bias, eps=1e-5)
    if residual_connection:
        hidden = hidden + residual
    return hidden


def sana_multiscale_linear_attention(
    x,
    *,
    to_q_weight,
    to_k_weight,
    to_v_weight,
    multiscale_weights,
    to_out_weight,
    norm_weight,
    norm_bias,
    attention_head_dim: int,
    norm_type: str = "rms_norm",
    residual_connection: bool = True,
    eps: float = 1e-15,
):
    residual = mx.array(x)
    batch, _, height, width = residual.shape
    hidden_nhwc = residual.transpose(0, 2, 3, 1)
    query = linear_last_dim(hidden_nhwc, to_q_weight)
    key = linear_last_dim(hidden_nhwc, to_k_weight)
    value = linear_last_dim(hidden_nhwc, to_v_weight)
    hidden = mx.concatenate([query, key, value], axis=-1).transpose(0, 3, 1, 2)

    multi_scale = [hidden]
    for item in multiscale_weights:
        projected = conv2d_nchw(
            hidden,
            item["proj_in_weight"],
            padding=mx.array(item["proj_in_weight"]).shape[2] // 2,
            groups=hidden.shape[1],
        )
        projected = conv2d_nchw(
            projected,
            item["proj_out_weight"],
            groups=3 * (mx.array(to_q_weight).shape[0] // attention_head_dim),
        )
        multi_scale.append(projected)

    hidden = mx.concatenate(multi_scale, axis=1)
    tokens = height * width
    use_linear_attention = tokens > attention_head_dim
    if use_linear_attention:
        hidden = hidden.astype(mx.float32)
    hidden = hidden.reshape(batch, -1, 3 * attention_head_dim, tokens)
    query, key, value = mx.split(hidden, 3, axis=2)
    query = relu(query)
    key = relu(key)
    if use_linear_attention:
        hidden = _linear_attention(query, key, value, eps=eps)
    else:
        hidden = _quadratic_attention(query, key, value, eps=eps)
    hidden = hidden.reshape(batch, -1, height, width)
    hidden = linear_last_dim(hidden.transpose(0, 2, 3, 1), to_out_weight).transpose(0, 3, 1, 2)
    if norm_type == "rms_norm":
        hidden = rms_norm_nchw(hidden, norm_weight, norm_bias, eps=1e-5)
    else:
        raise ValueError(f"unsupported norm_type: {norm_type}")
    if residual_connection:
        hidden = hidden + residual
    return hidden


def linear_last_dim(x, weight, bias=None):
    output = mx.matmul(mx.array(x), mx.array(weight).T)
    if bias is not None:
        output = output + mx.array(bias)
    return output


def relu(x):
    return mx.maximum(mx.array(x), 0)


def _linear_attention(query, key, value, *, eps: float):
    ones = mx.ones((*value.shape[:2], 1, value.shape[3]), dtype=value.dtype)
    value = mx.concatenate([value, ones], axis=2)
    scores = mx.matmul(value, key.transpose(0, 1, 3, 2))
    hidden = mx.matmul(scores, query).astype(mx.float32)
    return hidden[:, :, :-1] / (hidden[:, :, -1:] + eps)


def _quadratic_attention(query, key, value, *, eps: float):
    scores = mx.matmul(key.transpose(0, 1, 3, 2), query).astype(mx.float32)
    scores = scores / (mx.sum(scores, axis=2, keepdims=True) + eps)
    return mx.matmul(value, scores.astype(value.dtype))


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
