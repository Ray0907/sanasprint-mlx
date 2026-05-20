from __future__ import annotations

import math

import mlx.core as mx

from sanasprint_mlx.primitives.feed_forward import linear
from sanasprint_mlx.primitives.norm import rms_norm


class ToyTransformerBlock:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def __call__(self, x, encoder_hidden_states, encoder_attention_mask=None):
        x = mx.array(x)
        encoder = mx.array(encoder_hidden_states)
        if encoder_attention_mask is None:
            context = mx.mean(encoder, axis=1, keepdims=True)
        else:
            mask = mx.array(encoder_attention_mask).astype(encoder.dtype)[:, :, None]
            denom = mx.maximum(mx.sum(mask, axis=1, keepdims=True), mx.array(1.0, dtype=encoder.dtype))
            context = mx.sum(encoder * mask, axis=1, keepdims=True) / denom
        return x + context


class RealSanaAttentionBlock:
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_cross_attention_heads: int,
        cross_attention_head_dim: int,
        block_index: int = 0,
    ):
        if num_attention_heads * attention_head_dim != hidden_size:
            raise ValueError("num_attention_heads * attention_head_dim must equal hidden_size")
        if num_cross_attention_heads * cross_attention_head_dim != hidden_size:
            raise ValueError("num_cross_attention_heads * cross_attention_head_dim must equal hidden_size")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_cross_attention_heads = num_cross_attention_heads
        self.cross_attention_head_dim = cross_attention_head_dim
        self.block_index = block_index
        self._parameters = {
            key: mx.array(shape_value)
            for key, shape_value in _initial_attention_parameters(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                num_cross_attention_heads=num_cross_attention_heads,
                cross_attention_head_dim=cross_attention_head_dim,
                block_index=block_index,
            ).items()
        }

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(value.shape) for key, value in self._parameters.items()}

    def parameters(self) -> dict[str, object]:
        return {key: mx.array(value) for key, value in self._parameters.items()}

    def load_parameters(self, parameters: dict, *, strict: bool = True) -> None:
        expected = self.parameter_shapes()
        unknown = [key for key in parameters if key not in expected]
        if unknown:
            raise KeyError(unknown[0])
        if strict:
            missing = [key for key in expected if key not in parameters]
            if missing:
                raise KeyError(missing[0])
        for key, value in parameters.items():
            tensor = mx.array(value)
            if tuple(tensor.shape) != expected[key]:
                raise ValueError(f"{key}: expected shape {expected[key]}, got {tuple(tensor.shape)}")
            self._parameters[key] = tensor

    def __call__(self, x, encoder_hidden_states, encoder_attention_mask=None):
        x = mx.array(x)
        encoder_hidden_states = mx.array(encoder_hidden_states)
        self_output = self._self_attention(_layer_norm(x))
        x = x + self_output
        cross_output = self._cross_attention(x, encoder_hidden_states, encoder_attention_mask)
        return x + cross_output

    def _self_attention(self, x):
        prefix = self._prefix("attn1")
        query = linear(x, self._parameters[f"{prefix}.to_q.weight"])
        key = linear(x, self._parameters[f"{prefix}.to_k.weight"])
        value = linear(x, self._parameters[f"{prefix}.to_v.weight"])
        query = rms_norm(query, self._parameters[f"{prefix}.norm_q.weight"], eps=1e-5)
        key = rms_norm(key, self._parameters[f"{prefix}.norm_k.weight"], eps=1e-5)
        output = _sana_linear_attention(
            query,
            key,
            value,
            heads=self.num_attention_heads,
            head_dim=self.attention_head_dim,
        )
        return linear(output, self._parameters[f"{prefix}.to_out.0.weight"], self._parameters[f"{prefix}.to_out.0.bias"])

    def _cross_attention(self, x, encoder_hidden_states, encoder_attention_mask):
        prefix = self._prefix("attn2")
        query = linear(x, self._parameters[f"{prefix}.to_q.weight"], self._parameters[f"{prefix}.to_q.bias"])
        key = linear(
            encoder_hidden_states,
            self._parameters[f"{prefix}.to_k.weight"],
            self._parameters[f"{prefix}.to_k.bias"],
        )
        value = linear(
            encoder_hidden_states,
            self._parameters[f"{prefix}.to_v.weight"],
            self._parameters[f"{prefix}.to_v.bias"],
        )
        query = rms_norm(query, self._parameters[f"{prefix}.norm_q.weight"], eps=1e-5)
        key = rms_norm(key, self._parameters[f"{prefix}.norm_k.weight"], eps=1e-5)
        output = _scaled_attention(
            query,
            key,
            value,
            heads=self.num_cross_attention_heads,
            head_dim=self.cross_attention_head_dim,
            mask=encoder_attention_mask,
        )
        return linear(output, self._parameters[f"{prefix}.to_out.0.weight"], self._parameters[f"{prefix}.to_out.0.bias"])

    def _prefix(self, attention_name: str) -> str:
        return f"mlx_transformer.transformer_blocks.{self.block_index}.{attention_name}"


def _initial_attention_parameters(
    *,
    hidden_size: int,
    num_attention_heads: int,
    attention_head_dim: int,
    num_cross_attention_heads: int,
    cross_attention_head_dim: int,
    block_index: int,
) -> dict[str, object]:
    del num_attention_heads, attention_head_dim, num_cross_attention_heads, cross_attention_head_dim
    prefix = f"mlx_transformer.transformer_blocks.{block_index}"
    params = {}
    for attention in ("attn1", "attn2"):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            params[f"{prefix}.{attention}.{projection}.weight"] = mx.eye(hidden_size)
        params[f"{prefix}.{attention}.to_out.0.bias"] = mx.zeros((hidden_size,))
        params[f"{prefix}.{attention}.norm_q.weight"] = mx.ones((hidden_size,))
        params[f"{prefix}.{attention}.norm_k.weight"] = mx.ones((hidden_size,))
    for projection in ("to_q", "to_k", "to_v"):
        params[f"{prefix}.attn2.{projection}.bias"] = mx.zeros((hidden_size,))
    return params


def _layer_norm(x, eps: float = 1e-6):
    x = mx.array(x)
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(variance + eps)


def _sana_linear_attention(query, key, value, *, heads: int, head_dim: int):
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = mx.maximum(query.reshape(batch, query_tokens, heads, head_dim).transpose(0, 2, 3, 1), 0)
    key = mx.maximum(key.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3), 0)
    value = value.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 3, 1)
    ones = mx.ones((batch, heads, 1, key_tokens), dtype=value.dtype)
    value = mx.concatenate([value, ones], axis=2)
    scores = mx.matmul(value.astype(mx.float32), key.astype(mx.float32))
    hidden = mx.matmul(scores, query.astype(mx.float32))
    hidden = hidden[:, :, :-1] / (hidden[:, :, -1:] + 1e-15)
    return hidden.transpose(0, 3, 1, 2).reshape(batch, query_tokens, heads * head_dim)


def _scaled_attention(query, key, value, *, heads: int, head_dim: int, mask=None):
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    if mask is not None:
        keep = mx.array(mask).astype(mx.bool_)[:, None, None, :]
        scores = mx.where(keep, scores, mx.array(-1e9, dtype=scores.dtype))
    weights = mx.softmax(scores, axis=-1)
    output = mx.matmul(weights, value)
    return output.transpose(0, 2, 1, 3).reshape(batch, query_tokens, heads * head_dim)
