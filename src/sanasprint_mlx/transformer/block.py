from __future__ import annotations

import mlx.core as mx


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
