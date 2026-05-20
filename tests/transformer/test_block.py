import numpy as np
import mlx.core as mx
import pytest
from safetensors.numpy import save_file

from sanasprint_mlx.transformer.block import RealSanaAttentionBlock, ToyTransformerBlock
from sanasprint_mlx.transformer.block_weights import load_block_attention_weights_from_snapshot


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


def real_attention_state(block_index=0, value=0.0):
    prefix = f"mlx_transformer.transformer_blocks.{block_index}"
    state = {f"{prefix}.scale_shift_table": np.zeros((6, 4), dtype=np.float32)}
    for attention in ("attn1", "attn2"):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            state[f"{prefix}.{attention}.{projection}.weight"] = np.eye(4, dtype=np.float32)
        state[f"{prefix}.{attention}.to_out.0.bias"] = np.full((4,), value, dtype=np.float32)
        state[f"{prefix}.{attention}.norm_q.weight"] = np.ones((4,), dtype=np.float32)
        state[f"{prefix}.{attention}.norm_k.weight"] = np.ones((4,), dtype=np.float32)
    for projection in ("to_q", "to_k", "to_v"):
        state[f"{prefix}.attn2.{projection}.bias"] = np.zeros((4,), dtype=np.float32)
    return state


def test_real_sana_attention_block_preserves_shape_and_is_deterministic():
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )
    block.load_parameters(real_attention_state(value=0.25))
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10.0
    encoder = np.ones((1, 3, 4), dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int32)

    first = np.array(block(x, encoder, mask))
    second = np.array(block(x, encoder, mask))

    assert first.shape == x.shape
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-5, rtol=0)


def test_real_sana_attention_block_matches_numpy_reference_with_asymmetric_weights():
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )
    state = real_attention_state()
    for index, key in enumerate(state):
        if key.endswith(".weight") and "norm_" not in key:
            state[key] = (np.arange(16, dtype=np.float32).reshape(4, 4) + index + 1) / 25.0
        elif key.endswith(".bias"):
            state[key] = (np.arange(4, dtype=np.float32) + index) / 50.0
    block.load_parameters(state)
    x = np.array([[[0.2, -0.4, 0.5, 0.7], [1.0, -0.1, 0.3, -0.2]]], dtype=np.float32)
    encoder = np.array(
        [[[0.3, -0.2, 0.9, 0.1], [1.1, 0.4, -0.5, 0.6], [-0.7, 0.8, 0.2, -0.3]]],
        dtype=np.float32,
    )
    mask = np.array([[1, 0, 1]], dtype=np.int32)

    result = np.array(block(x, encoder, mask))
    expected = _numpy_attention_block_reference(x, encoder, mask, state)

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


def test_real_sana_attention_block_applies_timestep_self_attention_modulation():
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )
    state = real_attention_state()
    state["mlx_transformer.transformer_blocks.0.scale_shift_table"] = (
        np.arange(24, dtype=np.float32).reshape(6, 4) / 100.0
    )
    for index, key in enumerate(state):
        if key.endswith(".weight") and "norm_" not in key:
            state[key] = (np.arange(16, dtype=np.float32).reshape(4, 4) + index + 1) / 30.0
        elif key.endswith(".bias"):
            state[key] = (np.arange(4, dtype=np.float32) + index) / 60.0
    block.load_parameters(state)
    x = np.array([[[0.2, -0.4, 0.5, 0.7], [1.0, -0.1, 0.3, -0.2]]], dtype=np.float32)
    encoder = np.array(
        [[[0.3, -0.2, 0.9, 0.1], [1.1, 0.4, -0.5, 0.6], [-0.7, 0.8, 0.2, -0.3]]],
        dtype=np.float32,
    )
    mask = np.array([[1, 0, 1]], dtype=np.int32)
    timestep_embedding = np.arange(24, dtype=np.float32).reshape(1, 24) / 200.0

    result = np.array(block(x, encoder, mask, timestep_embedding=timestep_embedding))
    expected = _numpy_attention_block_reference(x, encoder, mask, state, timestep_embedding=timestep_embedding)

    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)


def test_real_sana_attention_block_rejects_incompatible_head_dimensions():
    with pytest.raises(ValueError, match="num_attention_heads"):
        RealSanaAttentionBlock(
            hidden_size=4,
            num_attention_heads=3,
            attention_head_dim=2,
            num_cross_attention_heads=2,
            cross_attention_head_dim=2,
        )


def test_real_sana_attention_block_rejects_bad_timestep_embedding_size():
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )

    with pytest.raises(ValueError, match="timestep_embedding"):
        block(
            np.zeros((1, 2, 4), dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            np.ones((1, 2), dtype=np.int32),
            timestep_embedding=np.zeros((1, 23), dtype=np.float32),
        )


def test_block_attention_loader_reads_exact_block_zero_tensors(tmp_path):
    snapshot = tmp_path / "snapshot"
    transformer_dir = snapshot / "transformer"
    transformer_dir.mkdir(parents=True)
    tensors = {
        key.removeprefix("mlx_transformer."): value
        for key, value in real_attention_state(value=0.5).items()
    }
    save_file(tensors, transformer_dir / "model.safetensors")
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )

    report = load_block_attention_weights_from_snapshot(block, snapshot, block_index=0, mlx_dtype=mx.float16)

    assert report["block_index"] == 0
    assert report["loaded_keys"] == list(real_attention_state())
    assert report["source_tensors"]["mlx_transformer.transformer_blocks.0.attn1.to_q.weight"]["final_dtype"] == "float16"
    assert block.parameters()["mlx_transformer.transformer_blocks.0.attn1.to_q.weight"].dtype == mx.float16


def test_block_attention_loader_rejects_missing_required_tensor(tmp_path):
    snapshot = tmp_path / "snapshot"
    transformer_dir = snapshot / "transformer"
    transformer_dir.mkdir(parents=True)
    tensors = {
        key.removeprefix("mlx_transformer."): value
        for key, value in real_attention_state().items()
        if not key.endswith("attn2.to_v.bias")
    }
    save_file(tensors, transformer_dir / "model.safetensors")
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
    )

    with pytest.raises(KeyError, match="attn2.to_v.bias"):
        load_block_attention_weights_from_snapshot(block, snapshot, block_index=0)


def _numpy_attention_block_reference(x, encoder, mask, state, *, timestep_embedding=None):
    prefix = "mlx_transformer.transformer_blocks.0"
    return _numpy_attention_block_reference_with_prefix(x, encoder, mask, state, prefix, timestep_embedding=timestep_embedding)


def _numpy_attention_block_reference_with_prefix(x, encoder, mask, state, prefix, timestep_embedding):
    norm_x = _numpy_layer_norm(x)
    gate_msa = None
    if timestep_embedding is not None:
        modulation = state[f"{prefix}.scale_shift_table"][None] + timestep_embedding.reshape(x.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa = modulation[:, 0:1], modulation[:, 1:2], modulation[:, 2:3]
        norm_x = norm_x * (1 + scale_msa) + shift_msa
    self_out = _numpy_self_attention(norm_x, state, f"{prefix}.attn1")
    x = x + (gate_msa * self_out if gate_msa is not None else self_out)
    cross_out = _numpy_cross_attention(x, encoder, mask, state, f"{prefix}.attn2")
    return x + cross_out


def _numpy_self_attention(x, state, prefix):
    q = _numpy_rms_norm(_numpy_linear(x, state[f"{prefix}.to_q.weight"]), state[f"{prefix}.norm_q.weight"])
    k = _numpy_rms_norm(_numpy_linear(x, state[f"{prefix}.to_k.weight"]), state[f"{prefix}.norm_k.weight"])
    v = _numpy_linear(x, state[f"{prefix}.to_v.weight"])
    out = _numpy_sana_linear_attention(q, k, v, heads=2, head_dim=2)
    return _numpy_linear(out, state[f"{prefix}.to_out.0.weight"], state[f"{prefix}.to_out.0.bias"])


def _numpy_cross_attention(x, encoder, mask, state, prefix):
    q = _numpy_rms_norm(_numpy_linear(x, state[f"{prefix}.to_q.weight"], state[f"{prefix}.to_q.bias"]), state[f"{prefix}.norm_q.weight"])
    k = _numpy_rms_norm(
        _numpy_linear(encoder, state[f"{prefix}.to_k.weight"], state[f"{prefix}.to_k.bias"]),
        state[f"{prefix}.norm_k.weight"],
    )
    v = _numpy_linear(encoder, state[f"{prefix}.to_v.weight"], state[f"{prefix}.to_v.bias"])
    out = _numpy_scaled_attention(q, k, v, heads=2, head_dim=2, mask=mask)
    return _numpy_linear(out, state[f"{prefix}.to_out.0.weight"], state[f"{prefix}.to_out.0.bias"])


def _numpy_linear(x, weight, bias=None):
    y = np.matmul(x, weight.T)
    if bias is not None:
        y = y + bias
    return y


def _numpy_layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def _numpy_rms_norm(x, weight, eps=1e-5):
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def _numpy_sana_linear_attention(query, key, value, *, heads, head_dim):
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = np.maximum(query.reshape(batch, query_tokens, heads, head_dim).transpose(0, 2, 3, 1), 0)
    key = np.maximum(key.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3), 0)
    value = value.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 3, 1)
    value = np.concatenate([value, np.ones((batch, heads, 1, key_tokens), dtype=value.dtype)], axis=2)
    scores = np.matmul(value, key)
    hidden = np.matmul(scores, query)
    hidden = hidden[:, :, :-1] / (hidden[:, :, -1:] + 1e-15)
    return hidden.transpose(0, 3, 1, 2).reshape(batch, query_tokens, heads * head_dim)


def _numpy_scaled_attention(query, key, value, *, heads, head_dim, mask):
    batch, query_tokens, _ = query.shape
    key_tokens = key.shape[1]
    query = query.reshape(batch, query_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(batch, key_tokens, heads, head_dim).transpose(0, 2, 1, 3)
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = np.where(mask.astype(bool)[:, None, None, :], scores, -1e9)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = np.matmul(weights, value)
    return out.transpose(0, 2, 1, 3).reshape(batch, query_tokens, heads * head_dim)
