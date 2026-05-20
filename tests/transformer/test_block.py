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


def real_ffn_state(block_index=0, *, hidden_size=4, hidden_channels=8):
    prefix = f"mlx_transformer.transformer_blocks.{block_index}.ff"
    return {
        f"{prefix}.conv_inverted.weight": np.zeros((hidden_channels * 2, hidden_size, 1, 1), dtype=np.float32),
        f"{prefix}.conv_inverted.bias": np.zeros((hidden_channels * 2,), dtype=np.float32),
        f"{prefix}.conv_depth.weight": np.zeros((hidden_channels * 2, 1, 3, 3), dtype=np.float32),
        f"{prefix}.conv_depth.bias": np.zeros((hidden_channels * 2,), dtype=np.float32),
        f"{prefix}.conv_point.weight": np.zeros((hidden_size, hidden_channels, 1, 1), dtype=np.float32),
    }


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


def test_real_sana_attention_block_applies_modulated_glumbconv_ffn():
    block = RealSanaAttentionBlock(
        hidden_size=4,
        num_attention_heads=2,
        attention_head_dim=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=2,
        include_ffn=True,
        mlp_ratio=2.0,
    )
    state = real_attention_state()
    state.update(real_ffn_state(hidden_channels=8))
    state["mlx_transformer.transformer_blocks.0.scale_shift_table"] = (
        np.arange(24, dtype=np.float32).reshape(6, 4) / 100.0
    )
    for index, key in enumerate(state):
        if key.endswith(".weight") and "norm_" not in key and "scale_shift" not in key:
            size = int(np.prod(state[key].shape))
            state[key] = (np.arange(size, dtype=np.float32).reshape(state[key].shape) + index + 1) / 100.0
        elif key.endswith(".bias"):
            state[key] = (np.arange(state[key].shape[0], dtype=np.float32) + index) / 80.0
    block.load_parameters(state)
    x = np.array(
        [[[0.2, -0.4, 0.5, 0.7], [1.0, -0.1, 0.3, -0.2], [0.4, 0.6, -0.8, 0.2], [-0.5, 0.9, 0.1, -0.3]]],
        dtype=np.float32,
    )
    encoder = np.array(
        [[[0.3, -0.2, 0.9, 0.1], [1.1, 0.4, -0.5, 0.6], [-0.7, 0.8, 0.2, -0.3], [0.5, -0.4, 0.7, 0.8]]],
        dtype=np.float32,
    )
    mask = np.array([[1, 0, 1, 1]], dtype=np.int32)
    timestep_embedding = np.arange(24, dtype=np.float32).reshape(1, 24) / 200.0

    result = np.array(block(x, encoder, mask, timestep_embedding=timestep_embedding, height=2, width=2))
    expected = _numpy_attention_block_reference(
        x,
        encoder,
        mask,
        state,
        timestep_embedding=timestep_embedding,
        include_ffn=True,
        height=2,
        width=2,
    )

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


def _numpy_attention_block_reference(
    x,
    encoder,
    mask,
    state,
    *,
    timestep_embedding=None,
    include_ffn=False,
    height=None,
    width=None,
):
    prefix = "mlx_transformer.transformer_blocks.0"
    return _numpy_attention_block_reference_with_prefix(
        x,
        encoder,
        mask,
        state,
        prefix,
        timestep_embedding=timestep_embedding,
        include_ffn=include_ffn,
        height=height,
        width=width,
    )


def _numpy_attention_block_reference_with_prefix(
    x,
    encoder,
    mask,
    state,
    prefix,
    timestep_embedding,
    include_ffn=False,
    height=None,
    width=None,
):
    norm_x = _numpy_layer_norm(x)
    gate_msa = gate_mlp = shift_mlp = scale_mlp = None
    if timestep_embedding is not None:
        modulation = state[f"{prefix}.scale_shift_table"][None] + timestep_embedding.reshape(x.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa = modulation[:, 0:1], modulation[:, 1:2], modulation[:, 2:3]
        shift_mlp, scale_mlp, gate_mlp = modulation[:, 3:4], modulation[:, 4:5], modulation[:, 5:6]
        norm_x = norm_x * (1 + scale_msa) + shift_msa
    self_out = _numpy_self_attention(norm_x, state, f"{prefix}.attn1")
    x = x + (gate_msa * self_out if gate_msa is not None else self_out)
    cross_out = _numpy_cross_attention(x, encoder, mask, state, f"{prefix}.attn2")
    x = x + cross_out
    if include_ffn:
        norm_x = _numpy_layer_norm(x)
        if shift_mlp is not None and scale_mlp is not None:
            norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        ffn_out = _numpy_glumbconv(norm_x, state, f"{prefix}.ff", height=height, width=width)
        x = x + (gate_mlp * ffn_out if gate_mlp is not None else ffn_out)
    return x


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


def _numpy_glumbconv(x, state, prefix, *, height, width):
    batch, tokens, channels = x.shape
    image = x.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
    hidden = _numpy_conv2d_nchw(
        image,
        state[f"{prefix}.conv_inverted.weight"],
        state[f"{prefix}.conv_inverted.bias"],
    )
    hidden = _numpy_silu(hidden)
    hidden = _numpy_depthwise_conv2d_nchw(
        hidden,
        state[f"{prefix}.conv_depth.weight"],
        state[f"{prefix}.conv_depth.bias"],
        padding=1,
    )
    hidden, gate = np.split(hidden, 2, axis=1)
    hidden = hidden * _numpy_silu(gate)
    hidden = _numpy_conv2d_nchw(hidden, state[f"{prefix}.conv_point.weight"], None)
    return hidden.transpose(0, 2, 3, 1).reshape(batch, tokens, channels)


def _numpy_silu(x):
    return x / (1.0 + np.exp(-x))


def _numpy_conv2d_nchw(x, weight, bias=None):
    batch, _, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    out = np.zeros((batch, out_channels, height - kernel_height + 1, width - kernel_width + 1), dtype=np.float32)
    for b in range(batch):
        for oc in range(out_channels):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    patch = x[b, :, i : i + kernel_height, j : j + kernel_width]
                    out[b, oc, i, j] = np.sum(patch * weight[oc])
            if bias is not None:
                out[b, oc] += bias[oc]
    return out


def _numpy_depthwise_conv2d_nchw(x, weight, bias=None, padding=0):
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    batch, channels, height, width = x.shape
    out = np.zeros((batch, channels, height - 2, width - 2), dtype=np.float32)
    for b in range(batch):
        for c in range(channels):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    patch = x[b, c : c + 1, i : i + 3, j : j + 3]
                    out[b, c, i, j] = np.sum(patch * weight[c])
            if bias is not None:
                out[b, c] += bias[c]
    return out
