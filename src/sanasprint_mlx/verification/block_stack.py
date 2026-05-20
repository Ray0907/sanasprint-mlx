from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sanasprint_mlx.transformer.block import RealSanaAttentionBlock
from sanasprint_mlx.transformer.block_weights import load_block_attention_weights_from_snapshot
from sanasprint_mlx.verification.block_attention import _grid_side, _hidden_states, _mlx_dtype, _prompt_inputs, _timestep_embedding
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_block_stack_smoke(
    snapshot: str | Path,
    *,
    dtype: str = "bfloat16",
    seed: int = 0,
    sequence_length: int = 4,
    block_count: int = 2,
) -> dict:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    snapshot_path = Path(snapshot)
    config = load_transformer_config(snapshot_path)
    summary = summarize_transformer_config(config)
    if block_count > summary.num_layers:
        raise ValueError("block_count must be less than or equal to num_layers")

    side = _grid_side(sequence_length)
    hidden_states = _hidden_states(summary.hidden_size, sequence_length=sequence_length, seed=seed)
    encoder_hidden_states, encoder_attention_mask = _prompt_inputs(
        summary.hidden_size,
        seed=seed,
        sequence_length=sequence_length,
    )
    timestep_embedding = _timestep_embedding(summary.hidden_size, seed=seed)

    start = time.perf_counter()
    block_reports = []
    x = hidden_states
    for block_index in range(block_count):
        block = RealSanaAttentionBlock(
            hidden_size=summary.hidden_size,
            num_attention_heads=summary.num_attention_heads,
            attention_head_dim=summary.attention_head_dim,
            num_cross_attention_heads=int(config.get("num_cross_attention_heads", summary.num_attention_heads)),
            cross_attention_head_dim=int(config.get("cross_attention_head_dim", summary.attention_head_dim)),
            block_index=block_index,
            include_ffn=True,
            mlp_ratio=float(config.get("mlp_ratio", 2.5)),
        )
        weight_report = load_block_attention_weights_from_snapshot(
            block,
            snapshot_path,
            block_index=block_index,
            mlx_dtype=_mlx_dtype(dtype),
            strict=True,
        )
        x = block(
            x,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep_embedding=timestep_embedding,
            height=side,
            width=side,
        )
        block_reports.append(
            {
                "block_index": block_index,
                "loaded_keys": {
                    "count": len(weight_report["loaded_keys"]),
                    "keys": weight_report["loaded_keys"],
                },
                "weights": weight_report,
            }
        )
    elapsed = time.perf_counter() - start

    output = np.array(x)
    finite = bool(np.isfinite(output).all())
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "block_count": block_count,
        "scope": "block_stack_core_with_timestep_modulation_not_full_model_parity",
        "dtype": dtype,
        "seed": seed,
        "loaded_keys": {
            "count": sum(block["loaded_keys"]["count"] for block in block_reports),
        },
        "blocks": block_reports,
        "prompt_source": "synthetic_projected_hidden_states",
        "prompt": {
            "embeds_shape": list(encoder_hidden_states.shape),
            "embeds_dtype": str(encoder_hidden_states.dtype),
            "attention_mask_shape": list(encoder_attention_mask.shape),
            "attention_mask_dtype": str(encoder_attention_mask.dtype),
        },
        "timestep": {
            "embedding_shape": list(timestep_embedding.shape),
            "embedding_dtype": str(timestep_embedding.dtype),
        },
        "ffn": {
            "active": True,
            "grid_shape": [side, side],
        },
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": finite,
            "mean": float(output.mean()),
            "std": float(output.std()),
        },
        "runtime": {"wall_time_seconds": elapsed},
    }
