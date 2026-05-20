from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sanasprint_mlx.transformer.block import RealSanaAttentionBlock
from sanasprint_mlx.transformer.block_weights import load_block_attention_weights_from_snapshot
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_block0_attention_smoke(
    snapshot: str | Path,
    *,
    dtype: str = "bfloat16",
    seed: int = 0,
    sequence_length: int = 4,
) -> dict:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    snapshot_path = Path(snapshot)
    config = load_transformer_config(snapshot_path)
    summary = summarize_transformer_config(config)
    block = RealSanaAttentionBlock(
        hidden_size=summary.hidden_size,
        num_attention_heads=summary.num_attention_heads,
        attention_head_dim=summary.attention_head_dim,
        num_cross_attention_heads=int(config.get("num_cross_attention_heads", summary.num_attention_heads)),
        cross_attention_head_dim=int(config.get("cross_attention_head_dim", summary.attention_head_dim)),
        block_index=0,
        include_ffn=True,
        mlp_ratio=float(config.get("mlp_ratio", 2.5)),
    )

    start = time.perf_counter()
    weight_report = load_block_attention_weights_from_snapshot(
        block,
        snapshot_path,
        block_index=0,
        mlx_dtype=_mlx_dtype(dtype),
        strict=True,
    )
    hidden_states = _hidden_states(summary.hidden_size, sequence_length=sequence_length, seed=seed)
    encoder_hidden_states, encoder_attention_mask = _prompt_inputs(
        summary.hidden_size,
        seed=seed,
        sequence_length=sequence_length,
    )
    timestep_embedding = _timestep_embedding(summary.hidden_size, seed=seed)
    output = np.array(
        block(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep_embedding=timestep_embedding,
            height=_grid_side(sequence_length),
            width=_grid_side(sequence_length),
        )
    )
    elapsed = time.perf_counter() - start
    finite = bool(np.isfinite(output).all())
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "block_index": 0,
        "scope": "block0_core_with_timestep_modulation_not_full_model_parity",
        "dtype": dtype,
        "seed": seed,
        "loaded_keys": {
            "count": len(weight_report["loaded_keys"]),
            "keys": weight_report["loaded_keys"],
        },
        "weights": weight_report,
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
            "grid_shape": [_grid_side(sequence_length), _grid_side(sequence_length)],
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


def _hidden_states(hidden_size: int, *, sequence_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 11)
    return rng.standard_normal((1, sequence_length, hidden_size), dtype=np.float32)


def _prompt_inputs(
    hidden_size: int,
    *,
    seed: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 12)
    prompt_embeds = rng.standard_normal((1, sequence_length, hidden_size), dtype=np.float32)
    prompt_attention_mask = np.ones((1, sequence_length), dtype=np.int32)
    return prompt_embeds, prompt_attention_mask


def _timestep_embedding(hidden_size: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 13)
    return rng.standard_normal((1, 6 * hidden_size), dtype=np.float32)


def _grid_side(sequence_length: int) -> int:
    side = int(sequence_length**0.5)
    if side * side != sequence_length:
        raise ValueError("sequence_length must be a square when FFN is active")
    return side


def _mlx_dtype(dtype: str):
    import mlx.core as mx

    values = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if dtype not in values:
        raise ValueError(f"dtype must be one of {', '.join(values)}")
    return values[dtype]
