from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from sanasprint_mlx.primitives.patch import patchify_nchw, unpatchify_nchw
from sanasprint_mlx.transformer.block import RealSanaAttentionBlock
from sanasprint_mlx.transformer.block_weights import load_block_attention_weights_from_snapshot
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.weights import load_scaffold_weights_from_snapshot
from sanasprint_mlx.verification.block_attention import _grid_side, _timestep_embedding
from sanasprint_mlx.verification.scaffold_denoise import _latents, _mlx_dtype, _prompt_inputs
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_real_block_denoise_smoke(
    snapshot: str | Path,
    *,
    dtype: str = "bfloat16",
    seed: int = 0,
    sample_size: int = 2,
    prompt_sequence_length: int = 4,
    block_count: int = 2,
) -> dict:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if prompt_sequence_length <= 0:
        raise ValueError("prompt_sequence_length must be positive")
    if block_count <= 0:
        raise ValueError("block_count must be positive")

    snapshot_path = Path(snapshot)
    config_dict = load_transformer_config(snapshot_path)
    summary = summarize_transformer_config(config_dict)
    if block_count > summary.num_layers:
        raise ValueError("block_count must be less than or equal to num_layers")

    config = _transformer_config_from_summary(summary, sample_size=sample_size, num_layers=block_count)
    model = SanaTransformerDenoiser(config)
    mlx_dtype = _mlx_dtype(dtype)

    start = time.perf_counter()
    scaffold_report = load_scaffold_weights_from_snapshot(
        model,
        snapshot_path,
        mlx_dtype=mlx_dtype,
        strict=True,
        include_caption_projection=True,
    )
    prompt_embeds, prompt_attention_mask, prompt_report = _prompt_inputs(
        config,
        prompt_cache=None,
        seed=seed,
        sequence_length=prompt_sequence_length,
    )
    latents = _latents(config, seed=seed)
    tokens = patchify_nchw(mx.array(latents), config.patch_size)
    x = mx.matmul(tokens, model.input_weight.T) + model.input_bias
    encoder_hidden_states = model._project_encoder_hidden_states(mx.array(prompt_embeds))
    encoder_attention_mask = mx.array(prompt_attention_mask)
    timestep_embedding = _timestep_embedding(summary.hidden_size, seed=seed)
    side = _grid_side(x.shape[1])

    block_reports = []
    for block_index in range(block_count):
        block = RealSanaAttentionBlock(
            hidden_size=summary.hidden_size,
            num_attention_heads=summary.num_attention_heads,
            attention_head_dim=summary.attention_head_dim,
            num_cross_attention_heads=int(config_dict.get("num_cross_attention_heads", summary.num_attention_heads)),
            cross_attention_head_dim=int(config_dict.get("cross_attention_head_dim", summary.attention_head_dim)),
            block_index=block_index,
            include_ffn=True,
            mlp_ratio=float(config_dict.get("mlp_ratio", 2.5)),
        )
        weight_report = load_block_attention_weights_from_snapshot(
            block,
            snapshot_path,
            block_index=block_index,
            mlx_dtype=mlx_dtype,
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

    out_tokens = mx.matmul(x, model.output_weight.T) + model.output_bias
    output = unpatchify_nchw(
        out_tokens,
        patch_size=config.patch_size,
        height=sample_size,
        width=sample_size,
        channels=config.out_channels,
    )
    elapsed = time.perf_counter() - start

    output_array = np.array(output)
    finite = bool(np.isfinite(output_array).all())
    block_loaded_count = sum(block["loaded_keys"]["count"] for block in block_reports)
    scaffold_count = len(scaffold_report["loaded_keys"])
    caption_count = len(scaffold_report["loaded_caption_keys"])
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "scope": "real_block_denoise_smoke_not_full_model_parity",
        "dtype": dtype,
        "seed": seed,
        "sample_size": sample_size,
        "block_count": block_count,
        "loaded_keys": {
            "scaffold_count": scaffold_count,
            "caption_count": caption_count,
            "block_count": block_loaded_count,
            "total_count": scaffold_count + caption_count + block_loaded_count,
        },
        "caption_projection_source": scaffold_report["caption_projection_source"],
        "scaffold_weights": scaffold_report,
        "blocks": block_reports,
        "prompt_source": prompt_report["source"],
        "prompt": {
            "embeds_shape": list(prompt_embeds.shape),
            "embeds_dtype": str(prompt_embeds.dtype),
            "attention_mask_shape": list(prompt_attention_mask.shape),
            "attention_mask_dtype": str(prompt_attention_mask.dtype),
            "projected_shape": list(encoder_hidden_states.shape),
            "projected_dtype": str(encoder_hidden_states.dtype),
        },
        "timestep": {
            "embedding_shape": list(timestep_embedding.shape),
            "embedding_dtype": str(timestep_embedding.dtype),
        },
        "latents": {
            "input_shape": list(latents.shape),
            "input_dtype": str(latents.dtype),
        },
        "output": {
            "shape": list(output_array.shape),
            "dtype": str(output_array.dtype),
            "finite": finite,
            "mean": float(output_array.mean()),
            "std": float(output_array.std()),
        },
        "runtime": {"wall_time_seconds": elapsed},
    }


def _transformer_config_from_summary(summary, *, sample_size: int, num_layers: int) -> SanaTransformerConfig:
    return SanaTransformerConfig(
        hidden_size=summary.hidden_size,
        in_channels=summary.in_channels,
        out_channels=summary.out_channels,
        caption_channels=summary.caption_channels,
        num_layers=num_layers,
        num_attention_heads=summary.num_attention_heads,
        attention_head_dim=summary.attention_head_dim,
        patch_size=summary.patch_size,
        sample_size=sample_size,
        guidance_embeds_scale=summary.guidance_embeds_scale,
    )
