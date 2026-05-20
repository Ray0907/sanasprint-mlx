from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.real_model import RealSanaTransformerDenoiser
from sanasprint_mlx.verification.scaffold_denoise import _latents, _prompt_inputs
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_real_block_denoise_smoke(
    snapshot: str | Path,
    *,
    dtype: str = "bfloat16",
    prompt_cache: str | Path | None = None,
    seed: int = 0,
    sample_size: int = 2,
    prompt_sequence_length: int = 4,
    block_count: int = 2,
    timestep: float = 0.5,
    guidance_scale: float = 4.5,
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

    start = time.perf_counter()
    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot_path,
        sample_size=sample_size,
        block_count=block_count,
        dtype=dtype,
    )
    prompt_embeds, prompt_attention_mask, prompt_report = _prompt_inputs(
        config,
        prompt_cache=prompt_cache,
        seed=seed,
        sequence_length=prompt_sequence_length,
    )
    latents = _latents(config, seed=seed)
    timestep_values = np.full((latents.shape[0],), timestep, dtype=np.float32)
    guidance = np.full((latents.shape[0],), guidance_scale * summary.guidance_embeds_scale, dtype=np.float32)
    output = transformer(
        latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        guidance=guidance,
        timestep=timestep_values,
        return_dict=False,
    )
    elapsed = time.perf_counter() - start

    output_array = np.array(output[0])
    finite = bool(np.isfinite(output_array).all())
    weight_report = transformer.weight_report
    tensor_dtype = str(transformer.dtype).removeprefix("mlx.core.")
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "scope": "real_block_denoise_smoke_not_full_model_parity",
        "dtype": dtype,
        "seed": seed,
        "sample_size": sample_size,
        "block_count": block_count,
        "loaded_keys": weight_report["loaded_keys"],
        "caption_projection_source": weight_report["caption_projection_source"],
        "time_embedding_source": weight_report["time_embedding_source"],
        "output_norm_source": weight_report["output_norm_source"],
        "scaffold_weights": weight_report["scaffold_weights"],
        "time_embedding_weights": weight_report["time_embedding_weights"],
        "output_norm_weights": weight_report["output_norm_weights"],
        "blocks": weight_report["blocks"],
        "prompt_source": prompt_report["source"],
        "prompt_cache": prompt_report.get("cache"),
        "prompt": {
            "embeds_shape": list(prompt_embeds.shape),
            "embeds_dtype": str(prompt_embeds.dtype),
            "attention_mask_shape": list(prompt_attention_mask.shape),
            "attention_mask_dtype": str(prompt_attention_mask.dtype),
            "projected_shape": [
                prompt_embeds.shape[0],
                prompt_embeds.shape[1],
                transformer.config.hidden_size,
            ],
            "projected_dtype": tensor_dtype,
        },
        "timestep": {
            "value": timestep,
            "guidance_scale": guidance_scale,
            "embedding_shape": [latents.shape[0], 6 * transformer.config.hidden_size],
            "embedding_dtype": tensor_dtype,
            "conditioning_shape": [latents.shape[0], transformer.config.hidden_size],
            "conditioning_dtype": tensor_dtype,
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
