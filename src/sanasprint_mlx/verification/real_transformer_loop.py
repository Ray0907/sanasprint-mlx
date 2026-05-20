from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.real_model import RealSanaTransformerDenoiser
from sanasprint_mlx.verification.scaffold_denoise import _latents, _prompt_inputs, _seeded_noise_fn
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_real_transformer_loop_smoke(
    snapshot: str | Path,
    *,
    dtype: str = "bfloat16",
    prompt_cache: str | Path | None = None,
    seed: int = 0,
    steps: int = 1,
    sample_size: int = 2,
    prompt_sequence_length: int = 4,
    block_count: int = 2,
    guidance_scale: float = 4.5,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
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
    input_latents = _latents(config, seed=seed)
    result = run_denoising_loop(
        transformer=transformer,
        scheduler=SCMScheduler(),
        latents=input_latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        num_inference_steps=steps,
        intermediate_timesteps=None,
        guidance_scale=guidance_scale,
        noise_fn=_seeded_noise_fn(seed + 2),
    )
    elapsed = time.perf_counter() - start
    latents_out = np.array(result.latents)
    finite = bool(np.isfinite(latents_out).all())
    weight_report = transformer.weight_report
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "scope": "real_transformer_loop_smoke_not_full_image_generation",
        "dtype": dtype,
        "seed": seed,
        "steps": steps,
        "sample_size": sample_size,
        "block_count": block_count,
        "guidance_scale": guidance_scale,
        "loaded_keys": weight_report["loaded_keys"],
        "caption_projection_source": weight_report["caption_projection_source"],
        "time_embedding_source": weight_report["time_embedding_source"],
        "output_norm_source": weight_report["output_norm_source"],
        "blocks": weight_report["blocks"],
        "prompt_source": prompt_report["source"],
        "prompt_cache": prompt_report.get("cache"),
        "prompt": {
            "embeds_shape": list(prompt_embeds.shape),
            "embeds_dtype": str(prompt_embeds.dtype),
            "attention_mask_shape": list(prompt_attention_mask.shape),
            "attention_mask_dtype": str(prompt_attention_mask.dtype),
        },
        "input_latents": {
            "shape": list(input_latents.shape),
            "dtype": str(input_latents.dtype),
        },
        "latents": {
            "shape": list(latents_out.shape),
            "dtype": str(latents_out.dtype),
            "finite": finite,
            "mean": float(latents_out.mean()),
            "std": float(latents_out.std()),
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
