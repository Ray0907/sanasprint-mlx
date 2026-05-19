from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.text.cache import read_prompt_cache
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.weights import load_scaffold_weights_from_snapshot
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_scaffold_denoise_smoke(
    snapshot: str | Path,
    *,
    prompt_cache: str | Path | None = None,
    dtype: str = "bfloat16",
    seed: int = 0,
    steps: int = 1,
    sequence_length: int = 4,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    mlx_dtype = _mlx_dtype(dtype)
    snapshot_path = Path(snapshot)
    summary = summarize_transformer_config(load_transformer_config(snapshot_path))
    config = _transformer_config_from_summary(summary)
    model = SanaTransformerDenoiser(config)

    start = time.perf_counter()
    weight_report = load_scaffold_weights_from_snapshot(model, snapshot_path, mlx_dtype=mlx_dtype, strict=True)
    prompt_embeds, prompt_attention_mask, prompt_report = _prompt_inputs(
        config,
        prompt_cache=prompt_cache,
        seed=seed,
        sequence_length=sequence_length,
    )
    latents = _latents(config, seed=seed)
    result = run_denoising_loop(
        transformer=model,
        scheduler=SCMScheduler(),
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        num_inference_steps=steps,
        intermediate_timesteps=None,
        noise_fn=_seeded_noise_fn(seed + 2),
    )
    elapsed = time.perf_counter() - start
    latents_out = np.asarray(result.latents)
    finite = bool(np.isfinite(latents_out).all())
    return {
        "status": "PASS" if finite else "FAIL",
        "snapshot_path": str(snapshot_path),
        "dtype": dtype,
        "steps": steps,
        "seed": seed,
        "loaded_keys": weight_report["loaded_keys"],
        "weights": weight_report,
        "prompt_source": prompt_report["source"],
        "prompt": {
            "embeds_shape": list(prompt_embeds.shape),
            "embeds_dtype": str(prompt_embeds.dtype),
            "attention_mask_shape": list(prompt_attention_mask.shape),
            "attention_mask_dtype": str(prompt_attention_mask.dtype),
        },
        "prompt_cache": prompt_report.get("cache"),
        "latents": {
            "shape": list(latents_out.shape),
            "dtype": str(latents_out.dtype),
            "finite": finite,
            "mean": float(latents_out.mean()),
            "std": float(latents_out.std()),
        },
        "runtime": {"wall_time_seconds": elapsed},
    }


def _transformer_config_from_summary(summary) -> SanaTransformerConfig:
    return SanaTransformerConfig(
        hidden_size=summary.hidden_size,
        in_channels=summary.in_channels,
        out_channels=summary.out_channels,
        caption_channels=summary.caption_channels,
        num_layers=1,
        num_attention_heads=summary.num_attention_heads,
        attention_head_dim=summary.attention_head_dim,
        patch_size=summary.patch_size,
        sample_size=summary.sample_size,
        guidance_embeds_scale=summary.guidance_embeds_scale,
    )


def _prompt_inputs(
    config: SanaTransformerConfig,
    *,
    prompt_cache: str | Path | None,
    seed: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if prompt_cache is not None:
        arrays, metadata = read_prompt_cache(prompt_cache)
        if "prompt_embeds" not in arrays or "prompt_attention_mask" not in arrays:
            raise ValueError("prompt cache requires prompt_embeds and prompt_attention_mask")
        prompt_embeds = np.asarray(arrays["prompt_embeds"], dtype=np.float32)
        prompt_attention_mask = np.asarray(arrays["prompt_attention_mask"], dtype=np.int32)
        return prompt_embeds, prompt_attention_mask, {
            "source": "prompt_cache",
            "cache": {"path": str(prompt_cache), "metadata": metadata},
        }

    rng = np.random.default_rng(seed)
    prompt_embeds = rng.standard_normal((1, sequence_length, config.caption_channels), dtype=np.float32)
    prompt_attention_mask = np.ones((1, sequence_length), dtype=np.int32)
    return prompt_embeds, prompt_attention_mask, {"source": "synthetic"}


def _latents(config: SanaTransformerConfig, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return rng.standard_normal(
        (1, config.in_channels, config.sample_size, config.sample_size),
        dtype=np.float32,
    )


def _seeded_noise_fn(seed: int):
    rng = np.random.default_rng(seed)

    def noise(shape, dtype):
        del dtype
        return rng.standard_normal(shape, dtype=np.float32)

    return noise


def _mlx_dtype(dtype: str):
    values = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if dtype not in values:
        raise ValueError(f"dtype must be one of {', '.join(values)}")
    return values[dtype]
