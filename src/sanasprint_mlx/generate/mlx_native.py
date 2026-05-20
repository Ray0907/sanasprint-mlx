from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sanasprint_mlx.autoencoder.mlx_decoder import MLXAutoencoderDCDecoder
from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.text.cache import read_prompt_cache
from sanasprint_mlx.text.instruction import DEFAULT_COMPLEX_HUMAN_INSTRUCTION
from sanasprint_mlx.text.mlx_encoder import encode_prompt_mlx
from sanasprint_mlx.transformer.real_model import RealSanaTransformerDenoiser
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


def run_mlx_generation(
    *,
    prompt: str | None = None,
    prompt_cache: str | Path | None = None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    output: str | Path,
    snapshot: str | Path | None,
    mlx_dtype: str = "bfloat16",
) -> dict:
    start = time.perf_counter()
    snapshot_path = _require_local_snapshot(snapshot)
    output_path = Path(output)
    prompt_embeds, prompt_attention_mask, prompt_source = _prompt_inputs(
        prompt=prompt,
        prompt_cache=prompt_cache,
        snapshot=snapshot_path,
    )
    summary = summarize_transformer_config(load_transformer_config(snapshot_path))
    sample_size = max(height // 32, 1)
    latents = _latents(
        channels=summary.in_channels,
        height=sample_size,
        width=max(width // 32, 1),
        seed=seed,
    )
    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot_path,
        sample_size=sample_size,
        block_count=None,
        dtype=mlx_dtype,
    )
    result = run_denoising_loop(
        transformer=transformer,
        scheduler=SCMScheduler(),
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        num_inference_steps=steps,
    )
    decoder = MLXAutoencoderDCDecoder.from_snapshot(snapshot_path, dtype=mlx_dtype)
    decoded = np.array(decoder.decode(np.array(result.latents, dtype=np.float32) / _scaling_factor(snapshot_path)))
    image = _postprocess(decoded[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "mode": "mlx_transformer_mlx_decode",
        "output": str(output_path),
        "model": str(snapshot_path),
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "mlx_dtype": mlx_dtype,
        "prompt_source": prompt_source,
        "latents_shape": [int(dim) for dim in np.array(result.latents).shape],
        "loaded_keys": transformer.weight_report["loaded_keys"],
        "runtime": {"wall_time_seconds": time.perf_counter() - start},
        "memory": {"max_rss_bytes": _max_rss_bytes()},
    }


def _prompt_inputs(
    *,
    prompt: str | None,
    prompt_cache: str | Path | None,
    snapshot: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    if prompt_cache is not None:
        prompt_embeds, prompt_attention_mask = _read_prompt_inputs(prompt_cache)
        return prompt_embeds, prompt_attention_mask, "prompt_cache"
    if prompt is None:
        raise ValueError("native MLX generation requires prompt or prompt_cache")
    encoded = encode_prompt_mlx(
        prompt=prompt,
        snapshot=snapshot,
        complex_human_instruction=DEFAULT_COMPLEX_HUMAN_INSTRUCTION,
    )
    return encoded.prompt_embeds, encoded.prompt_attention_mask, "mlx_text_encoder"


def _read_prompt_inputs(prompt_cache: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arrays, _ = read_prompt_cache(prompt_cache)
    if "prompt_embeds" not in arrays or "prompt_attention_mask" not in arrays:
        raise ValueError("prompt cache requires prompt_embeds and prompt_attention_mask")
    return (
        np.asarray(arrays["prompt_embeds"], dtype=np.float32),
        np.asarray(arrays["prompt_attention_mask"], dtype=np.int32),
    )


def _latents(*, channels: int, height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, channels, height, width), dtype=np.float32)


def _postprocess(sample: np.ndarray) -> Image.Image:
    sample = np.asarray(sample, dtype=np.float32)
    sample = np.clip(sample / 2.0 + 0.5, 0.0, 1.0)
    sample = np.transpose(sample, (1, 2, 0))
    sample = (sample * 255.0).round().astype(np.uint8)
    return Image.fromarray(sample)


def _scaling_factor(snapshot_path: Path) -> float:
    config_path = snapshot_path / "vae" / "config.json"
    if not config_path.exists():
        return 1.0
    return float(json.loads(config_path.read_text()).get("scaling_factor", 1.0))


def _require_local_snapshot(snapshot: str | Path | None) -> Path:
    if snapshot is None:
        raise ValueError("MLX generation requires a local snapshot path")
    text = str(snapshot)
    if text.startswith(("http://", "https://", "hf://")) or (
        "/" in text and not text.startswith(("/", "./", "../")) and not Path(text).exists()
    ):
        raise ValueError("MLX generation requires a local snapshot path")
    path = Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024
