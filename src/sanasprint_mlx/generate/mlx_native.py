from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig
from sanasprint_mlx.autoencoder.decode import AutoencoderDCDecode
from sanasprint_mlx.autoencoder.mlx_decoder import MLXAutoencoderDCDecoder
from sanasprint_mlx.memory.mlx_cache import mlx_cache_limit, trim_mlx_cache
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
    tiled_decode: bool = False,
    allow_download: bool = False,
) -> dict:
    return run_mlx_batch_generation(
        prompt=prompt,
        prompt_cache=prompt_cache,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        outputs=[output],
        snapshot=snapshot,
        mlx_dtype=mlx_dtype,
        tiled_decode=tiled_decode,
        allow_download=allow_download,
    )[0]


def run_mlx_batch_generation(
    *,
    prompt: str | None = None,
    prompt_cache: str | Path | None = None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    outputs: list[str | Path],
    snapshot: str | Path | None,
    mlx_dtype: str = "bfloat16",
    tiled_decode: bool = False,
    allow_download: bool = False,
) -> list[dict]:
    if not outputs:
        raise ValueError("batch generation requires at least one output")
    with mlx_cache_limit(0):
        return _run_mlx_batch_generation_uncached(
            prompt=prompt,
            prompt_cache=prompt_cache,
            height=height,
            width=width,
            steps=steps,
            seed=seed,
            outputs=outputs,
            snapshot=snapshot,
            mlx_dtype=mlx_dtype,
            tiled_decode=tiled_decode,
            allow_download=allow_download,
        )


def _run_mlx_batch_generation_uncached(
    *,
    prompt: str | None,
    prompt_cache: str | Path | None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    outputs: list[str | Path],
    snapshot: str | Path | None,
    mlx_dtype: str,
    tiled_decode: bool,
    allow_download: bool,
) -> list[dict]:
    start = time.perf_counter()
    snapshot_path = _resolve_snapshot(snapshot, allow_download=allow_download)
    prompt_embeds, prompt_attention_mask, prompt_source = _prompt_inputs(
        prompt=prompt,
        prompt_cache=prompt_cache,
        snapshot=snapshot_path,
    )
    summary = summarize_transformer_config(load_transformer_config(snapshot_path))
    sample_size = max(height // 32, 1)
    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot_path,
        sample_size=sample_size,
        block_count=None,
        dtype=mlx_dtype,
    )
    denoised_items = []
    for index, output in enumerate(outputs):
        item_seed = seed + index
        latents = _latents(
            channels=summary.in_channels,
            height=sample_size,
            width=max(width // 32, 1),
            seed=item_seed,
        )
        result = run_denoising_loop(
            transformer=transformer,
            scheduler=SCMScheduler(),
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_inference_steps=steps,
        )
        denoised_latents = np.array(result.latents, dtype=np.float32)
        denoised_items.append((Path(output), item_seed, denoised_latents, [int(dim) for dim in denoised_latents.shape]))
    loaded_keys = transformer.weight_report["loaded_keys"]
    del transformer, result
    _release_mlx_memory()
    decoder = MLXAutoencoderDCDecoder.from_snapshot(snapshot_path, dtype=mlx_dtype)
    scale = _scaling_factor(snapshot_path)
    reports = []
    for output_path, item_seed, denoised_latents, latents_shape in denoised_items:
        decoded = _decode_latents(
            decoder,
            denoised_latents / scale,
            tiled_decode=tiled_decode,
        )
        image = _postprocess(decoded[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        reports.append(
            {
                "mode": "mlx_transformer_mlx_decode",
                "output": str(output_path),
                "model": str(snapshot_path),
                "height": height,
                "width": width,
                "steps": steps,
                "seed": item_seed,
                "mlx_dtype": mlx_dtype,
                "prompt_source": prompt_source,
                "decode_mode": "tiled_mlx_decode" if tiled_decode else "mlx_decode",
                "latents_shape": latents_shape,
                "loaded_keys": loaded_keys,
                "runtime": {"wall_time_seconds": time.perf_counter() - start},
                "memory": {"max_rss_bytes": _max_rss_bytes()},
            }
        )
    return reports


def _decode_latents(decoder: MLXAutoencoderDCDecoder, latents: np.ndarray, *, tiled_decode: bool) -> np.ndarray:
    if not tiled_decode:
        return np.array(decoder.decode(latents))
    wrapped = AutoencoderDCDecode(
        _DecoderCallable(decoder),
        AutoencoderDecodeConfig(use_tiling=True),
    )
    return np.asarray(wrapped.decode(latents, return_dict=False)[0])


class _DecoderCallable:
    def __init__(self, decoder: MLXAutoencoderDCDecoder):
        self.decoder = decoder

    def __call__(self, latents):
        return np.array(self.decoder.decode(np.asarray(latents, dtype=np.float32)))


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


def _resolve_snapshot(snapshot: str | Path | None, *, allow_download: bool = False) -> Path:
    if snapshot is None:
        raise ValueError("MLX generation requires a local snapshot path")
    text = str(snapshot)
    if _looks_remote_snapshot(text):
        if allow_download:
            return Path(snapshot_download(text.removeprefix("hf://")))
        raise ValueError("MLX generation requires a local snapshot path")
    path = Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _looks_remote_snapshot(text: str) -> bool:
    return text.startswith(("http://", "https://", "hf://")) or (
        "/" in text and not text.startswith(("/", "./", "../")) and not Path(text).exists()
    )


def snapshot_download(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download as download
    except ImportError as error:
        raise ImportError("native MLX remote snapshots require huggingface_hub") from error
    return download(repo_id)


def _release_mlx_memory() -> None:
    gc.collect()
    trim_mlx_cache()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024
