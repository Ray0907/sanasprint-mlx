from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import SanaSprintPipeline

from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.text.cache import read_prompt_cache
from sanasprint_mlx.transformer.real_model import RealSanaTransformerDenoiser


DEFAULT_COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def run_mlx_reference_decode_generation(
    *,
    prompt: str | None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    output: str | Path,
    snapshot: str | Path | None,
    allow_download: bool,
    prompt_cache: str | Path | None = None,
    low_memory: bool = False,
    torch_dtype: str = "bfloat16",
    mlx_dtype: str = "bfloat16",
) -> dict:
    del allow_download
    snapshot_path = _require_local_snapshot(snapshot)
    output_path = Path(output)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = SanaSprintPipeline.from_pretrained(
        str(snapshot_path),
        torch_dtype=getattr(torch, torch_dtype),
        local_files_only=True,
    )
    if low_memory and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif low_memory and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe.to(device)

    prompt_embeds, prompt_attention_mask, prompt_source = _prompt_inputs(
        pipe,
        prompt=prompt,
        prompt_cache=prompt_cache,
        device=device,
    )
    latent_channels = int(pipe.transformer.config.in_channels)
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 32))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = pipe.prepare_latents(
        1,
        latent_channels,
        height,
        width,
        torch.float32,
        device,
        generator,
        None,
    )
    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot_path,
        sample_size=max(height // vae_scale_factor, 1),
        block_count=None,
        dtype=mlx_dtype,
    )
    result = run_denoising_loop(
        transformer=transformer,
        scheduler=SCMScheduler(),
        latents=latents.detach().float().cpu().numpy(),
        prompt_embeds=prompt_embeds.detach().float().cpu().numpy(),
        prompt_attention_mask=prompt_attention_mask.detach().cpu().numpy(),
        num_inference_steps=steps,
    )
    denoised = torch.from_numpy(np.array(result.latents)).to(device=device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(denoised / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "mode": "mlx_transformer_reference_decode",
        "output": str(output_path),
        "model": str(snapshot_path),
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "device": device,
        "low_memory": low_memory,
        "torch_dtype": torch_dtype,
        "mlx_dtype": mlx_dtype,
        "prompt_source": prompt_source,
        "latents_shape": [int(dim) for dim in np.array(result.latents).shape],
        "loaded_keys": transformer.weight_report["loaded_keys"],
    }


def _prompt_inputs(pipe, *, prompt: str | None, prompt_cache: str | Path | None, device: str):
    if prompt_cache is not None:
        arrays, _ = read_prompt_cache(prompt_cache)
        if "prompt_embeds" not in arrays or "prompt_attention_mask" not in arrays:
            raise ValueError("prompt cache requires prompt_embeds and prompt_attention_mask")
        return (
            torch.from_numpy(np.asarray(arrays["prompt_embeds"], dtype=np.float32)).to(device),
            torch.from_numpy(np.asarray(arrays["prompt_attention_mask"], dtype=np.int32)).to(device),
            "prompt_cache",
        )
    if not prompt:
        raise ValueError("MLX reference decode generation requires --prompt or --prompt-cache")
    prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        clean_caption=False,
        max_sequence_length=300,
        complex_human_instruction=DEFAULT_COMPLEX_HUMAN_INSTRUCTION,
    )
    return prompt_embeds, prompt_attention_mask, "reference_text_encoder"


def _require_local_snapshot(snapshot: str | Path | None) -> Path:
    if snapshot is None:
        raise ValueError("MLX reference decode generation requires a local snapshot path")
    text = str(snapshot)
    if text.startswith(("http://", "https://", "hf://")) or (
        "/" in text and not text.startswith(("/", "./", "../")) and not Path(text).exists()
    ):
        raise ValueError("MLX reference decode generation requires a local snapshot path")
    path = Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
