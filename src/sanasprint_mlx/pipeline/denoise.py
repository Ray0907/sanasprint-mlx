from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from sanasprint_mlx.fixtures.manifest import FixtureManifest, file_sha256, validate_manifest
from sanasprint_mlx.pipeline.debug import build_step_telemetry
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.transformer.parity import compare_arrays
from sanasprint_mlx.transformer.real_parity import _build_fixture_contract_model, _decode_tensor


@dataclass(frozen=True)
class DenoisingLoopResult:
    latents: object
    debug_steps: list[dict] = field(default_factory=list)


def run_denoising_loop(
    *,
    transformer,
    scheduler: SCMScheduler,
    latents,
    prompt_embeds,
    prompt_attention_mask,
    num_inference_steps: int,
    timesteps=None,
    max_timesteps: float | None = 1.57080,
    intermediate_timesteps: float | None = 1.3,
    guidance_scale: float = 4.5,
    attention_kwargs: dict | None = None,
    debug: bool = False,
    memory_snapshot=None,
    noise_fn=None,
) -> DenoisingLoopResult:
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        timesteps=timesteps,
        max_timesteps=max_timesteps,
        intermediate_timesteps=intermediate_timesteps,
    )
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)

    latents = mx.array(latents, dtype=mx.float32) * scheduler.config.sigma_data
    prompt_embeds = mx.array(prompt_embeds)
    prompt_attention_mask = mx.array(prompt_attention_mask)
    batch = latents.shape[0]

    guidance = mx.full((batch,), guidance_scale, dtype=mx.float32)
    guidance = guidance * float(transformer.config.guidance_embeds_scale)

    debug_steps: list[dict] = []
    denoised = latents
    for step_index, raw_timestep in enumerate(np.array(scheduler.timesteps[:-1], dtype=np.float32)):
        timestep = mx.full((batch,), float(raw_timestep), dtype=mx.float32)
        latents_model_input = latents / scheduler.config.sigma_data

        scm_timestep = mx.sin(timestep) / (mx.cos(timestep) + mx.sin(timestep))
        scm_timestep_expanded = scm_timestep.reshape((batch, 1, 1, 1))
        scale = mx.sqrt(scm_timestep_expanded**2 + (1.0 - scm_timestep_expanded) ** 2)
        latent_model_input = latents_model_input * scale

        noise_pred = transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            guidance=guidance,
            timestep=scm_timestep,
            return_dict=False,
            attention_kwargs=attention_kwargs,
        )[0]
        noise_pred = mx.array(noise_pred, dtype=mx.float32)
        noise_pred = (
            (1.0 - 2.0 * scm_timestep_expanded) * latent_model_input
            + (1.0 - 2.0 * scm_timestep_expanded + 2.0 * scm_timestep_expanded**2) * noise_pred
        ) / scale
        noise_pred = noise_pred.astype(mx.float32) * scheduler.config.sigma_data

        latents, denoised = scheduler.step(
            noise_pred,
            timestep,
            latents,
            noise_fn=noise_fn,
            return_dict=False,
        )
        if debug:
            debug_steps.append(
                build_step_telemetry(
                    step_index=step_index,
                    timestep=timestep,
                    scm_timestep=scm_timestep,
                    latents=latents,
                    memory_snapshot=memory_snapshot,
                )
            )

    return DenoisingLoopResult(latents=denoised / scheduler.config.sigma_data, debug_steps=debug_steps)


def run_real_loop_fixture_parity(fixture_path: str | Path, *, transformer=None, scheduler: SCMScheduler | None = None) -> dict:
    root = Path(fixture_path)
    if root.is_file() and root.name == "manifest.json":
        root = root.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fixture manifest: {manifest_path}")
    manifest = FixtureManifest.from_json(manifest_path)
    validate_manifest(manifest)
    tensor_file_name = manifest.tensor_files[0]
    tensor_file = root / tensor_file_name
    if file_sha256(tensor_file) != manifest.tensor_hashes.get(tensor_file_name):
        raise ValueError(f"fixture tensor hash mismatch: {tensor_file_name}")

    arrays = _load_loop_fixture_arrays(tensor_file, manifest)
    active_transformer = transformer or _build_fixture_contract_model(arrays)
    active_scheduler = scheduler or SCMScheduler()
    result = run_denoising_loop(
        transformer=active_transformer,
        scheduler=active_scheduler,
        latents=arrays["latents"],
        prompt_embeds=arrays["prompt_embeds"],
        prompt_attention_mask=arrays["prompt_attention_mask"],
        num_inference_steps=len(arrays["timesteps"]) - 1,
        timesteps=arrays["timesteps"].tolist(),
        max_timesteps=None,
        intermediate_timesteps=None,
    )
    expected = arrays.get("expected_final_latents", arrays["expected_noise_pred"])
    report = compare_arrays(np.array(result.latents), expected)
    report.update({"fixture_path": str(root), "fixture_tier": manifest.fixture_tier})
    return report


def _load_loop_fixture_arrays(tensor_file: Path, manifest: FixtureManifest) -> dict[str, np.ndarray]:
    required = {"prompt_embeds", "prompt_attention_mask", "latents", "timesteps"}
    with np.load(tensor_file) as npz:
        missing = sorted(required - set(npz.files))
        if missing:
            raise ValueError(f"fixture tensor file is missing: {', '.join(missing)}")
        arrays = {}
        for name in set(npz.files) & (required | {"expected_final_latents", "expected_noise_pred"}):
            metadata = manifest.tensor_metadata.get(name)
            arrays[name] = _decode_tensor(npz[name], metadata.dtype if metadata else "")
        return arrays
