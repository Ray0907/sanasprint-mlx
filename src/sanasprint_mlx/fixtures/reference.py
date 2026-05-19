from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from sanasprint_mlx.fixtures.manifest import (
    FixtureManifest,
    TensorMetadata,
    file_sha256,
)
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO, _array_sha256, _write_deterministic_npz


EXPECTED_TENSOR_NAMES = (
    "prompt_embeds",
    "prompt_attention_mask",
    "latents",
    "timesteps",
    "timestep",
    "guidance",
    "expected_noise_pred",
)

REFERENCE_DEPENDENCIES = ("torch", "diffusers", "transformers")
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
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


def check_reference_dependencies(packages: Iterable[str] = REFERENCE_DEPENDENCIES) -> list[str]:
    return [package for package in packages if importlib.util.find_spec(package) is None]


def generate_reference_fixture(
    output_dir: str | Path,
    *,
    model_repo: str = MODEL_REPO,
    revision: str,
    allow_download: bool,
    seed: int = 7,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 2,
    prompt: str = "synthetic prompt",
    torch_dtype: str = "bfloat16",
    diffusers_commit: str | None = None,
) -> Path:
    if not allow_download:
        raise PermissionError("reference fixture generation requires --allow-download")
    validate_pinned_revision(revision)

    missing = check_reference_dependencies()
    if missing:
        raise RuntimeError(
            "missing reference dependencies: "
            + ", ".join(missing)
            + '; install with `python3 -m pip install -e ".[reference]"`'
        )

    # Heavy reference dependencies are intentionally imported only inside this opt-in path.
    import torch
    from diffusers import SanaSprintPipeline
    from huggingface_hub import snapshot_download

    resolved_diffusers_commit = resolve_diffusers_commit(diffusers_commit, _package_commit("diffusers"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    local_model_dir = Path(
        snapshot_download(
            repo_id=model_repo,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer.*",
                "*.model",
                "*.txt",
            ],
        )
    )

    dtype = getattr(torch, torch_dtype)
    pipe = SanaSprintPipeline.from_pretrained(str(local_model_dir), torch_dtype=dtype)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(device)

    arrays, reference_dtypes = _capture_reference_tensors(
        pipe=pipe,
        torch=torch,
        device=device,
        dtype=dtype,
        seed=seed,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
    )

    tensor_path = output / "fixture.npz"
    _write_deterministic_npz(tensor_path, {name: arrays[name] for name in EXPECTED_TENSOR_NAMES})

    config_files, config_hashes, model_weight_files, model_weight_hashes = hash_reproduction_files(local_model_dir)
    tensor_file_hash = file_sha256(tensor_path)

    tensor_metadata = {
        name: TensorMetadata(
            file="fixture.npz",
            array_name=name,
            shape=list(array.shape),
            dtype=reference_dtypes[name],
            sha256=_array_sha256(array),
        )
        for name, array in arrays.items()
    }

    manifest = FixtureManifest(
        schema_version=1,
        fixture_tier=1,
        model_repo=model_repo,
        model_revision=revision,
        diffusers_version=_package_version("diffusers"),
        diffusers_commit=resolved_diffusers_commit,
        transformers_version=_package_version("transformers"),
        mlx_version=_package_version("mlx"),
        python_version=sys.version.split()[0],
        dtype=torch_dtype,
        seed=seed,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        tensor_files=["fixture.npz"],
        tensor_hashes={"fixture.npz": tensor_file_hash},
        tensor_metadata=tensor_metadata,
        config_files=config_files,
        config_hashes=config_hashes,
        model_weight_files=model_weight_files,
        model_weight_hashes=model_weight_hashes,
        notes=["opt-in real Diffusers reference fixture"],
    )
    manifest_path = output / "manifest.json"
    manifest.to_json(manifest_path)
    return manifest_path


def _capture_reference_tensors(
    *,
    pipe,
    torch,
    device: str,
    dtype,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    prompt: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent_channels = int(pipe.transformer.config.in_channels)
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 32))
    latent_shape = (
        1,
        latent_channels,
        max(height // vae_scale_factor, 1),
        max(width // vae_scale_factor, 1),
    )
    latents = torch.randn(latent_shape, generator=generator, dtype=dtype).to(device)

    prompt_embeds, prompt_attention_mask = _encode_prompt_tensors(pipe, device, prompt)

    captured: dict[str, np.ndarray] = {}
    reference_dtypes: dict[str, str] = {}
    _capture_tensor(captured, reference_dtypes, "prompt_embeds", prompt_embeds)
    _capture_tensor(captured, reference_dtypes, "prompt_attention_mask", prompt_attention_mask)

    original_forward = pipe.transformer.forward

    def forward_with_capture(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        sample = output.sample if hasattr(output, "sample") else output[0] if isinstance(output, tuple) else output
        if "expected_noise_pred" not in captured:
            latent_input = args[0] if args else kwargs["hidden_states"]
            _capture_tensor(captured, reference_dtypes, "latents", latent_input)
            _capture_tensor(captured, reference_dtypes, "timestep", kwargs["timestep"])
            _capture_tensor(captured, reference_dtypes, "guidance", kwargs["guidance"])
            _capture_tensor(captured, reference_dtypes, "expected_noise_pred", sample)
        return output

    pipe.transformer.forward = forward_with_capture
    try:
        pipe(
            prompt=None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            output_type="latent",
        )
    finally:
        pipe.transformer.forward = original_forward

    if hasattr(pipe.scheduler, "timesteps"):
        _capture_tensor(captured, reference_dtypes, "timesteps", pipe.scheduler.timesteps)

    missing = [name for name in EXPECTED_TENSOR_NAMES if name not in captured]
    if missing:
        raise RuntimeError(f"reference capture did not produce tensors: {', '.join(missing)}")

    return (
        {name: captured[name] for name in EXPECTED_TENSOR_NAMES},
        {name: reference_dtypes[name] for name in EXPECTED_TENSOR_NAMES},
    )


def _encode_prompt_tensors(pipe, device: str, prompt: str):
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("pipeline does not expose encode_prompt")

    try:
        return pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=DEFAULT_COMPLEX_HUMAN_INSTRUCTION,
        )
    except TypeError:
        return pipe.encode_prompt(prompt, 1, device, None, None, False, 300, DEFAULT_COMPLEX_HUMAN_INSTRUCTION)


def _torch_to_numpy(tensor) -> np.ndarray:
    tensor = tensor.detach().to("cpu")
    if str(tensor.dtype) == "torch.bfloat16":
        import torch

        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


def _capture_tensor(
    captured: dict[str, np.ndarray],
    reference_dtypes: dict[str, str],
    name: str,
    tensor,
) -> None:
    captured[name] = _torch_to_numpy(tensor)
    reference_dtypes[name] = str(tensor.dtype)


def validate_pinned_revision(revision: str) -> None:
    if not PINNED_REVISION_RE.match(revision):
        raise ValueError("model revision must be a pinned 40-character commit SHA")


def resolve_diffusers_commit(explicit_commit: str | None, discovered_commit: str) -> str:
    commit = explicit_commit or discovered_commit
    if not commit or commit == "unknown" or not PINNED_REVISION_RE.match(commit):
        raise ValueError("diffusers commit must be a pinned 40-character commit SHA")
    return commit


def hash_reproduction_files(root: Path) -> tuple[list[str], dict[str, str], list[str], dict[str, str]]:
    root = Path(root)
    weight_files = _collect_relative_files(root, ("*.safetensors",))
    config_files = _collect_relative_files(
        root,
        (
            "*.json",
            "*.model",
            "*.txt",
            "tokenizer.*",
            "*.vocab",
            "vocab.*",
            "merges.*",
            "spiece.model",
        ),
        exclude=set(weight_files),
    )
    return (
        config_files,
        {relative: file_sha256(root / relative) for relative in config_files},
        weight_files,
        {relative: file_sha256(root / relative) for relative in weight_files},
    )


def _collect_relative_files(root: Path, patterns: tuple[str, ...], exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    files: set[str] = set()
    for pattern in patterns:
        files.update(str(path.relative_to(root)) for path in root.rglob(pattern) if path.is_file())
    return sorted(files - excluded)


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _package_commit(package: str) -> str:
    try:
        module = __import__(package)
    except ImportError:
        return "not-installed"
    return str(getattr(module, "__commit__", "unknown"))
