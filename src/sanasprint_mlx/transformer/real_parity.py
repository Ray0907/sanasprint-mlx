from __future__ import annotations

from pathlib import Path

import numpy as np

from sanasprint_mlx.fixtures.manifest import FixtureManifest, file_sha256, validate_manifest
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.parity import compare_arrays


REQUIRED_TENSORS = {
    "prompt_embeds",
    "prompt_attention_mask",
    "latents",
    "timestep",
    "guidance",
    "expected_noise_pred",
}


def run_real_fixture_parity(fixture_path: str | Path, model: SanaTransformerDenoiser | None = None) -> dict:
    root = Path(fixture_path)
    if root.is_file() and root.name == "manifest.json":
        root = root.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fixture manifest: {manifest_path}")

    manifest = FixtureManifest.from_json(manifest_path)
    validate_manifest(manifest)
    tensor_file = _resolve_tensor_file(root, manifest)
    _validate_tensor_file_hash(tensor_file, manifest)
    arrays = _load_required_tensors(tensor_file, manifest)

    denoiser = model or _build_fixture_contract_model(arrays)
    actual = denoiser(
        arrays["latents"],
        encoder_hidden_states=arrays["prompt_embeds"],
        encoder_attention_mask=arrays["prompt_attention_mask"],
        guidance=arrays["guidance"],
        timestep=arrays["timestep"],
        return_dict=False,
    )[0]
    report = compare_arrays(np.array(actual), arrays["expected_noise_pred"])
    report.update(
        {
            "fixture_path": str(root),
            "model_repo": manifest.model_repo,
            "model_revision": manifest.model_revision,
            "fixture_tier": manifest.fixture_tier,
        }
    )
    return report


def _resolve_tensor_file(root: Path, manifest: FixtureManifest) -> Path:
    if not manifest.tensor_files:
        raise ValueError("fixture manifest does not list tensor files")
    tensor_file = root / manifest.tensor_files[0]
    if not tensor_file.exists():
        raise FileNotFoundError(f"missing fixture tensor file: {tensor_file}")
    return tensor_file


def _validate_tensor_file_hash(tensor_file: Path, manifest: FixtureManifest) -> None:
    expected = manifest.tensor_hashes.get(tensor_file.name)
    if expected and file_sha256(tensor_file) != expected:
        raise ValueError(f"fixture tensor hash mismatch: {tensor_file.name}")


def _load_required_tensors(tensor_file: Path, manifest: FixtureManifest) -> dict[str, np.ndarray]:
    with np.load(tensor_file) as npz:
        missing = sorted(REQUIRED_TENSORS - set(npz.files))
        if missing:
            raise ValueError(f"fixture tensor file is missing: {', '.join(missing)}")
        return {
            name: _decode_tensor(npz[name], manifest.tensor_metadata.get(name).dtype if name in manifest.tensor_metadata else "")
            for name in sorted(REQUIRED_TENSORS)
        }


def _decode_tensor(array: np.ndarray, dtype_name: str) -> np.ndarray:
    if "bfloat16" in dtype_name and array.dtype == np.uint16:
        return (array.astype(np.uint32) << 16).view(np.float32)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    return array


def _build_fixture_contract_model(arrays: dict[str, np.ndarray]) -> SanaTransformerDenoiser:
    latents = arrays["latents"]
    prompt_embeds = arrays["prompt_embeds"]
    expected = arrays["expected_noise_pred"]
    if latents.ndim != 4:
        raise ValueError("latents must have NCHW shape")
    if prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must have batch, sequence, channels shape")
    hidden_size = int(prompt_embeds.shape[-1])
    config = SanaTransformerConfig(
        hidden_size=hidden_size,
        in_channels=int(latents.shape[1]),
        out_channels=int(expected.shape[1]),
        caption_channels=hidden_size,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=hidden_size,
        patch_size=1,
        sample_size=max(int(latents.shape[2]), int(latents.shape[3])),
        guidance_embeds_scale=1000.0,
    )
    return SanaTransformerDenoiser(config)
