from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

from sanasprint_mlx.fixtures.manifest import (
    FixtureManifest,
    TensorMetadata,
    file_sha256,
)


MODEL_REPO = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
TENSOR_FILE = "fixture.npz"
TENSOR_NAMES = [
    "prompt_embeds",
    "prompt_attention_mask",
    "latents",
    "timesteps",
    "timestep",
    "guidance",
    "expected_noise_pred",
]


def generate_synthetic_fixture(
    output_dir: str | Path,
    *,
    seed: int,
    height: int = 8,
    width: int = 8,
    num_inference_steps: int = 2,
    prompt: str = "synthetic prompt",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    arrays = _synthetic_arrays(seed, height, width, num_inference_steps)
    tensor_path = output / TENSOR_FILE
    _write_deterministic_npz(tensor_path, arrays)
    tensor_file_hash = file_sha256(tensor_path)

    tensor_metadata = {
        name: TensorMetadata(
            file=TENSOR_FILE,
            array_name=name,
            shape=list(array.shape),
            dtype=str(array.dtype),
            sha256=_array_sha256(array),
        )
        for name, array in arrays.items()
    }

    manifest = FixtureManifest(
        schema_version=1,
        fixture_tier=0,
        model_repo=MODEL_REPO,
        model_revision="synthetic",
        diffusers_version="not-used",
        diffusers_commit="not-used",
        transformers_version="not-used",
        mlx_version="not-used",
        python_version=sys.version.split()[0],
        dtype="float32",
        seed=seed,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        tensor_files=[TENSOR_FILE],
        tensor_hashes={TENSOR_FILE: tensor_file_hash},
        tensor_metadata=tensor_metadata,
        config_files=[],
        config_hashes={},
        model_weight_files=[],
        model_weight_hashes={},
        notes=["synthetic Tier 0 fixture; no model weights used"],
    )

    manifest_path = output / "manifest.json"
    manifest.to_json(manifest_path)
    return manifest_path


def _synthetic_arrays(
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent_height = max(height // 2, 1)
    latent_width = max(width // 2, 1)

    latents = rng.standard_normal((1, 4, latent_height, latent_width), dtype=np.float32)
    expected_noise_pred = rng.standard_normal(latents.shape, dtype=np.float32)

    return {
        "prompt_embeds": rng.standard_normal((1, 4, 8), dtype=np.float32),
        "prompt_attention_mask": np.ones((1, 4), dtype=np.int64),
        "latents": latents,
        "timesteps": np.linspace(1.0, 0.0, num_inference_steps, dtype=np.float32),
        "timestep": np.array([0.5], dtype=np.float32),
        "guidance": np.array([4.5], dtype=np.float32),
        "expected_noise_pred": expected_noise_pred,
    }


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in TENSOR_NAMES:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, buffer.getvalue())
