from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass
class TensorMetadata:
    file: str
    array_name: str
    shape: list[int]
    dtype: str
    sha256: str


@dataclass
class FixtureManifest:
    schema_version: int
    fixture_tier: int
    model_repo: str
    model_revision: str
    diffusers_version: str
    diffusers_commit: str
    transformers_version: str
    mlx_version: str
    python_version: str
    dtype: str
    seed: int
    height: int
    width: int
    num_inference_steps: int
    prompt: str
    tensor_files: list[str]
    tensor_hashes: dict[str, str]
    tensor_metadata: dict[str, TensorMetadata]
    config_files: list[str] = field(default_factory=list)
    config_hashes: dict[str, str] = field(default_factory=dict)
    model_weight_files: list[str] = field(default_factory=list)
    model_weight_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixtureManifest":
        tensor_metadata = {
            name: TensorMetadata(**metadata)
            for name, metadata in data.get("tensor_metadata", {}).items()
        }
        return cls(
            **{
                **data,
                "tensor_metadata": tensor_metadata,
            }
        )

    def to_json(self, path: str | Path) -> None:
        validate_manifest(self)
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "FixtureManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: FixtureManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("schema_version must be 1")

    required_strings = [
        "model_repo",
        "model_revision",
        "diffusers_version",
        "diffusers_commit",
        "transformers_version",
        "mlx_version",
        "python_version",
        "dtype",
        "prompt",
    ]
    for field_name in required_strings:
        if not getattr(manifest, field_name):
            raise ValueError(f"{field_name} is required")

    if manifest.fixture_tier < 0:
        raise ValueError("fixture_tier must be non-negative")
    if manifest.height <= 0 or manifest.width <= 0:
        raise ValueError("height and width must be positive")
    if manifest.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    for tensor_file in manifest.tensor_files:
        if tensor_file not in manifest.tensor_hashes:
            raise ValueError(f"missing tensor hash for {tensor_file}")

    for config_file in manifest.config_files:
        if config_file not in manifest.config_hashes:
            raise ValueError(f"missing config hash for {config_file}")

    for weight_file in manifest.model_weight_files:
        if weight_file not in manifest.model_weight_hashes:
            raise ValueError(f"missing model weight hash for {weight_file}")

    for tensor_name, metadata in manifest.tensor_metadata.items():
        if metadata.array_name != tensor_name:
            raise ValueError(f"tensor metadata key {tensor_name} must match array name")
        if metadata.file not in manifest.tensor_files:
            raise ValueError(f"tensor metadata references unknown file {metadata.file}")
        if not metadata.shape:
            raise ValueError(f"tensor metadata {tensor_name} requires shape")
        if not metadata.dtype:
            raise ValueError(f"tensor metadata {tensor_name} requires dtype")
        if not metadata.sha256:
            raise ValueError(f"tensor metadata {tensor_name} requires sha256")
