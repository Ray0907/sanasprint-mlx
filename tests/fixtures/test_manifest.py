import json

import pytest

from sanasprint_mlx.fixtures.manifest import (
    FixtureManifest,
    TensorMetadata,
    file_sha256,
    validate_manifest,
)


def make_manifest(tmp_path):
    tensor_file = tmp_path / "fixture.npz"
    tensor_file.write_bytes(b"tensor-bytes")
    config_file = tmp_path / "config.json"
    config_file.write_text('{"model": "synthetic"}')
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"weight-bytes")

    return FixtureManifest(
        schema_version=1,
        fixture_tier=0,
        model_repo="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        model_revision="synthetic",
        diffusers_version="not-used",
        diffusers_commit="not-used",
        transformers_version="not-used",
        mlx_version="not-used",
        python_version="3.14.3",
        dtype="float32",
        seed=7,
        height=8,
        width=8,
        num_inference_steps=2,
        prompt="synthetic prompt",
        tensor_files=["fixture.npz"],
        tensor_hashes={"fixture.npz": file_sha256(tensor_file)},
        tensor_metadata={
            "latents": TensorMetadata(
                file="fixture.npz",
                array_name="latents",
                shape=[1, 4, 4, 4],
                dtype="float32",
                sha256=file_sha256(tensor_file),
            )
        },
        config_files=["config.json"],
        config_hashes={"config.json": file_sha256(config_file)},
        model_weight_files=["model.safetensors"],
        model_weight_hashes={"model.safetensors": file_sha256(weight_file)},
        notes=["unit test"],
    )


def test_manifest_round_trips_json(tmp_path):
    manifest = make_manifest(tmp_path)
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = FixtureManifest.from_json(path)

    assert loaded == manifest
    assert json.loads(path.read_text())["schema_version"] == 1


def test_manifest_requires_pinning_fields(tmp_path):
    manifest = make_manifest(tmp_path)
    manifest.model_revision = ""

    with pytest.raises(ValueError, match="model_revision"):
        validate_manifest(manifest)


def test_manifest_hashes_file_contents(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")

    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_records_tensor_metadata(tmp_path):
    manifest = make_manifest(tmp_path)

    validate_manifest(manifest)

    latents = manifest.tensor_metadata["latents"]
    assert latents.shape == [1, 4, 4, 4]
    assert latents.dtype == "float32"
    assert latents.array_name == "latents"


def test_manifest_records_model_weight_hashes(tmp_path):
    manifest = make_manifest(tmp_path)

    validate_manifest(manifest)

    assert manifest.model_weight_files == ["model.safetensors"]
    assert manifest.model_weight_hashes["model.safetensors"]
