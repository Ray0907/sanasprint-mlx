import numpy as np
from safetensors.numpy import save_file

from sanasprint_mlx.weights.inspect import (
    TensorInfo,
    classify_component,
    inspect_safetensors_file,
    inspect_snapshot,
    inspect_with_reader,
)


class MetadataOnlyReader:
    def __init__(self):
        self._keys = ["transformer.patch_embed.proj.weight"]

    def keys(self):
        return self._keys

    def get_slice(self, key):
        assert key in self._keys
        return np.zeros((2, 3), dtype=np.float32)

    def get_tensor(self, key):
        raise AssertionError("get_tensor must not be called")


def test_inspect_safetensors_reads_names_shapes_and_dtypes(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "transformer.patch_embed.proj.weight": np.zeros((4, 3), dtype=np.float32),
            "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float16),
        },
        path,
    )

    infos = inspect_safetensors_file(path, relative_to=tmp_path)

    assert infos == [
        TensorInfo(
            file="model.safetensors",
            name="transformer.patch_embed.proj.bias",
            shape=[4],
            dtype="F16",
            parameter_count=4,
            component="transformer",
        ),
        TensorInfo(
            file="model.safetensors",
            name="transformer.patch_embed.proj.weight",
            shape=[4, 3],
            dtype="F32",
            parameter_count=12,
            component="transformer",
        ),
    ]


def test_inspect_snapshot_collects_multiple_safetensors_files(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "text_encoder").mkdir()
    save_file({"transformer.block.weight": np.zeros((2, 2), dtype=np.float32)}, tmp_path / "transformer" / "model.safetensors")
    save_file({"text_encoder.embed.weight": np.zeros((3, 2), dtype=np.float32)}, tmp_path / "text_encoder" / "model.safetensors")

    infos = inspect_snapshot(tmp_path)

    assert [info.component for info in infos] == ["text_encoder", "transformer"]
    assert sum(info.parameter_count for info in infos) == 10


def test_inspect_snapshot_does_not_read_tensor_values():
    infos = inspect_with_reader(MetadataOnlyReader(), file="model.safetensors")

    assert infos[0].shape == [2, 3]
    assert infos[0].parameter_count == 6


def test_classify_component_from_path_and_key():
    assert classify_component("text_encoder/model.safetensors", "embed.weight") == "text_encoder"
    assert classify_component("transformer/model.safetensors", "patch.weight") == "transformer"
    assert classify_component("vae/diffusion_pytorch_model.safetensors", "decoder.weight") == "vae"
    assert classify_component("unknown/model.safetensors", "scheduler.foo") == "scheduler"
    assert classify_component("misc/model.safetensors", "foo") == "unknown"
