from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


def load_decoder_weights_from_snapshot(snapshot_path: str | Path, *, mlx_dtype=None) -> dict:
    snapshot = Path(snapshot_path)
    tensor_infos = inspect_snapshot(snapshot)
    decoder_infos = sorted(
        [info for info in tensor_infos if info.component == "vae" and info.name.startswith("decoder.")],
        key=lambda item: item.name,
    )
    tensors = {}
    source_tensors = {}
    for info in decoder_infos:
        decoded = read_selected_tensors(snapshot / info.file, [info.name])[info.name]
        value = decoded.array
        if mlx_dtype is not None:
            value = value.astype(mlx_dtype)
        tensors[info.name] = mx.array(value)
        source_tensors[info.name] = {
            "source_key": info.name,
            "source_file": info.file,
            "source_dtype": decoded.source_dtype,
            "decoded_dtype": decoded.decoded_dtype,
            "source_shape": decoded.source_shape,
            "target_shape": [int(dim) for dim in tensors[info.name].shape],
            "final_dtype": str(tensors[info.name].dtype).removeprefix("mlx.core."),
        }

    return {
        "source": "real_weights" if tensors else "missing",
        "loaded_keys": {
            "count": len(tensors),
            "keys": list(tensors),
        },
        "source_tensors": source_tensors,
        "tensors": tensors,
    }
