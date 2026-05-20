from __future__ import annotations

from pathlib import Path

from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


def load_block_attention_weights_from_snapshot(
    block,
    snapshot_path: str | Path,
    *,
    block_index: int = 0,
    mlx_dtype=None,
    strict: bool = True,
) -> dict:
    snapshot = Path(snapshot_path)
    source_by_target = _source_key_by_target(block.parameter_shapes(), block_index=block_index)
    tensor_infos = inspect_snapshot(snapshot)
    infos_by_key = {
        info.name: info
        for info in tensor_infos
        if info.component == "transformer" and info.name in set(source_by_target.values())
    }
    missing = [target for target, source in source_by_target.items() if source not in infos_by_key]
    if strict and missing:
        raise KeyError(missing[0])

    source_tensors = {}
    source_metadata = {}
    for target_key, source_key in source_by_target.items():
        info = infos_by_key.get(source_key)
        if info is None:
            continue
        decoded = read_selected_tensors(snapshot / info.file, [source_key])[source_key]
        value = decoded.array
        if mlx_dtype is not None:
            value = value.astype(mlx_dtype)
        source_tensors[target_key] = value
        source_metadata[target_key] = {
            "source_key": source_key,
            "target_key": target_key,
            "source_file": info.file,
            "source_dtype": decoded.source_dtype,
            "decoded_dtype": decoded.decoded_dtype,
            "source_shape": decoded.source_shape,
        }

    block.load_parameters(source_tensors, strict=strict)
    parameters = block.parameters()
    loaded_keys = [key for key in block.parameter_shapes() if key in source_tensors]
    return {
        "block_index": block_index,
        "loaded_keys": loaded_keys,
        "source_tensors": _source_tensor_diagnostics(source_metadata, parameters),
    }


def _source_key_by_target(parameter_shapes: dict[str, tuple[int, ...]], *, block_index: int) -> dict[str, str]:
    target_prefix = f"mlx_transformer.transformer_blocks.{block_index}."
    source_prefix = f"transformer_blocks.{block_index}."
    return {key: source_prefix + key.removeprefix(target_prefix) for key in parameter_shapes}


def _source_tensor_diagnostics(source_metadata: dict, parameters: dict) -> dict:
    diagnostics = {}
    for key, metadata in source_metadata.items():
        value = dict(metadata)
        parameter = parameters[key]
        value["target_shape"] = [int(dim) for dim in parameter.shape]
        value["final_dtype"] = str(parameter.dtype).removeprefix("mlx.core.")
        diagnostics[key] = value
    return diagnostics
