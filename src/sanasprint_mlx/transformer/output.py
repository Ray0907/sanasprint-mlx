from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


class SanaOutputNorm:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self._parameters = {
            "mlx_transformer.scale_shift_table": mx.zeros((2, hidden_size)),
        }
        self.source = "synthetic"

    def parameters(self) -> dict:
        return {key: mx.array(value) for key, value in self._parameters.items()}

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(value.shape) for key, value in self._parameters.items()}

    def load_parameters(self, parameters: dict, *, strict: bool = True) -> None:
        expected = self.parameter_shapes()
        unknown = [key for key in parameters if key not in expected]
        if unknown:
            raise KeyError(unknown[0])
        if strict:
            missing = [key for key in expected if key not in parameters]
            if missing:
                raise KeyError(missing[0])
        for key, value in parameters.items():
            tensor = mx.array(value)
            if tuple(tensor.shape) != expected[key]:
                raise ValueError(f"{key}: expected shape {expected[key]}, got {tuple(tensor.shape)}")
            self._parameters[key] = tensor
        if all(key in parameters for key in expected):
            self.source = "real_weights"

    def __call__(self, hidden_states, conditioning):
        hidden_states = _layer_norm(hidden_states)
        modulation = self._parameters["mlx_transformer.scale_shift_table"][None] + mx.array(conditioning)[:, None]
        shift, scale = mx.split(modulation, 2, axis=1)
        return hidden_states * (1 + scale) + shift


def load_output_norm_weights_from_snapshot(
    norm: SanaOutputNorm,
    snapshot_path: str | Path,
    *,
    mlx_dtype=None,
    strict: bool = True,
) -> dict:
    snapshot = Path(snapshot_path)
    source_by_target = {"mlx_transformer.scale_shift_table": "scale_shift_table"}
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

    norm.load_parameters(source_tensors, strict=strict)
    parameters = norm.parameters()
    loaded_keys = [key for key in norm.parameter_shapes() if key in source_tensors]
    return {
        "loaded_keys": loaded_keys,
        "source": norm.source,
        "source_tensors": _source_tensor_diagnostics(source_metadata, parameters),
    }


def _layer_norm(x, eps: float = 1e-6):
    x = mx.array(x)
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(variance + eps)


def _source_tensor_diagnostics(source_metadata: dict, parameters: dict) -> dict:
    diagnostics = {}
    for key, metadata in source_metadata.items():
        value = dict(metadata)
        parameter = parameters[key]
        value["target_shape"] = [int(dim) for dim in parameter.shape]
        value["final_dtype"] = str(parameter.dtype).removeprefix("mlx.core.")
        diagnostics[key] = value
    return diagnostics
