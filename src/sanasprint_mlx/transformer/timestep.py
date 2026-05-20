from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.primitives.feed_forward import linear, silu
from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


class SanaTimestepGuidanceEmbedding:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self._parameters = {
            key: mx.array(value)
            for key, value in _initial_timestep_guidance_parameters(hidden_size).items()
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

    def __call__(self, *, timestep, guidance, hidden_dtype=None):
        timestep_emb = self._timestep_embedder("timestep_embedder", timestep, hidden_dtype=hidden_dtype)
        guidance_emb = self._timestep_embedder("guidance_embedder", guidance, hidden_dtype=hidden_dtype)
        conditioning = timestep_emb + guidance_emb
        prefix = "mlx_transformer.time_embed"
        modulation = linear(
            silu(conditioning),
            self._parameters[f"{prefix}.linear.weight"],
            self._parameters[f"{prefix}.linear.bias"],
        )
        return modulation, conditioning

    def _timestep_embedder(self, name: str, values, *, hidden_dtype=None):
        prefix = f"mlx_transformer.time_embed.{name}"
        projection = _diffusers_timestep_projection(values, dim=256)
        if hidden_dtype is not None:
            projection = projection.astype(hidden_dtype)
        hidden = linear(
            projection,
            self._parameters[f"{prefix}.linear_1.weight"],
            self._parameters[f"{prefix}.linear_1.bias"],
        )
        hidden = silu(hidden)
        return linear(
            hidden,
            self._parameters[f"{prefix}.linear_2.weight"],
            self._parameters[f"{prefix}.linear_2.bias"],
        )


def load_timestep_guidance_weights_from_snapshot(
    embedding: SanaTimestepGuidanceEmbedding,
    snapshot_path: str | Path,
    *,
    mlx_dtype=None,
    strict: bool = True,
) -> dict:
    snapshot = Path(snapshot_path)
    source_by_target = _source_key_by_target(embedding.parameter_shapes())
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

    embedding.load_parameters(source_tensors, strict=strict)
    parameters = embedding.parameters()
    loaded_keys = [key for key in embedding.parameter_shapes() if key in source_tensors]
    return {
        "loaded_keys": loaded_keys,
        "source": embedding.source,
        "source_tensors": _source_tensor_diagnostics(source_metadata, parameters),
    }


def _initial_timestep_guidance_parameters(hidden_size: int) -> dict[str, object]:
    prefix = "mlx_transformer.time_embed"
    return {
        f"{prefix}.timestep_embedder.linear_1.weight": mx.zeros((hidden_size, 256)),
        f"{prefix}.timestep_embedder.linear_1.bias": mx.zeros((hidden_size,)),
        f"{prefix}.timestep_embedder.linear_2.weight": mx.zeros((hidden_size, hidden_size)),
        f"{prefix}.timestep_embedder.linear_2.bias": mx.zeros((hidden_size,)),
        f"{prefix}.guidance_embedder.linear_1.weight": mx.zeros((hidden_size, 256)),
        f"{prefix}.guidance_embedder.linear_1.bias": mx.zeros((hidden_size,)),
        f"{prefix}.guidance_embedder.linear_2.weight": mx.zeros((hidden_size, hidden_size)),
        f"{prefix}.guidance_embedder.linear_2.bias": mx.zeros((hidden_size,)),
        f"{prefix}.linear.weight": mx.zeros((6 * hidden_size, hidden_size)),
        f"{prefix}.linear.bias": mx.zeros((6 * hidden_size,)),
    }


def _diffusers_timestep_projection(values, *, dim: int, max_period: int = 10_000):
    if dim <= 0:
        raise ValueError("dim must be positive")
    values = mx.array(values).reshape(-1)
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / half
    args = values[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros((values.shape[0], 1), dtype=embedding.dtype)], axis=-1)
    return embedding


def _source_key_by_target(parameter_shapes: dict[str, tuple[int, ...]]) -> dict[str, str]:
    target_prefix = "mlx_transformer."
    return {key: key.removeprefix(target_prefix) for key in parameter_shapes}


def _source_tensor_diagnostics(source_metadata: dict, parameters: dict) -> dict:
    diagnostics = {}
    for key, metadata in source_metadata.items():
        value = dict(metadata)
        parameter = parameters[key]
        value["target_shape"] = [int(dim) for dim in parameter.shape]
        value["final_dtype"] = str(parameter.dtype).removeprefix("mlx.core.")
        diagnostics[key] = value
    return diagnostics
