from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Protocol

from safetensors import safe_open


@dataclass(frozen=True)
class TensorInfo:
    file: str
    name: str
    shape: list[int]
    dtype: str
    parameter_count: int
    component: str


class SafetensorsReader(Protocol):
    def keys(self): ...

    def get_slice(self, key): ...


def inspect_snapshot(snapshot_path: str | Path) -> list[TensorInfo]:
    snapshot = Path(snapshot_path)
    infos: list[TensorInfo] = []
    for path in sorted(snapshot.rglob("*.safetensors")):
        infos.extend(inspect_safetensors_file(path, relative_to=snapshot))
    return sorted(infos, key=lambda info: (info.component, info.file, info.name))


def inspect_safetensors_file(path: str | Path, *, relative_to: str | Path | None = None) -> list[TensorInfo]:
    file_path = Path(path)
    file_name = str(file_path.relative_to(relative_to)) if relative_to is not None else str(file_path)
    with safe_open(file_path, framework="numpy") as reader:
        return inspect_with_reader(reader, file=file_name)


def inspect_with_reader(reader: SafetensorsReader, *, file: str) -> list[TensorInfo]:
    infos = []
    for key in sorted(reader.keys()):
        tensor_slice = reader.get_slice(key)
        shape = _shape(tensor_slice)
        dtype = str(tensor_slice.get_dtype()) if hasattr(tensor_slice, "get_dtype") else _dtype_name(tensor_slice)
        infos.append(
            TensorInfo(
                file=file,
                name=key,
                shape=shape,
                dtype=dtype,
                parameter_count=int(prod(shape)) if shape else 1,
                component=classify_component(file, key),
            )
        )
    return infos


def classify_component(file: str, key: str) -> str:
    parts = set(Path(file).parts)
    if "text_encoder" in parts or key.startswith("text_encoder."):
        return "text_encoder"
    if "transformer" in parts or key.startswith("transformer."):
        return "transformer"
    if "vae" in parts or key.startswith(("vae.", "decoder.", "encoder.")):
        return "vae"
    if "scheduler" in parts or key.startswith("scheduler."):
        return "scheduler"
    return "unknown"


def _dtype_name(array) -> str:
    dtype = getattr(array, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def _shape(tensor_slice) -> list[int]:
    if hasattr(tensor_slice, "get_shape"):
        return [int(dim) for dim in tensor_slice.get_shape()]
    return [int(dim) for dim in tensor_slice.shape]
