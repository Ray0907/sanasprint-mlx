from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class SelectedTensor:
    array: object
    source_dtype: str
    decoded_dtype: str
    source_shape: list[int]


DTYPE_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
}


def read_selected_tensors(path: str | Path, keys: list[str]) -> dict[str, SelectedTensor]:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        file_size = file_path.stat().st_size
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"safetensors header is truncated: {file_path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > file_size - 8:
            raise ValueError(f"safetensors header extends beyond file bounds: {file_path}")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"safetensors header is truncated: {file_path}")
        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(f"safetensors header is invalid JSON: {file_path}") from error

        data_start = 8 + header_size
        return {key: _read_tensor(handle, file_size, data_start, header, key) for key in keys}


def _read_tensor(handle, file_size: int, data_start: int, header: dict, key: str) -> SelectedTensor:
    if key not in header:
        raise KeyError(key)
    metadata = header[key]
    source_dtype = metadata.get("dtype")
    if source_dtype not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype for {key}: {source_dtype}")
    shape = metadata.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"shape is missing or invalid for {key}")
    offsets = metadata.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"data_offsets are missing or invalid for {key}")
    start, end = [int(offset) for offset in offsets]
    absolute_start = data_start + start
    absolute_end = data_start + end
    expected_nbytes = math.prod(int(dim) for dim in shape) * DTYPE_SIZES[source_dtype]
    if start < 0 or end < start or absolute_end > file_size or end - start != expected_nbytes:
        raise ValueError(f"data offsets are invalid for {key}: {offsets}")

    handle.seek(absolute_start)
    raw = handle.read(end - start)
    if len(raw) != expected_nbytes:
        raise ValueError(f"tensor data is truncated for {key}")
    decoded = _decode_tensor(raw, source_dtype, shape)
    return SelectedTensor(
        array=mx.array(decoded),
        source_dtype=source_dtype,
        decoded_dtype=str(decoded.dtype),
        source_shape=[int(dim) for dim in shape],
    )


def _decode_tensor(raw: bytes, source_dtype: str, shape: list[int]):
    if source_dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if source_dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).copy()
    if source_dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (words << 16).view(np.float32).reshape(shape).copy()
    raise ValueError(f"unsupported dtype: {source_dtype}")
