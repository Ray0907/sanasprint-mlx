import json
import struct

import numpy as np
import pytest

from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


def write_raw_safetensors(path, entries):
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in entries:
        raw = bytes(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def bf16_bytes(values):
    float_bits = np.array(values, dtype="<f4").view("<u4")
    return ((float_bits >> 16).astype("<u2")).tobytes()


def test_selective_reader_decodes_bf16_values_from_little_endian_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    write_raw_safetensors(path, [("bf", "BF16", [3], bf16_bytes([1.0, -2.0, 0.5]))])

    tensors = read_selected_tensors(path, ["bf"])

    assert tensors["bf"].source_dtype == "BF16"
    assert tensors["bf"].decoded_dtype == "float32"
    np.testing.assert_array_equal(np.array(tensors["bf"].array), np.array([1.0, -2.0, 0.5], dtype=np.float32))


def test_selective_reader_reads_only_requested_f32_and_f16_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    write_raw_safetensors(
        path,
        [
            ("keep_f32", "F32", [2], np.array([1.0, 2.0], dtype="<f4").tobytes()),
            ("keep_f16", "F16", [2], np.array([3.0, 4.0], dtype="<f2").tobytes()),
            ("skip", "F32", [1], np.array([99.0], dtype="<f4").tobytes()),
        ],
    )

    tensors = read_selected_tensors(path, ["keep_f16", "keep_f32"])

    assert list(tensors) == ["keep_f16", "keep_f32"]
    assert tensors["keep_f16"].source_shape == [2]
    np.testing.assert_array_equal(np.array(tensors["keep_f32"].array), np.array([1.0, 2.0], dtype=np.float32))


def test_selective_reader_reports_missing_requested_tensor(tmp_path):
    path = tmp_path / "model.safetensors"
    write_raw_safetensors(path, [("present", "F32", [1], np.array([1.0], dtype="<f4").tobytes())])

    with pytest.raises(KeyError, match="missing"):
        read_selected_tensors(path, ["missing"])


def test_selective_reader_rejects_unsupported_dtype(tmp_path):
    path = tmp_path / "model.safetensors"
    write_raw_safetensors(path, [("bad", "I8", [1], b"\x01")])

    with pytest.raises(ValueError, match="unsupported dtype.*bad"):
        read_selected_tensors(path, ["bad"])


def test_selective_reader_rejects_missing_data_offsets(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"bad": {"dtype": "F32", "shape": [1]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00\x00\x80?")

    with pytest.raises(ValueError, match="data_offsets.*bad"):
        read_selected_tensors(path, ["bad"])


def test_selective_reader_rejects_offsets_outside_file_bounds(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"bad": {"dtype": "F32", "shape": [1], "data_offsets": [0, 16]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00\x00\x80?")

    with pytest.raises(ValueError, match="offsets.*bad"):
        read_selected_tensors(path, ["bad"])


def test_selective_reader_rejects_truncated_header(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 32) + b"{")

    with pytest.raises(ValueError, match="header"):
        read_selected_tensors(path, ["bad"])
