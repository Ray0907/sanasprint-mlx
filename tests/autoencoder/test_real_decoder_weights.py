from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.autoencoder.real_decoder import load_decoder_weights_from_snapshot
from sanasprint_mlx.cli.weights import make_synthetic_snapshot


def test_load_decoder_weights_from_snapshot_loads_only_vae_decoder_tensors(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    report = load_decoder_weights_from_snapshot(snapshot, mlx_dtype=mx.float16)

    assert report["source"] == "real_weights"
    assert report["loaded_keys"]["count"] == 1
    assert report["loaded_keys"]["keys"] == ["decoder.conv.weight"]
    assert report["source_tensors"]["decoder.conv.weight"]["final_dtype"] == "float16"
    assert "encoder.conv.weight" not in report["source_tensors"]


def test_load_decoder_weights_from_real_snapshot_reports_full_decoder_count():
    snapshot = Path(
        "/Users/ray/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/"
        "snapshots/a7d9fc31dd5c3f5e22dbfd78360777ceed56ae97"
    )
    if not snapshot.exists():
        return

    report = load_decoder_weights_from_snapshot(snapshot, mlx_dtype=mx.bfloat16)

    assert report["loaded_keys"]["count"] == 196
    assert report["source_tensors"]["decoder.conv_in.weight"]["final_dtype"] == "bfloat16"
    assert report["source_tensors"]["decoder.conv_out.weight"]["source_shape"] == [3, 128, 3, 3]
