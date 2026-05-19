import pytest

from sanasprint_mlx.autoencoder.weights import build_vae_decoder_mapping_report


def test_vae_weight_mapping_reports_decoder_keys():
    report = build_vae_decoder_mapping_report(
        [
            {"component": "vae", "key": "decoder.conv.weight", "shape": [3, 4, 3, 3]},
            {"component": "transformer", "key": "transformer.weight", "shape": [1]},
        ]
    )

    assert report["decoder_keys"] == ["decoder.conv.weight"]


def test_vae_weight_mapping_blocks_unknown_entries():
    with pytest.raises(ValueError, match="unknown"):
        build_vae_decoder_mapping_report([{"component": "vae", "key": "encoder.conv.weight", "shape": [1]}])
