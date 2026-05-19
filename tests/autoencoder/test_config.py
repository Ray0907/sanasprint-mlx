import pytest

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig


def test_autoencoder_decode_config_defaults():
    config = AutoencoderDecodeConfig()

    assert config.spatial_compression_ratio == 32
    assert config.use_tiling is False


def test_autoencoder_decode_config_rejects_invalid_compression_ratio():
    with pytest.raises(ValueError, match="spatial_compression_ratio"):
        AutoencoderDecodeConfig(spatial_compression_ratio=0).validate()


def test_autoencoder_decode_config_rejects_invalid_tile_stride():
    with pytest.raises(ValueError, match="stride"):
        AutoencoderDecodeConfig(tile_sample_min_height=8, tile_sample_stride_height=16).validate()
