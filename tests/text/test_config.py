import pytest

from sanasprint_mlx.text.config import TextEncoderConfig


def test_text_encoder_config_defaults():
    config = TextEncoderConfig()

    assert config.max_sequence_length == 300
    assert config.clean_caption is False
    assert config.dtype == "float32"


def test_text_encoder_config_rejects_invalid_sequence_length():
    with pytest.raises(ValueError, match="max_sequence_length"):
        TextEncoderConfig(max_sequence_length=0).validate()
