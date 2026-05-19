import pytest

from sanasprint_mlx.transformer.config import SanaTransformerConfig


def config_dict():
    return {
        "hidden_size": 4,
        "in_channels": 2,
        "out_channels": 2,
        "caption_channels": 3,
        "num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 4,
        "patch_size": 1,
        "sample_size": 2,
        "guidance_embeds_scale": 1000.0,
    }


def test_transformer_config_from_dict():
    config = SanaTransformerConfig.from_dict(config_dict())

    assert config.hidden_size == 4
    assert config.in_channels == 2


def test_transformer_config_validates_required_dimensions():
    data = config_dict()
    data["hidden_size"] = 0

    with pytest.raises(ValueError, match="hidden_size"):
        SanaTransformerConfig.from_dict(data)
