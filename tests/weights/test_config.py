import json

import pytest

from sanasprint_mlx.weights.config import (
    REQUIRED_TRANSFORMER_CONFIG_FIELDS,
    TransformerConfigSummary,
    load_transformer_config,
    summarize_transformer_config,
)


def write_config(snapshot, data):
    config_dir = snapshot / "transformer"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps(data))


def minimal_config():
    return {
        "_class_name": "SanaTransformer2DModel",
        "num_attention_heads": 36,
        "attention_head_dim": 64,
        "in_channels": 32,
        "out_channels": 32,
        "num_layers": 28,
        "caption_channels": 2304,
        "sample_size": 32,
        "patch_size": 1,
        "guidance_embeds_scale": 1000.0,
    }


def test_load_transformer_config_reads_json(tmp_path):
    write_config(tmp_path, minimal_config())

    config = load_transformer_config(tmp_path)

    assert config["_class_name"] == "SanaTransformer2DModel"
    assert config["num_layers"] == 28


def test_load_transformer_config_requires_expected_fields(tmp_path):
    data = minimal_config()
    del data["caption_channels"]
    write_config(tmp_path, data)

    with pytest.raises(ValueError, match="caption_channels"):
        load_transformer_config(tmp_path)


def test_config_summary_extracts_dimensions(tmp_path):
    write_config(tmp_path, minimal_config())

    summary = summarize_transformer_config(load_transformer_config(tmp_path))

    assert summary == TransformerConfigSummary(
        class_name="SanaTransformer2DModel",
        num_layers=28,
        num_attention_heads=36,
        attention_head_dim=64,
        hidden_size=2304,
        in_channels=32,
        out_channels=32,
        caption_channels=2304,
        sample_size=32,
        patch_size=1,
        guidance_embeds_scale=1000.0,
    )
    assert set(REQUIRED_TRANSFORMER_CONFIG_FIELDS).issubset(minimal_config())
