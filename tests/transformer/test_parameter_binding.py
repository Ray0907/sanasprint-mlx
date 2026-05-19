import numpy as np
import pytest
import mlx.core as mx

from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.weights import load_mapped_weights_into_denoiser, load_scaffold_weights_from_snapshot


PATCH_WEIGHT = "mlx_transformer.patch_embed.proj.weight"
PATCH_BIAS = "mlx_transformer.patch_embed.proj.bias"
OUT_WEIGHT = "mlx_transformer.proj_out.weight"
OUT_BIAS = "mlx_transformer.proj_out.bias"
SCAFFOLD_KEYS = [PATCH_WEIGHT, PATCH_BIAS, OUT_WEIGHT, OUT_BIAS]


def tiny_config(**overrides):
    data = {
        "hidden_size": 4,
        "in_channels": 2,
        "out_channels": 2,
        "caption_channels": 4,
        "num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 4,
        "patch_size": 1,
        "sample_size": 2,
        "guidance_embeds_scale": 1000.0,
    }
    data.update(overrides)
    return SanaTransformerConfig(**data)


def call_kwargs(config):
    return {
        "hidden_states": np.ones((1, config.in_channels, config.sample_size, config.sample_size), dtype=np.float32),
        "encoder_hidden_states": np.ones((1, 3, config.caption_channels), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 3), dtype=np.int32),
        "guidance": np.array([4.5], dtype=np.float32),
        "timestep": np.array([0.5], dtype=np.float32),
    }


def scaffold_state(config, *, value=1.0):
    return {
        PATCH_WEIGHT: np.full(
            (config.hidden_size, config.in_channels, config.patch_size, config.patch_size),
            value,
            dtype=np.float32,
        ),
        PATCH_BIAS: np.full((config.hidden_size,), value, dtype=np.float32),
        OUT_WEIGHT: np.full(
            (config.out_channels * config.patch_size * config.patch_size, config.hidden_size),
            value,
            dtype=np.float32,
        ),
        OUT_BIAS: np.full((config.out_channels * config.patch_size * config.patch_size,), value, dtype=np.float32),
    }


def mapped_entry(source, target, status="mapped"):
    return {
        "source_key": source,
        "target_key": target,
        "status": status,
        "transpose_required": False,
        "suggested_action": "test",
    }


def test_scaffold_parameters_expose_mapper_compatible_snapshot_shapes_for_non_default_patch_size():
    config = tiny_config(hidden_size=3, out_channels=1, patch_size=2, sample_size=4)
    model = SanaTransformerDenoiser(config)

    params = model.parameters()
    params[PATCH_BIAS] = mx.ones((3,), dtype=mx.float32)

    assert list(params) == SCAFFOLD_KEYS
    assert model.parameters()[PATCH_WEIGHT].shape == (3, 2, 2, 2)
    assert model.parameters()[PATCH_BIAS].shape == (3,)
    assert model.parameters()[OUT_WEIGHT].shape == (4, 3)
    assert model.parameters()[OUT_BIAS].shape == (4,)
    np.testing.assert_array_equal(np.array(model.input_bias), np.zeros((3,), dtype=np.float32))


def test_scaffold_parameters_do_not_expose_private_caption_projection_for_wide_caption_channels():
    config = tiny_config(hidden_size=4, caption_channels=8)
    model = SanaTransformerDenoiser(config)

    assert list(model.parameters()) == SCAFFOLD_KEYS


def test_scaffold_load_parameters_changes_forward_output():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    baseline = np.array(model(**call_kwargs(config))[0])

    model.load_parameters(scaffold_state(config, value=0.25))
    changed = np.array(model(**call_kwargs(config))[0])

    assert not np.allclose(baseline, changed)


def test_scaffold_load_parameters_rejects_missing_and_unknown_keys_in_strict_mode():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    state = scaffold_state(config)
    del state[OUT_BIAS]

    with pytest.raises(KeyError, match=OUT_BIAS):
        model.load_parameters(state)

    state = scaffold_state(config)
    state["mlx_transformer.extra"] = np.ones((1,), dtype=np.float32)
    with pytest.raises(KeyError, match="mlx_transformer.extra"):
        model.load_parameters(state)


def test_scaffold_load_parameters_allows_partial_updates_when_not_strict():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)

    model.load_parameters({OUT_BIAS: np.full((2,), 7.0, dtype=np.float32)}, strict=False)

    np.testing.assert_array_equal(np.array(model.output_bias), np.full((2,), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(np.array(model.input_bias), np.zeros((4,), dtype=np.float32))


def test_scaffold_load_parameters_accepts_numpy_and_mlx_inputs_and_reports_shape_errors():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    state = scaffold_state(config)
    state[PATCH_BIAS] = mx.array(state[PATCH_BIAS])

    model.load_parameters(state)

    assert model.input_bias.shape == (4,)
    with pytest.raises(ValueError, match=f"{PATCH_WEIGHT}.*expected.*got"):
        model.load_parameters({PATCH_WEIGHT: np.ones((4, 2), dtype=np.float32)}, strict=False)


def test_scaffold_load_parameters_is_deterministic_and_does_not_mutate_caller_state():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    state = scaffold_state(config, value=0.5)
    original_patch_shape = state[PATCH_WEIGHT].shape

    model.load_parameters(state)
    first = np.array(model.input_weight)
    model.load_parameters(state)
    second = np.array(model.input_weight)

    assert state[PATCH_WEIGHT].shape == original_patch_shape
    np.testing.assert_array_equal(first, second)


def test_load_mapped_weights_into_scaffold_mutates_model_and_reports_loaded_keys():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    source = {
        "transformer.patch_embed.proj.weight": np.full((4, 2, 1, 1), 0.25, dtype=np.float32),
        "transformer.patch_embed.proj.bias": np.full((4,), 0.5, dtype=np.float32),
        "transformer.proj_out.weight": np.full((2, 4), 0.75, dtype=np.float32),
        "transformer.proj_out.bias": np.full((2,), 1.0, dtype=np.float32),
    }
    report = {
        "mapping": [
            mapped_entry("transformer.patch_embed.proj.weight", PATCH_WEIGHT),
            mapped_entry("transformer.patch_embed.proj.bias", PATCH_BIAS),
            mapped_entry("transformer.proj_out.weight", OUT_WEIGHT),
            mapped_entry("transformer.proj_out.bias", OUT_BIAS),
            mapped_entry("transformer.transformer_blocks.0.attn1.to_q.weight", "mlx_transformer.transformer_blocks.0.attn1.to_q.weight"),
        ]
    }

    diagnostics = load_mapped_weights_into_denoiser(model, source, report, mlx_dtype=mx.float16, strict=True)

    assert diagnostics["loaded_keys"] == SCAFFOLD_KEYS
    assert diagnostics["ignored_entry_count"] == 1
    assert model.input_weight.shape == (4, 2)
    assert model.input_weight.dtype == mx.float16
    np.testing.assert_array_equal(np.array(model.input_bias), np.full((4,), 0.5, dtype=np.float16))


@pytest.mark.parametrize("status", ["requires_review", "missing", "unexpected", "shape_mismatch"])
def test_load_mapped_weights_into_scaffold_rejects_unsafe_relevant_entries(status):
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    target = PATCH_WEIGHT if status != "missing" else "mlx_transformer.patch_embed.*"
    source = "transformer.patch_embed.proj.weight" if status != "missing" else "transformer.patch_embed.*"
    if status == "unexpected":
        target = None
        source = "transformer.proj_out.extra"
    report = {"mapping": [mapped_entry(source, target, status)]}

    with pytest.raises(ValueError, match=status):
        load_mapped_weights_into_denoiser(model, {}, report, strict=False)


def test_load_mapped_weights_into_scaffold_allows_partial_safe_reports_when_not_strict():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    source = {"transformer.proj_out.bias": np.full((2,), 3.0, dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.proj_out.bias", OUT_BIAS)]}

    diagnostics = load_mapped_weights_into_denoiser(model, source, report, strict=False)

    assert diagnostics["loaded_keys"] == [OUT_BIAS]
    np.testing.assert_array_equal(np.array(model.output_bias), np.full((2,), 3.0, dtype=np.float32))


def test_load_mapped_weights_into_scaffold_accepts_real_snapshot_style_unprefixed_source_keys():
    config = tiny_config()
    model = SanaTransformerDenoiser(config)
    source = {"patch_embed.proj.bias": np.full((4,), 2.0, dtype=np.float32)}
    report = {"mapping": [mapped_entry("patch_embed.proj.bias", PATCH_BIAS)]}

    diagnostics = load_mapped_weights_into_denoiser(model, source, report, strict=False)

    assert diagnostics["loaded_keys"] == [PATCH_BIAS]
    np.testing.assert_array_equal(np.array(model.input_bias), np.full((4,), 2.0, dtype=np.float32))


def test_load_scaffold_weights_from_snapshot_loads_four_synthetic_projection_tensors(tmp_path):
    from sanasprint_mlx.cli.weights import make_synthetic_snapshot

    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    config = tiny_config(hidden_size=4, in_channels=4, out_channels=4)
    model = SanaTransformerDenoiser(config)

    diagnostics = load_scaffold_weights_from_snapshot(model, snapshot, mlx_dtype=mx.float16)

    assert diagnostics["loaded_keys"] == SCAFFOLD_KEYS
    assert diagnostics["source_tensors"][PATCH_WEIGHT]["source_key"] == "transformer.patch_embed.proj.weight"
    assert diagnostics["source_tensors"][PATCH_WEIGHT]["source_shape"] == [4, 4, 1, 1]
    assert diagnostics["source_tensors"][PATCH_WEIGHT]["final_dtype"] == "float16"
    assert model.input_weight.dtype == mx.float16


def test_loaded_synthetic_scaffold_can_forward_with_wide_caption_channels(tmp_path):
    from sanasprint_mlx.cli.weights import make_synthetic_snapshot

    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    config = tiny_config(hidden_size=4, in_channels=4, out_channels=4, caption_channels=8)
    model = SanaTransformerDenoiser(config)

    diagnostics = load_scaffold_weights_from_snapshot(model, snapshot, mlx_dtype=mx.float32)
    output = model(
        np.ones((1, config.in_channels, config.sample_size, config.sample_size), dtype=np.float32),
        encoder_hidden_states=np.ones((1, 3, config.caption_channels), dtype=np.float32),
        encoder_attention_mask=np.ones((1, 3), dtype=np.int32),
        guidance=np.array([4.5], dtype=np.float32),
        timestep=np.array([0.5], dtype=np.float32),
    )[0]

    assert diagnostics["loaded_keys"] == SCAFFOLD_KEYS
    assert output.shape == (1, config.out_channels, config.sample_size, config.sample_size)
    assert np.isfinite(np.array(output)).all()


def test_load_scaffold_weights_from_snapshot_strict_mode_requires_all_four_keys(tmp_path):
    from pathlib import Path

    from safetensors.numpy import save_file

    transformer_dir = Path(tmp_path) / "snapshot" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        """
{
  "_class_name": "SanaTransformer2DModel",
  "num_attention_heads": 2,
  "attention_head_dim": 2,
  "in_channels": 4,
  "out_channels": 4,
  "num_layers": 1,
  "caption_channels": 4,
  "sample_size": 2,
  "patch_size": 1,
  "guidance_embeds_scale": 1000.0
}
""".strip()
        + "\n"
    )
    save_file(
        {
            "transformer.patch_embed.proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
            "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
            "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
        },
        transformer_dir / "model.safetensors",
    )
    model = SanaTransformerDenoiser(tiny_config(hidden_size=4, in_channels=4, out_channels=4))

    with pytest.raises(KeyError, match=OUT_WEIGHT):
        load_scaffold_weights_from_snapshot(model, Path(tmp_path) / "snapshot")


def test_load_scaffold_weights_from_snapshot_rejects_duplicate_source_tensors(tmp_path):
    from pathlib import Path

    from safetensors.numpy import save_file

    snapshot = Path(tmp_path) / "snapshot"
    transformer_dir = snapshot / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        """
{
  "_class_name": "SanaTransformer2DModel",
  "num_attention_heads": 2,
  "attention_head_dim": 2,
  "in_channels": 4,
  "out_channels": 4,
  "num_layers": 1,
  "caption_channels": 4,
  "sample_size": 2,
  "patch_size": 1,
  "guidance_embeds_scale": 1000.0
}
""".strip()
        + "\n"
    )
    tensors = {
        "transformer.patch_embed.proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
        "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
        "transformer.proj_out.weight": np.zeros((4, 4), dtype=np.float32),
        "transformer.proj_out.bias": np.zeros((4,), dtype=np.float32),
        "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
    }
    save_file(tensors, transformer_dir / "a.safetensors")
    save_file({"transformer.patch_embed.proj.bias": np.ones((4,), dtype=np.float32)}, transformer_dir / "b.safetensors")
    model = SanaTransformerDenoiser(tiny_config(hidden_size=4, in_channels=4, out_channels=4))

    with pytest.raises(ValueError, match=f"duplicate scaffold target.*{PATCH_BIAS}"):
        load_scaffold_weights_from_snapshot(model, snapshot)


def test_load_scaffold_weights_from_snapshot_rejects_duplicate_target_aliases(tmp_path):
    from pathlib import Path

    from safetensors.numpy import save_file

    snapshot = Path(tmp_path) / "snapshot"
    transformer_dir = snapshot / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        """
{
  "_class_name": "SanaTransformer2DModel",
  "num_attention_heads": 2,
  "attention_head_dim": 2,
  "in_channels": 4,
  "out_channels": 4,
  "num_layers": 1,
  "caption_channels": 4,
  "sample_size": 2,
  "patch_size": 1,
  "guidance_embeds_scale": 1000.0
}
""".strip()
        + "\n"
    )
    save_file(
        {
            "transformer.patch_embed.proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
            "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
            "transformer.proj_out.weight": np.zeros((4, 4), dtype=np.float32),
            "proj_out.weight": np.ones((4, 4), dtype=np.float32),
            "transformer.proj_out.bias": np.zeros((4,), dtype=np.float32),
            "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
        },
        transformer_dir / "model.safetensors",
    )
    model = SanaTransformerDenoiser(tiny_config(hidden_size=4, in_channels=4, out_channels=4))

    with pytest.raises(ValueError, match=f"duplicate scaffold target.*{OUT_WEIGHT}"):
        load_scaffold_weights_from_snapshot(model, snapshot)


def test_load_scaffold_weights_from_snapshot_rejects_scaffold_shape_mismatch(tmp_path):
    from pathlib import Path

    from safetensors.numpy import save_file

    snapshot = Path(tmp_path) / "snapshot"
    transformer_dir = snapshot / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        """
{
  "_class_name": "SanaTransformer2DModel",
  "num_attention_heads": 2,
  "attention_head_dim": 2,
  "in_channels": 4,
  "out_channels": 4,
  "num_layers": 1,
  "caption_channels": 4,
  "sample_size": 2,
  "patch_size": 1,
  "guidance_embeds_scale": 1000.0
}
""".strip()
        + "\n"
    )
    save_file(
        {
            "transformer.patch_embed.proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
            "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
            "transformer.proj_out.weight": np.zeros((3, 4), dtype=np.float32),
            "transformer.proj_out.bias": np.zeros((4,), dtype=np.float32),
            "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
        },
        transformer_dir / "model.safetensors",
    )
    model = SanaTransformerDenoiser(tiny_config(hidden_size=4, in_channels=4, out_channels=4))

    with pytest.raises(ValueError, match="shape_mismatch"):
        load_scaffold_weights_from_snapshot(model, snapshot)
