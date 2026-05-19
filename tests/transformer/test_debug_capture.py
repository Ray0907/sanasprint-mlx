import numpy as np

from sanasprint_mlx.transformer.debug import DebugCapture, compare_debug_captures
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser


def tiny_model():
    config = SanaTransformerConfig(
        hidden_size=4,
        in_channels=2,
        out_channels=2,
        caption_channels=4,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        patch_size=1,
        sample_size=2,
        guidance_embeds_scale=1000.0,
    )
    return SanaTransformerDenoiser(config)


def call_kwargs():
    return {
        "hidden_states": np.ones((1, 2, 2, 2), dtype=np.float32),
        "encoder_hidden_states": np.ones((1, 3, 4), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 3), dtype=np.int32),
        "guidance": np.array([4.5], dtype=np.float32),
        "timestep": np.array([0.5], dtype=np.float32),
    }


def test_debug_capture_records_named_activations():
    capture = DebugCapture()

    capture.record("input", np.array([1.0]))

    assert "input" in capture.activations


def test_debug_capture_compares_layer_outputs():
    left = DebugCapture()
    right = DebugCapture()
    left.record("layer", np.array([1.0]))
    right.record("layer", np.array([1.1]))

    report = compare_debug_captures(left, right)

    assert report["layer"]["max_abs_error"] > 0


def test_debug_capture_reports_missing_and_extra_activations():
    actual = DebugCapture()
    expected = DebugCapture()
    actual.record("present", np.array([1.0]))
    actual.record("extra", np.array([2.0]))
    expected.record("present", np.array([1.0]))
    expected.record("missing", np.array([3.0]))

    report = compare_debug_captures(actual, expected)

    assert report["missing"]["status"] == "missing_actual"
    assert report["missing"]["passes_full_denoiser_tolerance"] is False
    assert report["extra"]["status"] == "missing_expected"
    assert report["extra"]["passes_full_denoiser_tolerance"] is False


def test_model_debug_mode_returns_expected_names():
    model = tiny_model()

    result = model(**call_kwargs(), return_dict=False, debug=True)

    debug = result[1]
    assert {"input_projection", "conditioning", "block_0", "output_projection", "final_output"}.issubset(debug.activations)
