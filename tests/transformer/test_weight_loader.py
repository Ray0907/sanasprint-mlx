import numpy as np
import pytest

from sanasprint_mlx.transformer.weights import load_mapped_weights


def mapped_entry(source, target, status="mapped"):
    return {
        "source_key": source,
        "target_key": target,
        "status": status,
        "suggested_action": "test",
    }


def test_weight_loader_accepts_mapped_entries():
    params = {"mlx_transformer.patch_embed.proj.bias": None}
    tensors = {"transformer.patch_embed.proj.bias": np.ones((2,), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.patch_embed.proj.bias", "mlx_transformer.patch_embed.proj.bias")]}

    loaded = load_mapped_weights(params, tensors, report)

    np.testing.assert_array_equal(loaded["mlx_transformer.patch_embed.proj.bias"], np.ones((2,), dtype=np.float32))


def test_weight_loader_rejects_requires_review_by_default():
    params = {"mlx_transformer.weight": None}
    report = {"mapping": [mapped_entry("transformer.weight", "mlx_transformer.weight", "requires_review")]}

    with pytest.raises(ValueError, match="requires_review"):
        load_mapped_weights(params, {"transformer.weight": np.ones((1,))}, report)


def test_weight_loader_rejects_unexpected_transformer_entries_by_default():
    with pytest.raises(ValueError, match="unexpected"):
        load_mapped_weights({}, {"transformer.extra": np.ones((1,))}, {"mapping": [mapped_entry("transformer.extra", None, "unexpected")]})


def test_weight_loader_override_requires_reason():
    with pytest.raises(ValueError, match="reason"):
        load_mapped_weights({}, {"transformer.extra": np.ones((1,))}, {"mapping": [mapped_entry("transformer.extra", None, "unexpected")]}, allow_unexpected=True)


def test_weight_loader_override_reports_reason_in_diagnostics():
    loaded, diagnostics = load_mapped_weights(
        {},
        {"transformer.extra": np.ones((1,))},
        {"mapping": [mapped_entry("transformer.extra", None, "unexpected")]},
        allow_unexpected=True,
        override_reason="feature test allows an extra key",
        return_diagnostics=True,
    )

    assert loaded == {}
    assert diagnostics == [
        {
            "source_key": "transformer.extra",
            "target_key": None,
            "status": "unexpected",
            "override_reason": "feature test allows an extra key",
        }
    ]


def test_weight_loader_reports_missing_target_parameter():
    report = {"mapping": [mapped_entry("transformer.bias", "mlx_transformer.missing")]}

    with pytest.raises(KeyError, match="mlx_transformer.missing"):
        load_mapped_weights({}, {"transformer.bias": np.ones((1,))}, report)
