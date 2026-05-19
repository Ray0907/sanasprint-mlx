import numpy as np
import pytest
import mlx.core as mx

from sanasprint_mlx.transformer.weights import load_mapped_weights


def mapped_entry(source, target, status="mapped", transpose_required=False):
    return {
        "source_key": source,
        "target_key": target,
        "status": status,
        "transpose_required": transpose_required,
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


def test_weight_loader_rejects_unknown_backend():
    with pytest.raises(ValueError, match="output_backend"):
        load_mapped_weights({}, {}, {"mapping": []}, output_backend="torch")


def test_weight_loader_converts_mapped_entries_to_mlx_dtype():
    params = {"mlx_transformer.bias": mx.zeros((2,), dtype=mx.float32)}
    tensors = {"transformer.bias": np.ones((2,), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.bias", "mlx_transformer.bias")]}

    loaded = load_mapped_weights(params, tensors, report, output_backend="mlx", mlx_dtype=mx.float16)

    assert isinstance(loaded["mlx_transformer.bias"], mx.array([0]).__class__)
    assert loaded["mlx_transformer.bias"].dtype == mx.float16
    np.testing.assert_array_equal(np.array(loaded["mlx_transformer.bias"]), np.ones((2,), dtype=np.float16))


def test_weight_loader_applies_explicit_2d_transpose_for_mlx():
    params = {"mlx_transformer.proj.weight": mx.zeros((3, 2), dtype=mx.float32)}
    tensors = {"transformer.proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    report = {"mapping": [mapped_entry("transformer.proj.weight", "mlx_transformer.proj.weight", transpose_required=True)]}

    loaded = load_mapped_weights(params, tensors, report, output_backend="mlx")

    np.testing.assert_array_equal(np.array(loaded["mlx_transformer.proj.weight"]), np.arange(6, dtype=np.float32).reshape(2, 3).T)


def test_weight_loader_rejects_explicit_transpose_for_non_2d_tensor():
    params = {"mlx_transformer.proj.weight": mx.zeros((1, 2, 3), dtype=mx.float32)}
    tensors = {"transformer.proj.weight": np.ones((1, 2, 3), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.proj.weight", "mlx_transformer.proj.weight", transpose_required=True)]}

    with pytest.raises(ValueError, match="2D"):
        load_mapped_weights(params, tensors, report, output_backend="mlx")


def test_weight_loader_rejects_unknown_transpose_for_mlx():
    params = {"mlx_transformer.proj.weight": mx.zeros((2, 2), dtype=mx.float32)}
    tensors = {"transformer.proj.weight": np.ones((2, 2), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.proj.weight", "mlx_transformer.proj.weight", transpose_required="unknown")]}

    with pytest.raises(ValueError, match="transpose"):
        load_mapped_weights(params, tensors, report, output_backend="mlx")


def test_weight_loader_rejects_reversed_shape_without_explicit_transpose():
    params = {"mlx_transformer.proj.weight": mx.zeros((3, 2), dtype=mx.float32)}
    tensors = {"transformer.proj.weight": np.ones((2, 3), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.proj.weight", "mlx_transformer.proj.weight")]}

    with pytest.raises(ValueError, match="shape"):
        load_mapped_weights(params, tensors, report, output_backend="mlx")


def test_weight_loader_rejects_target_shape_mismatch():
    params = {"mlx_transformer.bias": mx.zeros((3,), dtype=mx.float32)}
    tensors = {"transformer.bias": np.ones((2,), dtype=np.float32)}
    report = {"mapping": [mapped_entry("transformer.bias", "mlx_transformer.bias")]}

    with pytest.raises(ValueError, match="shape"):
        load_mapped_weights(params, tensors, report, output_backend="mlx")
