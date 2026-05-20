import numpy as np

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.transformer.output import SanaOutputNorm, load_output_norm_weights_from_snapshot


def test_sana_output_norm_applies_layer_norm_and_time_modulation():
    norm = SanaOutputNorm(hidden_size=3)
    norm.load_parameters(
        {
            "mlx_transformer.scale_shift_table": np.array(
                [[0.1, -0.2, 0.3], [0.5, 0.0, -0.25]],
                dtype=np.float32,
            )
        }
    )
    x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    conditioning = np.array([[0.2, 0.3, -0.1]], dtype=np.float32)

    result = np.array(norm(x, conditioning))

    normalized = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-6)
    shift = np.array([[[0.3, 0.1, 0.2]]], dtype=np.float32)
    scale = np.array([[[0.7, 0.3, -0.35]]], dtype=np.float32)
    expected = normalized * (1 + scale) + shift
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_sana_output_norm_loads_synthetic_snapshot_weight(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    norm = SanaOutputNorm(hidden_size=4)

    report = load_output_norm_weights_from_snapshot(norm, snapshot)

    assert report["loaded_keys"] == ["mlx_transformer.scale_shift_table"]
    assert report["source_tensors"]["mlx_transformer.scale_shift_table"]["target_shape"] == [2, 4]
