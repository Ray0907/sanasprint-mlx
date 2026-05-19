import numpy as np

from sanasprint_mlx.transformer.parity import compare_arrays


def test_parity_report_records_max_and_mean_abs_error():
    report = compare_arrays(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    assert report["max_abs_error"] == 1.0
    assert report["mean_abs_error"] == 0.5


def test_parity_report_records_cosine_similarity():
    report = compare_arrays(np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    assert report["cosine_similarity"] == 1.0


def test_parity_report_fails_when_full_denoiser_tolerance_is_missed():
    report = compare_arrays(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))

    assert report["passes_full_denoiser_tolerance"] is False
