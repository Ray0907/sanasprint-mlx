from __future__ import annotations

import numpy as np


def compare_arrays(actual, expected) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return {
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "cosine_similarity": 0.0,
            "passes_full_denoiser_tolerance": False,
        }
    diff = actual - expected
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    denom = float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat))
    cosine = float(np.dot(actual_flat, expected_flat) / denom) if denom else 1.0
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cosine,
        "passes_full_denoiser_tolerance": cosine >= 0.995 and mean_abs <= 3e-2,
    }
