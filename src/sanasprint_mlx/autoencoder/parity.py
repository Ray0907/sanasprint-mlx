from __future__ import annotations

from pathlib import Path

import numpy as np

from sanasprint_mlx.transformer.parity import compare_arrays


def run_real_decode_fixture_parity(fixture_path: str | Path, *, decoder=None) -> dict:
    if decoder is None:
        raise RuntimeError("real decode parity requires a decoder adapter")
    root = Path(fixture_path)
    arrays = np.load(root / "decode_fixture.npz" if root.is_dir() else root)
    latents = arrays["latents"]
    expected = arrays["expected_decoded"]
    actual = decoder.decode(latents, return_dict=False)[0]
    report = compare_arrays(actual, expected)
    report["passes_decode_tolerance"] = report["passes_full_denoiser_tolerance"]
    return report
