import os

import pytest

from sanasprint_mlx.pipeline.denoise import run_real_loop_fixture_parity


def test_real_loop_fixture_parity_passes_full_denoiser_tolerance():
    fixture = os.environ.get("SANASPRINT_MLX_REAL_LOOP_FIXTURE")
    if not fixture:
        pytest.skip("SANASPRINT_MLX_REAL_LOOP_FIXTURE is required for real loop parity")

    report = run_real_loop_fixture_parity(fixture)

    assert report["passes_full_denoiser_tolerance"], report
