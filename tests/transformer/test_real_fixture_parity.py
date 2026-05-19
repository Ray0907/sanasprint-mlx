import os

import pytest

from sanasprint_mlx.transformer.real_parity import run_real_fixture_parity


def test_real_fixture_parity_requires_explicit_fixture_path():
    if not os.environ.get("SANASPRINT_MLX_REAL_FIXTURE"):
        pytest.skip("SANASPRINT_MLX_REAL_FIXTURE is required for Tier 2 parity")


def test_real_fixture_parity_passes_full_denoiser_tolerance():
    fixture = os.environ.get("SANASPRINT_MLX_REAL_FIXTURE")
    if not fixture:
        pytest.skip("SANASPRINT_MLX_REAL_FIXTURE is required for Tier 2 parity")

    report = run_real_fixture_parity(fixture)

    assert report["passes_full_denoiser_tolerance"], report
