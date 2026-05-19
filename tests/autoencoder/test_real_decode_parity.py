import os

import pytest

from sanasprint_mlx.autoencoder.parity import run_real_decode_fixture_parity


def test_real_decode_fixture_parity_passes_image_tolerance():
    fixture = os.environ.get("SANASPRINT_MLX_REAL_DECODE_FIXTURE")
    if not fixture:
        pytest.skip("SANASPRINT_MLX_REAL_DECODE_FIXTURE is required for real decode parity")

    report = run_real_decode_fixture_parity(fixture)

    assert report["passes_decode_tolerance"], report
