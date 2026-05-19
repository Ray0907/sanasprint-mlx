import numpy as np
import pytest

from sanasprint_mlx.primitives.patch import patchify_nchw, unpatchify_nchw


def test_patchify_unpatchify_round_trips_nchw():
    x = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)

    tokens = patchify_nchw(x, patch_size=2)
    restored = unpatchify_nchw(tokens, patch_size=2, height=4, width=4, channels=2)

    assert tokens.shape == (1, 4, 8)
    np.testing.assert_array_equal(np.array(restored), x)


def test_patchify_rejects_non_divisible_size():
    x = np.zeros((1, 2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="divisible"):
        patchify_nchw(x, patch_size=2)
