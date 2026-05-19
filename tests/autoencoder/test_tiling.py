import numpy as np

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig
from sanasprint_mlx.autoencoder.tiling import blend_h, blend_v, build_tiling_plan, tiled_decode


def test_tiling_plan_derives_latent_tile_sizes():
    plan = build_tiling_plan(
        AutoencoderDecodeConfig(
            spatial_compression_ratio=2,
            tile_sample_min_height=8,
            tile_sample_min_width=10,
            tile_sample_stride_height=4,
            tile_sample_stride_width=6,
        )
    )

    assert plan.tile_latent_min_height == 4
    assert plan.tile_latent_min_width == 5
    assert plan.tile_latent_stride_height == 2
    assert plan.tile_latent_stride_width == 3


def test_tiling_plan_derives_sample_space_blend_extents():
    plan = build_tiling_plan(
        AutoencoderDecodeConfig(
            spatial_compression_ratio=2,
            tile_sample_min_height=8,
            tile_sample_min_width=10,
            tile_sample_stride_height=4,
            tile_sample_stride_width=6,
        )
    )

    assert plan.blend_height == 4
    assert plan.blend_width == 4


def test_blend_vertical_matches_linear_overlap():
    above = np.ones((1, 1, 2, 2), dtype=np.float32)
    current = np.zeros((1, 1, 2, 2), dtype=np.float32)

    blended = blend_v(above, current, blend_extent=2)

    np.testing.assert_allclose(blended[:, :, 0, :], 1.0)
    np.testing.assert_allclose(blended[:, :, 1, :], 0.5)


def test_blend_horizontal_matches_linear_overlap():
    left = np.ones((1, 1, 2, 2), dtype=np.float32)
    current = np.zeros((1, 1, 2, 2), dtype=np.float32)

    blended = blend_h(left, current, blend_extent=2)

    np.testing.assert_allclose(blended[:, :, :, 0], 1.0)
    np.testing.assert_allclose(blended[:, :, :, 1], 0.5)


def test_tiled_decode_blends_crops_and_concatenates_tiles():
    config = AutoencoderDecodeConfig(
        spatial_compression_ratio=1,
        tile_sample_min_height=2,
        tile_sample_min_width=2,
        tile_sample_stride_height=1,
        tile_sample_stride_width=1,
    )
    calls = []

    def decoder(tile):
        calls.append(np.array(tile))
        return np.ones_like(tile) * len(calls)

    decoded = tiled_decode(np.zeros((1, 1, 2, 2), dtype=np.float32), decoder, config)

    assert len(calls) == 4
    assert decoded.shape == (1, 1, 2, 2)
