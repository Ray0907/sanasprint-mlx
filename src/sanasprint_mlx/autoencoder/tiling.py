from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig


@dataclass(frozen=True)
class TilingPlan:
    tile_latent_min_height: int
    tile_latent_min_width: int
    tile_latent_stride_height: int
    tile_latent_stride_width: int
    blend_height: int
    blend_width: int


def build_tiling_plan(config: AutoencoderDecodeConfig) -> TilingPlan:
    config.validate()
    ratio = config.spatial_compression_ratio
    return TilingPlan(
        tile_latent_min_height=config.tile_sample_min_height // ratio,
        tile_latent_min_width=config.tile_sample_min_width // ratio,
        tile_latent_stride_height=config.tile_sample_stride_height // ratio,
        tile_latent_stride_width=config.tile_sample_stride_width // ratio,
        blend_height=config.tile_sample_min_height - config.tile_sample_stride_height,
        blend_width=config.tile_sample_min_width - config.tile_sample_stride_width,
    )


def blend_v(above, current, blend_extent: int):
    above = np.asarray(above)
    blended = np.array(current, copy=True)
    blend_extent = min(above.shape[2], blended.shape[2], blend_extent)
    for y in range(blend_extent):
        blended[:, :, y, :] = above[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + blended[:, :, y, :] * (
            y / blend_extent
        )
    return blended


def blend_h(left, current, blend_extent: int):
    left = np.asarray(left)
    blended = np.array(current, copy=True)
    blend_extent = min(left.shape[3], blended.shape[3], blend_extent)
    for x in range(blend_extent):
        blended[:, :, :, x] = left[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + blended[:, :, :, x] * (
            x / blend_extent
        )
    return blended


def tiled_decode(z, decoder, config: AutoencoderDecodeConfig):
    z = np.asarray(z)
    plan = build_tiling_plan(config)
    rows = []
    for i in range(0, z.shape[2], plan.tile_latent_stride_height):
        row = []
        for j in range(0, z.shape[3], plan.tile_latent_stride_width):
            tile = z[:, :, i : i + plan.tile_latent_min_height, j : j + plan.tile_latent_min_width]
            row.append(np.asarray(decoder(tile)))
        rows.append(row)

    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for col_index, tile in enumerate(row):
            if row_index > 0:
                tile = blend_v(rows[row_index - 1][col_index], tile, plan.blend_height)
            if col_index > 0:
                tile = blend_h(row[col_index - 1], tile, plan.blend_width)
            result_row.append(tile[:, :, : config.tile_sample_stride_height, : config.tile_sample_stride_width])
        result_rows.append(np.concatenate(result_row, axis=3))
    return np.concatenate(result_rows, axis=2)
