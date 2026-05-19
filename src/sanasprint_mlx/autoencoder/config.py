from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoencoderDecodeConfig:
    spatial_compression_ratio: int = 32
    use_slicing: bool = False
    use_tiling: bool = False
    tile_sample_min_height: int = 512
    tile_sample_min_width: int = 512
    tile_sample_stride_height: int = 448
    tile_sample_stride_width: int = 448

    def validate(self) -> None:
        if self.spatial_compression_ratio <= 0:
            raise ValueError("spatial_compression_ratio must be positive")
        for field_name in (
            "tile_sample_min_height",
            "tile_sample_min_width",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.tile_sample_stride_height > self.tile_sample_min_height:
            raise ValueError("tile_sample_stride_height must be <= tile_sample_min_height")
        if self.tile_sample_stride_width > self.tile_sample_min_width:
            raise ValueError("tile_sample_stride_width must be <= tile_sample_min_width")
