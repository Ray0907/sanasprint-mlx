from __future__ import annotations

import numpy as np

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig
from sanasprint_mlx.autoencoder.tiling import build_tiling_plan, tiled_decode


class AutoencoderDCDecode:
    def __init__(self, decoder, config: AutoencoderDecodeConfig | None = None):
        self.decoder = decoder
        self.config = config or AutoencoderDecodeConfig()
        self.config.validate()

    def _decode(self, z):
        z = np.asarray(z)
        plan = build_tiling_plan(self.config)
        if self.config.use_tiling and (z.shape[3] > plan.tile_latent_min_width or z.shape[2] > plan.tile_latent_min_height):
            return tiled_decode(z, self.decoder, self.config)
        return self.decoder(z)

    def decode(self, z, *, return_dict: bool = True):
        z = np.asarray(z)
        if self.config.use_slicing and z.shape[0] > 1:
            decoded = np.concatenate([self._decode(z[index : index + 1]) for index in range(z.shape[0])], axis=0)
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return {"sample": decoded}
