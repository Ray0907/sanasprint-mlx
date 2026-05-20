from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import AutoencoderDC

from sanasprint_mlx.autoencoder.mlx_decoder import MLXAutoencoderDCDecoder


def test_mlx_autoencoder_decoder_matches_diffusers_on_tiny_real_latent():
    snapshot = Path(
        "/Users/ray/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/"
        "snapshots/a7d9fc31dd5c3f5e22dbfd78360777ceed56ae97"
    )
    if not snapshot.exists():
        pytest.skip("local Sana snapshot is required")
    rng = np.random.default_rng(19)
    latents = rng.standard_normal((1, 32, 1, 1), dtype=np.float32)

    mlx_decoder = MLXAutoencoderDCDecoder.from_snapshot(snapshot, dtype="float32")
    actual = np.array(mlx_decoder.decode(latents))

    torch_decoder = AutoencoderDC.from_pretrained(snapshot / "vae", torch_dtype=torch.float32, local_files_only=True)
    torch_decoder.eval()
    with torch.no_grad():
        expected = torch_decoder.decode(torch.from_numpy(latents), return_dict=False)[0].numpy()

    assert actual.shape == expected.shape == (1, 3, 32, 32)
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
