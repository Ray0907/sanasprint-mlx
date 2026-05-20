from pathlib import Path

import numpy as np
import pytest

from sanasprint_mlx.text.cache import read_prompt_cache
from sanasprint_mlx.text.mlx_encoder import encode_prompt_mlx, prefixed_mlx_lm_weight_key


SNAPSHOT = Path(
    "/Users/ray/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/"
    "snapshots/a7d9fc31dd5c3f5e22dbfd78360777ceed56ae97"
)
PROMPT_CACHE = Path("/tmp/sanasprint-mlx-real-prompt-cache")
PROMPT = (
    "a cinematic macro photograph of a translucent glass apple on a wet black stone table, "
    "sharp reflections, studio rim light, ultra detailed"
)


def test_prefixed_mlx_lm_weight_key_adds_model_namespace():
    assert prefixed_mlx_lm_weight_key("layers.0.self_attn.q_proj.weight") == "model.layers.0.self_attn.q_proj.weight"
    assert prefixed_mlx_lm_weight_key("model.layers.0.self_attn.q_proj.weight") == "model.layers.0.self_attn.q_proj.weight"


@pytest.mark.skipif(not SNAPSHOT.exists() or not PROMPT_CACHE.exists(), reason="real Sana snapshot and prompt cache are local fixtures")
def test_mlx_prompt_encoder_matches_reference_cache_on_valid_tokens():
    expected, metadata = read_prompt_cache(PROMPT_CACHE)

    result = encode_prompt_mlx(
        prompt=PROMPT,
        snapshot=SNAPSHOT,
        complex_human_instruction=metadata["complex_human_instruction"],
    )

    mask = result.prompt_attention_mask.astype(bool)
    diff = np.abs(result.prompt_embeds - expected["prompt_embeds"].astype(np.float32))
    assert result.prompt_embeds.shape == (1, 300, 2304)
    assert np.array_equal(result.prompt_attention_mask, expected["prompt_attention_mask"].astype(np.int32))
    assert float(diff[mask].mean()) < 0.05
    assert float(diff[mask].max()) < 1.0
