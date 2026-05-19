import os

import pytest
import numpy as np

from sanasprint_mlx.text.cache import write_prompt_cache
from sanasprint_mlx.text.parity import run_real_text_fixture_parity


def test_real_text_fixture_parity_requires_prompt_encoder(tmp_path):
    write_prompt_cache(
        tmp_path,
        prompt="hello",
        prompt_embeds=np.ones((1, 4, 2), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 4), dtype=np.int64),
        tokenizer_id="fake-tokenizer",
        model_id="fake-model",
        max_sequence_length=4,
        clean_caption=False,
        complex_human_instruction=[],
    )

    with pytest.raises(RuntimeError, match="prompt encoder"):
        run_real_text_fixture_parity(tmp_path)


def test_real_text_fixture_parity_passes_prompt_embedding_tolerance():
    fixture = os.environ.get("SANASPRINT_MLX_REAL_TEXT_FIXTURE")
    if not fixture:
        pytest.skip("SANASPRINT_MLX_REAL_TEXT_FIXTURE is required for real text parity")

    report = run_real_text_fixture_parity(fixture)

    assert report["passes_prompt_embedding_tolerance"], report
