from __future__ import annotations

from pathlib import Path

import numpy as np

from sanasprint_mlx.text.cache import read_prompt_cache
from sanasprint_mlx.transformer.parity import compare_arrays


def run_real_text_fixture_parity(fixture_path: str | Path, *, prompt_encoder=None) -> dict:
    if prompt_encoder is None:
        raise RuntimeError("real text parity requires a prompt encoder adapter")
    expected, _ = read_prompt_cache(fixture_path)
    actual = prompt_encoder()
    report = compare_arrays(np.asarray(actual.prompt_embeds), expected["prompt_embeds"])
    report["passes_prompt_embedding_tolerance"] = report["passes_full_denoiser_tolerance"]
    return report
