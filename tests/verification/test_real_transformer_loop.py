import numpy as np
import pytest

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.text.cache import write_prompt_cache
from sanasprint_mlx.verification.real_transformer_loop import run_real_transformer_loop_smoke


def test_real_transformer_loop_smoke_runs_scheduler_with_real_adapter(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=2)

    report = run_real_transformer_loop_smoke(
        snapshot,
        block_count=2,
        dtype="float16",
        seed=123,
        steps=1,
        sample_size=2,
        prompt_sequence_length=4,
    )

    assert report["status"] == "PASS"
    assert report["scope"] == "real_transformer_loop_smoke_not_full_image_generation"
    assert report["steps"] == 1
    assert report["block_count"] == 2
    assert report["loaded_keys"]["total_count"] == 66
    assert report["latents"]["shape"] == [1, 4, 2, 2]
    assert report["latents"]["finite"] is True


def test_real_transformer_loop_smoke_accepts_prompt_cache(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=1)
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake-text",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )

    report = run_real_transformer_loop_smoke(snapshot, prompt_cache=cache, sample_size=2, block_count=1)

    assert report["status"] == "PASS"
    assert report["prompt_source"] == "prompt_cache"
    assert report["prompt_cache"]["path"] == str(cache)
    assert report["prompt"]["embeds_shape"] == [1, 2, 4]


def test_real_transformer_loop_smoke_rejects_invalid_steps(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=1)

    with pytest.raises(ValueError, match="steps must be positive"):
        run_real_transformer_loop_smoke(snapshot, steps=0)
