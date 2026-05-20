import pytest

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.text.cache import write_prompt_cache
from sanasprint_mlx.verification.real_block_denoise import run_real_block_denoise_smoke


def test_real_block_denoise_smoke_runs_scaffold_caption_and_two_real_blocks(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=2)

    report = run_real_block_denoise_smoke(
        snapshot,
        block_count=2,
        dtype="float16",
        seed=123,
        sample_size=2,
        prompt_sequence_length=4,
    )

    assert report["status"] == "PASS"
    assert report["scope"] == "real_block_denoise_smoke_not_full_model_parity"
    assert report["block_count"] == 2
    assert report["caption_projection_source"] == "real_weights"
    assert report["time_embedding_source"] == "real_weights"
    assert report["output_norm_source"] == "real_weights"
    assert report["loaded_keys"]["scaffold_count"] == 4
    assert report["loaded_keys"]["caption_count"] == 5
    assert report["loaded_keys"]["time_embedding_count"] == 10
    assert report["loaded_keys"]["output_norm_count"] == 1
    assert report["loaded_keys"]["block_count"] == 46
    assert report["loaded_keys"]["total_count"] == 66
    assert [block["loaded_keys"]["count"] for block in report["blocks"]] == [23, 23]
    assert report["latents"]["input_shape"] == [1, 4, 2, 2]
    assert report["output"]["shape"] == [1, 4, 2, 2]
    assert report["output"]["finite"] is True


def test_real_block_denoise_smoke_rejects_block_count_larger_than_config(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=1)

    with pytest.raises(ValueError, match="block_count must be less than or equal to num_layers"):
        run_real_block_denoise_smoke(snapshot, block_count=2)


def test_real_block_denoise_smoke_accepts_prompt_cache(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=2)
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=[[[-0.25, 0.5, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4]]],
        prompt_attention_mask=[[1, 1]],
        tokenizer_id="fake",
        model_id="fake-text",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )

    report = run_real_block_denoise_smoke(snapshot, prompt_cache=cache, sample_size=2)

    assert report["status"] == "PASS"
    assert report["prompt_source"] == "prompt_cache"
    assert report["prompt_cache"]["path"] == str(cache)
    assert report["prompt"]["embeds_shape"] == [1, 2, 4]
