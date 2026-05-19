import numpy as np
import pytest

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.text.cache import write_prompt_cache
from sanasprint_mlx.verification.scaffold_denoise import run_scaffold_denoise_smoke


def test_scaffold_denoise_smoke_runs_with_synthetic_prompt_inputs(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    report = run_scaffold_denoise_smoke(snapshot, dtype="float16")

    assert report["status"] == "PASS"
    assert report["prompt_source"] == "synthetic"
    assert report["loaded_keys"] == [
        "mlx_transformer.patch_embed.proj.weight",
        "mlx_transformer.patch_embed.proj.bias",
        "mlx_transformer.proj_out.weight",
        "mlx_transformer.proj_out.bias",
    ]
    assert report["latents"]["shape"] == [1, 4, 2, 2]
    assert report["latents"]["finite"] is True
    assert report["latents"]["dtype"] == "float32"
    assert report["weights"]["source_tensors"]["mlx_transformer.patch_embed.proj.weight"]["final_dtype"] == "float16"


def test_scaffold_denoise_smoke_uses_prompt_cache(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 3, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 3), dtype=np.int32),
        tokenizer_id="fake-tokenizer",
        model_id="fake-text-encoder",
        max_sequence_length=3,
        clean_caption=False,
        complex_human_instruction=[],
    )

    report = run_scaffold_denoise_smoke(snapshot, prompt_cache=cache)

    assert report["status"] == "PASS"
    assert report["prompt_source"] == "prompt_cache"
    assert report["prompt_cache"]["metadata"]["model_id"] == "fake-text-encoder"
    assert report["prompt"]["embeds_shape"] == [1, 3, 4]
    assert report["prompt"]["attention_mask_shape"] == [1, 3]


def test_scaffold_denoise_smoke_is_deterministic_for_multiple_steps(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    first = run_scaffold_denoise_smoke(snapshot, steps=2, seed=123)
    second = run_scaffold_denoise_smoke(snapshot, steps=2, seed=123)

    assert first["latents"] == second["latents"]


def test_scaffold_denoise_smoke_rejects_prompt_cache_without_required_arrays(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 3, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 3), dtype=np.int32),
        tokenizer_id="fake-tokenizer",
        model_id="fake-text-encoder",
        max_sequence_length=3,
        clean_caption=False,
        complex_human_instruction=[],
    )
    np.savez(cache / "prompt_cache.npz", prompt_embeds=np.ones((1, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="prompt_attention_mask"):
        run_scaffold_denoise_smoke(snapshot, prompt_cache=cache)


def test_scaffold_denoise_smoke_rejects_unknown_dtype(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    with pytest.raises(ValueError, match="dtype"):
        run_scaffold_denoise_smoke(snapshot, dtype="int8")


def test_scaffold_denoise_smoke_rejects_non_positive_sequence_length(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    with pytest.raises(ValueError, match="sequence_length"):
        run_scaffold_denoise_smoke(snapshot, sequence_length=0)
