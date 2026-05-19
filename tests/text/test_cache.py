import json

import numpy as np
import pytest

from sanasprint_mlx.text.cache import read_prompt_cache, write_prompt_cache


def test_prompt_cache_roundtrips_arrays_and_metadata(tmp_path):
    output = tmp_path / "cache"
    prompt_embeds = np.ones((1, 4, 2), dtype=np.float32)
    prompt_attention_mask = np.ones((1, 4), dtype=np.int64)

    write_prompt_cache(
        output,
        prompt="hello",
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        tokenizer_id="fake-tokenizer",
        model_id="fake-model",
        max_sequence_length=4,
        clean_caption=False,
        complex_human_instruction=["be precise"],
    )
    arrays, metadata = read_prompt_cache(output)

    np.testing.assert_array_equal(arrays["prompt_embeds"], prompt_embeds)
    np.testing.assert_array_equal(arrays["prompt_attention_mask"], prompt_attention_mask)
    assert metadata["tokenizer_id"] == "fake-tokenizer"


def test_prompt_cache_detects_hash_mismatch(tmp_path):
    output = tmp_path / "cache"
    write_prompt_cache(
        output,
        prompt="hello",
        prompt_embeds=np.ones((1, 4, 2), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 4), dtype=np.int64),
        tokenizer_id="fake-tokenizer",
        model_id="fake-model",
        max_sequence_length=4,
        clean_caption=False,
        complex_human_instruction=[],
    )
    metadata_path = output / "prompt_cache.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["arrays"]["prompt_embeds"]["sha256"] = "bad"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="hash mismatch"):
        read_prompt_cache(output)


def test_prompt_cache_metadata_records_clean_caption_and_complex_instruction(tmp_path):
    output = tmp_path / "cache"

    write_prompt_cache(
        output,
        prompt="hello",
        prompt_embeds=np.ones((1, 4, 2), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 4), dtype=np.int64),
        tokenizer_id="fake-tokenizer",
        model_id="fake-model",
        max_sequence_length=4,
        clean_caption=True,
        complex_human_instruction=["be precise"],
    )

    metadata = json.loads((output / "prompt_cache.json").read_text())
    assert metadata["clean_caption"] is True
    assert metadata["complex_human_instruction"] == ["be precise"]
    assert metadata["complex_human_instruction_sha256"]
