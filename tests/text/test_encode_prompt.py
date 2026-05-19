from types import SimpleNamespace

import numpy as np
import pytest

from sanasprint_mlx.text.config import TextEncoderConfig
from sanasprint_mlx.text.encoder import encode_prompt


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "left"

    def __call__(self, prompt, **kwargs):
        batch = len(prompt)
        max_length = kwargs["max_length"]
        return SimpleNamespace(
            input_ids=np.arange(batch * max_length).reshape(batch, max_length),
            attention_mask=np.ones((batch, max_length), dtype=np.int64),
        )


class FakeEncoder:
    def __call__(self, input_ids, attention_mask=None):
        embeds = np.repeat(input_ids[..., None].astype(np.float32), 2, axis=-1)
        return (embeds,)


def test_encode_prompt_returns_embeddings_and_mask():
    result = encode_prompt(
        prompt=["hello"],
        tokenizer=FakeTokenizer(),
        text_encoder=FakeEncoder(),
        config=TextEncoderConfig(max_sequence_length=4),
    )

    assert result.prompt_embeds.shape == (1, 4, 2)
    assert result.prompt_attention_mask.shape == (1, 4)


def test_encode_prompt_duplicates_num_images_per_prompt():
    result = encode_prompt(
        prompt=["hello", "world"],
        tokenizer=FakeTokenizer(),
        text_encoder=FakeEncoder(),
        config=TextEncoderConfig(max_sequence_length=4),
        num_images_per_prompt=2,
    )

    assert result.prompt_embeds.shape == (4, 4, 2)
    assert result.prompt_attention_mask.shape == (4, 4)


def test_encode_prompt_requires_mask_with_cached_embeddings():
    with pytest.raises(ValueError, match="prompt_attention_mask"):
        encode_prompt(prompt_embeds=np.zeros((1, 4, 2), dtype=np.float32))
