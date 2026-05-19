from types import SimpleNamespace

import numpy as np

from sanasprint_mlx.text.config import TextEncoderConfig
from sanasprint_mlx.text.tokenizer import prepare_prompt_inputs, select_index_for_max_length


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.calls = []

    def encode(self, text):
        return list(range(len(text.split()) + 2))

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        batch = len(prompt)
        max_length = kwargs["max_length"]
        input_ids = np.arange(batch * max_length).reshape(batch, max_length)
        attention_mask = np.ones((batch, max_length), dtype=np.int64)
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


def test_tokenizer_called_with_right_padding_and_max_length():
    tokenizer = FakeTokenizer()

    prepared = prepare_prompt_inputs(["hello"], tokenizer, TextEncoderConfig(max_sequence_length=4))

    assert tokenizer.padding_side == "right"
    assert tokenizer.calls[0]["padding"] == "max_length"
    assert tokenizer.calls[0]["max_length"] == 4
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["add_special_tokens"] is True
    assert prepared.input_ids.shape == (1, 4)


def test_complex_human_instruction_extends_tokenizer_length():
    tokenizer = FakeTokenizer()

    prepare_prompt_inputs(
        ["draw a cat"],
        tokenizer,
        TextEncoderConfig(max_sequence_length=4),
        complex_human_instruction=["be precise", "avoid blur"],
    )

    chi_tokens = len(tokenizer.encode("be precise\navoid blur"))
    assert tokenizer.calls[0]["max_length"] == chi_tokens + 4 - 2
    assert tokenizer.calls[0]["prompt"][0].startswith("be precise\navoid blur")


def test_selected_index_keeps_first_and_last_tokens():
    assert select_index_for_max_length(4) == [0, -3, -2, -1]
