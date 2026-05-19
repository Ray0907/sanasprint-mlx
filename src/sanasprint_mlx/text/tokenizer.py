from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sanasprint_mlx.text.config import TextEncoderConfig


@dataclass(frozen=True)
class PreparedPromptInputs:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    select_index: list[int]
    max_length_all: int
    prompt: list[str]


def prepare_prompt_inputs(
    prompt: str | list[str],
    tokenizer,
    config: TextEncoderConfig | None = None,
    *,
    complex_human_instruction: list[str] | None = None,
) -> PreparedPromptInputs:
    config = config or TextEncoderConfig()
    config.validate()
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    max_length_all = config.max_sequence_length
    if complex_human_instruction:
        chi_prompt = "\n".join(complex_human_instruction)
        prompts = [chi_prompt + item for item in prompts]
        max_length_all = len(tokenizer.encode(chi_prompt)) + config.max_sequence_length - 2

    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "right"

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length_all,
        truncation=True,
        add_special_tokens=True,
        return_tensors="np",
    )
    input_ids = _field(text_inputs, "input_ids")
    attention_mask = _field(text_inputs, "attention_mask")
    return PreparedPromptInputs(
        input_ids=np.asarray(input_ids),
        attention_mask=np.asarray(attention_mask),
        select_index=select_index_for_max_length(config.max_sequence_length),
        max_length_all=max_length_all,
        prompt=prompts,
    )


def select_index_for_max_length(max_sequence_length: int) -> list[int]:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    return [0] + list(range(-max_sequence_length + 1, 0))


def select_prompt_tokens(values, select_index: list[int]) -> np.ndarray:
    return np.asarray(values)[:, select_index]


def _field(value, name: str):
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)
