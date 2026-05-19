from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sanasprint_mlx.text.config import TextEncoderConfig
from sanasprint_mlx.text.tokenizer import prepare_prompt_inputs, select_prompt_tokens


@dataclass(frozen=True)
class EncodedPrompt:
    prompt_embeds: np.ndarray
    prompt_attention_mask: np.ndarray


def encode_prompt(
    prompt: str | list[str] | None = None,
    *,
    tokenizer=None,
    text_encoder=None,
    config: TextEncoderConfig | None = None,
    prompt_embeds=None,
    prompt_attention_mask=None,
    num_images_per_prompt: int = 1,
    complex_human_instruction: list[str] | None = None,
) -> EncodedPrompt:
    config = config or TextEncoderConfig()
    config.validate()
    if prompt_embeds is not None:
        if prompt_attention_mask is None:
            raise ValueError("prompt_attention_mask is required with cached prompt_embeds")
        embeds = np.asarray(prompt_embeds)
        mask = np.asarray(prompt_attention_mask)
    else:
        if prompt is None or tokenizer is None or text_encoder is None:
            raise ValueError("prompt, tokenizer, and text_encoder are required when prompt_embeds are not supplied")
        prepared = prepare_prompt_inputs(
            prompt,
            tokenizer,
            config,
            complex_human_instruction=complex_human_instruction,
        )
        encoder_output = text_encoder(prepared.input_ids, attention_mask=prepared.attention_mask)
        embeds = select_prompt_tokens(_first_output(encoder_output), prepared.select_index)
        mask = select_prompt_tokens(prepared.attention_mask, prepared.select_index)

    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    embeds = np.repeat(embeds, num_images_per_prompt, axis=0)
    mask = np.repeat(mask, num_images_per_prompt, axis=0)
    return EncodedPrompt(prompt_embeds=embeds, prompt_attention_mask=mask)


def _first_output(output):
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output
