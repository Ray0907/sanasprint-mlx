from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from sanasprint_mlx.memory.mlx_cache import trim_mlx_cache
from sanasprint_mlx.text.config import TextEncoderConfig
from sanasprint_mlx.text.encoder import EncodedPrompt
from sanasprint_mlx.text.tokenizer import prepare_prompt_inputs, select_prompt_tokens


def encode_prompt_mlx(
    *,
    prompt: str | list[str],
    snapshot: str | Path,
    max_sequence_length: int = 300,
    complex_human_instruction: list[str] | None = None,
) -> EncodedPrompt:
    mx, gemma2, auto_tokenizer = _mlx_text_dependencies()
    snapshot_path = Path(snapshot)
    text_encoder_path = snapshot_path / "text_encoder"
    tokenizer_path = snapshot_path / "tokenizer"
    config = TextEncoderConfig(max_sequence_length=max_sequence_length, dtype="bfloat16")

    model = _load_gemma2_hidden_state_model(text_encoder_path, mx=mx, gemma2=gemma2)
    tokenizer = auto_tokenizer.from_pretrained(tokenizer_path)
    prepared = prepare_prompt_inputs(
        prompt,
        tokenizer,
        config,
        complex_human_instruction=complex_human_instruction,
    )

    hidden_states = model.model(mx.array(prepared.input_ids)).astype(mx.float32)
    mx.eval(hidden_states)
    prompt_embeds = select_prompt_tokens(np.array(hidden_states), prepared.select_index).astype(np.float32, copy=False)
    prompt_attention_mask = select_prompt_tokens(prepared.attention_mask, prepared.select_index).astype(np.int32, copy=False)

    del hidden_states, model
    gc.collect()
    trim_mlx_cache(mx)
    return EncodedPrompt(prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_attention_mask)


def prefixed_mlx_lm_weight_key(key: str) -> str:
    if key.startswith("model."):
        return key
    return f"model.{key}"


def _load_gemma2_hidden_state_model(text_encoder_path: Path, *, mx, gemma2):
    config = json.loads((text_encoder_path / "config.json").read_text())
    model = gemma2.Model(gemma2.ModelArgs.from_dict(config))
    weights = {}
    for weight_file in sorted(text_encoder_path.glob("model-*.safetensors")):
        weights.update({prefixed_mlx_lm_weight_key(key): value for key, value in mx.load(str(weight_file)).items()})
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def _mlx_text_dependencies():
    try:
        import mlx.core as mx
        from mlx_lm.models import gemma2
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("native MLX prompt encoding requires mlx-lm and transformers") from error
    return mx, gemma2, AutoTokenizer
