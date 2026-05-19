from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextEncoderConfig:
    max_sequence_length: int = 300
    clean_caption: bool = False
    tokenizer_id: str = "GemmaTokenizer"
    model_id: str = "Gemma2Model"
    dtype: str = "float32"

    def validate(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if not self.dtype:
            raise ValueError("dtype is required")
