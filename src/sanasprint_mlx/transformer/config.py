from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SanaTransformerConfig:
    hidden_size: int
    in_channels: int
    out_channels: int
    caption_channels: int
    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    patch_size: int
    sample_size: int
    guidance_embeds_scale: float

    @classmethod
    def from_dict(cls, data: dict) -> "SanaTransformerConfig":
        config = cls(**{field: data[field] for field in cls.__dataclass_fields__})
        config.validate()
        return config

    def validate(self) -> None:
        for field, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{field} must be positive")
