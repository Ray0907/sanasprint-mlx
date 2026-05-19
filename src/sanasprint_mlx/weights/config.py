from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TRANSFORMER_CONFIG_FIELDS = (
    "_class_name",
    "num_attention_heads",
    "attention_head_dim",
    "in_channels",
    "out_channels",
    "num_layers",
    "caption_channels",
    "sample_size",
    "patch_size",
    "guidance_embeds_scale",
)


@dataclass(frozen=True)
class TransformerConfigSummary:
    class_name: str
    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    hidden_size: int
    in_channels: int
    out_channels: int
    caption_channels: int
    sample_size: int
    patch_size: int
    guidance_embeds_scale: float


def load_transformer_config(snapshot_path: str | Path) -> dict[str, Any]:
    config_path = Path(snapshot_path) / "transformer" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"transformer config not found: {config_path}")

    config = json.loads(config_path.read_text())
    missing = [field for field in REQUIRED_TRANSFORMER_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"transformer config missing required fields: {', '.join(missing)}")
    return config


def summarize_transformer_config(config: dict[str, Any]) -> TransformerConfigSummary:
    missing = [field for field in REQUIRED_TRANSFORMER_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"transformer config missing required fields: {', '.join(missing)}")

    hidden_size = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    return TransformerConfigSummary(
        class_name=str(config["_class_name"]),
        num_layers=int(config["num_layers"]),
        num_attention_heads=int(config["num_attention_heads"]),
        attention_head_dim=int(config["attention_head_dim"]),
        hidden_size=hidden_size,
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        caption_channels=int(config["caption_channels"]),
        sample_size=int(config["sample_size"]),
        patch_size=int(config["patch_size"]),
        guidance_embeds_scale=float(config["guidance_embeds_scale"]),
    )
