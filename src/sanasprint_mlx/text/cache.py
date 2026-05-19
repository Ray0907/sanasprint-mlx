from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def write_prompt_cache(
    output_dir: str | Path,
    *,
    prompt: str,
    prompt_embeds,
    prompt_attention_mask,
    tokenizer_id: str,
    model_id: str,
    max_sequence_length: int,
    clean_caption: bool,
    complex_human_instruction: list[str] | None,
    input_ids=None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "prompt_embeds": np.asarray(prompt_embeds),
        "prompt_attention_mask": np.asarray(prompt_attention_mask),
    }
    if input_ids is not None:
        arrays["input_ids"] = np.asarray(input_ids)
    np.savez(output / "prompt_cache.npz", **arrays)
    metadata = {
        "prompt_sha256": _text_sha256(prompt),
        "tokenizer_id": tokenizer_id,
        "model_id": model_id,
        "max_sequence_length": max_sequence_length,
        "clean_caption": clean_caption,
        "complex_human_instruction": list(complex_human_instruction or []),
        "complex_human_instruction_sha256": _text_sha256("\n".join(complex_human_instruction or [])),
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
            }
            for name, array in arrays.items()
        },
    }
    (output / "prompt_cache.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output / "prompt_cache.json"


def read_prompt_cache(cache_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    root = Path(cache_dir)
    metadata = json.loads((root / "prompt_cache.json").read_text())
    with np.load(root / "prompt_cache.npz") as npz:
        arrays = {name: npz[name] for name in npz.files}
    for name, info in metadata.get("arrays", {}).items():
        if name not in arrays:
            raise ValueError(f"prompt cache missing array: {name}")
        if _array_sha256(arrays[name]) != info["sha256"]:
            raise ValueError(f"prompt cache hash mismatch: {name}")
    return arrays, metadata


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
