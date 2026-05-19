from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str | None
    height: int
    width: int
    steps: int
    seed: int
    output: str | Path
    snapshot: str | Path | None = None
    cached_fixture: str | Path | None = None
    prompt_cache: str | Path | None = None
    low_memory: bool = False
    allow_download: bool = False
    reference_decode: bool = False
    tiled_decode: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "seed": self.seed,
            "output": str(self.output),
            "snapshot": str(self.snapshot) if self.snapshot is not None else None,
            "cached_fixture": str(self.cached_fixture) if self.cached_fixture is not None else None,
            "prompt_cache": str(self.prompt_cache) if self.prompt_cache is not None else None,
            "low_memory": self.low_memory,
            "allow_download": self.allow_download,
            "reference_decode": self.reference_decode,
            "tiled_decode": self.tiled_decode,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class GenerationPhase:
    name: str
    mode: str | None = None
    loads: list[str] = field(default_factory=list)
    unloads: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "loads": list(self.loads),
            "unloads": list(self.unloads),
            "notes": list(self.notes),
        }


def build_phase_plan(request: GenerationRequest) -> list[GenerationPhase]:
    _validate_request(request)
    phases: list[GenerationPhase] = []
    if request.cached_fixture is None and request.prompt_cache is None:
        phases.append(
            GenerationPhase(
                name="text_encode",
                loads=["tokenizer", "text_encoder"],
                unloads=["tokenizer", "text_encoder"] if request.low_memory else [],
                notes=["produce prompt_embeds, prompt_attention_mask, and prompt cache"],
            )
        )
    phases.append(
        GenerationPhase(
            name="denoise",
            loads=["transformer"],
            unloads=["transformer"] if request.low_memory else [],
            notes=["run MLX scheduler and denoising loop"],
        )
    )
    phases.append(
        GenerationPhase(
            name="decode",
            mode=_decode_mode(request),
            loads=["vae"] if request.reference_decode else ["decoder"],
            unloads=["vae"] if request.low_memory and request.reference_decode else [],
            notes=[_decode_mode(request)],
        )
    )
    phases.append(GenerationPhase(name="write_png", notes=[f"write {request.output}"]))
    return phases


def _validate_request(request: GenerationRequest) -> None:
    if request.prompt is None and request.cached_fixture is None and request.prompt_cache is None:
        raise ValueError("generation requires prompt, cached_fixture, or prompt_cache")
    if request.height % 32 != 0 or request.width % 32 != 0:
        raise ValueError("height and width must be divisible by 32")
    if request.steps <= 0:
        raise ValueError("steps must be positive")
    if _looks_remote_snapshot(request.snapshot) and not request.allow_download:
        raise ValueError("remote snapshot identifiers require --allow-download")


def _looks_remote_snapshot(snapshot: str | Path | None) -> bool:
    if snapshot is None:
        return False
    text = str(snapshot)
    if text.startswith(("http://", "https://", "hf://")):
        return True
    if text.startswith(("/", "./", "../")):
        return False
    return "/" in text and not Path(text).exists()


def _decode_mode(request: GenerationRequest) -> str:
    if request.reference_decode:
        return "reference_decode"
    if request.tiled_decode:
        return "tiled_mlx_decode"
    return "mlx_decode"
