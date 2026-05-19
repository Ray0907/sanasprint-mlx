from __future__ import annotations

import importlib.util
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sanasprint_mlx.fixtures.synthetic import MODEL_REPO

REFERENCE_PACKAGES = ("diffusers", "torch", "transformers", "PIL")


def check_reference_dependencies(packages: tuple[str, ...] = REFERENCE_PACKAGES) -> dict:
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    return {"available": not missing, "missing": missing}


def validate_real_generation_request(snapshot: str | Path | None, allow_download: bool) -> None:
    if snapshot is None and not allow_download:
        raise ValueError("real generation requires a local snapshot or --allow-download")
    if snapshot is None:
        return
    if _looks_remote(snapshot):
        if not allow_download:
            raise ValueError("remote snapshot identifiers require --allow-download")
        return
    path = Path(snapshot)
    if not path.exists():
        raise FileNotFoundError(path)


@dataclass
class ReferencePipelineSession:
    snapshot: str | Path | None
    allow_download: bool
    low_memory: bool = False
    torch_dtype: str = "bfloat16"
    pipeline_cls: object | None = None

    def __post_init__(self) -> None:
        validate_real_generation_request(snapshot=self.snapshot, allow_download=self.allow_download)
        if self.pipeline_cls is None:
            deps = check_reference_dependencies()
            if not deps["available"]:
                raise RuntimeError(f"missing reference dependencies: {', '.join(deps['missing'])}")
            self.pipeline_cls = _load_sana_sprint_pipeline_class()

        import torch

        self._torch = torch
        self.model_id = str(self.snapshot) if self.snapshot is not None else MODEL_REPO
        dtype = getattr(torch, self.torch_dtype)
        self.pipe = self.pipeline_cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            local_files_only=not self.allow_download,
        )
        self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        if self.low_memory and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        elif self.low_memory and hasattr(self.pipe, "enable_sequential_cpu_offload"):
            self.pipe.enable_sequential_cpu_offload()
        elif hasattr(self.pipe, "to"):
            self.pipe.to(self.device)

    def generate_one(
        self,
        *,
        prompt: str | None,
        height: int,
        width: int,
        steps: int,
        seed: int,
        output: str | Path,
    ) -> dict:
        _validate_reference_prompt(prompt)
        generator = self._torch.Generator(device="cpu").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
        )
        image = result.images[0]
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return {
            "output": str(output_path),
            "model": self.model_id,
            "height": height,
            "width": width,
            "steps": steps,
            "seed": seed,
            "device": self.device,
            "low_memory": self.low_memory,
            "torch_dtype": self.torch_dtype,
        }


def run_reference_pipeline_generation(
    *,
    prompt: str | None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    output: str | Path,
    snapshot: str | Path | None,
    allow_download: bool,
    low_memory: bool = False,
    torch_dtype: str = "bfloat16",
    pipeline_cls=None,
) -> dict:
    _validate_reference_prompt(prompt)
    session = ReferencePipelineSession(
        snapshot=snapshot,
        allow_download=allow_download,
        low_memory=low_memory,
        torch_dtype=torch_dtype,
        pipeline_cls=pipeline_cls,
    )
    return session.generate_one(
        prompt=prompt,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        output=output,
    )


def run_reference_pipeline_batch_generation(
    *,
    prompt: str | None,
    height: int,
    width: int,
    steps: int,
    seed: int,
    outputs: Sequence[str | Path],
    snapshot: str | Path | None,
    allow_download: bool,
    low_memory: bool = False,
    torch_dtype: str = "bfloat16",
    pipeline_cls=None,
) -> list[dict]:
    _validate_reference_prompt(prompt)
    if not outputs:
        raise ValueError("batch generation requires at least one output")
    session = ReferencePipelineSession(
        snapshot=snapshot,
        allow_download=allow_download,
        low_memory=low_memory,
        torch_dtype=torch_dtype,
        pipeline_cls=pipeline_cls,
    )
    return [
        session.generate_one(
            prompt=prompt,
            height=height,
            width=width,
            steps=steps,
            seed=seed + index,
            output=output,
        )
        for index, output in enumerate(outputs)
    ]


def _validate_reference_prompt(prompt: str | None) -> None:
    if not prompt:
        raise ValueError("reference pipeline generation requires --prompt")


def write_synthetic_png(path: str | Path, *, width: int, height: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), 128))
        rows.append(bytes(row))
    raw = b"".join(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    output.write_bytes(png)
    return output


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _looks_remote(snapshot: str | Path) -> bool:
    text = str(snapshot)
    return text.startswith(("http://", "https://", "hf://")) or (
        "/" in text and not text.startswith(("/", "./", "../")) and not Path(text).exists()
    )


def _load_sana_sprint_pipeline_class():
    try:
        from diffusers import SanaSprintPipeline

        return SanaSprintPipeline
    except ImportError:
        from diffusers.pipelines.sana.pipeline_sana_sprint import SanaSprintPipeline

        return SanaSprintPipeline
