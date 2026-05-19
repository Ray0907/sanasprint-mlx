from __future__ import annotations

import importlib.util
import struct
import zlib
from pathlib import Path

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
    if not prompt:
        raise ValueError("reference pipeline generation requires --prompt")
    validate_real_generation_request(snapshot=snapshot, allow_download=allow_download)
    if pipeline_cls is None:
        deps = check_reference_dependencies()
        if not deps["available"]:
            raise RuntimeError(f"missing reference dependencies: {', '.join(deps['missing'])}")
        pipeline_cls = _load_sana_sprint_pipeline_class()

    import torch

    model_id = str(snapshot) if snapshot is not None else MODEL_REPO
    dtype = getattr(torch, torch_dtype)
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=not allow_download,
    )
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if low_memory and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif low_memory and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe.to(device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
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
        "model": model_id,
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "device": device,
        "low_memory": low_memory,
        "torch_dtype": torch_dtype,
    }


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
