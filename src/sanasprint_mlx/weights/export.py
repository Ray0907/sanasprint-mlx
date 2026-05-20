from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.weights.inspect import inspect_safetensors_file


COMPONENTS = ("transformer", "text_encoder", "vae")
FORMAT_NAME = "sanasprint-mlx-snapshot"
FORMAT_VERSION = 1


def export_mlx_snapshot(
    snapshot: str | Path,
    output_dir: str | Path,
    *,
    dtype: str = "bfloat16",
    overwrite: bool = False,
) -> dict:
    source = Path(snapshot)
    output = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(f"snapshot path does not exist: {source}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    target_dtype = _mlx_dtype(dtype)
    components = {}
    for component in COMPONENTS:
        source_component = source / component
        if not source_component.exists():
            continue
        output_component = output / component
        output_component.mkdir(parents=True, exist_ok=True)
        _copy_non_weight_files(source_component, output_component)
        components[component] = _export_component_weights(
            source_component,
            output_component,
            dtype=target_dtype,
        )

    if "tokenizer" in [path.name for path in source.iterdir()]:
        _copy_tree_without_safetensors(source / "tokenizer", output / "tokenizer")
    for filename in ("model_index.json", "scheduler", "README.md"):
        path = source / filename
        if path.is_file():
            shutil.copy2(path, output / filename)
        elif path.is_dir():
            _copy_tree_without_safetensors(path, output / filename)

    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_snapshot": str(source),
        "dtype": dtype,
        "components": components,
        "loadable_by": "sanasprint_mlx",
    }
    (output / "mlx_model.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_model_card(output, manifest)
    return manifest


def _export_component_weights(source: Path, output: Path, *, dtype) -> dict:
    files = []
    tensor_count = 0
    parameter_count = 0
    for path in sorted(source.rglob("*.safetensors")):
        relative = path.relative_to(source)
        output_file = output / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        weights = {key: value.astype(dtype) for key, value in mx.load(str(path)).items()}
        mx.save_safetensors(str(output_file), weights)
        mx.eval(*weights.values())
        infos = inspect_safetensors_file(output_file, relative_to=output)
        files.append(
            {
                "path": str(relative),
                "tensor_count": len(infos),
                "parameter_count": sum(info.parameter_count for info in infos),
            }
        )
        tensor_count += len(infos)
        parameter_count += sum(info.parameter_count for info in infos)
        del weights
        gc.collect()
        mx.clear_cache()
    return {
        "files": files,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
    }


def _copy_non_weight_files(source: Path, output: Path) -> None:
    for path in source.rglob("*"):
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        relative = path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_tree_without_safetensors(source: Path, output: Path) -> None:
    if not source.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        relative = path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _write_model_card(output: Path, manifest: dict) -> None:
    readme = output / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "\n".join(
            [
                "---",
                "library_name: sanasprint-mlx",
                "license: other",
                "tags:",
                "- mlx",
                "- sana-sprint",
                "- text-to-image",
                "- apple-silicon",
                "---",
                "",
                "# Converted MLX snapshot for SanaSprint 0.6B",
                "",
                "This repository contains a converted MLX-loadable snapshot for "
                "`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`.",
                "",
                "It is intended for use with the `sanasprint-mlx` runtime. The exported snapshot preserves the tokenizer, "
                "component configs, and safetensors keys expected by the native MLX loader.",
                "",
                "## Format",
                "",
                f"- Format: `{manifest['format']}`",
                f"- Format version: `{manifest['format_version']}`",
                f"- Weight dtype: `{manifest['dtype']}`",
                "- Components: `text_encoder`, `transformer`, `vae`",
                "",
                "## Usage",
                "",
                "```bash",
                "python -m sanasprint_mlx.cli.generate \\",
                "  --prompt \"a tiny astronaut hatching from an egg on the moon\" \\",
                "  --height 768 --width 768 --steps 2 --seed 42 \\",
                "  --snapshot /path/to/this/snapshot \\",
                "  --output /tmp/sanasprint-mlx.png \\",
                "  --tiled-decode",
                "```",
                "",
                "## License and Attribution",
                "",
                "These are converted model weights derived from "
                "`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`. The original model card lists "
                "NSCL v2-custom / NVIDIA License terms and Gemma terms for the text encoder. Those upstream terms "
                "continue to govern the converted weights.",
                "",
            ]
        )
        + "\n"
    )


def _mlx_dtype(dtype: str):
    values = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if dtype not in values:
        raise ValueError(f"dtype must be one of {', '.join(values)}")
    return values[dtype]
