from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.weights import load_scaffold_weights_from_snapshot
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config
from sanasprint_mlx.weights.export import export_mlx_snapshot
from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.mapping import build_mapping_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-weights")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("make-synthetic-snapshot", help="write a tiny local snapshot for tests")
    synthetic.add_argument("--output-dir", required=True, type=Path)

    inspect = subparsers.add_parser("inspect", help="inspect a local safetensors snapshot")
    inspect.add_argument("--snapshot", required=True, type=Path)
    inspect.add_argument("--output", required=True, type=Path)

    load_scaffold = subparsers.add_parser("load-scaffold", help="load scaffold projection weights from a local snapshot")
    load_scaffold.add_argument("--snapshot", required=True, type=Path)
    load_scaffold.add_argument("--output", required=True, type=Path)
    load_scaffold.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    load_scaffold.add_argument("--include-caption-projection", action="store_true")

    export_mlx = subparsers.add_parser("export-mlx", help="export a local snapshot into an MLX-loadable snapshot")
    export_mlx.add_argument("--snapshot", required=True, type=Path)
    export_mlx.add_argument("--output-dir", required=True, type=Path)
    export_mlx.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    export_mlx.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-synthetic-snapshot":
        make_synthetic_snapshot(args.output_dir)
        print(f"wrote synthetic snapshot: {args.output_dir}")
        return 0

    if args.command == "inspect":
        if _looks_remote(args.snapshot):
            parser.error("--snapshot must be a local path, not a remote URL")
        if not args.snapshot.exists():
            parser.error(f"snapshot path does not exist: {args.snapshot}")
        write_inspection_report(args.snapshot, args.output)
        print(f"wrote weight inspection report: {args.output}")
        return 0

    if args.command == "load-scaffold":
        if _looks_remote(args.snapshot):
            parser.error("--snapshot must be a local path, not a remote URL")
        if not args.snapshot.exists():
            parser.error(f"snapshot path does not exist: {args.snapshot}")
        write_scaffold_load_report(
            args.snapshot,
            args.output,
            dtype=args.dtype,
            include_caption_projection=args.include_caption_projection,
        )
        print(f"wrote scaffold load report: {args.output}")
        return 0

    if args.command == "export-mlx":
        if _looks_remote(args.snapshot):
            parser.error("--snapshot must be a local path, not a remote URL")
        if not args.snapshot.exists():
            parser.error(f"snapshot path does not exist: {args.snapshot}")
        manifest = export_mlx_snapshot(
            args.snapshot,
            args.output_dir,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
        print(f"wrote MLX snapshot: {args.output_dir} ({manifest['dtype']})")
        return 0

    parser.error(f"unknown command: {args.command}")


def make_synthetic_snapshot(output_dir: str | Path, *, num_layers: int = 1) -> Path:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    output = Path(output_dir)
    transformer_dir = output / "transformer"
    text_encoder_dir = output / "text_encoder"
    vae_dir = output / "vae"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    text_encoder_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)

    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "SanaTransformer2DModel",
                "num_attention_heads": 2,
                "attention_head_dim": 2,
                "in_channels": 4,
                "out_channels": 4,
                "num_layers": num_layers,
                "caption_channels": 4,
                "sample_size": 2,
                "patch_size": 1,
                "mlp_ratio": 2.0,
                "guidance_embeds_scale": 1000.0,
            },
            indent=2,
        )
        + "\n"
    )
    transformer_tensors = {
        "transformer.patch_embed.proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
        "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
        "transformer.caption_projection.linear_1.weight": np.eye(4, dtype=np.float32),
        "transformer.caption_projection.linear_1.bias": np.zeros((4,), dtype=np.float32),
        "transformer.caption_projection.linear_2.weight": np.eye(4, dtype=np.float32),
        "transformer.caption_projection.linear_2.bias": np.zeros((4,), dtype=np.float32),
        "transformer.caption_norm.weight": np.ones((4,), dtype=np.float32),
        "transformer.proj_out.weight": np.zeros((4, 4), dtype=np.float32),
        "transformer.proj_out.bias": np.zeros((4,), dtype=np.float32),
        "scale_shift_table": np.zeros((2, 4), dtype=np.float32),
        "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
        "transformer.transformer_blocks.0.ff.net.0.proj.weight": np.zeros((8, 4), dtype=np.float32),
    }
    transformer_tensors.update(_synthetic_time_embedding_tensors())
    for block_index in range(num_layers):
        transformer_tensors.update(_synthetic_block_attention_tensors(block_index=block_index))
        transformer_tensors.update(_synthetic_block_ffn_tensors(block_index=block_index))
    save_file(transformer_tensors, transformer_dir / "model.safetensors")
    save_file(
        {"text_encoder.embed_tokens.weight": np.zeros((8, 4), dtype=np.float16)},
        text_encoder_dir / "model.safetensors",
    )
    save_file(
        {"decoder.conv.weight": np.zeros((3, 4, 3, 3), dtype=np.float32)},
        vae_dir / "model.safetensors",
    )
    return output


def _synthetic_block_attention_tensors(*, block_index: int = 0) -> dict[str, np.ndarray]:
    tensors = {}
    prefix = f"transformer_blocks.{block_index}"
    tensors[f"{prefix}.scale_shift_table"] = np.zeros((6, 4), dtype=np.float32)
    for attention in ("attn1", "attn2"):
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            tensors[f"{prefix}.{attention}.{projection}.weight"] = np.eye(4, dtype=np.float32)
        tensors[f"{prefix}.{attention}.to_out.0.bias"] = np.zeros((4,), dtype=np.float32)
        tensors[f"{prefix}.{attention}.norm_q.weight"] = np.ones((4,), dtype=np.float32)
        tensors[f"{prefix}.{attention}.norm_k.weight"] = np.ones((4,), dtype=np.float32)
    for projection in ("to_q", "to_k", "to_v"):
        tensors[f"{prefix}.attn2.{projection}.bias"] = np.zeros((4,), dtype=np.float32)
    return tensors


def _synthetic_block_ffn_tensors(*, block_index: int = 0) -> dict[str, np.ndarray]:
    hidden_size = 4
    hidden_channels = 8
    prefix = f"transformer_blocks.{block_index}.ff"
    return {
        f"{prefix}.conv_inverted.weight": np.zeros((hidden_channels * 2, hidden_size, 1, 1), dtype=np.float32),
        f"{prefix}.conv_inverted.bias": np.zeros((hidden_channels * 2,), dtype=np.float32),
        f"{prefix}.conv_depth.weight": np.zeros((hidden_channels * 2, 1, 3, 3), dtype=np.float32),
        f"{prefix}.conv_depth.bias": np.zeros((hidden_channels * 2,), dtype=np.float32),
        f"{prefix}.conv_point.weight": np.zeros((hidden_size, hidden_channels, 1, 1), dtype=np.float32),
    }


def _synthetic_time_embedding_tensors() -> dict[str, np.ndarray]:
    hidden_size = 4
    prefix = "time_embed"
    return {
        f"{prefix}.timestep_embedder.linear_1.weight": np.zeros((hidden_size, 256), dtype=np.float32),
        f"{prefix}.timestep_embedder.linear_1.bias": np.zeros((hidden_size,), dtype=np.float32),
        f"{prefix}.timestep_embedder.linear_2.weight": np.zeros((hidden_size, hidden_size), dtype=np.float32),
        f"{prefix}.timestep_embedder.linear_2.bias": np.zeros((hidden_size,), dtype=np.float32),
        f"{prefix}.guidance_embedder.linear_1.weight": np.zeros((hidden_size, 256), dtype=np.float32),
        f"{prefix}.guidance_embedder.linear_1.bias": np.zeros((hidden_size,), dtype=np.float32),
        f"{prefix}.guidance_embedder.linear_2.weight": np.zeros((hidden_size, hidden_size), dtype=np.float32),
        f"{prefix}.guidance_embedder.linear_2.bias": np.zeros((hidden_size,), dtype=np.float32),
        f"{prefix}.linear.weight": np.zeros((6 * hidden_size, hidden_size), dtype=np.float32),
        f"{prefix}.linear.bias": np.zeros((6 * hidden_size,), dtype=np.float32),
    }


def write_inspection_report(snapshot: str | Path, output: str | Path) -> Path:
    snapshot_path = Path(snapshot)
    tensor_infos = inspect_snapshot(snapshot_path)
    config_summary = summarize_transformer_config(load_transformer_config(snapshot_path)).__dict__
    report = build_mapping_report(
        tensor_infos,
        snapshot_path=str(snapshot_path),
        config_summary=config_summary,
    )
    output_path = Path(output)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return output_path


def write_scaffold_load_report(
    snapshot: str | Path,
    output: str | Path,
    *,
    dtype: str = "float32",
    include_caption_projection: bool = False,
) -> Path:
    snapshot_path = Path(snapshot)
    summary = summarize_transformer_config(load_transformer_config(snapshot_path))
    config = SanaTransformerConfig(
        hidden_size=summary.hidden_size,
        in_channels=summary.in_channels,
        out_channels=summary.out_channels,
        caption_channels=summary.caption_channels,
        num_layers=summary.num_layers,
        num_attention_heads=summary.num_attention_heads,
        attention_head_dim=summary.attention_head_dim,
        patch_size=summary.patch_size,
        sample_size=summary.sample_size,
        guidance_embeds_scale=summary.guidance_embeds_scale,
    )
    model = SanaTransformerDenoiser(config)
    diagnostics = load_scaffold_weights_from_snapshot(
        model,
        snapshot_path,
        mlx_dtype=_mlx_dtype(dtype),
        strict=True,
        include_caption_projection=include_caption_projection,
    )
    output_path = Path(output)
    output_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return output_path


def _mlx_dtype(dtype: str):
    import mlx.core as mx

    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[dtype]


def _looks_remote(path: Path) -> bool:
    text = str(path)
    return text.startswith(("http://", "https://", "hf://"))


if __name__ == "__main__":
    raise SystemExit(main())
