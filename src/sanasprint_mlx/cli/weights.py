from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config
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

    parser.error(f"unknown command: {args.command}")


def make_synthetic_snapshot(output_dir: str | Path) -> Path:
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
                "num_layers": 1,
                "caption_channels": 4,
                "sample_size": 2,
                "patch_size": 1,
                "guidance_embeds_scale": 1000.0,
            },
            indent=2,
        )
        + "\n"
    )
    save_file(
        {
            "transformer.patch_embed.proj.weight": np.zeros((4, 4), dtype=np.float32),
            "transformer.patch_embed.proj.bias": np.zeros((4,), dtype=np.float32),
            "transformer.transformer_blocks.0.attn1.to_q.weight": np.zeros((4, 4), dtype=np.float32),
            "transformer.transformer_blocks.0.ff.net.0.proj.weight": np.zeros((8, 4), dtype=np.float32),
        },
        transformer_dir / "model.safetensors",
    )
    save_file(
        {"text_encoder.embed_tokens.weight": np.zeros((8, 4), dtype=np.float16)},
        text_encoder_dir / "model.safetensors",
    )
    save_file(
        {"decoder.conv.weight": np.zeros((3, 4, 3, 3), dtype=np.float32)},
        vae_dir / "model.safetensors",
    )
    return output


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


def _looks_remote(path: Path) -> bool:
    text = str(path)
    return text.startswith(("http://", "https://", "hf://"))


if __name__ == "__main__":
    raise SystemExit(main())
