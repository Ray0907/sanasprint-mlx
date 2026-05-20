from __future__ import annotations

import argparse
import os
from pathlib import Path

from sanasprint_mlx.verification.gates import build_verification_report, load_pass_evidence
from sanasprint_mlx.verification.block_attention import run_block0_attention_smoke
from sanasprint_mlx.verification.block_stack import run_block_stack_smoke
from sanasprint_mlx.verification.hygiene import check_repository_hygiene
from sanasprint_mlx.verification.real_block_denoise import run_real_block_denoise_smoke
from sanasprint_mlx.verification.real_transformer_loop import run_real_transformer_loop_smoke
from sanasprint_mlx.verification.report import write_verification_report
from sanasprint_mlx.verification.scaffold_denoise import run_scaffold_denoise_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-verify")
    subparsers = parser.add_subparsers(dest="command")
    scaffold = subparsers.add_parser("scaffold-denoise", help="run a local MLX scaffold denoise smoke check")
    scaffold.add_argument("--snapshot", required=True)
    scaffold.add_argument("--output", required=True, type=Path)
    scaffold.add_argument("--prompt-cache", type=Path)
    scaffold.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    scaffold.add_argument("--seed", type=int, default=0)
    scaffold.add_argument("--steps", type=int, default=1)
    scaffold.add_argument("--sequence-length", type=int, default=4)
    scaffold.add_argument("--real-caption-projection", action="store_true")

    block0 = subparsers.add_parser("block0-attention", help="run a local MLX block-0 attention smoke check")
    block0.add_argument("--snapshot", required=True)
    block0.add_argument("--output", required=True, type=Path)
    block0.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    block0.add_argument("--seed", type=int, default=0)
    block0.add_argument("--sequence-length", type=int, default=4)

    block_stack = subparsers.add_parser("block-stack", help="run a local MLX real block stack smoke check")
    block_stack.add_argument("--snapshot", required=True)
    block_stack.add_argument("--output", required=True, type=Path)
    block_stack.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    block_stack.add_argument("--seed", type=int, default=0)
    block_stack.add_argument("--sequence-length", type=int, default=4)
    block_stack.add_argument("--block-count", type=int, default=2)

    real_block_denoise = subparsers.add_parser(
        "real-block-denoise",
        help="run a local MLX denoise smoke through scaffold weights and real Sana blocks",
    )
    real_block_denoise.add_argument("--snapshot", required=True)
    real_block_denoise.add_argument("--output", required=True, type=Path)
    real_block_denoise.add_argument("--prompt-cache", type=Path)
    real_block_denoise.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    real_block_denoise.add_argument("--seed", type=int, default=0)
    real_block_denoise.add_argument("--sample-size", type=int, default=2)
    real_block_denoise.add_argument("--prompt-sequence-length", type=int, default=4)
    real_block_denoise.add_argument("--block-count", type=int, default=2)

    real_transformer_loop = subparsers.add_parser(
        "real-transformer-loop",
        help="run the SCM scheduler loop with the local MLX real Sana transformer adapter",
    )
    real_transformer_loop.add_argument("--snapshot", required=True)
    real_transformer_loop.add_argument("--output", required=True, type=Path)
    real_transformer_loop.add_argument("--prompt-cache", type=Path)
    real_transformer_loop.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    real_transformer_loop.add_argument("--seed", type=int, default=0)
    real_transformer_loop.add_argument("--steps", type=int, default=1)
    real_transformer_loop.add_argument("--sample-size", type=int, default=2)
    real_transformer_loop.add_argument("--prompt-sequence-length", type=int, default=4)
    real_transformer_loop.add_argument("--block-count", type=int, default=2)

    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--transformer-fixture")
    parser.add_argument("--loop-fixture")
    parser.add_argument("--text-fixture")
    parser.add_argument("--decode-fixture")
    parser.add_argument("--pass-evidence", type=Path)
    parser.add_argument("--check-hygiene", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "scaffold-denoise":
            if _looks_remote(args.snapshot):
                raise ValueError("--snapshot must be a local path, not a remote URL")
            report = run_scaffold_denoise_smoke(
                args.snapshot,
                prompt_cache=args.prompt_cache,
                dtype=args.dtype,
                seed=args.seed,
                steps=args.steps,
                sequence_length=args.sequence_length,
                real_caption_projection=args.real_caption_projection,
            )
            write_verification_report(report, args.output)
            print(f"wrote scaffold denoise report: {args.output}")
            return 0 if report["status"] == "PASS" else 2
        if args.command == "block0-attention":
            if _looks_remote(args.snapshot):
                raise ValueError("--snapshot must be a local path, not a remote URL")
            report = run_block0_attention_smoke(
                args.snapshot,
                dtype=args.dtype,
                seed=args.seed,
                sequence_length=args.sequence_length,
            )
            write_verification_report(report, args.output)
            print(f"wrote block0 attention report: {args.output}")
            return 0 if report["status"] == "PASS" else 2
        if args.command == "block-stack":
            if _looks_remote(args.snapshot):
                raise ValueError("--snapshot must be a local path, not a remote URL")
            report = run_block_stack_smoke(
                args.snapshot,
                dtype=args.dtype,
                seed=args.seed,
                sequence_length=args.sequence_length,
                block_count=args.block_count,
            )
            write_verification_report(report, args.output)
            print(f"wrote block stack report: {args.output}")
            return 0 if report["status"] == "PASS" else 2
        if args.command == "real-block-denoise":
            if _looks_remote(args.snapshot):
                raise ValueError("--snapshot must be a local path, not a remote URL")
            report = run_real_block_denoise_smoke(
                args.snapshot,
                dtype=args.dtype,
                prompt_cache=args.prompt_cache,
                seed=args.seed,
                sample_size=args.sample_size,
                prompt_sequence_length=args.prompt_sequence_length,
                block_count=args.block_count,
            )
            write_verification_report(report, args.output)
            print(f"wrote real block denoise report: {args.output}")
            return 0 if report["status"] == "PASS" else 2
        if args.command == "real-transformer-loop":
            if _looks_remote(args.snapshot):
                raise ValueError("--snapshot must be a local path, not a remote URL")
            report = run_real_transformer_loop_smoke(
                args.snapshot,
                dtype=args.dtype,
                prompt_cache=args.prompt_cache,
                seed=args.seed,
                steps=args.steps,
                sample_size=args.sample_size,
                prompt_sequence_length=args.prompt_sequence_length,
                block_count=args.block_count,
            )
            write_verification_report(report, args.output)
            print(f"wrote real transformer loop report: {args.output}")
            return 0 if report["status"] == "PASS" else 2

        if args.output is None:
            parser.error("--output is required")
        evidence = load_pass_evidence(args.pass_evidence) if args.pass_evidence else None
        report = build_verification_report(
            env=dict(os.environ),
            snapshot=args.snapshot,
            allow_download=args.allow_download,
            pass_evidence=evidence,
            fixture_overrides={
                "transformer_parity": args.transformer_fixture,
                "loop_parity": args.loop_fixture,
                "text_parity": args.text_fixture,
                "decode_parity": args.decode_fixture,
            },
        )
        if args.check_hygiene:
            report["hygiene"] = check_repository_hygiene()
        write_verification_report(report, args.output)
        print(f"wrote verification report: {args.output}")
        if args.check_hygiene and report["hygiene"]["status"] != "PASS":
            return 2
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2


def _looks_remote(path: str) -> bool:
    return path.startswith(("http://", "https://", "hf://", "Efficient-Large-Model/"))


if __name__ == "__main__":
    raise SystemExit(main())
