from __future__ import annotations

import argparse
import os
from pathlib import Path

from sanasprint_mlx.verification.gates import build_verification_report, load_pass_evidence
from sanasprint_mlx.verification.hygiene import check_repository_hygiene
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
