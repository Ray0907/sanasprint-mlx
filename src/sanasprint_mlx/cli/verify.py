from __future__ import annotations

import argparse
import os
from pathlib import Path

from sanasprint_mlx.verification.gates import build_verification_report, load_pass_evidence
from sanasprint_mlx.verification.report import write_verification_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-verify")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--transformer-fixture")
    parser.add_argument("--loop-fixture")
    parser.add_argument("--text-fixture")
    parser.add_argument("--decode-fixture")
    parser.add_argument("--pass-evidence", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
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
        write_verification_report(report, args.output)
        print(f"wrote verification report: {args.output}")
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
