from __future__ import annotations

import argparse
from pathlib import Path

from sanasprint_mlx.baseline.comparison import write_benchmark_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-compare")
    parser.add_argument("--cold-manifest", required=True, type=Path)
    parser.add_argument("--warm-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    if args.output.suffix.lower() != ".json":
        return _error("output path must end with .json")
    try:
        write_benchmark_comparison(
            cold_manifest=args.cold_manifest,
            warm_manifest=args.warm_manifest,
            output=args.output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return _error(str(error))
    print(f"wrote benchmark comparison: {args.output}")
    return 0


def _error(message: str) -> int:
    print(f"error: {message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
