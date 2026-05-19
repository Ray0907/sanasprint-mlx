from __future__ import annotations

import argparse
from pathlib import Path

from sanasprint_mlx.baseline.benchmark import (
    ensure_artifact_safe_path,
    run_locked_cold_diffusers_benchmark,
    run_warm_persistent_diffusers_benchmark,
)
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-benchmark")
    parser.add_argument(
        "--benchmark-class",
        default="locked_cold_diffusers",
        choices=["locked_cold_diffusers", "warm_persistent_diffusers"],
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sanasprint-mlx-benchmark-runs"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--low-memory", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    if args.runs <= 0:
        return _error("runs must be positive")
    if args.count <= 0:
        return _error("count must be positive")
    if args.output.suffix.lower() != ".json":
        return _error("output path must end with .json")
    snapshot = Path(args.snapshot)
    if _looks_remote(args.snapshot) or not snapshot.exists():
        return _error("snapshot must be an existing local path")
    try:
        ensure_artifact_safe_path(args.output)
        ensure_artifact_safe_path(args.output_dir)
    except ValueError as error:
        return _error(str(error))

    try:
        if args.benchmark_class == "locked_cold_diffusers":
            run_locked_cold_diffusers_benchmark(
                prompt=args.prompt,
                snapshot=snapshot,
                model_repo=args.model_repo,
                revision=args.revision,
                output=args.output,
                output_dir=args.output_dir,
                height=args.height,
                width=args.width,
                steps=args.steps,
                seed=args.seed,
                runs=args.runs,
                torch_dtype=args.torch_dtype,
                low_memory=args.low_memory,
            )
        else:
            run_warm_persistent_diffusers_benchmark(
                prompt=args.prompt,
                snapshot=snapshot,
                model_repo=args.model_repo,
                revision=args.revision,
                output=args.output,
                output_dir=args.output_dir,
                height=args.height,
                width=args.width,
                steps=args.steps,
                seed=args.seed,
                count=args.count,
                torch_dtype=args.torch_dtype,
                low_memory=args.low_memory,
            )
    except (OSError, RuntimeError, ValueError) as error:
        return _error(str(error))
    print(f"wrote benchmark manifest: {args.output}")
    return 0


def _looks_remote(snapshot: str) -> bool:
    return snapshot.startswith(("http://", "https://", "hf://")) or (
        "/" in snapshot and not snapshot.startswith(("/", "./", "../")) and not Path(snapshot).exists()
    )


def _error(message: str) -> int:
    print(f"error: {message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
