from __future__ import annotations

import argparse
import json
from pathlib import Path

from sanasprint_mlx.generate.plan import GenerationRequest, build_phase_plan
from sanasprint_mlx.generate.reference_bridge import (
    run_reference_pipeline_batch_generation,
    run_reference_pipeline_generation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-generate")
    parser.add_argument("--prompt")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--snapshot")
    parser.add_argument("--cached-fixture", type=Path)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--reference-decode", action="store_true")
    parser.add_argument("--tiled-decode", action="store_true")
    parser.add_argument("--reference-pipeline", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    if args.output.suffix.lower() != ".png":
        return _error("output path must end with .png")
    if args.count <= 0:
        return _error("--count must be positive")
    if args.count > 1 and args.output_dir is None:
        return _error("--count greater than 1 requires --output-dir")
    if args.dry_run and args.plan_output is None:
        return _error("--dry-run requires --plan-output")

    request = GenerationRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        snapshot=args.snapshot,
        cached_fixture=args.cached_fixture,
        prompt_cache=args.prompt_cache,
        low_memory=args.low_memory,
        allow_download=args.allow_download,
        reference_decode=args.reference_decode,
        tiled_decode=args.tiled_decode,
        dry_run=args.dry_run,
    )
    try:
        phases = build_phase_plan(request)
    except (FileNotFoundError, ValueError) as error:
        return _error(str(error))

    if args.dry_run:
        payload = {
            "request": request.to_dict(),
            "phases": [phase.to_dict() for phase in phases],
        }
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote dry-run phase plan: {args.plan_output}")
        return 0

    if args.reference_pipeline:
        try:
            if args.count == 1:
                report = run_reference_pipeline_generation(
                    prompt=args.prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    seed=args.seed,
                    output=args.output,
                    snapshot=args.snapshot,
                    allow_download=args.allow_download,
                    low_memory=args.low_memory,
                    torch_dtype=args.torch_dtype,
                )
            else:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                outputs = _batch_output_paths(args.output, args.output_dir, args.count)
                reports = run_reference_pipeline_batch_generation(
                    prompt=args.prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    seed=args.seed,
                    outputs=outputs,
                    snapshot=args.snapshot,
                    allow_download=args.allow_download,
                    low_memory=args.low_memory,
                    torch_dtype=args.torch_dtype,
                )
                report = _batch_report(reports)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            return _error(str(error))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    return _error("real image generation requires --reference-pipeline or use --dry-run")


def _error(message: str) -> int:
    print(f"error: {message}")
    return 2


def _batch_output_paths(output: Path, output_dir: Path, count: int) -> list[Path]:
    return [output_dir / f"{output.stem}-{index:04d}{output.suffix}" for index in range(1, count + 1)]


def _batch_report(reports: list[dict]) -> dict:
    first = reports[0]
    return {
        "model": first["model"],
        "device": first["device"],
        "low_memory": first["low_memory"],
        "torch_dtype": first["torch_dtype"],
        "count": len(reports),
        "outputs": reports,
    }


if __name__ == "__main__":
    raise SystemExit(main())
