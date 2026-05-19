from __future__ import annotations

import argparse
import json
from pathlib import Path

from sanasprint_mlx.memory.estimate import estimate_memory
from sanasprint_mlx.memory.mlx_probe import probe_mlx_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="estimate memory feasibility from a local weight report")
    estimate.add_argument("--weight-report", required=True, type=Path)
    estimate.add_argument("--output", required=True, type=Path)
    estimate.add_argument("--skip-mlx-probe", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "estimate":
        if _looks_remote(args.weight_report):
            parser.error("--weight-report must be a local path, not a remote URL")
        if not args.weight_report.exists():
            parser.error(f"weight report does not exist: {args.weight_report}")
        report = estimate_from_file(args.weight_report, skip_mlx_probe=args.skip_mlx_probe)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote memory feasibility report: {args.output}")
        return 0

    parser.error(f"unknown command: {args.command}")


def estimate_from_file(path: str | Path, *, skip_mlx_probe: bool = False) -> dict:
    weight_report = json.loads(Path(path).read_text())
    report = estimate_memory(weight_report)
    report["source_weight_report"] = str(path)
    if skip_mlx_probe:
        report["mlx_probe"] = {
            "available": False,
            "unavailable_reason": "skipped",
            "allocation_bytes": 0,
            "active_memory_bytes": None,
            "peak_memory_bytes": None,
            "cache_memory_bytes": None,
            "process_rss_bytes": None,
            "cache_cleared": False,
            "cleanup_error": None,
        }
    else:
        report["mlx_probe"] = probe_mlx_memory()
    return report


def _looks_remote(path: Path) -> bool:
    text = str(path)
    return text.startswith(("http://", "https://", "hf://"))


if __name__ == "__main__":
    raise SystemExit(main())
