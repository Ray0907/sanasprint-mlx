from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sanasprint_mlx.baseline.schema import validate_raw_benchmark_manifest
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO

ALLOWED_REPO_OUTPUT_PREFIXES = ("benchmark-runs", ".benchmarks", "raw-benchmarks")
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class RunnerResult:
    returncode: int
    stdout: str
    stderr: str
    max_rss_bytes: int
    peak_footprint_bytes: int


def summarize_metric(values: Sequence[float | int]) -> dict:
    if not values:
        raise ValueError("values must not be empty")
    minimum = min(values)
    maximum = max(values)
    median = statistics.median(values)
    return {
        "min": minimum,
        "median": median,
        "max": maximum,
        "positive_noise_bound": maximum - median,
        "rule": "max_minus_median",
    }


def ensure_artifact_safe_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=False)
    temp_roots = {Path(tempfile.gettempdir()).resolve(strict=False), Path("/tmp").resolve(strict=False)}
    is_temp_path = any(resolved == temp_root or temp_root in resolved.parents for temp_root in temp_roots)
    if repo_root is None and is_temp_path:
        return candidate

    try:
        root = _repo_root(repo_root)
        relative = resolved.relative_to(root)
    except (subprocess.CalledProcessError, ValueError) as error:
        if is_temp_path:
            return candidate
        roots = ", ".join(str(root) for root in sorted(temp_roots))
        raise ValueError(f"artifact path must be under {roots} or ignored benchmark roots") from error

    if not relative.parts or relative.parts[0] not in ALLOWED_REPO_OUTPUT_PREFIXES:
        raise ValueError("artifact path must be under benchmark-runs/, .benchmarks/, or raw-benchmarks/")
    _require_git_ignored(resolved, root)
    return candidate


def build_benchmark_command(
    *,
    prompt: str,
    snapshot: str | Path,
    output: str | Path,
    height: int,
    width: int,
    steps: int,
    seed: int,
    torch_dtype: str,
    low_memory: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sanasprint_mlx.cli.generate",
        "--prompt",
        prompt,
        "--height",
        str(height),
        "--width",
        str(width),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--reference-pipeline",
        "--torch-dtype",
        torch_dtype,
    ]
    if low_memory:
        command.append("--low-memory")
    return command


def build_warm_benchmark_command(
    *,
    prompt: str,
    snapshot: str | Path,
    image_output: str | Path,
    image_output_dir: str | Path,
    count: int,
    height: int,
    width: int,
    steps: int,
    seed: int,
    torch_dtype: str,
    low_memory: bool,
) -> list[str]:
    command = build_benchmark_command(
        prompt=prompt,
        snapshot=snapshot,
        output=image_output,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        torch_dtype=torch_dtype,
        low_memory=low_memory,
    )
    command.extend(["--output-dir", str(image_output_dir), "--count", str(count)])
    return command


def run_locked_cold_diffusers_benchmark(
    *,
    prompt: str,
    snapshot: str | Path,
    output: str | Path,
    output_dir: str | Path,
    height: int = 512,
    width: int = 512,
    steps: int = 2,
    seed: int = 42,
    runs: int = 3,
    torch_dtype: str = "bfloat16",
    low_memory: bool = False,
    model_repo: str = MODEL_REPO,
    revision: str | None = None,
    runner: Callable[[list[str]], RunnerResult] | None = None,
    clock: Callable[[], float] | None = None,
    environment_collector: Callable[[], dict] | None = None,
    image_metadata_collector: Callable[[Path], dict] | None = None,
) -> dict:
    if runs <= 0:
        raise ValueError("runs must be positive")
    output_path = ensure_artifact_safe_path(output)
    if output_path.suffix.lower() != ".json":
        raise ValueError("output path must end with .json")
    run_output_dir = ensure_artifact_safe_path(output_dir)
    snapshot_path = Path(snapshot)
    if _looks_remote(str(snapshot)) or not snapshot_path.exists():
        raise ValueError("snapshot must be an existing local path")
    resolved_revision = revision or infer_revision_from_snapshot(snapshot_path)
    if resolved_revision is None:
        raise ValueError("revision is required when it cannot be inferred from snapshot")

    runner = runner or run_command_with_time
    clock = clock or time.monotonic
    environment_collector = environment_collector or collect_environment
    image_metadata_collector = image_metadata_collector or collect_image_metadata
    run_output_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    reports = []
    image_metadata = None
    for index in range(1, runs + 1):
        run_image = run_output_dir / f"run-{index}.png"
        command = build_benchmark_command(
            prompt=prompt,
            snapshot=snapshot_path,
            output=run_image,
            height=height,
            width=width,
            steps=steps,
            seed=seed,
            torch_dtype=torch_dtype,
            low_memory=low_memory,
        )
        start = clock()
        result = runner(command)
        elapsed = clock() - start
        if result.returncode != 0:
            raise RuntimeError(f"benchmark run {index} failed: {result.stderr.strip()}")
        report = parse_generate_report(result.stdout)
        reports.append(report)
        image_metadata = image_metadata_collector(run_image)
        run_results.append(
            {
                "index": index,
                "wall_time_seconds": elapsed,
                "max_rss_bytes": result.max_rss_bytes,
                "peak_footprint_bytes": result.peak_footprint_bytes,
                "success": True,
            }
        )

    if image_metadata is None:
        raise RuntimeError("benchmark did not produce image metadata")
    last_report = reports[-1]
    device = last_report.get("device", "")
    engine = "diffusers_pytorch_mps" if device == "mps" else "diffusers_pytorch_cpu"
    manifest = {
        "schema_version": 1,
        "manifest_type": "raw_benchmark",
        "benchmark_class": "locked_cold_diffusers",
        "created_at": _utc_now(),
        "command": " ".join(build_benchmark_command(
            prompt=prompt,
            snapshot=snapshot_path,
            output=run_output_dir / "run-N.png",
            height=height,
            width=width,
            steps=steps,
            seed=seed,
            torch_dtype=torch_dtype,
            low_memory=low_memory,
        )),
        "model": {
            "repo": model_repo,
            "revision": resolved_revision,
        },
        "runtime": {
            "engine": engine,
            "dtype": torch_dtype,
            "device": str(device),
            "low_memory": low_memory,
        },
        "generation": {
            "prompt_hash": prompt_hash(prompt),
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
        },
        "environment": environment_collector(),
        "behavior": {
            "reference_pipeline_required": True,
            "allow_download": False,
            "default_behavior_changed": False,
        },
        "image": image_metadata,
        "runs": run_results,
        "summary": {
            "run_count": len(run_results),
            "wall_time_seconds": summarize_metric([run["wall_time_seconds"] for run in run_results]),
            "max_rss_bytes": summarize_metric([run["max_rss_bytes"] for run in run_results]),
            "peak_footprint_bytes": summarize_metric([run["peak_footprint_bytes"] for run in run_results]),
        },
    }
    validate_raw_benchmark_manifest(manifest)
    _write_json_atomic(output_path, manifest)
    return manifest


def run_warm_persistent_diffusers_benchmark(
    *,
    prompt: str,
    snapshot: str | Path,
    output: str | Path,
    output_dir: str | Path,
    height: int = 512,
    width: int = 512,
    steps: int = 2,
    seed: int = 42,
    count: int = 2,
    torch_dtype: str = "bfloat16",
    low_memory: bool = False,
    model_repo: str = MODEL_REPO,
    revision: str | None = None,
    runner: Callable[[list[str]], RunnerResult] | None = None,
    clock: Callable[[], float] | None = None,
    environment_collector: Callable[[], dict] | None = None,
    image_metadata_collector: Callable[[Path], dict] | None = None,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    output_path = ensure_artifact_safe_path(output)
    if output_path.suffix.lower() != ".json":
        raise ValueError("output path must end with .json")
    run_output_dir = ensure_artifact_safe_path(output_dir)
    image_output = ensure_artifact_safe_path(run_output_dir / "warm.png")
    snapshot_path = Path(snapshot)
    if _looks_remote(str(snapshot)) or not snapshot_path.exists():
        raise ValueError("snapshot must be an existing local path")
    resolved_revision = revision or infer_revision_from_snapshot(snapshot_path)
    if resolved_revision is None:
        raise ValueError("revision is required when it cannot be inferred from snapshot")

    runner = runner or run_command_with_time
    clock = clock or time.monotonic
    environment_collector = environment_collector or collect_environment
    image_metadata_collector = image_metadata_collector or collect_image_metadata
    run_output_dir.mkdir(parents=True, exist_ok=True)

    command = build_warm_benchmark_command(
        prompt=prompt,
        snapshot=snapshot_path,
        image_output=image_output,
        image_output_dir=run_output_dir,
        count=count,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        torch_dtype=torch_dtype,
        low_memory=low_memory,
    )
    start = clock()
    result = runner(command)
    elapsed = clock() - start
    if result.returncode != 0:
        raise RuntimeError(f"warm benchmark failed: {result.stderr.strip()}")
    report = parse_generate_report(result.stdout)
    output_paths, output_seeds = _validate_warm_generate_report(report, count=count, seed=seed, expected_output_dir=run_output_dir)
    image_metadata = image_metadata_collector(Path(output_paths[-1]))
    device = report.get("device", "")
    engine = "diffusers_pytorch_mps" if device == "mps" else "diffusers_pytorch_cpu"
    amortized = elapsed / count
    run_results = [
        {
            "index": 1,
            "wall_time_seconds": elapsed,
            "max_rss_bytes": result.max_rss_bytes,
            "peak_footprint_bytes": result.peak_footprint_bytes,
            "timing_source": "wrapper_total",
            "output_count": count,
            "output_paths": output_paths,
            "output_seeds": output_seeds,
            "success": True,
        }
    ]
    manifest = {
        "schema_version": 1,
        "manifest_type": "raw_benchmark",
        "benchmark_class": "warm_persistent_diffusers",
        "created_at": _utc_now(),
        "command": " ".join(command),
        "model": {
            "repo": model_repo,
            "revision": resolved_revision,
        },
        "runtime": {
            "engine": engine,
            "dtype": torch_dtype,
            "device": str(device),
            "low_memory": low_memory,
        },
        "generation": {
            "prompt_hash": prompt_hash(prompt),
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
        },
        "environment": environment_collector(),
        "behavior": {
            "reference_pipeline_required": True,
            "allow_download": False,
            "default_behavior_changed": False,
        },
        "image": image_metadata,
        "runs": run_results,
        "summary": {
            "run_count": 1,
            "output_count": count,
            "total_wall_time_seconds": summarize_metric([elapsed]),
            "amortized_wall_time_seconds_per_output": summarize_metric([amortized]),
            "max_rss_bytes": summarize_metric([result.max_rss_bytes]),
            "peak_footprint_bytes": summarize_metric([result.peak_footprint_bytes]),
        },
    }
    validate_raw_benchmark_manifest(manifest)
    _write_json_atomic(output_path, manifest)
    return manifest


def infer_revision_from_snapshot(snapshot: str | Path) -> str | None:
    path = Path(snapshot)
    if path.parent.name == "snapshots" and REVISION_RE.fullmatch(path.name):
        return path.name
    return None


def prompt_hash(prompt: str) -> str:
    return f"sha256:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"


def parse_generate_report(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    candidates = []
    for start, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not stdout[start + end :].strip():
            candidates.append(value)
    if not candidates:
        raise RuntimeError("generate output did not contain JSON report")
    return candidates[-1]


def _validate_warm_generate_report(report: dict, *, count: int, seed: int, expected_output_dir: Path) -> tuple[list[str], list[int]]:
    if report.get("count") != count:
        raise RuntimeError("warm generate report count mismatch")
    outputs = report.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != count:
        raise RuntimeError("warm generate report outputs mismatch")
    output_paths = []
    output_seeds = []
    for index, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise RuntimeError("warm generate report output must be an object")
        expected_seed = seed + index
        if item.get("seed") != expected_seed:
            raise RuntimeError("warm generate report seed mismatch")
        output = item.get("output")
        if not isinstance(output, str) or not output:
            raise RuntimeError("warm generate report output path is invalid")
        try:
            output_path = ensure_artifact_safe_path(output)
        except ValueError as error:
            raise RuntimeError("warm generate report output path is not artifact-safe") from error
        try:
            output_path.resolve(strict=False).relative_to(expected_output_dir.resolve(strict=False))
        except ValueError as error:
            raise RuntimeError("warm generate report output path is outside the expected output directory") from error
        output_paths.append(output)
        output_seeds.append(expected_seed)
    return output_paths, output_seeds


def run_command_with_time(command: list[str]) -> RunnerResult:
    time_bin = Path("/usr/bin/time")
    if not time_bin.exists():
        raise RuntimeError("/usr/bin/time is required to capture memory metrics")
    result = subprocess.run(
        [str(time_bin), "-l", *command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    max_rss, peak_footprint = parse_time_l_metrics(result.stderr)
    return RunnerResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        max_rss_bytes=max_rss,
        peak_footprint_bytes=peak_footprint,
    )


def parse_time_l_metrics(stderr: str) -> tuple[int, int]:
    max_rss = None
    peak_footprint = None
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.endswith("maximum resident set size"):
            max_rss = int(stripped.split()[0])
        elif stripped.endswith("peak memory footprint"):
            peak_footprint = int(stripped.split()[0])
    if max_rss is None or peak_footprint is None:
        raise RuntimeError("unable to parse memory metrics from /usr/bin/time -l")
    return max_rss, peak_footprint


def collect_image_metadata(path: str | Path) -> dict:
    from PIL import Image

    image_path = Path(path)
    with Image.open(image_path) as image:
        return {
            "path": str(image_path),
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "valid": True,
        }


def collect_environment() -> dict:
    return {
        "machine": platform.machine() or "unknown",
        "memory_gib": _memory_gib(),
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "diffusers_version": _package_version("diffusers"),
        "mlx_version": _package_version("mlx"),
    }


def _memory_gib() -> float:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return 1.0


def _package_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "not-used"


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve(strict=False)
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return Path(result.stdout.strip()).resolve(strict=False)


def _require_git_ignored(path: Path, repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", str(path)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"artifact path is not ignored by git: {path}")


def _looks_remote(snapshot: str) -> bool:
    return snapshot.startswith(("http://", "https://", "hf://")) or (
        "/" in snapshot and not snapshot.startswith(("/", "./", "../")) and not Path(snapshot).exists()
    )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
