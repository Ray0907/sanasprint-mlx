from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
BENCHMARK_CLASSES = {"locked_cold_diffusers", "warm_persistent_diffusers", "opt_in_mlx_hybrid"}
APPROVED_BASELINE_CLASSES = {"locked_cold_diffusers", "warm_persistent_diffusers"}
RUNTIME_ENGINES = {"diffusers_pytorch_mps", "diffusers_pytorch_cpu", "mlx", "hybrid"}
DTYPES = {"bfloat16", "float16", "float32", "int8", "int4"}
PASS_FAIL = {"PASS", "FAIL"}
FLOAT_TOLERANCE = 1e-9
ALLOWED_ARTIFACT_PREFIXES = {"benchmark-runs", ".benchmarks", "raw-benchmarks"}
COMPARISON_CHECKED_FIELDS = [
    "model.repo",
    "model.revision",
    "generation.prompt_hash",
    "generation.seed",
    "generation.width",
    "generation.height",
    "generation.steps",
    "runtime.engine",
    "runtime.dtype",
    "runtime.device",
    "runtime.low_memory",
    "environment.machine",
    "environment.os_version",
    "environment.python_version",
    "environment.torch_version",
    "environment.diffusers_version",
]
COMPARISON_CLAIM_SCOPE = "diffusers_warm_persistence_only_not_mlx_native"
COMPARISON_BASIS = "cold.summary.wall_time_seconds.median / warm.summary.amortized_wall_time_seconds_per_output.median"


def validate_raw_benchmark_manifest(data: dict) -> dict:
    _validate_benchmark_manifest(data, expected_manifest_type="raw_benchmark")
    return data


def validate_approved_baseline_manifest(data: dict) -> dict:
    _validate_benchmark_manifest(data, expected_manifest_type="approved_baseline")
    if data["benchmark_class"] not in APPROVED_BASELINE_CLASSES:
        raise ValueError("benchmark_class must be locked_cold_diffusers or warm_persistent_diffusers")
    approval = _object(
        data,
        "approval",
        required={"approved_at", "approved_by", "reviewer", "source_raw_manifest", "notes"},
    )
    _non_empty_string(approval, "approved_at")
    _non_empty_string(approval, "approved_by")
    _non_empty_string(approval, "reviewer")
    _non_empty_string(approval, "source_raw_manifest")
    _string(approval, "notes")
    return data


def validate_promotion_manifest(data: dict) -> dict:
    _object_data(data, "manifest", required={
        "schema_version",
        "manifest_type",
        "created_at",
        "old_baseline",
        "candidate_manifest",
        "intended_change",
        "behavior_compatibility",
        "metric_comparison",
        "image_validity",
        "user_approval",
        "reviewer_result",
        "commit_or_pr",
    })
    _schema_version(data)
    _literal(data, "manifest_type", "promotion")
    _non_empty_string(data, "created_at")
    _non_empty_string(data, "old_baseline")
    _non_empty_string(data, "candidate_manifest")
    _non_empty_string(data, "intended_change")
    _enum(data, "behavior_compatibility", PASS_FAIL)
    _enum(data, "metric_comparison", PASS_FAIL)
    _enum(data, "image_validity", PASS_FAIL)
    _non_empty_string(data, "user_approval")
    _enum(data, "reviewer_result", PASS_FAIL)
    _string(data, "commit_or_pr")
    return data


def validate_benchmark_comparison_manifest(data: dict) -> dict:
    _object_data(
        data,
        "manifest",
        required={
            "schema_version",
            "manifest_type",
            "created_at",
            "inputs",
            "compatibility",
            "metrics",
            "conclusion",
            "claim_scope",
        },
    )
    _schema_version(data)
    _literal(data, "manifest_type", "benchmark_comparison")
    _non_empty_string(data, "created_at")
    _validate_comparison_inputs(data)
    _validate_comparison_compatibility(data)
    _validate_comparison_metrics(data)
    _enum(data, "conclusion", {"warm_faster", "warm_slower_or_equal"})
    _literal(data, "claim_scope", COMPARISON_CLAIM_SCOPE)
    return data


def validate_manifest_file(path: str | Path, kind: str) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest JSON is invalid: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    if kind == "raw_benchmark":
        return validate_raw_benchmark_manifest(data)
    if kind == "approved_baseline":
        return validate_approved_baseline_manifest(data)
    if kind == "promotion":
        return validate_promotion_manifest(data)
    if kind == "benchmark_comparison":
        return validate_benchmark_comparison_manifest(data)
    raise ValueError(f"unsupported manifest kind: {kind}")


def _validate_comparison_inputs(data: dict) -> None:
    inputs = _object(data, "inputs", required={"cold", "warm"})
    for key in ("cold", "warm"):
        item = _object(inputs, key, required={"path", "sha256"})
        _non_empty_string(item, "path")
        digest = _non_empty_string(item, "sha256")
        hex_digest = digest.removeprefix("sha256:")
        if (
            not digest.startswith("sha256:")
            or len(hex_digest) != 64
            or any(character not in "0123456789abcdef" for character in hex_digest)
        ):
            raise ValueError(f"inputs.{key}.sha256 must be a sha256 digest")


def _validate_comparison_compatibility(data: dict) -> None:
    compatibility = _object(data, "compatibility", required={"status", "checked_fields"})
    _literal(compatibility, "status", "PASS", prefix="compatibility.")
    checked_fields = _list(compatibility, "checked_fields")
    if checked_fields != COMPARISON_CHECKED_FIELDS:
        raise ValueError("compatibility.checked_fields must match the benchmark comparison contract")


def _validate_comparison_metrics(data: dict) -> None:
    metrics = _object(
        data,
        "metrics",
        required={
            "cold_seconds_per_image",
            "warm_amortized_seconds_per_image",
            "warm_amortized_speedup_ratio",
            "seconds_saved_per_image",
            "warm_total_wall_time_seconds",
            "warm_output_count",
            "max_rss_bytes_delta",
            "peak_footprint_bytes_delta",
            "basis",
        },
    )
    cold = _positive_number(metrics, "cold_seconds_per_image", prefix="metrics.")
    warm = _positive_number(metrics, "warm_amortized_seconds_per_image", prefix="metrics.")
    speedup = _positive_number(metrics, "warm_amortized_speedup_ratio", prefix="metrics.")
    saved = _number(metrics, "seconds_saved_per_image", prefix="metrics.")
    _positive_number(metrics, "warm_total_wall_time_seconds", prefix="metrics.")
    output_count = _positive_int(metrics, "warm_output_count", prefix="metrics.")
    if output_count <= 1:
        raise ValueError("metrics.warm_output_count must be greater than 1")
    _number(metrics, "max_rss_bytes_delta", prefix="metrics.")
    _number(metrics, "peak_footprint_bytes_delta", prefix="metrics.")
    _literal(metrics, "basis", COMPARISON_BASIS, prefix="metrics.")
    if abs(speedup - (cold / warm)) > FLOAT_TOLERANCE:
        raise ValueError("metrics.warm_amortized_speedup_ratio must equal cold / warm")
    if abs(saved - (cold - warm)) > FLOAT_TOLERANCE:
        raise ValueError("metrics.seconds_saved_per_image must equal cold - warm")


def _validate_benchmark_manifest(data: dict, *, expected_manifest_type: str) -> None:
    _object_data(data, "manifest", required={
        "schema_version",
        "manifest_type",
        "benchmark_class",
        "created_at",
        "command",
        "model",
        "runtime",
        "generation",
        "environment",
        "behavior",
        "image",
        "runs",
        "summary",
    } | ({"approval"} if expected_manifest_type == "approved_baseline" else set()))
    _schema_version(data)
    _literal(data, "manifest_type", expected_manifest_type)
    _enum(data, "benchmark_class", BENCHMARK_CLASSES)
    _non_empty_string(data, "created_at")
    _non_empty_string(data, "command")
    _validate_model(data)
    runtime = _validate_runtime(data)
    _validate_generation(data)
    _validate_environment(data, runtime["engine"])
    _validate_behavior(data)
    _validate_image(data)
    _validate_runs_and_summary(data, data["benchmark_class"])


def _validate_model(data: dict) -> None:
    model = _object(data, "model", required={"repo", "revision"})
    _non_empty_string(model, "repo")
    _non_empty_string(model, "revision")


def _validate_runtime(data: dict) -> dict:
    runtime = _object(data, "runtime", required={"engine", "dtype", "device", "low_memory"})
    _enum(runtime, "engine", RUNTIME_ENGINES)
    _enum(runtime, "dtype", DTYPES)
    _non_empty_string(runtime, "device")
    _bool(runtime, "low_memory")
    return runtime


def _validate_generation(data: dict) -> None:
    generation = _object(data, "generation", required={"prompt_hash", "seed", "width", "height", "steps"})
    _non_empty_string(generation, "prompt_hash")
    _int(generation, "seed")
    _positive_int(generation, "width")
    _positive_int(generation, "height")
    _positive_int(generation, "steps")


def _validate_environment(data: dict, engine: str) -> None:
    environment = _object(
        data,
        "environment",
        required={
            "machine",
            "memory_gib",
            "os_version",
            "python_version",
            "torch_version",
            "diffusers_version",
            "mlx_version",
        },
    )
    _non_empty_string(environment, "machine")
    _positive_number(environment, "memory_gib")
    _non_empty_string(environment, "os_version")
    _non_empty_string(environment, "python_version")
    for key in ("torch_version", "diffusers_version", "mlx_version"):
        _non_empty_string(environment, key)

    if engine in {"diffusers_pytorch_mps", "diffusers_pytorch_cpu"}:
        _not_not_used(environment, "torch_version")
        _not_not_used(environment, "diffusers_version")
    elif engine == "mlx":
        _not_not_used(environment, "mlx_version")
        _literal(environment, "torch_version", "not-used")
        _literal(environment, "diffusers_version", "not-used")
    elif engine == "hybrid":
        _not_not_used(environment, "torch_version")
        _not_not_used(environment, "diffusers_version")
        _not_not_used(environment, "mlx_version")


def _validate_behavior(data: dict) -> None:
    behavior = _object(data, "behavior", required={"reference_pipeline_required", "allow_download", "default_behavior_changed"})
    _bool(behavior, "reference_pipeline_required")
    _bool(behavior, "allow_download")
    _bool(behavior, "default_behavior_changed")
    if behavior["default_behavior_changed"]:
        raise ValueError("behavior.default_behavior_changed must be false")


def _validate_image(data: dict) -> None:
    image = _object(data, "image", required={"path", "width", "height", "mode", "valid"})
    _non_empty_string(image, "path")
    _positive_int(image, "width")
    _positive_int(image, "height")
    _literal(image, "mode", "RGB", prefix="image.")
    _bool(image, "valid")
    if not image["valid"]:
        raise ValueError("image.valid must be true")


def _validate_runs_and_summary(data: dict, benchmark_class: str) -> None:
    if benchmark_class == "warm_persistent_diffusers":
        _validate_warm_runs_and_summary(data)
        return
    _validate_cold_runs_and_summary(data)


def _validate_cold_runs_and_summary(data: dict) -> None:
    runs = _list(data, "runs")
    if not runs:
        raise ValueError("runs must not be empty")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be an object")
        _object_data(
            run,
            f"runs[{index}]",
            required={"index", "wall_time_seconds", "max_rss_bytes", "peak_footprint_bytes", "success"},
        )
        _positive_int(run, "index", prefix=f"runs[{index}].")
        _positive_number(run, "wall_time_seconds", prefix=f"runs[{index}].")
        _positive_int(run, "max_rss_bytes", prefix=f"runs[{index}].")
        _positive_int(run, "peak_footprint_bytes", prefix=f"runs[{index}].")
        _bool(run, "success", prefix=f"runs[{index}].")
        if not run["success"]:
            raise ValueError(f"runs[{index}].success must be true")

    summary = _object(data, "summary", required={"run_count", "wall_time_seconds", "max_rss_bytes", "peak_footprint_bytes"})
    _positive_int(summary, "run_count")
    if summary["run_count"] != len(runs):
        raise ValueError("summary.run_count must match len(runs)")
    _validate_metric_summary(summary, "wall_time_seconds")
    _validate_metric_summary(summary, "max_rss_bytes")
    _validate_metric_summary(summary, "peak_footprint_bytes")


def _validate_warm_runs_and_summary(data: dict) -> None:
    runs = _list(data, "runs")
    if len(runs) != 1:
        raise ValueError("warm_persistent_diffusers runs must contain exactly one process run")
    run = runs[0]
    _object_data(
        run,
        "runs[0]",
        required={
            "index",
            "wall_time_seconds",
            "max_rss_bytes",
            "peak_footprint_bytes",
            "timing_source",
            "output_count",
            "output_paths",
            "output_seeds",
            "success",
        },
    )
    _positive_int(run, "index", prefix="runs[0].")
    _positive_number(run, "wall_time_seconds", prefix="runs[0].")
    _positive_int(run, "max_rss_bytes", prefix="runs[0].")
    _positive_int(run, "peak_footprint_bytes", prefix="runs[0].")
    _literal(run, "timing_source", "wrapper_total", prefix="runs[0].")
    output_count = _positive_int(run, "output_count", prefix="runs[0].")
    output_paths = _list(run, "output_paths")
    output_seeds = _list(run, "output_seeds")
    if len(output_paths) != output_count or len(output_seeds) != output_count:
        raise ValueError("runs[0].output_count must match output path and seed counts")
    for index, output_path in enumerate(output_paths):
        if not isinstance(output_path, str) or not output_path:
            raise ValueError(f"runs[0].output_paths[{index}] must be a non-empty string")
        _safe_artifact_path(output_path, f"runs[0].output_paths[{index}]")
    for index, output_seed in enumerate(output_seeds):
        if isinstance(output_seed, bool) or not isinstance(output_seed, int):
            raise ValueError(f"runs[0].output_seeds[{index}] must be an int")
    _bool(run, "success", prefix="runs[0].")
    if not run["success"]:
        raise ValueError("runs[0].success must be true")

    summary = _object(
        data,
        "summary",
        required={
            "run_count",
            "output_count",
            "total_wall_time_seconds",
            "amortized_wall_time_seconds_per_output",
            "max_rss_bytes",
            "peak_footprint_bytes",
        },
    )
    _positive_int(summary, "run_count")
    if summary["run_count"] != 1:
        raise ValueError("summary.run_count must be 1 for warm_persistent_diffusers")
    summary_output_count = _positive_int(summary, "output_count")
    if summary_output_count != output_count:
        raise ValueError("summary.output_count must match runs[0].output_count")
    _validate_metric_summary(summary, "total_wall_time_seconds")
    _validate_metric_summary(summary, "amortized_wall_time_seconds_per_output")
    _validate_metric_summary(summary, "max_rss_bytes")
    _validate_metric_summary(summary, "peak_footprint_bytes")
    _metric_summary_equals(summary["total_wall_time_seconds"], run["wall_time_seconds"], "summary.total_wall_time_seconds")
    _metric_summary_equals(
        summary["amortized_wall_time_seconds_per_output"],
        run["wall_time_seconds"] / output_count,
        "summary.amortized_wall_time_seconds_per_output",
    )
    _metric_summary_equals(summary["max_rss_bytes"], run["max_rss_bytes"], "summary.max_rss_bytes")
    _metric_summary_equals(summary["peak_footprint_bytes"], run["peak_footprint_bytes"], "summary.peak_footprint_bytes")


def _safe_artifact_path(path: str, label: str) -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        if ".." in candidate.parts:
            raise ValueError(f"{label} must not contain parent directory traversal")
        if not candidate.parts or candidate.parts[0] not in ALLOWED_ARTIFACT_PREFIXES:
            raise ValueError(f"{label} must be under /tmp or ignored benchmark roots")
        return
    resolved = candidate.resolve(strict=False)
    temp_roots = {Path(tempfile.gettempdir()).resolve(strict=False), Path("/tmp").resolve(strict=False)}
    if any(resolved == temp_root or temp_root in resolved.parents for temp_root in temp_roots):
        return
    if not _is_ignored_repo_artifact_path(resolved):
        raise ValueError(f"{label} must be under /tmp or ignored benchmark roots")


def _is_ignored_repo_artifact_path(path: Path) -> bool:
    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        root = Path(root_result.stdout.strip()).resolve(strict=False)
        relative = path.relative_to(root)
    except (subprocess.CalledProcessError, ValueError):
        return False
    if not relative.parts or relative.parts[0] not in ALLOWED_ARTIFACT_PREFIXES:
        return False
    result = subprocess.run(
        ["git", "check-ignore", str(path)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def _metric_summary_equals(metric: dict, expected: float | int, label: str) -> None:
    for key in ("min", "median", "max"):
        if abs(metric[key] - expected) > FLOAT_TOLERANCE:
            raise ValueError(f"{label} must match wrapper_total timing")
    if abs(metric["positive_noise_bound"]) > FLOAT_TOLERANCE:
        raise ValueError(f"{label}.positive_noise_bound must be 0 for a single warm process")


def _validate_metric_summary(summary: dict, key: str) -> None:
    metric = _object(summary, key, required={"min", "median", "max", "positive_noise_bound", "rule"})
    minimum = _positive_number(metric, "min", prefix=f"summary.{key}.")
    median = _positive_number(metric, "median", prefix=f"summary.{key}.")
    maximum = _positive_number(metric, "max", prefix=f"summary.{key}.")
    positive_noise_bound = _number(metric, "positive_noise_bound", prefix=f"summary.{key}.")
    if positive_noise_bound < 0:
        raise ValueError(f"summary.{key}.positive_noise_bound must be >= 0")
    _literal(metric, "rule", "max_minus_median", prefix=f"summary.{key}.")
    if not (minimum <= median <= maximum):
        raise ValueError(f"summary.{key} must satisfy min <= median <= max")
    if abs(positive_noise_bound - (maximum - median)) > FLOAT_TOLERANCE:
        raise ValueError(f"summary.{key}.positive_noise_bound must equal max - median")


def _schema_version(data: dict) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")


def _object(data: dict, key: str, *, required: set[str]) -> dict:
    if key not in data:
        raise ValueError(f"{key} is required")
    value = data[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    _object_data(value, key, required=required)
    return value


def _object_data(data: Any, path: str, *, required: set[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be an object")
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"{path}.{missing[0]} is required")
    extra = sorted(data.keys() - required)
    if extra:
        raise ValueError(f"{path}.{extra[0]} is not allowed")


def _list(data: dict, key: str) -> list:
    if key not in data:
        raise ValueError(f"{key} is required")
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _non_empty_string(data: dict, key: str, *, prefix: str = "") -> str:
    value = _string(data, key, prefix=prefix)
    if not value:
        raise ValueError(f"{prefix}{key} must not be empty")
    return value


def _string(data: dict, key: str, *, prefix: str = "") -> str:
    if key not in data:
        raise ValueError(f"{prefix}{key} is required")
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{prefix}{key} must be a string")
    return value


def _bool(data: dict, key: str, *, prefix: str = "") -> bool:
    if key not in data:
        raise ValueError(f"{prefix}{key} is required")
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{prefix}{key} must be a bool")
    return value


def _int(data: dict, key: str, *, prefix: str = "") -> int:
    if key not in data:
        raise ValueError(f"{prefix}{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{prefix}{key} must be an int")
    return value


def _positive_int(data: dict, key: str, *, prefix: str = "") -> int:
    value = _int(data, key, prefix=prefix)
    if value <= 0:
        raise ValueError(f"{prefix}{key} must be positive")
    return value


def _number(data: dict, key: str, *, prefix: str = "") -> float | int:
    if key not in data:
        raise ValueError(f"{prefix}{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}{key} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{prefix}{key} must be finite")
    return value


def _positive_number(data: dict, key: str, *, prefix: str = "") -> float | int:
    value = _number(data, key, prefix=prefix)
    if value <= 0:
        raise ValueError(f"{prefix}{key} must be positive")
    return value


def _enum(data: dict, key: str, allowed: set[str], *, prefix: str = "") -> str:
    value = _non_empty_string(data, key, prefix=prefix)
    if value not in allowed:
        raise ValueError(f"{prefix}{key} must be one of {sorted(allowed)}")
    return value


def _literal(data: dict, key: str, expected: str, *, prefix: str = "") -> None:
    value = _string(data, key, prefix=prefix)
    if value != expected:
        raise ValueError(f"{prefix}{key} must be {expected}")


def _not_not_used(data: dict, key: str) -> None:
    value = _non_empty_string(data, key)
    if value == "not-used":
        raise ValueError(f"{key} must not be not-used")
