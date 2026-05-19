from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sanasprint_mlx.baseline.benchmark import ensure_artifact_safe_path
from sanasprint_mlx.baseline.schema import (
    COMPARISON_BASIS,
    COMPARISON_CHECKED_FIELDS,
    COMPARISON_CLAIM_SCOPE,
    validate_benchmark_comparison_manifest,
    validate_manifest_file,
    validate_raw_benchmark_manifest,
)

CHECKED_FIELDS = COMPARISON_CHECKED_FIELDS


def compare_benchmark_manifests(
    cold: dict,
    warm: dict,
    *,
    cold_path: str | Path,
    warm_path: str | Path,
    cold_digest: str,
    warm_digest: str,
) -> dict:
    validate_raw_benchmark_manifest(cold)
    validate_raw_benchmark_manifest(warm)
    _validate_classes(cold, warm)
    _validate_runtime_scope(cold, warm)
    _validate_compatibility(cold, warm)

    cold_seconds = cold["summary"]["wall_time_seconds"]["median"]
    warm_seconds = warm["summary"]["amortized_wall_time_seconds_per_output"]["median"]
    warm_output_count = warm["summary"]["output_count"]
    if warm_output_count <= 1:
        raise ValueError("warm output_count must be greater than 1")
    if warm_seconds <= 0:
        raise ValueError("warm amortized seconds must be positive")

    metrics = {
        "cold_seconds_per_image": cold_seconds,
        "warm_amortized_seconds_per_image": warm_seconds,
        "warm_amortized_speedup_ratio": cold_seconds / warm_seconds,
        "seconds_saved_per_image": cold_seconds - warm_seconds,
        "warm_total_wall_time_seconds": warm["summary"]["total_wall_time_seconds"]["median"],
        "warm_output_count": warm_output_count,
        "max_rss_bytes_delta": warm["summary"]["max_rss_bytes"]["median"] - cold["summary"]["max_rss_bytes"]["median"],
        "peak_footprint_bytes_delta": warm["summary"]["peak_footprint_bytes"]["median"]
        - cold["summary"]["peak_footprint_bytes"]["median"],
        "basis": COMPARISON_BASIS,
    }
    comparison = {
        "schema_version": 1,
        "manifest_type": "benchmark_comparison",
        "created_at": _utc_now(),
        "inputs": {
            "cold": {"path": str(cold_path), "sha256": cold_digest},
            "warm": {"path": str(warm_path), "sha256": warm_digest},
        },
        "compatibility": {
            "status": "PASS",
            "checked_fields": CHECKED_FIELDS,
        },
        "metrics": metrics,
        "conclusion": "warm_faster" if metrics["warm_amortized_speedup_ratio"] > 1.0 else "warm_slower_or_equal",
        "claim_scope": COMPARISON_CLAIM_SCOPE,
    }
    validate_benchmark_comparison_manifest(comparison)
    return comparison


def write_benchmark_comparison(*, cold_manifest: str | Path, warm_manifest: str | Path, output: str | Path) -> dict:
    output_path = ensure_artifact_safe_path(output)
    if output_path.suffix.lower() != ".json":
        raise ValueError("output path must end with .json")

    cold = validate_manifest_file(cold_manifest, "raw_benchmark")
    warm = validate_manifest_file(warm_manifest, "raw_benchmark")
    comparison = compare_benchmark_manifests(
        cold,
        warm,
        cold_path=cold_manifest,
        warm_path=warm_manifest,
        cold_digest=_sha256_file(cold_manifest),
        warm_digest=_sha256_file(warm_manifest),
    )
    _write_json_atomic(output_path, comparison)
    return comparison


def _validate_classes(cold: dict, warm: dict) -> None:
    if cold["benchmark_class"] != "locked_cold_diffusers":
        raise ValueError("cold manifest benchmark_class must be locked_cold_diffusers")
    if warm["benchmark_class"] != "warm_persistent_diffusers":
        raise ValueError("warm manifest benchmark_class must be warm_persistent_diffusers")


def _validate_runtime_scope(cold: dict, warm: dict) -> None:
    allowed = {"diffusers_pytorch_mps", "diffusers_pytorch_cpu"}
    if cold["runtime"]["engine"] not in allowed or warm["runtime"]["engine"] not in allowed:
        raise ValueError("benchmark comparison only supports Diffusers runtime engines")


def _validate_compatibility(cold: dict, warm: dict) -> None:
    for field in CHECKED_FIELDS:
        cold_value = _field_value(cold, field)
        warm_value = _field_value(warm, field)
        if cold_value != warm_value:
            raise ValueError(f"comparison input mismatch: {field}")


def _field_value(data: dict, dotted_path: str):
    value = data
    for part in dotted_path.split("."):
        value = value[part]
    return value


def _sha256_file(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
