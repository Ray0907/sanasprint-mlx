import json
from pathlib import Path

import pytest

from sanasprint_mlx.baseline.comparison import (
    CHECKED_FIELDS,
    compare_benchmark_manifests,
    write_benchmark_comparison,
)
from sanasprint_mlx.baseline.schema import validate_benchmark_comparison_manifest
from tests.baseline.test_schema import metric_summary, raw_manifest, warm_manifest


def test_compare_benchmark_manifests_computes_exact_metrics(tmp_path):
    cold = raw_manifest()
    warm = warm_manifest()
    cold["summary"]["wall_time_seconds"] = metric_summary(35.0, 40.0, 45.0)
    cold["summary"]["max_rss_bytes"] = metric_summary(90, 100, 110)
    cold["summary"]["peak_footprint_bytes"] = metric_summary(120, 130, 140)
    warm["runs"][0]["wall_time_seconds"] = 40.0
    warm["runs"][0]["max_rss_bytes"] = 115
    warm["runs"][0]["peak_footprint_bytes"] = 120
    warm["summary"]["total_wall_time_seconds"] = metric_summary(40.0, 40.0, 40.0)
    warm["summary"]["amortized_wall_time_seconds_per_output"] = metric_summary(20.0, 20.0, 20.0)
    warm["summary"]["max_rss_bytes"] = metric_summary(115, 115, 115)
    warm["summary"]["peak_footprint_bytes"] = metric_summary(120, 120, 120)

    comparison = compare_benchmark_manifests(
        cold,
        warm,
        cold_path=tmp_path / "cold.json",
        warm_path=tmp_path / "warm.json",
        cold_digest="sha256:" + "a" * 64,
        warm_digest="sha256:" + "b" * 64,
    )

    validate_benchmark_comparison_manifest(comparison)
    assert comparison["compatibility"] == {"status": "PASS", "checked_fields": CHECKED_FIELDS}
    assert comparison["metrics"]["cold_seconds_per_image"] == 40.0
    assert comparison["metrics"]["warm_amortized_seconds_per_image"] == 20.0
    assert comparison["metrics"]["warm_amortized_speedup_ratio"] == 2.0
    assert comparison["metrics"]["seconds_saved_per_image"] == 20.0
    assert comparison["metrics"]["max_rss_bytes_delta"] == 15
    assert comparison["metrics"]["peak_footprint_bytes_delta"] == -10
    assert comparison["conclusion"] == "warm_faster"
    assert comparison["claim_scope"] == "diffusers_warm_persistence_only_not_mlx_native"


def test_compare_rejects_wrong_classes(tmp_path):
    cold = warm_manifest()
    warm = warm_manifest()

    with pytest.raises(ValueError, match="locked_cold_diffusers"):
        compare_benchmark_manifests(
            cold,
            warm,
            cold_path=tmp_path / "cold.json",
            warm_path=tmp_path / "warm.json",
            cold_digest="sha256:" + "a" * 64,
            warm_digest="sha256:" + "b" * 64,
        )


def test_compare_rejects_mismatched_generation(tmp_path):
    cold = raw_manifest()
    warm = warm_manifest()
    warm["generation"]["seed"] = 43

    with pytest.raises(ValueError, match="generation.seed"):
        compare_benchmark_manifests(
            cold,
            warm,
            cold_path=tmp_path / "cold.json",
            warm_path=tmp_path / "warm.json",
            cold_digest="sha256:" + "a" * 64,
            warm_digest="sha256:" + "b" * 64,
        )


def test_compare_rejects_warm_output_count_one(tmp_path):
    cold = raw_manifest()
    warm = warm_manifest()
    warm["runs"][0]["output_count"] = 1
    warm["runs"][0]["output_paths"] = ["/tmp/warm-0001.png"]
    warm["runs"][0]["output_seeds"] = [42]
    warm["summary"]["output_count"] = 1
    warm["summary"]["amortized_wall_time_seconds_per_output"] = warm["summary"]["total_wall_time_seconds"]

    with pytest.raises(ValueError, match="output_count"):
        compare_benchmark_manifests(
            cold,
            warm,
            cold_path=tmp_path / "cold.json",
            warm_path=tmp_path / "warm.json",
            cold_digest="sha256:" + "a" * 64,
            warm_digest="sha256:" + "b" * 64,
        )


def test_write_comparison_writes_atomically_and_validates(tmp_path):
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    output = tmp_path / "comparison.json"
    cold.write_text(json.dumps(raw_manifest()))
    warm.write_text(json.dumps(warm_manifest()))

    comparison = write_benchmark_comparison(cold_manifest=cold, warm_manifest=warm, output=output)

    assert comparison == json.loads(output.read_text())
    validate_benchmark_comparison_manifest(comparison)
    assert comparison["inputs"]["cold"]["path"] == str(cold)
    assert comparison["inputs"]["cold"]["sha256"].startswith("sha256:")


def test_write_comparison_does_not_write_on_failure(tmp_path):
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    output = tmp_path / "comparison.json"
    cold_data = raw_manifest()
    warm_data = warm_manifest()
    warm_data["model"]["revision"] = "different"
    cold.write_text(json.dumps(cold_data))
    warm.write_text(json.dumps(warm_data))

    with pytest.raises(ValueError, match="model.revision"):
        write_benchmark_comparison(cold_manifest=cold, warm_manifest=warm, output=output)

    assert not output.exists()
    assert not Path(str(output) + ".tmp").exists()


def test_write_comparison_rejects_unsafe_output_path(tmp_path):
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    cold.write_text(json.dumps(raw_manifest()))
    warm.write_text(json.dumps(warm_manifest()))

    with pytest.raises(ValueError, match="artifact path"):
        write_benchmark_comparison(cold_manifest=cold, warm_manifest=warm, output="baseline/comparison.json")
