import json

import pytest

from sanasprint_mlx.baseline.schema import (
    validate_approved_baseline_manifest,
    validate_manifest_file,
    validate_promotion_manifest,
    validate_raw_benchmark_manifest,
)


def raw_manifest():
    return {
        "schema_version": 1,
        "manifest_type": "raw_benchmark",
        "benchmark_class": "locked_cold_diffusers",
        "created_at": "2026-05-19T00:00:00Z",
        "command": "sanasprint-mlx-generate --reference-pipeline",
        "model": {
            "repo": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            "revision": "abc123",
        },
        "runtime": {
            "engine": "diffusers_pytorch_mps",
            "dtype": "bfloat16",
            "device": "mps",
            "low_memory": True,
        },
        "generation": {
            "prompt_hash": "sha256:abc",
            "seed": 42,
            "width": 512,
            "height": 512,
            "steps": 2,
        },
        "environment": {
            "machine": "Apple M4",
            "memory_gib": 16.0,
            "os_version": "macOS 15",
            "python_version": "3.14",
            "torch_version": "2.9.0",
            "diffusers_version": "0.36.0",
            "mlx_version": "not-used",
        },
        "behavior": {
            "reference_pipeline_required": True,
            "allow_download": False,
            "default_behavior_changed": False,
        },
        "image": {
            "path": "/tmp/sanasprint.png",
            "width": 512,
            "height": 512,
            "mode": "RGB",
            "valid": True,
        },
        "runs": [
            {
                "index": 1,
                "wall_time_seconds": 10.0,
                "max_rss_bytes": 100,
                "peak_footprint_bytes": 120,
                "success": True,
            },
            {
                "index": 2,
                "wall_time_seconds": 12.0,
                "max_rss_bytes": 110,
                "peak_footprint_bytes": 140,
                "success": True,
            },
            {
                "index": 3,
                "wall_time_seconds": 11.0,
                "max_rss_bytes": 105,
                "peak_footprint_bytes": 130,
                "success": True,
            },
        ],
        "summary": {
            "run_count": 3,
            "wall_time_seconds": metric_summary(10.0, 11.0, 12.0),
            "max_rss_bytes": metric_summary(100, 105, 110),
            "peak_footprint_bytes": metric_summary(120, 130, 140),
        },
    }


def warm_manifest():
    data = raw_manifest()
    data["benchmark_class"] = "warm_persistent_diffusers"
    data["runs"] = [
        {
            "index": 1,
            "wall_time_seconds": 30.0,
            "max_rss_bytes": 200,
            "peak_footprint_bytes": 240,
            "timing_source": "wrapper_total",
            "output_count": 2,
            "output_paths": ["/tmp/warm-0001.png", "/tmp/warm-0002.png"],
            "output_seeds": [42, 43],
            "success": True,
        }
    ]
    data["summary"] = {
        "run_count": 1,
        "output_count": 2,
        "total_wall_time_seconds": metric_summary(30.0, 30.0, 30.0),
        "amortized_wall_time_seconds_per_output": metric_summary(15.0, 15.0, 15.0),
        "max_rss_bytes": metric_summary(200, 200, 200),
        "peak_footprint_bytes": metric_summary(240, 240, 240),
    }
    data["image"]["path"] = "/tmp/warm-0002.png"
    return data


def metric_summary(minimum, median, maximum):
    return {
        "min": minimum,
        "median": median,
        "max": maximum,
        "positive_noise_bound": maximum - median,
        "rule": "max_minus_median",
    }


def approved_manifest():
    data = raw_manifest()
    data["manifest_type"] = "approved_baseline"
    data["approval"] = {
        "approved_at": "2026-05-19T00:00:00Z",
        "approved_by": "user",
        "reviewer": "subagent",
        "source_raw_manifest": "/tmp/raw.json",
        "notes": "",
    }
    return data


def promotion_manifest():
    return {
        "schema_version": 1,
        "manifest_type": "promotion",
        "created_at": "2026-05-19T00:00:00Z",
        "old_baseline": "baseline/approved/cold.json",
        "candidate_manifest": "/tmp/candidate.json",
        "intended_change": "make mlx default",
        "behavior_compatibility": "PASS",
        "metric_comparison": "PASS",
        "image_validity": "PASS",
        "user_approval": "chat approval",
        "reviewer_result": "PASS",
        "commit_or_pr": "",
    }


def test_valid_raw_manifest_passes():
    data = raw_manifest()

    assert validate_raw_benchmark_manifest(data) is data


def test_valid_warm_raw_manifest_passes():
    data = warm_manifest()

    assert validate_raw_benchmark_manifest(data) is data


def test_cold_manifest_rejects_warm_run_fields():
    data = raw_manifest()
    data["runs"][0]["timing_source"] = "wrapper_total"

    with pytest.raises(ValueError, match="timing_source"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_cold_summary_shape():
    data = warm_manifest()
    data["summary"] = raw_manifest()["summary"]

    with pytest.raises(ValueError, match="summary.amortized_wall_time_seconds_per_output"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_output_count_mismatch():
    data = warm_manifest()
    data["runs"][0]["output_count"] = 3

    with pytest.raises(ValueError, match="output_count"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_unsafe_output_paths():
    data = warm_manifest()
    data["runs"][0]["output_paths"][0] = "/etc/passwd"

    with pytest.raises(ValueError, match="output_paths"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_relative_output_path_traversal():
    data = warm_manifest()
    data["runs"][0]["output_paths"][0] = "benchmark-runs/../src/tracked.png"

    with pytest.raises(ValueError, match="output_paths"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_accepts_absolute_ignored_repo_output_path():
    data = warm_manifest()
    data["runs"][0]["output_paths"][0] = str((__import__("pathlib").Path.cwd() / "benchmark-runs" / "warm-0001.png"))

    validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_inconsistent_total_timing_summary():
    data = warm_manifest()
    data["summary"]["total_wall_time_seconds"] = metric_summary(99.0, 99.0, 99.0)

    with pytest.raises(ValueError, match="total_wall_time_seconds"):
        validate_raw_benchmark_manifest(data)


def test_warm_manifest_rejects_inconsistent_amortized_timing_summary():
    data = warm_manifest()
    data["summary"]["amortized_wall_time_seconds_per_output"] = metric_summary(1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="amortized_wall_time_seconds_per_output"):
        validate_raw_benchmark_manifest(data)


def test_valid_approved_baseline_passes():
    data = approved_manifest()

    assert validate_approved_baseline_manifest(data) is data


def test_valid_promotion_manifest_passes():
    data = promotion_manifest()

    assert validate_promotion_manifest(data) is data


def test_missing_required_field_names_field():
    data = raw_manifest()
    del data["model"]["revision"]

    with pytest.raises(ValueError, match="model.revision"):
        validate_raw_benchmark_manifest(data)


def test_wrong_manifest_type_raises():
    data = raw_manifest()
    data["manifest_type"] = "approved_baseline"

    with pytest.raises(ValueError, match="manifest_type"):
        validate_raw_benchmark_manifest(data)


def test_invalid_benchmark_class_raises():
    data = raw_manifest()
    data["benchmark_class"] = "other"

    with pytest.raises(ValueError, match="benchmark_class"):
        validate_raw_benchmark_manifest(data)


def test_run_count_mismatch_raises():
    data = raw_manifest()
    data["summary"]["run_count"] = 2

    with pytest.raises(ValueError, match="run_count"):
        validate_raw_benchmark_manifest(data)


def test_invalid_metric_ordering_raises():
    data = raw_manifest()
    data["summary"]["wall_time_seconds"] = {
        "min": 10.0,
        "median": 12.0,
        "max": 11.0,
        "positive_noise_bound": 0.0,
        "rule": "max_minus_median",
    }

    with pytest.raises(ValueError, match="min <= median <= max"):
        validate_raw_benchmark_manifest(data)


def test_positive_noise_bound_mismatch_raises():
    data = raw_manifest()
    data["summary"]["wall_time_seconds"]["positive_noise_bound"] = 99.0

    with pytest.raises(ValueError, match="positive_noise_bound"):
        validate_raw_benchmark_manifest(data)


def test_image_mode_not_rgb_raises():
    data = raw_manifest()
    data["image"]["mode"] = "RGBA"

    with pytest.raises(ValueError, match="image.mode"):
        validate_raw_benchmark_manifest(data)


def test_image_valid_false_raises():
    data = raw_manifest()
    data["image"]["valid"] = False

    with pytest.raises(ValueError, match="image.valid"):
        validate_raw_benchmark_manifest(data)


def test_run_success_false_raises():
    data = raw_manifest()
    data["runs"][0]["success"] = False

    with pytest.raises(ValueError, match=r"runs\[0\].success"):
        validate_raw_benchmark_manifest(data)


def test_default_behavior_changed_true_raises():
    data = raw_manifest()
    data["behavior"]["default_behavior_changed"] = True

    with pytest.raises(ValueError, match="default_behavior_changed"):
        validate_raw_benchmark_manifest(data)


def test_approved_baseline_rejects_opt_in_mlx_hybrid():
    data = approved_manifest()
    data["benchmark_class"] = "opt_in_mlx_hybrid"

    with pytest.raises(ValueError, match="benchmark_class"):
        validate_approved_baseline_manifest(data)


def test_diffusers_runtime_requires_torch_and_diffusers_versions():
    data = raw_manifest()
    data["environment"]["torch_version"] = "not-used"

    with pytest.raises(ValueError, match="torch_version"):
        validate_raw_benchmark_manifest(data)


def test_mlx_runtime_requires_mlx_and_not_used_torch_diffusers():
    data = raw_manifest()
    data["runtime"]["engine"] = "mlx"
    data["environment"]["torch_version"] = "not-used"
    data["environment"]["diffusers_version"] = "not-used"
    data["environment"]["mlx_version"] = "0.29.0"

    validate_raw_benchmark_manifest(data)

    data["environment"]["torch_version"] = "2.9.0"
    with pytest.raises(ValueError, match="torch_version"):
        validate_raw_benchmark_manifest(data)


def test_hybrid_runtime_requires_all_versions():
    data = raw_manifest()
    data["runtime"]["engine"] = "hybrid"
    data["environment"]["mlx_version"] = "not-used"

    with pytest.raises(ValueError, match="mlx_version"):
        validate_raw_benchmark_manifest(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("generation", "seed"), True),
        (("generation", "width"), True),
        (("environment", "memory_gib"), True),
        (("runs", 0, "wall_time_seconds"), True),
    ],
)
def test_bool_is_rejected_for_numeric_fields(path, value):
    data = raw_manifest()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        validate_raw_benchmark_manifest(data)


def test_validate_manifest_file_loads_json_and_dispatches(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw_manifest()))

    assert validate_manifest_file(path, "raw_benchmark")["manifest_type"] == "raw_benchmark"


def test_validate_manifest_file_supports_approved_and_promotion(tmp_path):
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(approved_manifest()))
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps(promotion_manifest()))

    assert validate_manifest_file(approved, "approved_baseline")["manifest_type"] == "approved_baseline"
    assert validate_manifest_file(promotion, "promotion")["manifest_type"] == "promotion"


def test_validate_manifest_file_rejects_unsupported_kind(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw_manifest()))

    with pytest.raises(ValueError, match="unsupported manifest kind"):
        validate_manifest_file(path, "other")


def test_validate_manifest_file_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")

    with pytest.raises(ValueError, match="manifest JSON is invalid"):
        validate_manifest_file(path, "raw_benchmark")


def test_validate_manifest_file_rejects_top_level_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="manifest must be an object"):
        validate_manifest_file(path, "raw_benchmark")


def test_unknown_extra_keys_are_rejected():
    data = raw_manifest()
    data["extra"] = "not allowed"

    with pytest.raises(ValueError, match="manifest.extra"):
        validate_raw_benchmark_manifest(data)
