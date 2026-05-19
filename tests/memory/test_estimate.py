from sanasprint_mlx.memory.estimate import (
    BYTES_PER_DTYPE,
    estimate_memory,
    weight_memory_by_component,
)


def weight_report(parameter_count=1_000_000):
    return {
        "schema_version": 1,
        "snapshot_path": "/tmp/snapshot",
        "config_summary": {
            "hidden_size": 64,
            "in_channels": 32,
            "sample_size": 32,
        },
        "components": {
            "text_encoder": {
                "parameter_count": parameter_count,
                "parameter_count_by_dtype": {"F16": parameter_count},
                "largest_tensors": [{"name": "text.weight", "parameter_count": parameter_count, "dtype": "F16"}],
            },
            "transformer": {
                "parameter_count": parameter_count,
                "parameter_count_by_dtype": {"BF16": parameter_count},
                "largest_tensors": [{"name": "transformer.weight", "parameter_count": parameter_count, "dtype": "BF16"}],
            },
            "vae": {
                "parameter_count": parameter_count,
                "parameter_count_by_dtype": {"F32": parameter_count},
                "largest_tensors": [{"name": "decoder.weight", "parameter_count": parameter_count, "dtype": "F32"}],
            },
            "unknown": {
                "parameter_count": 0,
                "parameter_count_by_dtype": {},
                "largest_tensors": [],
            },
        },
    }


def test_weight_memory_uses_parameter_count_by_dtype():
    memory = weight_memory_by_component(weight_report(10))

    assert memory["text_encoder"]["weight_bytes_by_dtype"] == {"F16": 20}
    assert memory["transformer"]["weight_bytes_by_dtype"] == {"BF16": 20}
    assert memory["vae"]["weight_bytes_by_dtype"] == {"F32": 40}
    assert BYTES_PER_DTYPE["I4"] == 0.5


def test_resolution_estimates_apply_budget_statuses():
    report = estimate_memory(weight_report(1_000))

    by_resolution = {item["height"]: item for item in report["resolution_estimates"]}

    assert by_resolution[512]["status"] == "GO"
    assert by_resolution[768]["budget_gb"] == 13
    assert by_resolution[1024]["budget_gb"] == 15


def test_final_decision_blocks_feature_3_when_512_is_no_go():
    report = estimate_memory(
        weight_report(1_000),
        runtime_overhead_bytes=12 * 1024**3,
    )

    assert report["final_decision"]["status"] == "BLOCKED"
    assert report["final_decision"]["can_start_feature_3"] is False
    assert report["final_decision"]["requires_user_approval"] is True
    assert 512 in report["final_decision"]["blocking_resolutions"]


def test_final_decision_blocks_feature_3_when_required_components_missing():
    source = weight_report(1_000)
    del source["components"]["text_encoder"]
    del source["components"]["vae"]

    report = estimate_memory(source)

    assert report["resolution_estimates"][0]["status"] == "UNKNOWN"
    assert report["final_decision"]["status"] == "BLOCKED"
    assert report["final_decision"]["can_start_feature_3"] is False
    assert report["final_decision"]["requires_user_approval"] is True
    assert any("missing required component text_encoder" in warning for warning in report["warnings"])


def test_empty_required_component_is_warning_loaded():
    source = weight_report(1_000)
    source["components"]["vae"]["parameter_count_by_dtype"] = {}
    source["components"]["vae"]["parameter_count"] = 0

    report = estimate_memory(source)

    assert report["component_weight_memory"]["vae"]["warning"] == "missing or empty required component"
    assert report["resolution_estimates"][0]["status"] == "UNKNOWN"


def test_1024_no_go_marks_resolution_experimental():
    source = weight_report(1_000)
    source["config_summary"]["hidden_size"] = 600_000
    report = estimate_memory(
        source,
        runtime_overhead_bytes=int(1.5 * 1024**3),
    )

    assert report["final_decision"]["can_start_feature_3"] is True
    assert report["recommendations"]["1024"] == "experimental"


def test_report_includes_largest_tensor_groups():
    report = estimate_memory(weight_report(1_000))

    estimate_512 = report["resolution_estimates"][0]

    assert estimate_512["largest_tensor_groups"]
    assert estimate_512["largest_tensor_groups"][0]["name"] in {"decoder.weight", "text.weight", "transformer.weight"}


def test_unknown_dtype_warns_and_defaults_to_f32():
    source = weight_report(1_000)
    source["components"]["transformer"]["parameter_count_by_dtype"] = {"WEIRD": 3}

    report = estimate_memory(source)

    assert report["component_weight_memory"]["transformer"]["weight_bytes_by_dtype"] == {"WEIRD": 12}
    assert report["warnings"]


def test_phase_estimates_are_sequential_not_all_components_resident():
    report = estimate_memory(weight_report(1_000_000_000))
    phases = report["phase_estimates"]["512"]

    denoise = phases["denoise"]["estimated_bytes"]
    all_components = sum(
        component["total_weight_bytes"]
        for component in report["component_weight_memory"].values()
    )

    assert denoise < all_components + report["runtime_overhead_bytes"]
