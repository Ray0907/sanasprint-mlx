import json

from sanasprint_mlx.weights.inspect import TensorInfo
from sanasprint_mlx.weights.mapping import (
    Diagnostic,
    MappingEntry,
    build_mapping_report,
    component_summaries,
)


def infos():
    return [
        TensorInfo("transformer/model.safetensors", "transformer.patch_embed.proj.weight", [4, 3, 1, 1], "F32", 12, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.patch_embed.proj.bias", [4], "F32", 4, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.proj_out.weight", [3, 4], "F32", 12, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.proj_out.bias", [3], "F32", 3, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.transformer_blocks.0.attn1.to_q.weight", [4, 4], "F32", 16, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.transformer_blocks.0.ff.net.0.proj.weight", [8, 4], "F32", 32, "transformer"),
        TensorInfo("text_encoder/model.safetensors", "text_encoder.embed_tokens.weight", [10, 4], "F16", 40, "text_encoder"),
        TensorInfo("vae/model.safetensors", "decoder.conv.weight", [3, 4, 3, 3], "F32", 108, "vae"),
        TensorInfo("misc/model.safetensors", "misc.weight", [2, 2], "F32", 4, "unknown"),
    ]


def test_mapper_maps_known_transformer_prefixes():
    report = build_mapping_report(infos(), snapshot_path="/tmp/snapshot")

    mapped = {entry.source_key: entry for entry in report.mapping if entry.status == "mapped"}

    assert mapped["transformer.patch_embed.proj.bias"].target_key == "mlx_transformer.patch_embed.proj.bias"
    assert mapped["transformer.patch_embed.proj.bias"].component == "transformer"


def test_mapper_reports_unexpected_keys():
    report = build_mapping_report(infos(), snapshot_path="/tmp/snapshot")

    unexpected = [entry for entry in report.mapping if entry.status == "unexpected"]

    assert unexpected
    assert unexpected[0].source_key == "misc.weight"
    assert unexpected[0].suggested_action


def test_mapper_reports_missing_required_patterns():
    report = build_mapping_report([], snapshot_path="/tmp/snapshot")

    missing = [diagnostic for diagnostic in report.diagnostics if diagnostic.kind == "missing"]
    missing_entries = [entry for entry in report.mapping if entry.status == "missing"]

    assert missing
    assert missing[0].expected_pattern
    assert missing[0].suggested_action
    assert missing_entries
    assert missing_entries[0].target_key == "mlx_transformer.patch_embed.*"


def test_mapper_reports_shape_mismatch_from_config_summary():
    report = build_mapping_report(
        infos(),
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 5, "patch_size": 1},
    )

    mismatch = [entry for entry in report.mapping if entry.status == "shape_mismatch"][0]

    assert mismatch.source_key == "transformer.patch_embed.proj.weight"
    assert mismatch.source_shape == [4, 3, 1, 1]
    assert mismatch.suggested_action == "verify config dimensions or update the mapping rule"


def test_mapper_accepts_4d_patch_embed_shape_from_config_summary():
    report = build_mapping_report(
        infos(),
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 3, "patch_size": 1},
    )

    entry = [entry for entry in report.mapping if entry.source_key == "transformer.patch_embed.proj.weight"][0]

    assert entry.status == "mapped"
    assert entry.target_key == "mlx_transformer.patch_embed.proj.weight"


def test_mapper_uses_patch_size_for_patch_embed_shape():
    tensor_infos = [
        TensorInfo("transformer/model.safetensors", "transformer.patch_embed.proj.weight", [4, 3, 2, 2], "F32", 48, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer.patch_embed.proj.bias", [4], "F32", 4, "transformer"),
    ]

    report = build_mapping_report(
        tensor_infos,
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 3, "patch_size": 2},
    )

    entry = [entry for entry in report.mapping if entry.source_key == "transformer.patch_embed.proj.weight"][0]

    assert entry.status == "mapped"


def test_mapper_validates_proj_out_shape_from_config_summary():
    report = build_mapping_report(
        infos(),
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 3, "out_channels": 3, "patch_size": 1},
    )

    mapped = {entry.source_key: entry for entry in report.mapping if entry.status == "mapped"}

    assert mapped["transformer.proj_out.weight"].target_key == "mlx_transformer.proj_out.weight"
    assert mapped["transformer.proj_out.bias"].target_key == "mlx_transformer.proj_out.bias"


def test_mapper_reports_proj_out_shape_mismatch_from_config_summary():
    report = build_mapping_report(
        infos(),
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 3, "out_channels": 5, "patch_size": 1},
    )

    mismatches = {entry.source_key: entry for entry in report.mapping if entry.status == "shape_mismatch"}

    assert mismatches["transformer.proj_out.weight"].source_shape == [3, 4]
    assert mismatches["transformer.proj_out.bias"].source_shape == [3]


def test_mapper_required_patterns_accept_transformer_file_keys_without_transformer_prefix():
    tensor_infos = [
        TensorInfo("transformer/model.safetensors", "patch_embed.proj.weight", [4, 3, 1, 1], "F32", 12, "transformer"),
        TensorInfo("transformer/model.safetensors", "patch_embed.proj.bias", [4], "F32", 4, "transformer"),
        TensorInfo("transformer/model.safetensors", "transformer_blocks.0.attn1.to_q.weight", [4, 4], "F32", 16, "transformer"),
    ]

    report = build_mapping_report(
        tensor_infos,
        snapshot_path="/tmp/snapshot",
        config_summary={"hidden_size": 4, "in_channels": 3, "patch_size": 1},
    )

    missing_entries = [entry for entry in report.mapping if entry.status == "missing"]

    assert missing_entries == []


def test_mapper_required_patterns_ignore_unprefixed_keys_from_non_transformer_components():
    tensor_infos = [
        TensorInfo("misc/model.safetensors", "patch_embed.proj.weight", [4, 3, 1, 1], "F32", 12, "unknown"),
        TensorInfo("misc/model.safetensors", "transformer_blocks.0.attn1.to_q.weight", [4, 4], "F32", 16, "unknown"),
    ]

    report = build_mapping_report(tensor_infos, snapshot_path="/tmp/snapshot")

    missing_entries = [entry for entry in report.mapping if entry.status == "missing"]

    assert {entry.source_key for entry in missing_entries} == {
        "transformer.patch_embed.*",
        "transformer.transformer_blocks.*",
    }


def test_mapper_marks_transpose_sensitive_keys_for_review():
    report = build_mapping_report(infos(), snapshot_path="/tmp/snapshot")

    review = [
        entry
        for entry in report.mapping
        if entry.source_key == "transformer.transformer_blocks.0.attn1.to_q.weight"
    ][0]

    assert review.status == "requires_review"
    assert review.transpose_required == "unknown"


def test_mapper_diagnostics_include_actionable_fields():
    diagnostic = Diagnostic(
        severity="warning",
        kind="shape_mismatch",
        owning_rule="patch_embed",
        expected_pattern="transformer.patch_embed.*",
        matched_key="transformer.patch_embed.proj.weight",
        actual_shape=[4, 3],
        expected_shape=[4, 4],
        suggested_action="verify patch embedding dimensions",
    )

    assert diagnostic.to_dict()["suggested_action"] == "verify patch embedding dimensions"


def test_component_summary_groups_text_encoder_transformer_and_vae():
    summaries = component_summaries(infos())

    assert summaries["text_encoder"].parameter_count == 40
    assert summaries["text_encoder"].parameter_count_by_dtype == {"F16": 40}
    assert summaries["transformer"].tensor_count == 6
    assert summaries["vae"].largest_tensors[0]["name"] == "decoder.conv.weight"
    assert summaries["unknown"].parameter_count == 4


def test_mapping_report_serializes_to_json():
    report = build_mapping_report(infos(), snapshot_path="/tmp/snapshot")
    data = report.to_dict()

    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == 1
    assert decoded["components"]["transformer"]["parameter_count"] == 79
    assert decoded["mapping"]
