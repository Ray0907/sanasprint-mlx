import json

import pytest

from sanasprint_mlx.verification.gates import (
    build_verification_report,
    load_pass_evidence,
)


def gate(report, gate_id):
    return report["gates"][gate_id]


def test_missing_fixture_gate_is_blocked():
    report = build_verification_report(env={})

    assert gate(report, "transformer_parity")["status"] == "BLOCKED"
    assert "SANASPRINT_MLX_REAL_FIXTURE" in gate(report, "transformer_parity")["reason"]


def test_existing_fixture_gate_is_ready(tmp_path):
    report = build_verification_report(env={"SANASPRINT_MLX_REAL_FIXTURE": str(tmp_path)})

    assert gate(report, "transformer_parity")["status"] == "READY"


def test_snapshot_gate_blocks_remote_without_allow_download():
    report = build_verification_report(snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers")

    assert gate(report, "snapshot")["status"] == "BLOCKED"


def test_snapshot_gate_remote_with_allow_download_is_ready_not_pass():
    report = build_verification_report(
        snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        allow_download=True,
    )

    assert gate(report, "snapshot")["status"] == "READY"


def test_pass_evidence_can_mark_known_gate_passed(tmp_path):
    evidence = {
        "schema_version": 1,
        "gates": {
            "transformer_parity": {
                "status": "PASS",
                "command": "cmd",
                "artifact": str(tmp_path),
                "observed_at": "2026-05-19T00:00:00Z",
            }
        },
    }

    report = build_verification_report(env={}, pass_evidence=evidence)

    assert gate(report, "transformer_parity")["status"] == "PASS"


def test_pass_evidence_can_mark_scaffold_denoise_passed(tmp_path):
    evidence = {
        "schema_version": 1,
        "gates": {
            "scaffold_denoise": {
                "status": "PASS",
                "command": "cmd",
                "artifact": str(tmp_path),
                "observed_at": "2026-05-20T00:00:00Z",
            }
        },
    }

    report = build_verification_report(env={}, pass_evidence=evidence)

    assert gate(report, "scaffold_denoise")["status"] == "PASS"


def test_pass_evidence_can_mark_block0_attention_passed(tmp_path):
    evidence = {
        "schema_version": 1,
        "gates": {
            "block0_attention": {
                "status": "PASS",
                "command": "cmd",
                "artifact": str(tmp_path),
                "observed_at": "2026-05-20T00:00:00Z",
            }
        },
    }

    report = build_verification_report(env={}, pass_evidence=evidence)

    assert gate(report, "block0_attention")["status"] == "PASS"


def test_invalid_pass_evidence_is_rejected(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"schema_version": 1, "gates": {"feature_9_preconditions": {"status": "PASS"}}}))

    with pytest.raises(ValueError, match="feature_9_preconditions"):
        load_pass_evidence(path)


def test_pass_evidence_fields_must_be_non_empty_strings(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gates": {
                    "transformer_parity": {
                        "status": "PASS",
                        "command": 123,
                        "artifact": str(tmp_path),
                        "observed_at": "2026-05-19T00:00:00Z",
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="command"):
        load_pass_evidence(path)


def test_feature_9_readiness_requires_all_pass_evidence(tmp_path):
    partial = {
        "schema_version": 1,
        "gates": {
            "transformer_parity": {
                "status": "PASS",
                "command": "cmd",
                "artifact": str(tmp_path),
                "observed_at": "2026-05-19T00:00:00Z",
            }
        },
    }

    report = build_verification_report(env={}, pass_evidence=partial)

    assert gate(report, "feature_9_preconditions")["status"] == "BLOCKED"


def test_report_includes_real_verification_commands():
    report = build_verification_report(env={})

    assert "tests/transformer/test_real_fixture_parity.py" in gate(report, "transformer_parity")["command"]
    assert "sanasprint-mlx-verify block0-attention" in gate(report, "block0_attention")["command"]
    assert "sanasprint-mlx-verify scaffold-denoise" in gate(report, "scaffold_denoise")["command"]
    assert "sanasprint-mlx-generate" in gate(report, "smoke_512")["command"]


def test_block0_attention_gate_requires_local_snapshot(tmp_path):
    ready = build_verification_report(env={}, snapshot=tmp_path)
    remote = build_verification_report(
        env={},
        snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        allow_download=True,
    )

    assert gate(ready, "block0_attention")["status"] == "READY"
    assert gate(remote, "block0_attention")["status"] == "BLOCKED"


def test_scaffold_denoise_gate_requires_local_snapshot(tmp_path):
    ready = build_verification_report(env={}, snapshot=tmp_path)
    remote = build_verification_report(
        env={},
        snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        allow_download=True,
    )

    assert gate(ready, "scaffold_denoise")["status"] == "READY"
    assert gate(remote, "scaffold_denoise")["status"] == "BLOCKED"


def test_smoke_command_uses_reference_pipeline_and_allow_download_when_enabled():
    report = build_verification_report(
        env={},
        snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        allow_download=True,
    )

    command = gate(report, "smoke_512")["command"]
    assert "--reference-pipeline" in command
    assert "--allow-download" in command
    assert "--reference-decode" not in command
