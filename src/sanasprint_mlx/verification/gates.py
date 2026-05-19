from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KNOWN_EVIDENCE_GATES = {
    "transformer_parity",
    "loop_parity",
    "text_parity",
    "decode_parity",
    "smoke_512",
    "smoke_768",
}

REQUIRED_FOR_FEATURE_9 = tuple(sorted(KNOWN_EVIDENCE_GATES))

FIXTURE_GATES = {
    "transformer_parity": (
        "SANASPRINT_MLX_REAL_FIXTURE",
        "SANASPRINT_MLX_REAL_FIXTURE={path} python3 -m pytest tests/transformer/test_real_fixture_parity.py -v",
    ),
    "loop_parity": (
        "SANASPRINT_MLX_REAL_LOOP_FIXTURE",
        "SANASPRINT_MLX_REAL_LOOP_FIXTURE={path} python3 -m pytest tests/pipeline/test_real_loop_fixture_parity.py -v",
    ),
    "text_parity": (
        "SANASPRINT_MLX_REAL_TEXT_FIXTURE",
        "SANASPRINT_MLX_REAL_TEXT_FIXTURE={path} python3 -m pytest tests/text/test_real_text_fixture_parity.py -v",
    ),
    "decode_parity": (
        "SANASPRINT_MLX_REAL_DECODE_FIXTURE",
        "SANASPRINT_MLX_REAL_DECODE_FIXTURE={path} python3 -m pytest tests/autoencoder/test_real_decode_parity.py -v",
    ),
}


def build_verification_report(
    *,
    env: dict[str, str] | None = None,
    snapshot: str | Path | None = None,
    allow_download: bool = False,
    pass_evidence: dict | None = None,
    fixture_overrides: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    env = env or {}
    fixture_overrides = fixture_overrides or {}
    evidence = _validate_pass_evidence_dict(pass_evidence or {"schema_version": 1, "gates": {}})
    gates = {}
    for gate_id, (env_var, command_template) in FIXTURE_GATES.items():
        path = fixture_overrides.get(gate_id) or env.get(env_var)
        gates[gate_id] = _fixture_gate(gate_id, env_var, path, command_template, evidence)
    gates["snapshot"] = _snapshot_gate(snapshot, allow_download)
    gates["smoke_512"] = _smoke_gate(512, snapshot, allow_download, evidence)
    gates["smoke_768"] = _smoke_gate(768, snapshot, allow_download, evidence)
    gates["feature_9_preconditions"] = _feature_9_gate(gates)
    return {"schema_version": 1, "gates": gates}


def load_pass_evidence(path: str | Path) -> dict:
    return _validate_pass_evidence_dict(json.loads(Path(path).read_text()))


def _fixture_gate(gate_id: str, env_var: str, path, command_template: str, evidence: dict) -> dict:
    if gate_id in evidence["gates"]:
        item = dict(evidence["gates"][gate_id])
        return {"id": gate_id, "status": "PASS", "reason": "external pass evidence", **item}
    command = command_template.format(path=path or f"/path/from/{env_var}")
    if not path:
        return {
            "id": gate_id,
            "status": "BLOCKED",
            "reason": f"missing {env_var}",
            "env_var": env_var,
            "path": None,
            "command": command,
        }
    exists = Path(path).exists()
    return {
        "id": gate_id,
        "status": "READY" if exists else "BLOCKED",
        "reason": "path exists" if exists else "path does not exist",
        "env_var": env_var,
        "path": str(path),
        "command": command,
    }


def _snapshot_gate(snapshot, allow_download: bool) -> dict:
    if snapshot is None:
        return {
            "id": "snapshot",
            "status": "BLOCKED",
            "reason": "missing local snapshot path",
            "path": None,
            "allow_download": allow_download,
        }
    if _looks_remote(snapshot):
        return {
            "id": "snapshot",
            "status": "READY" if allow_download else "BLOCKED",
            "reason": "remote snapshot requires explicit download" if allow_download else "remote snapshot requires --allow-download",
            "path": str(snapshot),
            "allow_download": allow_download,
        }
    exists = Path(snapshot).exists()
    return {
        "id": "snapshot",
        "status": "READY" if exists else "BLOCKED",
        "reason": "local snapshot exists" if exists else "local snapshot path does not exist",
        "path": str(snapshot),
        "allow_download": allow_download,
    }


def _smoke_gate(size: int, snapshot, allow_download: bool, evidence: dict) -> dict:
    gate_id = f"smoke_{size}"
    allow_download_arg = " --allow-download" if allow_download else ""
    command = (
        "sanasprint-mlx-generate "
        f"--prompt 'a tiny astronaut hatching from an egg on the moon' --height {size} --width {size} "
        "--steps 2 --seed 7 "
        f"--snapshot {snapshot or '/path/to/Sana_Sprint_0.6B_1024px_diffusers'} "
        f"--output /tmp/sanasprint-mlx-{size}.png --low-memory --reference-pipeline{allow_download_arg}"
    )
    if gate_id in evidence["gates"]:
        item = dict(evidence["gates"][gate_id])
        return {"id": gate_id, "status": "PASS", "reason": "external pass evidence", "command": command, **item}
    snapshot_gate = _snapshot_gate(snapshot, allow_download)
    return {
        "id": gate_id,
        "status": "READY" if snapshot_gate["status"] == "READY" else "BLOCKED",
        "reason": "snapshot ready" if snapshot_gate["status"] == "READY" else "snapshot blocked",
        "command": command,
    }


def _feature_9_gate(gates: dict[str, dict]) -> dict:
    missing = [gate_id for gate_id in REQUIRED_FOR_FEATURE_9 if gates[gate_id]["status"] != "PASS"]
    return {
        "id": "feature_9_preconditions",
        "status": "READY" if not missing else "BLOCKED",
        "reason": "all required pass evidence present" if not missing else "missing pass evidence",
        "missing": missing,
    }


def _validate_pass_evidence_dict(data: dict) -> dict:
    if data.get("schema_version") != 1:
        raise ValueError("pass evidence schema_version must be 1")
    gates = data.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("pass evidence requires gates object")
    for gate_id, evidence in gates.items():
        if gate_id not in KNOWN_EVIDENCE_GATES:
            raise ValueError(f"unknown pass evidence gate: {gate_id}")
        if evidence.get("status") != "PASS":
            raise ValueError(f"pass evidence for {gate_id} must have status PASS")
        for field in ("command", "artifact", "observed_at"):
            value = evidence.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"pass evidence for {gate_id} requires {field}")
    return data


def _looks_remote(snapshot: str | Path) -> bool:
    text = str(snapshot)
    return text.startswith(("http://", "https://", "hf://")) or (
        "/" in text and not text.startswith(("/", "./", "../")) and not Path(text).exists()
    )
