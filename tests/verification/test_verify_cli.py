import json

import sanasprint_mlx.cli.verify as verify_cli
from sanasprint_mlx.cli.verify import main


def test_verify_cli_writes_report(tmp_path):
    output = tmp_path / "report.json"

    code = main(["--output", str(output)])

    assert code == 0
    report = json.loads(output.read_text())
    assert "gates" in report


def test_verify_cli_accepts_snapshot_path(tmp_path):
    output = tmp_path / "report.json"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    code = main(["--output", str(output), "--snapshot", str(snapshot)])

    assert code == 0
    report = json.loads(output.read_text())
    assert report["gates"]["snapshot"]["status"] == "READY"


def test_verify_cli_blocks_remote_snapshot_without_allow_download(tmp_path):
    output = tmp_path / "report.json"

    code = main(
        [
            "--output",
            str(output),
            "--snapshot",
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        ]
    )

    assert code == 0
    report = json.loads(output.read_text())
    assert report["gates"]["snapshot"]["status"] == "BLOCKED"


def test_verify_cli_check_hygiene_writes_hygiene_result(tmp_path):
    output = tmp_path / "report.json"

    code = main(["--output", str(output), "--check-hygiene"])

    assert code == 0
    report = json.loads(output.read_text())
    assert report["hygiene"]["status"] == "PASS"


def test_verify_cli_check_hygiene_failure_writes_report_and_returns_error(tmp_path, monkeypatch):
    output = tmp_path / "report.json"

    monkeypatch.setattr(
        verify_cli,
        "check_repository_hygiene",
        lambda: {"status": "FAIL", "violations": [{"category": "denied_prefix", "path": "docs/file.md"}]},
    )

    code = main(["--output", str(output), "--check-hygiene"])

    assert code == 2
    report = json.loads(output.read_text())
    assert report["hygiene"]["status"] == "FAIL"
