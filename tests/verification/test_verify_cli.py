import json

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
