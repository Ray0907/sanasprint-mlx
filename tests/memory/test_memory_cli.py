import json

import pytest

from sanasprint_mlx.cli.memory import main


def weight_report(parameter_count=1_000):
    return {
        "schema_version": 1,
        "snapshot_path": "/tmp/snapshot",
        "config_summary": {"hidden_size": 64, "in_channels": 32, "sample_size": 32},
        "components": {
            "transformer": {
                "parameter_count": parameter_count,
                "parameter_count_by_dtype": {"BF16": parameter_count},
                "largest_tensors": [],
            }
        },
    }


def test_memory_cli_help_imports(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "estimate" in capsys.readouterr().out


def test_memory_cli_estimate_writes_json(tmp_path):
    weight_report_path = tmp_path / "weights.json"
    output = tmp_path / "memory.json"
    weight_report_path.write_text(json.dumps(weight_report(1_000)))

    result = main(["estimate", "--weight-report", str(weight_report_path), "--output", str(output), "--skip-mlx-probe"])

    assert result == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert "final_decision" in report
    assert report["mlx_probe"]["available"] is False


def test_memory_cli_rejects_remote_report_path(tmp_path):
    output = tmp_path / "memory.json"

    with pytest.raises(SystemExit) as error:
        main(["estimate", "--weight-report", "https://example.com/report.json", "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()
