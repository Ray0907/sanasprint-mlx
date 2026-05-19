import json

import pytest

from sanasprint_mlx.cli.weights import main


def test_weights_cli_help_imports(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "inspect" in capsys.readouterr().out


def test_weights_cli_make_synthetic_snapshot(tmp_path, capsys):
    snapshot = tmp_path / "snapshot"

    result = main(["make-synthetic-snapshot", "--output-dir", str(snapshot)])

    assert result == 0
    assert (snapshot / "transformer" / "config.json").exists()
    assert (snapshot / "transformer" / "model.safetensors").exists()
    assert "wrote synthetic snapshot" in capsys.readouterr().out


def test_weights_cli_inspect_writes_json_report(tmp_path):
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "report.json"
    main(["make-synthetic-snapshot", "--output-dir", str(snapshot)])

    result = main(["inspect", "--snapshot", str(snapshot), "--output", str(output)])

    assert result == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["components"]["transformer"]["parameter_count"] > 0
    assert report["mapping"]


def test_weights_cli_requires_local_snapshot_path(tmp_path):
    output = tmp_path / "report.json"

    with pytest.raises(SystemExit) as error:
        main(["inspect", "--snapshot", "https://huggingface.co/example/model", "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()
