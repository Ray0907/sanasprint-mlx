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
    assert not [entry for entry in report["mapping"] if entry["status"] == "shape_mismatch"]


def test_weights_cli_load_scaffold_writes_json_diagnostics(tmp_path):
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "load-scaffold.json"
    main(["make-synthetic-snapshot", "--output-dir", str(snapshot)])

    result = main(["load-scaffold", "--snapshot", str(snapshot), "--output", str(output), "--dtype", "float16"])

    assert result == 0
    report = json.loads(output.read_text())
    assert report["loaded_keys"] == [
        "mlx_transformer.patch_embed.proj.weight",
        "mlx_transformer.patch_embed.proj.bias",
        "mlx_transformer.proj_out.weight",
        "mlx_transformer.proj_out.bias",
    ]
    assert report["source_tensors"]["mlx_transformer.patch_embed.proj.weight"]["source_file"] == "transformer/model.safetensors"
    assert report["source_tensors"]["mlx_transformer.patch_embed.proj.weight"]["final_dtype"] == "float16"


def test_weights_cli_load_scaffold_rejects_remote_snapshot_path(tmp_path):
    output = tmp_path / "load-scaffold.json"

    with pytest.raises(SystemExit) as error:
        main(["load-scaffold", "--snapshot", "https://huggingface.co/example/model", "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()


def test_weights_cli_load_scaffold_rejects_unknown_dtype(tmp_path):
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "load-scaffold.json"
    main(["make-synthetic-snapshot", "--output-dir", str(snapshot)])

    with pytest.raises(SystemExit) as error:
        main(["load-scaffold", "--snapshot", str(snapshot), "--output", str(output), "--dtype", "int8"])

    assert error.value.code == 2
    assert not output.exists()


def test_weights_cli_requires_local_snapshot_path(tmp_path):
    output = tmp_path / "report.json"

    with pytest.raises(SystemExit) as error:
        main(["inspect", "--snapshot", "https://huggingface.co/example/model", "--output", str(output)])

    assert error.value.code == 2
    assert not output.exists()
