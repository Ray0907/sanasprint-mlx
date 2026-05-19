import json
from pathlib import Path

import sanasprint_mlx.cli.compare as compare_cli
from sanasprint_mlx.cli.compare import main
from tests.baseline.test_schema import raw_manifest, warm_manifest


def test_compare_cli_writes_comparison(tmp_path):
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    output = tmp_path / "comparison.json"
    cold.write_text(json.dumps(raw_manifest()))
    warm.write_text(json.dumps(warm_manifest()))

    code = main(["--cold-manifest", str(cold), "--warm-manifest", str(warm), "--output", str(output)])

    assert code == 0
    assert json.loads(output.read_text())["manifest_type"] == "benchmark_comparison"


def test_compare_cli_rejects_mismatch_without_writing(tmp_path):
    cold = tmp_path / "cold.json"
    warm = tmp_path / "warm.json"
    output = tmp_path / "comparison.json"
    cold.write_text(json.dumps(raw_manifest()))
    warm_data = warm_manifest()
    warm_data["runtime"]["device"] = "cpu"
    warm.write_text(json.dumps(warm_data))

    code = main(["--cold-manifest", str(cold), "--warm-manifest", str(warm), "--output", str(output)])

    assert code == 2
    assert not output.exists()


def test_compare_cli_rejects_non_json_output(tmp_path, monkeypatch):
    calls = []

    def fake_write(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(compare_cli, "write_benchmark_comparison", fake_write)

    code = main(["--cold-manifest", "cold.json", "--warm-manifest", "warm.json", "--output", str(tmp_path / "out.txt")])

    assert code == 2
    assert calls == []


def test_compare_cli_registers_console_script():
    pyproject = Path("pyproject.toml").read_text()

    assert 'sanasprint-mlx-compare = "sanasprint_mlx.cli.compare:main"' in pyproject
