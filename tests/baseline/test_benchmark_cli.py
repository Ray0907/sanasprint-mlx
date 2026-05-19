from pathlib import Path

import sanasprint_mlx.cli.benchmark as benchmark_cli
from sanasprint_mlx.cli.benchmark import main
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO


def test_benchmark_cli_rejects_missing_snapshot(tmp_path):
    code = main(["--prompt", "p", "--snapshot", str(tmp_path / "missing"), "--output", str(tmp_path / "raw.json")])

    assert code == 2


def test_benchmark_cli_rejects_remote_snapshot(tmp_path):
    code = main(
        [
            "--prompt",
            "p",
            "--snapshot",
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            "--output",
            str(tmp_path / "raw.json"),
        ]
    )

    assert code == 2


def test_benchmark_cli_rejects_invalid_runs(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    code = main(["--prompt", "p", "--snapshot", str(snapshot), "--output", str(tmp_path / "raw.json"), "--runs", "0"])

    assert code == 2


def test_benchmark_cli_rejects_non_json_output(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    code = main(["--prompt", "p", "--snapshot", str(snapshot), "--output", str(tmp_path / "raw.txt")])

    assert code == 2


def test_benchmark_cli_rejects_unsafe_output_path(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_run(**kwargs):
        raise AssertionError("runner should not be called")

    monkeypatch.setattr(benchmark_cli, "run_locked_cold_diffusers_benchmark", fake_run)

    code = main(["--prompt", "p", "--snapshot", str(snapshot), "--output", "baseline/raw.json"])

    assert code == 2


def test_benchmark_cli_calls_runner_with_defaults(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(benchmark_cli, "run_locked_cold_diffusers_benchmark", fake_run)

    code = main(
        [
            "--prompt",
            "p",
            "--snapshot",
            str(snapshot),
            "--revision",
            "r",
            "--output",
            str(tmp_path / "raw.json"),
        ]
    )

    assert code == 0
    assert calls[0]["model_repo"] == MODEL_REPO
    assert calls[0]["runs"] == 3
    assert calls[0]["torch_dtype"] == "bfloat16"


def test_benchmark_cli_registers_console_script():
    pyproject = Path("pyproject.toml").read_text()

    assert 'sanasprint-mlx-benchmark = "sanasprint_mlx.cli.benchmark:main"' in pyproject
