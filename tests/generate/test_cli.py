import json

import sanasprint_mlx.cli.generate as generate_cli
from sanasprint_mlx.cli.generate import main


def test_generate_cli_dry_run_writes_phase_plan(tmp_path):
    output = tmp_path / "out.png"
    plan = tmp_path / "plan.json"

    code = main(
        [
            "--prompt",
            "a tiny astronaut",
            "--height",
            "512",
            "--width",
            "512",
            "--steps",
            "2",
            "--seed",
            "7",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--output",
            str(output),
            "--dry-run",
            "--plan-output",
            str(plan),
            "--low-memory",
            "--reference-decode",
        ]
    )

    assert code == 0
    data = json.loads(plan.read_text())
    assert data["request"]["prompt"] == "a tiny astronaut"
    assert [phase["name"] for phase in data["phases"]] == ["text_encode", "denoise", "decode", "write_png"]


def test_generate_cli_dry_run_accepts_prompt_cache(tmp_path):
    plan = tmp_path / "plan.json"

    code = main(
        [
            "--prompt-cache",
            str(tmp_path / "prompt-cache"),
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--dry-run",
            "--plan-output",
            str(plan),
        ]
    )

    assert code == 0
    data = json.loads(plan.read_text())
    assert data["request"]["prompt_cache"] == str(tmp_path / "prompt-cache")
    assert [phase["name"] for phase in data["phases"]] == ["denoise", "decode", "write_png"]


def test_generate_cli_dry_run_accepts_tiled_decode(tmp_path):
    plan = tmp_path / "plan.json"

    code = main(
        [
            "--prompt",
            "decode",
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--dry-run",
            "--plan-output",
            str(plan),
            "--tiled-decode",
        ]
    )

    assert code == 0
    data = json.loads(plan.read_text())
    assert data["phases"][2]["mode"] == "tiled_mlx_decode"


def test_generate_cli_validates_output_suffix(tmp_path):
    code = main(
        [
            "--prompt",
            "bad suffix",
            "--output",
            str(tmp_path / "out.jpg"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--dry-run",
            "--plan-output",
            str(tmp_path / "plan.json"),
        ]
    )

    assert code == 2


def test_generate_cli_reference_pipeline_runs_when_explicitly_requested(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        kwargs["output"].write_bytes(b"png")
        return {"output": str(kwargs["output"])}

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_generation", fake_run)

    code = main(
        [
            "--prompt",
            "real",
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--reference-pipeline",
        ]
    )

    assert code == 0
    assert calls[0]["prompt"] == "real"


def test_generate_cli_non_dry_run_requires_reference_pipeline(tmp_path):
    code = main(
        [
            "--prompt",
            "real",
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            str(tmp_path / "snapshot"),
        ]
    )

    assert code == 2


def test_generate_cli_registers_console_script():
    pyproject = open("pyproject.toml").read()

    assert 'sanasprint-mlx-generate = "sanasprint_mlx.cli.generate:main"' in pyproject
