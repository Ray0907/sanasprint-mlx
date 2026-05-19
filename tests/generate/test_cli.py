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
    assert calls[0]["torch_dtype"] == "bfloat16"


def test_generate_cli_reference_pipeline_batch_uses_output_dir_and_count(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run_batch(**kwargs):
        calls.append(kwargs)
        for output in kwargs["outputs"]:
            output.write_bytes(b"png")
        return [
            {
                "output": str(output),
                "model": str(kwargs["snapshot"]),
                "height": kwargs["height"],
                "width": kwargs["width"],
                "steps": kwargs["steps"],
                "seed": kwargs["seed"] + index,
                "device": "mps",
                "low_memory": kwargs["low_memory"],
                "torch_dtype": kwargs["torch_dtype"],
            }
            for index, output in enumerate(kwargs["outputs"])
        ]

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_batch_generation", fake_run_batch)

    code = main(
        [
            "--prompt",
            "warm",
            "--output",
            str(tmp_path / "sample.png"),
            "--output-dir",
            str(tmp_path / "batch"),
            "--count",
            "3",
            "--seed",
            "30",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--reference-pipeline",
            "--low-memory",
        ]
    )

    assert code == 0
    assert [path.name for path in calls[0]["outputs"]] == ["sample-0001.png", "sample-0002.png", "sample-0003.png"]
    assert calls[0]["seed"] == 30
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 3
    assert [item["seed"] for item in report["outputs"]] == [30, 31, 32]
    assert report["model"] == str(tmp_path / "snapshot")


def test_generate_cli_reference_pipeline_batch_requires_output_dir_before_generation(tmp_path, monkeypatch):
    calls = []

    def fake_run_batch(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_batch_generation", fake_run_batch)

    code = main(
        [
            "--prompt",
            "warm",
            "--output",
            str(tmp_path / "sample.png"),
            "--count",
            "2",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--reference-pipeline",
        ]
    )

    assert code == 2
    assert calls == []


def test_generate_cli_rejects_non_positive_count_before_generation(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"output": str(kwargs["output"])}

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_generation", fake_run)

    code = main(
        [
            "--prompt",
            "warm",
            "--output",
            str(tmp_path / "sample.png"),
            "--count",
            "0",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--reference-pipeline",
        ]
    )

    assert code == 2
    assert calls == []


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


def test_generate_cli_non_dry_run_without_reference_pipeline_does_not_call_reference(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
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
        ]
    )

    assert code == 2
    assert calls == []


def test_generate_cli_remote_snapshot_requires_allow_download_before_reference_call(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"output": str(kwargs["output"])}

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_generation", fake_run)

    code = main(
        [
            "--prompt",
            "real",
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            "--reference-pipeline",
        ]
    )

    assert code == 2
    assert calls == []


def test_generate_cli_remote_snapshot_with_allow_download_reaches_reference_call(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"output": str(kwargs["output"])}

    monkeypatch.setattr(generate_cli, "run_reference_pipeline_generation", fake_run)

    code = main(
        [
            "--prompt",
            "real",
            "--output",
            str(tmp_path / "out.png"),
            "--snapshot",
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            "--allow-download",
            "--reference-pipeline",
        ]
    )

    assert code == 0
    assert calls[0]["allow_download"] is True


def test_generate_cli_registers_console_script():
    pyproject = open("pyproject.toml").read()

    assert 'sanasprint-mlx-generate = "sanasprint_mlx.cli.generate:main"' in pyproject
