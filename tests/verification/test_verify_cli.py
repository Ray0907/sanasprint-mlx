import json

import numpy as np

import sanasprint_mlx.cli.verify as verify_cli
from sanasprint_mlx.cli.verify import main
from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.text.cache import write_prompt_cache


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


def test_verify_cli_scaffold_denoise_writes_report(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    output = tmp_path / "scaffold-denoise.json"

    code = main(["scaffold-denoise", "--snapshot", str(snapshot), "--output", str(output), "--dtype", "float16"])

    assert code == 0
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["prompt_source"] == "synthetic"
    assert report["latents"]["finite"] is True


def test_verify_cli_scaffold_denoise_accepts_prompt_cache(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    output = tmp_path / "scaffold-denoise.json"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 3, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 3), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake-text",
        max_sequence_length=3,
        clean_caption=False,
        complex_human_instruction=[],
    )

    code = main(["scaffold-denoise", "--snapshot", str(snapshot), "--prompt-cache", str(cache), "--output", str(output)])

    assert code == 0
    report = json.loads(output.read_text())
    assert report["prompt_source"] == "prompt_cache"
    assert report["prompt"]["embeds_shape"] == [1, 3, 4]


def test_verify_cli_scaffold_denoise_rejects_remote_snapshot(tmp_path):
    output = tmp_path / "scaffold-denoise.json"

    code = main(["scaffold-denoise", "--snapshot", "Efficient-Large-Model/Sana", "--output", str(output)])

    assert code == 2
    assert not output.exists()
