import json
from pathlib import Path

import pytest

from sanasprint_mlx.baseline.benchmark import (
    RunnerResult,
    build_benchmark_command,
    ensure_artifact_safe_path,
    infer_revision_from_snapshot,
    prompt_hash,
    run_locked_cold_diffusers_benchmark,
    summarize_metric,
)
from sanasprint_mlx.baseline.schema import validate_raw_benchmark_manifest
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO


def test_summarize_metric_computes_noise_bound():
    assert summarize_metric([3, 1, 2]) == {
        "min": 1,
        "median": 2,
        "max": 3,
        "positive_noise_bound": 1,
        "rule": "max_minus_median",
    }


def test_artifact_safe_path_allows_tmp(tmp_path):
    assert ensure_artifact_safe_path(tmp_path / "raw.json") == tmp_path / "raw.json"


def test_artifact_safe_path_allows_tmp_without_git(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("git should not be queried for temp paths")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.run", fail_if_called)

    output = Path("/tmp") / "sanasprint-mlx-test-raw.json"

    assert ensure_artifact_safe_path(output) == output


def test_artifact_safe_path_allows_ignored_repo_roots(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path = root / "benchmark-runs" / "raw.json"
    root.mkdir()

    def fake_run(args, **kwargs):
        class Result:
            returncode = 0
            stdout = str(path)
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert ensure_artifact_safe_path(path, repo_root=root) == path


def test_artifact_safe_path_rejects_allowed_root_when_not_ignored(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path = root / "benchmark-runs" / "raw.json"
    root.mkdir()

    def fake_run(args, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(ValueError, match="not ignored"):
        ensure_artifact_safe_path(path, repo_root=root)


@pytest.mark.parametrize("relative", ["assets/out.json", "baseline/raw.json", "src/raw.json", "tests/raw.json", "raw.json"])
def test_artifact_safe_path_rejects_unsafe_repo_paths(tmp_path, relative):
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(ValueError):
        ensure_artifact_safe_path(root / relative, repo_root=root)


def test_build_benchmark_command_uses_reference_pipeline_without_download(tmp_path):
    command = build_benchmark_command(
        prompt="p",
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "out.png",
        height=512,
        width=512,
        steps=2,
        seed=42,
        torch_dtype="bfloat16",
        low_memory=True,
    )

    assert "--reference-pipeline" in command
    assert "--allow-download" not in command
    assert "--low-memory" in command


def test_run_benchmark_builds_valid_manifest(tmp_path):
    snapshot = tmp_path / "models--repo--name" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    output = tmp_path / "raw.json"
    output_dir = tmp_path / "runs"

    manifest = run_locked_cold_diffusers_benchmark(
        prompt="koi",
        snapshot=snapshot,
        output=output,
        output_dir=output_dir,
        runs=3,
        runner=fake_runner(),
        clock=fake_clock(),
        environment_collector=fake_environment,
        image_metadata_collector=fake_image_metadata,
    )

    validate_raw_benchmark_manifest(manifest)
    assert output.exists()
    assert manifest["benchmark_class"] == "locked_cold_diffusers"
    assert manifest["model"]["repo"] == MODEL_REPO
    assert manifest["model"]["revision"] == "a" * 40
    assert manifest["behavior"] == {
        "reference_pipeline_required": True,
        "allow_download": False,
        "default_behavior_changed": False,
    }
    assert manifest["summary"]["run_count"] == 3
    assert manifest["image"]["path"].endswith("run-3.png")


def test_prompt_hash_is_deterministic():
    assert prompt_hash("abc") == prompt_hash("abc")
    assert prompt_hash("abc") != prompt_hash("def")


def test_failed_run_raises_and_does_not_write_manifest(tmp_path):
    snapshot = tmp_path / "models--repo--name" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    output = tmp_path / "raw.json"

    with pytest.raises(RuntimeError, match="failed"):
        run_locked_cold_diffusers_benchmark(
            prompt="koi",
            snapshot=snapshot,
            output=output,
            output_dir=tmp_path / "runs",
            runner=lambda command: RunnerResult(1, "", "boom", 1, 1),
            clock=fake_clock(),
            environment_collector=fake_environment,
            image_metadata_collector=fake_image_metadata,
        )

    assert not output.exists()


def test_validation_failure_does_not_write_manifest(tmp_path):
    snapshot = tmp_path / "models--repo--name" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    output = tmp_path / "raw.json"

    def bad_environment():
        data = fake_environment()
        data["torch_version"] = "not-used"
        return data

    with pytest.raises(ValueError, match="torch_version"):
        run_locked_cold_diffusers_benchmark(
            prompt="koi",
            snapshot=snapshot,
            output=output,
            output_dir=tmp_path / "runs",
            runner=fake_runner(),
            clock=fake_clock(),
            environment_collector=bad_environment,
            image_metadata_collector=fake_image_metadata,
        )

    assert not output.exists()
    assert not Path(str(output) + ".tmp").exists()


def test_infer_revision_from_hf_snapshot_path(tmp_path):
    snapshot = tmp_path / "snapshots" / ("0123456789abcdef0123456789abcdef01234567")

    assert infer_revision_from_snapshot(snapshot) == "0123456789abcdef0123456789abcdef01234567"


def test_infer_revision_rejects_non_hex_or_non_snapshot(tmp_path):
    assert infer_revision_from_snapshot(tmp_path / "snapshots" / ("g" * 40)) is None
    assert infer_revision_from_snapshot(tmp_path / ("a" * 40)) is None


def test_missing_revision_for_non_hf_snapshot_raises(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="revision"):
        run_locked_cold_diffusers_benchmark(
            prompt="koi",
            snapshot=snapshot,
            output=tmp_path / "raw.json",
            output_dir=tmp_path / "runs",
            runner=fake_runner(),
            clock=fake_clock(),
            environment_collector=fake_environment,
            image_metadata_collector=fake_image_metadata,
        )


def fake_runner():
    def run(command):
        output = command[command.index("--output") + 1]
        report = {
            "device": "mps",
            "height": 512,
            "low_memory": False,
            "model": "/local/snapshot",
            "output": output,
            "seed": 42,
            "steps": 2,
            "torch_dtype": "bfloat16",
            "width": 512,
        }
        return RunnerResult(0, "noise\n" + json.dumps(report), "stderr text", 100, 120)

    return run


def fake_clock():
    values = iter([0.0, 10.0, 10.0, 22.0, 22.0, 33.0])
    return lambda: next(values)


def fake_environment():
    return {
        "machine": "Apple M4",
        "memory_gib": 16.0,
        "os_version": "macOS",
        "python_version": "3.14",
        "torch_version": "2.9.0",
        "diffusers_version": "0.36.0",
        "mlx_version": "not-used",
    }


def fake_image_metadata(path):
    return {
        "path": str(path),
        "width": 512,
        "height": 512,
        "mode": "RGB",
        "valid": True,
    }
