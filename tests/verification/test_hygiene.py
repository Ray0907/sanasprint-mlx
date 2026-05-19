import subprocess

import sanasprint_mlx.verification.hygiene as hygiene
from sanasprint_mlx.verification.hygiene import (
    check_repository_hygiene,
    classify_hygiene_violations,
)


def categories_for(paths, allowlist=None):
    categories = {}
    for violation in classify_hygiene_violations(paths, allowlist or set()):
        categories.setdefault(violation["path"], set()).add(violation["category"])
    return categories


def test_hygiene_classifies_denied_prefixes():
    categories = categories_for(
        [
            "docs/file.md",
            "logs/run.txt",
            "benchmark-runs/run.json",
            ".benchmarks/run.json",
            "raw-benchmarks/run.json",
            "weights/model.index.json",
            "checkpoints/step-1/config.json",
            "fixtures/real/tensor.npz",
            ".fixtures/tensor.npz",
            "fixture-dumps/tensor.npz",
            "scratch-plans/a.md",
            "scratch-outputs/out.png",
        ]
    )

    assert all("denied_prefix" in categories[path] for path in categories)


def test_hygiene_classifies_denied_extensions():
    categories = categories_for(["run.log", "model.safetensors", "model.ckpt", "model.pt", "model.pth", "model.bin"])

    assert all(categories[path] == {"denied_extension"} for path in categories)


def test_hygiene_classifies_generated_images_outside_assets():
    categories = categories_for(["out.png"])

    assert categories["out.png"] == {"generated_image"}


def test_hygiene_classifies_unallowlisted_assets():
    categories = categories_for(["assets/other.png", "assets/subdir/foo.png"], {"showcase-reference-pipeline.png"})

    assert categories["assets/other.png"] == {"unallowlisted_asset"}
    assert categories["assets/subdir/foo.png"] == {"unallowlisted_asset"}


def test_hygiene_allows_allowlisted_asset_and_allowlist_file():
    violations = classify_hygiene_violations(
        ["assets/showcase-reference-pipeline.png", "assets/allowlist.txt"],
        {"showcase-reference-pipeline.png"},
    )

    assert violations == []


def test_current_repository_passes_hygiene():
    result = check_repository_hygiene()

    assert result["status"] == "PASS"


def test_hygiene_reports_git_error_when_git_cannot_run(monkeypatch):
    def fail_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = check_repository_hygiene()

    assert result["status"] == "FAIL"
    assert result["violations"][0]["category"] == "git_error"


def test_hygiene_reports_allowlist_error_when_allowlist_is_missing(tmp_path, monkeypatch):
    def fake_git(args, *, cwd):
        if args == ["rev-parse", "--show-toplevel"]:
            return str(tmp_path)
        if args == ["ls-files"]:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(hygiene, "_git", fake_git)

    result = check_repository_hygiene(tmp_path)

    assert result["status"] == "FAIL"
    assert result["violations"][0]["category"] == "allowlist_error"
