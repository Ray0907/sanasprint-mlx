from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

DENIED_PREFIXES = (
    "docs/",
    "logs/",
    "benchmark-runs/",
    ".benchmarks/",
    "raw-benchmarks/",
    "weights/",
    "checkpoints/",
    "fixtures/real/",
    ".fixtures/",
    "fixture-dumps/",
    "scratch-plans/",
    "scratch-outputs/",
)
DENIED_EXTENSIONS = (".log", ".safetensors", ".ckpt", ".pt", ".pth", ".bin")
ASSET_ALLOWLIST = "assets/allowlist.txt"


def classify_hygiene_violations(tracked_files: Iterable[str], asset_allowlist: set[str]) -> list[dict]:
    violations = []
    for raw_path in tracked_files:
        path = _normalize_path(raw_path)
        denied_prefix = next((prefix for prefix in DENIED_PREFIXES if path.startswith(prefix)), None)
        if denied_prefix:
            violations.append({"category": "denied_prefix", "path": path, "detail": denied_prefix})

        denied_extension = next((extension for extension in DENIED_EXTENSIONS if path.endswith(extension)), None)
        if denied_extension:
            violations.append({"category": "denied_extension", "path": path, "detail": denied_extension})

        if path.endswith(".png") and not path.startswith("assets/"):
            violations.append({"category": "generated_image", "path": path})

        if path.startswith("assets/") and path != ASSET_ALLOWLIST:
            asset_name = path.removeprefix("assets/")
            if asset_name not in asset_allowlist:
                violations.append({"category": "unallowlisted_asset", "path": path, "detail": asset_name})
    return violations


def load_asset_allowlist(root: str | Path | None = None) -> set[str]:
    repo_root = Path.cwd() if root is None else Path(root)
    path = repo_root / ASSET_ALLOWLIST
    lines = path.read_text().splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def list_tracked_files(root: str | Path | None = None) -> list[str]:
    cwd = Path.cwd() if root is None else Path(root)
    repo_root = _git(["rev-parse", "--show-toplevel"], cwd=cwd).strip()
    output = _git(["ls-files"], cwd=Path(repo_root))
    return [_normalize_path(line) for line in output.splitlines() if line.strip()]


def check_repository_hygiene(root: str | Path | None = None) -> dict:
    try:
        cwd = Path.cwd() if root is None else Path(root)
        repo_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=cwd).strip())
    except (OSError, subprocess.CalledProcessError) as error:
        violations = [
            {
                "category": "git_error",
                "path": "",
                "message": f"Unable to inspect tracked repository files: {error}",
            }
        ]
        return {"status": "FAIL", "violations": violations}
    except ValueError as error:
        violations = [
            {
                "category": "git_error",
                "path": "",
                "message": str(error),
            }
        ]
        return {"status": "FAIL", "violations": violations}

    try:
        allowlist = load_asset_allowlist(repo_root)
        tracked_files = list_tracked_files(repo_root)
        violations = classify_hygiene_violations(tracked_files, allowlist)
    except FileNotFoundError as error:
        violations = [
            {
                "category": "allowlist_error",
                "path": ASSET_ALLOWLIST,
                "message": f"Unable to read asset allowlist: {error}",
            }
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        violations = [
            {
                "category": "git_error",
                "path": "",
                "message": f"Unable to inspect tracked repository files: {error}",
            }
        ]
    except ValueError as error:
        violations = [
            {
                "category": "git_error",
                "path": "",
                "message": str(error),
            }
        ]
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def _git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/")
