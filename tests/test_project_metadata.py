import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_project_targets_sanasprint_06b():
    readme = (ROOT / "README.md").read_text()

    assert "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" in readme
    assert "16GB unified memory" in readme


def test_reference_dependencies_are_optional():
    pyproject = load_pyproject()
    dependencies = set(pyproject["project"]["dependencies"])
    optional = pyproject["project"]["optional-dependencies"]

    assert "torch" not in dependencies
    assert "diffusers" not in dependencies
    assert "transformers" not in dependencies
    assert {"accelerate", "torch", "diffusers", "transformers"}.issubset(set(optional["reference"]))
