from pathlib import Path
from types import SimpleNamespace

import pytest

from sanasprint_mlx.generate.reference_bridge import (
    check_reference_dependencies,
    run_reference_pipeline_generation,
    validate_real_generation_request,
    write_synthetic_png,
)


def test_reference_bridge_reports_missing_dependencies():
    report = check_reference_dependencies(packages=("definitely_missing_sanasprint_dependency",))

    assert report["available"] is False
    assert "definitely_missing_sanasprint_dependency" in report["missing"]


def test_write_synthetic_png(tmp_path):
    output = tmp_path / "synthetic.png"

    write_synthetic_png(output, width=8, height=8)

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_real_generate_requires_snapshot_or_allow_download():
    with pytest.raises(ValueError, match="snapshot"):
        validate_real_generation_request(snapshot=None, allow_download=False)


def test_real_generate_rejects_missing_local_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_real_generation_request(snapshot=Path(tmp_path / "missing"), allow_download=False)


def test_reference_pipeline_generation_saves_image_with_injected_pipeline(tmp_path):
    class FakeImage:
        def save(self, path):
            Path(path).write_bytes(b"png")

    class FakePipeline:
        dtype = "fake"

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == str(tmp_path / "snapshot")
            assert kwargs["local_files_only"] is True
            return cls()

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            assert kwargs["prompt"] == "run"
            assert kwargs["height"] == 64
            assert kwargs["width"] == 64
            return SimpleNamespace(images=[FakeImage()])

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "out.png"

    report = run_reference_pipeline_generation(
        prompt="run",
        height=64,
        width=64,
        steps=1,
        seed=7,
        output=output,
        snapshot=snapshot,
        allow_download=False,
        pipeline_cls=FakePipeline,
    )

    assert output.read_bytes() == b"png"
    assert report["output"] == str(output)
