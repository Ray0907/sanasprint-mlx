from pathlib import Path
from types import SimpleNamespace

import pytest

from sanasprint_mlx.generate.reference_bridge import (
    ReferencePipelineSession,
    check_reference_dependencies,
    run_reference_pipeline_batch_generation,
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


def test_reference_pipeline_session_reuses_loaded_pipeline_for_multiple_images(tmp_path):
    calls = {"loads": 0, "seeds": [], "outputs": []}

    class FakeImage:
        def save(self, path):
            calls["outputs"].append(Path(path).name)
            Path(path).write_bytes(b"png")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["loads"] += 1
            return cls()

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            calls["seeds"].append(kwargs["generator"].initial_seed())
            return SimpleNamespace(images=[FakeImage()])

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    session = ReferencePipelineSession(snapshot=snapshot, allow_download=False, pipeline_cls=FakePipeline)

    first = session.generate_one(
        prompt="warm",
        height=64,
        width=64,
        steps=1,
        seed=11,
        output=tmp_path / "first.png",
    )
    second = session.generate_one(
        prompt="warm",
        height=64,
        width=64,
        steps=1,
        seed=12,
        output=tmp_path / "second.png",
    )

    assert calls["loads"] == 1
    assert calls["seeds"] == [11, 12]
    assert calls["outputs"] == ["first.png", "second.png"]
    assert first["seed"] == 11
    assert second["seed"] == 12


def test_reference_pipeline_session_validates_missing_snapshot_before_load(tmp_path):
    calls = {"loads": 0}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["loads"] += 1
            return cls()

    with pytest.raises(FileNotFoundError):
        ReferencePipelineSession(snapshot=tmp_path / "missing", allow_download=False, pipeline_cls=FakePipeline)

    assert calls["loads"] == 0


def test_reference_pipeline_generation_validates_prompt_before_load(tmp_path):
    calls = {"loads": 0}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["loads"] += 1
            return cls()

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="prompt"):
        run_reference_pipeline_generation(
            prompt=None,
            height=64,
            width=64,
            steps=1,
            seed=1,
            output=tmp_path / "out.png",
            snapshot=snapshot,
            allow_download=False,
            pipeline_cls=FakePipeline,
        )

    assert calls["loads"] == 0


def test_reference_pipeline_batch_validates_prompt_before_load(tmp_path):
    calls = {"loads": 0}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["loads"] += 1
            return cls()

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="prompt"):
        run_reference_pipeline_batch_generation(
            prompt=None,
            height=64,
            width=64,
            steps=1,
            seed=1,
            outputs=[tmp_path / "out.png"],
            snapshot=snapshot,
            allow_download=False,
            pipeline_cls=FakePipeline,
        )

    assert calls["loads"] == 0


def test_reference_pipeline_batch_uses_one_session_and_reports_outputs(tmp_path):
    calls = {"loads": 0}

    class FakeImage:
        def save(self, path):
            Path(path).write_bytes(b"png")

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["loads"] += 1
            return cls()

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[FakeImage()])

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    reports = run_reference_pipeline_batch_generation(
        prompt="batch",
        height=64,
        width=64,
        steps=1,
        seed=21,
        outputs=[tmp_path / "image-0001.png", tmp_path / "image-0002.png", tmp_path / "image-0003.png"],
        snapshot=snapshot,
        allow_download=False,
        pipeline_cls=FakePipeline,
    )

    assert calls["loads"] == 1
    assert [report["seed"] for report in reports] == [21, 22, 23]
    assert [Path(report["output"]).name for report in reports] == [
        "image-0001.png",
        "image-0002.png",
        "image-0003.png",
    ]
