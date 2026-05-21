from __future__ import annotations

from pathlib import Path

from sanasprint_mlx.web.jobs import GenerationJobManager, GenerationRequest


def test_generation_job_manager_runs_batch_and_records_gallery(tmp_path):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        reports = []
        for index, output in enumerate(kwargs["outputs"]):
            Path(output).write_bytes(b"png")
            reports.append(
                {
                    "output": str(output),
                    "seed": kwargs["seed"] + index,
                    "height": kwargs["height"],
                    "width": kwargs["width"],
                    "steps": kwargs["steps"],
                    "runtime": {"wall_time_seconds": 1.25 + index},
                    "memory": {"max_rss_bytes": 1234},
                    "model": str(kwargs["snapshot"]),
                    "decode_mode": "tiled_mlx_decode" if kwargs["tiled_decode"] else "mlx_decode",
                    "prompt_source": "mlx_text_encoder",
                }
            )
        return reports

    manager = GenerationJobManager(output_dir=tmp_path, generator=fake_generate)
    job = manager.submit(
        GenerationRequest(
            prompt="a portrait in soft light",
            height=512,
            width=512,
            steps=2,
            seed=100,
            count=2,
            tiled_decode=True,
            snapshot="RayyTien/SanaSprint-0.6B-1024px-MLX",
            allow_download=True,
        )
    )

    finished = manager.wait(job.id, timeout=5)

    assert finished.status == "completed"
    assert [image.seed for image in finished.images] == [100, 101]
    assert [Path(image.file_path).name for image in finished.images] == [f"{job.id}-0001.png", f"{job.id}-0002.png"]
    assert calls[0]["prompt"] == "a portrait in soft light"
    assert calls[0]["outputs"] == [tmp_path / f"{job.id}-0001.png", tmp_path / f"{job.id}-0002.png"]
    assert manager.gallery()[0].job_id == job.id


def test_generation_job_manager_marks_failed_jobs(tmp_path):
    def fail_generate(**kwargs):
        raise RuntimeError("model failed")

    manager = GenerationJobManager(output_dir=tmp_path, generator=fail_generate)
    job = manager.submit(
        GenerationRequest(
            prompt="bad",
            height=512,
            width=512,
            steps=2,
            seed=7,
            count=1,
            tiled_decode=True,
            snapshot="local",
            allow_download=False,
        )
    )

    finished = manager.wait(job.id, timeout=5)

    assert finished.status == "failed"
    assert finished.error == "model failed"
