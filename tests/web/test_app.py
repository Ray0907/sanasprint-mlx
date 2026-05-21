from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from sanasprint_mlx.web.app import create_app
from sanasprint_mlx.web.jobs import GenerationJobManager


def test_web_api_creates_job_and_returns_gallery(tmp_path):
    def fake_generate(**kwargs):
        output = Path(kwargs["outputs"][0])
        output.write_bytes(b"png")
        return [
            {
                "output": str(output),
                "seed": kwargs["seed"],
                "height": kwargs["height"],
                "width": kwargs["width"],
                "steps": kwargs["steps"],
                "runtime": {"wall_time_seconds": 2.0},
                "memory": {"max_rss_bytes": 2048},
                "model": str(kwargs["snapshot"]),
                "decode_mode": "tiled_mlx_decode",
                "prompt_source": "mlx_text_encoder",
            }
        ]

    manager = GenerationJobManager(output_dir=tmp_path, generator=fake_generate)
    app = create_app(manager=manager)
    client = testclient.TestClient(app)

    response = client.post(
        "/api/jobs",
        json={
            "prompt": "a glass apple",
            "height": 512,
            "width": 512,
            "steps": 2,
            "seed": 42,
            "count": 1,
            "tiled_decode": True,
            "snapshot": "RayyTien/SanaSprint-0.6B-1024px-MLX",
            "allow_download": True,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    manager.wait(job_id, timeout=5)

    job_response = client.get(f"/api/jobs/{job_id}")
    gallery_response = client.get("/api/gallery")

    assert job_response.json()["status"] == "completed"
    assert job_response.json()["images"][0]["image_url"].startswith("/outputs/")
    assert gallery_response.json()["items"][0]["job_id"] == job_id
