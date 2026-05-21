from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from sanasprint_mlx.generate.mlx_native import run_mlx_batch_generation


DEFAULT_MODEL = "RayyTien/SanaSprint-0.6B-1024px-MLX"
JobStatus = Literal["queued", "running", "completed", "failed"]
Generator = Callable[..., list[dict]]


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 2
    seed: int = 42
    count: int = 1
    tiled_decode: bool = True
    snapshot: str = DEFAULT_MODEL
    allow_download: bool = True


@dataclass(frozen=True)
class GeneratedImage:
    job_id: str
    file_path: str
    image_url: str
    seed: int
    height: int
    width: int
    steps: int
    runtime_seconds: float | None
    max_rss_bytes: int | None
    model: str
    decode_mode: str
    prompt_source: str


@dataclass
class GenerationJob:
    id: str
    request: GenerationRequest
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    images: list[GeneratedImage] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["request"] = asdict(self.request)
        payload["images"] = [asdict(image) for image in self.images]
        return payload


class GenerationJobManager:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        generator: Generator = run_mlx_batch_generation,
        max_workers: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, GenerationJob] = {}

    def submit(self, request: GenerationRequest) -> GenerationJob:
        self._validate_request(request)
        job = GenerationJob(id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.id] = job
        self.executor.submit(self._run_job, job.id)
        return self.get(job.id)

    def get(self, job_id: str) -> GenerationJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._copy_job(self._jobs[job_id])

    def wait(self, job_id: str, *, timeout: float) -> GenerationJob:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(job_id)
            if job.status in ("completed", "failed"):
                return job
            time.sleep(0.01)
        raise TimeoutError(job_id)

    def gallery(self) -> list[GeneratedImage]:
        with self._lock:
            images = [image for job in self._jobs.values() for image in job.images if job.status == "completed"]
        return sorted(images, key=lambda image: image.file_path, reverse=True)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
            request = job.request
        try:
            outputs = self._output_paths(job_id, request.count)
            reports = self.generator(
                prompt=request.prompt,
                prompt_cache=None,
                height=request.height,
                width=request.width,
                steps=request.steps,
                seed=request.seed,
                outputs=outputs,
                snapshot=request.snapshot,
                tiled_decode=request.tiled_decode,
                allow_download=request.allow_download,
            )
            images = [self._image_from_report(job_id, report) for report in reports]
            with self._lock:
                job = self._jobs[job_id]
                job.status = "completed"
                job.completed_at = time.time()
                job.images = images
        except Exception as error:  # pragma: no cover - exercised through public failure state
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.completed_at = time.time()
                job.error = str(error)

    def _output_paths(self, job_id: str, count: int) -> list[Path]:
        if count == 1:
            return [self.output_dir / f"{job_id}.png"]
        return [self.output_dir / f"{job_id}-{index:04d}.png" for index in range(1, count + 1)]

    def _image_from_report(self, job_id: str, report: dict) -> GeneratedImage:
        path = Path(report["output"])
        return GeneratedImage(
            job_id=job_id,
            file_path=str(path),
            image_url=f"/outputs/{path.name}",
            seed=int(report["seed"]),
            height=int(report["height"]),
            width=int(report["width"]),
            steps=int(report["steps"]),
            runtime_seconds=_nested_float(report, "runtime", "wall_time_seconds"),
            max_rss_bytes=_nested_int(report, "memory", "max_rss_bytes"),
            model=str(report.get("model", "")),
            decode_mode=str(report.get("decode_mode", "")),
            prompt_source=str(report.get("prompt_source", "")),
        )

    def _validate_request(self, request: GenerationRequest) -> None:
        if not request.prompt.strip():
            raise ValueError("prompt is required")
        if request.height not in (512, 768):
            raise ValueError("height must be 512 or 768")
        if request.width not in (512, 768):
            raise ValueError("width must be 512 or 768")
        if request.steps <= 0:
            raise ValueError("steps must be positive")
        if request.count <= 0 or request.count > 4:
            raise ValueError("count must be between 1 and 4")

    def _copy_job(self, job: GenerationJob) -> GenerationJob:
        return GenerationJob(
            id=job.id,
            request=job.request,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            images=list(job.images),
        )


def _nested_float(payload: dict, key: str, nested_key: str) -> float | None:
    value = payload.get(key, {}).get(nested_key)
    return None if value is None else float(value)


def _nested_int(payload: dict, key: str, nested_key: str) -> int | None:
    value = payload.get(key, {}).get(nested_key)
    return None if value is None else int(value)
