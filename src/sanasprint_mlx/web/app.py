import os
from pathlib import Path

from sanasprint_mlx.web.jobs import DEFAULT_MODEL, GenerationJobManager, GenerationRequest


def create_app(*, manager: GenerationJobManager | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as error:  # pragma: no cover
        raise ImportError("SanaSprint WebUI requires the web extra: pip install -e '.[web]'") from error

    class JobRequestPayload(BaseModel):
        prompt: str = Field(min_length=1)
        height: int = 512
        width: int = 512
        steps: int = 2
        seed: int = 42
        count: int = 1
        tiled_decode: bool = True
        snapshot: str = DEFAULT_MODEL
        allow_download: bool = True

    output_dir = Path(os.environ.get("SANASPRINT_WEB_OUTPUT_DIR", "/tmp/sanasprint-mlx-web-outputs"))
    active_manager = manager or GenerationJobManager(output_dir=output_dir)
    app = FastAPI(title="SanaSprint MLX WebUI")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/outputs", StaticFiles(directory=str(active_manager.output_dir)), name="outputs")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ready",
            "default_model": DEFAULT_MODEL,
            "output_dir": str(active_manager.output_dir),
        }

    @app.post("/api/jobs")
    def create_job(payload: JobRequestPayload) -> dict:
        try:
            job = active_manager.submit(GenerationRequest(**payload.model_dump()))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return active_manager.get(job_id).to_dict()
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/api/gallery")
    def gallery() -> dict:
        return {"items": [image.__dict__ for image in active_manager.gallery()]}

    return app


app = create_app()


def main() -> int:
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover
        raise ImportError("SanaSprint WebUI requires the web extra: pip install -e '.[web]'") from error
    uvicorn.run("sanasprint_mlx.web.app:app", host="127.0.0.1", port=8008, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
