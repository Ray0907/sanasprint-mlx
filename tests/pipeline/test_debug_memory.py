import numpy as np

from sanasprint_mlx.pipeline.debug import build_step_telemetry


def test_debug_telemetry_records_step_shapes_and_timestep():
    telemetry = build_step_telemetry(
        step_index=1,
        timestep=np.array([1.3], dtype=np.float32),
        scm_timestep=np.array([0.7], dtype=np.float32),
        latents=np.zeros((1, 4, 2, 2), dtype=np.float32),
    )

    assert telemetry["step_index"] == 1
    assert telemetry["timestep"] == [1.3]
    assert telemetry["scm_timestep"] == [0.7]
    assert telemetry["latent_shape"] == [1, 4, 2, 2]
    assert telemetry["latent_dtype"] == "float32"


def test_debug_telemetry_allows_memory_snapshot_callback():
    telemetry = build_step_telemetry(
        step_index=0,
        timestep=np.array([1.0], dtype=np.float32),
        scm_timestep=np.array([0.5], dtype=np.float32),
        latents=np.zeros((1, 4, 2, 2), dtype=np.float32),
        memory_snapshot=lambda: {"rss_bytes": 1234},
    )

    assert telemetry["memory"] == {"rss_bytes": 1234}
