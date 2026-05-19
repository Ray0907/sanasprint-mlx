from __future__ import annotations

import numpy as np


def build_step_telemetry(
    *,
    step_index: int,
    timestep,
    scm_timestep,
    latents,
    memory_snapshot=None,
) -> dict:
    latents_array = np.array(latents)
    telemetry = {
        "step_index": step_index,
        "timestep": _rounded_list(timestep),
        "scm_timestep": _rounded_list(scm_timestep),
        "latent_shape": list(latents_array.shape),
        "latent_dtype": str(latents_array.dtype),
    }
    if memory_snapshot is not None:
        telemetry["memory"] = memory_snapshot()
    return telemetry


def _rounded_list(values) -> list[float]:
    return [round(float(value), 6) for value in np.array(values, dtype=np.float32).reshape(-1)]
