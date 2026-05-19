from __future__ import annotations

import mlx.core as mx

from sanasprint_mlx.primitives.embeddings import guidance_embedding, sinusoidal_embedding


def conditioning_vector(timestep, guidance, *, dim: int):
    timestep = mx.array(timestep)
    guidance = mx.array(guidance)
    return sinusoidal_embedding(timestep, dim) + guidance_embedding(guidance, dim)
