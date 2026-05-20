import numpy as np
import pytest
import mlx.core as mx

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.transformer.timestep import (
    SanaTimestepGuidanceEmbedding,
    load_timestep_guidance_weights_from_snapshot,
)


def test_timestep_guidance_embedding_produces_block_modulation_and_conditioning():
    embedding = SanaTimestepGuidanceEmbedding(hidden_size=4)
    state = embedding.parameters()
    state["mlx_transformer.time_embed.linear.bias"] = np.arange(24, dtype=np.float32) / 10.0
    embedding.load_parameters(state)

    modulation, conditioning = embedding(
        timestep=np.array([0.5], dtype=np.float32),
        guidance=np.array([4500.0], dtype=np.float32),
    )

    np.testing.assert_allclose(np.array(modulation), np.arange(24, dtype=np.float32).reshape(1, 24) / 10.0)
    assert conditioning.shape == (1, 4)


def test_timestep_guidance_embedding_loads_synthetic_snapshot_weights(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    embedding = SanaTimestepGuidanceEmbedding(hidden_size=4)

    report = load_timestep_guidance_weights_from_snapshot(embedding, snapshot)

    assert len(report["loaded_keys"]) == 10
    assert report["source_tensors"]["mlx_transformer.time_embed.linear.bias"]["target_shape"] == [24]


def test_timestep_guidance_embedding_casts_projection_to_hidden_dtype():
    embedding = SanaTimestepGuidanceEmbedding(hidden_size=4)
    embedding.load_parameters({key: value.astype(mx.float16) for key, value in embedding.parameters().items()})

    modulation, conditioning = embedding(
        timestep=np.array([0.5], dtype=np.float32),
        guidance=np.array([4500.0], dtype=np.float32),
        hidden_dtype=mx.float16,
    )

    assert modulation.dtype == mx.float16
    assert conditioning.dtype == mx.float16


def test_timestep_guidance_embedding_matches_diffusers_reference():
    torch = pytest.importorskip("torch")
    sana_transformer = pytest.importorskip("diffusers.models.transformers.sana_transformer")

    reference = sana_transformer.SanaCombinedTimestepGuidanceEmbeddings(embedding_dim=4)
    rng = np.random.default_rng(123)
    state = {}
    for key, tensor in reference.state_dict().items():
        values = rng.standard_normal(tuple(tensor.shape), dtype=np.float32) / 10.0
        state[key] = torch.from_numpy(values)
    reference.load_state_dict(state)

    embedding = SanaTimestepGuidanceEmbedding(hidden_size=4)
    embedding.load_parameters({f"mlx_transformer.time_embed.{key}": value.numpy() for key, value in state.items()})

    timestep = np.array([0.5, 1.25], dtype=np.float32)
    guidance = np.array([4500.0, 7000.0], dtype=np.float32)
    mlx_modulation, mlx_conditioning = embedding(timestep=timestep, guidance=guidance)
    torch_modulation, torch_conditioning = reference(
        torch.from_numpy(timestep),
        guidance=torch.from_numpy(guidance),
        hidden_dtype=torch.float32,
    )

    np.testing.assert_allclose(np.array(mlx_modulation), torch_modulation.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(mlx_conditioning), torch_conditioning.detach().numpy(), atol=3e-5, rtol=5e-5)
