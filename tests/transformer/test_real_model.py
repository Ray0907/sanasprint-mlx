import numpy as np

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler
from sanasprint_mlx.transformer.real_model import RealSanaTransformerDenoiser


def test_real_sana_transformer_loads_snapshot_and_runs_forward(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=2)

    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot,
        sample_size=2,
        block_count=2,
        dtype="float16",
    )

    output = transformer(
        np.ones((1, 4, 2, 2), dtype=np.float32),
        encoder_hidden_states=np.ones((1, 3, 4), dtype=np.float32),
        encoder_attention_mask=np.ones((1, 3), dtype=np.int32),
        guidance=np.array([4500.0], dtype=np.float32),
        timestep=np.array([0.5], dtype=np.float32),
        return_dict=False,
    )[0]

    assert transformer.weight_report["loaded_keys"]["total_count"] == 66
    assert transformer.weight_report["block_count"] == 2
    assert output.shape == (1, 4, 2, 2)
    assert np.isfinite(np.array(output)).all()


def test_real_sana_transformer_runs_inside_denoising_loop(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=1)
    transformer = RealSanaTransformerDenoiser.from_snapshot(
        snapshot,
        sample_size=2,
        block_count=1,
        dtype="float16",
    )

    result = run_denoising_loop(
        transformer=transformer,
        scheduler=SCMScheduler(),
        latents=np.ones((1, 4, 2, 2), dtype=np.float32),
        prompt_embeds=np.ones((1, 3, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 3), dtype=np.int32),
        num_inference_steps=1,
        intermediate_timesteps=None,
    )

    assert result.latents.shape == (1, 4, 2, 2)
    assert np.isfinite(np.array(result.latents)).all()
