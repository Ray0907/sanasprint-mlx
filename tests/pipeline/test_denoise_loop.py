from types import SimpleNamespace

import numpy as np

from sanasprint_mlx.pipeline.denoise import run_denoising_loop
from sanasprint_mlx.scheduler.scm import SCMScheduler


class RecordingTransformer:
    dtype = np.float32

    def __init__(self, scale=0.0):
        self.config = SimpleNamespace(guidance_embeds_scale=1000.0)
        self.scale = scale
        self.calls = []

    def __call__(
        self,
        hidden_states,
        *,
        encoder_hidden_states,
        encoder_attention_mask,
        guidance,
        timestep,
        return_dict=False,
        attention_kwargs=None,
    ):
        self.calls.append(
            {
                "hidden_states": np.array(hidden_states),
                "encoder_hidden_states": np.array(encoder_hidden_states),
                "encoder_attention_mask": np.array(encoder_attention_mask),
                "guidance": np.array(guidance),
                "timestep": np.array(timestep),
            }
        )
        return (np.array(hidden_states, dtype=np.float32) * self.scale,)


def cached_inputs():
    return {
        "latents": np.array([[[[0.2, -0.4], [0.1, 0.3]]]], dtype=np.float32),
        "prompt_embeds": np.ones((1, 3, 4), dtype=np.float32),
        "prompt_attention_mask": np.ones((1, 3), dtype=np.int32),
    }


def test_denoise_loop_calls_transformer_for_each_step():
    transformer = RecordingTransformer()
    scheduler = SCMScheduler()

    run_denoising_loop(transformer=transformer, scheduler=scheduler, num_inference_steps=2, **cached_inputs())

    assert len(transformer.calls) == 2


def test_denoise_loop_passes_scaled_guidance_and_scm_timestep():
    transformer = RecordingTransformer()
    scheduler = SCMScheduler()

    run_denoising_loop(
        transformer=transformer,
        scheduler=scheduler,
        num_inference_steps=1,
        intermediate_timesteps=None,
        guidance_scale=4.5,
        max_timesteps=1.5708,
        **cached_inputs(),
    )

    first = transformer.calls[0]
    expected_scm = np.sin(1.5708) / (np.cos(1.5708) + np.sin(1.5708))
    np.testing.assert_allclose(first["guidance"], np.array([4500.0], dtype=np.float32))
    np.testing.assert_allclose(first["timestep"], np.array([expected_scm], dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(first["encoder_attention_mask"], np.ones((1, 3), dtype=np.int32))


def test_denoise_loop_matches_numpy_reference_with_fake_transformer():
    transformer = RecordingTransformer(scale=0.0)
    scheduler = SCMScheduler(sigma_data=0.5)
    inputs = cached_inputs()

    result = run_denoising_loop(
        transformer=transformer,
        scheduler=scheduler,
        num_inference_steps=1,
        intermediate_timesteps=None,
        guidance_scale=4.5,
        max_timesteps=1.5708,
        **inputs,
    )

    scm = np.sin(1.5708) / (np.cos(1.5708) + np.sin(1.5708))
    scale = np.sqrt(scm**2 + (1.0 - scm) ** 2)
    latent_model_input = inputs["latents"] * scale
    noise_pred = ((1.0 - 2.0 * scm) * latent_model_input) / scale
    noise_pred = noise_pred * 0.5
    sample = inputs["latents"] * 0.5
    expected_pred_x0 = np.cos(1.5708) * sample - np.sin(1.5708) * noise_pred
    expected_final = expected_pred_x0 / 0.5
    np.testing.assert_allclose(np.array(result.latents), expected_final, atol=1e-5)


def test_denoise_loop_returns_debug_steps():
    transformer = RecordingTransformer()
    scheduler = SCMScheduler()

    result = run_denoising_loop(transformer=transformer, scheduler=scheduler, num_inference_steps=2, debug=True, **cached_inputs())

    assert len(result.debug_steps) == 2
    assert result.debug_steps[0]["step_index"] == 0
    assert result.debug_steps[0]["latent_shape"] == [1, 1, 2, 2]
