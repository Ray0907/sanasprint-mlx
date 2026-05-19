import numpy as np
import pytest

from sanasprint_mlx.scheduler.scm import SCMScheduler


def test_scm_scheduler_two_step_uses_intermediate_timestep():
    scheduler = SCMScheduler()

    scheduler.set_timesteps(num_inference_steps=2, max_timesteps=1.5708, intermediate_timesteps=1.3)

    np.testing.assert_allclose(np.array(scheduler.timesteps), np.array([1.5708, 1.3, 0.0]), atol=1e-6)


def test_scm_scheduler_linear_timesteps_without_intermediate():
    scheduler = SCMScheduler()

    scheduler.set_timesteps(num_inference_steps=4, max_timesteps=1.6, intermediate_timesteps=None)

    np.testing.assert_allclose(np.array(scheduler.timesteps), np.linspace(1.6, 0.0, 5), atol=1e-6)


def test_scm_scheduler_rejects_invalid_custom_timestep_length():
    scheduler = SCMScheduler()

    with pytest.raises(ValueError, match="length"):
        scheduler.set_timesteps(num_inference_steps=2, timesteps=[1.0, 0.0], max_timesteps=None)


def test_scm_scheduler_rejects_intermediate_for_non_two_step():
    scheduler = SCMScheduler()

    with pytest.raises(ValueError, match="Intermediate"):
        scheduler.set_timesteps(num_inference_steps=4, max_timesteps=1.5708, intermediate_timesteps=1.3)


def test_scm_scheduler_step_matches_numpy_trigflow_without_noise():
    scheduler = SCMScheduler(sigma_data=0.5)
    scheduler.set_timesteps(num_inference_steps=1, max_timesteps=1.5708, intermediate_timesteps=None)
    sample = np.array([[[[0.2, -0.4]]]], dtype=np.float32)
    model_output = np.array([[[[0.1, 0.3]]]], dtype=np.float32)

    prev, pred_x0 = scheduler.step(model_output, np.array([1.5708], dtype=np.float32), sample, return_dict=False)

    expected_pred_x0 = np.cos(1.5708) * sample - np.sin(1.5708) * model_output
    np.testing.assert_allclose(np.array(pred_x0), expected_pred_x0, atol=1e-6)
    np.testing.assert_allclose(np.array(prev), expected_pred_x0, atol=1e-6)


def test_scm_scheduler_step_uses_injected_noise_for_multistep():
    scheduler = SCMScheduler(sigma_data=0.5)
    scheduler.set_timesteps(num_inference_steps=2, max_timesteps=1.5708, intermediate_timesteps=1.3)
    sample = np.ones((1, 1, 1, 2), dtype=np.float32)
    model_output = np.full((1, 1, 1, 2), 0.25, dtype=np.float32)

    prev, pred_x0 = scheduler.step(
        model_output,
        np.array([1.5708], dtype=np.float32),
        sample,
        noise_fn=lambda shape, dtype: np.ones(shape, dtype=np.float32),
        return_dict=False,
    )

    expected_pred_x0 = np.cos(1.5708) * sample - np.sin(1.5708) * model_output
    expected_prev = np.cos(1.3) * expected_pred_x0 + np.sin(1.3) * np.ones_like(sample) * 0.5
    np.testing.assert_allclose(np.array(pred_x0), expected_pred_x0, atol=1e-6)
    np.testing.assert_allclose(np.array(prev), expected_prev, atol=1e-6)


def test_scm_scheduler_step_requires_set_timesteps():
    scheduler = SCMScheduler()

    with pytest.raises(ValueError, match="set_timesteps"):
        scheduler.step(np.zeros((1,)), np.array([1.0], dtype=np.float32), np.zeros((1,)))
