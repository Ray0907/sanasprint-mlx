from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class SCMSchedulerConfig:
    num_train_timesteps: int = 1000
    prediction_type: str = "trigflow"
    sigma_data: float = 0.5


@dataclass(frozen=True)
class SCMSchedulerOutput:
    prev_sample: object
    pred_original_sample: object | None = None


class SCMScheduler:
    order = 1

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        prediction_type: str = "trigflow",
        sigma_data: float = 0.5,
    ):
        self.config = SCMSchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            sigma_data=sigma_data,
        )
        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps = mx.array(np.arange(0, num_train_timesteps)[::-1].astype(np.float32))
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        timesteps=None,
        device=None,
        max_timesteps: float | None = 1.57080,
        intermediate_timesteps: float | None = 1.3,
    ) -> None:
        del device
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")
        if timesteps is not None and len(timesteps) != num_inference_steps + 1:
            raise ValueError("If providing custom timesteps, timesteps must be of length num_inference_steps + 1.")
        if timesteps is not None and max_timesteps is not None:
            raise ValueError("If providing custom timesteps, max_timesteps should not be provided.")
        if timesteps is None and max_timesteps is None:
            raise ValueError("Should provide either timesteps or max_timesteps.")
        if intermediate_timesteps is not None and num_inference_steps != 2:
            raise ValueError("Intermediate timesteps for SCM is not supported when num_inference_steps != 2.")

        self.num_inference_steps = num_inference_steps
        if timesteps is not None:
            self.timesteps = mx.array(timesteps, dtype=mx.float32)
        elif intermediate_timesteps is not None:
            self.timesteps = mx.array([max_timesteps, intermediate_timesteps, 0.0], dtype=mx.float32)
        else:
            self.timesteps = mx.array(np.linspace(max_timesteps, 0.0, num_inference_steps + 1, dtype=np.float32))
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None) -> int:
        schedule = np.array(self.timesteps if schedule_timesteps is None else schedule_timesteps, dtype=np.float32)
        value = float(np.array(timestep, dtype=np.float32).reshape(-1)[0])
        indices = np.flatnonzero(np.isclose(schedule, value, atol=1e-6))
        if len(indices) == 0:
            raise ValueError(f"timestep {value} is not in the scheduler timetable")
        position = 1 if len(indices) > 1 else 0
        return int(indices[position])

    def _init_step_index(self, timestep) -> None:
        self._step_index = self._begin_index if self._begin_index is not None else self.index_for_timestep(timestep)

    def step(
        self,
        model_output,
        timestep,
        sample,
        *,
        generator=None,
        noise_fn=None,
        return_dict: bool = True,
    ) -> SCMSchedulerOutput | tuple:
        del generator
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is None; run set_timesteps before step.")
        if self.step_index is None:
            self._init_step_index(timestep)
        if self.step_index is None or self.step_index + 1 >= len(np.array(self.timesteps)):
            raise ValueError("scheduler step index is out of range")

        model_output = mx.array(model_output)
        sample = mx.array(sample)
        current_timestep = self.timesteps[self.step_index]
        next_timestep = self.timesteps[self.step_index + 1]

        if self.config.prediction_type != "trigflow":
            raise ValueError(f"Unsupported parameterization: {self.config.prediction_type}")

        pred_x0 = mx.cos(current_timestep) * sample - mx.sin(current_timestep) * model_output
        if len(np.array(self.timesteps)) > 1:
            if noise_fn is None:
                noise = mx.random.normal(model_output.shape, dtype=model_output.dtype)
            else:
                noise = mx.array(noise_fn(model_output.shape, model_output.dtype), dtype=model_output.dtype)
            noise = noise * self.config.sigma_data
            prev_sample = mx.cos(next_timestep) * pred_x0 + mx.sin(next_timestep) * noise
        else:
            prev_sample = pred_x0

        self._step_index += 1
        if return_dict:
            return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
        return prev_sample, pred_x0

    def __len__(self) -> int:
        return self.config.num_train_timesteps
