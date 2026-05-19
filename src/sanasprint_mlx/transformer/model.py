from __future__ import annotations

import mlx.core as mx

from sanasprint_mlx.primitives.feed_forward import linear
from sanasprint_mlx.primitives.norm import rms_norm
from sanasprint_mlx.primitives.patch import patchify_nchw, unpatchify_nchw
from sanasprint_mlx.transformer.block import ToyTransformerBlock
from sanasprint_mlx.transformer.conditioning import conditioning_vector
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.debug import DebugCapture


PATCH_EMBED_WEIGHT_KEY = "mlx_transformer.patch_embed.proj.weight"
PATCH_EMBED_BIAS_KEY = "mlx_transformer.patch_embed.proj.bias"
PROJ_OUT_WEIGHT_KEY = "mlx_transformer.proj_out.weight"
PROJ_OUT_BIAS_KEY = "mlx_transformer.proj_out.bias"
CAPTION_LINEAR_1_WEIGHT_KEY = "mlx_transformer.caption_projection.linear_1.weight"
CAPTION_LINEAR_1_BIAS_KEY = "mlx_transformer.caption_projection.linear_1.bias"
CAPTION_LINEAR_2_WEIGHT_KEY = "mlx_transformer.caption_projection.linear_2.weight"
CAPTION_LINEAR_2_BIAS_KEY = "mlx_transformer.caption_projection.linear_2.bias"
CAPTION_NORM_WEIGHT_KEY = "mlx_transformer.caption_norm.weight"
SCAFFOLD_PARAMETER_KEYS = (
    PATCH_EMBED_WEIGHT_KEY,
    PATCH_EMBED_BIAS_KEY,
    PROJ_OUT_WEIGHT_KEY,
    PROJ_OUT_BIAS_KEY,
)
CAPTION_PARAMETER_KEYS = (
    CAPTION_LINEAR_1_WEIGHT_KEY,
    CAPTION_LINEAR_1_BIAS_KEY,
    CAPTION_LINEAR_2_WEIGHT_KEY,
    CAPTION_LINEAR_2_BIAS_KEY,
    CAPTION_NORM_WEIGHT_KEY,
)


class SanaTransformerDenoiser:
    def __init__(self, config: SanaTransformerConfig):
        config.validate()
        self.config = config
        self.blocks = [ToyTransformerBlock(config.hidden_size) for _ in range(config.num_layers)]
        self.input_weight = mx.ones((config.hidden_size, config.in_channels * config.patch_size * config.patch_size))
        self.input_bias = mx.zeros((config.hidden_size,))
        self.output_weight = mx.ones((config.out_channels * config.patch_size * config.patch_size, config.hidden_size))
        self.output_bias = mx.zeros((config.out_channels * config.patch_size * config.patch_size,))
        self.caption_projection_weight = self._build_caption_projection_weight()
        self.caption_projection_bias = mx.zeros((config.hidden_size,))
        self.caption_linear_1_weight = self.caption_projection_weight
        self.caption_linear_1_bias = mx.zeros((config.hidden_size,))
        self.caption_linear_2_weight = mx.eye(config.hidden_size)
        self.caption_linear_2_bias = mx.zeros((config.hidden_size,))
        self.caption_norm_weight = mx.ones((config.hidden_size,))
        self.caption_projection_source = "synthetic"

    def parameters(self) -> dict:
        patch_shape = (
            self.config.hidden_size,
            self.config.in_channels,
            self.config.patch_size,
            self.config.patch_size,
        )
        return {
            PATCH_EMBED_WEIGHT_KEY: mx.array(self.input_weight.reshape(patch_shape)),
            PATCH_EMBED_BIAS_KEY: mx.array(self.input_bias),
            PROJ_OUT_WEIGHT_KEY: mx.array(self.output_weight),
            PROJ_OUT_BIAS_KEY: mx.array(self.output_bias),
        }

    def load_parameters(self, parameters: dict, *, strict: bool = True) -> None:
        expected_shapes = self._external_parameter_shapes()
        unknown = [key for key in parameters if key not in expected_shapes]
        if unknown:
            raise KeyError(unknown[0])
        if strict:
            missing = [key for key in SCAFFOLD_PARAMETER_KEYS if key not in parameters]
            if missing:
                raise KeyError(missing[0])

        for key in SCAFFOLD_PARAMETER_KEYS:
            if key not in parameters:
                continue
            value = mx.array(parameters[key])
            expected_shape = expected_shapes[key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{key}: expected shape {expected_shape}, got {tuple(value.shape)}")
            if key == PATCH_EMBED_WEIGHT_KEY:
                self.input_weight = value.reshape(self.input_weight.shape)
            elif key == PATCH_EMBED_BIAS_KEY:
                self.input_bias = value
            elif key == PROJ_OUT_WEIGHT_KEY:
                self.output_weight = value
            elif key == PROJ_OUT_BIAS_KEY:
                self.output_bias = value

    def caption_parameters(self) -> dict:
        return {
            CAPTION_LINEAR_1_WEIGHT_KEY: mx.array(self.caption_linear_1_weight),
            CAPTION_LINEAR_1_BIAS_KEY: mx.array(self.caption_linear_1_bias),
            CAPTION_LINEAR_2_WEIGHT_KEY: mx.array(self.caption_linear_2_weight),
            CAPTION_LINEAR_2_BIAS_KEY: mx.array(self.caption_linear_2_bias),
            CAPTION_NORM_WEIGHT_KEY: mx.array(self.caption_norm_weight),
        }

    def load_caption_parameters(self, parameters: dict, *, strict: bool = True) -> None:
        expected_shapes = self._caption_parameter_shapes()
        unknown = [key for key in parameters if key not in expected_shapes]
        if unknown:
            raise KeyError(unknown[0])
        if strict:
            missing = [key for key in CAPTION_PARAMETER_KEYS if key not in parameters]
            if missing:
                raise KeyError(missing[0])

        for key in CAPTION_PARAMETER_KEYS:
            if key not in parameters:
                continue
            value = mx.array(parameters[key])
            expected_shape = expected_shapes[key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{key}: expected shape {expected_shape}, got {tuple(value.shape)}")
            if key == CAPTION_LINEAR_1_WEIGHT_KEY:
                self.caption_linear_1_weight = value
            elif key == CAPTION_LINEAR_1_BIAS_KEY:
                self.caption_linear_1_bias = value
            elif key == CAPTION_LINEAR_2_WEIGHT_KEY:
                self.caption_linear_2_weight = value
            elif key == CAPTION_LINEAR_2_BIAS_KEY:
                self.caption_linear_2_bias = value
            elif key == CAPTION_NORM_WEIGHT_KEY:
                self.caption_norm_weight = value
        if all(key in parameters for key in CAPTION_PARAMETER_KEYS):
            self.caption_projection_source = "real_weights"

    def _external_parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        patch_size = self.config.patch_size
        return {
            PATCH_EMBED_WEIGHT_KEY: (self.config.hidden_size, self.config.in_channels, patch_size, patch_size),
            PATCH_EMBED_BIAS_KEY: (self.config.hidden_size,),
            PROJ_OUT_WEIGHT_KEY: (self.config.out_channels * patch_size * patch_size, self.config.hidden_size),
            PROJ_OUT_BIAS_KEY: (self.config.out_channels * patch_size * patch_size,),
        }

    def _caption_parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            CAPTION_LINEAR_1_WEIGHT_KEY: (self.config.hidden_size, self.config.caption_channels),
            CAPTION_LINEAR_1_BIAS_KEY: (self.config.hidden_size,),
            CAPTION_LINEAR_2_WEIGHT_KEY: (self.config.hidden_size, self.config.hidden_size),
            CAPTION_LINEAR_2_BIAS_KEY: (self.config.hidden_size,),
            CAPTION_NORM_WEIGHT_KEY: (self.config.hidden_size,),
        }

    def __call__(
        self,
        hidden_states,
        *,
        encoder_hidden_states,
        encoder_attention_mask,
        guidance,
        timestep,
        return_dict: bool = False,
        attention_kwargs=None,
        debug: bool = False,
    ):
        hidden_states = mx.array(hidden_states)
        encoder_hidden_states = mx.array(encoder_hidden_states)
        encoder_attention_mask = mx.array(encoder_attention_mask)
        if len(encoder_hidden_states.shape) != 3:
            raise ValueError("encoder_hidden_states shape must be (batch, sequence, caption_channels)")
        if encoder_hidden_states.shape[-1] != self.config.caption_channels:
            raise ValueError(
                "encoder_hidden_states last dimension must equal config.caption_channels: "
                f"expected {self.config.caption_channels}, got {encoder_hidden_states.shape[-1]}"
            )
        if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask shape must match encoder hidden batch and sequence")

        capture = DebugCapture()
        batch, _, height, width = hidden_states.shape
        tokens = patchify_nchw(hidden_states, self.config.patch_size)
        x = mx.matmul(tokens, self.input_weight.T) + self.input_bias
        capture.record("input_projection", x)

        conditioning = conditioning_vector(timestep, guidance, dim=self.config.hidden_size)
        x = x + conditioning[:, None, :]
        capture.record("conditioning", conditioning)

        encoder_hidden_states = self._project_encoder_hidden_states(encoder_hidden_states)
        capture.record("caption_projection", encoder_hidden_states)

        for index, block in enumerate(self.blocks):
            x = block(x, encoder_hidden_states, encoder_attention_mask)
            capture.record(f"block_{index}", x)

        out_tokens = mx.matmul(x, self.output_weight.T) + self.output_bias
        capture.record("output_projection", out_tokens)
        output = unpatchify_nchw(
            out_tokens,
            patch_size=self.config.patch_size,
            height=height,
            width=width,
            channels=self.config.out_channels,
        )
        capture.record("final_output", output)

        if return_dict:
            result = {"sample": output}
            if debug:
                result["debug"] = capture
            return result
        return (output, capture) if debug else (output,)

    def _build_caption_projection_weight(self):
        if self.config.caption_channels == self.config.hidden_size:
            return mx.eye(self.config.hidden_size)
        scale = 1.0 / float(self.config.caption_channels)
        return mx.ones((self.config.hidden_size, self.config.caption_channels), dtype=mx.float32) * scale

    def _project_encoder_hidden_states(self, encoder_hidden_states):
        if self.caption_projection_source == "real_weights":
            hidden_states = linear(encoder_hidden_states, self.caption_linear_1_weight, self.caption_linear_1_bias)
            hidden_states = _gelu_tanh(hidden_states)
            hidden_states = linear(hidden_states, self.caption_linear_2_weight, self.caption_linear_2_bias)
            return rms_norm(hidden_states, self.caption_norm_weight, eps=1e-5)
        if self.config.caption_channels == self.config.hidden_size:
            return encoder_hidden_states
        return mx.matmul(encoder_hidden_states, self.caption_projection_weight.T) + self.caption_projection_bias


def _gelu_tanh(x):
    x = mx.array(x)
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
