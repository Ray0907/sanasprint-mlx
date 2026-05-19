from __future__ import annotations

import mlx.core as mx

from sanasprint_mlx.primitives.patch import patchify_nchw, unpatchify_nchw
from sanasprint_mlx.transformer.block import ToyTransformerBlock
from sanasprint_mlx.transformer.conditioning import conditioning_vector
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.debug import DebugCapture


class SanaTransformerDenoiser:
    def __init__(self, config: SanaTransformerConfig):
        config.validate()
        self.config = config
        self.blocks = [ToyTransformerBlock(config.hidden_size) for _ in range(config.num_layers)]
        self.input_weight = mx.ones((config.hidden_size, config.in_channels * config.patch_size * config.patch_size))
        self.input_bias = mx.zeros((config.hidden_size,))
        self.output_weight = mx.ones((config.out_channels * config.patch_size * config.patch_size, config.hidden_size))
        self.output_bias = mx.zeros((config.out_channels * config.patch_size * config.patch_size,))

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
