from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.autoencoder.decoder_ops import (
    conv2d_nchw,
    dc_up_block_interpolate,
    glumb_conv,
    res_block,
    rms_norm_nchw,
    relu,
    sana_multiscale_linear_attention,
)
from sanasprint_mlx.autoencoder.real_decoder import load_decoder_weights_from_snapshot


class MLXAutoencoderDCDecoder:
    def __init__(self, *, config: dict, tensors: dict):
        self.config = config
        self.tensors = tensors

    @classmethod
    def from_snapshot(cls, snapshot: str | Path, *, dtype: str = "bfloat16") -> "MLXAutoencoderDCDecoder":
        snapshot_path = Path(snapshot)
        config = json.loads((snapshot_path / "vae" / "config.json").read_text())
        report = load_decoder_weights_from_snapshot(snapshot_path, mlx_dtype=_mlx_dtype(dtype))
        return cls(config=config, tensors=report["tensors"])

    def decode(self, latents):
        hidden = mx.array(latents)
        channels = self.config["decoder_block_out_channels"][-1]
        hidden = conv2d_nchw(hidden, self._tensor("decoder.conv_in.weight"), self._tensor("decoder.conv_in.bias"), padding=1)
        hidden = hidden + mx.repeat(mx.array(latents), channels // self.config["latent_channels"], axis=1)

        block_out_channels = list(self.config["decoder_block_out_channels"])
        layers_per_block = list(self.config["decoder_layers_per_block"])
        block_types = list(self.config["decoder_block_types"])
        qkv_multiscales = list(self.config["decoder_qkv_multiscales"])
        attention_head_dim = int(self.config["attention_head_dim"])

        for block_index in reversed(range(len(block_out_channels))):
            layer_start = 0
            if block_index < len(block_out_channels) - 1 and layers_per_block[block_index] > 0:
                hidden = dc_up_block_interpolate(
                    hidden,
                    conv_weight=self._tensor(f"decoder.up_blocks.{block_index}.0.conv.weight"),
                    conv_bias=self._tensor(f"decoder.up_blocks.{block_index}.0.conv.bias"),
                )
                layer_start = 1

            for layer_offset in range(layers_per_block[block_index]):
                layer_index = layer_start + layer_offset
                prefix = f"decoder.up_blocks.{block_index}.{layer_index}"
                if block_types[block_index] == "ResBlock":
                    hidden = res_block(
                        hidden,
                        conv1_weight=self._tensor(f"{prefix}.conv1.weight"),
                        conv1_bias=self._tensor(f"{prefix}.conv1.bias"),
                        conv2_weight=self._tensor(f"{prefix}.conv2.weight"),
                        norm_weight=self._tensor(f"{prefix}.norm.weight"),
                        norm_bias=self._tensor(f"{prefix}.norm.bias"),
                    )
                elif block_types[block_index] == "EfficientViTBlock":
                    hidden = self._efficient_vit_block(
                        hidden,
                        prefix=prefix,
                        qkv_multiscales=qkv_multiscales[block_index],
                        attention_head_dim=attention_head_dim,
                    )
                else:
                    raise ValueError(f"unsupported decoder block type: {block_types[block_index]}")

        hidden = rms_norm_nchw(
            hidden,
            self._tensor("decoder.norm_out.weight"),
            self._tensor("decoder.norm_out.bias"),
            eps=1e-5,
        )
        hidden = relu(hidden)
        return conv2d_nchw(
            hidden,
            self._tensor("decoder.conv_out.weight"),
            self._tensor("decoder.conv_out.bias"),
            padding=1,
        )

    def _efficient_vit_block(self, hidden, *, prefix: str, qkv_multiscales: list[int], attention_head_dim: int):
        attn_prefix = f"{prefix}.attn"
        multiscale_weights = [
            {
                "proj_in_weight": self._tensor(f"{attn_prefix}.to_qkv_multiscale.{index}.proj_in.weight"),
                "proj_out_weight": self._tensor(f"{attn_prefix}.to_qkv_multiscale.{index}.proj_out.weight"),
            }
            for index, _ in enumerate(qkv_multiscales)
        ]
        hidden = sana_multiscale_linear_attention(
            hidden,
            to_q_weight=self._tensor(f"{attn_prefix}.to_q.weight"),
            to_k_weight=self._tensor(f"{attn_prefix}.to_k.weight"),
            to_v_weight=self._tensor(f"{attn_prefix}.to_v.weight"),
            multiscale_weights=multiscale_weights,
            to_out_weight=self._tensor(f"{attn_prefix}.to_out.weight"),
            norm_weight=self._tensor(f"{attn_prefix}.norm_out.weight"),
            norm_bias=self._tensor(f"{attn_prefix}.norm_out.bias"),
            attention_head_dim=attention_head_dim,
            norm_type="rms_norm",
            residual_connection=True,
        )
        conv_prefix = f"{prefix}.conv_out"
        return glumb_conv(
            hidden,
            conv_inverted_weight=self._tensor(f"{conv_prefix}.conv_inverted.weight"),
            conv_inverted_bias=self._tensor(f"{conv_prefix}.conv_inverted.bias"),
            conv_depth_weight=self._tensor(f"{conv_prefix}.conv_depth.weight"),
            conv_depth_bias=self._tensor(f"{conv_prefix}.conv_depth.bias"),
            conv_point_weight=self._tensor(f"{conv_prefix}.conv_point.weight"),
            norm_weight=self._tensor(f"{conv_prefix}.norm.weight"),
            norm_bias=self._tensor(f"{conv_prefix}.norm.bias"),
        )

    def _tensor(self, key: str):
        return self.tensors[key]


def _mlx_dtype(dtype: str):
    values = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if dtype not in values:
        raise ValueError(f"dtype must be one of {', '.join(values)}")
    return values[dtype]
