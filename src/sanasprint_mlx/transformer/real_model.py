from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from sanasprint_mlx.primitives.patch import patchify_nchw, unpatchify_nchw
from sanasprint_mlx.transformer.block import RealSanaAttentionBlock
from sanasprint_mlx.transformer.block_weights import load_block_attention_weights_from_snapshot
from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser
from sanasprint_mlx.transformer.output import SanaOutputNorm, load_output_norm_weights_from_snapshot
from sanasprint_mlx.transformer.timestep import SanaTimestepGuidanceEmbedding, load_timestep_guidance_weights_from_snapshot
from sanasprint_mlx.transformer.weights import load_scaffold_weights_from_snapshot
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config


class RealSanaTransformerDenoiser:
    def __init__(
        self,
        *,
        config: SanaTransformerConfig,
        blocks: list[RealSanaAttentionBlock],
        scaffold: SanaTransformerDenoiser,
        time_embedding: SanaTimestepGuidanceEmbedding,
        output_norm: SanaOutputNorm,
        dtype,
        weight_report: dict,
    ):
        self.config = config
        self.blocks = blocks
        self.scaffold = scaffold
        self.time_embedding = time_embedding
        self.output_norm = output_norm
        self.dtype = dtype
        self.weight_report = weight_report

    @classmethod
    def from_snapshot(
        cls,
        snapshot: str | Path,
        *,
        sample_size: int | None = None,
        block_count: int | None = None,
        dtype: str = "bfloat16",
        strict: bool = True,
    ) -> "RealSanaTransformerDenoiser":
        snapshot_path = Path(snapshot)
        config_dict = load_transformer_config(snapshot_path)
        summary = summarize_transformer_config(config_dict)
        active_block_count = summary.num_layers if block_count is None else block_count
        if active_block_count <= 0:
            raise ValueError("block_count must be positive")
        if active_block_count > summary.num_layers:
            raise ValueError("block_count must be less than or equal to num_layers")
        active_sample_size = summary.sample_size if sample_size is None else sample_size
        if active_sample_size <= 0:
            raise ValueError("sample_size must be positive")

        config = SanaTransformerConfig(
            hidden_size=summary.hidden_size,
            in_channels=summary.in_channels,
            out_channels=summary.out_channels,
            caption_channels=summary.caption_channels,
            num_layers=active_block_count,
            num_attention_heads=summary.num_attention_heads,
            attention_head_dim=summary.attention_head_dim,
            patch_size=summary.patch_size,
            sample_size=active_sample_size,
            guidance_embeds_scale=summary.guidance_embeds_scale,
        )
        scaffold = SanaTransformerDenoiser(config)
        mlx_dtype = _mlx_dtype(dtype)
        scaffold_report = load_scaffold_weights_from_snapshot(
            scaffold,
            snapshot_path,
            mlx_dtype=mlx_dtype,
            strict=strict,
            include_caption_projection=True,
        )

        time_embedding = SanaTimestepGuidanceEmbedding(summary.hidden_size)
        time_report = load_timestep_guidance_weights_from_snapshot(
            time_embedding,
            snapshot_path,
            mlx_dtype=mlx_dtype,
            strict=strict,
        )
        output_norm = SanaOutputNorm(summary.hidden_size)
        output_norm_report = load_output_norm_weights_from_snapshot(
            output_norm,
            snapshot_path,
            mlx_dtype=mlx_dtype,
            strict=strict,
        )

        blocks = []
        block_reports = []
        for block_index in range(active_block_count):
            block = RealSanaAttentionBlock(
                hidden_size=summary.hidden_size,
                num_attention_heads=summary.num_attention_heads,
                attention_head_dim=summary.attention_head_dim,
                num_cross_attention_heads=int(config_dict.get("num_cross_attention_heads", summary.num_attention_heads)),
                cross_attention_head_dim=int(config_dict.get("cross_attention_head_dim", summary.attention_head_dim)),
                block_index=block_index,
                include_ffn=True,
                mlp_ratio=float(config_dict.get("mlp_ratio", 2.5)),
            )
            block_report = load_block_attention_weights_from_snapshot(
                block,
                snapshot_path,
                block_index=block_index,
                mlx_dtype=mlx_dtype,
                strict=strict,
            )
            blocks.append(block)
            block_reports.append(
                {
                    "block_index": block_index,
                    "loaded_keys": {
                        "count": len(block_report["loaded_keys"]),
                        "keys": block_report["loaded_keys"],
                    },
                    "weights": block_report,
                }
            )

        scaffold_count = len(scaffold_report["loaded_keys"])
        caption_count = len(scaffold_report["loaded_caption_keys"])
        time_embedding_count = len(time_report["loaded_keys"])
        output_norm_count = len(output_norm_report["loaded_keys"])
        block_loaded_count = sum(block["loaded_keys"]["count"] for block in block_reports)
        weight_report = {
            "snapshot_path": str(snapshot_path),
            "block_count": active_block_count,
            "loaded_keys": {
                "scaffold_count": scaffold_count,
                "caption_count": caption_count,
                "time_embedding_count": time_embedding_count,
                "output_norm_count": output_norm_count,
                "block_count": block_loaded_count,
                "total_count": scaffold_count
                + caption_count
                + time_embedding_count
                + output_norm_count
                + block_loaded_count,
            },
            "caption_projection_source": scaffold_report["caption_projection_source"],
            "time_embedding_source": time_report["source"],
            "output_norm_source": output_norm_report["source"],
            "scaffold_weights": scaffold_report,
            "time_embedding_weights": time_report,
            "output_norm_weights": output_norm_report,
            "blocks": block_reports,
        }
        return cls(
            config=config,
            blocks=blocks,
            scaffold=scaffold,
            time_embedding=time_embedding,
            output_norm=output_norm,
            dtype=mlx_dtype,
            weight_report=weight_report,
        )

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
        del attention_kwargs
        if debug:
            raise ValueError("debug capture is not supported by RealSanaTransformerDenoiser yet")
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

        _, _, height, width = hidden_states.shape
        tokens = patchify_nchw(hidden_states, self.config.patch_size)
        x = mx.matmul(tokens, self.scaffold.input_weight.T) + self.scaffold.input_bias
        encoder_hidden_states = self.scaffold._project_encoder_hidden_states(encoder_hidden_states)
        timestep_embedding, conditioning = self.time_embedding(
            timestep=timestep,
            guidance=guidance,
            hidden_dtype=self.dtype,
        )
        token_height = height // self.config.patch_size
        token_width = width // self.config.patch_size
        for block in self.blocks:
            x = block(
                x,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep_embedding=timestep_embedding,
                height=token_height,
                width=token_width,
            )
        x = self.output_norm(x, conditioning)
        out_tokens = mx.matmul(x, self.scaffold.output_weight.T) + self.scaffold.output_bias
        output = unpatchify_nchw(
            out_tokens,
            patch_size=self.config.patch_size,
            height=height,
            width=width,
            channels=self.config.out_channels,
        )
        if return_dict:
            return {"sample": output}
        return (output,)


def _mlx_dtype(dtype: str):
    values = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if dtype not in values:
        raise ValueError(f"dtype must be one of {', '.join(values)}")
    return values[dtype]
