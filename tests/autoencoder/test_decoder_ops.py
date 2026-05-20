import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import SanaMultiscaleLinearAttention

from sanasprint_mlx.autoencoder.decoder_ops import (
    conv2d_nchw,
    dc_up_block_interpolate,
    glumb_conv,
    res_block,
    rms_norm_nchw,
    sana_multiscale_linear_attention,
)


def test_conv2d_nchw_matches_torch_conv2d():
    x = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3) / 10.0
    weight = np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3) / 20.0
    bias = np.linspace(-0.2, 0.2, 4, dtype=np.float32)

    actual = np.array(conv2d_nchw(x, weight, bias, padding=1))
    expected = F.conv2d(torch.from_numpy(x), torch.from_numpy(weight), torch.from_numpy(bias), padding=1).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_rms_norm_nchw_matches_torch_channel_last_rms_norm():
    x = np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2) / 10.0
    weight = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    bias = np.array([-0.1, 0.0, 0.1], dtype=np.float32)

    actual = np.array(rms_norm_nchw(x, weight, bias, eps=1e-5))
    moved = torch.from_numpy(x).movedim(1, -1)
    expected = torch.nn.functional.rms_norm(
        moved,
        normalized_shape=(3,),
        weight=torch.from_numpy(weight),
        eps=1e-5,
    )
    expected = expected + torch.from_numpy(bias)
    expected = expected.movedim(-1, 1).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_res_block_matches_diffusers_resblock_math():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
    conv1_weight = rng.standard_normal((2, 2, 3, 3), dtype=np.float32) * 0.05
    conv1_bias = rng.standard_normal((2,), dtype=np.float32) * 0.05
    conv2_weight = rng.standard_normal((2, 2, 3, 3), dtype=np.float32) * 0.05
    norm_weight = rng.standard_normal((2,), dtype=np.float32) * 0.05 + 1.0
    norm_bias = rng.standard_normal((2,), dtype=np.float32) * 0.05

    actual = np.array(
        res_block(
            x,
            conv1_weight=conv1_weight,
            conv1_bias=conv1_bias,
            conv2_weight=conv2_weight,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
        )
    )
    hidden = F.conv2d(torch.from_numpy(x), torch.from_numpy(conv1_weight), torch.from_numpy(conv1_bias), padding=1)
    hidden = F.silu(hidden)
    hidden = F.conv2d(hidden, torch.from_numpy(conv2_weight), bias=None, padding=1)
    hidden = F.rms_norm(hidden.movedim(1, -1), (2,), weight=torch.from_numpy(norm_weight), eps=1e-5)
    hidden = (hidden + torch.from_numpy(norm_bias)).movedim(-1, 1)
    expected = (hidden + torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_dc_up_block_interpolate_matches_diffusers_shortcut_path():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 2, 2, 2), dtype=np.float32)
    conv_weight = rng.standard_normal((4, 2, 3, 3), dtype=np.float32) * 0.05
    conv_bias = rng.standard_normal((4,), dtype=np.float32) * 0.05

    actual = np.array(dc_up_block_interpolate(x, conv_weight=conv_weight, conv_bias=conv_bias))
    hidden = F.interpolate(torch.from_numpy(x), scale_factor=2, mode="nearest")
    hidden = F.conv2d(hidden, torch.from_numpy(conv_weight), torch.from_numpy(conv_bias), padding=1)
    shortcut = torch.repeat_interleave(torch.from_numpy(x), repeats=4 * 4 // 2, dim=1)
    shortcut = F.pixel_shuffle(shortcut, 2)
    expected = (hidden + shortcut).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_glumb_conv_matches_diffusers_math():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
    conv_inverted_weight = rng.standard_normal((16, 2, 1, 1), dtype=np.float32) * 0.05
    conv_inverted_bias = rng.standard_normal((16,), dtype=np.float32) * 0.05
    conv_depth_weight = rng.standard_normal((16, 1, 3, 3), dtype=np.float32) * 0.05
    conv_depth_bias = rng.standard_normal((16,), dtype=np.float32) * 0.05
    conv_point_weight = rng.standard_normal((2, 8, 1, 1), dtype=np.float32) * 0.05
    norm_weight = rng.standard_normal((2,), dtype=np.float32) * 0.05 + 1.0
    norm_bias = rng.standard_normal((2,), dtype=np.float32) * 0.05

    actual = np.array(
        glumb_conv(
            x,
            conv_inverted_weight=conv_inverted_weight,
            conv_inverted_bias=conv_inverted_bias,
            conv_depth_weight=conv_depth_weight,
            conv_depth_bias=conv_depth_bias,
            conv_point_weight=conv_point_weight,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
        )
    )
    hidden = F.conv2d(torch.from_numpy(x), torch.from_numpy(conv_inverted_weight), torch.from_numpy(conv_inverted_bias))
    hidden = F.silu(hidden)
    hidden = F.conv2d(
        hidden,
        torch.from_numpy(conv_depth_weight),
        torch.from_numpy(conv_depth_bias),
        padding=1,
        groups=16,
    )
    hidden, gate = torch.chunk(hidden, 2, dim=1)
    hidden = hidden * F.silu(gate)
    hidden = F.conv2d(hidden, torch.from_numpy(conv_point_weight), bias=None)
    hidden = F.rms_norm(hidden.movedim(1, -1), (2,), weight=torch.from_numpy(norm_weight), eps=1e-5)
    hidden = (hidden + torch.from_numpy(norm_bias)).movedim(-1, 1)
    expected = (hidden + torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_sana_multiscale_linear_attention_matches_diffusers_module():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((1, 4, 2, 2), dtype=np.float32)
    attn = SanaMultiscaleLinearAttention(
        in_channels=4,
        out_channels=4,
        attention_head_dim=2,
        kernel_sizes=(3,),
        norm_type="rms_norm",
        residual_connection=True,
    )
    state = attn.state_dict()
    for index, key in enumerate(sorted(state)):
        values = rng.standard_normal(tuple(state[key].shape), dtype=np.float32) * 0.05
        if key == "norm_out.weight":
            values = values + 1.0
        state[key].copy_(torch.from_numpy(values))

    actual = np.array(
        sana_multiscale_linear_attention(
            x,
            to_q_weight=state["to_q.weight"].numpy(),
            to_k_weight=state["to_k.weight"].numpy(),
            to_v_weight=state["to_v.weight"].numpy(),
            multiscale_weights=[
                {
                    "proj_in_weight": state["to_qkv_multiscale.0.proj_in.weight"].numpy(),
                    "proj_out_weight": state["to_qkv_multiscale.0.proj_out.weight"].numpy(),
                }
            ],
            to_out_weight=state["to_out.weight"].numpy(),
            norm_weight=state["norm_out.weight"].numpy(),
            norm_bias=state["norm_out.bias"].numpy(),
            attention_head_dim=2,
            norm_type="rms_norm",
            residual_connection=True,
        )
    )
    with torch.no_grad():
        expected = attn(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)
