from __future__ import annotations

import mlx.core as mx


def patchify_nchw(x, patch_size: int):
    x = mx.array(x)
    batch, channels, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")

    h_patches = height // patch_size
    w_patches = width // patch_size
    x = x.reshape(batch, channels, h_patches, patch_size, w_patches, patch_size)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(batch, h_patches * w_patches, channels * patch_size * patch_size)


def unpatchify_nchw(tokens, *, patch_size: int, height: int, width: int, channels: int):
    tokens = mx.array(tokens)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")

    batch = tokens.shape[0]
    h_patches = height // patch_size
    w_patches = width // patch_size
    expected_dim = channels * patch_size * patch_size
    if tokens.shape[1] != h_patches * w_patches or tokens.shape[2] != expected_dim:
        raise ValueError("token shape does not match requested image dimensions")

    x = tokens.reshape(batch, h_patches, w_patches, channels, patch_size, patch_size)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(batch, channels, height, width)
