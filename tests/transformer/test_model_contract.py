import numpy as np
import pytest

from sanasprint_mlx.transformer.config import SanaTransformerConfig
from sanasprint_mlx.transformer.model import SanaTransformerDenoiser


def tiny_model():
    config = SanaTransformerConfig(
        hidden_size=4,
        in_channels=2,
        out_channels=2,
        caption_channels=4,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        patch_size=1,
        sample_size=2,
        guidance_embeds_scale=1000.0,
    )
    return SanaTransformerDenoiser(config)


def call_kwargs():
    return {
        "hidden_states": np.ones((1, 2, 2, 2), dtype=np.float32),
        "encoder_hidden_states": np.ones((1, 3, 4), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 3), dtype=np.int32),
        "guidance": np.array([4.5], dtype=np.float32),
        "timestep": np.array([0.5], dtype=np.float32),
    }


def test_model_forward_return_dict_false_returns_tuple():
    model = tiny_model()

    result = model(**call_kwargs(), return_dict=False)

    assert isinstance(result, tuple)


def test_model_forward_preserves_spatial_shape():
    model = tiny_model()

    output = model(**call_kwargs(), return_dict=False)[0]

    assert output.shape == (1, 2, 2, 2)
    assert np.isfinite(np.array(output)).all()


def test_model_forward_uses_timestep_and_guidance_conditioning():
    model = tiny_model()
    kwargs = call_kwargs()

    baseline = np.array(model(**kwargs, return_dict=False)[0])
    kwargs["timestep"] = np.array([0.9], dtype=np.float32)
    kwargs["guidance"] = np.array([7.0], dtype=np.float32)
    conditioned = np.array(model(**kwargs, return_dict=False)[0])

    assert not np.allclose(baseline, conditioned)


def test_model_rejects_bad_encoder_mask_shape():
    model = tiny_model()
    kwargs = call_kwargs()
    kwargs["encoder_attention_mask"] = np.ones((1, 4), dtype=np.int32)

    with pytest.raises(ValueError, match="encoder_attention_mask"):
        model(**kwargs)
