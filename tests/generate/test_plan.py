import pytest

from sanasprint_mlx.generate.plan import GenerationRequest, build_phase_plan


def test_low_memory_phase_order_unloads_components(tmp_path):
    request = GenerationRequest(
        prompt="a tiny astronaut",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
        low_memory=True,
        reference_decode=True,
    )

    phases = build_phase_plan(request)

    assert [phase.name for phase in phases] == ["text_encode", "denoise", "decode", "write_png"]
    assert phases[0].unloads == ["tokenizer", "text_encoder"]
    assert phases[1].unloads == ["transformer"]
    assert phases[2].unloads == ["vae"]


def test_cached_fixture_skips_text_encode(tmp_path):
    request = GenerationRequest(
        prompt=None,
        cached_fixture=tmp_path / "fixture",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
        low_memory=True,
        reference_decode=True,
    )

    phases = build_phase_plan(request)

    assert [phase.name for phase in phases] == ["denoise", "decode", "write_png"]


def test_prompt_cache_skips_text_encode(tmp_path):
    request = GenerationRequest(
        prompt=None,
        prompt_cache=tmp_path / "prompt-cache",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
        low_memory=True,
        reference_decode=True,
    )

    phases = build_phase_plan(request)

    assert [phase.name for phase in phases] == ["denoise", "decode", "write_png"]


def test_plan_rejects_non_divisible_size(tmp_path):
    request = GenerationRequest(
        prompt="bad size",
        height=500,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
    )

    with pytest.raises(ValueError, match="divisible by 32"):
        build_phase_plan(request)


def test_plan_rejects_remote_snapshot_without_allow_download(tmp_path):
    request = GenerationRequest(
        prompt="remote",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    )

    with pytest.raises(ValueError, match="allow-download"):
        build_phase_plan(request)


def test_plan_requires_prompt_or_cached_fixture(tmp_path):
    request = GenerationRequest(
        prompt=None,
        cached_fixture=None,
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
    )

    with pytest.raises(ValueError, match="prompt"):
        build_phase_plan(request)


def test_decode_phase_labels_reference_decode(tmp_path):
    request = GenerationRequest(
        prompt="decode",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
        reference_decode=True,
    )

    decode = build_phase_plan(request)[2]

    assert decode.mode == "reference_decode"


def test_decode_phase_labels_tiled_mlx_decode(tmp_path):
    request = GenerationRequest(
        prompt="decode",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
        tiled_decode=True,
    )

    decode = build_phase_plan(request)[2]

    assert decode.mode == "tiled_mlx_decode"


def test_decode_phase_labels_mlx_decode(tmp_path):
    request = GenerationRequest(
        prompt="decode",
        height=512,
        width=512,
        steps=2,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path / "snapshot",
    )

    decode = build_phase_plan(request)[2]

    assert decode.mode == "mlx_decode"
