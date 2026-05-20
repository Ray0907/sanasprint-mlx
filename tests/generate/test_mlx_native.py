from pathlib import Path

import numpy as np
from contextlib import contextmanager

from sanasprint_mlx.generate.mlx_native import run_mlx_batch_generation, run_mlx_generation
from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.text.encoder import EncodedPrompt
from sanasprint_mlx.text.cache import write_prompt_cache


class FakeTransformer:
    def __init__(self):
        self.config = type("Config", (), {"guidance_embeds_scale": 1000.0})()
        self.weight_report = {"loaded_keys": {"total_count": 11}}


class FakeDecoder:
    def decode(self, latents):
        batch = latents.shape[0]
        return np.zeros((batch, 3, 2, 2), dtype=np.float32)


class FakeLoopResult:
    latents = np.zeros((1, 4, 2, 2), dtype=np.float32)


def test_mlx_generation_uses_prompt_cache_transformer_and_mlx_decoder(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )
    calls = {}

    def fake_transformer_from_snapshot(*args, **kwargs):
        calls["transformer"] = kwargs
        return FakeTransformer()

    def fake_decoder_from_snapshot(*args, **kwargs):
        calls["decoder"] = kwargs
        return FakeDecoder()

    def fake_loop(**kwargs):
        calls["loop"] = kwargs
        return FakeLoopResult()

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", fake_transformer_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", fake_decoder_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", fake_loop)

    output = tmp_path / "out.png"
    report = run_mlx_generation(
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=output,
        snapshot=snapshot,
    )

    assert output.exists()
    assert report["mode"] == "mlx_transformer_mlx_decode"
    assert report["loaded_keys"]["total_count"] == 11
    assert calls["transformer"]["sample_size"] == 1
    assert calls["loop"]["num_inference_steps"] == 1
    assert calls["decoder"]["dtype"] == "bfloat16"


def test_mlx_generation_uses_native_prompt_encoder_without_cache(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    calls = {}

    def fake_encode_prompt_mlx(**kwargs):
        calls["prompt"] = kwargs
        return EncodedPrompt(
            prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
            prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        )

    def fake_transformer_from_snapshot(*args, **kwargs):
        return FakeTransformer()

    def fake_decoder_from_snapshot(*args, **kwargs):
        return FakeDecoder()

    def fake_loop(**kwargs):
        calls["loop"] = kwargs
        return FakeLoopResult()

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.encode_prompt_mlx", fake_encode_prompt_mlx)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", fake_transformer_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", fake_decoder_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", fake_loop)

    output = tmp_path / "out.png"
    report = run_mlx_generation(
        prompt="raw prompt",
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=output,
        snapshot=snapshot,
    )

    assert output.exists()
    assert report["prompt_source"] == "mlx_text_encoder"
    assert calls["prompt"]["prompt"] == "raw prompt"
    assert calls["prompt"]["snapshot"] == snapshot
    np.testing.assert_array_equal(calls["loop"]["prompt_attention_mask"], np.ones((1, 2), dtype=np.int32))


def test_mlx_generation_releases_transformer_before_loading_decoder(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )
    events = []

    def fake_transformer_from_snapshot(*args, **kwargs):
        events.append("load_transformer")
        return FakeTransformer()

    def fake_loop(**kwargs):
        events.append("denoise")
        return FakeLoopResult()

    def fake_release():
        events.append("release")

    def fake_decoder_from_snapshot(*args, **kwargs):
        events.append("load_decoder")
        return FakeDecoder()

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", fake_transformer_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", fake_loop)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native._release_mlx_memory", fake_release)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", fake_decoder_from_snapshot)

    run_mlx_generation(
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=snapshot,
    )

    assert events == ["load_transformer", "denoise", "release", "load_decoder"]


def test_mlx_generation_uses_tiled_decode_wrapper_when_requested(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )
    calls = {}

    class FakeDecodeWrapper:
        def __init__(self, decoder, config):
            calls["wrapper"] = {"decoder": decoder, "config": config}

        def decode(self, latents, *, return_dict):
            calls["decode"] = {"latents": np.array(latents), "return_dict": return_dict}
            return (np.zeros((1, 3, 2, 2), dtype=np.float32),)

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", lambda *args, **kwargs: FakeTransformer())
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", lambda *args, **kwargs: FakeDecoder())
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", lambda **kwargs: FakeLoopResult())
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.AutoencoderDCDecode", FakeDecodeWrapper)

    report = run_mlx_generation(
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=snapshot,
        tiled_decode=True,
    )

    assert report["decode_mode"] == "tiled_mlx_decode"
    assert calls["wrapper"]["config"].use_tiling is True
    assert calls["decode"]["return_dict"] is False


def test_mlx_batch_generation_reuses_loaded_components_and_increments_seeds(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )
    calls = {"transformer_loads": 0, "decoder_loads": 0, "seeds": []}

    def fake_transformer_from_snapshot(*args, **kwargs):
        calls["transformer_loads"] += 1
        return FakeTransformer()

    def fake_decoder_from_snapshot(*args, **kwargs):
        calls["decoder_loads"] += 1
        return FakeDecoder()

    def fake_latents(*, channels, height, width, seed):
        calls["seeds"].append(seed)
        return np.zeros((1, channels, height, width), dtype=np.float32)

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", fake_transformer_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", fake_decoder_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native._latents", fake_latents)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", lambda **kwargs: FakeLoopResult())

    reports = run_mlx_batch_generation(
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=10,
        outputs=[tmp_path / "one.png", tmp_path / "two.png"],
        snapshot=snapshot,
    )

    assert calls["transformer_loads"] == 1
    assert calls["decoder_loads"] == 1
    assert calls["seeds"] == [10, 11]
    assert [Path(report["output"]).name for report in reports] == ["one.png", "two.png"]
    assert [report["seed"] for report in reports] == [10, 11]


def test_mlx_batch_generation_runs_inside_zero_cache_limit(tmp_path, monkeypatch):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "prompt-cache"
    write_prompt_cache(
        cache,
        prompt="cached",
        prompt_embeds=np.ones((1, 2, 4), dtype=np.float32),
        prompt_attention_mask=np.ones((1, 2), dtype=np.int32),
        tokenizer_id="fake",
        model_id="fake",
        max_sequence_length=2,
        clean_caption=False,
        complex_human_instruction=[],
    )
    events = []

    @contextmanager
    def fake_cache_limit(limit):
        events.append(("enter_cache_limit", limit))
        yield
        events.append("exit_cache_limit")

    def fake_transformer_from_snapshot(*args, **kwargs):
        events.append("load_transformer")
        return FakeTransformer()

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.mlx_cache_limit", fake_cache_limit)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.RealSanaTransformerDenoiser.from_snapshot", fake_transformer_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.MLXAutoencoderDCDecoder.from_snapshot", lambda *args, **kwargs: FakeDecoder())
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_native.run_denoising_loop", lambda **kwargs: FakeLoopResult())

    run_mlx_batch_generation(
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=10,
        outputs=[tmp_path / "one.png"],
        snapshot=snapshot,
    )

    assert events[0] == ("enter_cache_limit", 0)
    assert "load_transformer" in events
    assert events[-1] == "exit_cache_limit"
