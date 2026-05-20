import numpy as np
import torch

from sanasprint_mlx.generate.mlx_reference_decode import run_mlx_reference_decode_generation
from sanasprint_mlx.text.cache import write_prompt_cache


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.dtype = "fake"

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def to(self, **kwargs):
        return self


class FakePipeline:
    vae_scale_factor = 1
    last_from_pretrained_kwargs = None

    def __init__(self):
        self.transformer = type("Transformer", (), {"config": type("Config", (), {"in_channels": 4})()})()
        self.vae = type(
            "VAE",
            (),
            {
                "dtype": "fake",
                "config": type("VAEConfig", (), {"scaling_factor": 1.0})(),
                "decode": lambda _self, latents, return_dict=False: (latents,),
            },
        )()
        self.vae.dtype = torch.float32
        self.image_processor = type(
            "ImageProcessor",
            (),
            {"postprocess": lambda _self, image, output_type="pil": [FakeImage()]},
        )()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = kwargs
        return cls()

    def to(self, device):
        self.device = device

    def encode_prompt(self, **kwargs):
        return FakeTensor(np.ones((1, 2, 4), dtype=np.float32)), FakeTensor(np.ones((1, 2), dtype=np.int32))

    def prepare_latents(self, *args, **kwargs):
        return FakeTensor(np.ones((1, 4, 2, 2), dtype=np.float32))


class FakeImage:
    def save(self, path):
        path.write_bytes(b"png")


class FakeTransformer:
    def __init__(self):
        self.config = type("Config", (), {"guidance_embeds_scale": 1000.0})()
        self.weight_report = {"loaded_keys": {"total_count": 7}}


class FakeResult:
    latents = np.ones((1, 4, 2, 2), dtype=np.float32)


def test_mlx_reference_decode_runs_mlx_loop_and_writes_png(tmp_path, monkeypatch):
    calls = {}

    def fake_from_snapshot(*args, **kwargs):
        calls["transformer"] = kwargs
        return FakeTransformer()

    def fake_loop(**kwargs):
        calls["loop"] = kwargs
        return FakeResult()

    monkeypatch.setattr("sanasprint_mlx.generate.mlx_reference_decode.SanaSprintPipeline", FakePipeline)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_reference_decode.RealSanaTransformerDenoiser.from_snapshot", fake_from_snapshot)
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_reference_decode.run_denoising_loop", fake_loop)

    output = tmp_path / "out.png"
    report = run_mlx_reference_decode_generation(
        prompt="glass apple",
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=output,
        snapshot=tmp_path,
        allow_download=False,
    )

    assert output.read_bytes() == b"png"
    assert report["mode"] == "mlx_transformer_reference_decode"
    assert report["loaded_keys"]["total_count"] == 7
    assert report["runtime"]["wall_time_seconds"] >= 0.0
    assert report["memory"]["max_rss_bytes"] >= 0
    assert FakePipeline.last_from_pretrained_kwargs["transformer"] is None
    assert calls["transformer"]["sample_size"] == 2
    assert calls["loop"]["num_inference_steps"] == 1


def test_mlx_reference_decode_prompt_cache_skips_reference_text_components(tmp_path, monkeypatch):
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
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_reference_decode.SanaSprintPipeline", FakePipeline)
    monkeypatch.setattr(
        "sanasprint_mlx.generate.mlx_reference_decode.RealSanaTransformerDenoiser.from_snapshot",
        lambda *args, **kwargs: FakeTransformer(),
    )
    monkeypatch.setattr("sanasprint_mlx.generate.mlx_reference_decode.run_denoising_loop", lambda **kwargs: FakeResult())

    run_mlx_reference_decode_generation(
        prompt=None,
        prompt_cache=cache,
        height=2,
        width=2,
        steps=1,
        seed=7,
        output=tmp_path / "out.png",
        snapshot=tmp_path,
        allow_download=False,
    )

    assert FakePipeline.last_from_pretrained_kwargs["transformer"] is None
    assert FakePipeline.last_from_pretrained_kwargs["text_encoder"] is None
    assert FakePipeline.last_from_pretrained_kwargs["tokenizer"] is None
