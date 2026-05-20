import numpy as np
import torch

from sanasprint_mlx.generate.mlx_reference_decode import run_mlx_reference_decode_generation


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
    assert FakePipeline.last_from_pretrained_kwargs["transformer"] is None
    assert calls["transformer"]["sample_size"] == 2
    assert calls["loop"]["num_inference_steps"] == 1
