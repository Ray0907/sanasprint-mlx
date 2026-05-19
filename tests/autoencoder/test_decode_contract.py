import numpy as np

from sanasprint_mlx.autoencoder.config import AutoencoderDecodeConfig
from sanasprint_mlx.autoencoder.decode import AutoencoderDCDecode


class RecordingDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, z):
        self.calls.append(np.array(z))
        return np.array(z) + 1.0


def test_decode_return_dict_false_returns_tuple():
    model = AutoencoderDCDecode(RecordingDecoder(), AutoencoderDecodeConfig())

    result = model.decode(np.zeros((1, 1, 2, 2), dtype=np.float32), return_dict=False)

    assert isinstance(result, tuple)


def test_decode_return_dict_true_returns_sample_dict():
    model = AutoencoderDCDecode(RecordingDecoder(), AutoencoderDecodeConfig())

    result = model.decode(np.zeros((1, 1, 2, 2), dtype=np.float32), return_dict=True)

    assert "sample" in result


def test_decode_slicing_decodes_each_batch_item():
    decoder = RecordingDecoder()
    model = AutoencoderDCDecode(decoder, AutoencoderDecodeConfig(use_slicing=True))

    result = model.decode(np.zeros((2, 1, 2, 2), dtype=np.float32), return_dict=False)[0]

    assert len(decoder.calls) == 2
    assert result.shape == (2, 1, 2, 2)


def test_decode_uses_tiled_path_when_threshold_exceeded():
    decoder = RecordingDecoder()
    config = AutoencoderDecodeConfig(
        use_tiling=True,
        spatial_compression_ratio=2,
        tile_sample_min_height=4,
        tile_sample_min_width=4,
        tile_sample_stride_height=4,
        tile_sample_stride_width=4,
    )
    model = AutoencoderDCDecode(decoder, config)

    model.decode(np.zeros((1, 1, 3, 2), dtype=np.float32), return_dict=False)

    assert len(decoder.calls) > 1
