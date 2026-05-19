import numpy as np

from sanasprint_mlx.fixtures.manifest import FixtureManifest
from sanasprint_mlx.fixtures.synthetic import generate_synthetic_fixture


def load_fixture(output_dir):
    manifest = FixtureManifest.from_json(output_dir / "manifest.json")
    arrays = np.load(output_dir / "fixture.npz")
    return manifest, arrays


def test_synthetic_fixture_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_synthetic_fixture(first, seed=7)
    generate_synthetic_fixture(second, seed=7)

    first_manifest, first_arrays = load_fixture(first)
    second_manifest, second_arrays = load_fixture(second)

    assert first_manifest.tensor_hashes == second_manifest.tensor_hashes
    for name in first_manifest.tensor_metadata:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])


def test_synthetic_fixture_writes_manifest_and_npz(tmp_path):
    output_dir = tmp_path / "fixture"

    manifest_path = generate_synthetic_fixture(output_dir, seed=11)

    assert manifest_path == output_dir / "manifest.json"
    assert manifest_path.exists()
    assert (output_dir / "fixture.npz").exists()

    arrays = np.load(output_dir / "fixture.npz")
    assert set(arrays.files) == {
        "prompt_embeds",
        "prompt_attention_mask",
        "latents",
        "timesteps",
        "timestep",
        "guidance",
        "expected_noise_pred",
    }


def test_synthetic_fixture_uses_tier_0_metadata(tmp_path):
    output_dir = tmp_path / "fixture"

    generate_synthetic_fixture(output_dir, seed=13, height=8, width=8, num_inference_steps=2)
    manifest, arrays = load_fixture(output_dir)

    assert manifest.fixture_tier == 0
    assert manifest.model_revision == "synthetic"
    assert manifest.diffusers_version == "not-used"
    assert manifest.transformers_version == "not-used"
    assert manifest.seed == 13
    assert manifest.height == 8
    assert manifest.width == 8
    assert manifest.num_inference_steps == 2
    assert sorted(manifest.tensor_metadata) == sorted(arrays.files)
    assert manifest.model_weight_files == []
    assert manifest.model_weight_hashes == {}
