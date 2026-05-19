import pytest


def test_reference_module_imports_without_reference_dependencies():
    from sanasprint_mlx.fixtures import reference

    assert reference.EXPECTED_TENSOR_NAMES == (
        "prompt_embeds",
        "prompt_attention_mask",
        "latents",
        "timesteps",
        "timestep",
        "guidance",
        "expected_noise_pred",
    )


def test_reference_dependency_check_reports_missing_packages():
    from sanasprint_mlx.fixtures.reference import check_reference_dependencies

    missing = check_reference_dependencies(("definitely_missing_sanasprint_dependency",))

    assert missing == ["definitely_missing_sanasprint_dependency"]


def test_cli_help_imports_without_reference_dependencies(capsys):
    from sanasprint_mlx.cli.fixtures import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "synthetic" in capsys.readouterr().out


def test_reference_command_requires_explicit_download_flag(tmp_path):
    from sanasprint_mlx.cli.fixtures import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "reference",
                "--output-dir",
                str(tmp_path),
                "--model-repo",
                "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                "--revision",
                "abc123",
            ]
        )

    assert error.value.code == 2
    assert not (tmp_path / "manifest.json").exists()


def test_reference_plan_exposes_expected_tensor_names():
    from sanasprint_mlx.fixtures.reference import EXPECTED_TENSOR_NAMES

    assert "expected_noise_pred" in EXPECTED_TENSOR_NAMES
    assert "timestep" in EXPECTED_TENSOR_NAMES
    assert "guidance" in EXPECTED_TENSOR_NAMES
    assert len(EXPECTED_TENSOR_NAMES) == 7


def test_reference_revision_must_be_pinned_commit():
    from sanasprint_mlx.fixtures.reference import validate_pinned_revision

    validate_pinned_revision("a" * 40)

    with pytest.raises(ValueError, match="40-character commit SHA"):
        validate_pinned_revision("main")


def test_diffusers_commit_must_be_pinned():
    from sanasprint_mlx.fixtures.reference import resolve_diffusers_commit

    with pytest.raises(ValueError, match="diffusers commit"):
        resolve_diffusers_commit(None, "unknown")

    assert resolve_diffusers_commit("b" * 40, "unknown") == "b" * 40


def test_tokenizer_assets_are_hashed_with_configs(tmp_path):
    from sanasprint_mlx.fixtures.reference import hash_reproduction_files

    (tmp_path / "model_index.json").write_text("{}")
    (tmp_path / "tokenizer.model").write_text("tokenizer")
    (tmp_path / "spiece.model").write_text("spiece")
    (tmp_path / "merges.txt").write_text("merges")
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "model.safetensors").write_bytes(b"weights")

    config_files, config_hashes, weight_files, weight_hashes = hash_reproduction_files(tmp_path)

    assert "model_index.json" in config_files
    assert "tokenizer.model" in config_files
    assert "spiece.model" in config_files
    assert "merges.txt" in config_files
    assert "transformer/model.safetensors" in weight_files
    assert set(config_files) == set(config_hashes)
    assert set(weight_files) == set(weight_hashes)
