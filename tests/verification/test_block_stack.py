import pytest

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.verification.block_stack import run_block_stack_smoke


def test_block_stack_smoke_runs_two_synthetic_real_blocks(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=2)

    report = run_block_stack_smoke(snapshot, block_count=2, dtype="float16", seed=123, sequence_length=4)

    assert report["status"] == "PASS"
    assert report["scope"] == "block_stack_core_with_timestep_modulation_not_full_model_parity"
    assert report["block_count"] == 2
    assert report["loaded_keys"]["count"] == 46
    assert [block["loaded_keys"]["count"] for block in report["blocks"]] == [23, 23]
    assert report["output"]["shape"] == [1, 4, 4]
    assert report["output"]["finite"] is True


def test_block_stack_smoke_rejects_block_count_larger_than_config(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot", num_layers=1)

    with pytest.raises(ValueError, match="block_count must be less than or equal to num_layers"):
        run_block_stack_smoke(snapshot, block_count=2)
