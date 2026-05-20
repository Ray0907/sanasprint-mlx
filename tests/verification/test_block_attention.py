import numpy as np
import pytest

from sanasprint_mlx.cli.weights import make_synthetic_snapshot
from sanasprint_mlx.verification.block_attention import run_block0_attention_smoke


def test_block0_attention_smoke_loads_synthetic_snapshot_weights(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    report = run_block0_attention_smoke(snapshot, dtype="float16", seed=123, sequence_length=4)

    assert report["status"] == "PASS"
    assert report["block_index"] == 0
    assert report["scope"] == "block0_core_with_timestep_modulation_not_full_model_parity"
    assert report["prompt_source"] == "synthetic_projected_hidden_states"
    assert report["loaded_keys"]["count"] == 23
    assert report["ffn"]["active"] is True
    assert report["ffn"]["grid_shape"] == [2, 2]
    assert report["timestep"]["embedding_shape"] == [1, 24]
    assert report["output"]["shape"] == [1, 4, 4]
    assert report["output"]["finite"] is True
    assert report["weights"]["source_tensors"]["mlx_transformer.transformer_blocks.0.attn1.to_q.weight"]["final_dtype"] == "float16"


def test_block0_attention_smoke_is_deterministic(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    first = run_block0_attention_smoke(snapshot, seed=99)
    second = run_block0_attention_smoke(snapshot, seed=99)

    assert first["output"] == second["output"]


def test_block0_attention_smoke_rejects_non_positive_sequence_length(tmp_path):
    snapshot = make_synthetic_snapshot(tmp_path / "snapshot")

    with pytest.raises(ValueError, match="sequence_length"):
        run_block0_attention_smoke(snapshot, sequence_length=0)
