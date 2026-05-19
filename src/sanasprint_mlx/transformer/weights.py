from __future__ import annotations

from pathlib import Path
from typing import Literal

from sanasprint_mlx.transformer.model import SCAFFOLD_PARAMETER_KEYS
from sanasprint_mlx.weights.config import load_transformer_config, summarize_transformer_config
from sanasprint_mlx.weights.inspect import inspect_snapshot
from sanasprint_mlx.weights.mapping import build_mapping_report
from sanasprint_mlx.weights.safetensors_reader import read_selected_tensors


UNSAFE_STATUSES = {"requires_review", "missing", "unexpected", "shape_mismatch"}
OutputBackend = Literal["native", "mlx"]


def load_mapped_weights(
    parameters: dict,
    source_tensors: dict,
    mapping_report: dict,
    *,
    allow_unexpected: bool = False,
    override_reason: str | None = None,
    return_diagnostics: bool = False,
    output_backend: OutputBackend = "native",
    mlx_dtype=None,
) -> dict | tuple[dict, list[dict]]:
    if output_backend not in ("native", "mlx"):
        raise ValueError("output_backend must be 'native' or 'mlx'")
    loaded = dict(parameters)
    diagnostics: list[dict] = []
    for entry in mapping_report.get("mapping", []):
        status = entry.get("status")
        if status in UNSAFE_STATUSES:
            if status == "unexpected" and allow_unexpected:
                if not override_reason:
                    raise ValueError("override reason is required for unexpected entries")
                diagnostics.append(
                    {
                        "source_key": entry.get("source_key"),
                        "target_key": entry.get("target_key"),
                        "status": status,
                        "override_reason": override_reason,
                    }
                )
                continue
            raise ValueError(f"unsafe mapping entry status: {status}")
        source_key = entry["source_key"]
        target_key = entry["target_key"]
        if target_key not in loaded:
            raise KeyError(target_key)
        if source_key not in source_tensors:
            raise KeyError(source_key)
        value = _prepare_tensor(
            source_tensors[source_key],
            loaded[target_key],
            entry,
            output_backend=output_backend,
            mlx_dtype=mlx_dtype,
        )
        loaded[target_key] = value
    return (loaded, diagnostics) if return_diagnostics else loaded


def _prepare_tensor(source_tensor, target_tensor, entry: dict, *, output_backend: OutputBackend, mlx_dtype):
    value = source_tensor
    if output_backend == "native":
        return value

    import mlx.core as mx

    transpose_required = entry.get("transpose_required", False)
    if transpose_required == "unknown":
        raise ValueError(f"cannot load tensor with unknown transpose requirement: {entry.get('source_key')}")

    value = mx.array(value)
    if transpose_required is True:
        if len(value.shape) != 2:
            raise ValueError(f"explicit transpose requires a 2D tensor: {entry.get('source_key')}")
        value = value.T
    elif transpose_required not in (False, None):
        raise ValueError(f"unsupported transpose requirement: {transpose_required}")

    if mlx_dtype is not None:
        value = value.astype(mlx_dtype)
    _validate_target_shape(value, target_tensor, entry)
    return value


def _validate_target_shape(value, target_tensor, entry: dict) -> None:
    target_shape = getattr(target_tensor, "shape", None)
    if target_shape is None:
        return
    if tuple(value.shape) != tuple(target_shape):
        raise ValueError(
            f"shape mismatch for {entry.get('source_key')} -> {entry.get('target_key')}: "
            f"got {tuple(value.shape)}, expected {tuple(target_shape)}"
        )


def load_mapped_weights_into_denoiser(
    model,
    source_tensors: dict,
    mapping_report: dict,
    *,
    mlx_dtype=None,
    strict: bool = False,
) -> dict:
    scaffold_keys = set(SCAFFOLD_PARAMETER_KEYS)
    selected_entries = []
    ignored_entry_count = 0
    for entry in mapping_report.get("mapping", []):
        status = entry.get("status")
        if status in UNSAFE_STATUSES:
            if _is_scaffold_relevant_entry(entry):
                raise ValueError(f"unsafe scaffold mapping entry status: {status}")
            ignored_entry_count += 1
            continue

        if entry.get("target_key") in scaffold_keys:
            selected_entries.append(entry)
        else:
            ignored_entry_count += 1

    loaded = load_mapped_weights(
        model.parameters(),
        source_tensors,
        {"mapping": selected_entries},
        output_backend="mlx",
        mlx_dtype=mlx_dtype,
    )
    selected_target_keys = {entry.get("target_key") for entry in selected_entries}
    loaded_parameters = {key: loaded[key] for key in scaffold_keys if key in selected_target_keys}
    model.load_parameters(loaded_parameters, strict=strict)
    loaded_keys = [key for key in SCAFFOLD_PARAMETER_KEYS if key in loaded_parameters]
    return {"loaded_keys": loaded_keys, "ignored_entry_count": ignored_entry_count}


def load_scaffold_weights_from_snapshot(
    model,
    snapshot_path: str | Path,
    *,
    mlx_dtype=None,
    strict: bool = True,
) -> dict:
    snapshot = Path(snapshot_path)
    config_summary = summarize_transformer_config(load_transformer_config(snapshot)).__dict__
    tensor_infos = inspect_snapshot(snapshot)
    report = build_mapping_report(tensor_infos, snapshot_path=str(snapshot), config_summary=config_summary)
    report_dict = report.to_dict()
    for entry in report_dict["mapping"]:
        if entry.get("status") in UNSAFE_STATUSES and _is_scaffold_relevant_entry(entry):
            raise ValueError(f"unsafe scaffold mapping entry status: {entry.get('status')}")
    selected_entries = [
        entry
        for entry in report_dict["mapping"]
        if entry.get("status") == "mapped" and entry.get("target_key") in set(SCAFFOLD_PARAMETER_KEYS)
    ]
    _reject_duplicate_scaffold_targets(selected_entries)
    if strict:
        selected_targets = {entry.get("target_key") for entry in selected_entries}
        missing_targets = [key for key in SCAFFOLD_PARAMETER_KEYS if key not in selected_targets]
        if missing_targets:
            raise KeyError(missing_targets[0])

    source_tensors, source_metadata = _read_scaffold_source_tensors(snapshot, tensor_infos, selected_entries)
    diagnostics = load_mapped_weights_into_denoiser(
        model,
        source_tensors,
        report_dict,
        mlx_dtype=mlx_dtype,
        strict=strict,
    )
    parameters = model.parameters()
    diagnostics.update(
        {
            "snapshot_path": str(snapshot),
            "source_tensors": _source_tensor_diagnostics(source_metadata, parameters),
        }
    )
    return diagnostics


def _read_scaffold_source_tensors(snapshot: Path, tensor_infos, selected_entries: list[dict]) -> tuple[dict, dict]:
    infos_by_key: dict[str, list] = {}
    for info in tensor_infos:
        if info.component == "transformer":
            infos_by_key.setdefault(info.name, []).append(info)

    source_tensors = {}
    source_metadata = {}
    for entry in selected_entries:
        source_key = entry["source_key"]
        matching_infos = infos_by_key.get(source_key, [])
        if len(matching_infos) != 1:
            raise ValueError(f"ambiguous or missing source tensor for {source_key}: {len(matching_infos)} matches")
        info = matching_infos[0]
        file_path = snapshot / info.file
        decoded = read_selected_tensors(file_path, [source_key])[source_key]
        source_tensors[source_key] = decoded.array
        source_metadata[entry["target_key"]] = {
            "source_key": source_key,
            "target_key": entry["target_key"],
            "source_file": info.file,
            "source_dtype": decoded.source_dtype,
            "decoded_dtype": decoded.decoded_dtype,
            "source_shape": decoded.source_shape,
        }
    return source_tensors, source_metadata


def _reject_duplicate_scaffold_targets(selected_entries: list[dict]) -> None:
    counts: dict[str, int] = {}
    for entry in selected_entries:
        target_key = entry["target_key"]
        counts[target_key] = counts.get(target_key, 0) + 1
    duplicates = [key for key in SCAFFOLD_PARAMETER_KEYS if counts.get(key, 0) > 1]
    if duplicates:
        raise ValueError(f"duplicate scaffold target mapping: {duplicates[0]}")


def _source_tensor_diagnostics(source_metadata: dict, parameters: dict) -> dict:
    diagnostics = {}
    for key in SCAFFOLD_PARAMETER_KEYS:
        if key not in source_metadata:
            continue
        value = dict(source_metadata[key])
        parameter = parameters[key]
        value["target_shape"] = [int(dim) for dim in parameter.shape]
        value["final_dtype"] = _dtype_name(parameter.dtype)
        diagnostics[key] = value
    return diagnostics


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _is_scaffold_relevant_entry(entry: dict) -> bool:
    target_key = entry.get("target_key")
    source_key = entry.get("source_key")
    if isinstance(target_key, str) and _matches_any_prefix_or_wildcard(
        target_key,
        ("mlx_transformer.patch_embed.", "mlx_transformer.proj_out."),
    ):
        return True
    if isinstance(source_key, str) and _matches_any_prefix_or_wildcard(
        source_key,
        ("transformer.patch_embed.", "patch_embed.", "transformer.proj_out.", "proj_out."),
    ):
        return True
    return False


def _matches_any_prefix_or_wildcard(value: str, prefixes: tuple[str, ...]) -> bool:
    normalized = value.removesuffix("*")
    return any(normalized.startswith(prefix) or prefix.startswith(normalized) for prefix in prefixes)
