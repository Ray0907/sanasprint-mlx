from __future__ import annotations

from typing import Literal

from sanasprint_mlx.transformer.model import SCAFFOLD_PARAMETER_KEYS


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
    loaded_parameters = {key: loaded[key] for key in scaffold_keys if key in {entry.get("target_key") for entry in selected_entries}}
    model.load_parameters(loaded_parameters, strict=strict)
    loaded_keys = [key for key in SCAFFOLD_PARAMETER_KEYS if key in loaded_parameters]
    return {"loaded_keys": loaded_keys, "ignored_entry_count": ignored_entry_count}


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
