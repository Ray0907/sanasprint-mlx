from __future__ import annotations

from typing import Literal


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
