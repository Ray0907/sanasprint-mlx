from __future__ import annotations


UNSAFE_STATUSES = {"requires_review", "missing", "unexpected", "shape_mismatch"}


def load_mapped_weights(
    parameters: dict,
    source_tensors: dict,
    mapping_report: dict,
    *,
    allow_unexpected: bool = False,
    override_reason: str | None = None,
    return_diagnostics: bool = False,
) -> dict | tuple[dict, list[dict]]:
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
        loaded[target_key] = source_tensors[source_key]
    return (loaded, diagnostics) if return_diagnostics else loaded
