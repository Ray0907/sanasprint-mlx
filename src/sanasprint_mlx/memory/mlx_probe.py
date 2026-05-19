from __future__ import annotations

import os
import resource
from typing import Any


MAX_PROBE_ALLOCATION_BYTES = 64 * 1024**2


def probe_mlx_memory(*, allocation_bytes: int = 1024 * 1024) -> dict[str, Any]:
    allocation_bytes = min(int(allocation_bytes), MAX_PROBE_ALLOCATION_BYTES)
    report = {
        "available": False,
        "unavailable_reason": None,
        "allocation_bytes": allocation_bytes,
        "active_memory_bytes": None,
        "peak_memory_bytes": None,
        "cache_memory_bytes": None,
        "process_rss_bytes": _process_rss_bytes(),
        "cache_cleared": False,
        "cleanup_error": None,
    }

    try:
        import mlx.core as mx
    except ImportError as error:
        report["unavailable_reason"] = str(error)
        return report

    try:
        element_count = max(allocation_bytes // 4, 1)
        array = mx.ones((element_count,), dtype=mx.float32)
        mx.eval(array)
        report["available"] = True
        report["active_memory_bytes"] = _call_optional(mx, "get_active_memory")
        report["peak_memory_bytes"] = _call_optional(mx, "get_peak_memory")
        report["cache_memory_bytes"] = _call_optional(mx, "get_cache_memory")
        del array
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
            report["cache_cleared"] = True
        report["process_rss_bytes"] = _process_rss_bytes()
    except Exception as error:  # pragma: no cover - defensive for version-specific MLX failures.
        report["available"] = False
        report["unavailable_reason"] = str(error)
        report["cleanup_error"] = str(error)
    return report


def _call_optional(module, name: str):
    if not hasattr(module, name):
        return None
    return int(getattr(module, name)())


def _process_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(rss)
    return int(rss * 1024)
