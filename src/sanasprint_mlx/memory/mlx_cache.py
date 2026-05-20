from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def trim_mlx_cache(mx_module=None):
    mx = mx_module or _import_mlx()
    previous = None
    if hasattr(mx, "set_cache_limit"):
        previous = mx.set_cache_limit(0)
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return previous


@contextmanager
def mlx_cache_limit(limit: int = 0, mx_module=None) -> Iterator[None]:
    mx = mx_module or _import_mlx()
    previous = None
    if hasattr(mx, "set_cache_limit"):
        previous = mx.set_cache_limit(limit)
    try:
        yield
    finally:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if previous is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(previous)


def _import_mlx():
    import mlx.core as mx

    return mx
