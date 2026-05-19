import builtins

from sanasprint_mlx.memory.mlx_probe import MAX_PROBE_ALLOCATION_BYTES, probe_mlx_memory


def test_mlx_probe_returns_available_false_when_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("missing mlx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = probe_mlx_memory(allocation_bytes=1024)

    assert report["available"] is False
    assert report["unavailable_reason"]


def test_mlx_probe_limits_allocation_size():
    report = probe_mlx_memory(allocation_bytes=MAX_PROBE_ALLOCATION_BYTES * 2)

    assert report["allocation_bytes"] <= MAX_PROBE_ALLOCATION_BYTES


def test_mlx_probe_report_has_active_peak_cache_and_rss_fields():
    report = probe_mlx_memory(allocation_bytes=1024)

    assert set(report).issuperset(
        {
            "available",
            "unavailable_reason",
            "allocation_bytes",
            "active_memory_bytes",
            "peak_memory_bytes",
            "cache_memory_bytes",
            "process_rss_bytes",
        }
    )


def test_mlx_probe_records_cache_cleanup():
    report = probe_mlx_memory(allocation_bytes=1024)

    assert "cache_cleared" in report
    assert "cleanup_error" in report
