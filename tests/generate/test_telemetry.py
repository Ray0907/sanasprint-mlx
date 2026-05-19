from sanasprint_mlx.generate.telemetry import PhaseTelemetry


def test_phase_telemetry_records_start_end_and_memory_fields():
    telemetry = PhaseTelemetry("denoise", memory_snapshot=lambda: {"rss_bytes": 10})

    telemetry.start()
    telemetry.end()
    data = telemetry.to_dict()

    assert data["name"] == "denoise"
    assert data["started_at"] is not None
    assert data["ended_at"] is not None
    assert data["memory_start"] == {"rss_bytes": 10}
    assert data["memory_end"] == {"rss_bytes": 10}


def test_phase_telemetry_records_unload_events():
    telemetry = PhaseTelemetry("decode")

    telemetry.record_unload("vae")

    assert telemetry.to_dict()["unload_events"] == ["vae"]
