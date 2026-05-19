from __future__ import annotations

import time
from dataclasses import dataclass, field


def empty_memory_snapshot() -> dict:
    return {
        "rss_bytes": None,
        "mlx_active_bytes": None,
        "mlx_peak_bytes": None,
        "mlx_cache_bytes": None,
    }


@dataclass
class PhaseTelemetry:
    name: str
    memory_snapshot: object | None = None
    started_at: float | None = None
    ended_at: float | None = None
    memory_start: dict | None = None
    memory_end: dict | None = None
    unload_events: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.started_at = time.time()
        self.memory_start = self._snapshot()

    def end(self) -> None:
        self.ended_at = time.time()
        self.memory_end = self._snapshot()

    def record_unload(self, component: str) -> None:
        self.unload_events.append(component)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "memory_start": self.memory_start,
            "memory_end": self.memory_end,
            "unload_events": list(self.unload_events),
        }

    def _snapshot(self) -> dict:
        if self.memory_snapshot is None:
            return empty_memory_snapshot()
        return self.memory_snapshot()
