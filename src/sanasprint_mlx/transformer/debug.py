from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sanasprint_mlx.transformer.parity import compare_arrays


@dataclass
class DebugCapture:
    activations: dict[str, object] = field(default_factory=dict)

    def record(self, name: str, value) -> None:
        self.activations[name] = value


def compare_debug_captures(actual: DebugCapture, expected: DebugCapture) -> dict[str, dict]:
    report: dict[str, dict] = {}
    actual_names = set(actual.activations)
    expected_names = set(expected.activations)
    for name in sorted(actual_names | expected_names):
        if name not in actual_names:
            report[name] = {
                "status": "missing_actual",
                "passes_full_denoiser_tolerance": False,
            }
            continue
        if name not in expected_names:
            report[name] = {
                "status": "missing_expected",
                "passes_full_denoiser_tolerance": False,
            }
            continue
        metrics = compare_arrays(np.array(actual.activations[name]), np.array(expected.activations[name]))
        report[name] = {"status": "compared", **metrics}
    return report
