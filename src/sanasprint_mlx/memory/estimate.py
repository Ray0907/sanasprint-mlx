from __future__ import annotations

from math import prod
from typing import Any


GB = 1024**3
BYTES_PER_DTYPE = {
    "F64": 8,
    "float64": 8,
    "F32": 4,
    "float32": 4,
    "BF16": 2,
    "F16": 2,
    "float16": 2,
    "I8": 1,
    "U8": 1,
    "I4": 0.5,
    "U4": 0.5,
}
MEMORY_BUDGETS_GB = {512: 11, 768: 13, 1024: 15}
RESOLUTIONS = (512, 768, 1024)
REQUIRED_COMPONENTS = ("text_encoder", "transformer", "vae")


def estimate_memory(
    weight_report: dict[str, Any],
    *,
    runtime_overhead_bytes: int = int(1.5 * GB),
    mlx_cache_reserve_bytes: int = 512 * 1024**2,
) -> dict[str, Any]:
    warnings: list[str] = []
    component_memory = weight_memory_by_component(weight_report, warnings=warnings)
    has_required_component_data = _has_required_component_data(component_memory)
    config = weight_report.get("config_summary", {})
    phase_estimates: dict[str, dict[str, dict[str, int]]] = {}
    resolution_estimates = []

    for resolution in RESOLUTIONS:
        phases = _phase_estimates(
            resolution,
            component_memory,
            config,
            runtime_overhead_bytes,
            mlx_cache_reserve_bytes,
        )
        phase_estimates[str(resolution)] = phases
        largest_phase_name, largest_phase = max(phases.items(), key=lambda item: item[1]["estimated_bytes"])
        budget_gb = MEMORY_BUDGETS_GB[resolution]
        estimated_peak_bytes = largest_phase["estimated_bytes"]
        estimated_peak_gb = estimated_peak_bytes / GB
        headroom_gb = budget_gb - estimated_peak_gb
        status = _status(estimated_peak_gb, budget_gb, known=has_required_component_data)
        resolution_estimates.append(
            {
                "height": resolution,
                "width": resolution,
                "estimated_peak_bytes": estimated_peak_bytes,
                "estimated_peak_gb": estimated_peak_gb,
                "budget_gb": budget_gb,
                "headroom_gb": headroom_gb,
                "status": status,
                "largest_phase": largest_phase_name,
                "largest_tensor_groups": _largest_tensor_groups(component_memory),
            }
        )

    recommendations = {
        str(item["height"]): "experimental" if item["height"] == 1024 and item["status"] in ("NO_GO", "UNKNOWN") else item["status"].lower()
        for item in resolution_estimates
    }
    final_decision = _final_decision(resolution_estimates)

    return {
        "schema_version": 1,
        "source_weight_report": weight_report.get("snapshot_path", ""),
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "final_decision": final_decision,
        "budgets": {str(key): value for key, value in MEMORY_BUDGETS_GB.items()},
        "component_weight_memory": component_memory,
        "resolution_estimates": resolution_estimates,
        "phase_estimates": phase_estimates,
        "mlx_probe": {},
        "recommendations": recommendations,
        "warnings": warnings,
    }


def weight_memory_by_component(
    weight_report: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    warnings = warnings if warnings is not None else []
    result = {}
    components = weight_report.get("components", {})
    for component in REQUIRED_COMPONENTS:
        if component not in components:
            warnings.append(f"missing required component {component}; memory estimate is UNKNOWN")
            result[component] = {
                "component": component,
                "weight_bytes_by_dtype": {},
                "total_weight_bytes": 0,
                "largest_tensors": [],
                "warning": "missing or empty required component",
            }

    for component, summary in components.items():
        by_dtype = {}
        for dtype, count in summary.get("parameter_count_by_dtype", {}).items():
            bytes_per_param = BYTES_PER_DTYPE.get(dtype)
            if bytes_per_param is None:
                warnings.append(f"unknown dtype {dtype}; defaulting to F32 size")
                bytes_per_param = 4
            by_dtype[dtype] = int(count * bytes_per_param)

        warning = None
        if component in REQUIRED_COMPONENTS and not by_dtype:
            warning = "missing or empty required component"
            warnings.append(f"missing required component {component}; memory estimate is UNKNOWN")

        result[component] = {
            "component": component,
            "weight_bytes_by_dtype": by_dtype,
            "total_weight_bytes": sum(by_dtype.values()),
            "largest_tensors": summary.get("largest_tensors", []),
            "warning": warning,
        }
    return result


def _phase_estimates(
    resolution: int,
    component_memory: dict[str, dict[str, Any]],
    config: dict[str, Any],
    runtime_overhead_bytes: int,
    mlx_cache_reserve_bytes: int,
) -> dict[str, dict[str, int]]:
    latent_channels = int(config.get("in_channels", 32))
    hidden_size = int(config.get("hidden_size", 2304))
    vae_scale_factor = int(config.get("vae_scale_factor", 32))
    latent_h = max(resolution // vae_scale_factor, 1)
    latent_w = max(resolution // vae_scale_factor, 1)
    latent_bytes = latent_channels * latent_h * latent_w * 2
    hidden_working_bytes = latent_h * latent_w * hidden_size * 2
    rgb_bytes = resolution * resolution * 3 * 4

    text_weight = _component_bytes(component_memory, "text_encoder")
    transformer_weight = _component_bytes(component_memory, "transformer") + _component_bytes(component_memory, "unknown")
    vae_weight = _component_bytes(component_memory, "vae")

    return {
        "text_encode": {
            "estimated_bytes": runtime_overhead_bytes + text_weight + 32 * 1024**2,
        },
        "denoise": {
            "estimated_bytes": runtime_overhead_bytes
            + mlx_cache_reserve_bytes
            + transformer_weight
            + latent_bytes
            + int(hidden_working_bytes * 12),
        },
        "decode": {
            "estimated_bytes": runtime_overhead_bytes
            + vae_weight
            + latent_bytes
            + int((latent_bytes + rgb_bytes) * 8),
        },
    }


def _component_bytes(component_memory: dict[str, dict[str, Any]], component: str) -> int:
    return int(component_memory.get(component, {}).get("total_weight_bytes", 0))


def _status(estimated_peak_gb: float, budget_gb: int, *, known: bool) -> str:
    if not known:
        return "UNKNOWN"
    if estimated_peak_gb >= budget_gb:
        return "NO_GO"
    if (budget_gb - estimated_peak_gb) / budget_gb < 0.15:
        return "RISK"
    return "GO"


def _final_decision(resolution_estimates: list[dict[str, Any]]) -> dict[str, Any]:
    by_resolution = {item["height"]: item for item in resolution_estimates}
    blocking = [
        resolution
        for resolution in (512,)
        if by_resolution[resolution]["status"] in ("NO_GO", "UNKNOWN")
    ]
    if blocking:
        return {
            "status": "BLOCKED",
            "can_start_feature_3": False,
            "requires_user_approval": True,
            "reason": "512x512 estimate is not viable under the 11GB budget",
            "blocking_resolutions": blocking,
        }

    risky = [
        resolution
        for resolution in (768, 1024)
        if by_resolution[resolution]["status"] in ("NO_GO", "UNKNOWN", "RISK")
    ]
    return {
        "status": "RISK" if risky else "GO",
        "can_start_feature_3": True,
        "requires_user_approval": False,
        "reason": "512x512 is viable; later resolutions may still be risky" if risky else "512/768/1024 estimates are below budget with headroom",
        "blocking_resolutions": [],
    }


def _largest_tensor_groups(component_memory: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for component, summary in component_memory.items():
        for tensor in summary.get("largest_tensors", [])[:3]:
            groups.append({"component": component, **tensor})
    return sorted(groups, key=lambda item: item.get("parameter_count", 0), reverse=True)[:10]


def _has_required_component_data(component_memory: dict[str, dict[str, Any]]) -> bool:
    return all(
        component in component_memory
        and component_memory[component].get("total_weight_bytes", 0) > 0
        and not component_memory[component].get("warning")
        for component in REQUIRED_COMPONENTS
    )
