from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from sanasprint_mlx.weights.inspect import TensorInfo


Status = Literal["mapped", "requires_review", "missing", "unexpected", "shape_mismatch"]


@dataclass(frozen=True)
class MappingEntry:
    source_key: str
    target_key: str | None
    source_shape: list[int]
    source_dtype: str
    component: str
    rule: str
    status: Status
    transpose_required: bool | Literal["unknown"]
    suggested_action: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    kind: str
    owning_rule: str
    expected_pattern: str
    matched_key: str | None
    actual_shape: list[int] | None
    expected_shape: list[int] | None
    suggested_action: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ComponentSummary:
    name: str
    files: list[str]
    tensor_count: int
    parameter_count: int
    dtype_counts: dict[str, int]
    parameter_count_by_dtype: dict[str, int]
    largest_tensors: list[dict]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MappingReport:
    schema_version: int
    snapshot_path: str
    config_summary: dict
    components: dict[str, ComponentSummary]
    mapping: list[MappingEntry]
    diagnostics: list[Diagnostic]

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "snapshot_path": self.snapshot_path,
            "config_summary": self.config_summary,
            "components": {name: summary.to_dict() for name, summary in self.components.items()},
            "mapping": [entry.to_dict() for entry in self.mapping],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


REQUIRED_PATTERNS = {
    "patch_embed": "transformer.patch_embed.",
    "transformer_block": "transformer.transformer_blocks.",
}

TRANSPOSE_SENSITIVE_SUFFIXES = (".weight",)
TRANSPOSE_SENSITIVE_FRAGMENTS = (".attn", ".to_q", ".to_k", ".to_v", ".to_out", ".ff.")


def build_mapping_report(
    tensor_infos: list[TensorInfo],
    *,
    snapshot_path: str,
    config_summary: dict | None = None,
) -> MappingReport:
    mapping = [_map_tensor(info, config_summary or {}) for info in tensor_infos if info.component in ("transformer", "unknown")]
    mapping.extend(_missing_entries(tensor_infos))
    diagnostics = _diagnostics(tensor_infos, mapping)
    return MappingReport(
        schema_version=1,
        snapshot_path=snapshot_path,
        config_summary=config_summary or {},
        components=component_summaries(tensor_infos),
        mapping=mapping,
        diagnostics=diagnostics,
    )


def component_summaries(tensor_infos: list[TensorInfo]) -> dict[str, ComponentSummary]:
    components = {name: [] for name in ("text_encoder", "transformer", "vae", "scheduler", "unknown")}
    for info in tensor_infos:
        components.setdefault(info.component, []).append(info)

    return {
        name: _component_summary(name, infos)
        for name, infos in components.items()
        if infos or name in ("text_encoder", "transformer", "vae", "unknown")
    }


def _component_summary(name: str, infos: list[TensorInfo]) -> ComponentSummary:
    dtype_counts: dict[str, int] = {}
    parameter_count_by_dtype: dict[str, int] = {}
    for info in infos:
        dtype_counts[info.dtype] = dtype_counts.get(info.dtype, 0) + 1
        parameter_count_by_dtype[info.dtype] = parameter_count_by_dtype.get(info.dtype, 0) + info.parameter_count

    largest = sorted(infos, key=lambda info: info.parameter_count, reverse=True)[:10]
    return ComponentSummary(
        name=name,
        files=sorted({info.file for info in infos}),
        tensor_count=len(infos),
        parameter_count=sum(info.parameter_count for info in infos),
        dtype_counts=dtype_counts,
        parameter_count_by_dtype=parameter_count_by_dtype,
        largest_tensors=[
            {
                "name": info.name,
                "file": info.file,
                "shape": info.shape,
                "dtype": info.dtype,
                "parameter_count": info.parameter_count,
            }
            for info in largest
        ],
    )


def _map_tensor(info: TensorInfo, config_summary: dict) -> MappingEntry:
    if info.component != "transformer":
        return MappingEntry(
            source_key=info.name,
            target_key=None,
            source_shape=info.shape,
            source_dtype=info.dtype,
            component=info.component,
            rule="component_filter",
            status="unexpected",
            transpose_required=False,
            suggested_action="ignore for transformer mapping or handle in the component-specific mapper",
        )

    target_key = _target_key(info.name)
    expected_shape = _expected_shape(info.name, config_summary)
    if expected_shape is not None and expected_shape != info.shape:
        return MappingEntry(
            source_key=info.name,
            target_key=target_key,
            source_shape=info.shape,
            source_dtype=info.dtype,
            component=info.component,
            rule="shape_from_config",
            status="shape_mismatch",
            transpose_required=False,
            suggested_action="verify config dimensions or update the mapping rule",
        )

    if _requires_layout_review(info.name):
        return MappingEntry(
            source_key=info.name,
            target_key=target_key,
            source_shape=info.shape,
            source_dtype=info.dtype,
            component=info.component,
            rule="transpose_sensitive",
            status="requires_review",
            transpose_required="unknown",
            suggested_action="validate tensor layout with a PyTorch-vs-MLX parity test before loading",
        )

    return MappingEntry(
        source_key=info.name,
        target_key=target_key,
        source_shape=info.shape,
        source_dtype=info.dtype,
        component=info.component,
        rule="strip_transformer_prefix",
        status="mapped",
        transpose_required=False,
        suggested_action="load directly after shape parity is verified",
    )


def _requires_layout_review(key: str) -> bool:
    return key.endswith(TRANSPOSE_SENSITIVE_SUFFIXES) and any(fragment in key for fragment in TRANSPOSE_SENSITIVE_FRAGMENTS)


def _target_key(source_key: str) -> str:
    if source_key.startswith("transformer."):
        return "mlx_transformer." + source_key.removeprefix("transformer.")
    return "mlx_transformer." + source_key


def _expected_shape(source_key: str, config_summary: dict) -> list[int] | None:
    hidden_size = config_summary.get("hidden_size")
    in_channels = config_summary.get("in_channels")
    out_channels = config_summary.get("out_channels")
    caption_channels = config_summary.get("caption_channels")
    patch_size = config_summary.get("patch_size", 1)
    normalized_key = source_key.removeprefix("transformer.")
    if normalized_key == "patch_embed.proj.weight" and hidden_size is not None and in_channels is not None:
        patch_size = int(patch_size)
        return [int(hidden_size), int(in_channels), patch_size, patch_size]
    if normalized_key == "patch_embed.proj.bias" and hidden_size is not None:
        return [int(hidden_size)]
    if normalized_key == "proj_out.weight" and hidden_size is not None and out_channels is not None:
        patch_size = int(patch_size)
        return [int(out_channels) * patch_size * patch_size, int(hidden_size)]
    if normalized_key == "proj_out.bias" and out_channels is not None:
        patch_size = int(patch_size)
        return [int(out_channels) * patch_size * patch_size]
    if normalized_key == "caption_projection.linear_1.weight" and hidden_size is not None and caption_channels is not None:
        return [int(hidden_size), int(caption_channels)]
    if normalized_key == "caption_projection.linear_1.bias" and hidden_size is not None:
        return [int(hidden_size)]
    if normalized_key == "caption_projection.linear_2.weight" and hidden_size is not None:
        return [int(hidden_size), int(hidden_size)]
    if normalized_key == "caption_projection.linear_2.bias" and hidden_size is not None:
        return [int(hidden_size)]
    if normalized_key == "caption_norm.weight" and hidden_size is not None:
        return [int(hidden_size)]
    return None


def _missing_entries(tensor_infos: list[TensorInfo]) -> list[MappingEntry]:
    transformer_infos = [info for info in tensor_infos if info.component == "transformer"]
    entries = []
    for rule, pattern in REQUIRED_PATTERNS.items():
        if not any(_matches_required_pattern(info.name, pattern) for info in transformer_infos):
            entries.append(
                MappingEntry(
                    source_key=f"{pattern}*",
                    target_key=f"mlx_transformer.{pattern.removeprefix('transformer.')}*",
                    source_shape=[],
                    source_dtype="unknown",
                    component="transformer",
                    rule=rule,
                    status="missing",
                    transpose_required="unknown",
                    suggested_action="check the model revision or update mapping rules for this Diffusers version",
                )
            )
    return entries


def _diagnostics(tensor_infos: list[TensorInfo], mapping: list[MappingEntry]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    transformer_infos = [info for info in tensor_infos if info.component == "transformer"]
    for rule, pattern in REQUIRED_PATTERNS.items():
        if not any(_matches_required_pattern(info.name, pattern) for info in transformer_infos):
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    kind="missing",
                    owning_rule=rule,
                    expected_pattern=f"{pattern}*",
                    matched_key=None,
                    actual_shape=None,
                    expected_shape=None,
                    suggested_action="check the model revision or update mapping rules for this Diffusers version",
                )
            )

    for entry in mapping:
        if entry.status in ("unexpected", "requires_review", "shape_mismatch", "missing"):
            diagnostics.append(
                Diagnostic(
                    severity="error" if entry.status in ("missing", "shape_mismatch") else "warning" if entry.status != "unexpected" else "info",
                    kind=entry.status,
                    owning_rule=entry.rule,
                    expected_pattern="transformer.*",
                    matched_key=entry.source_key,
                    actual_shape=entry.source_shape,
                    expected_shape=None,
                    suggested_action=entry.suggested_action,
                )
            )
    return diagnostics


def _matches_required_pattern(key: str, pattern: str) -> bool:
    if key.startswith(pattern):
        return True
    if pattern.startswith("transformer.") and key.startswith(pattern.removeprefix("transformer.")):
        return True
    return False
