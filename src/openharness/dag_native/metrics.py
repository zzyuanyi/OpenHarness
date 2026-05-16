"""Metrics logger for DAG-Native execution experiments.

Records quantitative metrics for comparing baseline vs. optimized execution.
All metrics explicitly labeled as: actual, estimated, simulated, or not_accessible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


MetricSource = Literal["actual", "estimated", "simulated", "not_accessible"]


@dataclass
class MetricsRecord:
    """A single metrics record from an experiment run."""

    # Identification
    experiment_id: str
    timestamp: float = field(default_factory=time.time)
    is_baseline: bool = True

    # Execution metrics
    total_makespan: float = 0.0
    planning_overhead: float = 0.0
    critical_path_length: float = 0.0

    # DAG metrics
    num_dag_nodes: int = 0
    num_dag_edges: int = 0
    num_subagents: int = 0
    max_parallel_subagents: int = 1

    # Context metrics
    shared_context_size: int = 0
    per_agent_context_size: int = 0
    duplicated_context_cost_baseline: int = 0
    duplicated_context_cost_optimized: int = 0
    context_reuse_rate: float = 0.0
    cache_hit_rate: float = 0.0
    estimated_token_saving: int = 0
    stale_cache_count: int = 0

    # Quality metrics
    test_pass_rate: float = 0.0
    code_generation_success: bool = False
    final_artifact_completeness: float = 0.0
    trace_length: float = 0.0
    failure_count: int = 0

    # Source annotations — per-field source classification
    sources: dict[str, MetricSource] = field(default_factory=dict)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = self._default_sources()

    @staticmethod
    def _default_sources() -> dict[str, MetricSource]:
        return {
            "total_makespan": "simulated",
            "planning_overhead": "actual",
            "critical_path_length": "actual",
            "num_dag_nodes": "actual",
            "num_dag_edges": "actual",
            "num_subagents": "actual",
            "max_parallel_subagents": "actual",
            "shared_context_size": "simulated",
            "per_agent_context_size": "estimated",
            "duplicated_context_cost_baseline": "estimated",
            "duplicated_context_cost_optimized": "simulated",
            "context_reuse_rate": "simulated",
            "cache_hit_rate": "simulated",
            "estimated_token_saving": "estimated",
            "stale_cache_count": "simulated",
            "test_pass_rate": "actual",
            "code_generation_success": "actual",
            "final_artifact_completeness": "estimated",
            "trace_length": "actual",
            "failure_count": "actual",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "is_baseline": self.is_baseline,
            "total_makespan": self.total_makespan,
            "planning_overhead": self.planning_overhead,
            "critical_path_length": self.critical_path_length,
            "num_dag_nodes": self.num_dag_nodes,
            "num_dag_edges": self.num_dag_edges,
            "num_subagents": self.num_subagents,
            "max_parallel_subagents": self.max_parallel_subagents,
            "shared_context_size": self.shared_context_size,
            "per_agent_context_size": self.per_agent_context_size,
            "duplicated_context_cost_baseline": self.duplicated_context_cost_baseline,
            "duplicated_context_cost_optimized": self.duplicated_context_cost_optimized,
            "context_reuse_rate": self.context_reuse_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "estimated_token_saving": self.estimated_token_saving,
            "stale_cache_count": self.stale_cache_count,
            "test_pass_rate": self.test_pass_rate,
            "code_generation_success": self.code_generation_success,
            "final_artifact_completeness": self.final_artifact_completeness,
            "trace_length": self.trace_length,
            "failure_count": self.failure_count,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class MetricsLogger:
    """Collects and persists metrics records."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._records: list[MetricsRecord] = []
        self._output_dir = Path(output_dir) if output_dir else Path("metrics_output")

    def log(self, record: MetricsRecord) -> None:
        """Record a metrics entry."""
        self._records.append(record)

    def get_records(self, baseline_only: bool | None = None) -> list[MetricsRecord]:
        """Get records, optionally filtered."""
        if baseline_only is None:
            return list(self._records)
        return [r for r in self._records if r.is_baseline == baseline_only]

    def get_comparison(self) -> dict[str, Any] | None:
        """Compare the latest baseline and optimized records."""
        baselines = [r for r in self._records if r.is_baseline]
        optimizeds = [r for r in self._records if not r.is_baseline]
        if not baselines or not optimizeds:
            return None

        baseline = baselines[-1]
        optimized = optimizeds[-1]

        return {
            "makespan_reduction": baseline.total_makespan - optimized.total_makespan,
            "makespan_reduction_pct": (
                (baseline.total_makespan - optimized.total_makespan)
                / max(baseline.total_makespan, 0.001) * 100
            ),
            "token_saving": baseline.duplicated_context_cost_baseline
                            - optimized.duplicated_context_cost_optimized,
            "context_reuse_rate_improvement": (
                optimized.context_reuse_rate - baseline.context_reuse_rate
            ),
            "cache_hit_rate_improvement": (
                optimized.cache_hit_rate - baseline.cache_hit_rate
            ),
        }

    def save(self, filename: str = "metrics.json") -> Path:
        """Save all metrics records to a JSON file."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename
        data = {
            "records": [r.to_dict() for r in self._records],
            "comparison": self.get_comparison(),
            "record_count": len(self._records),
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)
