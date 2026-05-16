"""Critical path identification for task DAGs.

Computes the critical path through a TaskDAG based on estimated node durations.
The critical path determines the minimum makespan of the task execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openharness.dag_native.graph import TaskDAG


@dataclass
class CriticalPathResult:
    """Result of critical path analysis on a TaskDAG."""

    critical_path: list[str]           # node_ids on the critical path
    critical_path_length: float        # sum of estimated durations on critical path
    earliest_start: dict[str, float]   # node_id -> earliest start time
    latest_start: dict[str, float]     # node_id -> latest start time
    slack: dict[str, float]            # node_id -> total float (0 = critical)
    node_durations: dict[str, float]   # node_id -> estimated duration
    total_nodes: int
    total_edges: int
    bottleneck_nodes: list[str] = field(default_factory=list)

    @property
    def critical_nodes(self) -> set[str]:
        """Nodes with zero slack (on the critical path)."""
        return {nid for nid, s in self.slack.items() if s == 0.0}

    @property
    def parallelizable_nodes(self) -> set[str]:
        """Nodes NOT on the critical path (have positive slack)."""
        return {nid for nid, s in self.slack.items() if s > 0.0}


def identify_critical_path(
    dag: "TaskDAG",
    duration_attr: str = "estimated_duration",
) -> CriticalPathResult:
    """Identify the critical path through a TaskDAG using the CPM algorithm.

    Uses the Critical Path Method (CPM):
    1. Forward pass: compute earliest start (ES) for each node
    2. Backward pass: compute latest start (LS) for each node
    3. Slack = LS - ES; nodes with zero slack are on the critical path

    Args:
        dag: The task DAG to analyze.
        duration_attr: Attribute name on TaskNode for duration (default: estimated_duration).

    Returns:
        CriticalPathResult with full analysis.
    """
    from openharness.dag_native.algorithms import topological_sort

    order = topological_sort(dag)

    # Collect node durations
    durations: dict[str, float] = {}
    for nid, node in dag.nodes.items():
        dur = getattr(node, duration_attr, 1.0)
        durations[nid] = max(float(dur), 0.01)  # ensure positive

    # Build predecessor map
    predecessors: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        predecessors[edge.target_id].append(edge.source_id)
    for nid, node in dag.nodes.items():
        for dep_id in node.input_dependencies:
            if dep_id in dag.nodes and dep_id not in predecessors[nid]:
                predecessors[nid].append(dep_id)

    # ---- Forward pass: Earliest Start ----
    earliest_start: dict[str, float] = {}
    earliest_finish: dict[str, float] = {}

    for nid in order:
        preds = predecessors.get(nid, [])
        if not preds:
            es = 0.0
        else:
            es = max(earliest_finish.get(p, 0.0) for p in preds)
        earliest_start[nid] = es
        earliest_finish[nid] = es + durations[nid]

    # ---- Backward pass: Latest Start ----
    reverse_order = list(reversed(order))
    project_duration = max(earliest_finish.values()) if earliest_finish else 0.0

    latest_finish: dict[str, float] = {}
    latest_start: dict[str, float] = {}

    # Build successor map for backward pass
    successors: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for nid, preds in predecessors.items():
        for pred in preds:
            if nid not in successors.get(pred, []):
                successors.setdefault(pred, []).append(nid)

    for nid in reverse_order:
        succs = successors.get(nid, [])
        if not succs:
            lf = project_duration
        else:
            lf = min(latest_start.get(s, project_duration) for s in succs)
        latest_finish[nid] = lf
        latest_start[nid] = lf - durations[nid]

    # ---- Compute Slack ----
    slack: dict[str, float] = {}
    for nid in dag.nodes:
        slack[nid] = latest_start[nid] - earliest_start[nid]

    # ---- Identify critical path ----
    critical_path = [nid for nid in order if slack.get(nid, 1.0) == 0.0]
    critical_path_length = sum(durations.get(nid, 0.0) for nid in critical_path)

    # Bottleneck nodes: on critical path AND have high fan-in
    bottleneck_nodes = [
        nid for nid in critical_path
        if len(predecessors.get(nid, [])) >= 3
    ]

    return CriticalPathResult(
        critical_path=critical_path,
        critical_path_length=critical_path_length,
        earliest_start=earliest_start,
        latest_start=latest_start,
        slack=slack,
        node_durations=durations,
        total_nodes=dag.node_count,
        total_edges=dag.edge_count,
        bottleneck_nodes=bottleneck_nodes,
    )
