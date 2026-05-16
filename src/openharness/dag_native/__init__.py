"""DAG-Native Coding Harness — incremental extension on OpenHarness.

Provides:
- Task parsing to structured task DAG nodes
- Topological / reverse topological sort
- Critical path identification
- Subagent planning and coordination
- Same-path shared prerequisite context management
- Experiment runner with metrics and trace logging
"""

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.algorithms import topological_sort, reverse_topological_sort
from openharness.dag_native.critical_path import identify_critical_path, CriticalPathResult
from openharness.dag_native.planner import DAGPlanner
from openharness.dag_native.shared_context import SharedContextManager, PathSignature
from openharness.dag_native.experiments import ExperimentRunner, ExperimentConfig
from openharness.dag_native.metrics import MetricsLogger, MetricsRecord

__all__ = [
    "TaskDAG",
    "TaskNode",
    "EdgeType",
    "topological_sort",
    "reverse_topological_sort",
    "identify_critical_path",
    "CriticalPathResult",
    "DAGPlanner",
    "SharedContextManager",
    "PathSignature",
    "ExperimentRunner",
    "ExperimentConfig",
    "MetricsLogger",
    "MetricsRecord",
]
