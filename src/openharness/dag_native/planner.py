"""DAG Planner — orchestrates the full DAG-native planning pipeline.

Pipeline:
1. Parse input task → structured nodes
2. Build TaskDAG
3. Topological sort
4. Reverse topological sort
5. Critical path identification
6. Subagent assignment
7. Execution plan generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.algorithms import (
    topological_sort,
    reverse_topological_sort,
    topological_levels,
)
from openharness.dag_native.critical_path import (
    identify_critical_path,
    CriticalPathResult,
)
from openharness.dag_native.task_parser import parse_coding_task, ParsedTask


@dataclass
class ExecutionPlan:
    """A complete execution plan for a coding task."""
    task_name: str
    dag: TaskDAG
    topological_order: list[str]
    reverse_topological_order: list[str]
    levels: list[list[str]]
    critical_path_result: CriticalPathResult
    subagent_assignments: dict[str, str]  # node_id -> subagent type
    parallel_groups: list[list[str]]
    sequential_chain: list[str]

    # Timing estimates
    estimated_total_duration: float = 0.0
    estimated_critical_path_duration: float = 0.0

    # Metadata
    planning_time: float = 0.0  # seconds spent planning
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerConfig:
    """Configuration for the DAG planner."""
    min_nodes_for_dag: int = 3           # Below this, skip DAG decomposition
    max_parallel_agents: int = 8         # Max concurrent subagents
    default_complexity: float = 1.0      # Default node complexity
    enable_critical_path_priority: bool = True
    enable_shared_context: bool = True
    assign_subagents: bool = True


class DAGPlanner:
    """Orchestrates the full DAG-native planning pipeline.

    Integrates with OpenHarness's coordinator module for subagent definitions
    and the task manager for execution.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self._config = config or PlannerConfig()

    def plan(self, task_description: str, task_name: str = "coding_task") -> ExecutionPlan:
        """Execute the full planning pipeline.

        Args:
            task_description: Natural language description of the coding task.
            task_name: Name for the task.

        Returns:
            A complete ExecutionPlan.
        """
        import time
        start_time = time.monotonic()

        # Step 1: Parse input task
        parsed = parse_coding_task(task_description, task_name)

        # Step 2: Build TaskDAG
        dag = TaskDAG(name=task_name, description=task_description)

        for node in parsed.nodes:
            dag.add_node(node)

        for source_id, target_id, edge_type in parsed.edges:
            dag.add_edge(source_id, target_id, EdgeType(edge_type))

        # If too few nodes, add meta-edges for implicit ordering
        if dag.node_count >= self._config.min_nodes_for_dag and dag.edge_count == 0:
            self._add_implicit_edges(dag, parsed)

        dag.build()

        # Step 3: Topological sort
        topo_order = topological_sort(dag)

        # Step 4: Reverse topological sort
        rev_topo_order = reverse_topological_sort(dag)

        # Step 5: Critical path identification
        cp_result = identify_critical_path(dag)

        # Step 6: Topological levels
        levels = topological_levels(dag)

        # Step 7: Subagent assignment
        subagent_assignments = {}
        if self._config.assign_subagents:
            subagent_assignments = self._assign_subagents(dag, parsed)

        # Step 8: Identify parallel groups from levels
        parallel_groups = [level for level in levels if len(level) > 1]

        # Build sequential chain (critical path nodes in order)
        sequential_chain = [
            nid for nid in topo_order if nid in cp_result.critical_nodes
        ]

        planning_time = time.monotonic() - start_time

        return ExecutionPlan(
            task_name=task_name,
            dag=dag,
            topological_order=topo_order,
            reverse_topological_order=rev_topo_order,
            levels=levels,
            critical_path_result=cp_result,
            subagent_assignments=subagent_assignments,
            parallel_groups=parallel_groups,
            sequential_chain=sequential_chain,
            estimated_total_duration=cp_result.critical_path_length,
            estimated_critical_path_duration=cp_result.critical_path_length,
            planning_time=planning_time,
            metadata={
                "detected_phases": parsed.detected_phases,
                "total_nodes": dag.node_count,
                "total_edges": dag.edge_count,
                "config": {
                    "max_parallel_agents": self._config.max_parallel_agents,
                    "critical_path_priority": self._config.enable_critical_path_priority,
                    "shared_context": self._config.enable_shared_context,
                },
            },
        )

    def plan_from_nodes(
        self,
        nodes: list[TaskNode],
        edges: list[tuple[str, str, str]],
        task_name: str = "coding_task",
    ) -> ExecutionPlan:
        """Build a plan from pre-parsed nodes and edges."""
        import time
        start_time = time.monotonic()

        dag = TaskDAG(name=task_name)
        for node in nodes:
            dag.add_node(node)
        for source_id, target_id, edge_type in edges:
            dag.add_edge(source_id, target_id, EdgeType(edge_type))
        dag.build()

        topo_order = topological_sort(dag)
        rev_topo_order = reverse_topological_sort(dag)
        cp_result = identify_critical_path(dag)
        levels = topological_levels(dag)
        subagent_assignments = self._assign_subagents(dag, None)
        parallel_groups = [level for level in levels if len(level) > 1]
        sequential_chain = [nid for nid in topo_order if nid in cp_result.critical_nodes]

        return ExecutionPlan(
            task_name=task_name,
            dag=dag,
            topological_order=topo_order,
            reverse_topological_order=rev_topo_order,
            levels=levels,
            critical_path_result=cp_result,
            subagent_assignments=subagent_assignments,
            parallel_groups=parallel_groups,
            sequential_chain=sequential_chain,
            estimated_total_duration=cp_result.critical_path_length,
            estimated_critical_path_duration=cp_result.critical_path_length,
            planning_time=time.monotonic() - start_time,
        )

    def _add_implicit_edges(
        self,
        dag: TaskDAG,
        parsed: ParsedTask,
    ) -> None:
        """Add implicit sequential edges between nodes when no edges declared."""
        node_ids = list(dag.nodes.keys())
        for i in range(len(node_ids) - 1):
            source = node_ids[i]
            target = node_ids[i + 1]
            # Check if edge already exists
            exists = any(
                e.source_id == source and e.target_id == target
                for e in dag.edges
            )
            if not exists:
                dag.add_edge(source, target, EdgeType.CONTROL_DEP)

    def _assign_subagents(
        self,
        dag: TaskDAG,
        parsed: ParsedTask | None,
    ) -> dict[str, str]:
        """Assign subagent types to task nodes based on content and dependencies.

        Uses heuristics to determine the best subagent type for each node.
        """
        from openharness.coordinator.agent_definitions import get_builtin_agent_definitions

        available_agents = get_builtin_agent_definitions()
        agent_names = {a.name for a in available_agents} if available_agents else {
            "general-purpose", "Explore", "Plan",
        }

        assignments: dict[str, str] = {}

        for nid, node in dag.nodes.items():
            desc_lower = node.description.lower()

            if any(kw in desc_lower for kw in ["test", "pytest", "assert"]):
                assignments[nid] = "general-purpose"  # Testing agent
            elif any(kw in desc_lower for kw in ["design", "architecture", "plan"]):
                assignments[nid] = "Plan" if "Plan" in agent_names else "general-purpose"
            elif any(kw in desc_lower for kw in ["search", "find", "explore", "grep"]):
                assignments[nid] = "Explore" if "Explore" in agent_names else "general-purpose"
            elif any(kw in desc_lower for kw in ["implement", "code", "write", "edit"]):
                assignments[nid] = "general-purpose"
            elif any(kw in desc_lower for kw in ["review", "check", "audit"]):
                assignments[nid] = "general-purpose"
            elif any(kw in desc_lower for kw in ["document", "readme", "docs"]):
                assignments[nid] = "general-purpose"
            else:
                assignments[nid] = "general-purpose"

        return assignments
