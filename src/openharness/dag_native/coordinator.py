"""Subagent coordinator — dispatches task nodes to subagents.

Integrates with OpenHarness's:
- BackgroundTaskManager for subprocess management
- AgentDefinition for subagent specs
- TeamRegistry for agent coordination

Coordinates execution following the DAG plan:
1. Critical path nodes get priority scheduling
2. Same-level nodes can be dispatched in parallel (up to max_parallel)
3. Shared context is injected based on reuse policy
4. Results are collected and knowledge is stored for dependents
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from openharness.dag_native.graph import TaskDAG, TaskNode, NodeStatus
from openharness.dag_native.planner import ExecutionPlan
from openharness.dag_native.shared_context import (
    SharedContextManager,
    SharedKnowledgeStore,
    ContextBudgetTracker,
    KnowledgeEntry,
    KnowledgeEntryType,
)


class CoordinationMode(str, Enum):
    """How the coordinator dispatches work."""
    SEQUENTIAL = "sequential"           # One node at a time
    PARALLEL_SAME_LEVEL = "parallel"    # Same-topological-level nodes in parallel
    CRITICAL_PATH_FIRST = "critical"    # Prioritize critical path nodes


@dataclass
class SubagentSpec:
    """Specification for a subagent assigned to a task node."""
    node_id: str
    agent_type: str
    prompt: str
    context: str = ""
    context_size: int = 0
    isolation: str = "worktree"  # or "remote"
    max_turns: int = 24
    priority: float = 0.0  # Higher = more urgent


@dataclass
class SubagentResult:
    """Result returned by a subagent."""
    node_id: str
    success: bool
    output: str = ""
    error: str = ""
    duration: float = 0.0
    knowledge_produced: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass
class CoordinationState:
    """Current state of the coordination process."""
    plan: ExecutionPlan
    node_status: dict[str, NodeStatus]
    active_subagents: dict[str, SubagentSpec]  # node_id -> spec
    completed_results: dict[str, SubagentResult]
    pending_queue: list[str]                    # node_ids ready to run
    failed_nodes: list[str]
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def is_complete(self) -> bool:
        return all(
            s in (NodeStatus.COMPLETED, NodeStatus.SKIPPED, NodeStatus.FAILED)
            for s in self.node_status.values()
        )

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.completed_results.values() if r.success)

    @property
    def failure_count(self) -> int:
        return len(self.failed_nodes)


class SubagentCoordinator:
    """Coordinates subagent execution following a DAG plan.

    This is the executor that dispatches work based on the plan produced
    by DAGPlanner.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        shared_context: SharedContextManager | None = None,
        mode: CoordinationMode = CoordinationMode.CRITICAL_PATH_FIRST,
        max_parallel: int = 4,
        cwd: str | Path | None = None,
    ) -> None:
        self._plan = plan
        self._dag = plan.dag
        self._shared_context = shared_context or SharedContextManager(plan.dag)
        self._mode = mode
        self._max_parallel = max_parallel
        self._cwd = Path(cwd) if cwd else Path.cwd()

        # Execution callbacks (allow injecting mock executors for testing)
        self._node_executor: Callable[[SubagentSpec], SubagentResult] | None = None

    @property
    def dag(self) -> TaskDAG:
        return self._dag

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    def set_node_executor(
        self,
        executor: Callable[[SubagentSpec], SubagentResult],
    ) -> None:
        """Set a custom node executor (useful for testing/mocking)."""
        self._node_executor = executor

    def create_state(self) -> CoordinationState:
        """Create the initial coordination state."""
        node_status: dict[str, NodeStatus] = {
            nid: NodeStatus.PENDING for nid in self._dag.nodes
        }
        return CoordinationState(
            plan=self._plan,
            node_status=node_status,
            active_subagents={},
            completed_results={},
            pending_queue=[],
            failed_nodes=[],
        )

    def get_ready_nodes(self, state: CoordinationState) -> list[str]:
        """Find nodes whose dependencies are all satisfied."""
        ready: list[str] = []
        for nid, status in state.node_status.items():
            if status != NodeStatus.PENDING:
                continue
            node = self._dag.nodes.get(nid)
            if not node:
                continue
            # Check all dependencies are completed
            all_deps_met = True
            for dep_id in node.input_dependencies:
                if state.node_status.get(dep_id) not in (
                    NodeStatus.COMPLETED,
                    NodeStatus.SKIPPED,
                ):
                    all_deps_met = False
                    break
            # Also check edge-based dependencies
            for edge in self._dag.edges:
                if edge.target_id == nid:
                    if state.node_status.get(edge.source_id) not in (
                        NodeStatus.COMPLETED,
                        NodeStatus.SKIPPED,
                    ):
                        all_deps_met = False
                        break

            if all_deps_met:
                ready.append(nid)
        return ready

    def prioritize(self, ready_nodes: list[str]) -> list[str]:
        """Prioritize ready nodes based on the coordination mode."""
        if self._mode == CoordinationMode.SEQUENTIAL:
            # One at a time in topological order
            for nid in self._plan.topological_order:
                if nid in ready_nodes:
                    return [nid]
            return ready_nodes[:1]

        if self._mode == CoordinationMode.CRITICAL_PATH_FIRST:
            # Sort: critical path nodes first, then by complexity
            cp_nodes = set(self._plan.critical_path_result.critical_nodes)

            def priority(nid: str) -> tuple[float, float]:
                node = self._dag.nodes.get(nid)
                on_critical = 0.0 if nid in cp_nodes else 1.0
                complexity = node.estimated_complexity if node else 1.0
                return (on_critical, -complexity)  # lower tuple = higher priority

            return sorted(ready_nodes, key=priority)

        # PARALLEL_SAME_LEVEL: Keep level order
        for level in self._plan.levels:
            level_ready = [nid for nid in level if nid in ready_nodes]
            if level_ready:
                return level_ready

        return ready_nodes

    def build_spec(self, node_id: str) -> SubagentSpec:
        """Build a subagent spec for a task node."""
        node = self._dag.nodes[node_id]
        agent_type = self._plan.subagent_assignments.get(node_id, "general-purpose")

        # Get shared context for this node
        context, context_size = self._shared_context.get_context_for_node(
            node_id,
            base_context=self._dag.description,
        )

        # Build prompt from node description + dependencies summary
        prompt = f"Task: {node.name}\n\n{node.description}"
        if node.input_dependencies:
            deps_summary = []
            for dep_id in node.input_dependencies:
                dep_node = self._dag.nodes.get(dep_id)
                if dep_node:
                    deps_summary.append(f"- {dep_id}: {dep_node.name}")
            if deps_summary:
                prompt += f"\n\nDependencies (already completed):\n" + "\n".join(deps_summary)

        # Priority: critical path nodes get higher priority
        priority = 0.0
        if node_id in self._plan.critical_path_result.critical_nodes:
            priority = 10.0 - self._plan.critical_path_result.slack.get(node_id, 0.0)

        return SubagentSpec(
            node_id=node_id,
            agent_type=agent_type,
            prompt=prompt,
            context=context,
            context_size=context_size,
            priority=priority,
        )

    def execute_node(self, spec: SubagentSpec) -> SubagentResult:
        """Execute a single task node via its assigned subagent."""
        start = time.time()

        if self._node_executor is not None:
            result = self._node_executor(spec)
            result.duration = time.time() - start
            return result

        # Default: simulated execution (in production, this would spawn a real subagent)
        # In reality, this would use openharness.tasks.manager.BackgroundTaskManager
        # or openharness.tools.agent_tool.AgentTool
        duration = time.time() - start

        # Simulate knowledge production based on node content
        knowledge: list[dict[str, Any]] = []
        node = self._dag.nodes.get(spec.node_id)
        if node and node.produces_reusable_knowledge:
            knowledge.append({
                "type": KnowledgeEntryType.CODE_CONTEXT.value,
                "content": f"Completed task: {node.name}. Summary of work done.",
            })

        return SubagentResult(
            node_id=spec.node_id,
            success=True,
            output=f"[simulated] Task '{spec.node_id}' completed in {duration:.2f}s",
            duration=duration,
            knowledge_produced=knowledge,
        )

    def execute_all_sequential(self) -> CoordinationState:
        """Execute all nodes in topological order, one at a time."""
        state = self.create_state()
        order = self._plan.topological_order

        for nid in order:
            state.node_status[nid] = NodeStatus.RUNNING
            spec = self.build_spec(nid)
            result = self.execute_node(spec)
            state.completed_results[nid] = result

            if result.success:
                state.node_status[nid] = NodeStatus.COMPLETED
                # Store produced knowledge
                if result.knowledge_produced:
                    self._shared_context.store_node_knowledge(
                        nid, result.knowledge_produced
                    )
            else:
                state.node_status[nid] = NodeStatus.FAILED
                state.failed_nodes.append(nid)

        state.end_time = time.time()
        return state

    def execute_all_parallel(self) -> CoordinationState:
        """Execute all nodes respecting DAG dependencies with parallel dispatch.

        This is a simplified synchronous simulation. In production, this would
        use asyncio with OpenHarness's BackgroundTaskManager.
        """
        state = self.create_state()

        while not state.is_complete:
            # Find ready nodes
            ready = self.get_ready_nodes(state)
            if not ready:
                # Check for deadlock (nodes pending but none ready)
                pending = [
                    nid for nid, s in state.node_status.items()
                    if s == NodeStatus.PENDING
                ]
                if pending:
                    # Some dependencies may have failed — mark unstartable as failed
                    for nid in pending:
                        node = self._dag.nodes.get(nid)
                        deps_met = True
                        if node:
                            for dep_id in node.input_dependencies:
                                if state.node_status.get(dep_id) == NodeStatus.FAILED:
                                    deps_met = False
                                    break
                        if not deps_met:
                            state.node_status[nid] = NodeStatus.SKIPPED
                    continue
                break

            # Prioritize and cap to max_parallel
            prioritized = self.prioritize(ready)
            batch = prioritized[:self._max_parallel]

            # Execute batch
            for nid in batch:
                state.node_status[nid] = NodeStatus.RUNNING
                spec = self.build_spec(nid)
                result = self.execute_node(spec)
                state.completed_results[nid] = result

                if result.success:
                    state.node_status[nid] = NodeStatus.COMPLETED
                    if result.knowledge_produced:
                        self._shared_context.store_node_knowledge(
                            nid, result.knowledge_produced,
                        )
                else:
                    state.node_status[nid] = NodeStatus.FAILED
                    state.failed_nodes.append(nid)

        state.end_time = time.time()
        return state

    def get_execution_summary(self, state: CoordinationState) -> dict[str, Any]:
        """Generate an execution summary."""
        total_time = (state.end_time or time.time()) - state.start_time
        return {
            "task_name": self._plan.task_name,
            "total_nodes": len(state.node_status),
            "completed": state.success_count,
            "failed": state.failure_count,
            "skipped": sum(
                1 for s in state.node_status.values() if s == NodeStatus.SKIPPED
            ),
            "total_duration": total_time,
            "critical_path_length": self._plan.critical_path_result.critical_path_length,
            "max_parallel": self._max_parallel,
            "mode": self._mode.value,
            "context_metrics": self._shared_context.get_metrics(),
        }
