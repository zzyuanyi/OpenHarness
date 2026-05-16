"""Task DAG data model and builder.

Defines the TaskNode and TaskDAG types used throughout the dag_native pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class EdgeType(str, Enum):
    """Types of edges in a task DAG."""
    DATA_DEP = "data_dep"           # Output of A is input to B
    CONTROL_DEP = "control_dep"     # B must wait for A to complete
    RESOURCE_DEP = "resource_dep"   # A and B share a constrained resource


class NodeStatus(str, Enum):
    """Execution status of a task node."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """A single node in the task DAG.

    Each node represents a discrete coding sub-task that can be assigned to a subagent.
    """

    node_id: str
    name: str
    description: str = ""
    # Inputs
    input_dependencies: list[str] = field(default_factory=list)  # node_ids this depends on
    input_files: list[str] = field(default_factory=list)
    # Outputs
    output_artifacts: list[str] = field(default_factory=list)    # file paths produced
    output_knowledge: list[str] = field(default_factory=list)    # knowledge keys produced
    # Execution
    assigned_subagent: str | None = None  # subagent type name
    can_parallelize: bool = True
    estimated_complexity: float = 1.0     # relative 1-10 scale
    estimated_duration: float = 1.0       # relative time units
    # Context
    required_context_size: int = 0        # estimated tokens needed
    produces_reusable_knowledge: bool = False
    # Risk
    risk_points: list[str] = field(default_factory=list)
    windows_concerns: list[str] = field(default_factory=list)
    # State
    status: NodeStatus = NodeStatus.PENDING
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """True if this node has no dependents (can be checked after graph is built)."""
        return not self.output_artifacts and not self.output_knowledge


@dataclass
class DAGEdge:
    """A directed edge in the task DAG."""
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DATA_DEP
    weight: float = 1.0
    description: str = ""


@dataclass
class TaskDAG:
    """A directed acyclic graph of coding tasks.

    Provides methods for building, validating, and querying the task graph.
    """

    nodes: dict[str, TaskNode] = field(default_factory=dict)
    edges: list[DAGEdge] = field(default_factory=list)
    name: str = ""
    description: str = ""

    # Derived adjacency (computed lazily)
    _adj_out: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _adj_in: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _built: bool = field(default=False, repr=False)

    def add_node(self, node: TaskNode) -> TaskDAG:
        """Add a task node to the DAG."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in DAG")
        self.nodes[node.node_id] = node
        self._built = False
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DATA_DEP,
        weight: float = 1.0,
        description: str = "",
    ) -> TaskDAG:
        """Add a directed edge to the DAG."""
        if source_id not in self.nodes:
            raise ValueError(f"Source node '{source_id}' not in DAG")
        if target_id not in self.nodes:
            raise ValueError(f"Target node '{target_id}' not in DAG")
        self.edges.append(DAGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            description=description,
        ))
        self._built = False
        return self

    def build(self) -> TaskDAG:
        """Build adjacency lists and validate DAG structure (no cycles)."""
        self._adj_out = {nid: [] for nid in self.nodes}
        self._adj_in = {nid: [] for nid in self.nodes}

        for edge in self.edges:
            self._adj_out[edge.source_id].append(edge.target_id)
            self._adj_in[edge.target_id].append(edge.source_id)

        # Merge node.input_dependencies into adjacency
        for nid, node in self.nodes.items():
            for dep_id in node.input_dependencies:
                if dep_id in self.nodes:
                    if nid not in self._adj_out.setdefault(dep_id, []):
                        self._adj_out[dep_id].append(nid)
                    if dep_id not in self._adj_in.setdefault(nid, []):
                        self._adj_in[nid].append(dep_id)

        # Check for cycles via topological sort validation
        try:
            from openharness.dag_native.algorithms import topological_sort
            topological_sort(self)
        except ValueError as e:
            raise ValueError(f"TaskDAG contains a cycle: {e}")

        self._built = True
        return self

    def get_ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor node IDs for a given node."""
        if not self._built:
            self.build()
        ancestors: set[str] = set()
        stack = list(self._adj_in.get(node_id, []))
        while stack:
            parent = stack.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                stack.extend(self._adj_in.get(parent, []))
        return ancestors

    def get_descendants(self, node_id: str) -> set[str]:
        """Return all descendant node IDs for a given node."""
        if not self._built:
            self.build()
        descendants: set[str] = set()
        stack = list(self._adj_out.get(node_id, []))
        while stack:
            child = stack.pop()
            if child not in descendants:
                descendants.add(child)
                stack.extend(self._adj_out.get(child, []))
        return descendants

    def get_path_to(self, node_id: str) -> list[list[str]]:
        """Return all paths from root nodes to the given node."""
        if not self._built:
            self.build()
        roots = self.get_root_nodes()
        all_paths: list[list[str]] = []

        def dfs(current: str, path: list[str]) -> None:
            path = path + [current]
            if current == node_id:
                all_paths.append(path)
                return
            for child in self._adj_out.get(current, []):
                if child not in path:  # should not happen in DAG, but safety
                    dfs(child, path)

        for root in roots:
            dfs(root, [])
        return all_paths

    def get_root_nodes(self) -> list[str]:
        """Return nodes with no incoming edges."""
        if not self._built:
            self.build()
        return [nid for nid, in_edges in self._adj_in.items() if not in_edges]

    def get_leaf_nodes(self) -> list[str]:
        """Return nodes with no outgoing edges."""
        if not self._built:
            self.build()
        return [nid for nid, out_edges in self._adj_out.items() if not out_edges]

    def get_ancestor_set_signature(self, node_id: str) -> str:
        """Generate a stable signature for the ancestor set of a node.

        Used by the shared context manager to identify nodes with identical
        prerequisite knowledge requirements.
        """
        ancestors = sorted(self.get_ancestors(node_id))
        data = json.dumps(ancestors, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_path_signature(self, node_id: str) -> str:
        """Generate a stable signature from all paths leading to a node.

        Two nodes with the same path_signature share the same DAG prefix
        and can reuse the same shared prerequisite knowledge.
        """
        paths = self.get_path_to(node_id)
        normalized = sorted([tuple(p) for p in paths])
        data = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_levels(self) -> list[list[str]]:
        """Return nodes grouped by topological level (BFS layering)."""
        if not self._built:
            self.build()
        in_degree = {nid: len(self._adj_in.get(nid, [])) for nid in self.nodes}
        levels: list[list[str]] = []
        queue = [nid for nid, deg in in_degree.items() if deg == 0]

        while queue:
            levels.append(sorted(queue))
            next_queue = []
            for nid in queue:
                for child in self._adj_out.get(nid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        return levels

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "name": self.name,
            "description": self.description,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "name": n.name,
                    "description": n.description,
                    "input_dependencies": n.input_dependencies,
                    "assigned_subagent": n.assigned_subagent,
                    "estimated_complexity": n.estimated_complexity,
                    "estimated_duration": n.estimated_duration,
                    "status": n.status.value,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {"source": e.source_id, "target": e.target_id, "type": e.edge_type.value}
                for e in self.edges
            ],
        }
