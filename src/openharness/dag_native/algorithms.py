"""DAG algorithms: topological sort, reverse topological sort, and helpers.

All algorithms are self-contained implementations operating on the TaskDAG
data model. No external graph library required.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openharness.dag_native.graph import TaskDAG


class DAGCycleError(ValueError):
    """Raised when a cycle is detected in a task DAG."""


def topological_sort(dag: "TaskDAG") -> list[str]:
    """Kahn's algorithm — return a valid topological ordering of node IDs.

    Returns:
        List of node_ids in topological order (dependencies before dependents).

    Raises:
        DAGCycleError: If the graph contains a cycle.
    """
    # Build in-degree and adjacency
    in_degree: dict[str, int] = {nid: 0 for nid in dag.nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}

    for edge in dag.edges:
        adj[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] += 1

    # Also account for edges declared via node.input_dependencies
    for nid, node in dag.nodes.items():
        for dep_id in node.input_dependencies:
            if dep_id in dag.nodes and nid not in adj.get(dep_id, []):
                adj.setdefault(dep_id, []).append(nid)
                in_degree[nid] += 1

    # Initialize queue with zero in-degree nodes
    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbor in adj.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(dag.nodes):
        unprocessed = set(dag.nodes) - set(result)
        raise DAGCycleError(
            f"Cycle detected involving nodes: {unprocessed}"
        )

    return result


def reverse_topological_sort(dag: "TaskDAG") -> list[str]:
    """Return nodes in reverse topological order (dependents before dependencies).

    Useful for:
    - Identifying which nodes must be completed before a specific target
    - Backward dependency analysis
    - Verifying that all prerequisite infrastructure exists before dependent work
    """
    order = topological_sort(dag)
    return list(reversed(order))


def topological_levels(dag: "TaskDAG") -> list[list[str]]:
    """Group nodes into levels where each level contains nodes that can run in parallel.

    Level 0 = root nodes (no dependencies).
    Level n = nodes whose max dependency depth is n.
    """
    order = topological_sort(dag)

    # Compute the longest path from any root to each node
    depths: dict[str, int] = {}

    # Reverse adjacency for computing depths
    rev_adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        rev_adj[edge.target_id].append(edge.source_id)
    for nid, node in dag.nodes.items():
        for dep_id in node.input_dependencies:
            if dep_id in dag.nodes and dep_id not in rev_adj.get(nid, []):
                rev_adj[nid].append(dep_id)

    # Process in topological order to compute depths
    for nid in order:
        parents = rev_adj.get(nid, [])
        if not parents:
            depths[nid] = 0
        else:
            depths[nid] = 1 + max(depths.get(p, 0) for p in parents)

    # Group by depth
    max_depth = max(depths.values()) if depths else 0
    levels: list[list[str]] = [[] for _ in range(max_depth + 1)]
    for nid, depth in depths.items():
        levels[depth].append(nid)

    return [level for level in levels if level]


def get_ancestor_closure(dag: "TaskDAG", node_id: str) -> set[str]:
    """Return the transitive closure of ancestors for a node."""
    result: set[str] = set()
    stack: list[str] = [node_id]
    while stack:
        current = stack.pop()
        # Add direct dependencies
        node = dag.nodes.get(current)
        if node:
            for dep in node.input_dependencies:
                if dep not in result and dep in dag.nodes:
                    result.add(dep)
                    stack.append(dep)
        # Add edge-based dependencies
        for edge in dag.edges:
            if edge.target_id == current and edge.source_id not in result:
                if edge.source_id in dag.nodes:
                    result.add(edge.source_id)
                    stack.append(edge.source_id)
    return result


def longest_path_length(dag: "TaskDAG") -> int:
    """Return the length (in edges) of the longest path through the DAG."""
    order = topological_sort(dag)
    dist: dict[str, int] = {nid: 0 for nid in dag.nodes}

    for nid in order:
        node = dag.nodes.get(nid)
        if node:
            for dep in node.input_dependencies:
                if dep in dist:
                    dist[nid] = max(dist[nid], dist[dep] + 1)
        for edge in dag.edges:
            if edge.target_id == nid and edge.source_id in dist:
                dist[nid] = max(dist[nid], dist[edge.source_id] + 1)

    return max(dist.values()) if dist else 0
