"""Parse natural language coding tasks into structured TaskNode specifications.

Converts a high-level coding task description into a set of TaskNode definitions
and dependency edges suitable for building a TaskDAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from openharness.dag_native.graph import TaskNode, EdgeType


@dataclass
class ParsedTask:
    """Result of parsing a coding task description."""
    task_name: str
    task_description: str
    nodes: list[TaskNode]
    edges: list[tuple[str, str, str]]  # (source_id, target_id, edge_type)

    # Metadata extracted from parsing
    estimated_total_complexity: float = 0.0
    detected_phases: list[str] = field(default_factory=list)
    recommended_parallel_groups: list[list[str]] = field(default_factory=list)


def parse_coding_task(
    task_description: str,
    task_name: str = "coding_task",
) -> ParsedTask:
    """Parse a natural language coding task into structured TaskNode specs.

    This is a heuristic parser. In production, this would be backed by an LLM call.
    The parser recognizes:
    - Numbered/comma-separated sub-tasks
    - Phase markers ("Phase 0:", "Step 1:", etc.)
    - Dependency keywords ("depends on", "after", "requires", "prerequisite")
    - Parallel hints ("in parallel", "concurrently", "independently")

    Args:
        task_description: Natural language description of the coding task.
        task_name: Name for the overall task.

    Returns:
        ParsedTask with nodes, edges, and metadata.
    """

    # Split into phases / steps
    phases = _extract_phases(task_description)

    nodes: list[TaskNode] = []
    edges: list[tuple[str, str, str]] = []
    detected_phases: list[str] = []
    node_counter = 0

    if not phases:
        # Treat the entire description as one node
        node = TaskNode(
            node_id=f"{task_name}_0",
            name=task_name,
            description=task_description.strip()[:500],
        )
        nodes.append(node)
        return ParsedTask(
            task_name=task_name,
            task_description=task_description,
            nodes=nodes,
            edges=[],
            estimated_total_complexity=1.0,
            detected_phases=["single_task"],
        )

    prev_node_id: str | None = None

    for phase in phases:
        phase_name = phase["header"] or f"phase_{node_counter}"
        phase_body = phase["body"]
        detected_phases.append(phase_name)

        # Check for explicit dependency declarations
        explicit_deps = _extract_dependencies(phase_body)
        parallel_hint = _detect_parallel_hint(phase_body)

        # Determine sub-tasks within this phase
        sub_tasks = _split_subtasks(phase_body)

        for i, sub_task in enumerate(sub_tasks):
            node_id = f"{task_name}_{node_counter}"
            node_counter += 1

            # Resolve dependencies
            dependencies: list[str] = []

            # Implicit: depends on previous node in sequence
            if prev_node_id is not None and not parallel_hint and i == 0:
                dependencies.append(prev_node_id)

            # Explicit dependencies from text
            for dep_phase_name in explicit_deps:
                # Find matching node
                for existing_node in nodes:
                    if dep_phase_name.lower() in existing_node.name.lower():
                        dependencies.append(existing_node.node_id)

            complexity = _estimate_complexity(sub_task)

            node = TaskNode(
                node_id=node_id,
                name=f"{phase_name}_{i}" if len(sub_tasks) > 1 else phase_name,
                description=sub_task[:300],
                input_dependencies=dependencies,
                can_parallelize=bool(parallel_hint),
                estimated_complexity=complexity,
                estimated_duration=complexity,
                required_context_size=_estimate_context_size(sub_task),
                produces_reusable_knowledge=_detect_reusable_knowledge(sub_task),
            )
            nodes.append(node)

            # Add edges for explicit dependencies
            for dep_id in dependencies:
                edges.append((dep_id, node_id, EdgeType.DATA_DEP.value))

            if not parallel_hint:
                prev_node_id = node_id

        # Reset between phases if no cross-phase dependency declared
        if not explicit_deps and phases.index(phase) > 0 and prev_node_id:
            # Implicit phase ordering: last node of previous phase -> first node of this phase
            pass  # Already handled via prev_node_id

    # Calculate total complexity
    total_complexity = sum(n.estimated_complexity for n in nodes)

    # Identify parallel groups (nodes at same phase with no interdependencies)
    parallel_groups: list[list[str]] = []
    current_group: list[str] = []
    for node in nodes:
        if node.can_parallelize:
            current_group.append(node.node_id)
        else:
            if current_group:
                parallel_groups.append(current_group)
            current_group = []
    if current_group:
        parallel_groups.append(current_group)

    return ParsedTask(
        task_name=task_name,
        task_description=task_description,
        nodes=nodes,
        edges=edges,
        estimated_total_complexity=total_complexity,
        detected_phases=detected_phases,
        recommended_parallel_groups=parallel_groups,
    )


def _extract_phases(text: str) -> list[dict[str, str]]:
    """Split text into numbered/headed phases."""
    # Match patterns like "Phase 0:", "Step 1:", "# 1.", "### Phase A:"
    phase_pattern = re.compile(
        r'(?:^|\n)(?:#{1,3}\s*)?(?:Phase|Step|阶段|ステップ)\s*[\dA-Za-z]+[：:.]?\s*\n?',
        re.IGNORECASE,
    )

    splits = phase_pattern.split(text)
    headers = phase_pattern.findall(text)

    if not headers:
        return []

    phases = []
    for i, header in enumerate(headers):
        body = splits[i + 1] if i + 1 < len(splits) else ""
        phases.append({"header": header.strip().rstrip(":").rstrip("："), "body": body.strip()})
    return phases


def _extract_dependencies(text: str) -> list[str]:
    """Extract explicitly declared dependency names from text."""
    dep_patterns = [
        r'(?:depends?\s+on|after\s+|requires?\s+|prerequisite[：:]?\s+|→)\s*([A-Za-z][\w\s]*)',
    ]
    deps = []
    for pattern in dep_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        deps.extend([m.strip() for m in matches if len(m.strip()) > 2])
    return deps


def _detect_parallel_hint(text: str) -> bool:
    """Check if text suggests parallel execution."""
    parallel_keywords = [
        "in parallel", "parallel", "concurrently", "concurrent",
        "independently", "simultaneously", "并行", "同时",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in parallel_keywords)


def _split_subtasks(text: str) -> list[str]:
    """Split a phase body into sub-tasks based on list markers."""
    # Try bullet/list splitting
    lines = text.strip().split("\n")
    subtasks: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Check if this is a new list item
        is_list_item = bool(re.match(r'^[-*+]\s|^\d+[.)]\s', stripped))
        if is_list_item and current:
            subtasks.append(" ".join(current))
            current = [stripped]
        elif is_list_item:
            current = [stripped]
        elif current:
            current.append(stripped)

    if current:
        subtasks.append(" ".join(current))

    if not subtasks:
        # No list items found, use sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Merge short sentences
        merged: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if merged and len(s) < 30:
                merged[-1] += " " + s
            else:
                merged.append(s)
        return merged if merged else [text]

    return subtasks


def _estimate_complexity(text: str) -> float:
    """Heuristic complexity estimation based on text features."""
    score = 1.0
    # Longer descriptions tend to be more complex
    score += min(len(text) / 200.0, 3.0)
    # Keywords suggesting complexity
    complex_keywords = [
        "implement", "refactor", "migrate", "optimize", "design",
        "architecture", "algorithm", "database", "security", "test",
    ]
    text_lower = text.lower()
    score += sum(0.5 for kw in complex_keywords if kw in text_lower)
    return min(score, 10.0)


def _estimate_context_size(text: str) -> int:
    """Estimate token context needed for this task."""
    # Rough estimate: ~4 chars per token
    token_estimate = len(text) // 4
    return max(token_estimate, 100)


def _detect_reusable_knowledge(text: str) -> bool:
    """Check if this task likely produces reusable knowledge."""
    reusable_keywords = [
        "document", "design", "architecture", "interface", "api",
        "schema", "spec", "plan", "pattern", "summary", "analysis",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in reusable_keywords)
