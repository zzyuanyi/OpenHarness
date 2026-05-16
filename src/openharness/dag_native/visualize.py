"""DAG visualization utilities.

Produces text-based and optional matplotlib visualizations of task DAGs,
critical paths, topological levels, and subagent timelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openharness.dag_native.graph import TaskDAG
    from openharness.dag_native.critical_path import CriticalPathResult


def dag_to_text(dag: "TaskDAG", highlight_path: list[str] | None = None) -> str:
    """Render a TaskDAG as an ASCII representation.

    Args:
        dag: The task DAG to render.
        highlight_path: Optional node IDs to highlight (e.g., critical path).

    Returns:
        A multi-line string representation.
    """
    highlight = set(highlight_path or [])
    lines: list[str] = []
    lines.append(f"=== Task DAG: {dag.name} ===")
    lines.append(f"Nodes: {dag.node_count}, Edges: {dag.edge_count}")
    lines.append("")

    # Show levels
    levels = dag.get_levels()
    for i, level in enumerate(levels):
        level_str = ", ".join(
            f"*{nid}*" if nid in highlight else nid
            for nid in level
        )
        lines.append(f"Level {i}: [{level_str}]")

    lines.append("")
    lines.append("Dependencies:")
    for edge in dag.edges:
        lines.append(f"  {edge.source_id} → {edge.target_id} ({edge.edge_type.value})")

    return "\n".join(lines)


def critical_path_to_text(result: "CriticalPathResult") -> str:
    """Render critical path analysis as text."""
    lines: list[str] = []
    lines.append("=== Critical Path Analysis ===")
    lines.append(f"Critical path: {' → '.join(result.critical_path)}")
    lines.append(f"Critical path length: {result.critical_path_length:.1f}")
    lines.append(f"Total nodes: {result.total_nodes}")
    lines.append("")

    lines.append("Node details:")
    lines.append(f"{'Node':<12} {'Duration':>8} {'ES':>6} {'LS':>6} {'Slack':>6} {'Critical'}")
    lines.append("-" * 55)

    # Sort by topological order for display
    for nid in sorted(result.node_durations.keys()):
        dur = result.node_durations.get(nid, 0)
        es = result.earliest_start.get(nid, 0)
        ls = result.latest_start.get(nid, 0)
        sl = result.slack.get(nid, 0)
        is_crit = "YES" if sl == 0 else ""
        lines.append(f"{nid:<12} {dur:>8.1f} {es:>6.1f} {ls:>6.1f} {sl:>6.1f} {is_crit}")

    lines.append("")
    lines.append(f"Bottleneck nodes: {', '.join(result.bottleneck_nodes) if result.bottleneck_nodes else 'none'}")
    lines.append(f"Parallelizable (non-critical): {len(result.parallelizable_nodes)}")

    return "\n".join(lines)


def execution_timeline_to_text(
    topological_order: list[str],
    critical_path: list[str],
    levels: list[list[str]],
    subagent_assignments: dict[str, str] | None = None,
) -> str:
    """Render an execution timeline as text."""
    agents = subagent_assignments or {}
    cp_set = set(critical_path)
    lines: list[str] = []
    lines.append("=== Execution Timeline ===")
    lines.append("")

    for i, nid in enumerate(topological_order):
        marker = "★" if nid in cp_set else " "
        agent = agents.get(nid, "unassigned")
        lines.append(f"  {i+1:2d}. {marker} {nid} [{agent}]")

    # Show parallel windows
    lines.append("")
    lines.append("Parallel execution windows:")
    for i, level in enumerate(levels):
        if len(level) > 1:
            lines.append(f"  Level {i}: {len(level)} tasks in parallel — {level}")

    return "\n".join(lines)


def dag_to_mermaid(dag: "TaskDAG") -> str:
    """Export DAG to Mermaid flowchart syntax for rendering in markdown."""
    lines = ["```mermaid", "graph TD"]
    for nid, node in dag.nodes.items():
        label = node.name.replace("_", " ")
        lines.append(f"    {nid}[{label}]")
    for edge in dag.edges:
        lines.append(f"    {edge.source_id} --> {edge.target_id}")
    lines.append("```")
    return "\n".join(lines)


def dag_to_dot(dag: "TaskDAG", highlight: list[str] | None = None) -> str:
    """Export DAG to Graphviz DOT format."""
    hl = set(highlight or [])
    lines = ["digraph TaskDAG {", "    rankdir=TB;", '    node [shape=box, style=rounded];']
    for nid, node in dag.nodes.items():
        attrs = 'style="filled,rounded", fillcolor=lightyellow' if nid in hl else 'style=rounded'
        label = node.name.replace("_", "\\n")
        lines.append(f'    {nid} [label="{label}", {attrs}];')
    for edge in dag.edges:
        lines.append(f"    {edge.source_id} -> {edge.target_id};")
    lines.append("}")
    return "\n".join(lines)


def save_dag_visualization(
    dag: "TaskDAG",
    output_dir: str | Path,
    critical_path: list[str] | None = None,
) -> dict[str, Path]:
    """Save multiple visualization formats to disk.

    Returns dict mapping format name to output path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    # Text representation
    text_path = output_path / "dag_structure.txt"
    text_path.write_text(dag_to_text(dag, critical_path), encoding="utf-8")
    files["text"] = text_path

    # Dot format
    dot_path = output_path / "dag_structure.dot"
    dot_path.write_text(dag_to_dot(dag, critical_path), encoding="utf-8")
    files["dot"] = dot_path

    # Mermaid
    mermaid_path = output_path / "dag_structure.md"
    mermaid_path.write_text(dag_to_mermaid(dag), encoding="utf-8")
    files["mermaid"] = mermaid_path

    return files


def try_matplotlib_visualization(
    dag: "TaskDAG",
    critical_path: list[str] | None = None,
    output_path: str | Path | None = None,
) -> bool:
    """Attempt to create a matplotlib visualization of the DAG.

    Returns True if matplotlib is available and visualization was created.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    # Simple layered layout
    levels = dag.get_levels()
    cp_set = set(critical_path or [])

    fig, ax = plt.subplots(figsize=(12, 8))
    node_positions: dict[str, tuple[float, float]] = {}

    level_height = 1.0
    node_spacing = 1.2

    for level_idx, level in enumerate(levels):
        y = -level_idx * level_height
        for node_idx, nid in enumerate(level):
            x = (node_idx - (len(level) - 1) / 2) * node_spacing
            node_positions[nid] = (x, y)

    # Draw edges
    for edge in dag.edges:
        if edge.source_id in node_positions and edge.target_id in node_positions:
            sx, sy = node_positions[edge.source_id]
            tx, ty = node_positions[edge.target_id]
            ax.annotate(
                "", xy=(tx, ty), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

    # Draw nodes
    for nid, (x, y) in node_positions.items():
        node = dag.nodes[nid]
        color = "gold" if nid in cp_set else "lightblue"
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.5, y - 0.2), 1.0, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="black",
        ))
        ax.text(x, y, node.name.replace("_", "\n"), ha="center", va="center", fontsize=7)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-len(levels) * level_height, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"DAG: {dag.name}")

    legend_elements = [
        mpatches.Patch(facecolor="gold", label="Critical Path"),
        mpatches.Patch(facecolor="lightblue", label="Non-Critical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
