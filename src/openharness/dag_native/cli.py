"""CLI commands for DAG-Native operations.

Provides typer-based CLI subcommands integrated with the OpenHarness CLI.
Windows-friendly: uses pathlib, avoids Unix-specific constructs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.algorithms import topological_sort, reverse_topological_sort, topological_levels
from openharness.dag_native.critical_path import identify_critical_path
from openharness.dag_native.planner import DAGPlanner, PlannerConfig
from openharness.dag_native.coordinator import SubagentCoordinator, CoordinationMode
from openharness.dag_native.shared_context import SharedContextManager, SharedKnowledgeStore
from openharness.dag_native.experiments import ExperimentRunner, ExperimentConfig
from openharness.dag_native.metrics import MetricsLogger
from openharness.dag_native.trace import TraceRecorder
from openharness.dag_native.visualize import (
    dag_to_text,
    critical_path_to_text,
    execution_timeline_to_text,
    save_dag_visualization,
)

dag_app = typer.Typer(help="DAG-Native coding task planning and execution.")


@dag_app.command("plan")
def dag_plan(
    task: Annotated[str, typer.Option("--task", "-t", help="Task description to plan")] = "",
    task_file: Annotated[
        Optional[Path],
        typer.Option("--task-file", "-f", help="Read task description from file"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for plan artifacts"),
    ] = None,
    max_parallel: Annotated[
        int, typer.Option("--max-parallel", "-p", help="Max parallel subagents")
    ] = 4,
):
    """Parse a coding task and generate a DAG-Native execution plan."""
    # Read task
    if task_file:
        task_description = Path(task_file).read_text(encoding="utf-8")
    elif task:
        task_description = task
    else:
        typer.echo("Error: Provide --task or --task-file", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Planning task: {task_description[:80]}...")

    # Configure planner
    config = PlannerConfig(
        max_parallel_agents=max_parallel,
        enable_critical_path_priority=True,
        enable_shared_context=True,
    )
    planner = DAGPlanner(config)
    plan = planner.plan(task_description)

    # Display results
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Plan generated: {plan.task_name}")
    typer.echo(f"  Nodes: {plan.dag.node_count}")
    typer.echo(f"  Edges: {plan.dag.edge_count}")
    typer.echo(f"  Topological order: {' → '.join(plan.topological_order)}")
    typer.echo(f"  Reverse topo order: {' → '.join(plan.reverse_topological_order)}")
    typer.echo(f"  Levels: {len(plan.levels)}")
    typer.echo(f"  Critical path: {' → '.join(plan.critical_path_result.critical_path)}")
    typer.echo(f"  Critical path length: {plan.critical_path_result.critical_path_length:.1f}")
    typer.echo(f"  Planning time: {plan.planning_time:.3f}s")
    typer.echo(f"{'='*60}")

    # Save if requested
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        dag_json = output / "dag.json"
        dag_json.write_text(
            json.dumps(plan.dag.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.echo(f"\nDAG saved to: {dag_json}")

        # Visualizations
        save_dag_visualization(plan.dag, output, plan.critical_path_result.critical_path)
        typer.echo(f"Visualizations saved to: {output}")


@dag_app.command("analyze")
def dag_analyze(
    dag_file: Annotated[
        Path,
        typer.Argument(help="Path to a DAG JSON file produced by 'dag plan'"),
    ],
):
    """Analyze an existing DAG plan file."""
    if not dag_file.exists():
        typer.echo(f"Error: File not found: {dag_file}", err=True)
        raise typer.Exit(code=1)

    data = json.loads(dag_file.read_text(encoding="utf-8"))

    # Reconstruct DAG
    dag = TaskDAG(name=data.get("name", "loaded_dag"))
    for nid, ndata in data.get("nodes", {}).items():
        node = TaskNode(
            node_id=nid,
            name=ndata.get("name", nid),
            description=ndata.get("description", ""),
            estimated_complexity=float(ndata.get("estimated_complexity", 1.0)),
            estimated_duration=float(ndata.get("estimated_duration", 1.0)),
        )
        dag.add_node(node)
    for edge_data in data.get("edges", []):
        dag.add_edge(
            edge_data["source"],
            edge_data["target"],
            EdgeType(edge_data.get("type", "data_dep")),
        )
    dag.build()

    # Analyze
    topo = topological_sort(dag)
    rev_topo = reverse_topological_sort(dag)
    levels = topological_levels(dag)
    cp = identify_critical_path(dag)

    typer.echo(dag_to_text(dag, cp.critical_path))
    typer.echo()
    typer.echo(critical_path_to_text(cp))
    typer.echo()
    typer.echo(execution_timeline_to_text(topo, cp.critical_path, levels))


@dag_app.command("experiment")
def dag_experiment(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for experiment results"),
    ] = Path("experiment_results"),
    runs: Annotated[
        int, typer.Option("--runs", "-n", help="Number of experiment runs")
    ] = 1,
    max_parallel: Annotated[
        int, typer.Option("--max-parallel", "-p", help="Max parallel subagents")
    ] = 4,
):
    """Run a baseline vs DAG-Native comparison experiment."""
    typer.echo("=== DAG-Native Experiment Runner ===")
    typer.echo(f"Output: {output.resolve()}")
    typer.echo(f"Runs: {runs}")
    typer.echo(f"Max parallel: {max_parallel}")
    typer.echo()

    config = ExperimentConfig(
        experiment_name="dag_native_comparison",
        num_runs=runs,
        max_parallel_subagents=max_parallel,
        output_dir=output,
        save_traces=True,
        save_metrics=True,
    )

    runner = ExperimentRunner(config)
    comparison = runner.run_demo_experiment()

    # Display results
    typer.echo("=" * 70)
    typer.echo("EXPERIMENT RESULTS")
    typer.echo("=" * 70)
    typer.echo()

    typer.echo("--- Baseline (Sequential, No Shared Context) ---")
    typer.echo(f"  Makespan: {comparison.baseline.total_makespan:.3f}s")
    typer.echo(f"  Planning overhead: {comparison.baseline.planning_overhead:.3f}s")
    typer.echo(f"  Failure count: {comparison.baseline.coordination_state.failure_count}")
    typer.echo()

    typer.echo("--- Optimized (Critical Path Priority, Shared Context) ---")
    typer.echo(f"  Makespan: {comparison.optimized.total_makespan:.3f}s")
    typer.echo(f"  Planning overhead: {comparison.optimized.planning_overhead:.3f}s")
    typer.echo(f"  Failure count: {comparison.optimized.coordination_state.failure_count}")
    typer.echo()

    typer.echo("--- Comparison ---")
    typer.echo(f"  Makespan reduction: {comparison.makespan_reduction:.3f}s ({comparison.makespan_reduction_pct:.1f}%)")
    typer.echo(f"  Token saving (est): {comparison.token_saving}")
    typer.echo(f"  Token saving %: {comparison.token_saving_pct:.1f}%")
    typer.echo(f"  Context reuse improvement: {comparison.context_reuse_rate_improvement:.1%}")
    typer.echo()

    typer.echo("--- DAG Structure ---")
    typer.echo(dag_to_text(comparison.baseline.plan.dag, comparison.baseline.plan.critical_path_result.critical_path))
    typer.echo()

    typer.echo("--- Critical Path ---")
    typer.echo(critical_path_to_text(comparison.baseline.plan.critical_path_result))
    typer.echo()

    typer.echo("--- Subagent Timeline ---")
    typer.echo(execution_timeline_to_text(
        comparison.baseline.plan.topological_order,
        comparison.baseline.plan.critical_path_result.critical_path,
        comparison.baseline.plan.levels,
        comparison.baseline.plan.subagent_assignments,
    ))

    typer.echo(f"\nResults saved to: {output.resolve()}")


@dag_app.command("demo")
def dag_demo(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory"),
    ] = None,
):
    """Run a quick demo showing all DAG-Native pipeline stages."""
    from openharness.dag_native.experiments import ExperimentRunner

    runner = ExperimentRunner()
    dag = runner._build_demo_dag()

    print("=" * 70)
    print("DAG-NATIVE PIPELINE DEMO")
    print("=" * 70)
    print()

    # Step 1: Topological sort
    topo = topological_sort(dag)
    print(f"1. Topological Sort ({len(topo)} nodes):")
    print(f"   {' → '.join(topo)}")
    print()

    # Step 2: Reverse topological sort
    rev = reverse_topological_sort(dag)
    print(f"2. Reverse Topological Sort:")
    print(f"   {' → '.join(rev)}")
    print()

    # Step 3: Levels
    levels = topological_levels(dag)
    print(f"3. Topological Levels ({len(levels)} levels):")
    for i, level in enumerate(levels):
        parallel = "PARALLEL" if len(level) > 1 else "sequential"
        print(f"   Level {i}: {level} ({parallel})")
    print()

    # Step 4: Critical path
    cp = identify_critical_path(dag)
    print(f"4. Critical Path:")
    print(f"   Path: {' → '.join(cp.critical_path)}")
    print(f"   Length: {cp.critical_path_length:.1f}")
    print(f"   Bottlenecks: {cp.bottleneck_nodes}")
    print()

    # Step 5: Path signatures
    ctx = SharedContextManager(dag)
    print("5. Path Signatures:")
    for nid in ["D", "G", "I", "L", "O", "X"]:
        sig = ctx.get_path_signature(nid)
        if sig:
            policy = ctx.get_reuse_policy(nid)
            print(f"   {nid}: sig={sig.path_signature[:8]}... policy={policy.value}")
    print()

    # Step 6: Shared context demo
    print("6. Shared Context Demo:")
    ctx.store_node_knowledge("A", [{
        "type": "upstream_summary",
        "content": "OpenHarness v0.1.9, Python 3.10+, Windows compatible, subprocess/agent loop/task manager available",
    }])
    ctx.store_node_knowledge("C", [{
        "type": "architecture_decision",
        "content": "Incremental approach: add dag_native/ package under src/openharness/, extend CLI with 'dag' subcommand",
    }])
    ctx.store_node_knowledge("D", [{
        "type": "design_constraint",
        "content": "All path handling uses pathlib; shell calls compatible with PowerShell/cmd; no Docker dependency",
    }])

    lcp = ctx.get_longest_common_prefix_context(["G", "H", "I"])
    print(f"   LCP for G,H,I: {lcp[:200]}...")
    print(f"   Store entries: {ctx._store.entry_count}")
    print()

    # Step 7: Experiment comparison
    print("7. Experiment Comparison:")
    comparison = runner.run_demo_experiment()
    print(f"   Baseline makespan: {comparison.baseline.total_makespan:.3f}s")
    print(f"   Optimized makespan: {comparison.optimized.total_makespan:.3f}s")
    print(f"   Reduction: {comparison.makespan_reduction:.3f}s ({comparison.makespan_reduction_pct:.1f}%)")
    print(f"   Token saving: {comparison.token_saving} (est)")

    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_dag_visualization(dag, output, cp.critical_path)
        # Save metrics
        logger = MetricsLogger(output / "metrics")
        if comparison.baseline.metrics_record:
            logger.log(comparison.baseline.metrics_record)
        if comparison.optimized.metrics_record:
            logger.log(comparison.optimized.metrics_record)
        logger.save()
        print(f"\n   Output saved to: {output}")


if __name__ == "__main__":
    dag_app()
