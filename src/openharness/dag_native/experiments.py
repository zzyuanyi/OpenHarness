"""Experiment runner for comparing Baseline vs DAG-Native execution.

Runs controlled experiments measuring:
- Total makespan
- Planning overhead
- Context reuse rates
- Token savings
- Success rates
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.planner import DAGPlanner, PlannerConfig, ExecutionPlan
from openharness.dag_native.coordinator import (
    SubagentCoordinator,
    CoordinationMode,
    CoordinationState,
    SubagentSpec,
    SubagentResult,
)
from openharness.dag_native.shared_context import (
    SharedContextManager,
    SharedKnowledgeStore,
    ContextBudgetTracker,
)
from openharness.dag_native.context_cache import ContextCache
from openharness.dag_native.metrics import MetricsLogger, MetricsRecord


@dataclass
class ExperimentConfig:
    """Configuration for a comparison experiment."""
    experiment_name: str = "dag_native_experiment"
    num_runs: int = 1
    max_parallel_subagents: int = 4
    # Baseline config
    baseline_mode: CoordinationMode = CoordinationMode.SEQUENTIAL
    baseline_shared_context: bool = False
    # Optimized config
    optimized_mode: CoordinationMode = CoordinationMode.CRITICAL_PATH_FIRST
    optimized_shared_context: bool = True
    # Output
    output_dir: str | Path = "experiment_results"
    save_traces: bool = True
    save_metrics: bool = True


@dataclass
class ExperimentRunResult:
    """Result of a single experiment run."""
    run_id: int
    is_baseline: bool
    plan: ExecutionPlan
    coordination_state: CoordinationState
    context_metrics: dict[str, Any]
    metrics_record: MetricsRecord | None = None
    total_makespan: float = 0.0
    planning_overhead: float = 0.0


@dataclass
class ExperimentComparison:
    """Comparison between baseline and optimized execution."""
    experiment_name: str
    baseline: ExperimentRunResult
    optimized: ExperimentRunResult

    # Computed comparisons
    makespan_reduction: float = 0.0
    makespan_reduction_pct: float = 0.0
    token_saving: int = 0
    token_saving_pct: float = 0.0
    context_reuse_rate_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.baseline.total_makespan > 0:
            self.makespan_reduction = (
                self.baseline.total_makespan - self.optimized.total_makespan
            )
            self.makespan_reduction_pct = (
                self.makespan_reduction / self.baseline.total_makespan * 100
            )

        baseline_tokens = self.baseline.context_metrics.get(
            "duplicated_context_cost_baseline", 0
        )
        optimized_tokens = self.optimized.context_metrics.get(
            "duplicated_context_cost_optimized", 0
        )
        if baseline_tokens > 0:
            self.token_saving = baseline_tokens - optimized_tokens
            self.token_saving_pct = self.token_saving / baseline_tokens * 100

        optimized_reuse = self.optimized.context_metrics.get("cache_hit_rate", 0.0)
        baseline_reuse = self.baseline.context_metrics.get("cache_hit_rate", 0.0)
        self.context_reuse_rate_improvement = optimized_reuse - baseline_reuse


class ExperimentRunner:
    """Runs controlled experiments comparing execution strategies."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config or ExperimentConfig()
        self._metrics_logger = MetricsLogger()
        self._output_dir = Path(self._config.output_dir)

    def run_demo_experiment(self) -> ExperimentComparison:
        """Run a demo experiment with a synthetic coding task.

        Returns:
            ExperimentComparison with baseline vs optimized results.
        """
        # Build demo task
        task_desc = self._demo_task_description()
        dag = self._build_demo_dag()

        planner = DAGPlanner(PlannerConfig(
            max_parallel_agents=self._config.max_parallel_subagents,
        ))
        plan = planner.plan_from_nodes(
            nodes=list(dag.nodes.values()),
            edges=[
                (e.source_id, e.target_id, e.edge_type.value)
                for e in dag.edges
            ],
            task_name="demo_coding_task",
        )

        # ---- Baseline Run ----
        baseline = self._run_single(
            plan=plan,
            run_id=1,
            is_baseline=True,
            mode=self._config.baseline_mode,
            use_shared_context=self._config.baseline_shared_context,
        )

        # ---- Optimized Run ----
        optimized = self._run_single(
            plan=plan,
            run_id=2,
            is_baseline=False,
            mode=self._config.optimized_mode,
            use_shared_context=self._config.optimized_shared_context,
        )

        comparison = ExperimentComparison(
            experiment_name=self._config.experiment_name,
            baseline=baseline,
            optimized=optimized,
        )

        # Save outputs
        if self._config.save_metrics:
            self._save_comparison(comparison)

        return comparison

    def _run_single(
        self,
        plan: ExecutionPlan,
        run_id: int,
        is_baseline: bool,
        mode: CoordinationMode,
        use_shared_context: bool,
    ) -> ExperimentRunResult:
        """Execute a single experiment run."""
        start = time.time()

        # Setup shared context manager
        knowledge_store = SharedKnowledgeStore() if use_shared_context else None
        budget_tracker = ContextBudgetTracker() if use_shared_context else None

        shared_ctx = SharedContextManager(
            dag=plan.dag,
            knowledge_store=knowledge_store,
            budget_tracker=budget_tracker,
        ) if use_shared_context else SharedContextManager(plan.dag)

        # Setup coordinator
        coordinator = SubagentCoordinator(
            plan=plan,
            shared_context=shared_ctx,
            mode=mode,
            max_parallel=self._config.max_parallel_subagents,
        )

        # Execute
        if mode == CoordinationMode.SEQUENTIAL:
            state = coordinator.execute_all_sequential()
        else:
            state = coordinator.execute_all_parallel()

        total_makespan = time.time() - start

        # Collect context metrics
        context_metrics = shared_ctx.get_metrics() if use_shared_context else {
            "cache_hit_rate": 0.0,
            "duplicated_context_cost_baseline": self._estimate_baseline_context_cost(plan),
            "duplicated_context_cost_optimized": 0,
            "estimated_token_saving": 0,
        }

        # Build metrics record
        record = self._build_metrics_record(
            plan=plan,
            state=state,
            is_baseline=is_baseline,
            context_metrics=context_metrics,
            total_makespan=total_makespan,
        )

        if self._config.save_metrics:
            self._metrics_logger.log(record)

        return ExperimentRunResult(
            run_id=run_id,
            is_baseline=is_baseline,
            plan=plan,
            coordination_state=state,
            context_metrics=context_metrics,
            metrics_record=record,
            total_makespan=total_makespan,
            planning_overhead=plan.planning_time,
        )

    def _estimate_baseline_context_cost(self, plan: ExecutionPlan) -> int:
        """Estimate duplicated context cost for baseline (no sharing)."""
        num_agents = plan.dag.node_count
        # Each agent in baseline carries full project context plus its node context
        per_agent = 2000  # estimated tokens per agent
        return num_agents * per_agent

    def _build_metrics_record(
        self,
        plan: ExecutionPlan,
        state: CoordinationState,
        is_baseline: bool,
        context_metrics: dict[str, Any],
        total_makespan: float,
    ) -> MetricsRecord:
        """Build a metrics record from experiment run data."""
        label = "baseline" if is_baseline else "dag_native_optimized"
        return MetricsRecord(
            experiment_id=f"{self._config.experiment_name}_{label}",
            is_baseline=is_baseline,
            total_makespan=total_makespan,
            planning_overhead=plan.planning_time,
            critical_path_length=plan.critical_path_result.critical_path_length,
            num_dag_nodes=plan.dag.node_count,
            num_dag_edges=plan.dag.edge_count,
            num_subagents=plan.dag.node_count,
            max_parallel_subagents=self._config.max_parallel_subagents,
            shared_context_size=context_metrics.get("shared_context_size", 0),
            per_agent_context_size=0,  # Would need per-agent tracking
            duplicated_context_cost_baseline=context_metrics.get(
                "duplicated_context_cost_baseline", 0
            ),
            duplicated_context_cost_optimized=context_metrics.get(
                "duplicated_context_cost_optimized", 0
            ),
            context_reuse_rate=context_metrics.get("context_reuse_rate", 0.0),
            cache_hit_rate=context_metrics.get("cache_hit_rate", 0.0),
            estimated_token_saving=context_metrics.get("estimated_token_saving", 0),
            stale_cache_count=context_metrics.get("stale_cache_count", 0),
            test_pass_rate=1.0 if state.failure_count == 0 else (
                state.success_count / max(state.success_count + state.failure_count, 1)
            ),
            code_generation_success=state.failure_count == 0,
            final_artifact_completeness=1.0 if state.failure_count == 0 else 0.5,
            trace_length=total_makespan,
            failure_count=state.failure_count,
            metadata={
                "mode": "sequential" if is_baseline else "critical_path_first",
                "num_levels": len(plan.levels),
                "topological_order": plan.topological_order,
            },
        )

    def _save_comparison(self, comparison: ExperimentComparison) -> None:
        """Save experiment comparison results to disk."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "experiment_name": comparison.experiment_name,
            "baseline": {
                "total_makespan": comparison.baseline.total_makespan,
                "planning_overhead": comparison.baseline.planning_overhead,
                "context_metrics": comparison.baseline.context_metrics,
                "dag_summary": comparison.baseline.plan.dag.to_dict(),
            },
            "optimized": {
                "total_makespan": comparison.optimized.total_makespan,
                "planning_overhead": comparison.optimized.planning_overhead,
                "context_metrics": comparison.optimized.context_metrics,
                "coordination_summary": {
                    nid: {
                        "node_id": r.node_id,
                        "success": r.success,
                        "duration": r.duration,
                    }
                    for nid, r in comparison.optimized.coordination_state.completed_results.items()
                },
            },
            "comparison": {
                "makespan_reduction": comparison.makespan_reduction,
                "makespan_reduction_pct": comparison.makespan_reduction_pct,
                "token_saving": comparison.token_saving,
                "token_saving_pct": comparison.token_saving_pct,
                "context_reuse_rate_improvement": comparison.context_reuse_rate_improvement,
            },
            "dag": comparison.baseline.plan.dag.to_dict(),
            "critical_path": comparison.baseline.plan.critical_path_result.critical_path,
            "critical_path_length": comparison.baseline.plan.critical_path_result.critical_path_length,
            "topological_order": comparison.baseline.plan.topological_order,
            "levels": comparison.baseline.plan.levels,
        }

        output_path = self._output_dir / f"{comparison.experiment_name}.json"
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _demo_task_description() -> str:
        return """Implement a DAG-native coding harness on top of OpenHarness.

Phase 1: Clone and analyze the upstream repository.
Phase 2: Design the incremental architecture.
Phase 3: Implement core DAG algorithms (topological sort, critical path).
Phase 4: Implement task parser and DAG builder.
Phase 5: Implement subagent coordinator.
Phase 6: Implement shared context manager.
Phase 7: Implement experiment runner and metrics.
Phase 8: Write tests and documentation.
"""

    @staticmethod
    def _build_demo_dag() -> TaskDAG:
        """Build the demo task DAG representing the development process."""
        dag = TaskDAG(name="dag_native_development")

        # Phase 1: Clone and analyze
        dag.add_node(TaskNode(
            node_id="A", name="clone_and_analyze",
            description="Clone OpenHarness and analyze structure",
            estimated_complexity=2.0, estimated_duration=2.0,
            produces_reusable_knowledge=True,
        ))
        dag.add_node(TaskNode(
            node_id="B", name="install_and_test",
            description="Install dependencies and run baseline tests",
            estimated_complexity=1.5, estimated_duration=1.5,
        ))
        dag.add_node(TaskNode(
            node_id="C", name="identify_reusable",
            description="Identify reusable harness abstractions",
            estimated_complexity=2.0, estimated_duration=2.0,
            produces_reusable_knowledge=True,
        ))

        # A → B, A → C (parallel after clone)
        dag.add_edge("A", "B", EdgeType.DATA_DEP, description="Need clone before install")
        dag.add_edge("A", "C", EdgeType.DATA_DEP, description="Need clone before analysis")

        # Phase 2: Design
        dag.add_node(TaskNode(
            node_id="D", name="design_architecture",
            description="Design incremental architecture and interfaces",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["B", "C"],
            produces_reusable_knowledge=True,
        ))

        # Phase 3: Core DAG algorithms
        dag.add_node(TaskNode(
            node_id="E", name="implement_task_parser",
            description="Implement coding task parser",
            estimated_complexity=2.5, estimated_duration=2.5,
            input_dependencies=["D"],
        ))
        dag.add_node(TaskNode(
            node_id="F", name="implement_dag_model",
            description="Implement task DAG data model",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["D"],
        ))

        dag.add_node(TaskNode(
            node_id="G", name="implement_topo_sort",
            description="Implement topological sort algorithm",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["F"],
        ))
        dag.add_node(TaskNode(
            node_id="H", name="implement_reverse_topo",
            description="Implement reverse topological sort",
            estimated_complexity=1.5, estimated_duration=1.5,
            input_dependencies=["F"],
        ))
        dag.add_node(TaskNode(
            node_id="I", name="implement_critical_path",
            description="Implement critical path algorithm",
            estimated_complexity=2.5, estimated_duration=2.5,
            input_dependencies=["F"],
        ))

        # Phase 4: Planner and coordinator
        dag.add_node(TaskNode(
            node_id="J", name="implement_planner",
            description="Implement DAG planner orchestrating pipeline",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["E", "F", "I"],
            can_parallelize=False,
        ))
        dag.add_node(TaskNode(
            node_id="K", name="implement_subagent_adapter",
            description="Implement subagent spec adapter for OpenHarness",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["J"],
        ))
        dag.add_node(TaskNode(
            node_id="L", name="implement_coordinator",
            description="Implement subagent coordinator adapter",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["K"],
        ))

        # Phase 5: Shared context
        dag.add_node(TaskNode(
            node_id="M", name="implement_shared_context",
            description="Implement same-path shared prerequisite knowledge manager",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["J"],
            produces_reusable_knowledge=True,
        ))
        dag.add_node(TaskNode(
            node_id="N", name="implement_context_cache",
            description="Implement shared context cache with TTL",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["M"],
        ))

        # Phase 6: Experiments and metrics
        dag.add_node(TaskNode(
            node_id="O", name="implement_experiment_runner",
            description="Implement experiment runner",
            estimated_complexity=2.5, estimated_duration=2.5,
            input_dependencies=["J", "M", "N"],
        ))
        dag.add_node(TaskNode(
            node_id="P", name="implement_metrics_trace",
            description="Implement metrics logger and trace recorder",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["J"],
        ))

        # Phase 7: CLI
        dag.add_node(TaskNode(
            node_id="Q", name="implement_cli",
            description="Implement Windows-friendly CLI command",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["O", "P"],
        ))

        # Phase 8: Tests and docs
        dag.add_node(TaskNode(
            node_id="R", name="add_demo_task",
            description="Add demo coding task for testing",
            estimated_complexity=1.5, estimated_duration=1.5,
            input_dependencies=["Q"],
        ))
        dag.add_node(TaskNode(
            node_id="S", name="add_unit_tests",
            description="Add unit tests for all modules",
            estimated_complexity=3.0, estimated_duration=3.0,
            input_dependencies=["Q"],
        ))
        dag.add_node(TaskNode(
            node_id="T", name="add_integration_tests",
            description="Add integration tests for full pipeline",
            estimated_complexity=2.5, estimated_duration=2.5,
            input_dependencies=["S"],
        ))
        dag.add_node(TaskNode(
            node_id="U", name="update_docs",
            description="Update README and documentation",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["R", "T"],
        ))

        # Final
        dag.add_node(TaskNode(
            node_id="V", name="run_experiments",
            description="Run baseline vs optimized experiments",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["O", "P", "R", "S"],
        ))
        dag.add_node(TaskNode(
            node_id="W", name="final_consistency_check",
            description="Final consistency check of all modules",
            estimated_complexity=2.0, estimated_duration=2.0,
            input_dependencies=["V", "U"],
        ))
        dag.add_node(TaskNode(
            node_id="X", name="produce_patch_report",
            description="Produce final patch report and risk checklist",
            estimated_complexity=1.5, estimated_duration=1.5,
            input_dependencies=["W"],
        ))

        dag.build()
        return dag
