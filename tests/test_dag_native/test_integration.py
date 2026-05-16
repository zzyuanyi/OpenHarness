"""Integration tests for the full DAG-Native pipeline."""

import json
from pathlib import Path

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.algorithms import topological_sort, reverse_topological_sort, topological_levels
from openharness.dag_native.critical_path import identify_critical_path
from openharness.dag_native.planner import DAGPlanner, PlannerConfig
from openharness.dag_native.coordinator import SubagentCoordinator, CoordinationMode
from openharness.dag_native.shared_context import SharedContextManager
from openharness.dag_native.experiments import ExperimentRunner, ExperimentConfig
from openharness.dag_native.metrics import MetricsLogger, MetricsRecord
from openharness.dag_native.trace import TraceRecorder, TraceEventType
from openharness.dag_native.visualize import (
    dag_to_text,
    critical_path_to_text,
    execution_timeline_to_text,
)


def _make_test_dag() -> TaskDAG:
    """Create a small test DAG for integration tests."""
    dag = TaskDAG(name="integration_test")
    dag.add_node(TaskNode(node_id="design", name="Design", estimated_duration=2.0))
    dag.add_node(TaskNode(node_id="backend", name="Backend", estimated_duration=4.0))
    dag.add_node(TaskNode(node_id="frontend", name="Frontend", estimated_duration=3.0))
    dag.add_node(TaskNode(node_id="integration", name="Integration", estimated_duration=2.0))
    dag.add_node(TaskNode(node_id="testing", name="Testing", estimated_duration=2.0))
    dag.add_edge("design", "backend")
    dag.add_edge("design", "frontend")
    dag.add_edge("backend", "integration")
    dag.add_edge("frontend", "integration")
    dag.add_edge("integration", "testing")
    dag.build()
    return dag


class TestPipelineIntegration:
    """Test the full DAG-native pipeline end-to-end."""

    def test_full_pipeline_parsing_to_execution(self):
        """Full pipeline: parse → plan → execute → verify."""
        task_desc = """
        Phase 1: Design the system architecture.
        Phase 2: Implement the backend API (depends on Phase 1).
        Phase 3: Implement the frontend UI (depends on Phase 1).
        Phase 4: Integration testing (depends on Phase 2 and Phase 3).
        """

        planner = DAGPlanner(PlannerConfig(max_parallel_agents=2))
        plan = planner.plan(task_desc, "test_system")

        # Verify plan structure
        assert plan.dag.node_count > 0
        assert len(plan.topological_order) == plan.dag.node_count
        assert plan.critical_path_result.total_nodes == plan.dag.node_count

        # Execute sequentially (baseline)
        coordinator_seq = SubagentCoordinator(
            plan=plan,
            mode=CoordinationMode.SEQUENTIAL,
        )
        state_seq = coordinator_seq.execute_all_sequential()

        assert state_seq.is_complete
        assert state_seq.failure_count == 0

        # Execute with critical path priority
        coordinator_opt = SubagentCoordinator(
            plan=plan,
            mode=CoordinationMode.CRITICAL_PATH_FIRST,
            max_parallel=2,
        )
        state_opt = coordinator_opt.execute_all_parallel()

        assert state_opt.is_complete
        assert state_opt.failure_count == 0

    def test_plan_from_nodes_api(self):
        """Test the plan_from_nodes API."""
        dag = _make_test_dag()

        planner = DAGPlanner()
        plan = planner.plan_from_nodes(
            nodes=list(dag.nodes.values()),
            edges=[
                (e.source_id, e.target_id, e.edge_type.value)
                for e in dag.edges
            ],
            task_name="test_api",
        )

        assert plan.task_name == "test_api"
        assert plan.dag.node_count == 5
        assert len(plan.topological_order) == 5
        assert "design" == plan.topological_order[0]

    def test_coordination_summary(self):
        """Test that coordination summary includes expected fields."""
        dag = _make_test_dag()
        planner = DAGPlanner()
        plan = planner.plan_from_nodes(
            nodes=list(dag.nodes.values()),
            edges=[(e.source_id, e.target_id, e.edge_type.value) for e in dag.edges],
            task_name="summary_test",
        )

        coordinator = SubagentCoordinator(plan=plan, mode=CoordinationMode.SEQUENTIAL)
        state = coordinator.execute_all_sequential()
        summary = coordinator.get_execution_summary(state)

        assert summary["total_nodes"] == 5
        assert summary["completed"] == 5
        assert summary["failed"] == 0
        assert "context_metrics" in summary


class TestExperimentIntegration:
    """Test the experiment runner end-to-end."""

    def test_demo_experiment_runs(self):
        """The demo experiment should complete without errors."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            num_runs=1,
            max_parallel_subagents=2,
            output_dir=Path("experiment_results"),
            save_metrics=False,
        )
        runner = ExperimentRunner(config)
        comparison = runner.run_demo_experiment()

        assert comparison.experiment_name == "test_experiment"
        assert comparison.baseline is not None
        assert comparison.optimized is not None
        # Optimized should generally be faster or equal
        assert comparison.optimized.total_makespan >= 0


class TestMetricsIntegration:
    """Test metrics logging."""

    def test_metrics_record_and_compare(self):
        logger = MetricsLogger()

        # Baseline record
        baseline = MetricsRecord(
            experiment_id="test_baseline",
            is_baseline=True,
            total_makespan=10.0,
            duplicated_context_cost_baseline=5000,
            duplicated_context_cost_optimized=5000,
            num_dag_nodes=5,
            num_dag_edges=4,
            failure_count=0,
        )
        logger.log(baseline)

        # Optimized record
        optimized = MetricsRecord(
            experiment_id="test_optimized",
            is_baseline=False,
            total_makespan=6.0,
            duplicated_context_cost_baseline=5000,
            duplicated_context_cost_optimized=2000,
            num_dag_nodes=5,
            num_dag_edges=4,
            failure_count=0,
            context_reuse_rate=0.8,
            cache_hit_rate=0.75,
            estimated_token_saving=3000,
        )
        logger.log(optimized)

        comparison = logger.get_comparison()
        assert comparison is not None
        assert comparison["makespan_reduction"] == 4.0
        assert comparison["token_saving"] == 3000


class TestTraceIntegration:
    """Test trace recording."""

    def test_trace_recording_pipeline(self):
        trace = TraceRecorder()

        trace.record(TraceEventType.PLAN_GENERATED, node_id="", data={"nodes": 5})
        trace.record(TraceEventType.NODE_STATUS_CHANGE, node_id="A", data={"from": "pending", "to": "running"})
        trace.record(TraceEventType.SUBAGENT_DISPATCHED, node_id="A", agent_id="general-purpose")
        trace.record(TraceEventType.NODE_STATUS_CHANGE, node_id="A", data={"from": "running", "to": "completed"})
        trace.record(TraceEventType.CONTEXT_REUSED, node_id="B", data={"policy": "same_path"})

        assert trace.event_count == 5
        timeline = trace.get_timeline()
        assert len(timeline) == 5

        # Check node events
        node_a_events = trace.get_events_by_node("A")
        assert len(node_a_events) == 3


class TestVisualizeIntegration:
    """Test visualization outputs."""

    def test_text_outputs(self):
        dag = _make_test_dag()
        cp = identify_critical_path(dag)

        text = dag_to_text(dag, cp.critical_path)
        assert "integration_test" in text
        assert "Nodes:" in text

        cp_text = critical_path_to_text(cp)
        assert "Critical Path Analysis" in cp_text
        assert "Critical path:" in cp_text

        topo = topological_sort(dag)
        levels = topological_levels(dag)
        timeline = execution_timeline_to_text(topo, cp.critical_path, levels)
        assert "Execution Timeline" in timeline

    def test_dag_to_dict(self):
        dag = _make_test_dag()
        d = dag.to_dict()
        assert d["node_count"] == 5
        assert len(d["edges"]) == 5
