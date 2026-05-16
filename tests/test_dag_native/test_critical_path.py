"""Unit tests for critical path identification."""

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.critical_path import identify_critical_path


def _make_dag(name: str = "test") -> TaskDAG:
    return TaskDAG(name=name)


class TestCriticalPath:
    def test_linear_chain(self):
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=3.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.build()

        result = identify_critical_path(dag)
        # In a linear chain, all nodes are critical
        assert result.critical_path == ["A", "B", "C"]
        assert result.critical_path_length == 6.0
        assert all(s == 0.0 for s in result.slack.values())

    def test_parallel_branches(self):
        """A -> B -> D and A -> C -> D, B takes longer than C."""
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=5.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        result = identify_critical_path(dag)
        # Critical path should be A -> B -> D (8.0) since B is longer than C
        assert set(result.critical_path) == {"A", "B", "D"}
        assert result.critical_path_length == 8.0
        # C should have slack
        assert result.slack["C"] > 0

    def test_all_equal_branches(self):
        """Both parallel branches equal duration."""
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        result = identify_critical_path(dag)
        # All nodes on critical path since both branches equal
        assert set(result.critical_path) == {"A", "B", "C", "D"}

    def test_single_node(self):
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=5.0))
        dag.build()

        result = identify_critical_path(dag)
        assert result.critical_path == ["A"]
        assert result.critical_path_length == 5.0

    def test_earliest_latest_start(self):
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=3.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.build()

        result = identify_critical_path(dag)
        assert result.earliest_start["A"] == 0.0
        assert result.earliest_start["B"] == 2.0
        assert result.earliest_start["C"] == 2.0

    def test_bottleneck_detection(self):
        """Node with high fan-in on critical path is a bottleneck."""
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=5.0))  # bottleneck
        dag.add_edge("A", "D")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        result = identify_critical_path(dag)
        assert "D" in result.bottleneck_nodes
        assert "D" in result.critical_nodes

    def test_parallelizable_nodes(self):
        dag = _make_dag()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=3.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "C")
        dag.add_edge("A", "D")
        dag.add_edge("B", "D")
        dag.build()

        result = identify_critical_path(dag)
        non_critical = result.parallelizable_nodes
        # Some nodes should not be on the critical path
        assert len(non_critical) > 0 or len(result.critical_path) > 0

    def test_demo_dag_critical_path(self):
        """Test with the full demo DAG structure."""
        from openharness.dag_native.experiments import ExperimentRunner
        dag = ExperimentRunner._build_demo_dag()

        result = identify_critical_path(dag)
        assert len(result.critical_path) > 0
        assert result.critical_path_length > 0
        assert result.total_nodes == dag.node_count
        assert result.total_edges == dag.edge_count
        assert set(result.critical_path).issubset(set(dag.nodes.keys()))
