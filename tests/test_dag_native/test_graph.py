"""Unit tests for the TaskDAG graph model."""

import pytest

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType, NodeStatus


class TestTaskNode:
    def test_create_minimal_node(self):
        node = TaskNode(node_id="test_1", name="Test Node")
        assert node.node_id == "test_1"
        assert node.name == "Test Node"
        assert node.status == NodeStatus.PENDING
        assert node.estimated_complexity == 1.0

    def test_create_node_with_deps(self):
        node = TaskNode(
            node_id="B",
            name="Task B",
            input_dependencies=["A"],
            estimated_duration=3.0,
            produces_reusable_knowledge=True,
        )
        assert node.input_dependencies == ["A"]
        assert node.estimated_duration == 3.0
        assert node.produces_reusable_knowledge


class TestTaskDAG:
    def test_create_empty_dag(self):
        dag = TaskDAG(name="test")
        assert dag.name == "test"
        assert dag.node_count == 0
        assert dag.edge_count == 0

    def test_add_node(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="Node A"))
        assert dag.node_count == 1
        assert "A" in dag.nodes

    def test_add_duplicate_node_raises(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="Node A"))
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node(TaskNode(node_id="A", name="Duplicate"))

    def test_add_edge(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_edge("A", "B", EdgeType.DATA_DEP)
        assert dag.edge_count == 1

    def test_add_edge_missing_node_raises(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A"))
        with pytest.raises(ValueError, match="not in DAG"):
            dag.add_edge("A", "B")

    def test_build_linear_dag(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=2.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.build()
        assert dag._built

    def test_cycle_detection(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "A")
        with pytest.raises(ValueError, match="cycle"):
            dag.build()

    def test_get_ancestors(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        ancestors_D = dag.get_ancestors("D")
        assert ancestors_D == {"A", "B", "C"}

        ancestors_A = dag.get_ancestors("A")
        assert ancestors_A == set()

    def test_get_descendants(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.build()

        assert dag.get_descendants("A") == {"B", "C"}
        assert dag.get_descendants("B") == set()

    def test_get_root_nodes(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "C")
        dag.add_edge("B", "C")
        dag.build()

        roots = dag.get_root_nodes()
        assert set(roots) == {"A", "B"}

    def test_get_leaf_nodes(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.build()

        assert dag.get_leaf_nodes() == ["B"]

    def test_path_signature_stable(self):
        """Path signatures should be stable for same DAG structure."""
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.build()

        sig1 = dag.get_path_signature("B")
        sig2 = dag.get_path_signature("B")
        assert sig1 == sig2

    def test_ancestor_set_signature(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_edge("A", "C")
        dag.add_edge("B", "C")
        dag.build()

        sig = dag.get_ancestor_set_signature("C")
        assert len(sig) == 16  # SHA256 hex digest truncated to 16 chars

    def test_to_dict(self):
        dag = TaskDAG(name="test_dag")
        dag.add_node(TaskNode(node_id="A", name="Node A"))
        dag.add_node(TaskNode(node_id="B", name="Node B"))
        dag.add_edge("A", "B")
        dag.build()

        d = dag.to_dict()
        assert d["name"] == "test_dag"
        assert d["node_count"] == 2
        assert d["edge_count"] == 1
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_levels_linear(self):
        dag = TaskDAG()
        for nid in ["A", "B", "C", "D"]:
            dag.add_node(TaskNode(node_id=nid, name=nid, estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "D")
        dag.build()

        levels = dag.get_levels()
        assert len(levels) == 4
        for i, level in enumerate(levels):
            assert len(level) == 1

    def test_levels_diamond(self):
        dag = TaskDAG()
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        levels = dag.get_levels()
        assert len(levels) == 3  # A, {B,C}, D
        assert len(levels[1]) == 2  # B and C at same level
