"""Unit tests for DAG algorithms."""

import pytest

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.algorithms import (
    topological_sort,
    reverse_topological_sort,
    topological_levels,
    get_ancestor_closure,
    longest_path_length,
    DAGCycleError,
)


def _make_dag(name: str = "test") -> TaskDAG:
    return TaskDAG(name=name)


def _add_node(dag: TaskDAG, nid: str, duration: float = 1.0, **kwargs) -> TaskNode:
    node = TaskNode(node_id=nid, name=nid, estimated_duration=duration, **kwargs)
    dag.add_node(node)
    return node


class TestTopologicalSort:
    def test_linear_chain(self):
        dag = _make_dag()
        for nid in ["A", "B", "C", "D"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "D")
        dag.build()

        order = topological_sort(dag)
        assert order == ["A", "B", "C", "D"]

    def test_diamond(self):
        dag = _make_dag()
        _add_node(dag, "A")
        _add_node(dag, "B")
        _add_node(dag, "C")
        _add_node(dag, "D")
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        order = topological_sort(dag)
        assert order[0] == "A"
        assert order[3] == "D"
        assert set(order[1:3]) == {"B", "C"}

    def test_parallel_roots(self):
        dag = _make_dag()
        _add_node(dag, "A")
        _add_node(dag, "B")
        _add_node(dag, "C")
        dag.add_edge("A", "C")
        dag.add_edge("B", "C")
        dag.build()

        order = topological_sort(dag)
        assert order[2] == "C"
        assert set(order[:2]) == {"A", "B"}

    def test_single_node(self):
        dag = _make_dag()
        _add_node(dag, "A")
        dag.build()
        assert topological_sort(dag) == ["A"]

    def test_cycle_detection_algo(self):
        dag = _make_dag()
        _add_node(dag, "A")
        _add_node(dag, "B")
        _add_node(dag, "C")
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "A")
        # Build without validation first, then run sort directly
        dag._adj_out = {"A": ["B"], "B": ["C"], "C": ["A"]}
        dag._adj_in = {"A": ["C"], "B": ["A"], "C": ["B"]}
        dag._built = True
        with pytest.raises(DAGCycleError):
            topological_sort(dag)

    def test_node_dependency_edges(self):
        dag = _make_dag()
        _add_node(dag, "A")
        _add_node(dag, "B", input_dependencies=["A"])
        dag.build()

        order = topological_sort(dag)
        assert order == ["A", "B"]


class TestReverseTopologicalSort:
    def test_linear_chain(self):
        dag = _make_dag()
        for nid in ["A", "B", "C"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.build()

        rev = reverse_topological_sort(dag)
        assert rev == ["C", "B", "A"]

    def test_diamond(self):
        dag = _make_dag()
        for nid in ["A", "B", "C", "D"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        rev = reverse_topological_sort(dag)
        assert rev[0] == "D"  # leaf first
        assert rev[3] == "A"  # root last


class TestTopologicalLevels:
    def test_linear(self):
        dag = _make_dag()
        for nid in ["A", "B", "C"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.build()

        levels = topological_levels(dag)
        assert len(levels) == 3
        assert levels == [["A"], ["B"], ["C"]]

    def test_diamond(self):
        dag = _make_dag()
        for nid in ["A", "B", "C", "D"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        levels = topological_levels(dag)
        assert len(levels) == 3
        assert len(levels[1]) == 2  # B, C at same level


class TestLongestPath:
    def test_linear(self):
        dag = _make_dag()
        for nid in ["A", "B", "C", "D"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.add_edge("C", "D")
        dag.build()

        assert longest_path_length(dag) == 3

    def test_diamond(self):
        dag = _make_dag()
        for nid in ["A", "B", "C", "D"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        assert longest_path_length(dag) == 2  # A -> B -> D or A -> C -> D


class TestAncestorClosure:
    def test_linear(self):
        dag = _make_dag()
        for nid in ["A", "B", "C"]:
            _add_node(dag, nid)
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        dag.build()

        assert get_ancestor_closure(dag, "C") == {"A", "B"}
        assert get_ancestor_closure(dag, "A") == set()
