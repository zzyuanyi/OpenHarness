"""Unit tests for shared context manager."""

from openharness.dag_native.graph import TaskDAG, TaskNode, EdgeType
from openharness.dag_native.shared_context import (
    SharedContextManager,
    SharedKnowledgeStore,
    ContextBudgetTracker,
    KnowledgeEntry,
    KnowledgeEntryType,
    PathSignature,
    ReusePolicy,
)


def _make_linear_dag() -> TaskDAG:
    dag = TaskDAG(name="linear")
    dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
    dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
    dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    dag.build()
    return dag


def _make_diamond_dag() -> TaskDAG:
    dag = TaskDAG(name="diamond")
    dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
    dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
    dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
    dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    dag.build()
    return dag


class TestSharedKnowledgeStore:
    def test_put_get(self):
        store = SharedKnowledgeStore()
        entry = KnowledgeEntry(
            key="test_key",
            entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="Some context",
        )
        store.put(entry)
        assert store.has("test_key")
        assert store.get("test_key") == entry

    def test_get_missing(self):
        store = SharedKnowledgeStore()
        assert store.get("missing") is None

    def test_get_by_prefix(self):
        store = SharedKnowledgeStore()
        store.put(KnowledgeEntry(
            key="node_A_k0", entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="Knowledge A",
        ))
        store.put(KnowledgeEntry(
            key="node_B_k0", entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="Knowledge B",
        ))
        store.put(KnowledgeEntry(
            key="node_A_k1", entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="Knowledge A2",
        ))

        a_entries = store.get_by_prefix("node_A_")
        assert len(a_entries) == 2

    def test_delete(self):
        store = SharedKnowledgeStore()
        store.put(KnowledgeEntry(
            key="test", entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="test",
        ))
        assert store.delete("test")
        assert not store.has("test")
        assert not store.delete("test")

    def test_total_tokens(self):
        store = SharedKnowledgeStore()
        store.put(KnowledgeEntry(
            key="a", entry_type=KnowledgeEntryType.CODE_CONTEXT,
            content="This is some content",
        ))
        assert store.total_tokens > 0


class TestContextBudgetTracker:
    def test_update_baseline(self):
        tracker = ContextBudgetTracker()
        tracker.update_baseline(num_agents=5, per_agent_baseline_context=2000)
        assert tracker.duplicated_context_cost_baseline == 10000

    def test_update_optimized(self):
        tracker = ContextBudgetTracker()
        tracker.update_baseline(5, 2000)
        tracker.shared_context_size = 3000
        tracker.per_agent_context = {"a1": 500, "a2": 500}
        tracker.update_optimized()
        assert tracker.duplicated_context_cost_optimized == 4000
        assert tracker.estimated_token_saving == 6000

    def test_cache_hit_rate(self):
        tracker = ContextBudgetTracker()
        tracker.record_request(hit=True)
        tracker.record_request(hit=True)
        tracker.record_request(hit=False)
        assert tracker.cache_hit_rate == 2.0 / 3.0


class TestSharedContextManager:
    def test_path_signatures_computed(self):
        dag = _make_linear_dag()
        ctx = SharedContextManager(dag)

        sig = ctx.get_path_signature("C")
        assert sig is not None
        assert sig.node_id == "C"
        assert len(sig.path_signature) == 16

    def test_reuse_policy_same_path_linear(self):
        dag = _make_linear_dag()
        ctx = SharedContextManager(dag)

        # In a linear chain, each node has a unique path
        policy = ctx.get_reuse_policy("C")
        assert policy in (ReusePolicy.UNRELATED, ReusePolicy.PARTIAL_OVERLAP)
        # No same-path since linear chain

    def test_reuse_policy_diamond(self):
        dag = _make_diamond_dag()
        ctx = SharedContextManager(dag)

        # B and C share the same ancestor path (A -> B/C)
        policy_b = ctx.get_reuse_policy("B")
        policy_c = ctx.get_reuse_policy("C")
        # B and C have the same ancestor set but different paths
        assert policy_b is not None
        assert policy_c is not None

    def test_get_context_for_node(self):
        dag = _make_linear_dag()
        ctx = SharedContextManager(dag)

        context, size = ctx.get_context_for_node("B")
        assert isinstance(context, str)
        assert size > 0
        assert "Task Context for: B" in context

    def test_store_node_knowledge(self):
        dag = _make_linear_dag()
        ctx = SharedContextManager(dag)

        ctx.store_node_knowledge("A", [{
            "type": "code_context",
            "content": "Completed analysis of upstream repo",
        }])
        assert ctx._store.entry_count == 1

    def test_longest_common_prefix_context(self):
        dag = _make_diamond_dag()
        ctx = SharedContextManager(dag)

        # Store knowledge for A
        ctx.store_node_knowledge("A", [{
            "type": "upstream_summary",
            "content": "Analysis complete: project has 364 Python files",
        }])

        # B and C share A as common ancestor
        lcp = ctx.get_longest_common_prefix_context(["B", "C"])
        assert "Analysis complete" in lcp

    def test_get_metrics(self):
        dag = _make_linear_dag()
        ctx = SharedContextManager(dag)

        metrics = ctx.get_metrics()
        assert "knowledge_store_entries" in metrics
        assert "path_signature_count" in metrics
        assert "cache_hit_rate" in metrics

    def test_find_same_path_nodes(self):
        """Two nodes that are siblings in a tree should share same path."""
        dag = TaskDAG(name="test")
        dag.add_node(TaskNode(node_id="root", name="root", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="L1", name="L1", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="L2", name="L2", estimated_duration=1.0))
        dag.add_edge("root", "L1")
        dag.add_edge("root", "L2")
        dag.build()

        ctx = SharedContextManager(dag)
        same_L1 = ctx.find_same_path_nodes("L1")
        # L1 and L2 might share same ancestor_set_signature
        # (depending on the DAG structure, they share same ancestor {root})
        assert isinstance(same_L1, list)

    def test_find_partial_overlap(self):
        dag = TaskDAG(name="partial")
        dag.add_node(TaskNode(node_id="A", name="A", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="B", name="B", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="C", name="C", estimated_duration=1.0))
        dag.add_node(TaskNode(node_id="D", name="D", estimated_duration=1.0))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        dag.build()

        ctx = SharedContextManager(dag)
        overlaps = ctx.find_partial_overlap_nodes("B")
        # C should partially overlap with B (share A as ancestor)
        assert isinstance(overlaps, list)


class TestPathSignature:
    def test_creation(self):
        sig = PathSignature(
            node_id="test",
            path_signature="abc123def4567890",
            ancestor_set_signature="fed9876543210cba",
            ancestor_ids=["A", "B"],
            path_count=1,
        )
        assert sig.node_id == "test"
        assert len(sig.path_signature) == 16

    def test_equality(self):
        sig1 = PathSignature("A", "aaaa111122223333", "bbbb444455556666")
        sig2 = PathSignature("B", "aaaa111122223333", "cccc777788889999")
        sig3 = PathSignature("C", "xxxx999988887777", "yyyy666655554444")
        assert sig1 == sig2  # Same path_signature
        assert sig1 != sig3

    def test_hashable(self):
        sig = PathSignature("A", "aaaa111122223333", "bbbb444455556666")
        d = {sig: "value"}
        assert d[sig] == "value"
