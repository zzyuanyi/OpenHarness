"""Same-path shared prerequisite knowledge manager.

When multiple subagents depend on the same DAG prefix path or identical
ancestor task sets, they share structured, compressed prerequisite knowledge
instead of each carrying full context.

Key concepts:
- path_signature: Stable hash of all paths from roots to a node
- ancestor_set_signature: Hash of the ancestor set (for multi-parent DAGs)
- longest_common_prefix_context: Reuse knowledge for partially overlapping paths
- shared_knowledge_store: Central store of prerequisite knowledge entries
- context_budget_tracker: Track global/per-agent context sizes
- reuse_policy: Rules for when to share vs. isolate context
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from openharness.dag_native.graph import TaskDAG, TaskNode


class KnowledgeEntryType(str, Enum):
    """Types of shared knowledge entries."""
    UPSTREAM_SUMMARY = "upstream_summary"
    ARCHITECTURE_DECISION = "architecture_decision"
    DEPENDENCY_SUMMARY = "dependency_summary"
    CODE_CONTEXT = "code_context"
    DESIGN_CONSTRAINT = "design_constraint"
    KNOWN_RISK = "known_risk"
    TEST_RESULT = "test_result"
    REUSABLE_ARTIFACT = "reusable_artifact"


@dataclass
class KnowledgeEntry:
    """A single piece of shared prerequisite knowledge."""
    key: str
    entry_type: KnowledgeEntryType
    content: str
    source_node_id: str = ""
    token_size_estimate: int = 0
    created_at: float = field(default_factory=time.time)
    version: int = 1
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]
        if self.token_size_estimate == 0:
            self.token_size_estimate = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return max(len(text) // 4, 1)

    def is_stale(self, current_hash: str) -> bool:
        """Check if this entry is stale relative to a new content hash."""
        return self.content_hash != current_hash


@dataclass
class PathSignature:
    """A stable signature for the DAG paths leading to a node.

    Two nodes with the same PathSignature share identical DAG prefixes
    and can reuse the same shared prerequisite knowledge.
    """
    node_id: str
    path_signature: str          # hash of all paths from roots to this node
    ancestor_set_signature: str  # hash of ancestor set
    ancestor_ids: list[str] = field(default_factory=list)
    path_count: int = 0

    def __hash__(self) -> int:
        return hash(self.path_signature)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathSignature):
            return False
        return self.path_signature == other.path_signature


class SharedKnowledgeStore:
    """Central store for shared prerequisite knowledge."""

    def __init__(self) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}

    def put(self, entry: KnowledgeEntry) -> None:
        """Store a knowledge entry."""
        self._entries[entry.key] = entry

    def get(self, key: str) -> KnowledgeEntry | None:
        """Retrieve a knowledge entry."""
        return self._entries.get(key)

    def get_by_prefix(self, prefix: str) -> list[KnowledgeEntry]:
        """Get all entries with keys starting with a prefix."""
        return [e for k, e in self._entries.items() if k.startswith(prefix)]

    def has(self, key: str) -> bool:
        """Check if an entry exists."""
        return key in self._entries

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_size_estimate for e in self._entries.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "total_tokens": self.total_tokens,
            "entries": {
                k: {
                    "key": v.key,
                    "type": v.entry_type.value,
                    "source_node": v.source_node_id,
                    "tokens": v.token_size_estimate,
                    "version": v.version,
                }
                for k, v in self._entries.items()
            },
        }


class ContextBudgetTracker:
    """Track context sizes and costs across agents."""

    def __init__(self) -> None:
        self.global_context_size: int = 0
        self.shared_context_size: int = 0
        self.per_agent_context: dict[str, int] = {}
        self.duplicated_context_cost_baseline: int = 0
        self.duplicated_context_cost_optimized: int = 0
        self.estimated_token_saving: int = 0
        self.shared_memory_budget: int = 100_000  # estimated

        # Metrics
        self.context_reuse_rate: float = 0.0
        self.cache_hit_rate: float = 0.0
        self.duplicated_context_cost_reduction: float = 0.0
        self.stale_cache_count: int = 0
        self.total_context_requests: int = 0
        self.total_cache_hits: int = 0

    def record_agent_context(self, agent_id: str, context_size: int) -> None:
        self.per_agent_context[agent_id] = context_size

    def update_baseline(self, num_agents: int, per_agent_baseline_context: int) -> None:
        self.duplicated_context_cost_baseline = num_agents * per_agent_baseline_context

    def update_optimized(self) -> None:
        self.duplicated_context_cost_optimized = (
            self.shared_context_size
            + sum(self.per_agent_context.values())
        )
        saving = self.duplicated_context_cost_baseline - self.duplicated_context_cost_optimized
        self.estimated_token_saving = max(saving, 0)
        if self.duplicated_context_cost_baseline > 0:
            self.duplicated_context_cost_reduction = (
                self.estimated_token_saving / self.duplicated_context_cost_baseline
            )

    def record_request(self, hit: bool) -> None:
        self.total_context_requests += 1
        if hit:
            self.total_cache_hits += 1
        if self.total_context_requests > 0:
            self.cache_hit_rate = self.total_cache_hits / self.total_context_requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_context_size": self.global_context_size,
            "shared_context_size": self.shared_context_size,
            "per_agent_context": dict(self.per_agent_context),
            "duplicated_context_cost_baseline": self.duplicated_context_cost_baseline,
            "duplicated_context_cost_optimized": self.duplicated_context_cost_optimized,
            "estimated_token_saving": self.estimated_token_saving,
            "context_reuse_rate": self.context_reuse_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "duplicated_context_cost_reduction": self.duplicated_context_cost_reduction,
            "stale_cache_count": self.stale_cache_count,
            "shared_memory_budget": self.shared_memory_budget,
        }


class ReusePolicy(str, Enum):
    """Policy for when to share vs. isolate context."""
    SAME_PATH = "same_path"              # Directly reuse shared knowledge
    PARTIAL_OVERLAP = "partial_overlap"   # Reuse longest common prefix
    UNRELATED = "unrelated"               # Only share global project context
    STALE = "stale"                       # Knowledge is stale, must refresh


class SharedContextManager:
    """Manages same-path shared prerequisite knowledge for subagents.

    Determines what context each subagent should receive based on its
    position in the DAG, avoiding duplicated context transmission.
    """

    def __init__(
        self,
        dag: TaskDAG,
        knowledge_store: SharedKnowledgeStore | None = None,
        budget_tracker: ContextBudgetTracker | None = None,
    ) -> None:
        self._dag = dag
        self._store = knowledge_store or SharedKnowledgeStore()
        self._budget = budget_tracker or ContextBudgetTracker()
        self._path_signatures: dict[str, PathSignature] = {}
        self._compute_all_path_signatures()

    def _compute_all_path_signatures(self) -> None:
        """Pre-compute path signatures for all nodes in the DAG."""
        for nid in self._dag.nodes:
            sig = PathSignature(
                node_id=nid,
                path_signature=self._dag.get_path_signature(nid),
                ancestor_set_signature=self._dag.get_ancestor_set_signature(nid),
                ancestor_ids=sorted(self._dag.get_ancestors(nid)),
                path_count=len(self._dag.get_path_to(nid)),
            )
            self._path_signatures[nid] = sig

    def get_path_signature(self, node_id: str) -> PathSignature | None:
        """Get the path signature for a specific node."""
        return self._path_signatures.get(node_id)

    def find_same_path_nodes(self, node_id: str) -> list[str]:
        """Find all nodes that share the same DAG prefix path."""
        sig = self._path_signatures.get(node_id)
        if not sig:
            return []
        return [
            nid for nid, s in self._path_signatures.items()
            if s.path_signature == sig.path_signature and nid != node_id
        ]

    def find_partial_overlap_nodes(self, node_id: str) -> list[str]:
        """Find nodes with partially overlapping ancestor sets."""
        sig = self._path_signatures.get(node_id)
        if not sig:
            return []
        node_ancestors = set(sig.ancestor_ids)
        overlaps: list[tuple[str, float]] = []

        for nid, other_sig in self._path_signatures.items():
            if nid == node_id:
                continue
            if other_sig.path_signature == sig.path_signature:
                continue  # Same path, not partial
            other_ancestors = set(other_sig.ancestor_ids)
            if not other_ancestors and not node_ancestors:
                continue
            # Jaccard similarity
            union = len(node_ancestors | other_ancestors)
            if union == 0:
                continue
            intersection = len(node_ancestors & other_ancestors)
            similarity = intersection / union
            if similarity > 0.3:  # Threshold for "partial overlap"
                overlaps.append((nid, similarity))

        overlaps.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in overlaps]

    def get_reuse_policy(self, node_id: str) -> ReusePolicy:
        """Determine the reuse policy for a given node."""
        sig = self._path_signatures.get(node_id)
        if not sig:
            return ReusePolicy.UNRELATED

        # Check for same-path peers
        same_path_nodes = self.find_same_path_nodes(node_id)
        if same_path_nodes:
            return ReusePolicy.SAME_PATH

        # Check for partial overlaps
        partial_nodes = self.find_partial_overlap_nodes(node_id)
        if partial_nodes:
            return ReusePolicy.PARTIAL_OVERLAP

        return ReusePolicy.UNRELATED

    def get_context_for_node(
        self,
        node_id: str,
        base_context: str = "",
    ) -> tuple[str, int]:
        """Get the appropriate shared context for a subagent working on a node.

        Returns:
            Tuple of (context_string, estimated_token_size).
        """
        policy = self.get_reuse_policy(node_id)
        node = self._dag.nodes.get(node_id)

        if policy == ReusePolicy.SAME_PATH:
            # Find the shared knowledge from same-path peers
            same_path_nodes = self.find_same_path_nodes(node_id)
            # Use the first same-path node's knowledge (arbitration)
            peer_knowledge = ""
            for peer_id in same_path_nodes:
                entries = self._store.get_by_prefix(f"node_{peer_id}_")
                if entries:
                    for entry in entries:
                        peer_knowledge += f"\n[{entry.entry_type.value}] {entry.content[:200]}"
                    break

            context = self._build_context_package(node, base_context, peer_knowledge)

        elif policy == ReusePolicy.PARTIAL_OVERLAP:
            # Use longest common prefix knowledge
            partial_nodes = self.find_partial_overlap_nodes(node_id)
            lcp_knowledge = ""
            if partial_nodes:
                # Take knowledge from the most overlapping node
                best_peer = partial_nodes[0]
                entries = self._store.get_by_prefix(f"node_{best_peer}_")
                for entry in entries:
                    lcp_knowledge += f"\n[{entry.entry_type.value}] {entry.content[:200]}"

            context = self._build_context_package(node, base_context, lcp_knowledge)

        else:
            # Unrelated — only global project context
            context = self._build_context_package(node, base_context, "")

        token_size = KnowledgeEntry._estimate_tokens(context)
        self._budget.record_request(policy != ReusePolicy.UNRELATED)
        return context, token_size

    def _build_context_package(
        self,
        node: TaskNode | None,
        base_context: str,
        shared_knowledge: str,
    ) -> str:
        """Build a structured context package for a subagent."""
        parts = [f"# Task Context for: {node.name if node else 'unknown'}"]
        if node:
            parts.append(f"\n## Task Description\n{node.description}")
            if node.input_dependencies:
                parts.append(f"\n## Dependencies\n- " + "\n- ".join(node.input_dependencies))

        if shared_knowledge:
            parts.append(f"\n## Shared Prerequisite Knowledge{shared_knowledge}")

        if base_context:
            parts.append(f"\n## Global Project Context\n{base_context[:500]}")

        return "\n".join(parts)

    def store_node_knowledge(
        self,
        node_id: str,
        knowledge: list[dict[str, Any]],
    ) -> None:
        """Store knowledge produced by a completed node for reuse by dependents."""
        for i, entry_data in enumerate(knowledge):
            entry = KnowledgeEntry(
                key=f"node_{node_id}_k{i}",
                entry_type=KnowledgeEntryType(entry_data.get("type", "code_context")),
                content=entry_data.get("content", ""),
                source_node_id=node_id,
            )
            self._store.put(entry)
            self._budget.shared_context_size += entry.token_size_estimate

    def get_longest_common_prefix_context(
        self,
        node_ids: list[str],
        max_tokens: int = 2000,
    ) -> str:
        """Extract shared knowledge that is common across multiple nodes.

        Finds the longest common prefix in terms of ancestor sets.
        """
        if not node_ids:
            return ""

        # Find common ancestors across all given nodes
        common_ancestors: set[str] | None = None
        for nid in node_ids:
            ancestors = self._dag.get_ancestors(nid)
            if common_ancestors is None:
                common_ancestors = ancestors
            else:
                common_ancestors &= ancestors

        if not common_ancestors:
            return ""

        # Collect knowledge from common ancestors
        parts: list[str] = []
        total_tokens = 0
        for ancestor_id in sorted(common_ancestors):
            entries = self._store.get_by_prefix(f"node_{ancestor_id}_")
            for entry in entries:
                snippet = entry.content[:300]
                tokens = KnowledgeEntry._estimate_tokens(snippet)
                if total_tokens + tokens > max_tokens:
                    break
                parts.append(f"[{entry.entry_type.value}] {snippet}")
                total_tokens += tokens

        return "\n".join(parts)

    def get_metrics(self) -> dict[str, Any]:
        """Get context reuse metrics."""
        metrics = self._budget.to_dict()
        metrics["knowledge_store_entries"] = self._store.entry_count
        metrics["knowledge_store_total_tokens"] = self._store.total_tokens
        metrics["path_signature_count"] = len(self._path_signatures)

        # Count unique path signatures
        unique_sigs = len({s.path_signature for s in self._path_signatures.values()})
        metrics["unique_path_signatures"] = unique_sigs

        # Reuse rate
        if self._budget.total_context_requests > 0:
            metrics["context_reuse_rate"] = self._budget.cache_hit_rate
        else:
            metrics["context_reuse_rate"] = 0.0

        return metrics
