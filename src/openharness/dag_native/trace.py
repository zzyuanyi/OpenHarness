"""Trace recorder for DAG-Native execution.

Records a structured trace of the entire planning-to-execution pipeline,
including node transitions, subagent dispatch, context decisions, and timing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from openharness.dag_native.graph import NodeStatus


class TraceEventType(str, Enum):
    """Types of trace events."""
    # Planning
    TASK_PARSED = "task_parsed"
    DAG_BUILT = "dag_built"
    TOPO_SORTED = "topo_sorted"
    REVERSE_TOPO_SORTED = "reverse_topo_sorted"
    CRITICAL_PATH_IDENTIFIED = "critical_path_identified"
    PLAN_GENERATED = "plan_generated"

    # Execution
    NODE_STATUS_CHANGE = "node_status_change"
    SUBAGENT_DISPATCHED = "subagent_dispatched"
    SUBAGENT_COMPLETED = "subagent_completed"
    SUBAGENT_FAILED = "subagent_failed"

    # Context
    CONTEXT_REUSED = "context_reused"
    CONTEXT_CACHE_HIT = "context_cache_hit"
    CONTEXT_CACHE_MISS = "context_cache_miss"
    KNOWLEDGE_STORED = "knowledge_stored"
    CONTEXT_STALE = "context_stale"

    # Experiment
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    METRICS_RECORDED = "metrics_recorded"


@dataclass
class TraceEvent:
    """A single trace event."""
    event_id: str
    event_type: TraceEventType
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    agent_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class TraceRecorder:
    """Records structured trace events throughout DAG-Native execution."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._event_counter: int = 0
        self._start_time: float = time.time()

    def record(
        self,
        event_type: TraceEventType,
        node_id: str = "",
        agent_id: str = "",
        data: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> TraceEvent:
        """Record a trace event."""
        self._event_counter += 1
        event = TraceEvent(
            event_id=f"evt_{self._event_counter:04d}",
            event_type=event_type,
            timestamp=time.time(),
            node_id=node_id,
            agent_id=agent_id,
            data=data or {},
            duration_ms=duration_ms,
        )
        self._events.append(event)
        return event

    def record_node_transition(
        self,
        node_id: str,
        from_status: NodeStatus | None,
        to_status: NodeStatus,
    ) -> TraceEvent:
        """Record a node status transition."""
        return self.record(
            TraceEventType.NODE_STATUS_CHANGE,
            node_id=node_id,
            data={
                "from": from_status.value if from_status else None,
                "to": to_status.value,
            },
        )

    def record_subagent_dispatch(
        self,
        node_id: str,
        agent_type: str,
        context_size: int,
    ) -> TraceEvent:
        """Record a subagent dispatch."""
        return self.record(
            TraceEventType.SUBAGENT_DISPATCHED,
            node_id=node_id,
            agent_id=agent_type,
            data={
                "agent_type": agent_type,
                "context_size": context_size,
            },
        )

    def record_context_reuse(
        self,
        node_id: str,
        policy: str,
        source_node_id: str = "",
    ) -> TraceEvent:
        """Record a context reuse decision."""
        return self.record(
            TraceEventType.CONTEXT_REUSED,
            node_id=node_id,
            data={
                "policy": policy,
                "source_node": source_node_id,
            },
        )

    def get_events_by_type(self, event_type: TraceEventType) -> list[TraceEvent]:
        """Filter events by type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_node(self, node_id: str) -> list[TraceEvent]:
        """Filter events by node."""
        return [e for e in self._events if e.node_id == node_id]

    def get_timeline(self) -> list[dict[str, Any]]:
        """Get a chronological timeline of all events."""
        timeline = []
        for e in sorted(self._events, key=lambda x: x.timestamp):
            timeline.append({
                "event_id": e.event_id,
                "type": e.event_type.value,
                "timestamp": e.timestamp,
                "elapsed_ms": (e.timestamp - self._start_time) * 1000,
                "node_id": e.node_id,
                "agent_id": e.agent_id,
                "data": e.data,
            })
        return timeline

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": len(self._events),
            "duration_ms": (time.time() - self._start_time) * 1000,
            "timeline": self.get_timeline(),
            "summary": {
                "node_transitions": len(
                    self.get_events_by_type(TraceEventType.NODE_STATUS_CHANGE)
                ),
                "subagent_dispatches": len(
                    self.get_events_by_type(TraceEventType.SUBAGENT_DISPATCHED)
                ),
                "context_reuses": len(
                    self.get_events_by_type(TraceEventType.CONTEXT_REUSED)
                ),
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save trace to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Clear all trace events."""
        self._events.clear()
        self._event_counter = 0
        self._start_time = time.time()

    @property
    def event_count(self) -> int:
        return len(self._events)
