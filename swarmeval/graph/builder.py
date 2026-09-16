"""Execution graph construction and analysis.

Two views are built from a trajectory:

* **Agent graph** -- nodes are agents, edges are messages, artifact flows and
  parent/child invocations.  Used for bottleneck, cycle and density analysis.
* **Work-unit DAG** -- nodes are LLM and tool calls with durations; edges are
  inferred dependencies.  Used for critical-path and parallelism analysis.

Dependencies in the work-unit DAG are *inferred* from what was recorded:
sequential order within an agent invocation, messages, artifact production
and consumption, and nesting.  If a swarm passes information between agents
outside the recorder, the DAG will under-estimate dependencies (and thus
over-estimate potential parallelism).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schema import Event, EventType, Span, Trajectory


@dataclass
class WorkUnit:
    unit_id: str
    agent_id: str
    span_id: str
    kind: str              # llm_call | tool_call
    start: float
    end: float
    tokens: int = 0
    event: Optional[Event] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AgentEdge:
    source: str
    target: str
    kind: str              # message | artifact | invocation
    count: int = 0
    tokens: int = 0
    used_downstream: int = 0   # messages/artifacts that were consumed by the receiver

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind,
                "count": self.count, "tokens": self.tokens, "used_downstream": self.used_downstream}


@dataclass
class ExecutionGraph:
    trajectory: Trajectory
    agents: List[str] = field(default_factory=list)
    roles: Dict[str, Optional[str]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], AgentEdge] = field(default_factory=dict)
    units: List[WorkUnit] = field(default_factory=list)
    deps: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))   # unit -> prerequisite units
    _critical: Optional[Tuple[float, List[str]]] = None

    # -- agent-level queries --------------------------------------------------
    def adjacency(self, kinds: Tuple[str, ...] = ("message", "artifact")) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for (s, t, k), e in self.edges.items():
            if k in kinds:
                adj[s].add(t)
        return adj

    def fan_in(self, agent: str, kinds: Tuple[str, ...] = ("message", "artifact")) -> int:
        """Number of distinct agents with an edge into ``agent``."""
        return len({s for (s, t, k) in self.edges if t == agent and k in kinds and s != agent})

    def fan_out(self, agent: str, kinds: Tuple[str, ...] = ("message", "artifact")) -> int:
        """Number of distinct agents ``agent`` has an edge to."""
        return len({t for (s, t, k) in self.edges if s == agent and k in kinds and t != agent})

    def message_edges(self) -> List[AgentEdge]:
        return [e for (_, _, k), e in self.edges.items() if k == "message"]

    def cycles(self) -> List[List[str]]:
        """Simple cycles in the message/artifact agent graph (bounded search)."""
        adj = self.adjacency()
        found: List[List[str]] = []
        seen: Set[Tuple[str, ...]] = set()

        def dfs(start: str, node: str, path: List[str]) -> None:
            if len(path) > 8:
                return
            for nxt in adj.get(node, ()):
                if nxt == start:
                    cyc = path[:]
                    key = tuple(sorted(cyc))
                    if key not in seen:
                        seen.add(key)
                        found.append(cyc)
                elif nxt not in path:
                    dfs(start, nxt, path + [nxt])

        for a in self.agents:
            dfs(a, a, [a])
        return found

    def isolated_agents(self) -> List[str]:
        """Agents with no message/artifact edges to any other agent."""
        connected: Set[str] = set()
        for (s, t, k) in self.edges:
            if k in ("message", "artifact") and s != t:
                connected.add(s)
                connected.add(t)
        return [a for a in self.agents if a not in connected]

    def communication_density(self) -> float:
        n = len(self.agents)
        if n < 2:
            return 0.0
        pairs = {(s, t) for (s, t, k) in self.edges if k == "message" and s != t}
        return len(pairs) / (n * (n - 1))

    def depth(self) -> int:
        """Longest dependency chain (in agent invocations) through the DAG."""
        return self._longest_chain_len()

    def _longest_chain_len(self) -> int:
        # longest path over units counted in distinct spans
        order = self._topo()
        best: Dict[str, int] = {}
        unit_by_id = {u.unit_id: u for u in self.units}
        for uid in order:
            preds = self.deps.get(uid, ())
            span = unit_by_id[uid].span_id
            b = 1
            for p in preds:
                pv = best.get(p, 1)
                b = max(b, pv + (0 if unit_by_id[p].span_id == span else 1))
            best[uid] = b
        return max(best.values()) if best else 0

    # -- work-unit DAG --------------------------------------------------------
    def _topo(self) -> List[str]:
        indeg: Dict[str, int] = {u.unit_id: 0 for u in self.units}
        succ: Dict[str, List[str]] = defaultdict(list)
        for uid, preds in self.deps.items():
            for p in preds:
                if p in indeg and uid in indeg:
                    indeg[uid] += 1
                    succ[p].append(uid)
        ready = sorted([u for u, d in indeg.items() if d == 0])
        order: List[str] = []
        while ready:
            u = ready.pop(0)
            order.append(u)
            for s in succ[u]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        # cycles (should not happen with time-ordered deps) -> append remaining
        for u in indeg:
            if u not in order:
                order.append(u)
        return order

    def critical_path(self) -> Tuple[float, List[str]]:
        """Longest-duration path through the work-unit DAG: the minimum
        makespan achievable with unlimited parallelism given the observed
        dependencies."""
        if self._critical is not None:
            return self._critical
        unit_by_id = {u.unit_id: u for u in self.units}
        dist: Dict[str, float] = {}
        prev: Dict[str, Optional[str]] = {}
        for uid in self._topo():
            u = unit_by_id[uid]
            best, bp = 0.0, None
            for p in self.deps.get(uid, ()):
                if p in dist and dist[p] > best:
                    best, bp = dist[p], p
            dist[uid] = best + u.duration
            prev[uid] = bp
        if not dist:
            self._critical = (0.0, [])
            return self._critical
        end = max(dist, key=lambda k: dist[k])
        path: List[str] = []
        cur: Optional[str] = end
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        self._critical = (dist[end], path)
        return self._critical

    def critical_path_agents(self) -> Dict[str, float]:
        unit_by_id = {u.unit_id: u for u in self.units}
        _, path = self.critical_path()
        share: Dict[str, float] = defaultdict(float)
        for uid in path:
            share[unit_by_id[uid].agent_id] += unit_by_id[uid].duration
        return dict(share)

    def total_work(self) -> float:
        return sum(u.duration for u in self.units)

    def to_dict(self) -> Dict[str, Any]:
        cp_len, cp = self.critical_path()
        return {
            "agents": self.agents,
            "roles": self.roles,
            "edges": [e.to_dict() for e in self.edges.values()],
            "units": [{"unit_id": u.unit_id, "agent_id": u.agent_id, "span_id": u.span_id, "kind": u.kind,
                       "start": u.start, "end": u.end, "tokens": u.tokens} for u in self.units],
            "deps": {k: sorted(v) for k, v in self.deps.items()},
            "critical_path": cp,
            "critical_path_s": cp_len,
            "total_work_s": self.total_work(),
            "depth": self.depth(),
            "cycles": self.cycles(),
            "density": self.communication_density(),
        }


def build_graph(traj: Trajectory) -> ExecutionGraph:
    g = ExecutionGraph(trajectory=traj)
    spans = traj.spans
    g.agents = traj.agent_ids()
    for s in spans.values():
        g.roles.setdefault(s.agent_id, s.role)

    def edge(s: str, t: str, kind: str) -> AgentEdge:
        key = (s, t, kind)
        if key not in g.edges:
            g.edges[key] = AgentEdge(s, t, kind)
        return g.edges[key]

    # invocation edges (parent -> child span)
    for s in spans.values():
        if s.parent_span_id and s.parent_span_id in spans:
            p = spans[s.parent_span_id]
            if p.agent_id != s.agent_id:
                e = edge(p.agent_id, s.agent_id, "invocation")
                e.count += 1

    # work units, in time order
    units: List[WorkUnit] = []
    for ev in traj.events:
        if ev.event_type in (EventType.LLM_CALL.value, EventType.TOOL_CALL.value) and ev.span_id:
            units.append(WorkUnit(ev.event_id, ev.agent_id or "?", ev.span_id, ev.event_type,
                                  ev.timestamp, ev.end_timestamp, ev.total_tokens, ev))
    units.sort(key=lambda u: (u.start, u.end))
    g.units = units
    by_span: Dict[str, List[WorkUnit]] = defaultdict(list)
    for u in units:
        by_span[u.span_id].append(u)

    # 1. sequential dependency within a span
    for su in by_span.values():
        for a, b in zip(su, su[1:]):
            g.deps[b.unit_id].add(a.unit_id)

    def last_unit_before(span_id: str, t: float) -> Optional[WorkUnit]:
        cands = [u for u in by_span.get(span_id, []) if u.end <= t + 1e-9]
        return cands[-1] if cands else None

    def first_unit_after(span_id: str, t: float) -> Optional[WorkUnit]:
        cands = [u for u in by_span.get(span_id, []) if u.start >= t - 1e-9]
        return cands[0] if cands else None

    def spans_of(agent: str) -> List[Span]:
        return traj.spans_for(agent)

    def last_unit_of_agent_before(agent: str, t: float) -> Optional[WorkUnit]:
        best: Optional[WorkUnit] = None
        for s in spans_of(agent):
            u = last_unit_before(s.span_id, t)
            if u and (best is None or u.end > best.end):
                best = u
        return best

    def first_unit_of_agent_after(agent: str, t: float) -> Optional[WorkUnit]:
        best: Optional[WorkUnit] = None
        for s in spans_of(agent):
            u = first_unit_after(s.span_id, t)
            if u and (best is None or u.start < best.start):
                best = u
        return best

    # 2. parent/child nesting
    for s in spans.values():
        if s.parent_span_id and s.parent_span_id in spans:
            child_first = by_span[s.span_id][0] if by_span.get(s.span_id) else None
            child_last = by_span[s.span_id][-1] if by_span.get(s.span_id) else None
            if child_first:
                p_prev = last_unit_before(s.parent_span_id, s.start)
                if p_prev:
                    g.deps[child_first.unit_id].add(p_prev.unit_id)
            if child_last and s.end is not None:
                p_next = first_unit_after(s.parent_span_id, s.end)
                if p_next:
                    g.deps[p_next.unit_id].add(child_last.unit_id)

    # 3. messages
    consumed_by: Dict[str, Set[str]] = defaultdict(set)   # artifact -> consuming agents
    for ev in traj.events:
        if ev.event_type == EventType.ARTIFACT_USE.value or ev.input_artifacts:
            for a in ev.input_artifacts:
                if ev.agent_id:
                    consumed_by[a].add(ev.agent_id)

    for ev in traj.messages():
        if not ev.sender or not ev.receiver:
            continue
        e = edge(ev.sender, ev.receiver, "message")
        e.count += 1
        e.tokens += ev.content_tokens
        if ev.status not in ("success", "injected"):
            continue
        src = last_unit_of_agent_before(ev.sender, ev.timestamp + 1e-6)
        dst = first_unit_of_agent_after(ev.receiver, ev.timestamp - 1e-6)
        if src and dst and src.unit_id != dst.unit_id:
            g.deps[dst.unit_id].add(src.unit_id)
        # downstream use: receiver has any later work
        if dst is not None:
            e.used_downstream += 1

    # 4. artifacts
    for art in traj.artifacts.values():
        for consumer in consumed_by.get(art.artifact_id, ()):
            if art.producer and consumer != art.producer:
                e = edge(art.producer, consumer, "artifact")
                e.count += 1
                e.tokens += art.tokens
                e.used_downstream += 1
        producer_unit = last_unit_of_agent_before(art.producer, art.produced_at + 1e-6) if art.producer else None
        for ev in traj.events:
            if art.artifact_id in ev.input_artifacts and ev.agent_id and ev.agent_id != art.producer:
                dst = first_unit_of_agent_after(ev.agent_id, ev.timestamp - 1e-6)
                if producer_unit and dst and producer_unit.unit_id != dst.unit_id:
                    g.deps[dst.unit_id].add(producer_unit.unit_id)

    return g
