"""Layer 3 -- coordination quality.

All measurements here are *observational*: they describe what happened in the
recorded trajectory.  They are inputs to hypotheses (swarm smells) and must
not be read as causal contribution -- see ``swarmeval.causal`` for that.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..graph.builder import ExecutionGraph
from ..schema import COORDINATION_KINDS, EventType, Trajectory, estimate_tokens, to_text
from ..stats import containment, jaccard, shingles, text_similarity

VERIFICATION_ROLES = {"critic", "verifier", "reviewer", "validator", "judge", "checker"}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class AgentUtilization:
    agent_id: str
    role: Optional[str] = None
    actions: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    token_share: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    artifacts_produced: int = 0
    artifacts_used_downstream: int = 0
    messages_sent: int = 0
    messages_used: int = 0
    messages_received: int = 0
    downstream_uses: int = 0
    final_output_overlap: float = 0.0
    produced_final_output: bool = False
    observed_contribution: float = 0.0
    dead: bool = False
    critical_path_s: float = 0.0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommunicationMetrics:
    messages: int = 0
    delivered: int = 0
    dropped: int = 0
    message_tokens: int = 0
    llm_tokens: int = 0
    communication_ratio: float = 0.0        # message tokens / llm tokens (clipped to 1)
    coordination_messages: int = 0
    information_messages: int = 0
    useful_work_fraction: float = 1.0
    information_transfer_fraction: float = 0.0
    coordination_fraction: float = 0.0
    repeated_messages: int = 0
    broadcast_messages: int = 0
    hub_agent: Optional[str] = None
    hub_share: float = 0.0                  # share of messages involving the hub
    density: float = 0.0
    per_channel: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RedundancyMetrics:
    total_actions: int = 0
    redundant_actions: int = 0
    redundant_action_fraction: float = 0.0
    total_artifact_tokens: int = 0
    redundant_artifact_tokens: int = 0
    redundant_token_fraction: float = 0.0
    ratio: float = 0.0                      # headline redundancy ratio
    accidental: int = 0
    intentional: int = 0
    duplicate_tool_calls: int = 0
    self_repeats: int = 0
    pair_overlap: Dict[str, float] = field(default_factory=dict)   # "a|b" -> jaccard
    highest_overlap: Optional[Tuple[str, str, float]] = None
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["highest_overlap"] = list(self.highest_overlap) if self.highest_overlap else None
        return d


@dataclass
class DependencyMetrics:
    depth: int = 0
    edges: int = 0
    max_fan_in: int = 0
    max_fan_in_agent: Optional[str] = None
    max_fan_out: int = 0
    max_fan_out_agent: Optional[str] = None
    cycles: List[List[str]] = field(default_factory=list)
    isolated_agents: List[str] = field(default_factory=list)
    serial_bottlenecks: List[str] = field(default_factory=list)
    unused_channels: List[str] = field(default_factory=list)      # messages whose receiver never used them
    fan_in: Dict[str, int] = field(default_factory=dict)
    fan_out: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelismMetrics:
    total_work_s: float = 0.0
    wall_clock_s: float = 0.0
    critical_path_s: float = 0.0
    actual_parallel_fraction: float = 0.0     # 1 - wall / work
    potential_parallel_fraction: float = 0.0  # 1 - critical_path / work
    actual_speedup: float = 1.0
    potential_speedup: float = 1.0
    opportunity: float = 0.0                   # potential - actual
    max_concurrency: int = 0
    mean_concurrency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextBloatMetrics:
    calls_measured: int = 0
    bloated_calls: int = 0
    mean_bloat_ratio: float = 0.0
    excess_tokens: int = 0
    worst: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoordinationMetrics:
    utilization: Dict[str, AgentUtilization] = field(default_factory=dict)
    communication: CommunicationMetrics = field(default_factory=CommunicationMetrics)
    redundancy: RedundancyMetrics = field(default_factory=RedundancyMetrics)
    dependency: DependencyMetrics = field(default_factory=DependencyMetrics)
    parallelism: ParallelismMetrics = field(default_factory=ParallelismMetrics)
    context_bloat: ContextBloatMetrics = field(default_factory=ContextBloatMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utilization": {k: v.to_dict() for k, v in self.utilization.items()},
            "communication": self.communication.to_dict(),
            "redundancy": self.redundancy.to_dict(),
            "dependency": self.dependency.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "context_bloat": self.context_bloat.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _message_used(traj: Trajectory, msg_event, min_containment: float = 0.3) -> bool:
    """Did the receiver observably use this message afterwards?

    If the message carried artifacts, usage means the receiver later
    referenced one of them (explicit provenance).  Otherwise usage means a
    later event of the receiver (prompt, output, artifact, outgoing message,
    tool arguments) contains the message text.  Very short messages count as
    used if the receiver did any work after receiving them."""
    receiver = msg_event.receiver
    t = msg_event.timestamp
    attached = set(msg_event.input_artifacts)
    content = to_text(msg_event.content)
    short = len(content.split()) < 4
    later = [e for e in traj.events if e.agent_id == receiver and e.timestamp >= t - 1e-9
             and e.event_id != msg_event.event_id]
    if attached:
        return any(attached.intersection(e.input_artifacts) for e in later)
    for e in later:
        if e.event_type in (EventType.LLM_CALL.value, EventType.TOOL_CALL.value) and short:
            return True
        texts = []
        if e.event_type == EventType.LLM_CALL.value:
            texts.append(to_text(e.metadata.get("prompt", "")))
            texts.append(to_text(e.content))
        elif e.event_type in (EventType.ARTIFACT.value, EventType.MESSAGE.value):
            texts.append(to_text(e.content))
        elif e.event_type == EventType.TOOL_CALL.value:
            texts.append(to_text(e.tool_args))
        for tx in texts:
            if tx and containment(content, tx) >= min_containment:
                return True
    return False


def final_lineage(traj: Trajectory) -> Set[str]:
    """Artifacts that transitively fed the final output, following the
    ``input_artifacts`` recorded on events in each producing span."""
    lineage: Set[str] = set()
    frontier = list(traj.output_artifacts)
    while frontier:
        a = frontier.pop()
        if a in lineage or a not in traj.artifacts:
            continue
        lineage.add(a)
        art = traj.artifacts[a]
        for e in traj.events:
            if e.span_id == art.span_id and e.timestamp <= art.produced_at + 1e-9:
                frontier.extend(x for x in e.input_artifacts if x not in lineage)
    return lineage


def _concurrency(units) -> Tuple[int, float]:
    points = []
    for u in units:
        points.append((u.start, 1))
        points.append((u.end, -1))
    if not points:
        return 0, 0.0
    points.sort(key=lambda p: (p[0], p[1]))
    cur = 0
    max_c = 0
    area = 0.0
    last_t = points[0][0]
    for t, d in points:
        area += cur * (t - last_t)
        last_t = t
        cur += d
        max_c = max(max_c, cur)
    span = points[-1][0] - points[0][0]
    return max_c, (area / span if span > 0 else 0.0)


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #
def compute_coordination(traj: Trajectory, graph: ExecutionGraph,
                         bloat_ratio_threshold: float = 0.6, bloat_min_tokens: int = 2000,
                         artifact_similarity: float = 0.6) -> CoordinationMetrics:
    cm = CoordinationMetrics()
    agents = traj.agent_ids()
    final_text = to_text(traj.final_output)
    llm_tokens = traj.total_tokens
    cp_share = graph.critical_path_agents()

    # --- utilization ---------------------------------------------------------
    util: Dict[str, AgentUtilization] = {a: AgentUtilization(a, graph.roles.get(a)) for a in agents}
    for ev in traj.events:
        if not ev.agent_id or ev.agent_id not in util:
            continue
        u = util[ev.agent_id]
        if ev.event_type == EventType.LLM_CALL.value:
            u.actions += 1
            u.llm_calls += 1
            u.tokens += ev.total_tokens
            if ev.status == "error":
                u.errors += 1
        elif ev.event_type == EventType.TOOL_CALL.value:
            u.actions += 1
            u.tool_calls += 1
            if ev.tool_name and ev.tool_name not in u.tools_used:
                u.tools_used.append(ev.tool_name)
            if ev.status == "error":
                u.errors += 1
        elif ev.event_type == EventType.ARTIFACT.value:
            u.artifacts_produced += 1
    for ev in traj.messages():
        if ev.sender in util:
            util[ev.sender].messages_sent += 1
        if ev.receiver in util and ev.status in ("success", "injected"):
            util[ev.receiver].messages_received += 1

    # artifact downstream usage
    consumers: Dict[str, Set[str]] = defaultdict(set)
    for ev in traj.events:
        for a in ev.input_artifacts:
            if ev.agent_id:
                consumers[a].add(ev.agent_id)
    output_arts = set(traj.output_artifacts)
    for art in traj.artifacts.values():
        if not art.producer or art.producer not in util:
            continue
        u = util[art.producer]
        others = {c for c in consumers.get(art.artifact_id, set()) if c != art.producer}
        if others or art.artifact_id in output_arts:
            u.artifacts_used_downstream += 1
        if art.artifact_id in output_arts:
            u.produced_final_output = True
        if final_text:
            u.final_output_overlap = max(u.final_output_overlap, containment(to_text(art.content), final_text))
    # llm outputs also count toward final-output overlap
    for ev in traj.llm_calls():
        if ev.agent_id in util and ev.content is not None and final_text:
            util[ev.agent_id].final_output_overlap = max(util[ev.agent_id].final_output_overlap,
                                                        containment(to_text(ev.content), final_text))
    # message usage
    for ev in traj.messages():
        if ev.sender in util and ev.status in ("success", "injected") and ev.receiver != ev.sender:
            if _message_used(traj, ev):
                util[ev.sender].messages_used += 1
    provenance = any(ev.input_artifacts for ev in traj.events)
    lineage = final_lineage(traj)
    for a, u in util.items():
        u.downstream_uses = u.artifacts_used_downstream + u.messages_used
        u.token_share = (u.tokens / llm_tokens) if llm_tokens else 0.0
        u.critical_path_s = cp_share.get(a, 0.0)
        in_lineage = any(art.producer == a and art.artifact_id in lineage for art in traj.artifacts.values())
        if in_lineage:
            u.produced_final_output = u.produced_final_output or any(
                art.producer == a and art.artifact_id in output_arts for art in traj.artifacts.values())
        art_frac = (u.artifacts_used_downstream / u.artifacts_produced) if u.artifacts_produced else 0.0
        msg_frac = (u.messages_used / u.messages_sent) if u.messages_sent else 0.0
        contribution = max(art_frac, msg_frac, 1.0 if in_lineage else 0.0,
                           0.0 if provenance else u.final_output_overlap)
        if u.produced_final_output:
            contribution = 1.0
        u.observed_contribution = max(0.0, min(1.0, contribution))
        if provenance:
            u.dead = u.actions > 0 and u.downstream_uses == 0 and not in_lineage and not u.produced_final_output
        else:
            u.dead = (u.actions > 0 and u.downstream_uses == 0 and u.final_output_overlap < 0.1
                      and not u.produced_final_output)
    cm.utilization = util

    # --- communication -------------------------------------------------------
    com = CommunicationMetrics()
    msgs = traj.messages()
    com.messages = len(msgs)
    com.llm_tokens = llm_tokens
    involvement: Dict[str, int] = defaultdict(int)
    seen_hash: Set[Tuple[str, str, str]] = set()
    by_sender_hash: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    coord_tokens = 0
    info_tokens = 0
    for ev in msgs:
        com.message_tokens += ev.content_tokens
        if ev.status in ("success", "injected"):
            com.delivered += 1
        else:
            com.dropped += 1
        if (ev.kind or "information") in COORDINATION_KINDS:
            com.coordination_messages += 1
            coord_tokens += ev.content_tokens
        else:
            com.information_messages += 1
            info_tokens += ev.content_tokens
        if ev.sender:
            involvement[ev.sender] += 1
        if ev.receiver and ev.receiver != ev.sender:
            involvement[ev.receiver] += 1
        key = (ev.sender or "", ev.receiver or "", ev.content_hash or "")
        if key in seen_hash:
            com.repeated_messages += 1
        seen_hash.add(key)
        by_sender_hash[(ev.sender or "", ev.content_hash or "")].add(ev.receiver or "")
        ch = f"{ev.sender}->{ev.receiver}"
        com.per_channel[ch] = com.per_channel.get(ch, 0) + 1
    com.broadcast_messages = sum(1 for r in by_sender_hash.values() if len(r) >= 3)
    denom = max(1, llm_tokens)
    com.communication_ratio = min(1.0, com.message_tokens / denom)
    com.coordination_fraction = min(1.0, coord_tokens / denom)
    com.information_transfer_fraction = min(1.0, info_tokens / denom)
    com.useful_work_fraction = max(0.0, 1.0 - com.coordination_fraction - com.information_transfer_fraction)
    if involvement and com.messages:
        hub = max(involvement, key=lambda k: involvement[k])
        com.hub_agent = hub
        com.hub_share = involvement[hub] / com.messages
    com.density = graph.communication_density()
    cm.communication = com

    # --- redundancy ----------------------------------------------------------
    red = RedundancyMetrics()
    fingerprints: Dict[str, Set[str]] = defaultdict(set)   # agent -> fingerprints for pairwise overlap
    seen_fp: Dict[str, str] = {}                            # fingerprint -> first agent
    verification_agents = {a for a, r in graph.roles.items() if (r or "").lower() in VERIFICATION_ROLES}
    for ev in traj.tool_calls():
        if ev.status not in ("success",) or not ev.agent_id:
            continue
        fp = f"tool:{ev.tool_name}:{' '.join(to_text(ev.tool_args).lower().split())}"
        red.total_actions += 1
        fingerprints[ev.agent_id].add(fp)
        if fp in seen_fp:
            red.redundant_actions += 1
            red.duplicate_tool_calls += 1
            first = seen_fp[fp]
            if first == ev.agent_id:
                red.self_repeats += 1
            intentional = ev.agent_id in verification_agents
            if intentional:
                red.intentional += 1
            else:
                red.accidental += 1
            red.details.append({"type": "tool_call", "agent": ev.agent_id, "duplicates": first,
                                "tool": ev.tool_name, "intentional": intentional})
        else:
            seen_fp[fp] = ev.agent_id
    arts = sorted(traj.artifacts.values(), key=lambda a: a.produced_at)
    art_shingles = {a.artifact_id: shingles(to_text(a.content)) for a in arts}
    # agents that consumed another agent's artifacts: composition is not duplication
    consumed_from: Dict[str, Set[str]] = defaultdict(set)
    for aid, cs in consumers.items():
        prod = traj.artifacts[aid].producer if aid in traj.artifacts else None
        for c in cs:
            if prod and c != prod:
                consumed_from[c].add(prod)
    for i, art in enumerate(arts):
        red.total_actions += 1
        red.total_artifact_tokens += art.tokens
        if art.producer:
            fingerprints[art.producer].update(f"art:{s}" for s in art_shingles[art.artifact_id])
        for prev in arts[:i]:
            if prev.producer == art.producer:
                continue
            if art.producer and prev.producer in consumed_from.get(art.producer, set()):
                continue
            sim = jaccard(art_shingles[art.artifact_id], art_shingles[prev.artifact_id])
            if sim >= artifact_similarity:
                red.redundant_actions += 1
                red.redundant_artifact_tokens += art.tokens
                intentional = (art.intent == "verification") or (art.producer in verification_agents)
                if intentional:
                    red.intentional += 1
                else:
                    red.accidental += 1
                red.details.append({"type": "artifact", "agent": art.producer, "duplicates": prev.producer,
                                    "artifact": art.name, "similarity": round(sim, 3), "intentional": intentional})
                break
    if red.total_actions:
        red.redundant_action_fraction = red.redundant_actions / red.total_actions
    if red.total_artifact_tokens:
        red.redundant_token_fraction = red.redundant_artifact_tokens / red.total_artifact_tokens
    red.ratio = red.redundant_action_fraction
    best: Optional[Tuple[str, str, float]] = None
    agent_list = [a for a in agents if fingerprints.get(a)]
    for i, a in enumerate(agent_list):
        for b in agent_list[i + 1:]:
            if b in consumed_from.get(a, set()) or a in consumed_from.get(b, set()):
                continue  # producer/consumer pairs legitimately share content
            j = jaccard(fingerprints[a], fingerprints[b])
            red.pair_overlap[f"{a}|{b}"] = j
            if best is None or j > best[2]:
                best = (a, b, j)
    red.highest_overlap = best
    cm.redundancy = red

    # --- dependency ----------------------------------------------------------
    dep = DependencyMetrics()
    dep.depth = graph.depth()
    dep.edges = sum(1 for (s, t, k) in graph.edges if k in ("message", "artifact"))
    for a in agents:
        fi, fo = graph.fan_in(a), graph.fan_out(a)
        dep.fan_in[a] = fi
        dep.fan_out[a] = fo
        if fi > dep.max_fan_in:
            dep.max_fan_in, dep.max_fan_in_agent = fi, a
        if fo > dep.max_fan_out:
            dep.max_fan_out, dep.max_fan_out_agent = fo, a
    dep.cycles = graph.cycles()
    dep.isolated_agents = graph.isolated_agents()
    adj = graph.adjacency(kinds=("message",))
    preds: Dict[str, Set[str]] = defaultdict(set)
    for s, ts in adj.items():
        for t in ts:
            preds[t].add(s)
    for a in agents:
        ps = {p for p in preds.get(a, set()) if p != a}
        ss = {s for s in adj.get(a, set()) if s != a}
        if len(ps) >= 2 and ss:
            bypass = any(s in adj.get(p, set()) for p in ps for s in ss)
            if not bypass:
                dep.serial_bottlenecks.append(a)
    for (s, t, k), e in graph.edges.items():
        if k == "message" and e.used_downstream == 0 and s != t:
            dep.unused_channels.append(f"{s}->{t}")
    cm.dependency = dep

    # --- parallelism ---------------------------------------------------------
    par = ParallelismMetrics()
    par.total_work_s = graph.total_work()
    par.wall_clock_s = traj.duration
    par.critical_path_s, _ = graph.critical_path()
    if par.total_work_s > 0:
        par.actual_parallel_fraction = max(0.0, 1.0 - par.wall_clock_s / par.total_work_s)
        par.potential_parallel_fraction = max(0.0, 1.0 - par.critical_path_s / par.total_work_s)
        par.actual_speedup = par.total_work_s / par.wall_clock_s if par.wall_clock_s > 0 else 1.0
        par.potential_speedup = par.total_work_s / par.critical_path_s if par.critical_path_s > 0 else 1.0
    par.opportunity = max(0.0, par.potential_parallel_fraction - par.actual_parallel_fraction)
    par.max_concurrency, par.mean_concurrency = _concurrency(graph.units)
    cm.parallelism = par

    # --- context bloat -------------------------------------------------------
    cb = ContextBloatMetrics()
    task_tokens = estimate_tokens(traj.task.input)
    ratios: List[float] = []
    for ev in traj.llm_calls():
        if ev.input_tokens <= 0:
            continue
        relevant = ev.metadata.get("relevant_tokens")
        if relevant is None:
            if not ev.input_artifacts:
                continue
            relevant = task_tokens + sum(traj.artifacts[a].tokens for a in ev.input_artifacts if a in traj.artifacts)
        relevant = int(relevant)
        ratio = max(0.0, 1.0 - min(1.0, relevant / ev.input_tokens))
        ratios.append(ratio)
        cb.calls_measured += 1
        excess = max(0, ev.input_tokens - relevant)
        if ratio >= bloat_ratio_threshold and ev.input_tokens >= bloat_min_tokens:
            cb.bloated_calls += 1
            cb.excess_tokens += excess
            cb.worst.append({"agent": ev.agent_id, "input_tokens": ev.input_tokens,
                             "relevant_tokens": relevant, "ratio": round(ratio, 3), "event_id": ev.event_id})
    cb.worst.sort(key=lambda w: -w["ratio"])
    cb.worst = cb.worst[:5]
    cb.mean_bloat_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    cm.context_bloat = cb
    return cm
