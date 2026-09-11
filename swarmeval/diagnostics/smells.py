"""Swarm smell detection -- "ESLint for agent architectures".

Every smell is reported as a Diagnostic that keeps three things separate:

* **observation** -- what the trajectory shows (facts),
* **hypothesis**  -- what might be wrong (an interpretation),
* **recommendation** -- an intervention to *test*, not a verdict.

Smells are hypotheses generated from trajectory evidence.  Validate them with
``swarmeval.causal`` before acting on them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..graph.builder import ExecutionGraph
from ..metrics.coordination import CoordinationMetrics
from ..metrics.efficiency import EfficiencyMetrics
from ..schema import COORDINATION_KINDS, EventType, Trajectory, to_text
from ..stats import text_similarity

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Diagnostic:
    smell: str
    severity: str
    title: str
    observation: str
    hypothesis: str
    recommendation: str
    agents: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    run_ids: List[str] = field(default_factory=list)
    prevalence: Optional[float] = None       # fraction of runs exhibiting the smell

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        return (self.smell, tuple(self.agents))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Diagnostic":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SmellConfig:
    hub_message_share: float = 0.6
    hub_token_share: float = 0.25
    min_messages_for_hub: int = 4
    under_specialization_overlap: float = 0.5
    max_agents_simple: int = 5
    low_contribution: float = 0.1
    bloat_ratio: float = 0.6
    bloat_min_tokens: int = 2000
    ping_pong_min_alternations: int = 3
    redundant_delegation_similarity: float = 0.6
    parallelism_opportunity: float = 0.25
    communication_ratio_warning: float = 0.35
    synthesizer_roles: Tuple[str, ...] = ("synthesizer", "writer", "aggregator", "summarizer", "composer")


def _fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"


def detect_smells(traj: Trajectory, graph: ExecutionGraph, eff: EfficiencyMetrics,
                  coord: CoordinationMetrics, config: Optional[SmellConfig] = None) -> List[Diagnostic]:
    cfg = config or SmellConfig()
    out: List[Diagnostic] = []
    rid = traj.run_id
    util = coord.utilization
    total_tokens = max(1, traj.total_tokens)

    # 13.1 redundant delegation ---------------------------------------------
    by_receiver: Dict[str, List[Any]] = defaultdict(list)
    for ev in traj.messages():
        if (ev.kind or "") in COORDINATION_KINDS and ev.sender and ev.receiver and ev.sender != ev.receiver:
            by_receiver[ev.receiver].append(ev)
    for receiver, evs in by_receiver.items():
        senders = {e.sender for e in evs}
        if len(senders) < 2:
            continue
        best = 0.0
        pair = None
        for i, a in enumerate(evs):
            for b in evs[i + 1:]:
                if a.sender == b.sender:
                    continue
                sim = text_similarity(to_text(a.content), to_text(b.content))
                if sim > best:
                    best, pair = sim, (a.sender, b.sender)
        if pair and best >= cfg.redundant_delegation_similarity:
            out.append(Diagnostic(
                "redundant_delegation", "warning", "Redundant delegation",
                f"{len(senders)} agents ({', '.join(sorted(senders))}) sent {receiver} similar requests "
                f"(max similarity {best:.2f}).",
                "The architecture may be creating duplicate requests or contention on one agent.",
                f"Route requests to {receiver} through a single owner, or de-duplicate before dispatch.",
                agents=[receiver], evidence={"senders": sorted(senders), "similarity": round(best, 3),
                                             "requests": len(evs)}, run_ids=[rid]))

    # 13.2 agent ping-pong ---------------------------------------------------
    pair_seq: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    pair_tokens: Dict[Tuple[str, str], int] = defaultdict(int)
    for ev in traj.messages():
        if not ev.sender or not ev.receiver or ev.sender == ev.receiver or ev.status not in ("success", "injected"):
            continue
        key = tuple(sorted((ev.sender, ev.receiver)))
        pair_seq[key].append(ev.sender)
        pair_tokens[key] += ev.content_tokens
    for key, seq in pair_seq.items():
        alternations = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        if alternations >= cfg.ping_pong_min_alternations:
            out.append(Diagnostic(
                "agent_ping_pong", "warning", "Agent ping-pong",
                f"{key[0]} and {key[1]} exchanged {len(seq)} messages with {alternations} direction changes "
                f"({pair_tokens[key]} message tokens).",
                "Work is bouncing back and forth: repeated reassignment, corrections or a circular dependency.",
                "Merge the two roles, cap revision rounds, or give one agent the final say.",
                agents=list(key), evidence={"messages": len(seq), "alternations": alternations,
                                            "message_tokens": pair_tokens[key]}, run_ids=[rid]))

    # 13.3 over-agentization -------------------------------------------------
    agents = traj.agent_ids()
    low_value = [a for a, u in util.items() if u.observed_contribution < cfg.low_contribution]
    simple = (traj.task.difficulty or "").lower() == "simple"
    if len(agents) >= cfg.max_agents_simple and (simple or (len(low_value) / max(1, len(agents))) >= 0.4):
        out.append(Diagnostic(
            "over_agentization", "info" if not simple else "warning", "Over-agentization",
            f"{len(agents)} agents were invoked" + (" for a task marked simple" if simple else "") +
            f"; {len(low_value)} showed observed contribution below {_fmt_pct(cfg.low_contribution)} "
            f"({', '.join(low_value) or 'none'}). Cost: " +
            (f"${eff.cost_usd:.2f}" if eff.cost_usd is not None else "unknown") + ".",
            "Additional agents may not be producing enough value to justify their cost.",
            "Ablate the low-contribution agents and compare success and cost against the full swarm.",
            agents=[], evidence={"agents": len(agents), "low_value_agents": low_value,
                                 "cost_usd": eff.cost_usd, "tokens": traj.total_tokens}, run_ids=[rid]))

    # 13.4 under-specialization ----------------------------------------------
    for pair_key, overlap in coord.redundancy.pair_overlap.items():
        a, b = pair_key.split("|", 1)
        if overlap >= cfg.under_specialization_overlap:
            out.append(Diagnostic(
                "under_specialization", "warning", "Under-specialization",
                f"{a} and {b} performed {_fmt_pct(overlap)} overlapping work (tool calls / artifact content).",
                "The two agents' roles are not distinct enough; they may be duplicating effort.",
                f"Partition scope between {a} and {b} before execution, or merge them into one agent.",
                agents=[a, b], evidence={"overlap": round(overlap, 3)}, run_ids=[rid]))

    # 13.5 coordinator bottleneck --------------------------------------------
    com = coord.communication
    hub = com.hub_agent
    if hub and com.messages >= cfg.min_messages_for_hub and com.hub_share >= cfg.hub_message_share:
        hub_tokens = util[hub].token_share if hub in util else 0.0
        serial = hub in coord.dependency.serial_bottlenecks
        others_wait = sum(a.idle_time_s for aid, a in eff.per_agent.items() if aid != hub)
        cp_share = util[hub].critical_path_s if hub in util else 0.0
        cp_total = max(1e-9, eff.critical_path_s)
        if hub_tokens >= cfg.hub_token_share or serial or cp_share / cp_total >= 0.3:
            out.append(Diagnostic(
                "coordinator_bottleneck", "warning", "Coordinator bottleneck",
                f"{_fmt_pct(com.hub_share)} of messages pass through {hub}; it accounts for "
                f"{_fmt_pct(hub_tokens)} of tokens and {cp_share:.1f}s of the {eff.critical_path_s:.1f}s critical path"
                + ("; no other agent pair communicates around it" if serial else "") + ".",
                "Forcing execution through one agent reduces parallelism, adds latency and creates a "
                "single point of failure.",
                "Allow producer agents to communicate directly with their consumers and re-measure the critical path.",
                agents=[hub], evidence={"message_share": round(com.hub_share, 3), "token_share": round(hub_tokens, 3),
                                        "critical_path_s": round(cp_share, 2), "serial_bottleneck": serial,
                                        "others_waiting_s": round(others_wait, 2)}, run_ids=[rid]))

    # 13.6 premature synthesis / late information ----------------------------
    art_by_id = traj.artifacts
    for span in traj.spans.values():
        span_calls = [e for e in traj.llm_calls() if e.span_id == span.span_id]
        if not span_calls:
            continue
        first = min(e.timestamp for e in span_calls)
        last_end = max(e.end_timestamp for e in span_calls)
        consumed: Set[str] = set()
        for e in traj.events:
            if e.span_id == span.span_id:
                consumed.update(e.input_artifacts)
        late_arts = [art_by_id[a] for a in consumed if a in art_by_id and art_by_id[a].produced_at > first + 1e-9
                     and art_by_id[a].producer != span.agent_id]
        late_msgs = [m for m in traj.messages() if m.receiver == span.agent_id and m.status in ("success", "injected")
                     and span.start <= m.timestamp <= (span.end or m.timestamp) and m.timestamp > last_end + 1e-9]
        is_synth = (span.role or "").lower() in cfg.synthesizer_roles
        if late_arts:
            out.append(Diagnostic(
                "premature_synthesis", "critical" if is_synth else "warning",
                "Premature synthesis" if is_synth else "Started before inputs were ready",
                f"{span.agent_id} began reasoning at t={first - (traj.started_at or 0):.1f}s but "
                f"{len(late_arts)} of its inputs ({', '.join(a.name for a in late_arts)}) were produced later.",
                "The agent may have reasoned without required information and either wasted a call or "
                "produced an output missing that information.",
                f"Make {span.agent_id} wait for all upstream artifacts before its first LLM call.",
                agents=[span.agent_id], evidence={"late_inputs": [a.name for a in late_arts]}, run_ids=[rid]))
        if late_msgs:
            out.append(Diagnostic(
                "late_information", "warning", "Information arrived too late",
                f"{span.agent_id} received {len(late_msgs)} message(s) from "
                f"{', '.join(sorted({m.sender for m in late_msgs if m.sender}))} after its last LLM call.",
                "The message could not have influenced the agent's output.",
                "Re-order execution so the sender completes before the receiver starts, or drop the message.",
                agents=[span.agent_id], evidence={"late_messages": len(late_msgs)}, run_ids=[rid]))

    # 13.7 context bloat -----------------------------------------------------
    cb = coord.context_bloat
    if cb.bloated_calls:
        worst = cb.worst[0]
        bloated_agents = sorted({w["agent"] for w in cb.worst if w.get("agent")})
        out.append(Diagnostic(
            "context_bloat", "warning", "Context bloat",
            f"{cb.bloated_calls} LLM call(s) received far more context than they used; worst: {worst['agent']} "
            f"received {worst['input_tokens']:,} tokens with ~{worst['relevant_tokens']:,} relevant "
            f"({_fmt_pct(worst['ratio'])} excess). Excess tokens: {cb.excess_tokens:,}.",
            "Oversized context raises cost and latency and can degrade attention on relevant material.",
            "Pass only the artifacts the agent needs (or summaries of them) instead of full history.",
            agents=bloated_agents, evidence={"bloated_calls": cb.bloated_calls, "excess_tokens": cb.excess_tokens,
                                             "worst": cb.worst[:3]}, run_ids=[rid]))

    # 13.8 dead / zero-value agent -------------------------------------------
    for aid, u in util.items():
        if u.dead:
            a_eff = eff.per_agent.get(aid)
            out.append(Diagnostic(
                "dead_agent", "warning", "Zero-value agent",
                f"{aid} performed {u.actions} action(s) and {u.tokens:,} tokens "
                f"({_fmt_pct(u.token_share)} of total) but nothing it produced was used downstream or "
                f"appeared in the final output.",
                "The agent's output does not influence the result; it may be removable without loss.",
                f"Ablate {aid} and compare task success; if unchanged, remove it or fix the consumer that ignores it.",
                agents=[aid], evidence={"tokens": u.tokens, "token_share": round(u.token_share, 3),
                                        "actions": u.actions,
                                        "cost_usd": a_eff.cost_usd if a_eff else None}, run_ids=[rid]))

    # extra: serialized execution --------------------------------------------
    par = coord.parallelism
    if par.opportunity >= cfg.parallelism_opportunity and par.total_work_s > 0:
        out.append(Diagnostic(
            "serialized_execution", "info", "Unexploited parallelism",
            f"Dependencies allow {_fmt_pct(par.potential_parallel_fraction)} of work to overlap but only "
            f"{_fmt_pct(par.actual_parallel_fraction)} did (wall clock {par.wall_clock_s:.1f}s vs critical path "
            f"{par.critical_path_s:.1f}s).",
            "Independent agents are being run sequentially.",
            "Schedule independent agents concurrently; check rate limits and shared resources first.",
            agents=[], evidence={"opportunity": round(par.opportunity, 3), "wall_clock_s": round(par.wall_clock_s, 2),
                                 "critical_path_s": round(par.critical_path_s, 2)}, run_ids=[rid]))

    # extra: communication overhead ------------------------------------------
    if com.communication_ratio >= cfg.communication_ratio_warning and com.messages >= 3:
        out.append(Diagnostic(
            "communication_overhead", "info", "High communication overhead",
            f"Message tokens equal {_fmt_pct(com.communication_ratio)} of LLM tokens across {com.messages} messages "
            f"({com.repeated_messages} repeated, {com.broadcast_messages} broadcast).",
            "A large share of activity is spent moving information rather than producing it.",
            "Send references to artifacts instead of copies, and remove repeated or broadcast messages.",
            agents=[], evidence={"communication_ratio": round(com.communication_ratio, 3),
                                 "repeated": com.repeated_messages, "broadcast": com.broadcast_messages},
            run_ids=[rid]))

    out.sort(key=lambda d: -SEVERITY_ORDER.get(d.severity, 0))
    return out
