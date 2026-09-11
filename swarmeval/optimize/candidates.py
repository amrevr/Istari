"""Phase 4 -- from diagnosis to testable architecture hypotheses.

``propose`` turns diagnostics into candidate interventions with *predicted*
effects (heuristics derived from the observed trajectory).  ``validate``
runs the candidates as counterfactual experiments and compares measured
against predicted.  A candidate is never presented as an improvement until
it has been measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..causal.experiments import Comparison, CounterfactualRunner, compare
from ..causal.interventions import DisableChannel, Intervention, RemoveAgent, SetParam
from ..evaluate import EvaluationResult


@dataclass
class Candidate:
    name: str
    interventions: List[Intervention]
    rationale: str
    source_smell: str
    predicted: Dict[str, Optional[float]] = field(default_factory=dict)   # cost_pct, latency_pct, success_pp
    measured: Optional[Comparison] = None
    verdict: str = "untested hypothesis"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "interventions": [i.name for i in self.interventions], "rationale": self.rationale,
                "source_smell": self.source_smell, "predicted": self.predicted,
                "measured": self.measured.to_dict() if self.measured else None, "verdict": self.verdict}


@dataclass
class ValidationReport:
    baseline: EvaluationResult
    candidates: List[Candidate]

    def accepted(self) -> List[Candidate]:
        return [c for c in self.candidates if c.verdict.startswith("accept")]

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline": self.baseline.summary(), "candidates": [c.to_dict() for c in self.candidates]}

    def table(self) -> str:
        lines = [f"CANDIDATE ARCHITECTURES vs {self.baseline.label}",
                 f"  {'candidate':<30}{'pred Δcost':>11}{'pred Δlat':>10}{'Δ success':>11}{'Δ cost':>9}{'Δ lat':>8}  verdict"]
        for c in self.candidates:
            m = c.measured
            ms = f"{m.success_delta*100:+.1f}pp" if m else "   —"
            mc = _s(m.cost_delta_pct) if m else "—"
            ml = _s(m.latency_delta_pct) if m else "—"
            lines.append(f"  {c.name[:29]:<30}{_s(c.predicted.get('cost_pct')):>11}{_s(c.predicted.get('latency_pct')):>10}"
                         f"{ms:>11}{mc:>9}{ml:>8}  {c.verdict}")
        return "\n".join(lines)


def _s(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:+.0f}%"


def propose(result: EvaluationResult, extra_params: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Candidate]:
    """Generate candidate interventions from diagnostics.

    ``extra_params`` maps a smell name to swarm-specific params to set when
    that smell is present, e.g. ``{"coordinator_bottleneck": {"direct_channels": True}}``.
    Without it, only generic interventions (agent removal, channel changes)
    are proposed for smells that have a generic remedy."""
    extra_params = extra_params or {}
    cands: List[Candidate] = []
    eff, coord = result.efficiency, result.coordination
    total_tokens = max(1.0, eff.tokens.mean)
    total_cost = eff.cost_usd.mean or 0.0
    wall = max(1e-9, eff.wall_clock_s.mean)
    seen: set = set()
    for d in result.diagnostics:
        if d.smell == "dead_agent" and d.agents:
            a = d.agents[0]
            if ("remove", a) in seen:
                continue
            seen.add(("remove", a))
            pa = eff.per_agent.get(a, {})
            ua = coord.utilization.get(a, {})
            cost_pct = -(pa.get("cost_usd", 0.0) / total_cost) if total_cost else -(pa.get("tokens", 0.0) / total_tokens)
            lat_pct = -(ua.get("critical_path_s", 0.0) / wall)
            cands.append(Candidate(f"remove {a}", [RemoveAgent(a)],
                                   f"{a} output unused in {d.prevalence*100:.0f}% of runs", d.smell,
                                   {"cost_pct": cost_pct, "latency_pct": lat_pct, "success_pp": 0.0}))
        elif d.smell == "under_specialization" and len(d.agents) == 2:
            a, b = d.agents
            # remove the lower-contribution one
            ua, ub = coord.utilization.get(a, {}), coord.utilization.get(b, {})
            victim = a if ua.get("observed_contribution", 0) <= ub.get("observed_contribution", 0) else b
            if ("remove", victim) in seen:
                continue
            seen.add(("remove", victim))
            pv = eff.per_agent.get(victim, {})
            cost_pct = -(pv.get("cost_usd", 0.0) / total_cost) if total_cost else -(pv.get("tokens", 0.0) / total_tokens)
            cands.append(Candidate(f"remove {victim} (overlaps {a if victim == b else b})", [RemoveAgent(victim)],
                                   f"{a} and {b} overlap {d.evidence.get('overlap', 0)*100:.0f}%", d.smell,
                                   {"cost_pct": cost_pct, "latency_pct": 0.0, "success_pp": None}))
            if d.smell in extra_params:
                cands.append(Candidate(f"partition scope ({a}, {b})",
                                       [SetParam(k, v) for k, v in extra_params[d.smell].items()],
                                       "give overlapping agents distinct scopes", d.smell,
                                       {"cost_pct": 0.0, "latency_pct": 0.0, "success_pp": None}))
        elif d.smell == "coordinator_bottleneck" and d.agents:
            hub = d.agents[0]
            if d.smell in extra_params:
                cp = d.evidence.get("critical_path_s", 0.0) or 0.0
                ph = eff.per_agent.get(hub, {})
                cands.append(Candidate("direct channels (bypass hub)",
                                       [SetParam(k, v) for k, v in extra_params[d.smell].items()],
                                       f"{hub} relays {d.evidence.get('message_share', 0)*100:.0f}% of messages", d.smell,
                                       {"cost_pct": -(0.5 * ph.get("cost_usd", 0.0) / total_cost) if total_cost else None,
                                        "latency_pct": -(0.5 * cp / wall), "success_pp": 0.0}))
        elif d.smell == "agent_ping_pong" and len(d.agents) == 2:
            a, b = d.agents
            cands.append(Candidate(f"disable channel {b}->{a}", [DisableChannel(b, a)],
                                   "cut the revision loop", d.smell,
                                   {"cost_pct": -0.05, "latency_pct": -0.05, "success_pp": None}))
            if d.smell in extra_params:
                cands.append(Candidate("cap revision rounds",
                                       [SetParam(k, v) for k, v in extra_params[d.smell].items()],
                                       "limit ping-pong rounds", d.smell,
                                       {"cost_pct": -0.05, "latency_pct": -0.05, "success_pp": None}))
        elif d.smell in ("serialized_execution", "context_bloat", "communication_overhead") and d.smell in extra_params:
            opp = d.evidence.get("opportunity", 0.0) or 0.0
            cands.append(Candidate(f"fix {d.smell}", [SetParam(k, v) for k, v in extra_params[d.smell].items()],
                                   d.hypothesis, d.smell,
                                   {"cost_pct": -0.1 if d.smell != "serialized_execution" else 0.0,
                                    "latency_pct": -opp if d.smell == "serialized_execution" else 0.0, "success_pp": 0.0}))
    return cands


def validate(candidates: Sequence[Candidate], runner: CounterfactualRunner,
             success_tolerance_pp: float = 2.0, min_gain: float = 0.03) -> ValidationReport:
    """Run each candidate as a counterfactual and label it accepted, rejected
    or inconclusive.  Accepted = success not significantly worse (lower CI
    bound above -tolerance) AND cost or latency improved by at least
    ``min_gain``."""
    base = runner.baseline()
    for c in candidates:
        variant = runner.run(*c.interventions, label=c.name)
        m = compare(base, variant)
        c.measured = m
        lo = m.success_ci[0] * 100
        gain = min(m.cost_delta_pct if m.cost_delta_pct is not None else 0.0,
                   m.latency_delta_pct if m.latency_delta_pct is not None else 0.0)
        improved_success = m.success_ci[0] > 0
        succ = f"success {m.success_delta*100:+.1f} pp" + ("" if m.significant else " (n.s.)")
        if lo < -success_tolerance_pp and not improved_success:
            c.verdict = f"reject: {succ}"
        elif improved_success:
            c.verdict = f"accept: {succ}"
        elif gain <= -min_gain:
            which = "cost" if (m.cost_delta_pct or 0) <= gain else "latency"
            c.verdict = f"accept: {which} {gain*100:+.0f}%, {succ}"
        else:
            c.verdict = f"inconclusive: no measurable gain, {succ}"
    return ValidationReport(base, list(candidates))
