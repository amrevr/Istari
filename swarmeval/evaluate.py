"""Top-level evaluation: run a swarm on a benchmark and analyse the result."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .config import SwarmConfig
from .diagnostics.engine import aggregate_diagnostics
from .diagnostics.smells import Diagnostic, SmellConfig, detect_smells
from .graph.builder import ExecutionGraph, build_graph
from .metrics.coordination import CoordinationMetrics, compute_coordination
from .metrics.efficiency import EfficiencyMetrics, compute_efficiency
from .metrics.performance import PerformanceMetrics, compute_performance
from .pricing import PricingTable
from .recorder import Clock
from .runner import RunOptions, RunSet, SwarmRunner
from .schema import Benchmark, Trajectory
from .stats import Summary, mean, summarize
from .validity import check_leakage, repeatability


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
@dataclass
class RunAnalysis:
    trajectory: Trajectory
    graph: ExecutionGraph
    efficiency: EfficiencyMetrics
    coordination: CoordinationMetrics
    diagnostics: List[Diagnostic]

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.trajectory.run_id, "task_id": self.trajectory.task_id,
                "trial": self.trajectory.trial, "success": self.trajectory.succeeded,
                "score": self.trajectory.score, "status": self.trajectory.status,
                "efficiency": self.efficiency.to_dict(), "coordination": self.coordination.to_dict(),
                "graph": self.graph.to_dict(), "diagnostics": [d.to_dict() for d in self.diagnostics]}


def analyze_trajectory(traj: Trajectory, smell_config: Optional[SmellConfig] = None) -> RunAnalysis:
    graph = build_graph(traj)
    eff = compute_efficiency(traj, graph)
    cfg = smell_config or SmellConfig()
    coord = compute_coordination(traj, graph, cfg.bloat_ratio, cfg.bloat_min_tokens)
    diags = detect_smells(traj, graph, eff, coord, cfg)
    return RunAnalysis(traj, graph, eff, coord, diags)


# --------------------------------------------------------------------------- #
# Aggregated summaries
# --------------------------------------------------------------------------- #
@dataclass
class EfficiencySummary:
    tokens: Summary
    input_tokens: Summary
    output_tokens: Summary
    llm_calls: Summary
    tool_calls: Summary
    messages: Summary
    cost_usd: Summary
    cost_complete: bool
    cost_total: float
    cost_per_success: Optional[float]
    wall_clock_s: Summary
    critical_path_s: Summary
    total_work_s: Summary
    agent_compute_s: Summary
    waiting_time_s: Summary
    errors: Summary
    per_agent: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}
        return d


@dataclass
class CoordinationSummary:
    communication_ratio: Summary
    coordination_fraction: Summary
    information_fraction: Summary
    useful_work_fraction: Summary
    redundancy_ratio: Summary
    redundant_token_fraction: Summary
    accidental_redundancy: Summary
    intentional_redundancy: Summary
    actual_parallelism: Summary
    potential_parallelism: Summary
    parallel_opportunity: Summary
    depth: Summary
    density: Summary
    hub_agent: Optional[str]
    hub_share: Summary
    context_bloat_calls: Summary
    context_bloat_ratio: Summary
    utilization: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    highest_overlap: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Summary) else v) for k, v in self.__dict__.items()}


def summarize_efficiency(analyses: Sequence[RunAnalysis]) -> EfficiencySummary:
    effs = [a.efficiency for a in analyses]
    trajs = [a.trajectory for a in analyses]
    S = lambda xs: summarize(list(xs), keep_values=False)  # noqa: E731
    cost_complete = all(e.cost_complete for e in effs)
    costs = [e.cost_usd or 0.0 for e in effs]
    total_cost = sum(costs)
    successes = sum(1 for t in trajs if t.succeeded)
    per_agent: Dict[str, Dict[str, float]] = {}
    agent_runs: Dict[str, int] = Counter()
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for e in effs:
        for aid, ae in e.per_agent.items():
            agent_runs[aid] += 1
            acc[aid]["tokens"].append(ae.total_tokens)
            acc[aid]["input_tokens"].append(ae.input_tokens)
            acc[aid]["output_tokens"].append(ae.output_tokens)
            acc[aid]["cost_usd"].append(ae.cost_usd or 0.0)
            acc[aid]["llm_calls"].append(ae.llm_calls)
            acc[aid]["tool_calls"].append(ae.tool_calls)
            acc[aid]["self_time_s"].append(ae.self_time_s)
            acc[aid]["idle_time_s"].append(ae.idle_time_s)
            acc[aid]["errors"].append(ae.errors)
    n = max(1, len(effs))
    for aid, d in acc.items():
        per_agent[aid] = {k: mean(v) for k, v in d.items()}
        per_agent[aid]["invocation_rate"] = agent_runs[aid] / n
        per_agent[aid]["total_cost_usd"] = sum(d["cost_usd"])
    return EfficiencySummary(
        tokens=S(e.total_tokens for e in effs), input_tokens=S(e.input_tokens for e in effs),
        output_tokens=S(e.output_tokens for e in effs), llm_calls=S(e.llm_calls for e in effs),
        tool_calls=S(e.tool_calls for e in effs), messages=S(e.messages for e in effs),
        cost_usd=S(costs), cost_complete=cost_complete, cost_total=total_cost,
        cost_per_success=(total_cost / successes) if successes else None,
        wall_clock_s=S(e.wall_clock_s for e in effs), critical_path_s=S(e.critical_path_s for e in effs),
        total_work_s=S(e.total_work_s for e in effs), agent_compute_s=S(e.agent_compute_s for e in effs),
        waiting_time_s=S(e.waiting_time_s for e in effs), errors=S(e.errors for e in effs), per_agent=per_agent)


def summarize_coordination(analyses: Sequence[RunAnalysis]) -> CoordinationSummary:
    cs = [a.coordination for a in analyses]
    S = lambda xs: summarize(list(xs), keep_values=False)  # noqa: E731
    hubs = Counter(c.communication.hub_agent for c in cs if c.communication.hub_agent)
    hub = hubs.most_common(1)[0][0] if hubs else None
    util: Dict[str, Dict[str, Any]] = {}
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    roles: Dict[str, Optional[str]] = {}
    n = max(1, len(cs))
    for c in cs:
        for aid, u in c.utilization.items():
            roles[aid] = u.role
            acc[aid]["token_share"].append(u.token_share)
            acc[aid]["tokens"].append(u.tokens)
            acc[aid]["actions"].append(u.actions)
            acc[aid]["observed_contribution"].append(u.observed_contribution)
            acc[aid]["downstream_uses"].append(u.downstream_uses)
            acc[aid]["final_output_overlap"].append(u.final_output_overlap)
            acc[aid]["dead"].append(1.0 if u.dead else 0.0)
            acc[aid]["critical_path_s"].append(u.critical_path_s)
            acc[aid]["messages_sent"].append(u.messages_sent)
            acc[aid]["artifacts_produced"].append(u.artifacts_produced)
            acc[aid]["errors"].append(u.errors)
    for aid, d in acc.items():
        util[aid] = {k: mean(v) for k, v in d.items()}
        util[aid]["invocation_rate"] = len(d["tokens"]) / n
        util[aid]["dead_rate"] = util[aid].pop("dead")
        util[aid]["role"] = roles.get(aid)
        tools: List[str] = []
        for c in cs:
            u = c.utilization.get(aid)
            if u:
                for t in u.tools_used:
                    if t not in tools:
                        tools.append(t)
        util[aid]["tools_used"] = tools
    overlaps = [c.redundancy.highest_overlap for c in cs if c.redundancy.highest_overlap]
    highest = None
    if overlaps:
        pair_scores: Dict[tuple, List[float]] = defaultdict(list)
        for a, b, s in overlaps:
            pair_scores[(a, b)].append(s)
        (a, b), scores = max(pair_scores.items(), key=lambda kv: mean(kv[1]))
        highest = {"agents": [a, b], "overlap": mean(scores)}
    return CoordinationSummary(
        communication_ratio=S(c.communication.communication_ratio for c in cs),
        coordination_fraction=S(c.communication.coordination_fraction for c in cs),
        information_fraction=S(c.communication.information_transfer_fraction for c in cs),
        useful_work_fraction=S(c.communication.useful_work_fraction for c in cs),
        redundancy_ratio=S(c.redundancy.ratio for c in cs),
        redundant_token_fraction=S(c.redundancy.redundant_token_fraction for c in cs),
        accidental_redundancy=S(c.redundancy.accidental for c in cs),
        intentional_redundancy=S(c.redundancy.intentional for c in cs),
        actual_parallelism=S(c.parallelism.actual_parallel_fraction for c in cs),
        potential_parallelism=S(c.parallelism.potential_parallel_fraction for c in cs),
        parallel_opportunity=S(c.parallelism.opportunity for c in cs),
        depth=S(c.dependency.depth for c in cs), density=S(c.communication.density for c in cs),
        hub_agent=hub, hub_share=S(c.communication.hub_share for c in cs),
        context_bloat_calls=S(c.context_bloat.bloated_calls for c in cs),
        context_bloat_ratio=S(c.context_bloat.mean_bloat_ratio for c in cs),
        utilization=util, highest_overlap=highest)


# --------------------------------------------------------------------------- #
# Evaluation result
# --------------------------------------------------------------------------- #
class EvaluationResult:
    def __init__(self, runs: RunSet, analyses: List[RunAnalysis], performance: PerformanceMetrics,
                 efficiency: EfficiencySummary, coordination: CoordinationSummary,
                 diagnostics: List[Diagnostic], leakage: List[str], smell_config: Optional[SmellConfig] = None) -> None:
        self.runs = runs
        self.analyses = analyses
        self.performance = performance
        self.efficiency = efficiency
        self.coordination = coordination
        self.diagnostics = diagnostics
        self.leakage = leakage
        self.smell_config = smell_config or SmellConfig()
        self.repeatability = repeatability(runs.trajectories)

    # -- spec-style accessors -------------------------------------------------
    @property
    def label(self) -> str:
        return self.runs.label

    @property
    def config(self) -> SwarmConfig:
        return self.runs.config

    @property
    def benchmark(self) -> str:
        return self.runs.benchmark_name

    @property
    def success_rate(self) -> float:
        return self.performance.success_rate.mean

    @property
    def cost(self) -> Dict[str, Optional[float]]:
        return {"total": self.efficiency.cost_total, "per_run": self.efficiency.cost_usd.mean,
                "per_success": self.efficiency.cost_per_success, "complete": self.efficiency.cost_complete}

    @property
    def latency(self) -> Dict[str, float]:
        return {"wall_clock": self.efficiency.wall_clock_s.mean, "critical_path": self.efficiency.critical_path_s.mean,
                "agent_compute": self.efficiency.agent_compute_s.mean, "waiting": self.efficiency.waiting_time_s.mean}

    @property
    def smells(self) -> List[str]:
        seen: List[str] = []
        for d in self.diagnostics:
            if d.smell not in seen:
                seen.append(d.smell)
        return seen

    @property
    def agent_contribution(self) -> Dict[str, float]:
        """*Observed* contribution (0..1) -- how much of each agent's output was
        used downstream or appeared in the final output.  For causal (marginal)
        contribution use ``swarmeval.causal.ablate_agents``."""
        return {a: round(u["observed_contribution"], 3) for a, u in self.coordination.utilization.items()}

    @property
    def agents(self) -> List[str]:
        return list(self.coordination.utilization.keys())

    @property
    def tools(self) -> List[str]:
        out: List[str] = []
        for u in self.coordination.utilization.values():
            for t in u.get("tools_used", []):
                if t not in out:
                    out.append(t)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label, "benchmark": self.benchmark, "runs": len(self.runs),
            "success_rate": round(self.success_rate, 4),
            "success_ci": [round(self.performance.success_rate.ci_low, 4), round(self.performance.success_rate.ci_high, 4)],
            "score": round(self.performance.score.mean, 4),
            "cost": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.cost.items()},
            "latency": {k: round(v, 3) for k, v in self.latency.items()},
            "coordination": {
                "communication_ratio": round(self.coordination.communication_ratio.mean, 3),
                "redundancy_ratio": round(self.coordination.redundancy_ratio.mean, 3),
                "parallelism": round(self.coordination.actual_parallelism.mean, 3),
                "potential_parallelism": round(self.coordination.potential_parallelism.mean, 3),
            },
            "smells": self.smells,
            "agent_contribution_observed": self.agent_contribution,
        }

    def to_dict(self, include_runs: bool = False) -> Dict[str, Any]:
        d = {
            "summary": self.summary(),
            "config": self.config.to_dict(),
            "performance": self.performance.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "coordination": self.coordination.to_dict(),
            "diagnostics": [x.to_dict() for x in self.diagnostics],
            "leakage_warnings": self.leakage,
            "repeatability": self.repeatability,
            "per_run": [{"run_id": a.trajectory.run_id, "task_id": a.trajectory.task_id, "trial": a.trajectory.trial,
                         "success": a.trajectory.succeeded, "score": a.trajectory.score,
                         "cost_usd": a.efficiency.cost_usd, "tokens": a.efficiency.total_tokens,
                         "wall_clock_s": a.efficiency.wall_clock_s, "critical_path_s": a.efficiency.critical_path_s,
                         "smells": [x.smell for x in a.diagnostics]} for a in self.analyses],
        }
        if include_runs:
            d["runs"] = self.runs.to_dict()
        return d

    def save(self, path: str, include_runs: bool = True) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(include_runs=include_runs), f, indent=1, default=str)

    @classmethod
    def load(cls, path: str, smell_config: Optional[SmellConfig] = None) -> "EvaluationResult":
        with open(path) as f:
            d = json.load(f)
        if "runs" not in d:
            raise ValueError("file has no trajectories; save with include_runs=True to reload")
        return analyze_runs(RunSet.from_dict(d["runs"]), smell_config)

    def report(self, **kwargs: Any) -> str:
        from .report.text import render_text
        return render_text(self, **kwargs)

    def to_html(self, path: Optional[str] = None, **kwargs: Any) -> str:
        from .report.html import render_html
        return render_html(self, path=path, **kwargs)

    def __repr__(self) -> str:
        return (f"EvaluationResult(label={self.label!r}, runs={len(self.runs)}, success={self.success_rate:.3f}, "
                f"cost=${self.efficiency.cost_total:.2f}, smells={self.smells})")


def analyze_runs(runs: RunSet, smell_config: Optional[SmellConfig] = None) -> EvaluationResult:
    cfg = smell_config or SmellConfig()
    analyses = [analyze_trajectory(t, cfg) for t in runs.trajectories]
    performance = compute_performance(runs.trajectories)
    efficiency = summarize_efficiency(analyses)
    coordination = summarize_coordination(analyses)
    diagnostics = aggregate_diagnostics([a.diagnostics for a in analyses], len(analyses))
    leakage: List[str] = []
    for t in runs.trajectories:
        leakage.extend(check_leakage(t))
    return EvaluationResult(runs, analyses, performance, efficiency, coordination, diagnostics, leakage, cfg)


def evaluate(swarm: Any, benchmark: Benchmark, trials: int = 1, seed: int = 0,
             config: Optional[SwarmConfig] = None, interventions: Iterable[Any] = (),
             evaluator: Any = None, clock_factory: Optional[Callable[[], Clock]] = None,
             pricing: Optional[PricingTable] = None, smell_config: Optional[SmellConfig] = None,
             label: Optional[str] = None, progress: bool = False, raise_errors: bool = False,
             on_run: Optional[Callable[[Trajectory], None]] = None) -> EvaluationResult:
    """Run ``swarm`` on ``benchmark`` and return a full EvaluationResult.

    ``swarm`` may be a Swarm instance, a ``factory(config) -> Swarm`` or a
    bare ``run(task, ctx)`` function.  Pass ``interventions`` to evaluate a
    counterfactual configuration.
    """
    from .causal.interventions import apply_interventions
    cfg = apply_interventions(config or SwarmConfig(), list(interventions), label=label)
    if label:
        cfg.label = label
    opts = RunOptions(trials=trials, seed=seed, clock_factory=clock_factory, pricing=pricing, evaluator=evaluator,
                      progress=progress, raise_errors=raise_errors, on_run=on_run)
    runs = SwarmRunner(swarm, benchmark, cfg, opts).run()
    return analyze_runs(runs, smell_config)
