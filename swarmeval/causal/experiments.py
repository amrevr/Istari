"""Counterfactual experiments: ablation, failure injection, frontiers.

These produce *causal* evidence: the same swarm, benchmark and seeds are
re-run under an intervention and outcomes are compared pairwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..config import FailureSpec, SwarmConfig
from ..diagnostics.smells import SmellConfig
from ..evaluate import EvaluationResult, evaluate
from ..pricing import PricingTable
from ..recorder import Clock
from ..schema import Benchmark
from ..stats import mean, paired_bootstrap_delta
from .interventions import InjectFailure, Intervention, RemoveAgent, RemoveTool


# --------------------------------------------------------------------------- #
# Comparison of two evaluation results
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    baseline_label: str
    variant_label: str
    n_pairs: int
    success_delta: float
    success_ci: Tuple[float, float]
    score_delta: float
    score_ci: Tuple[float, float]
    cost_delta: float
    cost_delta_pct: Optional[float]
    cost_ci: Tuple[float, float]
    latency_delta: float
    latency_delta_pct: Optional[float]
    latency_ci: Tuple[float, float]
    critical_path_delta: float
    tokens_delta_pct: Optional[float]
    baseline: Optional[EvaluationResult] = None
    variant: Optional[EvaluationResult] = None

    @property
    def significant(self) -> bool:
        lo, hi = self.success_ci
        return lo > 0 or hi < 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ("baseline", "variant")}


def _pct(delta: float, base: float) -> Optional[float]:
    return (delta / base) if base else None


def compare(baseline: EvaluationResult, variant: EvaluationResult) -> Comparison:
    pairs = baseline.runs.paired_with(variant.runs)
    if pairs:
        b_succ = [1.0 if a.succeeded else 0.0 for a, _ in pairs]
        v_succ = [1.0 if b.succeeded else 0.0 for _, b in pairs]
        b_score = [a.score for a, _ in pairs]
        v_score = [b.score for _, b in pairs]
        b_cost = [a.cost_usd or 0.0 for a, _ in pairs]
        v_cost = [b.cost_usd or 0.0 for _, b in pairs]
        b_lat = [a.duration for a, _ in pairs]
        v_lat = [b.duration for _, b in pairs]
        b_tok = [float(a.total_tokens) for a, _ in pairs]
        v_tok = [float(b.total_tokens) for _, b in pairs]
    else:  # unpaired fallback
        b_succ = [1.0 if t.succeeded else 0.0 for t in baseline.runs]
        v_succ = [1.0 if t.succeeded else 0.0 for t in variant.runs]
        b_score = [t.score for t in baseline.runs]
        v_score = [t.score for t in variant.runs]
        b_cost = [t.cost_usd or 0.0 for t in baseline.runs]
        v_cost = [t.cost_usd or 0.0 for t in variant.runs]
        b_lat = [t.duration for t in baseline.runs]
        v_lat = [t.duration for t in variant.runs]
        b_tok = [float(t.total_tokens) for t in baseline.runs]
        v_tok = [float(t.total_tokens) for t in variant.runs]
    sd, sci = paired_bootstrap_delta(b_succ, v_succ)
    scd, scci = paired_bootstrap_delta(b_score, v_score)
    cd, cci = paired_bootstrap_delta(b_cost, v_cost)
    ld, lci = paired_bootstrap_delta(b_lat, v_lat)
    cp_delta = variant.efficiency.critical_path_s.mean - baseline.efficiency.critical_path_s.mean
    return Comparison(baseline.label, variant.label, len(pairs), sd, sci, scd, scci, cd, _pct(cd, mean(b_cost)), cci,
                      ld, _pct(ld, mean(b_lat)), lci, cp_delta,
                      _pct(mean(v_tok) - mean(b_tok), mean(b_tok)), baseline, variant)


# --------------------------------------------------------------------------- #
# Counterfactual runner
# --------------------------------------------------------------------------- #
class CounterfactualRunner:
    """Runs the same swarm/benchmark/seeds under different interventions."""

    def __init__(self, swarm: Any, benchmark: Benchmark, trials: int = 3, seed: int = 0,
                 base_config: Optional[SwarmConfig] = None, clock_factory: Optional[Callable[[], Clock]] = None,
                 pricing: Optional[PricingTable] = None, evaluator: Any = None,
                 smell_config: Optional[SmellConfig] = None, progress: bool = False) -> None:
        self.swarm = swarm
        self.benchmark = benchmark
        self.trials = trials
        self.seed = seed
        self.base_config = base_config or SwarmConfig()
        self.clock_factory = clock_factory
        self.pricing = pricing
        self.evaluator = evaluator
        self.smell_config = smell_config
        self.progress = progress
        self.results: Dict[str, EvaluationResult] = {}

    def run(self, *interventions: Intervention, label: Optional[str] = None) -> EvaluationResult:
        label = label or (" + ".join(i.name for i in interventions) if interventions else self.base_config.label)
        if label in self.results:
            return self.results[label]
        result = evaluate(self.swarm, self.benchmark, trials=self.trials, seed=self.seed, config=self.base_config,
                          interventions=interventions, evaluator=self.evaluator, clock_factory=self.clock_factory,
                          pricing=self.pricing, smell_config=self.smell_config, label=label, progress=self.progress)
        self.results[label] = result
        return result

    def baseline(self) -> EvaluationResult:
        return self.run(label=self.base_config.label)

    def compare(self, *interventions: Intervention, label: Optional[str] = None) -> Comparison:
        return compare(self.baseline(), self.run(*interventions, label=label))


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
@dataclass
class AblationRow:
    component: str
    kind: str                        # agent | tool | channel | ...
    comparison: Comparison
    invocation_rate: float = 1.0     # how often the component appeared in baseline runs
    observed_contribution: Optional[float] = None
    run_error_rate: float = 0.0      # runs that crashed under the ablation

    @property
    def marginal_success(self) -> float:
        """Marginal effect of *having* the component = -(delta when removed)."""
        return -self.comparison.success_delta

    def to_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "kind": self.kind, "invocation_rate": self.invocation_rate,
                "observed_contribution": self.observed_contribution, "run_error_rate": self.run_error_rate,
                "marginal_success": self.marginal_success, "comparison": self.comparison.to_dict()}


@dataclass
class AblationReport:
    baseline: EvaluationResult
    rows: List[AblationRow] = field(default_factory=list)

    def ranking(self) -> List[AblationRow]:
        return sorted(self.rows, key=lambda r: -r.marginal_success)

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline": self.baseline.summary(), "rows": [r.to_dict() for r in self.ranking()]}

    def table(self) -> str:
        from ..report.text import render_ablation
        return render_ablation(self)


def ablate_agents(runner: CounterfactualRunner, agents: Optional[Sequence[str]] = None) -> AblationReport:
    base = runner.baseline()
    agents = list(agents) if agents is not None else base.agents
    report = AblationReport(base)
    for a in agents:
        variant = runner.run(RemoveAgent(a))
        u = base.coordination.utilization.get(a, {})
        errors = sum(1 for t in variant.runs if t.status != "success") / max(1, len(variant.runs))
        report.rows.append(AblationRow(a, "agent", compare(base, variant), u.get("invocation_rate", 0.0),
                                       u.get("observed_contribution"), errors))
    return report


def ablate_tools(runner: CounterfactualRunner, tools: Optional[Sequence[str]] = None) -> AblationReport:
    base = runner.baseline()
    tools = list(tools) if tools is not None else base.tools
    report = AblationReport(base)
    for t in tools:
        variant = runner.run(RemoveTool(t))
        errors = sum(1 for x in variant.runs if x.status != "success") / max(1, len(variant.runs))
        report.rows.append(AblationRow(t, "tool", compare(base, variant), 1.0, None, errors))
    return report


def ablate(runner: CounterfactualRunner, variants: Dict[str, Sequence[Intervention]]) -> AblationReport:
    """Generic ablation over named intervention sets (architecture ablation)."""
    base = runner.baseline()
    report = AblationReport(base)
    for name, ivs in variants.items():
        variant = runner.run(*ivs, label=name)
        errors = sum(1 for x in variant.runs if x.status != "success") / max(1, len(variant.runs))
        report.rows.append(AblationRow(name, "architecture", compare(base, variant), 1.0, None, errors))
    return report


# --------------------------------------------------------------------------- #
# Failure injection / resilience
# --------------------------------------------------------------------------- #
@dataclass
class ResilienceRow:
    failure: str
    comparison: Comparison
    resilience: float             # variant success / baseline success (1.0 = unaffected)
    quality_retention: float      # variant score / baseline score
    run_error_rate: float
    recovery_cost_pct: Optional[float]
    recovery_latency_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"failure": self.failure, "resilience": self.resilience, "quality_retention": self.quality_retention,
                "run_error_rate": self.run_error_rate, "recovery_cost_pct": self.recovery_cost_pct,
                "recovery_latency_pct": self.recovery_latency_pct, "comparison": self.comparison.to_dict()}


@dataclass
class ResilienceReport:
    baseline: EvaluationResult
    rows: List[ResilienceRow] = field(default_factory=list)

    @property
    def resilience(self) -> float:
        return mean([r.resilience for r in self.rows]) if self.rows else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline": self.baseline.summary(), "resilience": self.resilience,
                "rows": [r.to_dict() for r in self.rows]}

    def table(self) -> str:
        from ..report.text import render_resilience
        return render_resilience(self)


def inject_failures(runner: CounterfactualRunner,
                    failures: Iterable[Union[InjectFailure, FailureSpec]]) -> ResilienceReport:
    base = runner.baseline()
    report = ResilienceReport(base)
    for f in failures:
        iv = f if isinstance(f, InjectFailure) else InjectFailure(f.kind, f.target, f.probability, f.on_call,
                                                                    f.delay_s, f.payload, f.label)
        variant = runner.run(iv)
        comp = compare(base, variant)
        b_s = base.success_rate
        res = (variant.success_rate / b_s) if b_s > 0 else (1.0 if variant.success_rate == 0 else 0.0)
        b_q = base.performance.score.mean
        qret = (variant.performance.score.mean / b_q) if b_q > 0 else 1.0
        errors = sum(1 for t in variant.runs if t.status != "success") / max(1, len(variant.runs))
        report.rows.append(ResilienceRow(iv.spec.key, comp, min(1.5, res), min(1.5, qret), errors,
                                         comp.cost_delta_pct, comp.latency_delta_pct))
    return report


# --------------------------------------------------------------------------- #
# Efficiency frontier & structured experiments
# --------------------------------------------------------------------------- #
@dataclass
class FrontierPoint:
    label: str
    success: float
    cost: float
    latency: float
    on_frontier: bool
    result: Optional[EvaluationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "success": self.success, "cost": self.cost, "latency": self.latency,
                "on_frontier": self.on_frontier}


def frontier(results: Iterable[EvaluationResult], cost_key: str = "per_run") -> List[FrontierPoint]:
    """Pareto frontier over (maximise success, minimise cost)."""
    pts = [FrontierPoint(r.label, r.success_rate, r.cost.get(cost_key) or 0.0, r.latency["wall_clock"], False, r)
           for r in results]
    for p in pts:
        dominated = any((q.success >= p.success and q.cost <= p.cost) and (q.success > p.success or q.cost < p.cost)
                        for q in pts if q is not p)
        p.on_frontier = not dominated
    return sorted(pts, key=lambda p: (p.cost, -p.success))


def intelligence_vs_coordination(runner: CounterfactualRunner,
                                 configurations: Dict[str, Sequence[Intervention]]) -> Dict[str, EvaluationResult]:
    """Run a set of named configurations (e.g. strong single agent, strong
    swarm, weak swarm, weak single agent) and return their results keyed by
    name.  Combine with ``frontier`` or ``compare`` for analysis."""
    out: Dict[str, EvaluationResult] = {}
    for name, ivs in configurations.items():
        out[name] = runner.run(*ivs, label=name)
    return out
