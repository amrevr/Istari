"""Human-readable text reports (terminal friendly)."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from ..stats import Summary


def bar(frac: float, width: int = 16) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _money(x: Optional[float], complete: bool = True) -> str:
    if x is None:
        return "unknown"
    s = f"${x:.2f}" if x >= 0.1 else f"${x:.4f}"
    return s if complete else s + "*"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _ci(s: Summary, pct: bool = False, unit: str = "", digits: int = 1) -> str:
    if s.n <= 1:
        return _pct(s.mean) if pct else f"{s.mean:.{digits}f}{unit}"
    return s.fmt(unit=unit, pct=pct, digits=digits)


def box(title: str, rows: Sequence[Any], width: int = 58) -> str:
    longest = max([len(r) + 3 for r in rows if isinstance(r, str)] + [width])
    inner = max(width, longest) - 2
    out = ["╭" + "─" * inner + "╮", "│" + title.center(inner) + "│", "├" + "─" * inner + "┤"]
    for row in rows:
        if row is None:
            out.append("├" + "─" * inner + "┤")
            continue
        if isinstance(row, tuple):
            label, value = row[0], row[1]
            flag = row[2] if len(row) > 2 else ""
            line = f" {label:<28}{value:>{inner - 32}} {flag:<2}"
        else:
            line = f" {row}"
        out.append("│" + line[:inner].ljust(inner) + "│")
    out.append("╰" + "─" * inner + "╯")
    return "\n".join(out)


def render_text(result: Any, max_diagnostics: int = 12, show_agents: bool = True, show_validity: bool = True) -> str:
    perf, eff, coord = result.performance, result.efficiency, result.coordination
    n_runs = len(result.runs)
    rows: List[Any] = [
        f"{result.label}  ·  {result.benchmark}  ·  {perf.n_tasks} tasks × {perf.trials} trials = {n_runs} runs",
        None,
        ("Task Success", _ci(perf.success_rate, pct=True)),
        ("Mean Score", _ci(perf.score, digits=3)),
    ]
    if perf.constraint_pass_rate:
        rows.append(("Constraint Pass", _ci(perf.constraint_pass_rate, pct=True)))
    if perf.run_errors:
        rows.append(("Runs Crashed", f"{perf.run_errors}/{n_runs}", "⚠"))
    rows += [
        None,
        ("Cost / Run", _money(eff.cost_usd.mean, eff.cost_complete)),
        ("Cost / Successful Task", _money(eff.cost_per_success, eff.cost_complete)),
        ("Tokens / Run", f"{eff.tokens.mean:,.0f}"),
        ("LLM Calls / Run", f"{eff.llm_calls.mean:.1f}"),
        ("Tool Calls / Run", f"{eff.tool_calls.mean:.1f}"),
        None,
        ("Wall Clock", f"{eff.wall_clock_s.mean:.1f}s"),
        ("Critical Path", f"{eff.critical_path_s.mean:.1f}s"),
        ("Agent Compute Time", f"{eff.agent_compute_s.mean:.1f}s"),
        ("Waiting Time", f"{eff.waiting_time_s.mean:.1f}s"),
        None,
        ("Communication / Work", _pct(coord.communication_ratio.mean),
         "⚠" if coord.communication_ratio.mean >= result.smell_config.communication_ratio_warning else ""),
        ("Redundant Work", _pct(coord.redundancy_ratio.mean), "⚠" if coord.redundancy_ratio.mean >= 0.2 else ""),
        ("Parallelism (actual)", _pct(coord.actual_parallelism.mean)),
        ("Parallelism (potential)", _pct(coord.potential_parallelism.mean),
         "⚠" if coord.parallel_opportunity.mean >= result.smell_config.parallelism_opportunity else ""),
        ("Context-bloated calls / run", f"{coord.context_bloat_calls.mean:.1f}",
         "⚠" if coord.context_bloat_calls.mean > 0 else ""),
    ]
    if result.leakage:
        rows.append(("Leakage Warnings", str(len(result.leakage)), "⚠"))
    parts = [box("SWARM EVALUATION", rows)]
    if not eff.cost_complete:
        parts.append("  * cost incomplete: some models have no pricing entry")

    if show_agents and coord.utilization:
        parts.append("")
        parts.append("AGENT UTILIZATION (observed, not causal)")
        parts.append(f"  {'agent':<16}{'role':<12}{'tokens':<26}{'contrib':>8}{'dead':>7}{'invoked':>9}{'cost':>10}")
        for aid, u in sorted(coord.utilization.items(), key=lambda kv: -kv[1]["token_share"]):
            cost = eff.per_agent.get(aid, {}).get("cost_usd")
            parts.append(f"  {aid:<16}{(u.get('role') or '-')[:11]:<12}{bar(u['token_share'])} {u['token_share']*100:>4.0f}%"
                         f"{u['observed_contribution']*100:>7.0f}%{u['dead_rate']*100:>6.0f}%{u['invocation_rate']*100:>8.0f}%"
                         f"{_money(cost):>10}")
        parts.append("  contrib = share of the agent's output used downstream / in the final answer (observed).")
        if coord.highest_overlap:
            a, b = coord.highest_overlap["agents"]
            parts.append(f"  Highest overlap: {a} ↔ {b}  ({_pct(coord.highest_overlap['overlap'])})")

    diags = result.diagnostics[:max_diagnostics]
    parts.append("")
    parts.append(f"DIAGNOSTICS ({len(result.diagnostics)} findings; hypotheses, not verdicts)")
    if not diags:
        parts.append("  none")
    for d in diags:
        icon = {"critical": "✖", "warning": "⚠", "info": "ℹ"}.get(d.severity, "•")
        prev = f" — in {_pct(d.prevalence)} of runs" if d.prevalence is not None else ""
        parts.append(f"\n{icon} {d.title} [{d.smell}]{prev}")
        parts.append(f"  Observed: {d.observation}")
        parts.append(f"  Hypothesis: {d.hypothesis}")
        parts.append(f"  Potential intervention: {d.recommendation}")

    if show_validity:
        parts.append("")
        parts.append("VALIDITY")
        parts.append(f"  Repeated trials: {perf.trials}; per-trial success: "
                     + ", ".join(_pct(x) for x in perf.per_trial))
        unstable = [t for t, r in result.repeatability.items() if r.get("unstable")]
        if unstable:
            parts.append(f"  Unstable tasks (mixed outcomes across trials): {', '.join(unstable)}")
        if perf.evaluator_agreement is not None:
            parts.append(f"  Evaluator agreement: {_pct(perf.evaluator_agreement)}; confidence: "
                         f"{_pct(perf.evaluator_confidence or 0)}")
        parts.append(f"  Evaluators: {', '.join(perf.evaluators) or 'none'}")
        for w in result.leakage[:5]:
            parts.append(f"  ⚠ leakage: {w}")
    return "\n".join(parts)


def _signed_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:+.0f}%"


def render_comparison(comp: Any) -> str:
    lines = [f"{comp.baseline_label}  →  {comp.variant_label}   ({comp.n_pairs} paired runs)",
             f"  Δ success   {comp.success_delta*100:+.1f} pp  [{comp.success_ci[0]*100:+.1f}, {comp.success_ci[1]*100:+.1f}]"
             + ("  *" if comp.significant else ""),
             f"  Δ score     {comp.score_delta:+.3f}  [{comp.score_ci[0]:+.3f}, {comp.score_ci[1]:+.3f}]",
             f"  Δ cost      {comp.cost_delta:+.4f} $/run ({_signed_pct(comp.cost_delta_pct)})",
             f"  Δ latency   {comp.latency_delta:+.2f} s ({_signed_pct(comp.latency_delta_pct)}); "
             f"Δ critical path {comp.critical_path_delta:+.2f} s",
             f"  Δ tokens    {_signed_pct(comp.tokens_delta_pct)}"]
    if comp.significant:
        lines.append("  * 95% CI excludes zero")
    return "\n".join(lines)


def render_ablation(report: Any) -> str:
    b = report.baseline
    head = (f"ABLATION vs {b.label}  (success {_pct(b.success_rate)}, {_money(b.cost['per_run'])}/run, "
            f"{b.latency['wall_clock']:.1f}s)")
    lines = [head, f"  {'component':<20}{'Δ success':>12}{'95% CI':>20}{'Δ cost':>10}{'Δ latency':>11}"
                   f"{'crashes':>9}{'observed':>10}"]
    for r in report.ranking():
        c = r.comparison
        obs = "n/a" if r.observed_contribution is None else _pct(r.observed_contribution)
        lines.append(f"  {r.component:<20}{c.success_delta*100:>+11.1f}p"
                     f"{'[' + f'{c.success_ci[0]*100:+.0f}, {c.success_ci[1]*100:+.0f}' + ']':>20}"
                     f"{_signed_pct(c.cost_delta_pct):>10}{_signed_pct(c.latency_delta_pct):>11}"
                     f"{_pct(r.run_error_rate):>9}{obs:>10}")
    lines.append("  Δ success = change in success rate when the component is REMOVED (paired, same seeds).")
    lines.append("  observed = observed contribution in the baseline; compare it with the causal column.")
    return "\n".join(lines)


def render_resilience(report: Any) -> str:
    b = report.baseline
    lines = [f"FAILURE INJECTION vs {b.label}  (baseline success {_pct(b.success_rate)})",
             f"  {'failure':<28}{'success':>9}{'resilience':>12}{'quality':>9}{'crashes':>9}{'Δ cost':>9}{'Δ latency':>11}"]
    for r in report.rows:
        v = r.comparison.variant
        lines.append(f"  {r.failure:<28}{_pct(v.success_rate) if v else 'n/a':>9}{r.resilience*100:>11.0f}%"
                     f"{r.quality_retention*100:>8.0f}%{_pct(r.run_error_rate):>9}"
                     f"{_signed_pct(r.recovery_cost_pct):>9}{_signed_pct(r.recovery_latency_pct):>11}")
    lines.append(f"  Mean resilience: {_pct(report.resilience)}  (variant success ÷ baseline success)")
    return "\n".join(lines)


def render_frontier(points: Sequence[Any], height: int = 8) -> str:
    if not points:
        return "no configurations"
    lines = ["EFFICIENCY FRONTIER (success vs cost per run)",
             f"  {'configuration':<34}{'success':>9}{'cost/run':>10}{'latency':>9}  frontier"]
    for p in points:
        lines.append(f"  {p.label[:33]:<34}{_pct(p.success):>9}{_money(p.cost):>10}{p.latency:>8.1f}s  "
                     f"{'●' if p.on_frontier else '·'}")
    # tiny ascii scatter
    costs = [p.cost for p in points]
    succ = [p.success for p in points]
    cmin, cmax = min(costs), max(costs)
    smin, smax = min(succ), max(succ)
    width = 40
    grid = [[" "] * (width + 1) for _ in range(height + 1)]
    for p in points:
        x = 0 if cmax == cmin else int(round((p.cost - cmin) / (cmax - cmin) * width))
        y = 0 if smax == smin else int(round((p.success - smin) / (smax - smin) * height))
        grid[height - y][x] = "●" if p.on_frontier else "·"
    lines.append("")
    for i, row in enumerate(grid):
        label = f"{(smax - (smax - smin) * i / height)*100:5.0f}% │" if height else "      │"
        lines.append("  " + label + "".join(row))
    lines.append("        └" + "─" * (width + 1))
    lines.append(f"         {_money(cmin):<20}{_money(cmax):>20}")
    return "\n".join(lines)
