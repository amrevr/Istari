"""Standalone HTML report with inline SVG execution graph and timeline."""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schema import EventType

CSS = """
:root{--bg:#fafafa;--fg:#1c1c1c;--muted:#6b6b6b;--card:#fff;--line:#e4e4e4;--acc:#2f6fed;--ok:#1a9c5b;--warn:#d98a00;--bad:#d1434b;--info:#5b7fa6}
@media (prefers-color-scheme: dark){:root{--bg:#141517;--fg:#ececec;--muted:#a3a3a3;--card:#1e1f23;--line:#33353a}}
body{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--muted);margin-bottom:16px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:20px;font-weight:600}.tile .ci{font-size:11px;color:var(--muted)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:13px;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.bar{display:inline-block;height:10px;background:var(--acc);border-radius:2px;vertical-align:middle}
.bar-wrap{display:inline-block;width:120px;height:10px;background:var(--line);border-radius:2px;vertical-align:middle;margin-right:6px}
.diag{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--info);border-radius:8px;padding:10px 14px;margin:10px 0}
.diag.warning{border-left-color:var(--warn)}.diag.critical{border-left-color:var(--bad)}.diag .t{font-weight:600}.diag .m{color:var(--muted);font-size:12px}
.diag p{margin:4px 0}.svgwrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;background:var(--line);color:var(--fg)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.muted{color:var(--muted)}.note{font-size:12px;color:var(--muted)}
svg text{font:11px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:var(--fg)}
"""


def _e(x: Any) -> str:
    return html.escape(str(x))


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:.1f}%"


def _money(x: Optional[float]) -> str:
    return "unknown" if x is None else (f"${x:.2f}" if x >= 0.1 else f"${x:.4f}")


def _tile(k: str, v: str, ci: str = "") -> str:
    return f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div><div class="ci">{_e(ci)}</div></div>'


def _bar(frac: float) -> str:
    w = max(0, min(100, int(round(frac * 100))))
    return f'<span class="bar-wrap"><span class="bar" style="width:{w}%"></span></span>'


# --------------------------------------------------------------------------- #
# SVG: agent graph
# --------------------------------------------------------------------------- #
def graph_svg(analysis: Any) -> str:
    g = analysis.graph
    traj = analysis.trajectory
    agents = g.agents
    if not agents:
        return "<p class='muted'>no agents recorded</p>"
    first_start = {a: min(s.start for s in traj.spans_for(a)) for a in agents}
    preds: Dict[str, set] = {a: set() for a in agents}
    for (s, t, k), e in g.edges.items():
        if s != t and s in preds and t in preds and first_start[s] <= first_start[t]:
            preds[t].add(s)
    rank: Dict[str, int] = {}
    for a in sorted(agents, key=lambda x: first_start[x]):
        rank[a] = max([rank.get(p, 0) + 1 for p in preds[a] if p in rank] or [0])
    cols: Dict[int, List[str]] = {}
    for a, r in rank.items():
        cols.setdefault(r, []).append(a)
    col_w, row_h, node_w, node_h = 190, 78, 150, 48
    ncols = max(cols) + 1 if cols else 1
    nrows = max(len(v) for v in cols.values()) if cols else 1
    W, H = 40 + ncols * col_w, 40 + nrows * row_h
    pos: Dict[str, Tuple[float, float]] = {}
    for r, names in cols.items():
        off = (nrows - len(names)) * row_h / 2
        for i, a in enumerate(sorted(names, key=lambda x: first_start[x])):
            pos[a] = (20 + r * col_w, 20 + off + i * row_h)
    util = analysis.coordination.utilization
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" fill="#888"/></marker></defs>']
    colors = {"message": "#2f6fed", "artifact": "#1a9c5b", "invocation": "#aaa"}
    for (s, t, k), e in g.edges.items():
        if s == t or s not in pos or t not in pos:
            continue
        x1, y1 = pos[s][0] + node_w, pos[s][1] + node_h / 2
        x2, y2 = pos[t][0], pos[t][1] + node_h / 2
        if x2 < x1:  # back edge
            x1, x2 = pos[s][0] + node_w / 2, pos[t][0] + node_w / 2
            y1, y2 = pos[s][1], pos[t][1]
        dash = ' stroke-dasharray="4 3"' if k == "invocation" else ""
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{colors.get(k, "#999")}" '
                     f'stroke-width="{1 + min(3, e.count / 2):.1f}" marker-end="url(#arr)"{dash} opacity="0.8"/>')
        if k != "invocation":
            parts.append(f'<text x="{mx:.0f}" y="{my - 4:.0f}" font-size="10" fill="#888">{e.count}×'
                         f'{(" " + str(e.tokens) + "t") if e.tokens else ""}</text>')
    for a, (x, y) in pos.items():
        u = util.get(a)
        dead = u.dead if u else False
        fill = "#ffe9e9" if dead else "#eef3ff"
        stroke = "#d1434b" if dead else "#2f6fed"
        parts.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{x + 10}" y="{y + 19}" font-weight="600" fill="#1c1c1c">{_e(a)}</text>')
        role = g.roles.get(a) or ""
        tok = f"{u.tokens:,}t" if u else ""
        parts.append(f'<text x="{x + 10}" y="{y + 36}" font-size="10" fill="#555">{_e(role)} · {tok}'
                     f'{" · unused" if dead else ""}</text>')
    parts.append("</svg>")
    legend = ('<p class="note">Blue edges: messages · green: artifact flow · dashed: invocation. '
              'Red nodes: output not used downstream in this run.</p>')
    return "".join(parts) + legend


# --------------------------------------------------------------------------- #
# SVG: timeline
# --------------------------------------------------------------------------- #
def timeline_svg(analysis: Any) -> str:
    traj = analysis.trajectory
    t0 = traj.started_at or 0.0
    total = max(traj.duration, 1e-6)
    spans = sorted(traj.spans.values(), key=lambda s: (s.start, s.agent_id))
    if not spans:
        return "<p class='muted'>no spans recorded</p>"
    W, left, row_h = 900, 130, 22
    scale = (W - left - 20) / total
    H = 30 + row_h * len(spans)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    ticks = 6
    for i in range(ticks + 1):
        x = left + (W - left - 20) * i / ticks
        parts.append(f'<line x1="{x:.0f}" y1="14" x2="{x:.0f}" y2="{H}" stroke="#e4e4e4"/>')
        parts.append(f'<text x="{x:.0f}" y="10" font-size="9" fill="#888" text-anchor="middle">{total * i / ticks:.1f}s</text>')
    cp_units = set(analysis.graph.critical_path()[1])
    for i, s in enumerate(spans):
        y = 20 + i * row_h
        x0 = left + (s.start - t0) * scale
        x1 = left + ((s.end or s.start) - t0) * scale
        col = "#f0d0d0" if s.status in ("error", "injected") else "#e6e9ef"
        parts.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(1.0, x1 - x0):.1f}" height="{row_h - 6}" fill="{col}" rx="3"/>')
        parts.append(f'<text x="4" y="{y + 12}" font-size="10">{_e(s.agent_id)}</text>')
        for ev in traj.events:
            if ev.span_id != s.span_id:
                continue
            if ev.event_type in (EventType.LLM_CALL.value, EventType.TOOL_CALL.value, EventType.WAIT.value):
                ex0 = left + (ev.timestamp - t0) * scale
                ex1 = left + (ev.end_timestamp - t0) * scale
                c = {"llm_call": "#2f6fed", "tool_call": "#1a9c5b", "wait": "#bbb"}[ev.event_type]
                if ev.status in ("error", "injected"):
                    c = "#d1434b"
                stroke = ' stroke="#111" stroke-width="1.5"' if ev.event_id in cp_units else ""
                parts.append(f'<rect x="{ex0:.1f}" y="{y + 2}" width="{max(1.5, ex1 - ex0):.1f}" height="{row_h - 10}" '
                             f'fill="{c}" rx="2"{stroke}><title>{_e(ev.event_type)} {_e(ev.tool_name or ev.model or "")} '
                             f'{ev.duration_s:.2f}s {ev.total_tokens}t</title></rect>')
            elif ev.event_type == EventType.MESSAGE.value and ev.sender == s.agent_id:
                mx = left + (ev.timestamp - t0) * scale
                parts.append(f'<path d="M{mx:.1f},{y - 1} l4,5 l-4,5 l-4,-5 z" fill="#d98a00"><title>→ {_e(ev.receiver)} '
                             f'({_e(ev.kind)})</title></path>')
    parts.append("</svg>")
    return "".join(parts) + ('<p class="note">Blue: LLM calls · green: tool calls · grey: waits · red: failed/injected · '
                             'orange diamonds: messages sent · black outline: on the critical path.</p>')


def frontier_svg(points: Sequence[Any]) -> str:
    if not points:
        return ""
    W, H, pad = 520, 260, 40
    costs = [p.cost for p in points]
    succ = [p.success for p in points]
    cmin, cmax = min(costs), max(costs)
    smin, smax = min(0.0, min(succ)), 1.0
    def X(c: float) -> float:
        return pad + (W - 2 * pad) * ((c - cmin) / (cmax - cmin) if cmax > cmin else 0.5)
    def Y(s: float) -> float:
        return H - pad - (H - 2 * pad) * ((s - smin) / (smax - smin) if smax > smin else 0.5)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#999"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#999"/>',
             f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-size="10" fill="#888">cost per run</text>',
             f'<text x="12" y="{H/2}" font-size="10" fill="#888" transform="rotate(-90 12,{H/2})" text-anchor="middle">success</text>']
    front = sorted([p for p in points if p.on_frontier], key=lambda p: p.cost)
    if len(front) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p.cost):.1f},{Y(p.success):.1f}" for i, p in enumerate(front))
        parts.append(f'<path d="{d}" fill="none" stroke="#2f6fed" stroke-dasharray="4 3"/>')
    for p in points:
        parts.append(f'<circle cx="{X(p.cost):.1f}" cy="{Y(p.success):.1f}" r="5" fill="{"#2f6fed" if p.on_frontier else "#bbb"}">'
                     f'<title>{_e(p.label)}: {_pct(p.success)} @ {_money(p.cost)}</title></circle>')
        parts.append(f'<text x="{X(p.cost)+7:.1f}" y="{Y(p.success)-6:.1f}" font-size="9">{_e(p.label[:24])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Main renderer
# --------------------------------------------------------------------------- #
def render_html(result: Any, path: Optional[str] = None, ablation: Any = None, resilience: Any = None,
                frontier_points: Optional[Sequence[Any]] = None, candidates: Any = None,
                title: Optional[str] = None) -> str:
    perf, eff, coord = result.performance, result.efficiency, result.coordination
    title = title or f"SwarmEval · {result.label}"
    out: List[str] = [f"<title>{_e(title)}</title><style>{CSS}</style>",
                      f"<h1>{_e(title)}</h1>",
                      f'<div class="sub">{_e(result.benchmark)} · {perf.n_tasks} tasks × {perf.trials} trials = '
                      f'{len(result.runs)} runs · config: {_e(result.config.describe())}</div>']
    out.append('<div class="tiles">' + "".join([
        _tile("Task success", _pct(perf.success_rate.mean),
              f"95% CI {_pct(perf.success_rate.ci_low)} – {_pct(perf.success_rate.ci_high)}"),
        _tile("Cost / successful task", _money(eff.cost_per_success), f"{_money(eff.cost_usd.mean)} per run"),
        _tile("Critical path", f"{eff.critical_path_s.mean:.1f}s", f"wall clock {eff.wall_clock_s.mean:.1f}s"),
        _tile("Communication / work", _pct(coord.communication_ratio.mean), "message tokens ÷ LLM tokens"),
        _tile("Redundant work", _pct(coord.redundancy_ratio.mean), f"{coord.accidental_redundancy.mean:.1f} accidental / run"),
        _tile("Parallelism", _pct(coord.actual_parallelism.mean), f"potential {_pct(coord.potential_parallelism.mean)}"),
    ]) + "</div>")

    # performance
    out.append("<h2>Task performance</h2><table><tr><th>Metric</th><th class=num>Mean</th><th class=num>95% CI</th><th class=num>n</th></tr>")
    rows = [("Success rate", perf.success_rate, True), ("Score", perf.score, False)]
    if perf.constraint_pass_rate:
        rows.append(("Constraint pass", perf.constraint_pass_rate, True))
    for name, s, is_pct in rows:
        f = _pct if is_pct else (lambda v: f"{v:.3f}")
        out.append(f"<tr><td>{name}</td><td class=num>{f(s.mean)}</td><td class=num>{f(s.ci_low)} – {f(s.ci_high)}</td><td class=num>{s.n}</td></tr>")
    out.append("</table>")
    out.append("<table><tr><th>Task</th><th class=num>Success</th><th class=num>Score</th><th class=num>Cost</th><th class=num>Latency</th><th class=num>Runs</th></tr>")
    for tid, r in perf.per_task.items():
        out.append(f"<tr><td>{_e(tid)}</td><td class=num>{_pct(r['success_rate'])}</td><td class=num>{r['score']:.2f}</td>"
                   f"<td class=num>{_money(r['cost_usd'])}</td><td class=num>{r['latency_s']:.1f}s</td><td class=num>{r['n']}</td></tr>")
    out.append("</table>")

    # efficiency
    out.append("<h2>Efficiency</h2><div class=tiles>" + "".join([
        _tile("Tokens / run", f"{eff.tokens.mean:,.0f}", f"in {eff.input_tokens.mean:,.0f} · out {eff.output_tokens.mean:,.0f}"),
        _tile("LLM calls / run", f"{eff.llm_calls.mean:.1f}", f"tool calls {eff.tool_calls.mean:.1f}"),
        _tile("Total cost", _money(eff.cost_total), "" if eff.cost_complete else "incomplete pricing"),
        _tile("Agent compute", f"{eff.agent_compute_s.mean:.1f}s", f"waiting {eff.waiting_time_s.mean:.1f}s"),
        _tile("Total work", f"{eff.total_work_s.mean:.1f}s", f"critical path {eff.critical_path_s.mean:.1f}s"),
        _tile("Messages / run", f"{eff.messages.mean:.1f}", f"errors {eff.errors.mean:.1f}"),
    ]) + "</div>")

    # agents
    out.append("<h2>Agent utilization <span class=badge>observed, not causal</span></h2>")
    out.append("<table><tr><th>Agent</th><th>Role</th><th>Token share</th><th class=num>Tokens</th><th class=num>Cost</th>"
               "<th class=num>Observed contribution</th><th class=num>Unused in</th><th class=num>Invoked</th><th class=num>Critical path</th><th>Tools</th></tr>")
    for aid, u in sorted(coord.utilization.items(), key=lambda kv: -kv[1]["token_share"]):
        c = eff.per_agent.get(aid, {})
        out.append(f"<tr><td><b>{_e(aid)}</b></td><td>{_e(u.get('role') or '-')}</td><td>{_bar(u['token_share'])}{_pct(u['token_share'])}</td>"
                   f"<td class=num>{u['tokens']:,.0f}</td><td class=num>{_money(c.get('cost_usd'))}</td>"
                   f"<td class=num>{_pct(u['observed_contribution'])}</td><td class=num>{_pct(u['dead_rate'])} of runs</td>"
                   f"<td class=num>{_pct(u['invocation_rate'])}</td><td class=num>{u['critical_path_s']:.1f}s</td>"
                   f"<td>{_e(', '.join(u.get('tools_used', [])))}</td></tr>")
    out.append("</table>")

    # coordination
    out.append("<h2>Coordination quality</h2><div class=tiles>" + "".join([
        _tile("Useful work", _pct(coord.useful_work_fraction.mean), "share of LLM tokens not spent on messaging"),
        _tile("Information transfer", _pct(coord.information_fraction.mean), "result / feedback messages"),
        _tile("Coordination", _pct(coord.coordination_fraction.mean), "assignments / requests"),
        _tile("Hub agent", str(coord.hub_agent or "—"), f"{_pct(coord.hub_share.mean)} of messages"),
        _tile("Dependency depth", f"{coord.depth.mean:.1f}", f"density {_pct(coord.density.mean)}"),
        _tile("Context-bloated calls", f"{coord.context_bloat_calls.mean:.1f}", f"mean excess {_pct(coord.context_bloat_ratio.mean)}"),
        _tile("Redundant tokens", _pct(coord.redundant_token_fraction.mean),
              f"intentional {coord.intentional_redundancy.mean:.1f} / run"),
        _tile("Parallel opportunity", _pct(coord.parallel_opportunity.mean), "potential − actual"),
    ]) + "</div>")
    if coord.highest_overlap:
        a, b = coord.highest_overlap["agents"]
        out.append(f"<p class=note>Highest work overlap: <b>{_e(a)}</b> ↔ <b>{_e(b)}</b> ({_pct(coord.highest_overlap['overlap'])})</p>")

    # graph + timeline of representative run
    rep = next((a for a in result.analyses if a.trajectory.succeeded), result.analyses[0] if result.analyses else None)
    if rep is not None:
        out.append(f"<h2>Execution graph <span class=badge>run {_e(rep.trajectory.task_id)} / trial {rep.trajectory.trial}</span></h2>")
        out.append(f"<div class=svgwrap>{graph_svg(rep)}</div>")
        out.append("<h2>Timeline</h2>")
        out.append(f"<div class=svgwrap>{timeline_svg(rep)}</div>")

    # diagnostics
    out.append(f"<h2>Diagnostics <span class=badge>{len(result.diagnostics)} findings · hypotheses, not verdicts</span></h2>")
    if not result.diagnostics:
        out.append("<p class=muted>No smells detected.</p>")
    for d in result.diagnostics:
        prev = f" · in {_pct(d.prevalence)} of runs" if d.prevalence is not None else ""
        out.append(f'<div class="diag {d.severity}"><div class="t">{_e(d.title)} <span class=badge>{_e(d.smell)}</span></div>'
                   f'<div class="m">{_e(d.severity)}{prev}{(" · " + _e(", ".join(d.agents))) if d.agents else ""}</div>'
                   f"<p><b>Observed:</b> {_e(d.observation)}</p><p><b>Hypothesis:</b> {_e(d.hypothesis)}</p>"
                   f"<p><b>Potential intervention:</b> {_e(d.recommendation)}</p></div>")

    # causal sections
    if ablation is not None:
        out.append("<h2>Agent ablation <span class=badge>causal · paired runs, same seeds</span></h2>")
        out.append("<table><tr><th>Component</th><th class=num>Δ success (removed)</th><th class=num>95% CI</th><th class=num>Δ cost</th>"
                   "<th class=num>Δ latency</th><th class=num>Crashes</th><th class=num>Observed contribution</th></tr>")
        for r in ablation.ranking():
            c = r.comparison
            cls = "bad" if c.success_delta < 0 else ("ok" if c.success_delta > 0 else "muted")
            out.append(f"<tr><td><b>{_e(r.component)}</b></td><td class='num {cls}'>{c.success_delta*100:+.1f} pp</td>"
                       f"<td class=num>[{c.success_ci[0]*100:+.0f}, {c.success_ci[1]*100:+.0f}]</td>"
                       f"<td class=num>{_signed(c.cost_delta_pct)}</td><td class=num>{_signed(c.latency_delta_pct)}</td>"
                       f"<td class=num>{_pct(r.run_error_rate)}</td><td class=num>{_pct(r.observed_contribution) if r.observed_contribution is not None else 'n/a'}</td></tr>")
        out.append("</table><p class=note>Δ success is the change when the component is removed; a large negative value means the component matters. "
                   "Compare with observed contribution: activity is not contribution.</p>")
    if resilience is not None:
        out.append(f"<h2>Failure injection <span class=badge>mean resilience {_pct(resilience.resilience)}</span></h2>")
        out.append("<table><tr><th>Failure</th><th class=num>Success</th><th class=num>Resilience</th><th class=num>Quality retained</th>"
                   "<th class=num>Crashes</th><th class=num>Δ cost</th><th class=num>Δ latency</th></tr>")
        for r in resilience.rows:
            v = r.comparison.variant
            out.append(f"<tr><td>{_e(r.failure)}</td><td class=num>{_pct(v.success_rate) if v else 'n/a'}</td>"
                       f"<td class=num>{_pct(r.resilience)}</td><td class=num>{_pct(r.quality_retention)}</td>"
                       f"<td class=num>{_pct(r.run_error_rate)}</td><td class=num>{_signed(r.recovery_cost_pct)}</td>"
                       f"<td class=num>{_signed(r.recovery_latency_pct)}</td></tr>")
        out.append("</table>")
    if frontier_points:
        out.append("<h2>Efficiency frontier</h2><div class=svgwrap>" + frontier_svg(frontier_points) + "</div>")
        out.append("<table><tr><th>Configuration</th><th class=num>Success</th><th class=num>Cost / run</th><th class=num>Latency</th><th>Frontier</th></tr>")
        for p in frontier_points:
            out.append(f"<tr><td>{_e(p.label)}</td><td class=num>{_pct(p.success)}</td><td class=num>{_money(p.cost)}</td>"
                       f"<td class=num>{p.latency:.1f}s</td><td>{'●' if p.on_frontier else ''}</td></tr>")
        out.append("</table>")
    if candidates is not None:
        out.append("<h2>Architecture candidates <span class=badge>predictions are hypotheses until validated</span></h2>")
        out.append("<table><tr><th>Candidate</th><th>Rationale</th><th class=num>Predicted Δ cost</th><th class=num>Predicted Δ latency</th>"
                   "<th class=num>Measured Δ success</th><th class=num>Measured Δ cost</th><th class=num>Measured Δ latency</th><th>Verdict</th></tr>")
        for c in candidates:
            m = c.measured
            out.append(f"<tr><td><b>{_e(c.name)}</b></td><td>{_e(c.rationale)}</td>"
                       f"<td class=num>{_signed(c.predicted.get('cost_pct'))}</td><td class=num>{_signed(c.predicted.get('latency_pct'))}</td>"
                       f"<td class=num>{(m.success_delta*100):+.1f} pp</td><td class=num>{_signed(m.cost_delta_pct)}</td>"
                       f"<td class=num>{_signed(m.latency_delta_pct)}</td><td>{_e(c.verdict)}</td></tr>"
                       if m else
                       f"<tr><td><b>{_e(c.name)}</b></td><td>{_e(c.rationale)}</td>"
                       f"<td class=num>{_signed(c.predicted.get('cost_pct'))}</td><td class=num>{_signed(c.predicted.get('latency_pct'))}</td>"
                       f"<td class=num colspan=3 class=muted>not validated</td><td>{_e(c.verdict)}</td></tr>")
        out.append("</table>")

    # validity + runs
    out.append("<h2>Validity</h2><ul>")
    out.append(f"<li>Repeated trials: {perf.trials}; per-trial success: {', '.join(_pct(x) for x in perf.per_trial)}</li>")
    unstable = [t for t, r in result.repeatability.items() if r.get("unstable")]
    if unstable:
        out.append(f"<li>Unstable tasks (mixed outcomes across trials): {_e(', '.join(unstable))}</li>")
    if perf.evaluator_agreement is not None:
        out.append(f"<li>Evaluator agreement {_pct(perf.evaluator_agreement)}, confidence {_pct(perf.evaluator_confidence or 0)}</li>")
    out.append(f"<li>Evaluators: {_e(', '.join(perf.evaluators) or 'none')}</li>")
    for w in result.leakage[:10]:
        out.append(f"<li class=bad>Leakage: {_e(w)}</li>")
    out.append("</ul>")
    out.append("<h2>Runs</h2><table><tr><th>Task</th><th class=num>Trial</th><th>Status</th><th class=num>Score</th><th class=num>Tokens</th>"
               "<th class=num>Cost</th><th class=num>Wall</th><th class=num>Critical path</th><th>Smells</th></tr>")
    for a in result.analyses:
        t = a.trajectory
        st = "✓" if t.succeeded else "✗"
        cls = "ok" if t.succeeded else "bad"
        out.append(f"<tr><td>{_e(t.task_id)}</td><td class=num>{t.trial}</td><td class={cls}>{st} {_e(t.status)}</td>"
                   f"<td class=num>{t.score:.2f}</td><td class=num>{a.efficiency.total_tokens:,}</td><td class=num>{_money(a.efficiency.cost_usd)}</td>"
                   f"<td class=num>{a.efficiency.wall_clock_s:.1f}s</td><td class=num>{a.efficiency.critical_path_s:.1f}s</td>"
                   f"<td>{_e(', '.join(sorted({d.smell for d in a.diagnostics})))}</td></tr>")
    out.append("</table>")
    out.append('<p class=note>Generated by SwarmEval. Observed metrics describe the trajectory; causal claims require ablation.</p>')
    doc = "\n".join(out)
    if path:
        with open(path, "w") as f:
            f.write("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'></head><body>"
                    + doc + "</body></html>")
    return doc


def _signed(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:+.0f}%"
