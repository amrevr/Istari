"""Aggregates per-run diagnostics across a benchmark into prevalence-weighted
findings."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Sequence

from .smells import SEVERITY_ORDER, Diagnostic


def _merge_evidence(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    keys: List[str] = []
    for ev in items:
        for k in ev:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [ev[k] for ev in items if k in ev and ev[k] is not None]
        if not vals:
            merged[k] = None
        elif all(isinstance(v, bool) for v in vals):
            merged[k] = sum(vals) / len(vals)
        elif all(isinstance(v, (int, float)) for v in vals):
            merged[k] = round(sum(vals) / len(vals), 3)
        elif all(isinstance(v, list) for v in vals):
            seen: List[Any] = []
            for v in vals:
                for x in v:
                    if x not in seen:
                        seen.append(x)
            merged[k] = seen[:10]
        else:
            merged[k] = vals[0]
    return merged


def aggregate_diagnostics(per_run: Sequence[Sequence[Diagnostic]], n_runs: int,
                          min_prevalence: float = 0.0) -> List[Diagnostic]:
    groups: "OrderedDict[Any, List[Diagnostic]]" = OrderedDict()
    for diags in per_run:
        seen_keys = set()
        for d in diags:
            if d.key in seen_keys:
                # same smell twice in one run (e.g. two late-artifact cases) -> keep first for prevalence
                groups[d.key].append(d)
                continue
            seen_keys.add(d.key)
            groups.setdefault(d.key, []).append(d)
    out: List[Diagnostic] = []
    for key, ds in groups.items():
        runs = sorted({r for d in ds for r in d.run_ids})
        prevalence = len(runs) / max(1, n_runs)
        if prevalence < min_prevalence:
            continue
        severity = max((d.severity for d in ds), key=lambda s: SEVERITY_ORDER.get(s, 0))
        first = ds[0]
        out.append(Diagnostic(first.smell, severity, first.title,
                              f"In {len(runs)}/{n_runs} runs: " + first.observation,
                              first.hypothesis, first.recommendation, list(first.agents),
                              _merge_evidence([d.evidence for d in ds]), runs, prevalence))
    out.sort(key=lambda d: (-SEVERITY_ORDER.get(d.severity, 0), -(d.prevalence or 0)))
    return out
