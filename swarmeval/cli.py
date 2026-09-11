"""Command-line interface.

    swarmeval demo [--trials N] [--out DIR]
    swarmeval run --swarm pkg.module:factory --benchmark bench.json [--trials N] [--out DIR] [--html]
    swarmeval report runs.json [--html out.html]
    swarmeval ablate --swarm ... --benchmark ... [--agents a,b] [--trials N]
    swarmeval inject --swarm ... --benchmark ... --failure agent_crash:critic [--failure ...]
    swarmeval compare baseline.json variant.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, List, Optional

from . import __version__
from .config import FailureSpec
from .causal.experiments import CounterfactualRunner, ablate_agents, compare, frontier, inject_failures
from .causal.interventions import InjectFailure, SetParam
from .evaluate import EvaluationResult, analyze_runs, evaluate
from .export.jsonl import write_events_jsonl
from .export.otel import to_otel
from .recorder import SimClock
from .report.text import render_ablation, render_comparison, render_frontier, render_resilience
from .runner import RunSet
from .schema import Benchmark


def _load_obj(spec: str) -> Any:
    if ":" not in spec:
        raise SystemExit(f"--swarm must look like package.module:attr (got {spec!r})")
    mod, attr = spec.split(":", 1)
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), attr)


def _clock_factory(name: Optional[str]) -> Optional[Callable[[], Any]]:
    if name == "sim":
        return SimClock
    if name in (None, "wall"):
        return None
    return _load_obj(name)


def _write_outputs(result: EvaluationResult, out: Optional[str], html: bool, **sections: Any) -> None:
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    result.save(os.path.join(out, f"{result.label}.json"))
    write_events_jsonl(result.runs.trajectories, os.path.join(out, f"{result.label}.events.jsonl"))
    with open(os.path.join(out, f"{result.label}.otel.json"), "w") as f:
        json.dump([to_otel(t) for t in result.runs.trajectories], f)
    if html:
        result.to_html(os.path.join(out, f"{result.label}.html"), **sections)
    print(f"\nwrote {out}/{result.label}.json (+ .events.jsonl, .otel.json{', .html' if html else ''})")


def cmd_demo(args: argparse.Namespace) -> int:
    from examples.research_swarm import build_benchmark, make_swarm  # type: ignore
    bench = build_benchmark()
    runner = CounterfactualRunner(make_swarm, bench, trials=args.trials, clock_factory=SimClock, progress=True)
    base = runner.baseline()
    print(base.report())
    sections: dict = {}
    if not args.quick:
        print("\n" + "=" * 70 + "\nCAUSAL ANALYSIS\n" + "=" * 70)
        abl = ablate_agents(runner)
        print(render_ablation(abl))
        res = inject_failures(runner, [InjectFailure("agent_crash", "critic"), InjectFailure("agent_crash", "researcher_a"),
                                       InjectFailure("tool_error", "web_search", probability=0.5),
                                       InjectFailure("message_drop", "coordinator->critic")])
        print()
        print(render_resilience(res))
        variants = {
            "single agent": [SetParam("single_agent", True)],
            "direct channels": [SetParam("direct_channels", True)],
            "partitioned scope": [SetParam("partition_scope", True)],
            "sequential research": [SetParam("parallel_research", False)],
        }
        results = [base] + [runner.run(*ivs, label=n) for n, ivs in variants.items()]
        pts = frontier(results)
        print()
        print(render_frontier(pts))
        from .optimize.candidates import propose, validate
        cands = propose(base, {"coordinator_bottleneck": {"direct_channels": True},
                               "under_specialization": {"partition_scope": True},
                               "agent_ping_pong": {"revision_rounds": 0}})
        val = validate(cands, runner)
        print()
        print(val.table())
        sections = {"ablation": abl, "resilience": res, "frontier_points": pts, "candidates": val.candidates}
    _write_outputs(base, args.out, True, **sections)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    swarm = _load_obj(args.swarm)
    bench = Benchmark.load(args.benchmark) if not ":" in args.benchmark else _load_obj(args.benchmark)
    if callable(bench) and not isinstance(bench, Benchmark):
        bench = bench()
    result = evaluate(swarm, bench, trials=args.trials, seed=args.seed, clock_factory=_clock_factory(args.clock),
                      label=args.label, progress=True)
    print(result.report())
    _write_outputs(result, args.out, args.html)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.path) as f:
        d = json.load(f)
    runs = RunSet.from_dict(d["runs"] if "runs" in d else d)
    result = analyze_runs(runs)
    print(result.report())
    if args.html:
        result.to_html(args.html)
        print(f"\nwrote {args.html}")
    return 0


def _runner(args: argparse.Namespace) -> CounterfactualRunner:
    swarm = _load_obj(args.swarm)
    bench = Benchmark.load(args.benchmark) if not ":" in args.benchmark else _load_obj(args.benchmark)
    if callable(bench) and not isinstance(bench, Benchmark):
        bench = bench()
    return CounterfactualRunner(swarm, bench, trials=args.trials, seed=args.seed,
                                clock_factory=_clock_factory(args.clock), progress=True)


def cmd_ablate(args: argparse.Namespace) -> int:
    runner = _runner(args)
    agents = args.agents.split(",") if args.agents else None
    report = ablate_agents(runner, agents)
    print(render_ablation(report))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "ablation.json"), "w") as f:
            json.dump(report.to_dict(), f, indent=1, default=str)
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    runner = _runner(args)
    specs: List[InjectFailure] = []
    for f in args.failure:
        parts = f.split(":")
        kind, target = parts[0], (parts[1] if len(parts) > 1 else "*")
        prob = float(parts[2]) if len(parts) > 2 else 1.0
        specs.append(InjectFailure(kind, target, probability=prob))
    report = inject_failures(runner, specs)
    print(render_resilience(report))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    def load(p: str) -> EvaluationResult:
        with open(p) as f:
            d = json.load(f)
        return analyze_runs(RunSet.from_dict(d["runs"] if "runs" in d else d))
    print(render_comparison(compare(load(args.baseline), load(args.variant))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="swarmeval", description="Evaluate, diagnose and ablate multi-agent systems.")
    p.add_argument("--version", action="version", version=f"swarmeval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the built-in simulated research swarm end to end")
    d.add_argument("--trials", type=int, default=3)
    d.add_argument("--out", default=None, help="directory for JSON/JSONL/OTel/HTML outputs")
    d.add_argument("--quick", action="store_true", help="skip the causal experiments")
    d.set_defaults(fn=cmd_demo)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--swarm", required=True, help="package.module:factory_or_swarm")
    common.add_argument("--benchmark", required=True, help="path to benchmark JSON/JSONL or package.module:builder")
    common.add_argument("--trials", type=int, default=1)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--clock", default=None, help="'sim', 'wall' (default) or package.module:factory")

    r = sub.add_parser("run", parents=[common], help="run a swarm on a benchmark and print the report")
    r.add_argument("--label", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--html", action="store_true")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="re-analyse a saved run set")
    rp.add_argument("path")
    rp.add_argument("--html", default=None, help="write an HTML report to this path")
    rp.set_defaults(fn=cmd_report)

    a = sub.add_parser("ablate", parents=[common], help="agent ablation (remove each agent, re-run, compare)")
    a.add_argument("--agents", default=None, help="comma-separated agent ids (default: all observed)")
    a.add_argument("--out", default=None)
    a.set_defaults(fn=cmd_ablate)

    i = sub.add_parser("inject", parents=[common], help="failure injection / resilience")
    i.add_argument("--failure", action="append", required=True, help="kind:target[:probability], e.g. agent_crash:critic")
    i.set_defaults(fn=cmd_inject)

    c = sub.add_parser("compare", help="compare two saved run sets")
    c.add_argument("baseline")
    c.add_argument("variant")
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
