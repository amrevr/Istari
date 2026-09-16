import pytest

from examples.research_swarm import build_benchmark, make_swarm
from swarmeval import (CounterfactualRunner, InjectFailure, RemoveAgent, SetParam, SimClock, ablate_agents, compare,
                       evaluate, frontier, inject_failures)
from swarmeval.optimize import propose, validate


@pytest.fixture(scope="module")
def runner():
    return CounterfactualRunner(make_swarm, build_benchmark(4), trials=2, seed=3, clock_factory=SimClock)


def test_baseline_is_deterministic_and_finds_smells(runner):
    a = runner.baseline()
    b = evaluate(make_swarm, build_benchmark(4), trials=2, seed=3, clock_factory=SimClock)
    assert a.success_rate == b.success_rate
    assert [t.total_tokens for t in a.runs] == [t.total_tokens for t in b.runs]
    assert {"dead_agent", "context_bloat", "under_specialization"} <= set(a.smells)
    assert a.agent_contribution["researcher_c"] == 0.0
    assert a.cost["complete"] is True and a.cost["total"] > 0
    assert a.leakage == []


def test_ablation_distinguishes_activity_from_contribution(runner):
    report = ablate_agents(runner, ["researcher_c", "critic", "coordinator"])
    rows = {r.component: r for r in report.rows}
    assert rows["researcher_c"].comparison.success_delta == 0.0        # dead agent: no causal effect
    assert rows["researcher_c"].comparison.cost_delta < 0               # but it costs money
    assert rows["critic"].comparison.success_delta < 0                  # critic matters
    assert rows["coordinator"].comparison.success_delta < 0
    assert rows["critic"].comparison.n_pairs == 8
    assert report.ranking()[0].component in ("coordinator", "critic")
    assert "ABLATION" in report.table()


def test_failure_injection_reveals_hidden_resilience_value(runner):
    rep = inject_failures(runner, [InjectFailure("agent_crash", "researcher_a"),
                                   InjectFailure("agent_crash", "critic")])
    rows = {r.failure: r for r in rep.rows}
    # researcher_c silently covers for researcher_a: partial resilience despite zero observed contribution
    assert rows["agent_crash:researcher_a"].resilience > rows["agent_crash:critic"].resilience
    assert 0.0 <= rep.resilience <= 1.5
    assert "FAILURE INJECTION" in rep.table()


def test_compare_and_frontier(runner):
    base = runner.baseline()
    single = runner.run(SetParam("single_agent", True), label="single")
    comp = compare(base, single)
    assert comp.n_pairs == len(base.runs)
    assert comp.cost_delta < 0
    pts = frontier([base, single])
    assert any(p.on_frontier for p in pts)
    assert pts[0].cost <= pts[-1].cost


def test_propose_and_validate(runner):
    base = runner.baseline()
    cands = propose(base, {"coordinator_bottleneck": {"direct_channels": True}})
    names = [c.name for c in cands]
    assert any("remove researcher_c" in n for n in names)
    rep = validate(cands, runner, min_gain=0.02)
    rc = next(c for c in rep.candidates if "remove researcher_c" in c.name)
    assert rc.measured is not None and rc.verdict.startswith("accept")
    assert "CANDIDATE" in rep.table()


def test_evaluate_with_interventions_and_report():
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock,
                   interventions=[RemoveAgent("critic")], label="no-critic")
    assert res.label == "no-critic" and "critic" not in res.agents
    text = res.report()
    assert "SWARM EVALUATION" in text and "no-critic" in text
    html = res.to_html()
    assert "<svg" in html and "Diagnostics" in html
