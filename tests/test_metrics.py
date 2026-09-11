"""Hand-built trajectories with known answers for the graph and metric layers."""
import pytest

from swarmeval import RunContext, SimClock, SwarmConfig, Task, analyze_trajectory
from swarmeval.graph import build_graph
from swarmeval.metrics import compute_coordination, compute_efficiency


def fan_out_trajectory(parallel=True, use_c=True):
    """coordinator -> (a, b, c in parallel) -> synthesizer.  c's notes are ignored when use_c=False."""
    clock = SimClock()
    ctx = RunContext(Task("t", "question"), SwarmConfig(), clock=clock, seed=0)
    ctx.start()
    with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as co:
        with co.llm_call() as call:
            clock.advance(1.0)
            call.input_tokens, call.output_tokens = 100, 50
        for w in ("a", "b", "c"):
            co.send(w, f"research topic {w}", kind="task_assignment")
    t0 = clock.now()
    ends = []
    arts = {}
    for w in ("a", "b", "c"):
        if parallel:
            clock.set(t0)
        with ctx.agent(w, role="research", model="claude-sonnet-5") as ag:
            ag.receive()
            ag.call_tool("search", lambda q: (clock.advance(0.5), f"result {q}")[1], q="topic" if w != "b" else "other")
            with ag.llm_call() as call:
                clock.advance(2.0)
                call.input_tokens, call.output_tokens = 200, 100
            arts[w] = ag.produce(f"notes_{w}", f"notes about {'topic' if w != 'b' else 'other'} findings " * 5)
            ag.send("synthesizer", "notes", kind="result", artifacts=[arts[w]])
        ends.append(clock.now())
    if parallel:
        clock.set(max(ends))
    with ctx.agent("synthesizer", role="synthesizer", model="claude-opus-5") as sy:
        sy.receive()
        used = [arts["a"], arts["b"]] + ([arts["c"]] if use_c else [])
        for a in used:
            sy.consume(a)
        with sy.llm_call(input_artifacts=used) as call:
            clock.advance(3.0)
            call.input_tokens, call.output_tokens, call.output = 5000, 300, "final answer"
        final = sy.produce("final", "final answer")
    ctx.set_output("final answer", artifacts=[final])
    return ctx.finish()


def test_critical_path_and_parallelism_parallel():
    traj = fan_out_trajectory(parallel=True)
    g = build_graph(traj)
    cp, path = g.critical_path()
    # coordinator llm (1.0) -> worker search (0.5) + llm (2.0) -> synthesizer (3.0) = 6.5
    assert cp == pytest.approx(6.5)
    assert g.total_work() == pytest.approx(1.0 + 3 * 2.5 + 3.0)
    eff = compute_efficiency(traj, g)
    assert eff.wall_clock_s == pytest.approx(6.5)
    coord = compute_coordination(traj, g)
    par = coord.parallelism
    assert par.actual_parallel_fraction == pytest.approx(1 - 6.5 / 11.5)
    assert par.potential_parallel_fraction == pytest.approx(par.actual_parallel_fraction)
    assert par.max_concurrency == 3
    assert eff.waiting_time_s == pytest.approx(0.0)


def test_sequential_execution_shows_opportunity():
    traj = fan_out_trajectory(parallel=False)
    g = build_graph(traj)
    cp, _ = g.critical_path()
    assert cp == pytest.approx(6.5)              # dependencies unchanged
    assert traj.duration == pytest.approx(11.5)  # but executed serially
    coord = compute_coordination(traj, g)
    assert coord.parallelism.actual_parallel_fraction == pytest.approx(0.0)
    assert coord.parallelism.opportunity > 0.4
    analysis = analyze_trajectory(traj)
    assert "serialized_execution" in {d.smell for d in analysis.diagnostics}


def test_dead_agent_and_redundancy_detection():
    traj = fan_out_trajectory(use_c=False)
    analysis = analyze_trajectory(traj)
    util = analysis.coordination.utilization
    assert util["c"].dead is True
    assert util["a"].dead is False and util["b"].dead is False
    assert util["c"].observed_contribution == 0.0
    assert util["synthesizer"].observed_contribution == 1.0
    smells = {d.smell for d in analysis.diagnostics}
    assert "dead_agent" in smells
    red = analysis.coordination.redundancy
    assert red.duplicate_tool_calls == 1          # c repeated a's search
    assert red.redundant_actions >= 2             # duplicate search + similar notes artifact
    assert red.highest_overlap[0:2] == ("a", "c")
    assert "under_specialization" in smells
    # provenance: c is not in the final lineage even though its content looks like a's
    assert util["c"].downstream_uses == 0


def test_agent_used_when_consumed():
    traj = fan_out_trajectory(use_c=True)
    analysis = analyze_trajectory(traj)
    assert analysis.coordination.utilization["c"].dead is False
    assert "dead_agent" not in {d.smell for d in analysis.diagnostics}


def test_context_bloat_detected():
    traj = fan_out_trajectory()
    analysis = analyze_trajectory(traj)
    cb = analysis.coordination.context_bloat
    assert cb.bloated_calls == 1 and cb.worst[0]["agent"] == "synthesizer"
    assert "context_bloat" in {d.smell for d in analysis.diagnostics}


def test_communication_and_dependency_metrics():
    traj = fan_out_trajectory()
    analysis = analyze_trajectory(traj)
    com = analysis.coordination.communication
    assert com.messages == 6 and com.coordination_messages == 3 and com.information_messages == 3
    assert 0.0 < com.communication_ratio < 1.0
    dep = analysis.coordination.dependency
    assert dep.fan_out["coordinator"] == 3
    assert dep.fan_in["synthesizer"] == 3
    assert dep.cycles == []
    assert dep.depth >= 3


def test_ping_pong_and_late_information():
    clock = SimClock()
    ctx = RunContext(Task("t", "q"), SwarmConfig(), clock=clock)
    ctx.start()
    for i in range(4):
        with ctx.agent("writer", role="writer") as w:
            w.receive()
            with w.llm_call() as c:
                clock.advance(1.0)
                c.input_tokens = 10
            w.send("critic", f"draft {i}", kind="request")
        with ctx.agent("critic", role="critic") as cr:
            cr.receive()
            with cr.llm_call() as c:
                clock.advance(1.0)
                c.input_tokens = 10
            cr.send("writer", f"feedback {i}", kind="feedback")
    ctx.set_output("done")
    traj = ctx.finish()
    smells = {d.smell for d in analyze_trajectory(traj).diagnostics}
    assert "agent_ping_pong" in smells
