import threading
import time

import pytest

from swarmeval import (AgentDisabled, BudgetExceeded, FailureSpec, InjectedAgentCrash, InjectedToolError, LLMResult,
                       RunContext, SimClock, SwarmConfig, Task, ToolDisabled, WallClock)
from swarmeval.schema import EventType


def make_ctx(config=None, clock=None):
    return RunContext(Task("t1", "do the thing", expected="42"), config or SwarmConfig(), clock=clock or SimClock(),
                      seed=1)


def test_spans_llm_tool_message_artifact_are_recorded():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a", role="worker", model="claude-sonnet-5") as a:
        out = a.llm(lambda p: LLMResult("hello", 100, 20), "prompt", prompt="prompt")
        assert out == "hello"
        r = a.call_tool("calc", lambda x, y: x + y, 1, 2)
        assert r == 3
        art = a.produce("notes", "some notes here")
        assert a.send("b", "take this", kind="result", artifacts=[art]) is not None
    with ctx.agent("b", role="writer") as b:
        msgs = b.receive()
        assert len(msgs) == 1 and msgs[0].artifacts == [art.artifact_id]
        b.consume(art)
        with b.llm_call(model="claude-opus-5", input_artifacts=[art]) as call:
            ctx.clock.advance(1.5)
            call.input_tokens, call.output_tokens, call.output = 500, 50, "final"
    ctx.set_output("final", artifacts=[art])
    traj = ctx.finish()
    assert traj.status == "success"
    assert len(traj.spans) == 2
    llm = traj.llm_calls()
    assert [e.agent_id for e in llm] == ["a", "b"]
    assert llm[0].input_tokens == 100 and llm[0].output_tokens == 20 and llm[0].cost_usd is not None
    assert llm[1].latency_ms == pytest.approx(1500.0)
    assert llm[1].input_artifacts == [art.artifact_id]
    tool = traj.tool_calls()[0]
    assert tool.tool_name == "calc" and tool.tool_args == {"args": [1, 2]} and tool.content == 3
    msg = traj.messages()[0]
    assert msg.sender == "a" and msg.receiver == "b" and msg.status == "success"
    assert art.artifact_id in traj.artifacts
    assert traj.total_tokens == 670


def test_nested_spans_have_parent():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("coord") as c:
        with c.sub_agent("child") as ch:
            assert ch.parent is c
    spans = list(ctx.finish().spans.values())
    child = next(s for s in spans if s.agent_id == "child")
    parent = next(s for s in spans if s.agent_id == "coord")
    assert child.parent_span_id == parent.span_id


def test_disabled_agent_and_tool_raise():
    cfg = SwarmConfig(disabled_agents={"x"}, disabled_tools={"search"})
    ctx = make_ctx(cfg)
    ctx.start()
    assert not ctx.is_enabled("x")
    with pytest.raises(AgentDisabled):
        with ctx.agent("x"):
            pass
    with ctx.agent("y") as y:
        with pytest.raises(ToolDisabled):
            y.call_tool("search", lambda: 1)
    traj = ctx.finish()
    kinds = {i["kind"] for i in traj.interventions}
    assert {"agent_disabled", "tool_disabled"} <= kinds
    assert any(e.status == "disabled" for e in traj.tool_calls())


def test_channel_disabled_and_message_drop():
    cfg = SwarmConfig(disabled_channels={"a->b"}, failures=[FailureSpec("message_drop", "a->c")])
    ctx = make_ctx(cfg)
    ctx.start()
    with ctx.agent("a") as a:
        assert a.send("b", "hi") is None
        assert a.send("c", "hi") is None
        assert a.send("d", "hi") is not None
    assert ctx.receive("b") == [] and ctx.receive("c") == [] and len(ctx.receive("d")) == 1
    statuses = sorted(m.status for m in ctx.finish().messages())
    assert statuses == ["disabled", "dropped", "success"]


def test_failure_injection_crash_and_malformed_tool():
    cfg = SwarmConfig(failures=[FailureSpec("agent_crash", "critic"),
                                FailureSpec("tool_malformed", "search", payload={"garbage": True}),
                                FailureSpec("tool_error", "fetch", on_call=2)])
    ctx = make_ctx(cfg)
    ctx.start()
    with pytest.raises(InjectedAgentCrash):
        with ctx.agent("critic"):
            pass
    with ctx.agent("w") as w:
        assert w.call_tool("search", lambda q: ["real"], q="x") == {"garbage": True}
        assert w.call_tool("fetch", lambda: "ok") == "ok"
        with pytest.raises(InjectedToolError):
            w.call_tool("fetch", lambda: "ok")
        assert w.call_tool("fetch", lambda: "ok") == "ok"
    traj = ctx.finish()
    critic_span = next(s for s in traj.spans.values() if s.agent_id == "critic")
    assert critic_span.status == "injected"
    assert any(e.status == "injected" and e.tool_name == "search" for e in traj.tool_calls())
    assert sum(1 for e in traj.tool_calls() if e.tool_name == "fetch" and e.status == "injected") == 1


def test_token_budget_enforced():
    ctx = make_ctx(SwarmConfig(token_budgets={"a": 150}))
    ctx.start()
    with ctx.agent("a") as a:
        a.llm(lambda: LLMResult("x", 100, 20))
        a.llm(lambda: LLMResult("y", 100, 20))  # cumulative now 240 >= 150 for the next call
        with pytest.raises(BudgetExceeded):
            a.llm(lambda: LLMResult("z", 1, 1))
    assert ctx.tokens_used("a") == 240


def test_model_override_and_params():
    ctx = make_ctx(SwarmConfig(model_overrides={"a": "claude-haiku-4-5"}, params={"k": 3}))
    ctx.start()
    with ctx.agent("a", model="claude-opus-5") as a:
        assert a.model == "claude-haiku-4-5"
        a.llm(lambda: LLMResult("x", 10, 10))
    assert ctx.param("k") == 3
    assert ctx.finish().llm_calls()[0].model == "claude-haiku-4-5"


def test_parallel_propagates_span_and_wall_clock_overlaps():
    ctx = make_ctx(clock=WallClock())
    ctx.start()

    def worker(name):
        with ctx.agent(name) as w:
            with w.llm_call(model="claude-sonnet-5") as call:
                time.sleep(0.05)
                call.input_tokens = 10
            return threading.current_thread().name

    with ctx.agent("coord") as c:
        names = ctx.parallel(lambda: worker("w1"), lambda: worker("w2"))
    traj = ctx.finish()
    assert len(set(names)) == 2
    children = [s for s in traj.spans.values() if s.agent_id in ("w1", "w2")]
    parent = next(s for s in traj.spans.values() if s.agent_id == "coord")
    assert all(s.parent_span_id == parent.span_id for s in children)
    # the two workers overlapped in time
    a, b = sorted(children, key=lambda s: s.start)
    assert b.start < a.end


def test_error_inside_span_is_recorded_and_reraised():
    ctx = make_ctx()
    ctx.start()
    with pytest.raises(ValueError):
        with ctx.agent("a") as a:
            with a.llm_call(model="m"):
                raise ValueError("boom")
    traj = ctx.finish("error", "boom")
    assert traj.llm_calls()[0].status == "error"
    assert any(e.event_type == EventType.ERROR.value for e in traj.events)
    assert list(traj.spans.values())[0].status == "error"


def test_trajectory_roundtrip():
    ctx = make_ctx()
    ctx.start()
    with ctx.agent("a") as a:
        art = a.produce("n", "content")
        a.send("b", "m", artifacts=[art])
    traj = ctx.finish()
    from swarmeval.schema import Trajectory
    back = Trajectory.from_dict(traj.to_dict())
    assert len(back.events) == len(traj.events)
    assert back.artifacts[art.artifact_id].content == "content"
    assert back.spans.keys() == traj.spans.keys()
