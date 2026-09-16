import pytest

from swarmeval import Task, evaluators as E


def T(**kw):
    return Task("t", "input", **kw)


def test_contains_all_and_forbidden():
    ev = E.ContainsAll(["alpha", "beta"], forbidden=["gamma"])
    r = ev.evaluate(T(), "Alpha and BETA are here")
    assert r.success and r.score == 1.0
    r = ev.evaluate(T(), "alpha only")
    assert not r.success and r.score == pytest.approx(0.5) and r.details["missing"] == ["beta"]
    r = ev.evaluate(T(), "alpha beta gamma")
    assert not r.success and r.constraints_passed is False


def test_exact_regex_numeric():
    assert E.ExactMatch().evaluate(T(expected="Paris"), " paris ").success
    assert E.Regex(r"\bok\b").evaluate(T(), "all OK here").success
    assert E.NumericTolerance(tolerance=0.01).evaluate(T(expected=3.14), "pi is about 3.141").success
    assert not E.NumericTolerance().evaluate(T(expected=3), "no numbers").success


def test_composite_and_constraints_and_spec_roundtrip():
    spec = {"type": "composite", "threshold": 0.6, "required": ["regex"],
            "evaluators": [{"type": "contains_all", "keywords": ["a", "b"]}, {"type": "regex", "pattern": "done"}]}
    ev = E.from_spec(spec)
    assert isinstance(ev, E.Composite)
    assert ev.evaluate(T(), "a b done").success
    assert not ev.evaluate(T(), "a b").success       # required regex fails
    assert E.from_spec(ev.to_spec()).evaluate(T(), "a b done").success
    cons = E.from_spec({"type": "constraints", "checks": [{"type": "regex", "pattern": "x"}, {"type": "regex", "pattern": "y"}]})
    assert cons.evaluate(T(), "xy").constraints_passed is True
    assert cons.evaluate(T(), "x").constraints_passed is False


def test_llm_judge_majority_and_agreement():
    judges = [("j1", lambda p: '{"score": 0.9, "success": true, "rationale": "good"}'),
              ("j2", lambda p: '{"score": 0.8, "success": true}'),
              ("j3", lambda p: "score: 0.2 -- this fails")]
    ev = E.LLMJudge(judges)
    r = ev.evaluate(T(success_criteria="be right"), "output")
    assert r.success and len(r.judgments) == 3
    assert r.agreement == pytest.approx(1 / 3)
    assert 0 < r.confidence <= 1
    assert r.score == pytest.approx((0.9 + 0.8 + 0.2) / 3)
    # a crashing judge is recorded, not fatal
    ev2 = E.LLMJudge(lambda p: (_ for _ in ()).throw(RuntimeError("down")), n_judgments=2)
    r2 = ev2.evaluate(T(), "x")
    assert not r2.success and all("judge error" in j.rationale for j in r2.judgments)


def test_parse_judgment_fallbacks():
    assert E.parse_judgment("PASS - correct")["success"]
    assert not E.parse_judgment("garbage")["success"]
    assert E.parse_judgment('prefix {"score": 1.7} suffix')["score"] == 1.0


def test_callable_and_unknown_spec():
    ev = E.from_spec(lambda task, out, traj: out == "yes")
    assert ev.evaluate(T(), "yes").success
    with pytest.raises(ValueError):
        E.from_spec({"type": "nope"})
