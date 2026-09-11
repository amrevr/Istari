import json
import os

from examples.research_swarm import build_benchmark, make_swarm
from swarmeval import EvaluationResult, SimClock, evaluate
from swarmeval.cli import main
from swarmeval.export import read_events_jsonl, to_otel, write_events_jsonl
from swarmeval.runner import RunSet


def test_runset_and_result_roundtrip(tmp_path):
    res = evaluate(make_swarm, build_benchmark(2), trials=1, clock_factory=SimClock)
    p = tmp_path / "r.json"
    res.save(str(p))
    back = EvaluationResult.load(str(p))
    assert back.success_rate == res.success_rate
    assert back.smells == res.smells
    rs = RunSet.load(str(p)) if False else RunSet.from_dict(json.load(open(p))["runs"])
    assert len(rs) == 2 and rs.config.label == "baseline"


def test_jsonl_and_otel(tmp_path):
    res = evaluate(make_swarm, build_benchmark(1), trials=1, clock_factory=SimClock)
    traj = res.runs.trajectories[0]
    p = tmp_path / "e.jsonl"
    n = write_events_jsonl(traj, str(p))
    assert n == len(traj.events) == len(read_events_jsonl(str(p)))
    otel = to_otel(traj)
    spans = otel["resourceSpans"][0]["scopeSpans"][0]["spans"]
    names = {s["name"] for s in spans}
    assert "agent:coordinator" in names and "llm_call" in names and "tool:web_search" in names
    llm = next(s for s in spans if s["name"] == "llm_call")
    keys = {a["key"] for a in llm["attributes"]}
    assert {"gen_ai.request.model", "gen_ai.usage.input_tokens"} <= keys
    json.dumps(otel)  # serialisable


def test_cli_demo_quick_and_run(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--quick", "--trials", "1", "--out", str(out)]) == 0
    assert (out / "baseline.html").exists() and (out / "baseline.events.jsonl").exists()
    assert main(["run", "--swarm", "examples.research_swarm:make_swarm", "--benchmark",
                 "examples.research_swarm:build_benchmark", "--trials", "1", "--clock", "sim",
                 "--label", "cli", "--out", str(tmp_path / "run")]) == 0
    assert main(["report", str(tmp_path / "run" / "cli.json")]) == 0
    assert main(["compare", str(out / "baseline.json"), str(tmp_path / "run" / "cli.json")]) == 0
    captured = capsys.readouterr()
    assert "SWARM EVALUATION" in captured.out and "baseline  →  cli" in captured.out
    assert main(["inject", "--swarm", "examples.research_swarm:make_swarm", "--benchmark",
                 "examples.research_swarm:build_benchmark", "--trials", "1", "--clock", "sim",
                 "--failure", "agent_crash:critic"]) == 0
