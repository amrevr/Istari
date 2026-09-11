# SwarmEval

> Evaluation, observability, diagnostics and causal analysis for multi-agent AI systems.

SwarmEval evaluates a multi-agent system not only on whether it produced the right answer, but on **how the agents worked together to produce it**, and, through counterfactual experiments, on **which agents actually mattered**.

```
Trajectory Observability  +  Causal Ablation
              ↓
       Coordination Analysis
              ↓
          Diagnostics
              ↓
        Optimization
```

* **Observe** – every agent invocation, LLM call, tool call, message, artifact and wait is recorded as a structured event.
* **Measure** – task performance, cost, latency, critical path, utilization, communication, redundancy, dependencies, parallelism, context bloat.
* **Diagnose** – swarm smells (coordinator bottleneck, dead agent, ping-pong, under-specialization, context bloat, premature synthesis, …) reported as *observation → hypothesis → intervention to test*.
* **Intervene** – remove agents/tools/channels, swap models, cap budgets, inject failures; re-run on the same seeds; compare with paired confidence intervals.
* **Optimize** – turn diagnostics into candidate architectures, then *validate* them experimentally before calling them improvements.

Pure Python 3.9+, no runtime dependencies. Optional Anthropic SDK adapter.

The full design document is in [docs/DESIGN.md](docs/DESIGN.md).

---

## Install

```bash
pip install -e ".[dev]"        # from this repo
pip install -e ".[anthropic]"  # optional: Anthropic SDK adapter
```

## 60-second demo

The repo ships a simulated research swarm (no API keys) with deliberate flaws:

```bash
swarmeval demo --trials 3 --out runs/demo
```

Observation layer:

```
╭──────────────────────────────────────────────────────────────╮
│                       SWARM EVALUATION                       │
├──────────────────────────────────────────────────────────────┤
│ Task Success                        83.3% [64.1, 93.3]       │
│ Cost / Successful Task                              $0.17    │
│ Critical Path                                       46.9s    │
│ Communication / Work                                 3.7%    │
│ Redundant Work                                      24.6% ⚠  │
│ Context-bloated calls / run                           2.5 ⚠  │
╰──────────────────────────────────────────────────────────────╯

⚠ Zero-value agent [dead_agent] — in 100% of runs
  Observed: researcher_c performed 4 actions and 529 tokens but nothing it
            produced was used downstream or appeared in the final output.
  Hypothesis: The agent's output does not influence the result.
  Potential intervention: Ablate researcher_c and compare task success.
```

Causal layer (same swarm, same seeds, one component removed at a time):

```
ABLATION vs baseline
  component        Δ success      95% CI    Δ cost  Δ latency  observed
  critic             -50.0p    [-71, -29]      -21%       -31%    100.0%
  researcher_c        +0.0p      [+0, +0]       -3%        +0%      0.0%
  researcher_a        +8.3p     [-8, +25]       -3%        +0%    100.0%

FAILURE INJECTION
  failure                     success  resilience
  agent_crash:critic             0.0%          0%
  agent_crash:researcher_a      91.7%        110%   ← researcher_c silently covers
```

`researcher_c` looks dead in every trajectory and removing it changes nothing, yet failure injection shows it provides real resilience when `researcher_a` crashes. **Activity is not contribution, and observation is not causation** – SwarmEval keeps the two apart.

The demo also writes `baseline.html` (execution graph, timeline, diagnostics), `baseline.events.jsonl` (one event per line) and `baseline.otel.json` (OpenTelemetry-style spans with `gen_ai.*` attributes).

---

## Instrumenting a swarm

A swarm is anything with `run(task, ctx)`. Everything the swarm does goes through the `RunContext`, which is both the recorder and the enforcement point for interventions.

```python
from swarmeval import Swarm, Task, Benchmark, evaluate, evaluators

class MySwarm(Swarm):
    def run(self, task, ctx):
        with ctx.agent("planner", role="coordinator", model="claude-opus-5") as planner:
            plan = planner.llm(call_model, f"Plan: {task.input}", prompt=task.input)  # LLMResult carries usage
            planner.send("researcher", plan, kind="task_assignment")

        with ctx.agent("researcher", role="research", model="claude-sonnet-5") as r:
            assignment = r.receive()[0]
            hits = r.call_tool("web_search", search, query=assignment.content)
            notes = r.produce("notes", summarise(hits))
            r.send("writer", notes.content, kind="result", artifacts=[notes])

        with ctx.agent("writer", role="synthesizer") as w:
            msg = w.receive()[0]
            with w.llm_call(input_artifacts=msg.artifacts, prompt=msg.content) as call:
                answer = call_model(msg.content)          # any client
                call.input_tokens, call.output_tokens = answer.usage.input, answer.usage.output
                call.output = answer.text
            final = w.produce("answer", answer.text)
        ctx.set_output(answer.text, artifacts=[final])
        return answer.text

bench = Benchmark("qa", [Task("q1", "What is …?", expected=["keyword"],
                              evaluator={"type": "contains_all", "keywords": ["keyword"]})])
result = evaluate(MySwarm(), bench, trials=3)
print(result.report())
result.to_html("report.html")
```

Key instrumentation calls:

| Call | Records |
|---|---|
| `ctx.agent(id, role, model)` | an agent invocation span (nest them for hierarchies; `ctx.parallel(...)` for threads) |
| `span.llm(fn, …)` / `span.llm_call(...)` | an LLM call with model, tokens, cost, latency, prompt, `input_artifacts` |
| `span.call_tool(name, fn, **args)` / `span.tool_call(...)` | a tool call with arguments and result |
| `span.send(to, content, kind, artifacts)` / `span.receive()` | messages through a mailbox (so dropped messages really are dropped) |
| `span.produce(name, content)` / `span.consume(artifact)` | artifacts and provenance |
| `span.wait(reason, seconds)` | explicit waiting |
| `span.rng` | a per-agent random stream (keeps ablations comparable) |

Record `input_artifacts` consistently: provenance is what lets SwarmEval tell "used downstream" from "looks similar to something that was used".

**Anthropic SDK**: `swarmeval.integrations.anthropic.instrument_client(client)` records every `messages.create` made inside an agent span with real usage; `anthropic_llm(...)` gives a prompt→`LLMResult` callable and `anthropic_judge(...)` a judge for `LLMJudge`.

---

## Evaluators

Deterministic: `ExactMatch`, `ContainsAll`/`ContainsAny` (with `forbidden`), `Regex`, `NumericTolerance`, `ToolUseEvaluator`, `Groundedness`, `Constraints`, `Composite`, or any callable. All are JSON-serialisable specs (`{"type": "contains_all", "keywords": [...]}`) so benchmarks are plain files.

LLM-based: `LLMJudge(judge_fn | [(name, fn), ...], rubric, n_judgments)` – majority vote, mean score, inter-judge agreement and a confidence estimate; failing judges are recorded, not fatal.

Validity checks run automatically: repeated trials with Wilson/bootstrap confidence intervals, per-task repeatability, evaluator agreement, and **leakage detection** (an expected-answer fragment appearing in an agent's prompt or tool arguments *before* any tool result supplied it).

---

## What gets measured

| Layer | Metrics |
|---|---|
| **Performance** | success rate + CI, score, constraint pass rate, per-task/per-trial breakdown, run crashes |
| **Efficiency** | tokens, LLM/tool calls, cost (pricing table, "unknown" rather than $0 for unpriced models), cost per success, wall clock, critical path, total work, agent compute vs waiting time |
| **Coordination** | per-agent utilization and *observed* contribution (provenance-based), dead agents, communication ratio and category split (useful work / information transfer / coordination), hub share, redundancy (duplicate tool calls, similar artifacts, accidental vs intentional verification), dependency depth, fan-in/out, cycles, serial bottlenecks, unused channels, actual vs potential parallelism, context bloat |
| **Causality** | Δ success/score/cost/latency with paired bootstrap CIs per intervention, resilience under injected failures, Pareto efficiency frontier |

Critical path and potential parallelism come from a work-unit DAG whose edges are inferred from recorded dependencies (sequential order within an agent, messages, artifact provenance, nesting). If a swarm passes information outside the recorder the DAG under-counts dependencies; instrument the channel.

---

## Diagnostics (swarm smells)

`redundant_delegation`, `agent_ping_pong`, `over_agentization`, `under_specialization`, `coordinator_bottleneck`, `premature_synthesis` / `late_information`, `context_bloat`, `dead_agent`, plus `serialized_execution` and `communication_overhead`. Thresholds live in `SmellConfig`. Each diagnostic carries prevalence across runs, evidence, and a recommendation phrased as an experiment.

---

## Causal experiments

```python
from swarmeval import (CounterfactualRunner, RemoveAgent, SetModel, SetParam, InjectFailure,
                       ablate_agents, inject_failures, frontier, compare)

runner = CounterfactualRunner(make_swarm, bench, trials=5, seed=0)
ablate_agents(runner).table()                         # remove each agent, re-run, paired comparison
inject_failures(runner, [InjectFailure("agent_crash", "critic"),
                         InjectFailure("tool_error", "web_search", probability=0.5),
                         InjectFailure("message_drop", "coordinator->critic")]).table()
weak = runner.run(SetModel("*", "claude-haiku-4-5"), label="weak swarm")
solo = runner.run(SetParam("single_agent", True), label="single agent")
frontier([runner.baseline(), weak, solo])             # Pareto frontier of success vs cost
compare(runner.baseline(), weak)                      # deltas with 95% CIs
```

Interventions: `RemoveAgent`, `RemoveTool`, `DisableChannel`, `SetModel`, `SetTokenBudget`, `SetContextLimit`, `SetParam` (swarm-specific knobs read via `ctx.param`), `InjectFailure` (`agent_crash`, `agent_delay`, `llm_error`, `llm_delay`, `tool_error`, `tool_malformed`, `tool_delay`, `message_drop`, `message_corrupt`; with probability or Nth-call targeting), `Compose`.

Disabled agents/tools raise `AgentDisabled`/`ToolDisabled` (subclasses of `InterventionError`); injected failures raise `InjectedFailure` subclasses that swarms should treat like real runtime errors. Every fired intervention is recorded in the trajectory.

### From diagnosis to validated optimization

```python
from swarmeval.optimize import propose, validate
candidates = propose(result, {"coordinator_bottleneck": {"direct_channels": True}})
print(validate(candidates, runner).table())
```

```
CANDIDATE ARCHITECTURES vs baseline
  candidate                    pred Δcost pred Δlat  Δ success  Δ cost  Δ lat  verdict
  remove researcher_c                 -3%       -0%     +0.0pp     -3%    +0%  accept: cost -3%, success +0.0 pp (n.s.)
  direct channels (bypass hub)       -13%      -12%    +16.7pp    -20%   -12%  accept: success +16.7 pp
```

Predictions are labelled hypotheses; verdicts come only from measurement.

---

## CLI

```bash
swarmeval demo [--trials N] [--out DIR] [--quick]
swarmeval run     --swarm pkg.mod:factory --benchmark bench.json --trials 3 --out runs --html
swarmeval report  runs/baseline.json --html report.html
swarmeval ablate  --swarm ... --benchmark ... --agents critic,researcher_c
swarmeval inject  --swarm ... --benchmark ... --failure agent_crash:critic --failure tool_error:web_search:0.5
swarmeval compare runs/baseline.json runs/variant.json
```

`--swarm` accepts a `Swarm` instance, a `factory(config) -> Swarm`, or a bare `run(task, ctx)` function. `--benchmark` accepts a JSON/JSONL file or `pkg.mod:builder`. `--clock sim` uses the virtual clock for simulated swarms.

## Outputs

* `EvaluationResult.report()` – text health report; `.to_html(path)` – standalone HTML with execution graph, timeline (critical path highlighted), diagnostics and causal tables; `.to_dict()` / `.save()` – JSON.
* `swarmeval.export.write_events_jsonl` – flat event stream; `swarmeval.export.to_otel` – OTLP/JSON spans.
* `RunSet.save/load` – trajectories are the source of truth; every analysis can be recomputed from them (`swarmeval report`).

---

## Design principles

1. **Observe before optimizing.** If it is not a structured event, it is not analysable.
2. **Activity is not contribution.** Utilization is reported as observed; contribution claims require ablation.
3. **Correlation is not causation.** Smells are hypotheses; the causal layer validates them.
4. **Recommendations are hypotheses** until a paired experiment says otherwise.
5. **Preserve underlying measurements.** No single opaque health score.
6. **Compare against strong baselines** (single agent, smaller swarms, other models).
7. **Measure the trade-offs** – success, cost and latency move together.
8. **Reproduce.** Seeds derive from `(seed, task, trial)`; each agent gets its own random stream so ablating one agent does not perturb the others.

## Layout

```
swarmeval/
  schema.py        events, artifacts, spans, tasks, trajectories
  recorder.py      RunContext / AgentSpan: recording + intervention enforcement
  config.py        SwarmConfig, FailureSpec
  runner.py        SwarmRunner, RunSet, repeated trials, seeds
  evaluators.py    deterministic + LLM judges, JSON specs
  graph/           agent graph, work-unit DAG, critical path
  metrics/         performance, efficiency, coordination
  diagnostics/     swarm smells, cross-run aggregation
  causal/          interventions, counterfactual runner, ablation, failure injection, frontier
  optimize/        candidate generation + experimental validation
  report/          text and HTML reports
  export/          JSONL and OpenTelemetry export
  integrations/    Anthropic SDK adapter
  validity.py      leakage detection, repeatability
  stats.py         Wilson/bootstrap CIs, agreement, text similarity
  cli.py
examples/research_swarm.py   simulated swarm + benchmark used by the demo and tests
tests/
```

## Status

Phases 1–3 of the design (observability & evaluation, coordination analysis, causal & research features) are implemented; Phase 4 (optimization) is implemented as candidate generation plus experimental validation. Not yet done: framework adapters beyond the Anthropic SDK, a learned cost/latency predictor, and automatic candidate-architecture search.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```
