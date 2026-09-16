"""A simulated multi-agent research swarm -- no API keys required.

The swarm is deliberately imperfect so the evaluation has something to find:

* ``coordinator`` relays every message (bottleneck, context bloat),
* ``researcher_a`` and ``researcher_c`` are given the same scope (redundancy),
* ``researcher_c``'s notes are ignored unless ``researcher_a`` is missing
  (zero observed value in normal runs, but real resilience value),
* ``critic`` catches planted wrong facts (removing it hurts correctness),
* ``synthesizer`` receives the whole history instead of the artifacts it needs.

Configuration knobs (``SetParam``):
    direct_channels   researchers/critic talk to their consumers directly
    partition_scope   give researcher_c its own scope (source s5)
    parallel_research run researchers concurrently (default True)
    single_agent      one agent does everything
    revision_rounds   max critic<->synthesizer revision rounds (default 2)

Models can be swapped per agent with ``SetModel``; a weaker model finds fewer
facts and catches fewer errors.

The simulated clock (``SimClock``) is advanced explicitly so runs are instant
but timings look real.
"""
from __future__ import annotations

import random
from functools import partial
from typing import Any, Dict, List, Optional

from swarmeval import (AgentDisabled, Benchmark, InjectedFailure, InterventionError, RunContext, SimClock, Swarm,
                       SwarmConfig, Task)
from swarmeval.schema import estimate_tokens

MODEL_QUALITY = {"claude-opus-5": 0.985, "claude-sonnet-5": 0.96, "claude-haiku-4-5": 0.85}
WRONG_FACT_PICKUP = 0.5   # how often a researcher records the planted wrong fact
MODEL_SPEED = {"claude-opus-5": 55.0, "claude-sonnet-5": 90.0, "claude-haiku-4-5": 160.0}   # output tok/s

TOPICS = [
    ("Aurora protocol", "aurora"), ("Helix compiler", "helix"), ("Quasar storage", "quasar"),
    ("Lumen scheduler", "lumen"), ("Tidal cache", "tidal"), ("Basalt runtime", "basalt"),
    ("Meridian mesh", "meridian"), ("Cinder allocator", "cinder"), ("Sable index", "sable"),
    ("Orchid planner", "orchid"),
]

_FACT_TEMPLATES = [
    "{name} was first released in {year}",
    "{name} is maintained by the {org} working group",
    "{name} version {ver} added {feature}",
    "{name} is licensed under the {lic} license",
    "{name} processes roughly {n} requests per second per node",
    "{name} depends on the {dep} library",
]
_WRONG_TEMPLATE = "{name} was deprecated in {year}"


def build_world(n_tasks: int = 8, seed: int = 7) -> Dict[str, Dict[str, Any]]:
    """Deterministic knowledge base: each task has five sources with facts and
    one planted wrong fact in source s3."""
    rng = random.Random(seed)
    world: Dict[str, Dict[str, Any]] = {}
    for i in range(n_tasks):
        name, slug = TOPICS[i % len(TOPICS)]
        year = rng.randint(2012, 2023)
        vals = {"name": name, "year": year, "org": rng.choice(["Tern", "Kestrel", "Ibis", "Heron"]),
                "ver": f"{rng.randint(1, 5)}.{rng.randint(0, 9)}",
                "feature": rng.choice(["streaming replication", "incremental snapshots", "zero-copy IO", "adaptive batching"]),
                "lic": rng.choice(["Apache-2.0", "MIT", "BSD-3", "MPL-2.0"]),
                "n": rng.choice([12000, 48000, 90000, 250000]),
                "dep": rng.choice(["libwarp", "zephyrio", "corundum", "nimbus-rt"])}
        facts = [t.format(**vals) for t in _FACT_TEMPLATES]
        wrong = _WRONG_TEMPLATE.format(name=name, year=year + rng.randint(1, 3))
        extra = f"{name} has an experimental {rng.choice(['WASM', 'GPU', 'RISC-V'])} backend"
        sources = {"s1": [facts[0], facts[1]], "s2": [facts[2]], "s3": [facts[3], wrong], "s4": [facts[4], facts[5]],
                   "s5": [extra]}
        keywords = [vals["org"], str(vals["ver"]), vals["feature"], vals["lic"], str(vals["n"]), vals["dep"]]
        world[f"task_{i+1:02d}"] = {"name": name, "slug": slug, "sources": sources, "required": facts,
                                    "keywords": keywords, "wrong": wrong,
                                    "wrong_keyword": "deprecated", "extra": extra}
    return world


WORLD = build_world()


def build_benchmark(n_tasks: int = 8) -> Benchmark:
    tasks = []
    for tid, w in list(WORLD.items())[:n_tasks]:
        tasks.append(Task(
            task_id=tid, category="research",
            input=f"Write a factual brief on the {w['name']}: release year, maintainer, notable version, "
                  f"license, throughput and dependencies.",
            expected=w["keywords"],
            success_criteria="All six facts present, no deprecated claim.",
            constraints=["Do not claim the project is deprecated."],
            evaluator={"type": "contains_all", "keywords": w["keywords"], "forbidden": [w["wrong_keyword"]]},
            difficulty="moderate",
        ))
    return Benchmark("research_briefs", tasks, metadata={"simulated": True})


class ResearchSwarm(Swarm):
    def __init__(self, config: Optional[SwarmConfig] = None) -> None:
        self.config = config or SwarmConfig()

    # -- simulation helpers ---------------------------------------------------
    @staticmethod
    def _advance(ctx: RunContext, seconds: float) -> None:
        clock = ctx.clock
        if isinstance(clock, SimClock):
            clock.advance(seconds)
        else:
            clock.sleep(seconds * 0.01)  # real clock: keep demos fast

    def _llm(self, ctx: RunContext, agent, prompt: str, out_tokens: int, extra_context_tokens: int = 0,
             input_artifacts=(), relevant_tokens: Optional[int] = None, output: Optional[str] = None) -> str:
        model = agent.model or "claude-sonnet-5"
        in_tokens = estimate_tokens(prompt) + extra_context_tokens
        limit = ctx.context_limit(agent.agent_id)
        if limit is not None and in_tokens > limit:
            in_tokens = limit
        meta = {"relevant_tokens": relevant_tokens} if relevant_tokens is not None else {}
        with agent.llm_call(model=model, input_artifacts=input_artifacts, prompt=prompt, **meta) as call:
            self._advance(ctx, 0.35 + out_tokens / MODEL_SPEED.get(model, 80.0) + in_tokens / 25000.0)
            call.input_tokens = in_tokens
            call.output_tokens = out_tokens
            call.output = output if output is not None else f"[{agent.agent_id} output for: {prompt[:60]}]"
        return call.output

    def _search(self, ctx: RunContext, world: Dict[str, Any], query: str) -> List[str]:
        self._advance(ctx, 0.4 + 0.15 * len(query))
        return list(world["sources"].get(query, []))

    @staticmethod
    def _quality(agent) -> float:
        return MODEL_QUALITY.get(agent.model or "", 0.8)

    # -- main flow ------------------------------------------------------------
    def run(self, task: Task, ctx: RunContext) -> Any:
        world = WORLD[task.task_id]
        if ctx.param("single_agent", False):
            return self._single_agent(task, ctx, world)
        direct = bool(ctx.param("direct_channels", False))
        partition = bool(ctx.param("partition_scope", False))
        parallel = bool(ctx.param("parallel_research", True))
        revision_rounds = int(ctx.param("revision_rounds", 2))
        researchers = ["researcher_a", "researcher_b", "researcher_c"]
        scopes = {"researcher_a": ["s1", "s2"], "researcher_b": ["s3", "s4"],
                  "researcher_c": ["s5"] if partition else ["s1", "s2"]}
        clock = ctx.clock

        # 1. coordinator plans and delegates -------------------------------
        try:
            with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
                self._llm(ctx, coord, f"Plan the research for: {task.input}", 300,
                          output="Plan: split sources between researchers, verify with critic, synthesize.")
                for r in researchers:
                    if ctx.is_enabled(r):
                        coord.send(r, f"Research {world['name']}. Cover sources {', '.join(scopes[r])} and report facts.",
                                   kind="task_assignment", scope=scopes[r])
        except (InterventionError, InjectedFailure):
            ctx.set_output("")
            return ""

        # 2. researchers ------------------------------------------------------
        def research(r: str):
            with ctx.agent(r, role="research", model="claude-sonnet-5") as agent:
                msgs = agent.receive()
                if not msgs:
                    return None
                scope = msgs[0].metadata.get("scope", [])
                q = self._quality(agent)
                rng = agent.rng
                self._llm(ctx, agent, f"Assignment: {msgs[0].content}. Decide search queries.", 120)
                found: List[str] = []
                for s in scope:
                    try:
                        hits = agent.call_tool("web_search", partial(self._search, ctx, world), query=s)
                    except (InterventionError, InjectedFailure):
                        hits = []
                    if not isinstance(hits, list):
                        hits = []
                    for fact in hits:
                        p_keep = WRONG_FACT_PICKUP if fact == world["wrong"] else q
                        if rng.random() < p_keep:
                            found.append(fact)
                notes = "\n".join(found) if found else "No reliable information found."
                self._llm(ctx, agent, f"Summarise findings:\n{notes}", 350, output=notes)
                art = agent.produce(f"notes_{r[-1]}", notes)
                target = "critic" if direct and ctx.is_enabled("critic") else "coordinator"
                agent.send(target, notes, kind="result", artifacts=[art])
                return art

        t0 = clock.now()
        ends: List[float] = []
        for r in researchers:
            if parallel and isinstance(clock, SimClock):
                clock.set(t0)
            try:
                research(r)
            except (InterventionError, InjectedFailure):
                pass
            ends.append(clock.now())
        if parallel and isinstance(clock, SimClock) and ends:
            clock.set(max(ends))

        # 3. coordinator relays notes to the critic ------------------------
        notes_msgs: Dict[str, Any] = {}
        if direct:
            pass
        else:
            try:
                with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
                    for m in coord.receive():
                        notes_msgs[m.sender] = m
                    chosen = [notes_msgs[a] for a in ("researcher_a", "researcher_b") if a in notes_msgs]
                    if "researcher_a" not in notes_msgs and "researcher_c" in notes_msgs:
                        chosen.append(notes_msgs["researcher_c"])
                    arts = [a for m in chosen for a in m.artifacts]
                    combined = "\n".join(str(m.content) for m in chosen)
                    self._llm(ctx, coord, f"Forward these findings for review:\n{combined}", 150,
                              extra_context_tokens=1800, input_artifacts=arts,
                              output="Forwarding findings to the critic.")
                    target = "critic" if ctx.is_enabled("critic") else "synthesizer"
                    coord.send(target, combined, kind="request", artifacts=arts)
            except (InterventionError, InjectedFailure):
                pass

        # 4. critic verifies ---------------------------------------------------
        critique_art = None
        facts: List[str] = []
        flagged: List[str] = []
        if ctx.is_enabled("critic"):
            try:
                with ctx.agent("critic", role="critic", model="claude-opus-5") as critic:
                    msgs = critic.receive()
                    arts = [a for m in msgs for a in m.artifacts]
                    for a in arts:
                        critic.consume(a, purpose="review")
                    facts = [ln for m in msgs for ln in str(m.content).splitlines() if ln.strip()]
                    q = self._quality(critic)
                    rng = critic.rng
                    flagged = [f for f in facts if f == world["wrong"] and rng.random() < q]
                    critique = ("Flagged as unsupported: " + "; ".join(flagged)) if flagged else "No issues found."
                    self._llm(ctx, critic, "Review these findings for errors:\n" + "\n".join(facts), 400,
                              input_artifacts=arts, output=critique)
                    critique_art = critic.produce("critique", critique, intent="verification")
                    target = "synthesizer" if direct else "coordinator"
                    critic.send(target, critique, kind="feedback", artifacts=arts + [critique_art])
            except (InterventionError, InjectedFailure):
                critique_art = None
        elif direct:
            # no critic: researchers sent to critic; nothing arrives. Fall back to the coordinator's mailbox.
            pass

        # 5. coordinator relays to synthesizer (bottleneck) -------------------
        if not direct:
            try:
                with ctx.agent("coordinator", role="coordinator", model="claude-opus-5") as coord:
                    msgs = coord.receive()
                    arts = [a for m in msgs for a in m.artifacts]
                    content = "\n".join(str(m.content) for m in msgs)
                    self._llm(ctx, coord, f"Hand off to synthesizer:\n{content}", 120, extra_context_tokens=2600,
                              input_artifacts=arts, output="Handing off to synthesizer.")
                    coord.send("synthesizer", content, kind="request", artifacts=arts)
            except (InterventionError, InjectedFailure):
                pass

        # 6. synthesizer writes the brief (with revision loop) ---------------
        answer = ""
        final_art = None
        try:
            with ctx.agent("synthesizer", role="synthesizer", model="claude-opus-5") as synth:
                msgs = synth.receive()
                arts = [a for m in msgs for a in m.artifacts]
                gathered: List[str] = []
                for a in arts:
                    art = synth.consume(a, purpose="synthesis")
                    if art is not None and art.name.startswith("notes"):
                        gathered.extend(ln for ln in str(art.content).splitlines() if ln.strip())
                if not gathered:
                    gathered = [ln for m in msgs for ln in str(m.content).splitlines() if ln.strip()]
                flagged_set = set(flagged)
                kept = [f for f in gathered if f not in flagged_set and f != "No reliable information found."]
                relevant = sum(ctx.trajectory.artifacts[a].tokens for a in arts if a in ctx.trajectory.artifacts)
                answer = f"Brief on {world['name']}:\n" + "\n".join(f"- {f}" for f in dict.fromkeys(kept))
                self._llm(ctx, synth, "Write the brief from:\n" + "\n".join(kept), 700, extra_context_tokens=9000,
                          input_artifacts=arts, relevant_tokens=relevant + 60, output=answer)
                final_art = synth.produce("final_brief", answer)
                # revision loop: critic asks for changes; synthesizer revises (ping-pong)
                rounds = 0
                rng = synth.rng
                while rounds < revision_rounds and ctx.is_enabled("critic") and rng.random() < 0.35:
                    rounds += 1
                    if synth.send("critic", answer, kind="request", artifacts=[final_art]) is None:
                        break  # channel disabled or message dropped: no review possible
                    try:
                        with ctx.agent("critic", role="critic", model="claude-opus-5") as critic:
                            m = critic.receive()
                            fb = "Please tighten the wording of the brief."
                            self._llm(ctx, critic, "Review brief:\n" + answer, 150,
                                      input_artifacts=[final_art], output=fb)
                            critic.send("synthesizer", fb, kind="feedback")
                    except (InterventionError, InjectedFailure):
                        break
                    synth.receive()
                    self._llm(ctx, synth, "Revise the brief per feedback.", 500, extra_context_tokens=4000,
                              input_artifacts=[final_art], output=answer)
                    final_art = synth.produce("final_brief", answer)
        except (InterventionError, InjectedFailure):
            answer = ""
        ctx.set_output(answer, artifacts=[final_art] if final_art else [])
        return answer

    # -- single-agent baseline ------------------------------------------------
    def _single_agent(self, task: Task, ctx: RunContext, world: Dict[str, Any]) -> str:
        with ctx.agent("solo", role="generalist", model="claude-opus-5") as agent:
            q = self._quality(agent)
            rng = agent.rng
            self._llm(ctx, agent, f"Research and write: {task.input}", 200)
            found: List[str] = []
            for s in ("s1", "s2", "s3", "s4"):
                try:
                    hits = agent.call_tool("web_search", partial(self._search, ctx, world), query=s)
                except (InterventionError, InjectedFailure):
                    hits = []
                for fact in (hits if isinstance(hits, list) else []):
                    p_keep = WRONG_FACT_PICKUP if fact == world["wrong"] else q
                    if rng.random() < p_keep:
                        found.append(fact)
            # a single agent self-checks less reliably than a dedicated critic
            kept = [f for f in found if not (f == world["wrong"] and rng.random() < q * 0.55)]
            answer = f"Brief on {world['name']}:\n" + "\n".join(f"- {f}" for f in kept)
            self._llm(ctx, agent, "Write the brief:\n" + "\n".join(kept), 700, output=answer)
            art = agent.produce("final_brief", answer)
        ctx.set_output(answer, artifacts=[art])
        return answer


def make_swarm(config: SwarmConfig) -> ResearchSwarm:
    """Factory used by ``swarmeval.evaluate`` and the CLI."""
    return ResearchSwarm(config)


def sim_clock() -> SimClock:
    return SimClock()


if __name__ == "__main__":  # pragma: no cover
    from swarmeval import evaluate
    res = evaluate(make_swarm, build_benchmark(), trials=3, clock_factory=sim_clock)
    print(res.report())
