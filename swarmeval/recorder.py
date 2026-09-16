"""Trajectory recorder and run context.

The recorder has two jobs:

1. **Observability** -- capture every agent invocation, LLM call, tool call,
   message, artifact and wait as a structured event.
2. **Intervention enforcement** -- when a SwarmConfig disables an agent or a
   tool, drops a message, or injects a failure, the recorder is the place
   where that takes effect.  This lets counterfactual experiments work on any
   swarm that is instrumented through it, without every swarm having to
   implement ablation logic.

Typical use inside a swarm::

    def run(self, task, ctx):
        with ctx.agent("coordinator", role="coordinator") as coord:
            plan = coord.llm(call_model, prompt, model="claude-opus-5")
            coord.send("researcher", plan, kind="task_assignment")
        with ctx.agent("researcher", role="research") as r:
            for msg in r.receive():
                notes = r.call_tool("web_search", search, msg.content)
                r.produce("notes", notes)
        ...
        return answer
"""
from __future__ import annotations

import contextvars
import hashlib
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .config import FailureSpec, SwarmConfig
from .pricing import DEFAULT_PRICING, PricingTable
from .schema import (Artifact, Event, EventType, Span, Status, Task, Trajectory,
                     content_hash, estimate_tokens, new_id, preview, to_text)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class InterventionError(Exception):
    """A configured intervention prevented an action (not a simulated crash)."""


class AgentDisabled(InterventionError):
    pass


class ToolDisabled(InterventionError):
    pass


class BudgetExceeded(InterventionError):
    pass


class InjectedFailure(Exception):
    """Base class for injected failures.  Deliberately *not* an
    InterventionError: swarms should treat these like any runtime error."""


class InjectedAgentCrash(InjectedFailure):
    pass


class InjectedLLMError(InjectedFailure):
    pass


class InjectedToolError(InjectedFailure):
    pass


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #
class Clock:
    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock(Clock):
    """A virtual clock.  Simulated swarms advance it explicitly so that
    demos and tests run instantly while still producing realistic timings."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, float(seconds))

    def set(self, t: float) -> None:
        with self._lock:
            self._t = float(t)


# --------------------------------------------------------------------------- #
# Value objects handed to swarm code
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    message_id: str
    sender: str
    receiver: str
    content: Any
    kind: str
    artifacts: List[str]
    timestamp: float
    tokens: int
    event_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    """Return this from a function passed to ``AgentSpan.llm`` to report real
    token usage."""
    text: Any
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None
    raw: Any = None


class LLMCall:
    """Mutable handle yielded by ``AgentSpan.llm_call``."""

    def __init__(self, model: Optional[str], input_tokens: int, output_tokens: int,
                 input_artifacts: List[str], prompt: Any) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_artifacts = input_artifacts
        self.prompt = prompt
        self.output: Any = None
        self.metadata: Dict[str, Any] = {}


class ToolCall:
    """Mutable handle yielded by ``AgentSpan.tool_call``."""

    def __init__(self, name: str, args: Any, input_artifacts: List[str]) -> None:
        self.name = name
        self.args = args
        self.input_artifacts = input_artifacts
        self.result: Any = None
        self.status: str = Status.SUCCESS.value
        self.metadata: Dict[str, Any] = {}


_current_span: contextvars.ContextVar = contextvars.ContextVar("swarmeval_current_span", default=None)


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #
class RunContext:
    """Everything a swarm needs during one run: recording, intervention
    queries, messaging, randomness and the clock."""

    def __init__(self, task: Task, config: Optional[SwarmConfig] = None, run_id: Optional[str] = None,
                 clock: Optional[Clock] = None, seed: int = 0, trial: int = 0,
                 pricing: Optional[PricingTable] = None, capture_text: bool = True,
                 max_text_chars: int = 20000) -> None:
        self.task = task
        self.config = config or SwarmConfig()
        self.clock = clock or WallClock()
        self.pricing = pricing or DEFAULT_PRICING
        self.seed = seed
        self.trial = trial
        self.rng = random.Random(seed)
        self._failure_rng = random.Random((seed * 7919 + 17) & 0xFFFFFFFF)
        self.capture_text = capture_text
        self.max_text_chars = max_text_chars
        self.trajectory = Trajectory(run_id=run_id or new_id("run"), task_id=task.task_id, task=task,
                                     config=self.config.to_dict(), seed=seed, trial=trial)
        self._lock = threading.RLock()
        self._mailbox: Dict[str, List[Message]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._agent_tokens: Dict[str, int] = {}
        self._agent_rngs: Dict[str, random.Random] = {}
        self._started = False
        self._finished = False

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.trajectory.started_at = self.now()
        self.record(EventType.RUN_START, content=preview(self.task.input), metadata={"config": self.config.label})

    def finish(self, status: str = Status.SUCCESS.value, error: Optional[str] = None) -> Trajectory:
        if self._finished:
            return self.trajectory
        self._finished = True
        self.trajectory.ended_at = self.now()
        self.trajectory.status = status
        self.trajectory.error = error
        self.record(EventType.RUN_END, status=status, error=error,
                    latency_ms=self.trajectory.duration * 1000.0)
        return self.trajectory

    def set_output(self, output: Any, artifacts: Iterable[Union[Artifact, str]] = ()) -> None:
        self.trajectory.final_output = output
        self.trajectory.output_artifacts = [a.artifact_id if isinstance(a, Artifact) else str(a) for a in artifacts]

    def now(self) -> float:
        return self.clock.now()

    # -- intervention queries -------------------------------------------------
    def is_enabled(self, agent_id: str) -> bool:
        return self.config.agent_enabled(agent_id)

    def enabled(self, *agent_ids: str) -> List[str]:
        return [a for a in agent_ids if self.is_enabled(a)]

    def tool_available(self, tool: str) -> bool:
        return self.config.tool_enabled(tool)

    def channel_open(self, sender: str, receiver: str) -> bool:
        return self.config.channel_open(sender, receiver)

    def model_for(self, agent_id: str, default: Optional[str] = None) -> Optional[str]:
        return self.config.model_for(agent_id, default)

    def param(self, key: str, default: Any = None) -> Any:
        return self.config.param(key, default)

    def context_limit(self, agent_id: str) -> Optional[int]:
        return self.config.context_limits.get(agent_id, self.config.context_limits.get("*"))

    def token_budget(self, agent_id: str) -> Optional[int]:
        return self.config.token_budgets.get(agent_id, self.config.token_budgets.get("*"))

    def tokens_used(self, agent_id: str) -> int:
        return self._agent_tokens.get(agent_id, 0)

    def rng_for(self, agent_id: str) -> random.Random:
        """A random stream private to ``agent_id`` (derived from the run seed).

        Use this instead of ``ctx.rng`` inside agents so that removing or
        re-ordering one agent does not perturb the random draws of the
        others -- the experimental control that makes ablations comparable."""
        with self._lock:
            rng = self._agent_rngs.get(agent_id)
            if rng is None:
                h = hashlib.sha1(f"{self.seed}|{agent_id}".encode()).hexdigest()
                rng = random.Random(int(h[:12], 16))
                self._agent_rngs[agent_id] = rng
            return rng

    # -- recording ------------------------------------------------------------
    def record(self, event_type: Union[EventType, str], **fields: Any) -> Event:
        span = _current_span.get()
        et = event_type.value if isinstance(event_type, EventType) else event_type
        fields.setdefault("timestamp", self.now())
        if span is not None:
            fields.setdefault("agent_id", span.agent_id)
            fields.setdefault("span_id", span.span_id)
            fields.setdefault("parent_event_id", span.start_event_id)
        ev = Event(event_id=new_id("ev"), run_id=self.trajectory.run_id, task_id=self.task.task_id,
                   event_type=et, **fields)
        with self._lock:
            self.trajectory.events.append(ev)
        return ev

    def annotate(self, note: str, **metadata: Any) -> Event:
        return self.record(EventType.ANNOTATION, content=note, metadata=metadata)

    def _text(self, obj: Any) -> Any:
        if not self.capture_text:
            return preview(obj, 120)
        if isinstance(obj, str) and len(obj) > self.max_text_chars:
            return obj[: self.max_text_chars] + "…[truncated]"
        return obj

    def _record_intervention(self, kind: str, target: str, detail: str, **metadata: Any) -> None:
        entry = {"kind": kind, "target": target, "detail": detail, "timestamp": self.now(), **metadata}
        with self._lock:
            self.trajectory.interventions.append(entry)
        self.record(EventType.INTERVENTION, kind=kind, content=detail, status=Status.INJECTED.value,
                    metadata={"target": target, **metadata})

    # -- failure injection ----------------------------------------------------
    def _matching(self, kind: str, subject: str, alt_subjects: Sequence[str] = ()) -> Iterator[FailureSpec]:
        subjects = {subject, "*", *alt_subjects}
        for spec in self.config.failures:
            if spec.kind == kind and spec.target in subjects:
                yield spec

    def _should_fire(self, spec: FailureSpec) -> bool:
        with self._lock:
            self._failure_counts[spec.key] = self._failure_counts.get(spec.key, 0) + 1
            n = self._failure_counts[spec.key]
        if spec.on_call is not None:
            return n == spec.on_call
        if spec.probability >= 1.0:
            return True
        return self._failure_rng.random() < spec.probability

    # -- spans ----------------------------------------------------------------
    def agent(self, agent_id: str, role: Optional[str] = None, model: Optional[str] = None,
              **metadata: Any) -> "AgentSpan":
        return AgentSpan(self, agent_id, role, model, metadata)

    @property
    def current_span(self) -> Optional["AgentSpan"]:
        return _current_span.get()

    # -- messaging ------------------------------------------------------------
    def send(self, sender: str, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Union[Artifact, str]] = (), **metadata: Any) -> Optional[Message]:
        """Record a message and deliver it to the receiver's mailbox.

        Returns the delivered Message, or ``None`` if the channel is disabled
        or the message was dropped by an injected failure."""
        art_ids = [a.artifact_id if isinstance(a, Artifact) else str(a) for a in artifacts]
        tokens = estimate_tokens(content) + sum(
            self.trajectory.artifacts[a].tokens for a in art_ids if a in self.trajectory.artifacts)
        status = Status.SUCCESS.value
        channel = f"{sender}->{receiver}"
        if not self.channel_open(sender, receiver):
            status = Status.DISABLED.value
            self._record_intervention("channel_disabled", channel, f"message {channel} not delivered")
        else:
            for spec in self._matching("message_drop", channel, [receiver, sender]):
                if self._should_fire(spec):
                    status = Status.DROPPED.value
                    self._record_intervention("message_drop", channel, f"dropped message {channel}", failure=spec.key)
                    break
            if status == Status.SUCCESS.value:
                for spec in self._matching("message_corrupt", channel, [receiver, sender]):
                    if self._should_fire(spec):
                        content = spec.payload
                        status = Status.INJECTED.value
                        self._record_intervention("message_corrupt", channel, f"corrupted message {channel}", failure=spec.key)
                        break
        span = _current_span.get()
        ev = self.record(EventType.MESSAGE, agent_id=sender, span_id=span.span_id if span else None,
                         sender=sender, receiver=receiver, kind=kind, content=self._text(content),
                         content_tokens=tokens, content_hash=content_hash(content), status=status,
                         input_artifacts=art_ids, metadata=dict(metadata))
        msg = Message(new_id("msg"), sender, receiver, content, kind, art_ids, ev.timestamp, tokens, ev.event_id,
                      dict(metadata))
        if status in (Status.SUCCESS.value, Status.INJECTED.value):
            with self._lock:
                self._mailbox.setdefault(receiver, []).append(msg)
            return msg
        return None

    def receive(self, agent_id: str) -> List[Message]:
        """Drain the mailbox of ``agent_id``."""
        with self._lock:
            msgs = self._mailbox.get(agent_id, [])
            self._mailbox[agent_id] = []
        return msgs

    def inbox(self, agent_id: str) -> List[Message]:
        with self._lock:
            return list(self._mailbox.get(agent_id, []))

    # -- parallel helper ------------------------------------------------------
    def parallel(self, *fns: Callable[[], Any], max_workers: Optional[int] = None) -> List[Any]:
        """Run callables concurrently in threads, propagating the current span
        so nested ``ctx.agent`` calls are parented correctly."""
        if not fns:
            return []
        contexts = [contextvars.copy_context() for _ in fns]
        with ThreadPoolExecutor(max_workers=max_workers or len(fns)) as ex:
            futures = [ex.submit(c.run, fn) for c, fn in zip(contexts, fns)]
            return [f.result() for f in futures]

    # -- cost -----------------------------------------------------------------
    def cost_of(self, model: Optional[str], input_tokens: int, output_tokens: int) -> Optional[float]:
        return self.pricing.cost(model, input_tokens, output_tokens)


# --------------------------------------------------------------------------- #
# Agent span
# --------------------------------------------------------------------------- #
class AgentSpan:
    """Context manager for one agent invocation."""

    def __init__(self, ctx: RunContext, agent_id: str, role: Optional[str], model: Optional[str],
                 metadata: Dict[str, Any]) -> None:
        self.ctx = ctx
        self.agent_id = agent_id
        self.role = role
        self.requested_model = model
        self.model: Optional[str] = model
        self.metadata = metadata
        self.span_id: str = ""
        self.start_event_id: Optional[str] = None
        self.parent: Optional["AgentSpan"] = None
        self._token: Any = None
        self._closed = False
        self.span: Optional[Span] = None

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "AgentSpan":
        ctx = self.ctx
        if not ctx.is_enabled(self.agent_id):
            ctx._record_intervention("agent_disabled", self.agent_id, f"agent {self.agent_id} is disabled")
            raise AgentDisabled(self.agent_id)
        self.parent = _current_span.get()
        self.model = ctx.model_for(self.agent_id, self.requested_model)
        self.span_id = new_id("span")
        start = ctx.now()
        ev = ctx.record(EventType.AGENT_START, agent_id=self.agent_id, span_id=self.span_id,
                        parent_event_id=self.parent.start_event_id if self.parent else None,
                        model=self.model, kind=self.role, metadata=dict(self.metadata), timestamp=start)
        self.start_event_id = ev.event_id
        self.span = Span(self.span_id, self.agent_id, self.role,
                         self.parent.span_id if self.parent else None, start, model=self.model,
                         start_event_id=ev.event_id, metadata=dict(self.metadata))
        with ctx._lock:
            ctx.trajectory.spans[self.span_id] = self.span
        self._token = _current_span.set(self)
        for spec in ctx._matching("agent_delay", self.agent_id):
            if ctx._should_fire(spec):
                self.wait("injected_delay", spec.delay_s, status=Status.INJECTED.value)
        for spec in ctx._matching("agent_crash", self.agent_id):
            if ctx._should_fire(spec):
                ctx._record_intervention("agent_crash", self.agent_id, f"injected crash in {self.agent_id}",
                                         failure=spec.key)
                self._close(Status.INJECTED.value, f"injected crash ({spec.key})")
                raise InjectedAgentCrash(f"{self.agent_id}: {spec.key}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._closed:
            return False
        if exc is not None:
            err = f"{type(exc).__name__}: {exc}"
            self.ctx.record(EventType.ERROR, error=err, content=traceback.format_exc()[-2000:]
                            if self.ctx.capture_text else None)
            self._close(Status.ERROR.value, err)
        else:
            self._close(Status.SUCCESS.value, None)
        return False

    def _close(self, status: str, error: Optional[str]) -> None:
        if self._closed:
            return
        self._closed = True
        end = self.ctx.now()
        if self.span is not None:
            self.span.end = end
            self.span.status = status
            self.span.error = error
        self.ctx.record(EventType.AGENT_END, agent_id=self.agent_id, span_id=self.span_id,
                        parent_event_id=self.start_event_id, status=status, error=error,
                        latency_ms=(end - self.span.start) * 1000.0 if self.span else None, timestamp=end)
        if self._token is not None:
            try:
                _current_span.reset(self._token)
            except ValueError:  # token created in a different context (threads)
                _current_span.set(self.parent)
            self._token = None

    # -- LLM calls ------------------------------------------------------------
    def _check_llm_preconditions(self, model: Optional[str]) -> None:
        ctx = self.ctx
        budget = ctx.token_budget(self.agent_id)
        if budget is not None and ctx.tokens_used(self.agent_id) >= budget:
            ctx._record_intervention("budget_exceeded", self.agent_id,
                                     f"{self.agent_id} exhausted token budget {budget}")
            raise BudgetExceeded(f"{self.agent_id}: {ctx.tokens_used(self.agent_id)} >= {budget}")
        for spec in ctx._matching("llm_delay", self.agent_id):
            if ctx._should_fire(spec):
                self.wait("injected_llm_delay", spec.delay_s, status=Status.INJECTED.value)
        for spec in ctx._matching("llm_error", self.agent_id):
            if ctx._should_fire(spec):
                ctx._record_intervention("llm_error", self.agent_id, f"injected LLM error in {self.agent_id}",
                                         failure=spec.key)
                ctx.record(EventType.LLM_CALL, model=model, status=Status.INJECTED.value,
                           error=f"injected ({spec.key})", latency_ms=0.0)
                raise InjectedLLMError(f"{self.agent_id}: {spec.key}")

    @contextmanager
    def llm_call(self, model: Optional[str] = None, input_tokens: int = 0, output_tokens: int = 0,
                 input_artifacts: Iterable[Union[Artifact, str]] = (), prompt: Any = None,
                 **metadata: Any) -> Iterator[LLMCall]:
        """Record one LLM call.  Set ``call.input_tokens`` / ``call.output_tokens``
        / ``call.output`` inside the block when they become known."""
        ctx = self.ctx
        model = model or self.model
        art_ids = [a.artifact_id if isinstance(a, Artifact) else str(a) for a in input_artifacts]
        self._check_llm_preconditions(model)
        call = LLMCall(model, input_tokens, output_tokens, art_ids, prompt)
        call.metadata.update(metadata)
        t0 = ctx.now()
        try:
            yield call
        except Exception as exc:
            self._record_llm(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            raise
        self._record_llm(call, t0, Status.SUCCESS.value, None)

    def _record_llm(self, call: LLMCall, t0: float, status: str, error: Optional[str]) -> Event:
        ctx = self.ctx
        end = ctx.now()
        limit = ctx.context_limit(self.agent_id)
        meta = dict(call.metadata)
        if call.prompt is not None:
            meta["prompt"] = ctx._text(to_text(call.prompt))
            meta["prompt_hash"] = content_hash(call.prompt)
        if limit is not None and call.input_tokens > limit:
            meta["context_limit"] = limit
            meta["context_limit_exceeded"] = True
        with ctx._lock:
            ctx._agent_tokens[self.agent_id] = ctx._agent_tokens.get(self.agent_id, 0) + call.input_tokens + call.output_tokens
        return ctx.record(EventType.LLM_CALL, model=call.model, input_tokens=int(call.input_tokens),
                          output_tokens=int(call.output_tokens),
                          cost_usd=ctx.cost_of(call.model, int(call.input_tokens), int(call.output_tokens)),
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.output) if call.output is not None else None,
                          content_tokens=estimate_tokens(call.output) if call.output is not None else 0,
                          content_hash=content_hash(call.output) if call.output is not None else None,
                          input_artifacts=list(call.input_artifacts), metadata=meta, timestamp=t0)

    def llm(self, fn: Callable[..., Any], *args: Any, model: Optional[str] = None,
            input_artifacts: Iterable[Union[Artifact, str]] = (), prompt: Any = None,
            input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
            **kwargs: Any) -> Any:
        """Managed LLM call: runs ``fn`` and records it.  If ``fn`` returns an
        ``LLMResult`` its token usage is recorded and ``result.text`` returned."""
        with self.llm_call(model=model, input_artifacts=input_artifacts, prompt=prompt) as call:
            result = fn(*args, **kwargs)
            if isinstance(result, LLMResult):
                call.input_tokens = result.input_tokens
                call.output_tokens = result.output_tokens
                if result.model:
                    call.model = result.model
                call.output = result.text
                out = result.text
            else:
                call.output = result
                out = result
            if input_tokens is not None:
                call.input_tokens = input_tokens
            if output_tokens is not None:
                call.output_tokens = output_tokens
            if not call.input_tokens and prompt is not None:
                call.input_tokens = estimate_tokens(prompt)
            if not call.output_tokens and out is not None:
                call.output_tokens = estimate_tokens(out)
        return out

    # -- tool calls -----------------------------------------------------------
    @contextmanager
    def tool_call(self, name: str, args: Any = None,
                  input_artifacts: Iterable[Union[Artifact, str]] = ()) -> Iterator[ToolCall]:
        ctx = self.ctx
        art_ids = [a.artifact_id if isinstance(a, Artifact) else str(a) for a in input_artifacts]
        if not ctx.tool_available(name):
            ctx._record_intervention("tool_disabled", name, f"tool {name} is disabled")
            ctx.record(EventType.TOOL_CALL, tool_name=name, tool_args=args, status=Status.DISABLED.value,
                       error="tool disabled", latency_ms=0.0, input_artifacts=art_ids)
            raise ToolDisabled(name)
        for spec in ctx._matching("tool_delay", name):
            if ctx._should_fire(spec):
                self.wait("injected_tool_delay", spec.delay_s, status=Status.INJECTED.value)
        for spec in ctx._matching("tool_error", name):
            if ctx._should_fire(spec):
                ctx._record_intervention("tool_error", name, f"injected error in tool {name}", failure=spec.key)
                ctx.record(EventType.TOOL_CALL, tool_name=name, tool_args=args, status=Status.INJECTED.value,
                           error=f"injected ({spec.key})", latency_ms=0.0, input_artifacts=art_ids)
                raise InjectedToolError(f"{name}: {spec.key}")
        call = ToolCall(name, args, art_ids)
        t0 = ctx.now()
        try:
            yield call
        except Exception as exc:
            self._record_tool(call, t0, Status.ERROR.value, f"{type(exc).__name__}: {exc}")
            raise
        self._record_tool(call, t0, call.status, None)

    def _record_tool(self, call: ToolCall, t0: float, status: str, error: Optional[str]) -> Event:
        ctx = self.ctx
        end = ctx.now()
        return ctx.record(EventType.TOOL_CALL, tool_name=call.name, tool_args=call.args,
                          latency_ms=(end - t0) * 1000.0, status=status, error=error,
                          content=ctx._text(call.result) if call.result is not None else None,
                          content_tokens=estimate_tokens(call.result) if call.result is not None else 0,
                          content_hash=content_hash(call.result) if call.result is not None else None,
                          input_artifacts=list(call.input_artifacts), metadata=dict(call.metadata), timestamp=t0)

    def call_tool(self, name: str, fn: Callable[..., Any], *args: Any,
                  input_artifacts: Iterable[Union[Artifact, str]] = (), **kwargs: Any) -> Any:
        """Managed tool call: runs ``fn(*args, **kwargs)`` and records it.
        Supports ``tool_malformed`` failure injection (the real function is
        skipped and ``payload`` returned instead)."""
        recorded_args: Any = {"args": list(args), **kwargs} if args else dict(kwargs)
        with self.tool_call(name, recorded_args, input_artifacts=input_artifacts) as call:
            for spec in self.ctx._matching("tool_malformed", name):
                if self.ctx._should_fire(spec):
                    self.ctx._record_intervention("tool_malformed", name, f"malformed output from {name}",
                                                  failure=spec.key)
                    call.result = spec.payload
                    call.status = Status.INJECTED.value
                    return spec.payload
            call.result = fn(*args, **kwargs)
            return call.result

    # -- messaging ------------------------------------------------------------
    def send(self, receiver: str, content: Any, kind: str = "information",
             artifacts: Iterable[Union[Artifact, str]] = (), **metadata: Any) -> Optional[Message]:
        return self.ctx.send(self.agent_id, receiver, content, kind=kind, artifacts=artifacts, **metadata)

    def receive(self) -> List[Message]:
        return self.ctx.receive(self.agent_id)

    def inbox(self) -> List[Message]:
        return self.ctx.inbox(self.agent_id)

    # -- artifacts ------------------------------------------------------------
    def produce(self, name: str, content: Any, intent: Optional[str] = None, **metadata: Any) -> Artifact:
        ctx = self.ctx
        art = Artifact(new_id("art"), name, self.agent_id, self.span_id, ctx.now(), content,
                       content_hash(content), estimate_tokens(content), intent, None, dict(metadata))
        ev = ctx.record(EventType.ARTIFACT, kind=name, content=ctx._text(content), content_tokens=art.tokens,
                        content_hash=art.content_hash, output_artifacts=[art.artifact_id],
                        metadata={"intent": intent, **metadata})
        art.event_id = ev.event_id
        with ctx._lock:
            ctx.trajectory.artifacts[art.artifact_id] = art
        return art

    def consume(self, artifact: Union[Artifact, str], purpose: Optional[str] = None) -> Optional[Artifact]:
        art_id = artifact.artifact_id if isinstance(artifact, Artifact) else str(artifact)
        art = self.ctx.trajectory.artifacts.get(art_id)
        self.ctx.record(EventType.ARTIFACT_USE, kind=art.name if art else None, input_artifacts=[art_id],
                        metadata={"purpose": purpose} if purpose else {})
        return art

    # -- waiting --------------------------------------------------------------
    def wait(self, reason: str, seconds: float, status: str = Status.SUCCESS.value) -> None:
        t0 = self.ctx.now()
        self.ctx.clock.sleep(seconds)
        self.ctx.record(EventType.WAIT, kind=reason, latency_ms=(self.ctx.now() - t0) * 1000.0,
                        status=status, timestamp=t0)

    @contextmanager
    def waiting(self, reason: str) -> Iterator[None]:
        t0 = self.ctx.now()
        try:
            yield
        finally:
            self.ctx.record(EventType.WAIT, kind=reason, latency_ms=(self.ctx.now() - t0) * 1000.0, timestamp=t0)

    # -- misc -----------------------------------------------------------------
    @property
    def rng(self) -> random.Random:
        """Per-agent random stream (see ``RunContext.rng_for``)."""
        return self.ctx.rng_for(self.agent_id)

    def note(self, text: str, **metadata: Any) -> Event:
        return self.ctx.record(EventType.ANNOTATION, content=text, metadata=metadata)

    def error(self, message: str, **metadata: Any) -> Event:
        return self.ctx.record(EventType.ERROR, error=message, metadata=metadata)

    def sub_agent(self, agent_id: str, role: Optional[str] = None, model: Optional[str] = None,
                  **metadata: Any) -> "AgentSpan":
        """Alias for ``ctx.agent`` that reads naturally when nesting."""
        return self.ctx.agent(agent_id, role=role, model=model, **metadata)
