"""Evaluation validity helpers: leakage detection and repeatability."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import EventType, Trajectory, to_text
from .stats import mean, stdev


def _expected_strings(task_expected: Any, min_len: int = 12) -> List[str]:
    items: List[str] = []
    if task_expected is None:
        return items
    if isinstance(task_expected, (list, tuple, set)):
        items = [to_text(x) for x in task_expected]
    elif isinstance(task_expected, dict):
        items = [to_text(v) for v in task_expected.values()]
    else:
        items = [to_text(task_expected)]
    return [s for s in items if len(s) >= min_len]


def check_leakage(traj: Trajectory) -> List[str]:
    """Warn if an expected-answer fragment appears in an agent's *input*
    (LLM prompt or tool arguments) before the environment supplied it
    (a tool result).  Knowledge that shows up in prompts only after a tool
    returned it is normal; knowledge that shows up first in a prompt means
    the swarm saw the answer key."""
    warnings: List[str] = []
    needles = _expected_strings(traj.task.expected)
    if not needles:
        return warnings
    for n in needles:
        first_env: Optional[float] = None
        first_input: Optional[Tuple[float, str, str]] = None
        for ev in traj.events:
            if ev.event_type == EventType.TOOL_CALL.value:
                if ev.content is not None and n in to_text(ev.content):
                    if first_env is None or ev.end_timestamp < first_env:
                        first_env = ev.end_timestamp
                if n in to_text(ev.tool_args):
                    if first_input is None or ev.timestamp < first_input[0]:
                        first_input = (ev.timestamp, ev.agent_id or "?", "tool_call arguments")
            elif ev.event_type == EventType.LLM_CALL.value:
                if n in to_text(ev.metadata.get("prompt", "")):
                    if first_input is None or ev.timestamp < first_input[0]:
                        first_input = (ev.timestamp, ev.agent_id or "?", "llm_call prompt")
        if first_input is not None and (first_env is None or first_input[0] < first_env - 1e-9):
            warnings.append(f"{traj.run_id}: {first_input[1]} {first_input[2]} contained expected answer fragment "
                            f"{n[:40]!r} before any tool result did")
    return warnings


def repeatability(trajectories: Sequence[Trajectory]) -> Dict[str, Dict[str, float]]:
    """Per-task variance across repeated trials."""
    by_task: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)
    out: Dict[str, Dict[str, float]] = {}
    for tid, ts in by_task.items():
        succ = [1.0 if t.succeeded else 0.0 for t in ts]
        costs = [t.cost_usd or 0.0 for t in ts]
        lat = [t.duration for t in ts]
        out[tid] = {"n": len(ts), "success_mean": mean(succ), "success_std": stdev(succ),
                    "cost_mean": mean(costs), "cost_std": stdev(costs),
                    "latency_mean": mean(lat), "latency_std": stdev(lat),
                    "unstable": 0.0 < mean(succ) < 1.0}
    return out
