"""Evaluators: deterministic and LLM-based judgments of task outcomes.

Every evaluator implements ``evaluate(task, output, trajectory) -> Evaluation``
and, where serialisable, ``to_spec()`` so benchmarks can be stored as JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .schema import Evaluation, Judgment, Task, Trajectory, to_text
from .stats import containment, mean, pairwise_agreement, stdev


class Evaluator:
    name: str = "evaluator"

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:  # pragma: no cover
        raise NotImplementedError

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name}

    def __call__(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        return self.evaluate(task, output, trajectory)


# --------------------------------------------------------------------------- #
# Deterministic evaluators
# --------------------------------------------------------------------------- #
class ExactMatch(Evaluator):
    name = "exact_match"

    def __init__(self, expected: Any = None, normalize: bool = True) -> None:
        self.expected = expected
        self.normalize = normalize

    def _norm(self, x: Any) -> str:
        s = to_text(x)
        return " ".join(s.lower().split()) if self.normalize else s

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        expected = self.expected if self.expected is not None else task.expected
        ok = self._norm(output) == self._norm(expected)
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"expected": expected})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "expected": self.expected, "normalize": self.normalize}


class ContainsAll(Evaluator):
    """Success if every keyword appears in the output and no forbidden one does.
    Score is the fraction of required keywords found (minus forbidden hits)."""
    name = "contains_all"

    def __init__(self, keywords: Optional[Sequence[str]] = None, forbidden: Sequence[str] = (),
                 case_insensitive: bool = True) -> None:
        self.keywords = list(keywords) if keywords is not None else None
        self.forbidden = list(forbidden)
        self.case_insensitive = case_insensitive

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        keywords = self.keywords if self.keywords is not None else (
            task.expected if isinstance(task.expected, list) else [task.expected])
        keywords = [str(k) for k in keywords if k is not None]
        text = to_text(output)
        if self.case_insensitive:
            text_cmp = text.lower()
            found = [k for k in keywords if k.lower() in text_cmp]
            bad = [f for f in self.forbidden if f.lower() in text_cmp]
        else:
            found = [k for k in keywords if k in text]
            bad = [f for f in self.forbidden if f in text]
        frac = len(found) / len(keywords) if keywords else 1.0
        score = max(0.0, frac - 0.5 * len(bad) / max(1, len(self.forbidden) or 1)) if bad else frac
        ok = len(found) == len(keywords) and not bad
        return Evaluation(ok, score, self.name, {"found": found, "missing": [k for k in keywords if k not in found],
                                                 "forbidden_hits": bad},
                          constraints_passed=not bad)

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "keywords": self.keywords, "forbidden": self.forbidden,
                "case_insensitive": self.case_insensitive}


class ContainsAny(ContainsAll):
    name = "contains_any"

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        ev = super().evaluate(task, output, trajectory)
        found = ev.details["found"]
        ok = bool(found) and not ev.details["forbidden_hits"]
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, ev.details, constraints_passed=ev.constraints_passed)


class Regex(Evaluator):
    name = "regex"

    def __init__(self, pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> None:
        self.pattern = pattern
        self.flags = flags

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        ok = re.search(self.pattern, to_text(output), self.flags) is not None
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"pattern": self.pattern})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "pattern": self.pattern}


class NumericTolerance(Evaluator):
    name = "numeric"

    def __init__(self, expected: Optional[float] = None, tolerance: float = 1e-6, relative: bool = False) -> None:
        self.expected = expected
        self.tolerance = tolerance
        self.relative = relative

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        expected = self.expected if self.expected is not None else task.expected
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", to_text(output))
        ok = False
        value: Optional[float] = None
        if nums and expected is not None:
            try:
                exp = float(expected)
                for n in reversed(nums):
                    value = float(n)
                    tol = self.tolerance * abs(exp) if self.relative else self.tolerance
                    if abs(value - exp) <= tol:
                        ok = True
                        break
            except ValueError:
                pass
        return Evaluation(ok, 1.0 if ok else 0.0, self.name, {"expected": expected, "parsed": value})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "expected": self.expected, "tolerance": self.tolerance, "relative": self.relative}


class CallableEvaluator(Evaluator):
    """Wrap any ``fn(task, output, trajectory) -> bool | float | Evaluation``."""
    name = "callable"

    def __init__(self, fn: Callable[..., Any], label: str = "callable") -> None:
        self.fn = fn
        self.name = label

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        r = self.fn(task, output, trajectory)
        if isinstance(r, Evaluation):
            return r
        if isinstance(r, bool):
            return Evaluation(r, 1.0 if r else 0.0, self.name)
        score = float(r)
        return Evaluation(score >= 0.5, score, self.name)


class ToolUseEvaluator(Evaluator):
    """Checks tool-use correctness from the trajectory: expected tools were
    called, forbidden tools were not, and no tool call errored."""
    name = "tool_use"

    def __init__(self, expected_tools: Sequence[str] = (), forbidden_tools: Sequence[str] = (),
                 allow_errors: bool = False) -> None:
        self.expected_tools = list(expected_tools)
        self.forbidden_tools = list(forbidden_tools)
        self.allow_errors = allow_errors

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        if trajectory is None:
            return Evaluation(False, 0.0, self.name, {"reason": "no trajectory"})
        calls = trajectory.tool_calls()
        used = {c.tool_name for c in calls if c.status == "success"}
        missing = [t for t in self.expected_tools if t not in used]
        forbidden = [t for t in self.forbidden_tools if t in used]
        errors = [c for c in calls if c.status == "error"]
        checks = [not missing, not forbidden, self.allow_errors or not errors]
        score = sum(checks) / len(checks)
        return Evaluation(all(checks), score, self.name,
                          {"missing": missing, "forbidden": forbidden, "errors": len(errors)})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "expected_tools": self.expected_tools,
                "forbidden_tools": self.forbidden_tools, "allow_errors": self.allow_errors}


class Groundedness(Evaluator):
    """Deterministic groundedness proxy: fraction of the output's 3-gram
    shingles that appear in recorded tool results or artifacts."""
    name = "groundedness"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        if trajectory is None:
            return Evaluation(False, 0.0, self.name, {"reason": "no trajectory"})
        evidence = "\n".join(to_text(e.content) for e in trajectory.tool_calls() if e.content is not None)
        evidence += "\n" + "\n".join(to_text(a.content) for a in trajectory.artifacts.values())
        score = containment(to_text(output), evidence)
        return Evaluation(score >= self.threshold, score, self.name, {"threshold": self.threshold})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "threshold": self.threshold}


class Constraints(Evaluator):
    """All sub-evaluators must pass (hard constraints)."""
    name = "constraints"

    def __init__(self, checks: Sequence[Evaluator]) -> None:
        self.checks = list(checks)

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        results = [(c.name, c.evaluate(task, output, trajectory)) for c in self.checks]
        passed = all(r.success for _, r in results)
        score = mean([r.score for _, r in results]) if results else 1.0
        return Evaluation(passed, score, self.name, {n: r.to_dict() for n, r in results}, constraints_passed=passed)

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "checks": [c.to_spec() for c in self.checks]}


class Composite(Evaluator):
    """Weighted combination of evaluators.  Success requires the weighted score
    to reach ``threshold`` and every ``required`` evaluator to pass."""
    name = "composite"

    def __init__(self, evaluators: Sequence[Evaluator], weights: Optional[Sequence[float]] = None,
                 threshold: float = 0.5, required: Sequence[str] = ()) -> None:
        self.evaluators = list(evaluators)
        self.weights = list(weights) if weights else [1.0] * len(self.evaluators)
        self.threshold = threshold
        self.required = list(required)

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        results = [(e.name, e.evaluate(task, output, trajectory)) for e in self.evaluators]
        total_w = sum(self.weights) or 1.0
        score = sum(w * r.score for w, (_, r) in zip(self.weights, results)) / total_w
        req_ok = all(r.success for n, r in results if n in self.required)
        ok = score >= self.threshold and req_ok
        return Evaluation(ok, score, self.name, {n: r.to_dict() for n, r in results})

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "evaluators": [e.to_spec() for e in self.evaluators],
                "weights": self.weights, "threshold": self.threshold, "required": self.required}


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #
DEFAULT_RUBRIC = (
    "You are grading the output of an AI system on a task. Judge whether the output "
    "correctly and completely accomplishes the task. Respond with JSON only: "
    '{"score": <0.0-1.0>, "success": <true|false>, "rationale": "<one or two sentences>"}'
)


def build_judge_prompt(task: Task, output: Any, rubric: str = DEFAULT_RUBRIC) -> str:
    parts = [rubric, "", "TASK:", to_text(task.input)]
    if task.success_criteria:
        parts += ["", "SUCCESS CRITERIA:", task.success_criteria]
    if task.constraints:
        parts += ["", "CONSTRAINTS:"] + [f"- {c}" for c in task.constraints]
    if task.reference is not None:
        parts += ["", "REFERENCE SOLUTION:", to_text(task.reference)]
    elif task.expected is not None:
        parts += ["", "EXPECTED:", to_text(task.expected)]
    parts += ["", "OUTPUT TO GRADE:", to_text(output)]
    return "\n".join(parts)


def parse_judgment(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            score = float(d.get("score", 1.0 if d.get("success") else 0.0))
            success = bool(d.get("success", score >= 0.5))
            return {"score": max(0.0, min(1.0, score)), "success": success,
                    "rationale": str(d.get("rationale", ""))}
        except (ValueError, TypeError):
            pass
    low = text.lower()
    m2 = re.search(r"score\s*[:=]\s*([01](?:\.\d+)?)", low)
    if m2:
        score = float(m2.group(1))
        return {"score": score, "success": score >= 0.5, "rationale": text[:300]}
    if "pass" in low or "success: true" in low or "correct" in low:
        return {"score": 1.0, "success": True, "rationale": text[:300]}
    return {"score": 0.0, "success": False, "rationale": text[:300]}


class LLMJudge(Evaluator):
    """LLM-based evaluator supporting multiple judgments and agreement.

    ``judges`` is a list of ``(name, fn)`` where ``fn(prompt) -> str``.  A
    single callable may be passed and will be invoked ``n_judgments`` times.
    Success is decided by majority vote; score is the mean; ``agreement`` is
    the pairwise agreement on the success label and ``confidence`` combines
    agreement with score dispersion.
    """
    name = "llm_judge"

    def __init__(self, judge: Any, rubric: str = DEFAULT_RUBRIC, n_judgments: int = 1,
                 threshold: float = 0.5) -> None:
        if callable(judge):
            self.judges: List = [(f"judge_{i+1}", judge) for i in range(max(1, n_judgments))]
        else:
            self.judges = [(n, f) for n, f in judge]
        self.rubric = rubric
        self.threshold = threshold

    def evaluate(self, task: Task, output: Any, trajectory: Optional[Trajectory] = None) -> Evaluation:
        prompt = build_judge_prompt(task, output, self.rubric)
        judgments: List[Judgment] = []
        for name, fn in self.judges:
            try:
                raw = fn(prompt)
                parsed = parse_judgment(to_text(raw))
                judgments.append(Judgment(name, parsed["score"], parsed["success"], parsed["rationale"], raw))
            except Exception as exc:  # a failing judge is recorded, not fatal
                judgments.append(Judgment(name, 0.0, False, f"judge error: {exc}", None))
        scores = [j.score for j in judgments]
        labels = [j.success for j in judgments]
        score = mean(scores)
        success = sum(labels) * 2 > len(labels) if labels else False
        agreement = pairwise_agreement(labels)
        dispersion = stdev(scores) if len(scores) > 1 else 0.0
        confidence = (agreement if agreement is not None else 1.0) * (1.0 - min(1.0, dispersion))
        return Evaluation(success, score, self.name, {"prompt_chars": len(prompt)}, judgments,
                          agreement, confidence)

    def to_spec(self) -> Dict[str, Any]:
        return {"type": self.name, "rubric": self.rubric, "n_judgments": len(self.judges),
                "threshold": self.threshold, "note": "judge callable not serialisable"}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Evaluator]] = {}


def register(name: str, factory: Callable[[Dict[str, Any]], Evaluator]) -> None:
    _REGISTRY[name] = factory


register("exact_match", lambda s: ExactMatch(s.get("expected"), s.get("normalize", True)))
register("contains_all", lambda s: ContainsAll(s.get("keywords"), s.get("forbidden", ()), s.get("case_insensitive", True)))
register("contains_any", lambda s: ContainsAny(s.get("keywords"), s.get("forbidden", ()), s.get("case_insensitive", True)))
register("regex", lambda s: Regex(s["pattern"]))
register("numeric", lambda s: NumericTolerance(s.get("expected"), s.get("tolerance", 1e-6), s.get("relative", False)))
register("tool_use", lambda s: ToolUseEvaluator(s.get("expected_tools", ()), s.get("forbidden_tools", ()), s.get("allow_errors", False)))
register("groundedness", lambda s: Groundedness(s.get("threshold", 0.5)))
register("constraints", lambda s: Constraints([from_spec(c) for c in s.get("checks", [])]))
register("composite", lambda s: Composite([from_spec(e) for e in s.get("evaluators", [])], s.get("weights"),
                                          s.get("threshold", 0.5), s.get("required", ())))


def from_spec(spec: Any) -> Evaluator:
    """Build an evaluator from an instance, a spec dict, or a bare type name."""
    if isinstance(spec, Evaluator):
        return spec
    if callable(spec) and not isinstance(spec, dict):
        return CallableEvaluator(spec)
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"cannot build evaluator from {spec!r}")
    t = spec["type"]
    if t not in _REGISTRY:
        raise ValueError(f"unknown evaluator type {t!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[t](spec)


def resolve(task: Task, *fallbacks: Any) -> Optional[Evaluator]:
    for candidate in (task.evaluator, *fallbacks):
        if candidate is not None:
            return from_spec(candidate)
    return None
