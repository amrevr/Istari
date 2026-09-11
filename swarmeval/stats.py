"""Small, dependency-free statistics toolkit.

SwarmEval reports distributions and confidence intervals rather than point
estimates wherever it can, because agent behaviour is stochastic.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def percentile(xs: Sequence[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] = mean,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(stat(draw))
    samples.sort()
    lo = samples[int((alpha / 2) * n_boot)]
    hi = samples[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], n_boot: int = 1000,
                           alpha: float = 0.05, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """Mean of (b - a) with a bootstrap CI over paired observations."""
    if len(a) != len(b) or not a:
        return (mean(b) - mean(a), (0.0, 0.0))
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    return (mean(diffs), bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed))


@dataclass
class Summary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    minimum: float = 0.0
    maximum: float = 0.0
    values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std,
                "ci_low": self.ci_low, "ci_high": self.ci_high,
                "min": self.minimum, "max": self.maximum}

    def fmt(self, unit: str = "", pct: bool = False, digits: int = 1) -> str:
        if pct:
            return f"{self.mean*100:.{digits}f}% [{self.ci_low*100:.{digits}f}, {self.ci_high*100:.{digits}f}]"
        return f"{self.mean:.{digits}f}{unit} [{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}]"


def summarize(values: Sequence[float], seed: int = 0, keep_values: bool = True) -> Summary:
    values = [float(v) for v in values]
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    lo, hi = bootstrap_ci(values, seed=seed)
    return Summary(len(values), mean(values), stdev(values), lo, hi,
                   min(values), max(values), values if keep_values else [])


def proportion_summary(flags: Sequence[bool]) -> Summary:
    flags = [bool(f) for f in flags]
    n = len(flags)
    k = sum(flags)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    lo, hi = wilson_interval(k, n)
    p = k / n
    return Summary(n, p, math.sqrt(p * (1 - p)), lo, hi, 0.0 if k < n else 1.0,
                   1.0 if k > 0 else 0.0, [1.0 if f else 0.0 for f in flags])


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(1 for x in a if x) / n
    pb = sum(1 for y in b if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pairwise_agreement(labels: Sequence[bool]) -> Optional[float]:
    """Fraction of judge pairs that agree on a binary label."""
    labels = list(labels)
    n = len(labels)
    if n < 2:
        return None
    agree = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            agree += 1 if labels[i] == labels[j] else 0
    return agree / pairs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def shingles(text: str, k: int = 3) -> set:
    """Word k-gram shingles for cheap text-similarity comparisons."""
    words = [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if w]
    if len(words) < k:
        return set([" ".join(words)]) if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def text_similarity(a: str, b: str, k: int = 3) -> float:
    return jaccard(shingles(a, k), shingles(b, k))


def containment(needle: str, haystack: str, k: int = 3) -> float:
    """Fraction of `needle` shingles that appear in `haystack`."""
    n = shingles(needle, k)
    if not n:
        return 0.0
    h = shingles(haystack, k)
    return len(n & h) / len(n)
