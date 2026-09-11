import pytest

from swarmeval import stats


def test_wilson_and_proportion():
    lo, hi = stats.wilson_interval(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
    s = stats.proportion_summary([True] * 9 + [False])
    assert s.mean == pytest.approx(0.9) and s.n == 10


def test_bootstrap_and_paired_delta():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = stats.bootstrap_ci(vals, seed=1)
    assert lo <= 3.0 <= hi
    d, (lo, hi) = stats.paired_bootstrap_delta([0, 0, 0, 0], [1, 1, 1, 1])
    assert d == 1.0 and lo == hi == 1.0
    d, ci = stats.paired_bootstrap_delta([1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1])
    assert ci[0] <= d <= ci[1]


def test_text_similarity_and_containment():
    a = "the quick brown fox jumps over the lazy dog"
    assert stats.text_similarity(a, a) == 1.0
    assert stats.text_similarity(a, "completely different words here now") == 0.0
    assert stats.containment("quick brown fox", a) == 1.0
    assert stats.jaccard(set(), set()) == 1.0


def test_agreement_and_kappa():
    assert stats.pairwise_agreement([True, True, False]) == pytest.approx(1 / 3)
    assert stats.cohens_kappa([True, False, True], [True, False, True]) == 1.0
    assert stats.pairwise_agreement([True]) is None
