# -*- coding: utf-8 -*-
import math

from jamo_bench.metrics import (
    bootstrap_ci,
    chance_corrected_acc,
    chance_model,
    chance_target,
    chance_uniform,
    confusion_matrix,
)


def test_chance_uniform_matches_structural_space():
    c = chance_uniform()
    assert c["onset"] == 1 / 19
    assert c["nucleus"] == 1 / 21
    assert c["coda"] == 1 / 28


def test_chance_target_two_category_balanced():
    labels = ["a", "a", "b", "b"]
    assert math.isclose(chance_target(labels), 0.5)


def test_chance_target_natural_distribution_beats_balanced():
    # 자연 분포(55% no-coda)의 우연일치가 균형 분포(1/3씩)보다 커야 한다
    # (design §8.5.3의 정성적 결론: 자연분포 31.0% vs 균형 12.5%)
    natural = ["no_T"] * 55 + ["simple_T"] * 30 + ["cluster_T"] * 15
    balanced = ["no_T"] * 33 + ["simple_T"] * 34 + ["cluster_T"] * 33
    assert chance_target(natural) > chance_target(balanced)


def test_chance_target_empty_is_zero():
    assert chance_target([]) == 0.0


def test_chance_model_is_deterministic_with_seed():
    target = ["ㄱ", "ㄴ", "ㄱ", "ㄷ"] * 25
    pred = ["ㄱ", "ㄱ", "ㄷ", "ㄴ"] * 25
    r1 = chance_model(target, pred, n_permutations=200, seed=7)
    r2 = chance_model(target, pred, n_permutations=200, seed=7)
    assert r1 == r2
    assert 0.0 <= r1 <= 1.0


def test_chance_model_requires_equal_length():
    import pytest

    with pytest.raises(ValueError):
        chance_model(["a"], ["a", "b"])


def test_chance_corrected_acc_boundaries():
    assert math.isclose(chance_corrected_acc(0.8, 0.5), 0.6)
    assert chance_corrected_acc(0.5, 0.5) == 0.0
    assert chance_corrected_acc(1.0, 0.5) == 1.0
    assert chance_corrected_acc(0.9, 1.0) == 0.0  # 퇴화 케이스: 분모 0 방지


def test_bootstrap_ci_constant_values():
    lo, hi = bootstrap_ci([1.0] * 50, n_boot=200, seed=0)
    assert lo == hi == 1.0


def test_bootstrap_ci_brackets_true_mean():
    values = [1.0, 0.0] * 100  # 평균 0.5
    lo, hi = bootstrap_ci(values, n_boot=1000, seed=0)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_confusion_matrix_counts_pairs():
    pairs = [("ㄱ", "ㄱ"), ("ㄱ", "ㄴ"), ("ㄱ", "ㄱ")]
    cm = confusion_matrix(pairs)
    assert cm[("ㄱ", "ㄱ")] == 2
    assert cm[("ㄱ", "ㄴ")] == 1
