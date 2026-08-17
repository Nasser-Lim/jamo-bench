# -*- coding: utf-8 -*-
"""Chance 보정, 신뢰구간, 혼동표 (JAMO_benchmark_design.md §8.5.3, §8.6).

jamo_chance_bias.py의 수치 검증 로직을 함수화한다. 3종 baseline을 항상 함께
보고해야 한다는 설계 원칙(§8.5.3) — 하나만 노출하면 "받침 전부 탈락" 같은
퇴화 전략이 어느 baseline에서는 숨겨질 수 있다.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

# 균등 무작위 우연일치 (구조 공간 전체 기준: 초성 19 / 중성 21 / 종성 28)
CHANCE_UNIFORM = {
    "onset": 1 / 19,
    "nucleus": 1 / 21,
    "coda": 1 / 28,
}


def chance_uniform() -> dict:
    """Chance_uniform (§8.5.3) — 균등 무작위 예측 시 초/중/종 우연일치."""
    return dict(CHANCE_UNIFORM)


def chance_target(labels: Sequence) -> float:
    """Chance_target (§8.5.3) — 평가셋 타깃 주변분포 기반 우연일치.

    두 독립적인 타깃 라벨이 우연히 같을 확률 = Σ p_i^2 (i는 라벨 카테고리).
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return sum((c / n) ** 2 for c in counts.values())


def chance_model(
    target_labels: Sequence,
    pred_labels: Sequence,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """Chance_model (§8.5.3) — 모델이 실제로 낸 예측 라벨을 샘플 간
    permutation한 baseline. 모델이 "받침 전부 탈락" 같은 퇴화 전략을 쓰면
    Chance_target보다 훨씬 강한(공정한) 기준선이 된다."""
    if len(target_labels) != len(pred_labels):
        raise ValueError("target_labels and pred_labels must be the same length")
    target_arr = np.asarray(target_labels)
    pred_arr = np.asarray(pred_labels)
    n = len(target_arr)
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    accs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(pred_arr)
        accs[i] = np.mean(perm == target_arr)
    return float(accs.mean())


def chance_corrected_acc(observed: float, chance: float) -> float:
    """(관측 − 우연) / (1 − 우연). chance가 1에 도달하면(퇴화 케이스) 0을
    반환한다 — 분모 0으로 나누는 대신 "보정 불가"를 명시적으로 표현."""
    if chance >= 1.0:
        return 0.0
    return (observed - chance) / (1 - chance)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple:
    """values(0/1 정오 라벨 또는 실수 지표)의 평균에 대한 percentile bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def confusion_matrix(pairs: Sequence[tuple]) -> Counter:
    """(target_jamo, pred_jamo) 쌍의 카운터. 초/중/종 각각 별도로 호출한다.

    예: confusion_matrix([(t.onset, p.onset) for t, p in scored_pairs])
    """
    return Counter(pairs)
