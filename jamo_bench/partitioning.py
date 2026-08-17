# -*- coding: utf-8 -*-
"""표본 파티셔닝 — Core 540 / Cross-shared 250 / Cross-Struct 180.

JAMO_v51_patch.md §19 "데이터 파티션 단일 스펙"의 실행 순서를 그대로
파이프라인화한다. 순서를 바꾸면 오염이 난다는 것이 원문의 경고이므로, 여기서도
그 순서를 하드코딩한 함수 호출 체인으로 강제한다:

    1. 금칙어 목록 제외
    2. Minimal-pair reserve 확보                 (§2, v5.1)
    3. private-test holdout 분리 (구조 층화)      (§19 step 3)
    4. Core 540 확정 (18셀×30, 층화 제약 포함)     (§19 step 4, §3~4 제약)
    5. Cross-shared 250 ⊂ Core 540                (§1)
    6. Cross-Struct 180 ⊂ Core 540                (§12)

Word-Freq/Word-Struct(§19 step 7)는 실제 한국어 단어 빈도 코퍼스가 있어야
하므로 이 모듈에서는 구현하지 않는다 — `build_word_subsets`는 그 자리를
표시하는 스텁이다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

# `from . import decompose` would resolve to the *function* `decompose`
# re-exported by jamo_bench/__init__.py (package attribute shadowing — the
# package's `decompose` name gets overwritten by `from .decompose import
# decompose` in __init__.py). Importing directly from the submodule sidesteps
# that shadowing entirely.
from .decompose import ALL_18_CELLS, ALL_6_CELLS, cell_index_18
from .decompose import decompose as decompose_char

Cell18 = Tuple[str, str, str]
Cell6 = Tuple[str, str]

# --- Minimal-pair reserve set (design §8.10 + v5.1 §2) ---------------------
# 메인 피겨(최소쌍 분석)를 약속했으면 셀 무작위 층화만으로는 사다리가 통째로
# 빠질 수 있다(v5.1 §2). 그래서 샘플링 "전에" 예약하고 셀 quota에 우선
# 배정한다.
MINIMAL_PAIR_RESERVE_GROUPS: Dict[str, List[str]] = {
    "coda_ladder": ["가", "각", "간", "갈", "감", "갑", "값"],
    "coda_complexity": ["일", "입", "잉", "읽"],
    "complex_vowel": ["와", "워", "왜", "의", "외", "위"],
    "tensed_contrast": ["각", "갂", "갓", "갔"],  # v5.1 §2 신규
}


def minimal_pair_reserve_map() -> Dict[str, str]:
    """음절 → 소속 reserve 그룹. 그룹 간 중복(예: '각')은 먼저 등장한
    그룹으로 귀속시킨다(어느 그룹으로 보고되든 quota 확보 목적은 동일)."""
    mapping: Dict[str, str] = {}
    for group, chars in MINIMAL_PAIR_RESERVE_GROUPS.items():
        for c in chars:
            mapping.setdefault(c, group)
    return mapping


def _cell_char_lists() -> Dict[Cell18, List[str]]:
    return cell_index_18()


def _stratified_cell_sample(
    cell: Cell18,
    cell_chars: Sequence[str],
    reserved_in_cell: Sequence[str],
    quota: int,
    rng: np.random.Generator,
    tensed_min_frac: float,
    vertical_min_frac: float,
    vertical_max_frac: float,
) -> List[str]:
    """한 셀 안에서 reserve 우선 배정 + tensed_double/vertical_derived 층화
    제약(v5.1 §3, §4)을 만족하는 quota개 표본을 뽑는다."""
    v, t, _o = cell

    reserved = sorted(set(reserved_in_cell))
    if len(reserved) > quota:
        reserved = reserved[:quota]
    selected: List[str] = list(reserved)
    selected_set = set(selected)

    remaining = [c for c in cell_chars if c not in selected_set]
    rng.shuffle(remaining)

    def syl(c: str):
        return decompose_char(c)

    # 제약 1: simple_T 셀은 tensed_double(ㄲㅆ 받침)을 20% 이상 포함
    if t == "simple_T":
        current = sum(1 for c in selected if syl(c).coda_class_4 == "tensed_double")
        min_needed = math.ceil(tensed_min_frac * quota)
        gap = max(0, min_needed - current)
        if gap > 0:
            cands = [c for c in remaining if syl(c).coda_class_4 == "tensed_double"]
            take = cands[: min(gap, len(cands), quota - len(selected))]
            selected.extend(take)
            selected_set.update(take)
            remaining = [c for c in remaining if c not in selected_set]

    # 제약 2: simple_V 셀은 vertical_derived(ㅐㅒㅔㅖ)를 30~40% 포함
    if v == "simple_V":
        min_vd = math.ceil(vertical_min_frac * quota)
        max_vd = math.floor(vertical_max_frac * quota)
        current = sum(1 for c in selected if syl(c).vowel_shape == "vertical_derived")
        gap = max(0, min_vd - current)
        if gap > 0:
            cands = [c for c in remaining if syl(c).vowel_shape == "vertical_derived"]
            take = cands[: min(gap, len(cands), quota - len(selected))]
            selected.extend(take)
            selected_set.update(take)
            remaining = [c for c in remaining if c not in selected_set]
        current = sum(1 for c in selected if syl(c).vowel_shape == "vertical_derived")
        room = max_vd - current
        vd_in_remaining = [c for c in remaining if syl(c).vowel_shape == "vertical_derived"]
        if room <= 0:
            remaining = [c for c in remaining if syl(c).vowel_shape != "vertical_derived"]
        elif len(vd_in_remaining) > room:
            # 무작위 최종 채움에서 vertical_derived가 room개를 넘게 뽑히지
            # 않도록, 남길 후보 수를 room개로 미리 잘라둔다(이미 shuffle된
            # remaining에서 뽑은 것이므로 여전히 무작위 부분집합이다).
            keep = set(vd_in_remaining[:room])
            remaining = [c for c in remaining if syl(c).vowel_shape != "vertical_derived" or c in keep]

    needed = quota - len(selected)
    if needed > 0:
        if len(remaining) <= needed:
            selected.extend(remaining)
        else:
            idx = rng.choice(len(remaining), size=needed, replace=False)
            selected.extend(remaining[i] for i in idx)

    return selected[:quota]


@dataclass(frozen=True)
class PartitionResult:
    seed: int
    banned_syllables: FrozenSet[str]
    reserve_map: Dict[str, str]
    private_test_holdout: Tuple[str, ...]
    core_by_cell: Dict[Cell18, Tuple[str, ...]]
    core_540: Tuple[str, ...]
    cell_available_counts: Dict[Cell18, int]
    cross_shared_250: Tuple[str, ...]
    cross_struct_by_cell6: Dict[Cell6, Tuple[str, ...]]
    cross_struct_180: Tuple[str, ...]
    warnings: Tuple[str, ...]
    plan_b_prime_triggered: bool

    def summary(self) -> str:
        lines = [
            f"seed={self.seed}  banned={len(self.banned_syllables)}  "
            f"reserve={len(self.reserve_map)}  private_holdout={len(self.private_test_holdout)}",
            f"Core 540: {len(self.core_540)} unique syllables "
            f"({'OK' if len(self.core_540) == 540 else 'MISMATCH'})",
        ]
        lines.append(f"{'cell':<30}{'available':>10}{'selected':>10}")
        for cell in ALL_18_CELLS:
            n_avail = self.cell_available_counts[cell]
            n_sel = len(self.core_by_cell[cell])
            lines.append(f"{'/'.join(cell):<30}{n_avail:>10}{n_sel:>10}")
        lines.append(
            f"Cross-shared 250: {len(self.cross_shared_250)}  "
            f"Cross-Struct 180: {len(self.cross_struct_180)}"
        )
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.plan_b_prime_triggered:
            lines.append("Plan B' TRIGGER: 30 미만 셀이 2개 이상 (§6.5 폴백 사다리 참고)")
        return "\n".join(lines)

    def to_json(self) -> dict:
        def key18(cell: Cell18) -> str:
            return "|".join(cell)

        def key6(cell: Cell6) -> str:
            return "|".join(cell)

        return {
            "seed": self.seed,
            "banned_syllables": sorted(self.banned_syllables),
            "reserve_map": self.reserve_map,
            "private_test_holdout": list(self.private_test_holdout),
            "core_by_cell": {key18(k): list(v) for k, v in self.core_by_cell.items()},
            "core_540": list(self.core_540),
            "cell_available_counts": {key18(k): v for k, v in self.cell_available_counts.items()},
            "cross_shared_250": list(self.cross_shared_250),
            "cross_struct_by_cell6": {key6(k): list(v) for k, v in self.cross_struct_by_cell6.items()},
            "cross_struct_180": list(self.cross_struct_180),
            "warnings": list(self.warnings),
            "plan_b_prime_triggered": self.plan_b_prime_triggered,
        }

    def to_json_str(self, **kwargs) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, **kwargs)


def partition(
    seed: int,
    banned_syllables: FrozenSet[str] = frozenset(),
    quota_per_cell: int = 30,
    private_holdout_per_cell: int = 5,
    cross_shared_total: int = 250,
    tensed_min_frac: float = 0.20,
    vertical_min_frac: float = 0.30,
    vertical_max_frac: float = 0.40,
) -> PartitionResult:
    """v5.1 §19 실행 순서를 그대로 따르는 파티셔닝 파이프라인.

    같은 seed는 항상 같은 결과를 낸다(재현성 요구, §6.4/§19).
    """
    rng = np.random.default_rng(seed)
    cell_index = _cell_char_lists()

    # 1. 금칙어 제외
    pool = {c for chars in cell_index.values() for c in chars} - set(banned_syllables)

    # 2. minimal-pair reserve 확보
    reserve_map = minimal_pair_reserve_map()
    reserve_chars = set(reserve_map) & pool

    # 3. private-test holdout 분리 (reserve는 holdout 후보에서 제외 — Core에
    #    남아 있어야 하는 확정 항목이므로)
    private_holdout: set = set()
    for cell in ALL_18_CELLS:
        candidates = [c for c in cell_index[cell] if c in pool and c not in reserve_chars]
        rng.shuffle(candidates)
        private_holdout.update(candidates[:private_holdout_per_cell])

    core_pool = pool - private_holdout

    # 4. Core 540 확정
    core_by_cell: Dict[Cell18, List[str]] = {}
    cell_available_counts: Dict[Cell18, int] = {}
    warnings: List[str] = []
    for cell in ALL_18_CELLS:
        cell_chars_all = [c for c in cell_index[cell] if c in core_pool]
        cell_available_counts[cell] = len(cell_chars_all)
        reserved_in_cell = [c for c in reserve_chars if c in cell_chars_all]
        selected = _stratified_cell_sample(
            cell,
            cell_chars_all,
            reserved_in_cell,
            quota_per_cell,
            rng,
            tensed_min_frac,
            vertical_min_frac,
            vertical_max_frac,
        )
        core_by_cell[cell] = selected
        if len(cell_chars_all) < quota_per_cell:
            warnings.append(
                f"cell {cell} available={len(cell_chars_all)} < quota={quota_per_cell}"
            )

    core_540 = sorted({c for chars in core_by_cell.values() for c in chars})
    plan_b_prime = sum(1 for n in cell_available_counts.values() if n < quota_per_cell) >= 2

    # 5. Cross-shared 250 ⊂ Core 540 (18셀 층화, 셀당 ~14개 → 250 조정)
    n_cells = len(ALL_18_CELLS)
    base_quota = cross_shared_total // n_cells
    remainder = cross_shared_total - base_quota * n_cells  # 초과분 (양수) 또는 삭감분 (음수)
    quotas = {cell: base_quota for cell in ALL_18_CELLS}
    order = list(ALL_18_CELLS)
    rng.shuffle(order)
    if remainder > 0:
        for cell in order[:remainder]:
            quotas[cell] += 1
    elif remainder < 0:
        for cell in order[: -remainder]:
            quotas[cell] -= 1

    cross_shared: List[str] = []
    for cell in ALL_18_CELLS:
        chars = list(core_by_cell[cell])
        rng.shuffle(chars)
        q = min(quotas[cell], len(chars))
        cross_shared.extend(chars[:q])
    cross_shared_250 = sorted(cross_shared)

    # 6. Cross-Struct 180 ⊂ Core 540 (6셀 = 2모음×3종성, 초성군 접음)
    cross_struct_by_cell6: Dict[Cell6, List[str]] = {}
    for cell6 in ALL_6_CELLS:
        v, t = cell6
        combined: List[str] = []
        for o in ("simple_O", "aspir_O", "tense_O"):
            combined.extend(core_by_cell[(v, t, o)])
        rng.shuffle(combined)
        cross_struct_by_cell6[cell6] = sorted(combined[:30])
    cross_struct_180 = sorted({c for chars in cross_struct_by_cell6.values() for c in chars})

    return PartitionResult(
        seed=seed,
        banned_syllables=frozenset(banned_syllables),
        reserve_map=reserve_map,
        private_test_holdout=tuple(sorted(private_holdout)),
        core_by_cell={k: tuple(v) for k, v in core_by_cell.items()},
        core_540=tuple(core_540),
        cell_available_counts=cell_available_counts,
        cross_shared_250=tuple(cross_shared_250),
        cross_struct_by_cell6={k: tuple(v) for k, v in cross_struct_by_cell6.items()},
        cross_struct_180=tuple(cross_struct_180),
        warnings=tuple(warnings),
        plan_b_prime_triggered=plan_b_prime,
    )


def build_word_subsets(*args, **kwargs):
    """Word-Freq(125)/Word-Struct(125) 서브셋 (설계서 §6.8) — 스텁.

    Word-Freq는 실사용 상위 빈도 2~3음절 한국어 단어 목록(실제 코퍼스)이
    있어야 정의할 수 있고, 이 저장소에는 아직 그 데이터가 없다. 자모
    조합만으로 "실사용 빈도가 높은 단어"를 합성할 수 없으므로 이 함수는
    의도적으로 미구현 상태로 남겨둔다 — 코퍼스 확보 후 채운다.
    """
    raise NotImplementedError(
        "Word-Freq/Word-Struct 서브셋은 실제 한국어 단어 빈도 코퍼스가 필요합니다. "
        "docs/partitioning.md의 §19 step 7 참고 — 코퍼스 확보 후 구현 예정."
    )
