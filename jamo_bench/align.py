# -*- coding: utf-8 -*-
"""다음절(Word) 정렬 규칙 (JAMO_benchmark_design.md §8.5.2, JAMO_v51_patch.md §8).

Word 서브셋은 "책 읽기"→"챽 엮기"처럼 1:1 정렬이 되는 경우도 있지만, 병합·
분리·탈락(삽입/삭제)이 생기면 어느 음절을 어느 음절에 대응시켜 자모를 채점할지
정해야 한다. 이 모듈은 그 귀속 규칙(음절 단위 편집거리 정렬 → 치환 쌍만 자모
분해 → 삽입/삭제는 음절 수준 오류로 별도 계상)을 구현한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from .score import ScoreResult, score

AlignOpType = Literal["match", "substitute", "insert", "delete"]


@dataclass(frozen=True)
class AlignOp:
    op: AlignOpType
    target_char: Optional[str]
    pred_char: Optional[str]
    score_result: Optional[ScoreResult] = None


@dataclass(frozen=True)
class WordScoreResult:
    target: str
    pred: str
    ops: tuple  # tuple[AlignOp, ...]
    edit_distance: int
    overgen: bool
    syllable_insert_count: int
    syllable_delete_count: int
    route_hint: Optional[Literal["C"]] = None

    @property
    def substitution_scores(self) -> list:
        """자모 분해 대상(match/substitute)만 모은 ScoreResult 리스트."""
        return [op.score_result for op in self.ops if op.score_result is not None]


def align_syllables(target: str, pred: str) -> tuple:
    """음절 단위 Levenshtein 정렬. (ops, edit_distance) 대신 ops만 필요하면
    align_syllables(...)[0]으로 쓴다 — 이 함수는 (tuple[AlignOp,...], int)를
    반환한다."""
    n, m = len(target), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete target[i-1]
                dp[i][j - 1] + 1,       # insert pred[j-1]
                dp[i - 1][j - 1] + cost,  # match/substitute
            )

    ops: list = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and target[i - 1] == pred[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(AlignOp("match", target[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(AlignOp("substitute", target[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(AlignOp("insert", None, pred[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(AlignOp("delete", target[i - 1], None))
            i -= 1
        else:  # pragma: no cover - DP invariant guarantees one branch matches
            raise AssertionError("edit-distance backtrace reached an inconsistent state")
    ops.reverse()
    return tuple(ops), dp[n][m]


def score_word(
    target: str,
    pred: str,
    ocr_confidence: Optional[float] = None,
    length_guard_factor: int = 3,
) -> WordScoreResult:
    """다음절 타깃/예측을 정렬한 뒤 치환 쌍만 자모 분해한다.

    - insert/delete는 syllable_insert / syllable_delete로만 계상하고 자모
      분해 분모에는 넣지 않는다.
    - 정렬 자체가 불가능(edit_distance > len(target)) 하거나 예측 길이가
      타깃 길이의 length_guard_factor배를 넘으면 route_hint="C"를 반환한다
      (호출부가 Route C로 넘길지 판단하는 신호일 뿐, 여기서 예외를 던지지
      않는다 — 통계용 필드는 채워서 반환해야 후속 분석에서 드롭 사유를
      추적할 수 있다).
    - overgen(과생성)은 "타깃 전체가 삭제 없이 매치/치환되고, 남는 예측
      음절이 꼬리로 삽입되는" 경우로 정의한다(v5.1 §8.3, 예: 읽→읽기).
    """
    route_hint: Optional[Literal["C"]] = None
    if len(pred) > len(target) * length_guard_factor:
        route_hint = "C"

    ops, edit_distance = align_syllables(target, pred)

    if edit_distance > len(target):
        route_hint = "C"

    insert_count = sum(1 for op in ops if op.op == "insert")
    delete_count = sum(1 for op in ops if op.op == "delete")
    overgen = insert_count > 0 and delete_count == 0

    scored_ops = []
    for op in ops:
        if op.op in ("match", "substitute"):
            result = score(op.target_char, op.pred_char, ocr_confidence=ocr_confidence)
            scored_ops.append(AlignOp(op.op, op.target_char, op.pred_char, result))
        else:
            scored_ops.append(op)

    return WordScoreResult(
        target=target,
        pred=pred,
        ops=tuple(scored_ops),
        edit_distance=edit_distance,
        overgen=overgen,
        syllable_insert_count=insert_count,
        syllable_delete_count=delete_count,
        route_hint=route_hint,
    )
