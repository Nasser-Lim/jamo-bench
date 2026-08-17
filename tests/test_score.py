# -*- coding: utf-8 -*-
"""jamo_scorer_probe.py의 13종 엣지케이스 regression test (설계서 §8.5 요구사항).

기대값은 jamo_scorer_probe.py를 실제로 실행해 얻은 출력을 그대로 옮긴 것 —
새 라이브러리가 검증 스크립트와 어긋나면 이 테스트가 실패해야 한다.
"""
import pytest

from jamo_bench.score import score

# (target, pred, verdict, onset_ok, nucleus_ok, coda_ok, description)
CASES = [
    ("읽", "엮", "VALID", True, False, False, "파일럿 실측: 겹받침 붕괴"),
    ("책", "챽", "VALID", True, False, True, "파일럿 실측: 중성 오류"),
    ("일", "엄", "VALID", True, False, False, "파일럿 실측: 중성+종성"),
    ("입", "업", "VALID", True, False, True, "파일럿 실측: 중성만"),
    ("가", "가", "VALID", True, True, True, "정답"),
    ("이", "익", "VALID", True, True, False, "종성 삽입(원래 없음)"),
    ("읽", "익", "VALID", True, True, False, "겹받침→단순종성"),
    ("읽", "읽기", "OVERGEN", None, None, None, "과생성 2글자"),
    ("읽", "", "EMPTY", None, None, None, "빈 출력"),
    ("읽", "ag", "NON_HANGUL", None, None, None, "비한글"),
    ("읽", "ㅇㅣㄺ", "NON_HANGUL", None, None, None, "자모 분리 출력"),
    ("의", "이", "VALID", True, False, True, "복합모음→단순"),
    ("값", "갑", "VALID", True, True, False, "겹받침 일부 탈락"),
]


@pytest.mark.parametrize("target,pred,verdict,onset_ok,nucleus_ok,coda_ok,desc", CASES, ids=[c[-1] for c in CASES])
def test_13_edge_cases(target, pred, verdict, onset_ok, nucleus_ok, coda_ok, desc):
    result = score(target, pred)
    assert result.verdict == verdict, f"{desc}: {target}->{pred!r}"
    assert result.onset_ok == onset_ok
    assert result.nucleus_ok == nucleus_ok
    assert result.coda_ok == coda_ok


def test_score_requires_valid_target():
    with pytest.raises(ValueError):
        score("A", "가")


class TestHallucination:
    """v5.1 §6 — 환각 분리. confidence 없이는 기존 13종 케이스 그대로."""

    def test_no_confidence_keeps_normal_substitution(self):
        result = score("읽", "책")
        assert result.verdict == "VALID"
        assert (result.onset_ok, result.nucleus_ok, result.coda_ok) == (False, False, False)

    def test_full_mismatch_with_high_confidence_is_hallucination(self):
        result = score("읽", "책", ocr_confidence=0.97)
        assert result.verdict == "HALLUCINATED"

    def test_full_mismatch_with_low_confidence_stays_valid(self):
        result = score("읽", "책", ocr_confidence=0.5)
        assert result.verdict == "VALID"

    def test_partial_overlap_is_not_hallucination_even_at_high_confidence(self):
        # 읽→엮: 초성(ㅇ)이 보존되므로 "완전히 다른 글자"가 아니다.
        result = score("읽", "엮", ocr_confidence=0.99)
        assert result.verdict == "VALID"
