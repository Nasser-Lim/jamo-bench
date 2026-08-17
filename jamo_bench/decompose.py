# -*- coding: utf-8 -*-
"""자모 분해 + 셀 분류 메타데이터.

JAMO_benchmark_design.md §6.2, §6.3, §8.5 및 JAMO_v51_patch.md §3, §4의
분류 규칙을 정식 함수로 승격한 모듈. jamo_18cell_design.py /
jamo_v51_verification.py의 검증 로직과 동일한 상수를 쓴다 — 두 코드베이스의
숫자가 어긋나면 설계 검증 스크립트가 라이브러리를 감사하는 의미가 없어진다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

SBASE = 0xAC00
SCOUNT = 11172
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT  # 588

ONSETS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
VOWELS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
FINALS = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

assert len(ONSETS) == LCOUNT
assert len(VOWELS) == VCOUNT
assert len(FINALS) == TCOUNT

# --- 모음 분류 -----------------------------------------------------------
# 샘플링 축(18셀)에서는 이중자모(ㅐㅒㅔㅖ)를 단순모음에 병합한다(v5 §6.2).
# 단, ㅐ↔ㅔ는 한글 최대 혼동쌍이므로 메타데이터 축(vowel_shape)은 별도 보존한다
# (v5.1 §4, B-5).
COMPLEX_V = set("ㅘㅙㅚㅝㅞㅟㅢ")   # 3획 이상 결합 모음 (7종)
DIPH_V = set("ㅐㅒㅔㅖ")             # 이중자모, 시각적으론 단순 세로형 (4종)

# --- 종성 분류 -------------------------------------------------------------
# 3-way: 샘플링 축(18셀) 유지용. 쌍받침(ㄲㅆ)은 여기서는 simple_T에 섞인다.
# 4-way: 메타데이터/2차 분석용(v5.1 §3, B-4) — 쌍받침을 분리한다.
CLUSTER_T = set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")  # 이질 겹받침 11종
TENSED_T = set("ㄲㅆ")                      # 쌍받침 2종

# --- 초성 분류 (시각 복잡도 기반, v4의 음운 분류를 대체 — v5 §6.3, C-2) ------
SIMPLE_ONSET = set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈ")  # 기본형 9종 (ㅇ 포함)
ASPIR_ONSET = set("ㅊㅋㅌㅍㅎ")            # 가획형 5종
TENSE_ONSET = set("ㄲㄸㅃㅆㅉ")            # 쌍자음 5종

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Syllable:
    """분해된 한글 완성형 음절 1개."""

    char: str
    onset: str
    nucleus: str
    coda: str  # 종성 없으면 ""
    onset_idx: int
    nucleus_idx: int
    coda_idx: int

    @property
    def vowel_class_2(self) -> str:
        return "complex_V" if self.nucleus in COMPLEX_V else "simple_V"

    @property
    def vowel_shape(self) -> str:
        if self.nucleus in COMPLEX_V:
            return "complex_block"
        if self.nucleus in DIPH_V:
            return "vertical_derived"
        return "vertical_simple"

    @property
    def is_ae_e_pair_member(self) -> bool:
        return self.nucleus in DIPH_V

    @property
    def coda_class_3(self) -> str:
        if self.coda == "":
            return "no_T"
        if self.coda in CLUSTER_T:
            return "cluster_T"
        return "simple_T"

    @property
    def coda_class_4(self) -> str:
        if self.coda == "":
            return "none"
        if self.coda in TENSED_T:
            return "tensed_double"
        if self.coda in CLUSTER_T:
            return "cluster_mixed"
        return "simple_single"

    @property
    def onset_group(self) -> str:
        if self.onset in ASPIR_ONSET:
            return "aspir_O"
        if self.onset in TENSE_ONSET:
            return "tense_O"
        return "simple_O"

    @property
    def is_ieung_onset(self) -> bool:
        return self.onset == "ㅇ"

    @property
    def cell_id(self) -> tuple:
        """18셀 키: (모음유형 2way, 종성유형 3way, 초성군)."""
        return (self.vowel_class_2, self.coda_class_3, self.onset_group)

    @property
    def cell_id_6(self) -> tuple:
        """Cross-Struct 6셀 키: 초성군을 접은 (모음유형, 종성유형)."""
        return (self.vowel_class_2, self.coda_class_3)


def decompose(ch: str) -> Optional[Syllable]:
    """단일 문자를 자모로 분해한다. 현대 한글 완성형이 아니면 None."""
    if len(ch) != 1:
        return None
    s = ord(ch) - SBASE
    if not (0 <= s < SCOUNT):
        return None
    onset_idx = s // NCOUNT
    nucleus_idx = (s % NCOUNT) // TCOUNT
    coda_idx = s % TCOUNT
    return Syllable(
        char=ch,
        onset=ONSETS[onset_idx],
        nucleus=VOWELS[nucleus_idx],
        coda=FINALS[coda_idx],
        onset_idx=onset_idx,
        nucleus_idx=nucleus_idx,
        coda_idx=coda_idx,
    )


def is_hangul_syllable(ch: str) -> bool:
    return decompose(ch) is not None


def compose(onset: str, nucleus: str, coda: str = "") -> Optional[str]:
    """decompose()의 역함수 — 초/중/종성 자모로 완성형 음절 1개를 조립한다.

    VLM judge(§8.1 폴백)가 "무슨 글자로 보이는가"를 통짜로 답하게 하면
    CLOVA와 같은 사전 편향("그럴듯한 실제 단어"로 보정)에 빠질 위험이
    있다. 대신 모델에게 **보이는 자모 각각**을 답하게 하고, 최종 글자는
    이 함수로 우리가 직접 조립한다 — 모델이 자모 인식 단계를 건너뛰고
    바로 "그럴듯한 단어"를 내놓을 여지를 없앤다.

    유효하지 않은 자모 조합(예: 목록에 없는 문자)이면 None을 반환한다.
    """
    try:
        onset_idx = ONSETS.index(onset)
        nucleus_idx = VOWELS.index(nucleus)
        coda_idx = FINALS.index(coda)
    except ValueError:
        return None
    return chr(SBASE + onset_idx * NCOUNT + nucleus_idx * TCOUNT + coda_idx)


def normalize(text: str) -> str:
    """Word 서브셋 타깃/예측 정규화 (v5.1 §8.1).

    NFC 정규화 → 공백 전부 제거 → 구두점 제거. 순서를 바꾸면 안 된다
    (NFC를 먼저 해야 결합 자모가 완성형으로 합쳐진 뒤 공백/구두점 판정이 된다).
    """
    text = unicodedata.normalize("NFC", text)
    return _PUNCT_RE.sub("", text)


@lru_cache(maxsize=1)
def all_syllables() -> tuple:
    """11,172 전수 음절의 Syllable 튜플 (모듈 1회 계산, 캐시)."""
    return tuple(decompose(chr(SBASE + i)) for i in range(SCOUNT))


@lru_cache(maxsize=1)
def cell_index_18() -> dict:
    """18셀 → 해당 셀에 속하는 음절 문자 리스트."""
    idx: dict = {}
    for syl in all_syllables():
        idx.setdefault(syl.cell_id, []).append(syl.char)
    return idx


@lru_cache(maxsize=1)
def cell_index_6() -> dict:
    """Cross-Struct 6셀(초성군 접음) → 해당 셀 음절 문자 리스트."""
    idx: dict = {}
    for syl in all_syllables():
        idx.setdefault(syl.cell_id_6, []).append(syl.char)
    return idx


ALL_18_CELLS = tuple(
    (v, t, o)
    for v in ("simple_V", "complex_V")
    for t in ("no_T", "simple_T", "cluster_T")
    for o in ("simple_O", "aspir_O", "tense_O")
)

ALL_6_CELLS = tuple(
    (v, t) for v in ("simple_V", "complex_V") for t in ("no_T", "simple_T", "cluster_T")
)
