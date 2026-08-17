# -*- coding: utf-8 -*-
"""사람 감사(§9) 큐 빌더 — target-blind transcription용 문항 생성.

judge_ceiling.json에 이미 기록된 clean 표본(음절·폰트·CLOVA 판독 결과)을
그대로 재사용한다. 이미지는 저장돼 있지 않지만 forge_render.render_clean()이
결정론적이라(char, font_name, canvas_size, text_area_frac이 같으면 항상
같은 픽셀) 재생성 비용 없이 CLOVA가 실제로 봤던 것과 동일한 이미지를
복원할 수 있다.

문항 순서는 clean_accuracy가 낮은 셀부터 — "가장 의심스러운 셀부터
사람이 먼저 본다"는 감사 우선순위 원칙(§9.1)을 그대로 따른다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

CEILING_JSON = Path(__file__).resolve().parent.parent / "results" / "judge_ceiling.json"
CANVAS_SIZE = 1024
TEXT_AREA_FRAC = 0.3

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_JSONL = REPO_ROOT / "results" / "pilot" / "pilot_results_v2.jsonl"
TEMPLATE_MATCH_EVAL_JSON = REPO_ROOT / "results" / "template_match_full_eval.json"
CLOVA_OCC010_EVAL_JSON = REPO_ROOT / "results" / "clova_occ010_full_eval.json"


@dataclass(frozen=True)
class AuditItem:
    item_id: str
    char: str  # 타깃 (사람에게는 숨김)
    font_name: str
    cell_key: str
    cell_clean_accuracy: float
    clova_reading: Optional[str]
    clova_correct: bool


def build_audit_queue(ceiling_json_path: Path = CEILING_JSON) -> List[AuditItem]:
    data = json.loads(ceiling_json_path.read_text(encoding="utf-8"))
    cells = sorted(data["per_cell"].items(), key=lambda kv: kv[1]["clean_accuracy"])

    items: List[AuditItem] = []
    for cell_key, cell in cells:
        clean_samples = [s for s in cell["samples"] if s["condition"] == "clean"]
        for i, s in enumerate(clean_samples):
            item_id = f"{cell_key}__{s['font_name']}__{s['char']}__{i}"
            items.append(
                AuditItem(
                    item_id=item_id,
                    char=s["char"],
                    font_name=s["font_name"],
                    cell_key=cell_key,
                    cell_clean_accuracy=cell["clean_accuracy"],
                    clova_reading=s["candidate_text"],
                    clova_correct=bool(s["correct"]),
                )
            )
    return items


@dataclass(frozen=True)
class PilotAuditItem:
    """실제 Seedream 파일럿 이미지용 감사 문항. `AuditItem`과 달리 이미지를
    렌더링이 아니라 디스크에서 읽는다 — 판독 대상이 폰트가 아니라 실제
    생성물이기 때문이다. 판정자 두 개(CLOVA, template_match)의 판독을
    함께 들고 있어 감사 결과 하나로 셋 다(사람 vs 타깃, 사람 vs 각 판정자,
    판정자끼리) 비교할 수 있다."""

    item_id: str
    target: str  # 사람에게는 숨김
    image_path: str
    template_id: str
    coda_type: str  # no_T / simple_T / cluster_T — 요약 표 그룹핑 기준
    clova_reading: Optional[str]
    clova_correct: bool
    template_match_reading: Optional[str]
    template_match_correct: bool


def build_pilot_audit_queue(
    pilot_jsonl_path: Path = PILOT_JSONL,
    tm_eval_path: Path = TEMPLATE_MATCH_EVAL_JSON,
    clova_eval_path: Path = CLOVA_OCC010_EVAL_JSON,
) -> List[PilotAuditItem]:
    """Route B + jamo_valid=VALID(OVERGEN 등 제외) 256건 전체를 문항으로
    만든다. CLOVA와 template_match가 **서로 다르게 읽은 문항을 먼저**
    배치한다 — 의견이 갈리는 지점이 가장 정보가치가 높다(§9.1 원칙과
    동일한 논리)."""
    pilot_recs = [
        json.loads(line)
        for line in pilot_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clean = [r for r in pilot_recs if r.get("route") == "B" and r.get("jamo_valid") == "VALID"]

    tm_eval = json.loads(tm_eval_path.read_text(encoding="utf-8"))
    tm_mismatch_by_path: Dict[str, dict] = {m["image_path"]: m for m in tm_eval["mismatches"]}

    clova_eval = json.loads(clova_eval_path.read_text(encoding="utf-8"))
    clova_by_path: Dict[str, dict] = {c["image_path"]: c for c in clova_eval}

    items: List[PilotAuditItem] = []
    for r in clean:
        image_path = r["image_path"]
        target = r["target"]

        tm_mismatch = tm_mismatch_by_path.get(image_path)
        tm_reading = tm_mismatch["pred"] if tm_mismatch else target
        tm_correct = tm_mismatch is None

        clova_rec = clova_by_path.get(image_path)
        clova_reading = clova_rec["pred"] if clova_rec else None
        clova_correct = bool(clova_rec["correct"]) if clova_rec else False

        item_id = f"{image_path}"
        items.append(
            PilotAuditItem(
                item_id=item_id,
                target=target,
                image_path=image_path,
                template_id=r["template_id"],
                coda_type=r["final_class"],
                clova_reading=clova_reading,
                clova_correct=clova_correct,
                template_match_reading=tm_reading,
                template_match_correct=tm_correct,
            )
        )

    # 의견이 갈리는 문항 우선
    items.sort(key=lambda it: it.clova_reading == it.template_match_reading)
    return items


# --- Phase 2 라벨 체계 -------------------------------------------------------
#
# 1차 감사(256건)의 라벨은 text / illegible / extra_text 3종이었는데, 감사자가
# `illegible`에 실제로 적용한 기준은 "읽을 수 없음"이 아니라 **"윈도우 키보드로
# 입력이 불가능한 형태"**(= 11,172 완성형 집합의 원소가 아님)였다. 이름이 기준을
# 표현하지 못하므로 2번째 감사자에게 같은 UI를 주면 다른 기준으로 누르게 되어
# Krippendorff α가 무의미해진다(`docs/PROGRESS.md` 12단계).
#
# 또한 `illegible` 44건 안에는 성질이 다른 것들이 섞여 있다 — 획이 틀린 한글
# (`감`의 받침이 田), 다중 글자(`쫄읙`), 진짜 비한글 기호. 이들은 각각 모델의
# 다른 실패 유형이고 자동 판정자에게 요구하는 능력도 다르므로 분리해야 한다.
PHASE2_LABELS = (
    # (라벨, 화면 표시, 전사 요구 여부)
    ("valid_syllable", "유효한 한글 한 글자 (키보드로 입력 가능)", True),
    ("malformed", "한글 구조지만 획이 틀려 입력 불가능", False),
    ("multi_syllable", "글자가 2개 이상", False),
    ("non_hangul", "한글이 아님 (로마자·기호·도형)", False),
    ("no_text", "글자 시도 자체가 없음", False),
)

# 1차 감사 라벨 → Phase 2 라벨. `illegible`은 malformed와 non_hangul 양쪽에
# 걸쳐 있어 자동 변환이 불가능하다 — 그래서 재라벨링이 필요하다.
PHASE1_LABEL_MAP = {
    "text": "valid_syllable",
    "extra_text": "multi_syllable",
    "illegible": None,  # 재라벨링 필요
}


def build_pilot_phase2_queue(
    pilot_jsonl_path: Path = PILOT_JSONL,
    phase1_results_path: Optional[Path] = None,
) -> List[PilotAuditItem]:
    """Phase 2 감사 큐 — 두 집단을 합쳐 파일럿 300장의 라벨을 완결한다.

    **집단 1: 미감사 44장** (route != B 또는 jamo_valid != VALID).
    `build_pilot_audit_queue()`가 `route == "B" and jamo_valid == "VALID"`로
    걸러 정확히 256건을 만들었기 때문에, Route A1/C와 OVERGEN 이미지는 사람이
    한 번도 본 적이 없다. 그래서 "Route 분류기가 무효 출력을 걸러내는가"라는
    질문에 아직 답할 수 없다 — 무효 44건이 전부 Route B에 있는 것은 Route의
    실패가 아니라 표본 구성상 필연이었다(`docs/PROGRESS.md` 12단계).
    이 집단을 감사하면 Route A1/C의 precision을 처음으로 측정할 수 있다.

    **집단 2: 1차에서 `illegible`로 표기된 44장.** Phase 2 라벨로 재분류해
    malformed / multi_syllable / non_hangul을 분리한다.

    두 집단의 크기가 우연히 같다(44/44) — 서로 다른 집합이므로 혼동하지 말 것.
    """
    pilot_recs = [
        json.loads(line)
        for line in pilot_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    audited_paths = set()
    relabel_paths = set()
    path = phase1_results_path or (REPO_ROOT / "results" / "audit" / "human_audit_pilot.jsonl")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            audited_paths.add(rec["image_path"])
            if PHASE1_LABEL_MAP.get(rec["label"]) is None:
                relabel_paths.add(rec["image_path"])

    items: List[PilotAuditItem] = []
    for r in pilot_recs:
        image_path = r["image_path"]
        unaudited = image_path not in audited_paths
        if not (unaudited or image_path in relabel_paths):
            continue
        # group을 item_id에 박아 1차 결과 파일과 id가 충돌하지 않게 한다.
        group = "unaudited" if unaudited else "relabel"
        items.append(
            PilotAuditItem(
                item_id=f"{group}::{image_path}",
                target=r["target"],
                image_path=image_path,
                template_id=r["template_id"],
                coda_type=r["final_class"],
                # 미감사 집단은 판정자 평가(256건 대상)에 포함되지 않아 판독값이
                # 없다. 감사는 target-blind라 판독값 없이도 성립한다.
                clova_reading=None,
                clova_correct=False,
                template_match_reading=None,
                template_match_correct=False,
            )
        )

    # 미감사 집단을 먼저 — Route precision이라는 미답 질문에 직결된다.
    items.sort(key=lambda it: not it.item_id.startswith("unaudited::"))
    return items
