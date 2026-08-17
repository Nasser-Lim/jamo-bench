# -*- coding: utf-8 -*-
"""정형성(well-formedness) 신호의 분리력 측정.

**푸는 문제.** 판정자가 "그려진 모양이 11,172 유효 음절 중 어느 것도
아니다"를 답할 수 있는가. 파일럿 256장 사람 감사(2026-08-10)에서 44건
(17.2%)이 "윈도우 키보드로 입력 불가능한 형태"였는데, CLOVA도
template_match도 전부 가장 가까운 유효 음절로 스냅해 답했고 그중 5건은
"정답"으로까지 처리됐다.

**왜 기존 시도가 실패했는가.** `docs/PROGRESS.md` 10단계는 template_match
유사도 점수 하나로 갈라보려다 실패했다(중앙값 0.734 vs 0.642, 중첩).
절대 점수의 전역 임계값은 음절마다 달성 가능한 최대치가 달라 원리적으로
작동하지 않는다. 여기서 재는 것은 전부 **상대적** 신호다.

  unexplained_ink  최적 템플릿으로 설명 안 되는 잉크 (여분 획)
  font_agreement   3개 폰트 뱅크가 독립적으로 같은 음절을 뽑는가
  margin           1위 − 2위 점수차
  score            (대조군 — 이미 실패한 것으로 알려진 신호)

**금지.** 신호는 전부 target-independent여야 한다. 타깃 순위·타깃과의
편집거리를 섞으면 "정답에 가까우면 통과"가 되어 자동 판정 구간이 정답
쪽으로 선별되고 정확도가 인위적으로 부풀려진다.

사람 라벨(label == "text" 여부)을 정답으로 두고 AUC와 최적 운용점을 낸다.
API 호출 0건 — 전부 로컬.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 기본 콘솔은 cp949라 한글·em dash 출력에서 죽는다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.template_match import read_by_template_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "results" / "audit" / "human_audit_pilot.jsonl"
OUT = ROOT / "results" / "wellformedness_eval.json"

# 방향: +1이면 값이 클수록 "무효(non-well-formed)", -1이면 작을수록 무효
SIGNALS = {
    "unexplained_ink": +1,
    "font_agreement": -1,
    "margin": -1,
    "score": -1,
}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """무효(pos)가 유효(neg)보다 높은 점수를 받을 확률. 0.5 = 무작위."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_operating_point(pos: np.ndarray, neg: np.ndarray) -> dict:
    """무효를 걸러내는 임계값 스윕. recall(무효 검출)과
    false_flag(유효를 무효로 오분류) 중 Youden J 최대점을 고른다."""
    best = {"threshold": None, "recall": 0.0, "false_flag": 0.0, "youden": -1.0}
    for t in np.unique(np.concatenate([pos, neg])):
        recall = float((pos >= t).mean())
        false_flag = float((neg >= t).mean())
        j = recall - false_flag
        if j > best["youden"]:
            best = {
                "threshold": float(t),
                "recall": recall,
                "false_flag": false_flag,
                "youden": float(j),
            }
    return best


def main() -> None:
    records = [json.loads(line) for line in AUDIT.open(encoding="utf-8")]
    print(f"감사 기록 {len(records)}건 — template_match 스윕 시작 (로컬, API 0건)")

    rows = []
    for i, rec in enumerate(records, 1):
        path = ROOT / rec["image_path"]
        if not path.is_file():
            path = Path(rec["image_path"])
        reading = read_by_template_match(Image.open(path))
        rows.append(
            {
                "image_path": rec["image_path"],
                "target": rec["target"],
                "coda_type": rec["coda_type"],
                "human_label": rec["label"],
                # 사람이 "유효 완성형 한 글자"로 인정했는가
                "well_formed": rec["label"] == "text",
                "predicted_char": reading.predicted_char,
                "score": reading.score,
                "margin": reading.margin,
                "font_agreement": reading.font_agreement,
                "unexplained_ink": reading.unexplained_ink,
                "missing_ink": reading.missing_ink,
                "per_font": [list(p) for p in reading.per_font],
            }
        )
        if i % 32 == 0:
            print(f"  {i}/{len(records)}")

    wf = np.array([r["well_formed"] for r in rows], dtype=bool)
    print(f"\n유효 {int(wf.sum())}건 / 무효 {int((~wf).sum())}건\n")

    summary = {}
    print(f"{'signal':18}{'AUC':>7}{'무효중앙':>10}{'유효중앙':>10}"
          f"{'임계':>8}{'검출률':>8}{'오탐률':>8}")
    print("-" * 69)
    for name, direction in SIGNALS.items():
        vals = np.array([r[name] for r in rows], dtype=float) * direction
        pos, neg = vals[~wf], vals[wf]
        a = auc(pos, neg)
        op = best_operating_point(pos, neg)
        summary[name] = {
            "auc": a,
            "direction": direction,
            "median_invalid": float(np.median(pos)) * direction,
            "median_valid": float(np.median(neg)) * direction,
            **op,
        }
        print(f"{name:18}{a:7.3f}{summary[name]['median_invalid']:10.3f}"
              f"{summary[name]['median_valid']:10.3f}"
              f"{op['threshold'] * direction:8.3f}{op['recall']:8.1%}{op['false_flag']:8.1%}")

    # 두 신호 결합 — OR 규칙(둘 중 하나라도 걸리면 무효 의심)
    ue = np.array([r["unexplained_ink"] for r in rows])
    fa = np.array([r["font_agreement"] for r in rows])
    best_combo = {"youden": -1.0}
    for t_ue in np.quantile(ue, np.linspace(0.5, 0.99, 40)):
        for t_fa in (1.0, 0.67):
            flag = (ue >= t_ue) | (fa < t_fa)
            recall = float(flag[~wf].mean())
            false_flag = float(flag[wf].mean())
            j = recall - false_flag
            if j > best_combo["youden"]:
                best_combo = {
                    "rule": f"unexplained_ink >= {t_ue:.3f} OR font_agreement < {t_fa}",
                    "recall": recall,
                    "false_flag": false_flag,
                    "youden": float(j),
                }
    print(f"\n결합 규칙: {best_combo['rule']}")
    print(f"  무효 검출률 {best_combo['recall']:.1%} / 유효 오탐률 {best_combo['false_flag']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"n": len(rows), "n_invalid": int((~wf).sum()), "signals": summary,
             "combined": best_combo, "rows": rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
