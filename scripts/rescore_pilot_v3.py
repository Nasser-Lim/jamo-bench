# -*- coding: utf-8 -*-
"""파일럿 300장 v3 재채점 — 확정된 `judging_protocol`로 첫 유효 채점본을 만든다.

**이건 새 실험이 아니라 마무리 작업이다.** 이미지는 4단계(파일럿 배치)에서
다 생성됐고, 이후 v1(Route 버그, 4~5단계) → v2(occupancy 버그, 5·8단계) →
hybrid_judge(15단계에서 confidence 게이트로 대체) 순으로 채점 시도가
있었지만 전부 무효이거나 상위 버전으로 대체됐다. `judging_protocol.py`가
16단계에서 18셀 세 축(종성·모음·초성군) 전부 편향 검증을 통과한 뒤에야
"인용해도 되는" 채점을 만들 수 있게 됐다.

**규칙 (judging_protocol과 동일, 그대로 적용만 한다):**

    confidence >= 0.80 AND 초성군 != tense_O  →  CLOVA 판독 채택 (label_tier=silver)
    그 외                                      →  사람 판정 채택 (label_tier=gold)

confidence는 15단계에서 측정한 256장 부분집합(Route B ∩ jamo_valid==VALID)
에만 있다 — 나머지 44장(Route A1/C, OVERGEN)은 confidence가 없으므로
`judging_protocol.route()`가 항상 사람에게 넘긴다(설계상 자연스러운 동작).

사람이 "유효 완성형 아님"이라 판정한 경우 `score()`에 `pred=None`을 넘겨
EMPTY verdict로 기록한다 — score.py 원래 의미(OCR이 후보를 못 찾음)와
정확히 같지는 않지만, "채점 가능한 완성형이 없다"는 사실은 동일하므로
재사용한다. `human_valid=False`가 근본 원인이라는 것은 `label_tier`/
`route_source` 필드로 항상 함께 기록되므로 혼동되지 않는다.

API 호출 0건, Seedream 재생성 0건 — 전부 저장된 결과 재사용.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.judging_protocol import (  # noqa: E402
    CALIBRATION,
    CONFIDENCE_THRESHOLD,
    MEASURED_BIAS_PP,
    resolve_human,
    route,
)
from jamo_bench.score import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_ROWS = ROOT / "results" / "pilot" / "pilot_results_v3.jsonl"
OUT_SUMMARY = ROOT / "results" / "pilot" / "pilot_v3_summary.json"


def load_confidence_by_path() -> dict:
    """CLOVA occ0.10 confidence — 15단계에서 측정한 256장(Route B ∩
    jamo_valid==VALID) 부분집합. 리스트 순서로만 식별되므로 같은 필터로
    pilot_results_v2를 재구성해 순서를 맞춘다(검증된 전제, 15단계)."""
    conf = json.loads((ROOT / "results" / "clova_confidence_eval.json").read_text(encoding="utf-8"))
    pilot = [
        json.loads(line)
        for line in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sub = [r for r in pilot if r["route"] == "B" and r["jamo_valid"] == "VALID"]
    assert len(sub) == len(conf), f"순서 전제 붕괴: pilot {len(sub)} vs conf {len(conf)}"
    out = {}
    for pilot_rec, conf_rec in zip(sub, conf):
        assert conf_rec["target"] == pilot_rec["target"]
        out[pilot_rec["image_path"]] = {"reading": conf_rec["pred"], "confidence": conf_rec["confidence"]}
    return out


def load_human_labels() -> dict:
    """이미지 경로 -> {"valid": bool, "transcription": str|None}.
    Phase 2 재라벨링(이진, α=0.942)이 있으면 그걸 쓰고 없으면 1차 라벨로
    폴백 — judging_protocol이 실전에서 쓰는 것과 같은 우선순위."""
    phase1 = {
        r["image_path"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / "results" / "audit" / "human_audit_pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    phase2 = [
        json.loads(line)
        for line in (ROOT / "results" / "audit" / "human_audit_pilot_phase2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relabel = {r["image_path"]: r for r in phase2 if r["annotator"] == "Nasser Lim"}

    # phase1 ∪ phase2 전체를 순회한다. phase1만 돌면 **1차 미감사 44장**
    # (Route A1/C·OVERGEN — 13단계에서 Phase 2로 처음 감사한 집단)이 통째로
    # 빠져 호출부에서 전부 무효로 처리된다. 그중 23장은 사람이 유효 완성형
    # 으로 판정하고 전사까지 남긴 것이라, 빠뜨리면 유효 완성형 생성률이
    # 크게 과소평가된다.
    out = {}
    for path in set(phase1) | set(relabel):
        if path in relabel:
            valid = relabel[path]["label"] == "valid_syllable"
            # Phase 2 UI도 valid_syllable일 때는 전사를 받는다
            # (audit_queue.PHASE2_LABELS의 needs_transcription=True).
            transcription = relabel[path].get("transcription") or None
        else:
            rec = phase1[path]
            valid = rec["label"] == "text"
            transcription = rec["transcription"] if valid else None
        out[path] = {"valid": valid, "transcription": transcription}
    return out


def main() -> None:
    pilot = [
        json.loads(line)
        for line in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    confidence_by_path = load_confidence_by_path()
    human_labels = load_human_labels()

    rows = []
    for r in pilot:
        path = r["image_path"]
        clova = confidence_by_path.get(path, {"reading": None, "confidence": None})
        decision = route(
            clova_reading=clova["reading"],
            confidence=clova["confidence"],
            coda_class_3=r["final_class"],
            onset_group=r["onset_group"],
        )
        if decision.needs_human:
            human = human_labels.get(path, {"valid": False, "transcription": None})
            decision = resolve_human(decision, human["valid"], human["transcription"])

        result = score(r["target"], decision.reading)
        rows.append({
            "image_path": path,
            "target": r["target"],
            "vowel_class": r["vowel_class"],
            "final_class": r["final_class"],
            "onset_group": r["onset_group"],
            "reading": decision.reading,
            "label_tier": "silver" if decision.source == "clova" else "gold",
            "route_source": decision.source,
            "expected_bias_pp": decision.expected_bias_pp,
            "verdict": result.verdict,
            "onset_ok": result.onset_ok,
            "nucleus_ok": result.nucleus_ok,
            "coda_ok": result.coda_ok,
        })

    n = len(rows)
    n_gold = sum(1 for r in rows if r["label_tier"] == "gold")
    n_valid = sum(1 for r in rows if r["verdict"] == "VALID")
    n_target_match = sum(1 for r in rows if r["verdict"] == "VALID" and r["onset_ok"] and r["nucleus_ok"] and r["coda_ok"])

    def well_formed_rate_by(axis: str) -> dict:
        """Primary 1(유효 완성형 생성률)을 축별로. 분모는 전체(무효 포함)."""
        out = {}
        for v in sorted({r[axis] for r in rows}):
            sub = [r for r in rows if r[axis] == v]
            n_wf = sum(1 for r in sub if r["verdict"] == "VALID")
            out[v] = {"n": len(sub), "well_formed_rate": n_wf / len(sub)}
        return out

    def target_match_rate_by(axis: str) -> dict:
        """Primary 2(타깃 일치율)을 축별로. 분모는 유효 완성형만(conditional)
        — Primary 1을 통과한 것 중에서 타깃과 일치하는 비율."""
        out = {}
        for v in sorted({r[axis] for r in rows}):
            sub = [r for r in rows if r[axis] == v and r["verdict"] == "VALID"]
            if not sub:
                continue
            exact = sum(1 for r in sub if r["onset_ok"] and r["nucleus_ok"] and r["coda_ok"])
            out[v] = {"n": len(sub), "exact_match_rate": exact / len(sub)}
        return out

    def jamo_error_rate(conditional: bool) -> dict:
        pool = [r for r in rows if r["verdict"] == "VALID"] if conditional else rows
        n_pool = len(pool)
        errs = {"onset": 0, "nucleus": 0, "coda": 0}
        for r in pool:
            if r["verdict"] != "VALID":
                errs["onset"] += 1
                errs["nucleus"] += 1
                errs["coda"] += 1
                continue
            errs["onset"] += not r["onset_ok"]
            errs["nucleus"] += not r["nucleus_ok"]
            errs["coda"] += not r["coda_ok"]
        return {k: v / n_pool for k, v in errs.items()}

    summary = {
        "n": n,
        "n_gold": n_gold,
        "n_silver": n - n_gold,
        "gold_frac": n_gold / n,
        "well_formed_rate": n_valid / n,
        "target_match_rate_unconditional": n_target_match / n,
        "target_match_rate_conditional": n_target_match / n_valid if n_valid else 0.0,
        "well_formed_rate_by_coda": well_formed_rate_by("final_class"),
        "target_match_rate_by_coda": target_match_rate_by("final_class"),
        "jamo_error_rate_conditional": jamo_error_rate(conditional=True),
        "jamo_error_rate_unconditional": jamo_error_rate(conditional=False),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "measured_bias_pp": MEASURED_BIAS_PP,
        "calibration": CALIBRATION,
    }

    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ROWS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"n={n}  Gold(사람) {n_gold}건({n_gold/n:.1%})  Silver(CLOVA) {n-n_gold}건({(n-n_gold)/n:.1%})")
    print(f"\n유효 완성형 생성률(Primary 1): {n_valid/n:.1%}")
    print(f"타깃 일치율 | 유효 완성형(Primary 2, conditional): {n_target_match/n_valid:.1%}" if n_valid else "")
    print(f"타깃 일치율(unconditional): {n_target_match/n:.1%}")
    print(f"\n[반드시 병기] 자동 판정 편향: {MEASURED_BIAS_PP}")
    print(f"\n종성유형별 유효 완성형 생성률(Primary 1):")
    for k, v in summary["well_formed_rate_by_coda"].items():
        print(f"  {k:10} n={v['n']:3}  {v['well_formed_rate']:.1%}")
    print(f"\n종성유형별 타깃 일치율(유효 완성형 중):")
    for k, v in summary["target_match_rate_by_coda"].items():
        print(f"  {k:10} n={v['n']:3}  {v['exact_match_rate']:.1%}")
    print(f"\n자모 위치별 오류율 — Conditional (유효만, 재현 못함=제외):")
    for k, v in summary["jamo_error_rate_conditional"].items():
        print(f"  {k:8} {v:.1%}")
    print(f"자모 위치별 오류율 — Unconditional (무효는 3위치 전부 오류):")
    for k, v in summary["jamo_error_rate_unconditional"].items():
        print(f"  {k:8} {v:.1%}")

    print(f"\n저장: {OUT_ROWS.relative_to(ROOT)}")
    print(f"저장: {OUT_SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
