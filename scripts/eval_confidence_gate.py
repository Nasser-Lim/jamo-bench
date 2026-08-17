# -*- coding: utf-8 -*-
"""confidence 게이트 임계값 스윕 — 판정 프로토콜의 유일한 자유 파라미터를 정한다.

**프로토콜.** CLOVA `inferConfidence`가 임계값 이상이면 CLOVA 판독을 그대로
채택하고, 그 외는 사람이 판정한다(이진 유효성 + 유효하면 전사).

**왜 이게 작동하는가(반직관적).** 11단계에서 confidence를 "한글 여부" 신호로
쓰려다 실패했고, 12단계에서는 "정답 여부와 상관된 신호로 abstain하면 자동
판정 구간이 정답 쪽으로 선별되어 정확도가 부풀려진다"고 경고했다. 그런데
실측하면 반대다 — **CLOVA는 자기가 못 믿을 곳에서 정확히 자신감을 잃는다.**
자동 채택률이 종성 복잡도를 따라 무너지고(no_T 75.9% → cluster_T 18.8%),
어려운 구간이 통째로 사람에게 넘어가므로 편향이 커지는 게 아니라 줄어든다.

임계값은 **사람 큐 비율 ↔ 편향** 트레이드오프다. 이 스크립트는 그 곡선을
그려 운용점을 고른다. 판정 기준:

  1. 축별 편향 변동폭(max-min)이 작을 것 — 6단계에서 세운 판정자 자격 기준
     ("오차가 측정 축과 상관되지 않을 것")
  2. 겹받침−단순종성 격차를 사람 기준 대비 ±5%p 안에서 복원할 것
  3. 사람 큐 비율이 감당 가능할 것

API 호출 0건 — 기존 `results/clova_confidence_eval.json`과 사람 라벨 재사용.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "confidence_gate_sweep.json"
CODA_ORDER = ["no_T", "simple_T", "cluster_T"]


def load_rows() -> list:
    """confidence 평가 결과에 사람 라벨을 붙인다.

    `clova_confidence_eval.json`은 이미지 경로를 안 들고 있고 리스트 순서로만
    식별된다. 그 순서는 `pilot_results_v2.jsonl`의 `route == B and
    jamo_valid == VALID` 부분집합 순서와 일치한다(검증됨) — 이 전제가 깨지면
    아래 assert가 잡는다.
    """
    conf = json.loads((ROOT / "results" / "clova_confidence_eval.json").read_text(encoding="utf-8"))
    pilot = [
        json.loads(line)
        for line in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sub = [r for r in pilot if r["route"] == "B" and r["jamo_valid"] == "VALID"]
    assert len(sub) == len(conf), f"순서 전제 붕괴: pilot {len(sub)} vs conf {len(conf)}"

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

    rows = []
    for pilot_rec, conf_rec in zip(sub, conf):
        path = pilot_rec["image_path"]
        assert conf_rec["target"] == pilot_rec["target"], f"순서 전제 붕괴: {path}"
        if path in relabel:
            is_valid = relabel[path]["label"] == "valid_syllable"
        else:
            is_valid = phase1[path]["label"] == "text"
        rows.append({
            "target": conf_rec["target"],
            "clova": conf_rec["pred"],
            "confidence": conf_rec["confidence"],
            "human_transcription": phase1[path]["transcription"],
            "human_valid": is_valid,
            "coda": pilot_rec["final_class"],
        })
    return rows


def simulate(rows: list, threshold: float) -> dict:
    """게이트를 적용했을 때의 정확도. confidence가 없으면(CLOVA 검출 실패)
    사람에게 넘긴다."""
    correct = 0
    human_queue = 0
    for r in rows:
        auto = r["confidence"] is not None and r["confidence"] >= threshold
        if auto:
            pred = r["clova"]
        else:
            human_queue += 1
            # 사람은 무효 판정 시 전사하지 않는다 — 그 경우 어떤 타깃과도 불일치
            pred = r["human_transcription"] if r["human_valid"] else None
        correct += pred == r["target"]
    return {"accuracy": correct / len(rows), "human_queue_frac": human_queue / len(rows)}


def truth(rows: list) -> float:
    return sum(1 for r in rows if r["human_valid"] and r["human_transcription"] == r["target"]) / len(rows)


def main() -> None:
    rows = load_rows()
    n_no_conf = sum(1 for r in rows if r["confidence"] is None)
    print(f"n={len(rows)} (confidence 없음 {n_no_conf}건은 항상 사람 큐로)")

    by_coda = {c: [r for r in rows if r["coda"] == c] for c in CODA_ORDER}
    truth_all = truth(rows)
    truth_coda = {c: truth(v) for c, v in by_coda.items()}
    gap_truth = truth_coda["cluster_T"] - truth_coda["simple_T"]
    print(f"사람(진실) 전체 {truth_all:.1%} | "
          + " ".join(f"{c} {truth_coda[c]:.1%}" for c in CODA_ORDER))
    print(f"사람 기준 겹받침−단순종성 격차 {gap_truth:+.1%}\n")

    header = (f"{'임계값':>8}{'사람큐':>9}{'전체정확도':>12}{'편향':>9}"
              f"{'편향변동폭':>12}{'격차복원오차':>14}")
    print(header)
    print("-" * 66)

    sweep = []
    for t in [0.0, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.01]:
        overall = simulate(rows, t)
        biases = {c: simulate(v, t)["accuracy"] - truth_coda[c] for c, v in by_coda.items()}
        spread = max(biases.values()) - min(biases.values())
        gap_sim = simulate(by_coda["cluster_T"], t)["accuracy"] - simulate(by_coda["simple_T"], t)["accuracy"]
        entry = {
            "threshold": t,
            "human_queue_frac": overall["human_queue_frac"],
            "accuracy": overall["accuracy"],
            "bias": overall["accuracy"] - truth_all,
            "bias_by_coda": biases,
            "bias_spread": spread,
            "gap_recovery_error": gap_sim - gap_truth,
        }
        sweep.append(entry)
        label = f"{t:.2f}" if t <= 1.0 else "사람전수"
        print(f"{label:>8}{overall['human_queue_frac']:9.1%}{overall['accuracy']:12.1%}"
              f"{entry['bias']:+9.1%}{spread:12.1%}{entry['gap_recovery_error']:+14.1%}")

    # 판정 기준: 제약을 만족하는 것 중 **사람 큐가 가장 작은** 임계값.
    # 편향 변동폭만 최소화하면 "전수 사람 판정"(변동폭 0)으로 수렴해 자동화의
    # 목적 자체가 사라진다 — 자동화는 사람 부담을 줄이려고 하는 것이므로
    # 품질은 제약(constraint)으로 걸고 비용을 최소화한다.
    MAX_GAP_ERROR = 0.05   # 겹받침 효과를 사람 기준 대비 ±5%p 안에서 복원
    MAX_BIAS_SPREAD = 0.025  # 축별 편향 변동폭 — 10단계에서 hybrid_judge에 적용한 기준과 동일
    eligible = [
        e for e in sweep
        if e["threshold"] <= 1.0
        and abs(e["gap_recovery_error"]) <= MAX_GAP_ERROR
        and e["bias_spread"] <= MAX_BIAS_SPREAD
    ]
    best = min(eligible, key=lambda e: e["human_queue_frac"])
    print(f"\n권장 임계값: {best['threshold']:.2f}")
    print(f"  사람 큐 {best['human_queue_frac']:.1%} / 전체 편향 {best['bias']:+.1%} "
          f"/ 편향 변동폭 {best['bias_spread']:.1%} / 격차 복원 오차 {best['gap_recovery_error']:+.1%}p")
    print("  축별 편향: " + " ".join(f"{c} {best['bias_by_coda'][c]:+.1%}" for c in CODA_ORDER))

    OUT.write_text(
        json.dumps({"n": len(rows), "truth_overall": truth_all, "truth_by_coda": truth_coda,
                    "gap_truth": gap_truth, "recommended": best, "sweep": sweep},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
