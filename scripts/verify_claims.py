# -*- coding: utf-8 -*-
"""기술노트·데이터셋 카드에 인용된 **모든 수치를 원본 데이터에서 재계산**한다.

공개 릴리스의 재현성 장치다 — 문서에 적힌 숫자와 이 스크립트 출력이
불일치하면 문서가 틀린 것이다. 누구나 `python scripts/verify_claims.py`로
확인할 수 있다.

**이 스크립트가 존재하는 이유(실제 사고).** 2026-08-12에 `rescore_pilot_v3.py`
와 `eval_oss_ocr.py`가 사람 라벨을 로드할 때 (1) Phase 2 감사에서 받은
전사를 버리고, (2) 1차 미감사 44장을 통째로 누락하는 버그가 발견됐다.
그 결과 `docs/PROGRESS.md` 18·19단계에 기록된 수치가 전부 틀렸고, "과소평가
폭이 일정하다"·"OCR이 구조 격차를 왜곡하지 않는다" 같은 결론이 정반대로
뒤집혔다. 수치를 손으로 옮겨 적는 대신 항상 원본에서 재계산해야 한다.

정답(ground truth) 정의 — 우선순위:
  1. Phase 2 감사(2026-08-11, 이진 α=0.942) 라벨이 있으면 그것
  2. 없으면 1차 감사 라벨(text/illegible/extra_text)을 매핑
전사는 두 감사 모두 valid 판정 시에만 존재한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.decompose import decompose  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
B = 5000
PRIMARY_ANNOTATOR = "Nasser Lim"
CODA_ORDER = ("no_T", "simple_T", "cluster_T")

# 1차 감사 라벨 -> Phase 2 스킴. `illegible`은 "키보드로 입력 불가능한 형태"라는
# 감사자 기준이 적용된 것으로, Phase 2에서 재라벨링한 43건은 86%가 malformed
# 였다(13단계). 재라벨링되지 않은 건은 malformed로 보수적 매핑한다 — 어느
# 쪽이든 "유효 완성형 아님"이라는 이진 판정은 동일하다.
PHASE1_MAP = {"text": "valid_syllable", "extra_text": "multi_syllable", "illegible": "malformed"}


def _read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load() -> dict:
    pilot = {r["image_path"]: r for r in _read_jsonl(RESULTS / "pilot" / "pilot_results_v2.jsonl")}
    phase1 = {r["image_path"]: r for r in _read_jsonl(RESULTS / "audit" / "human_audit_pilot.jsonl")}
    phase2 = _read_jsonl(RESULTS / "audit" / "human_audit_pilot_phase2.jsonl")
    a1 = {r["image_path"]: r for r in phase2 if r["annotator"] == PRIMARY_ANNOTATOR}
    a2 = {r["image_path"]: r for r in phase2 if r["annotator"] != PRIMARY_ANNOTATOR}
    easy = {r["path"]: r for r in json.loads((RESULTS / "oss_ocr_eval.json").read_text(encoding="utf-8"))["rows"]}
    padd_path = RESULTS / "oss_ocr_eval_paddleocr.json"
    padd = {}
    if padd_path.is_file():
        padd = {r["path"]: r for r in json.loads(padd_path.read_text(encoding="utf-8"))["rows"]}

    rows = []
    for path, p in pilot.items():
        if path in a1:
            label = a1[path]["label"]
            transcription = a1[path].get("transcription") or None
        else:
            label = PHASE1_MAP[phase1[path]["label"]]
            transcription = phase1[path]["transcription"] if label == "valid_syllable" else None
        rows.append({
            "path": path,
            "target": p["target"],
            "coda": p["final_class"],
            "vowel": p["vowel_class"],
            "onset": p["onset_group"],
            "human_label": label,
            "human_valid": label == "valid_syllable",
            "human_reading": transcription,
            "clova": p["selected_candidate"],
            "easyocr": easy.get(path, {}).get("easyocr_reading"),
            "paddleocr": padd.get(path, {}).get("engine_reading"),
        })
    return {"rows": rows, "a1": a1, "a2": a2}


def boot_mean(flags, rng, b=B):
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(a), (b, len(a)))
    d = a[idx].mean(axis=1)
    return a.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main() -> None:
    rng = np.random.default_rng(0)
    data = load()
    rows = data["rows"]
    n = len(rows)

    truth = lambda r: r["human_valid"] and r["human_reading"] == r["target"]  # noqa: E731
    engines = {
        "CLOVA": lambda r: r["clova"] == r["target"],
        "EasyOCR": lambda r: r["easyocr"] == r["target"],
        "PaddleOCR": lambda r: r["paddleocr"] == r["target"],
    }
    has_paddle = any(r["paddleocr"] is not None for r in rows)
    if not has_paddle:
        engines.pop("PaddleOCR")

    print("=" * 68)
    print(f"[C1] 파일럿 사람 라벨 분포 (n={n})")
    counts: dict = {}
    for r in rows:
        counts[r["human_label"]] = counts.get(r["human_label"], 0) + 1
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"      {k:16}{counts[k]:4}  {counts[k]/n:6.1%}")
    n_invalid = sum(1 for r in rows if not r["human_valid"])
    print(f"      → 유효 완성형 아님: {n_invalid}/{n} = {n_invalid/n:.1%}")

    print()
    print("[C2] 감사자 간 일치도 (Phase 2 공통 문항)")
    common = [p for p in data["a1"] if p in data["a2"]]
    b1 = [data["a1"][p]["label"] == "valid_syllable" for p in common]
    b2 = [data["a2"][p]["label"] == "valid_syllable" for p in common]
    agree = sum(1 for x, y in zip(b1, b2) if x == y)
    po = agree / len(common)
    pe = sum((np.mean([b1, b2], axis=0) == v).mean() for v in [0]) * 0  # placeholder, computed below
    # Krippendorff alpha (nominal, 2 coders, no missing)
    pairs = list(zip(b1, b2))
    n_units = len(pairs)
    do = sum(1 for x, y in pairs if x != y) / n_units
    marg = {}
    for x, y in pairs:
        marg[x] = marg.get(x, 0) + 1
        marg[y] = marg.get(y, 0) + 1
    total = 2 * n_units
    de = 1 - sum((v / total) ** 2 for v in marg.values())
    alpha = 1 - do / de if de > 0 else float("nan")
    print(f"      이진(유효/무효)  raw 일치율 {agree}/{len(common)} = {po:.1%}   Krippendorff α = {alpha:.3f}")
    both_valid = [p for p in common if data["a1"][p]["label"] == "valid_syllable" and data["a2"][p]["label"] == "valid_syllable"]
    same_tr = sum(1 for p in both_valid if (data["a1"][p].get("transcription") or "") == (data["a2"][p].get("transcription") or ""))
    print(f"      둘 다 유효 판정한 {len(both_valid)}건의 전사 일치: {same_tr}/{len(both_valid)}")

    print()
    print("[C3] 은폐율 — 사람이 '유효 완성형 아님'이라 한 건에서 엔진이 유효 음절을 답한 비율")
    invalid_rows = [r for r in rows if not r["human_valid"]]
    print(f"      (분모 n={len(invalid_rows)})")
    for name in ["CLOVA", "EasyOCR", "PaddleOCR"]:
        if name == "PaddleOCR" and not has_paddle:
            continue
        key = {"CLOVA": "clova", "EasyOCR": "easyocr", "PaddleOCR": "paddleocr"}[name]
        answered = [r for r in invalid_rows if r[key] and len(r[key]) == 1]
        fp = [r for r in invalid_rows if r[key] == r["target"]]
        print(f"      {name:10} 은폐 {len(answered)}/{len(invalid_rows)} = {len(answered)/len(invalid_rows):6.1%}"
              f"   위양성(타깃과 일치) {len(fp)}")

    both = [r for r in invalid_rows if r["clova"] and len(r["clova"]) == 1 and r["easyocr"]]
    same = sum(1 for r in both if r["clova"] == r["easyocr"])
    print(f"      CLOVA·EasyOCR 둘 다 답한 {len(both)}건 중 서로 다른 글자로 스냅: "
          f"{len(both)-same}/{len(both)} = {(len(both)-same)/len(both):.1%}")

    print()
    print("[C4] 유효 완성형 생성률 (사람 기준) — 종성유형별")
    for coda in CODA_ORDER:
        sub = [r for r in rows if r["coda"] == coda]
        m, lo, hi = boot_mean([r["human_valid"] for r in sub], rng)
        print(f"      {coda:10} n={len(sub):3}  {m:6.1%} [{lo:.1%},{hi:.1%}]")
    a = np.array([r["human_valid"] for r in rows if r["coda"] == "cluster_T"], float)
    b = np.array([r["human_valid"] for r in rows if r["coda"] == "no_T"], float)
    ia, ib = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    verdict = "유의" if (lo > 0 or hi < 0) else "유의하지 않음"
    print(f"      cluster_T − no_T: {a.mean()-b.mean():+.1%} CI[{lo:+.1%},{hi:+.1%}]  → {verdict}")

    print()
    print("[C5] 실패 구성비 (사람 기준)")
    print(f"      {'종성':10}{'n':>5}{'정답':>9}{'유효-오답':>11}{'무효':>9}")
    for coda in CODA_ORDER:
        sub = [r for r in rows if r["coda"] == coda]
        ok = sum(1 for r in sub if truth(r))
        wrong = sum(1 for r in sub if r["human_valid"] and not truth(r))
        inv = sum(1 for r in sub if not r["human_valid"])
        print(f"      {coda:10}{len(sub):5}{ok/len(sub):9.1%}{wrong/len(sub):11.1%}{inv/len(sub):9.1%}")

    print()
    print("[C6] 타깃 일치율: 사람 vs OCR (unconditional)")
    header = f"      {'종성':10}{'진실':>9}" + "".join(f"{k:>11}" for k in engines)
    print(header)
    for coda in CODA_ORDER:
        sub = [r for r in rows if r["coda"] == coda]
        line = f"      {coda:10}{np.mean([truth(r) for r in sub]):9.1%}"
        line += "".join(f"{np.mean([f(r) for r in sub]):11.1%}" for f in engines.values())
        print(line)
    line = f"      {'전체':10}{np.mean([truth(r) for r in rows]):9.1%}"
    line += "".join(f"{np.mean([f(r) for r in rows]):11.1%}" for f in engines.values())
    print(line)

    print()
    print("[C7] 과소평가 폭 (진실 − OCR), 쌍대응 bootstrap")
    for name, f in engines.items():
        for coda in CODA_ORDER:
            sub = [r for r in rows if r["coda"] == coda]
            a = np.array([truth(r) for r in sub], float)
            b = np.array([f(r) for r in sub], float)
            i = rng.integers(0, len(sub), (B, len(sub)))
            d = (a[i] - b[i]).mean(axis=1)
            print(f"      {name:10}{coda:11} {(a-b).mean():+7.1%} CI[{np.percentile(d,2.5):+.1%},{np.percentile(d,97.5):+.1%}]")

    print()
    print("[C8] 구조 격차 왜곡 (겹받침 − 단순종성)")
    cl = [r for r in rows if r["coda"] == "cluster_T"]
    si = [r for r in rows if r["coda"] == "simple_T"]
    gt = np.mean([truth(r) for r in cl]) - np.mean([truth(r) for r in si])
    print(f"      진실       {gt:+.1%}")
    for name, f in engines.items():
        g = np.mean([f(r) for r in cl]) - np.mean([f(r) for r in si])
        ds = []
        for _ in range(B):
            ic = rng.integers(0, len(cl), len(cl))
            isi = rng.integers(0, len(si), len(si))
            gg = np.mean([f(cl[i]) for i in ic]) - np.mean([f(si[i]) for i in isi])
            tt = np.mean([truth(cl[i]) for i in ic]) - np.mean([truth(si[i]) for i in isi])
            ds.append(gg - tt)
        lo, hi = np.percentile(ds, 2.5), np.percentile(ds, 97.5)
        verdict = "유의" if (lo > 0 or hi < 0) else "유의하지 않음"
        print(f"      {name:10} {g:+.1%}  과장 {g-gt:+.1%}p CI[{lo:+.1%},{hi:+.1%}] {verdict}  배율 {g/gt:.1f}x")

    print()
    print("[C9] 자모 위치별 오류율 (사람 기준)")
    for cond_label, conditional in [("Conditional(유효만)", True), ("Unconditional(무효=3위치 오류)", False)]:
        pool = [r for r in rows if r["human_valid"]] if conditional else rows
        print(f"      --- {cond_label}  n={len(pool)} ---")
        for pos, attr in [("초성", "onset"), ("중성", "nucleus"), ("종성", "coda")]:
            flags = []
            for r in pool:
                if not r["human_valid"] or not r["human_reading"]:
                    flags.append(1.0)
                    continue
                t, p = decompose(r["target"]), decompose(r["human_reading"])
                flags.append(0.0 if (p and getattr(t, attr) == getattr(p, attr)) else 1.0)
            m, lo, hi = boot_mean(flags, rng)
            print(f"      {pos:6} {m:6.1%} [{lo:.1%},{hi:.1%}]")

    print()
    print("[C10] 합성 malformed 실험 (H1) — 생성 모델 미사용, 정답 구성상 보장")
    syn_path = RESULTS / "synthetic_malformed_eval.json"
    if syn_path.is_file():
        from scipy import stats

        syn = json.loads(syn_path.read_text(encoding="utf-8"))["rows"]
        ctl = [r for r in syn if r["is_control"]]
        mal = [r for r in syn if not r["is_control"]]
        ca = [bool(r.get("easyocr_reading")) for r in ctl]
        ma = [bool(r.get("easyocr_reading")) for r in mal]
        pc, pm = np.mean(ca), np.mean(ma)
        pp = (sum(ca) + sum(ma)) / (len(ca) + len(ma))
        se = np.sqrt(pp * (1 - pp) * (1 / len(ca) + 1 / len(ma)))
        z = (pm - pc) / se
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        lo, hi = (pm - pc) - 1.96 * se, (pm - pc) + 1.96 * se
        margin = 0.10
        eq = "통과" if (lo > -margin and hi < margin) else "미통과"
        print(f"      EasyOCR 답변율: 대조군 {pc:.1%}(n={len(ca)})  실험군 {pm:.1%}(n={len(ma)})")
        print(f"        차이 {pm-pc:+.1%}  z={z:.2f}  p={pval:.3f}   동등성검정(±{margin:.0%}) {eq}  CI[{lo:+.1%},{hi:+.1%}]")
        cc = [r.get("easyocr_confidence") or 0.0 for r in ctl]
        mc = [r.get("easyocr_confidence") or 0.0 for r in mal]
        _, pmw = stats.mannwhitneyu(cc, mc, alternative="two-sided")
        print(f"      EasyOCR confidence: 대조군 {np.mean(cc):.3f}  실험군 {np.mean(mc):.3f}  Mann-Whitney p={pmw:.3f}")
        tc = [r.get("tm_matches_original") for r in ctl]
        tm_ = [r.get("tm_matches_original") for r in mal]
        if all(v is not None for v in tc + tm_):
            print(f"      template_match: 대조군 정확도 {np.mean(tc):.1%}  실험군 원본스냅 {np.mean(tm_):.1%}")
    else:
        print("      (results/synthetic_malformed_eval.json 없음 — scripts/eval_synthetic_malformed.py 먼저 실행)")

    print()
    print("=" * 68)
    print("모든 수치는 results/ 원본에서 재계산됐다. 문서와 불일치하면 문서가 틀린 것이다.")


if __name__ == "__main__":
    main()
