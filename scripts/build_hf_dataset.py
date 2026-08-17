# -*- coding: utf-8 -*-
"""HuggingFace Dataset Viewer용 parquet 빌드 — `release/data/`.

**왜 필요한가.** `release/`의 원본 산출물은 전부 jsonl/json이라 HF 데이터셋
페이지에 자동 미리보기(Dataset Viewer)가 뜨지 않는다. `configs:`가 가리키는
parquet을 만들어두면 표(또는 이미지) 미리보기가 자동 생성되어 검색 노출·
클릭률에 도움이 된다.

**gold_pilot (이미지 없음, 300행).** 이미지 원본은 배포하지 않으므로(약관
미확인, RELEASE.md 참고) 이 config는 표만 있고 썸네일은 없다. 이미 검증된
release 산출물들을 `image_id`로 조인한다 — 원본 오디팅 로그(사람 라벨 간
불일치 해소 로직 등)를 다시 해석하지 않고, 이미 계산이 끝난 결과 파일만
합치므로 조인 과정에서 새로운 판단이 끼어들 여지가 없다:
  - `judge_outputs/image_manifest.jsonl` — 타깃·프롬프트·18셀 메타데이터
  - `results/pilot/pilot_results_v2.jsonl`(비공개 원본) — CLOVA 판독
    (image_path를 `image_id`로 해시하는 방식은 build_release.py와 동일해야
    조인이 맞는다 — 아래 `image_id()`를 그대로 복사해 쓴다)
  - `judge_outputs/oss_ocr_eval.json` — 사람 라벨(human_valid/human_reading)
    + EasyOCR 판독
  - `judge_outputs/oss_ocr_eval_paddleocr.json` — PaddleOCR 판독

**synthetic_malformed (이미지 포함, 581행).** 원본이 SIL OFL 폰트 렌더링
+ 프로그램적 변형이라 재배포에 아무 제약이 없다. 이미지 파일 자체가
저장돼 있지 않으므로 `jamo_bench.synthetic_malformed.build_dataset()`을
**생성 당시와 동일한 파라미터(seed=0 포함)로 재실행**해 결정론적으로
재현한다 — `scripts/eval_synthetic_malformed.py`의 `stratified_syllables`
+ `build_dataset` 호출 순서를 그대로 복제했다. 재현된 아이템과
`results/synthetic_malformed_eval.json`의 저장된 행을
(char, font_name, kind, severity)로 매 인덱스마다 대조해 순서가 어긋나면
즉시 assert로 중단한다 — 조용히 잘못된 이미지-메타데이터 쌍을 만들지
않기 위함.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import datasets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"
DATA = RELEASE / "data"


def image_id(image_path: str) -> str:
    """build_release.py의 image_id()와 완전히 동일해야 조인 키가 맞는다."""
    norm = image_path.replace("\\", "/")
    return "img_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_gold_pilot() -> None:
    print("=== gold_pilot (이미지 없음) ===")
    manifest = {r["image_id"]: r for r in load_jsonl(RELEASE / "judge_outputs" / "image_manifest.jsonl")}

    clova = {}
    for r in load_jsonl(ROOT / "results" / "pilot" / "pilot_results_v2.jsonl"):
        clova[image_id(r["image_path"])] = r["selected_candidate"]

    easy = {r["image_id"]: r for r in json.loads((RELEASE / "judge_outputs" / "oss_ocr_eval.json").read_text(encoding="utf-8"))["rows"]}
    paddle = {r["image_id"]: r for r in json.loads((RELEASE / "judge_outputs" / "oss_ocr_eval_paddleocr.json").read_text(encoding="utf-8"))["rows"]}

    missing = [iid for iid in manifest if iid not in clova or iid not in easy or iid not in paddle]
    if missing:
        raise SystemExit(f"조인 실패 — 다음 image_id가 일부 소스에 없음: {missing[:5]} 외 {len(missing)}건")

    rows = []
    for iid, m in manifest.items():
        e, p = easy[iid], paddle[iid]
        rows.append({
            "image_id": iid,
            "target": m["target"],
            "template_id": m["template_id"],
            "n_sample_idx": m["n_sample_idx"],
            "prompt": m["prompt"],
            "vowel_class": m["vowel_class"],
            "coda_class": m["final_class"],
            "onset_group": m["onset_group"],
            "human_valid": e["human_valid"],
            "human_reading": e["human_reading"],
            "clova_reading": clova[iid],
            "easyocr_reading": e["easyocr_reading"],
            "easyocr_confidence": e["easyocr_confidence"],
            "paddleocr_reading": p["engine_reading"],
            "paddleocr_confidence": p["engine_confidence"],
        })

    ds = datasets.Dataset.from_list(rows)
    out = DATA / "gold_pilot"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out / "gold_pilot-00000-of-00001.parquet"))
    print(f"  {len(rows)}행 -> {out.relative_to(ROOT)}/")


def _stratified_syllables(n_per_coda: int, seed: int) -> list[str]:
    """scripts/eval_synthetic_malformed.py의 stratified_syllables()와 동일 —
    표본을 재현하려면 이 함수도 완전히 똑같아야 한다."""
    import random

    from jamo_bench.decompose import all_syllables

    rng = random.Random(seed)
    by_coda: dict = {}
    for s in all_syllables():
        by_coda.setdefault(s.coda_class_3, []).append(s.char)
    out = []
    for coda in ("no_T", "simple_T", "cluster_T"):
        pool = by_coda[coda]
        out.extend(rng.sample(pool, min(n_per_coda, len(pool))))
    return out


def build_synthetic_malformed() -> None:
    print("\n=== synthetic_malformed (이미지 포함) ===")
    from jamo_bench.synthetic_malformed import build_dataset

    eval_data = json.loads((ROOT / "results" / "synthetic_malformed_eval.json").read_text(encoding="utf-8"))
    cfg = eval_data["config"]
    syllables = _stratified_syllables(cfg["n_per_coda"], cfg["seed"])
    items = build_dataset(syllables, fonts=tuple(cfg["fonts"]), severities=tuple(cfg["severities"]), seed=cfg["seed"])
    items = [it for it in items if it.verified_novel]

    rows_meta = eval_data["rows"]
    if len(items) != len(rows_meta):
        raise SystemExit(f"재현된 아이템 수({len(items)})가 저장된 행 수({len(rows_meta)})와 다름 — 재현 실패")

    records = []
    for it, row in zip(items, rows_meta):
        if (it.char, it.font_name, it.kind, it.severity) != (row["char"], row["font_name"], row["kind"], row["severity"]):
            raise SystemExit(f"순서 불일치 — 재현={it.char, it.font_name, it.kind, it.severity} 저장={row['char'], row['font_name'], row['kind'], row['severity']}")
        records.append({
            "image": it.image,
            "char": it.char,
            "font_name": it.font_name,
            "kind": it.kind,
            "severity": it.severity,
            "is_control": row["is_control"],
            "tm_reading": row.get("tm_reading"),
            "tm_score": row.get("tm_score"),
            "tm_matches_original": row.get("tm_matches_original"),
            "easyocr_reading": row.get("easyocr_reading"),
            "easyocr_confidence": row.get("easyocr_confidence"),
            "easyocr_matches_original": row.get("easyocr_matches_original"),
        })

    ds = datasets.Dataset.from_list(records)
    ds = ds.cast_column("image", datasets.Image())
    out = DATA / "synthetic_malformed"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out / "synthetic_malformed-00000-of-00001.parquet"))
    print(f"  {len(records)}행 (이미지 포함) -> {out.relative_to(ROOT)}/")


def main() -> None:
    if not (RELEASE / "judge_outputs" / "image_manifest.jsonl").is_file():
        raise SystemExit("release/가 없음 — 먼저 scripts/build_release.py를 실행하세요")
    build_gold_pilot()
    build_synthetic_malformed()
    total = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    print(f"\nparquet 크기: {total/1024/1024:.1f} MB -> {DATA.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
