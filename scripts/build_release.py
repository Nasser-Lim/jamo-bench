# -*- coding: utf-8 -*-
"""공개용 릴리스 번들 생성 — TechRxiv / HuggingFace 업로드 대상.

**무엇을 공개하고 무엇을 빼는가.**

  공개  사람 라벨(익명화) · 판정자 판독 결과 · 집계 통계 · 코드
  제외  생성 이미지 원본  — 모델 약관 확인 전(설계서 §10.2)이므로
                            "미공개 기본값"을 따른다. 대신 이미지
                            해시와 프롬프트를 공개해 재생성 경로를 남긴다
  제외  API 키(.env) · 템플릿 캐시(295MB, 코드로 재생성 가능)

**익명화.** 감사자 실명을 `annotator_1` / `annotator_2`로 치환한다.
`annotator_1`은 저자 본인이며 이 사실은 데이터셋 카드에 명시한다(자기
라벨링은 한계로 보고해야 할 사항이지 숨길 사항이 아니다).

**이미지 경로.** 원본 경로에 타깃 음절이 파일명으로 들어 있어(`퐋_T1_0_*.jpeg`)
그대로 두면 target-blind 재현이 불가능하다. 이미지 해시 기반 ID로 치환한다.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"

ANNOTATOR_MAP = {"Nasser Lim": "annotator_1", "2번감사자": "annotator_2"}


def image_id(image_path: str) -> str:
    """타깃 음절이 노출되지 않는 안정적 ID. 경로 문자열의 해시를 쓴다
    (이미지 바이트가 아니라 경로 — 이미지를 공개하지 않으므로 경로만으로
    충분하고, 원본 리포지토리와 대조 가능해야 하므로 결정론적이어야 한다)."""
    norm = image_path.replace("\\", "/")
    return "img_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def sanitize(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k == "annotator":
            out[k] = ANNOTATOR_MAP.get(v, v)
        elif k in ("image_path", "path"):
            out["image_id"] = image_id(v)
        elif k == "id":
            # phase2의 id는 "group::경로" 형태 — 경로 부분만 치환
            if isinstance(v, str) and "::" in v:
                group, path = v.split("::", 1)
                out[k] = f"{group}::{image_id(path)}"
            elif isinstance(v, str) and ("\\" in v or "/" in v):
                out[k] = image_id(v)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def copy_jsonl(src: Path, dst: Path) -> int:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize(r), ensure_ascii=False) + "\n")
    return len(rows)


def copy_json(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data["rows"] = [sanitize(r) for r in data["rows"]]
    elif isinstance(data, list):
        data = [sanitize(r) if isinstance(r, dict) else r for r in data]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir()

    print("=== 사람 라벨 (익명화) ===")
    for name in ("human_audit.jsonl", "human_audit_pilot.jsonl", "human_audit_pilot_phase2.jsonl"):
        n = copy_jsonl(ROOT / "results" / "audit" / name, RELEASE / "human_labels" / name)
        print(f"  {name:36} {n}행")

    print("\n=== 판정자 판독 · 집계 결과 ===")
    for rel in (
        "pilot/pilot_results_v3.jsonl",
    ):
        n = copy_jsonl(ROOT / "results" / rel, RELEASE / "judge_outputs" / Path(rel).name)
        print(f"  {Path(rel).name:36} {n}행")

    optional = {"oss_ocr_eval_paddleocr.json", "synthetic_malformed_paddleocr_eval.json"}
    for rel in (
        "synthetic_malformed_eval.json",
        "synthetic_malformed_paddleocr_eval.json",
        "oss_ocr_eval.json",
        "oss_ocr_eval_paddleocr.json",
        "wellformedness_eval.json",
        "confidence_gate_sweep.json",
        "clova_confidence_eval.json",
        "pilot/pilot_v3_summary.json",
    ):
        src = ROOT / "results" / rel
        if not src.is_file():
            # PaddleOCR 결과는 해당 엔진을 돌린 환경에서만 존재한다.
            print(f"  {Path(rel).name:36} 없음 — 건너뜀"
                  f"{'' if Path(rel).name in optional else '  (!! 필수 파일 누락)'}")
            if Path(rel).name not in optional:
                sys.exit(1)
            continue
        copy_json(src, RELEASE / "judge_outputs" / Path(rel).name)
        print(f"  {Path(rel).name:36} 복사됨")

    print("\n=== 이미지 매니페스트 (원본 이미지는 미공개 — 약관 §10.2) ===")
    pilot = [
        json.loads(l)
        for l in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    manifest = []
    for r in pilot:
        manifest.append({
            "image_id": image_id(r["image_path"]),
            "target": r["target"],
            "template_id": r["template_id"],
            "n_sample_idx": r["n_sample_idx"],
            "prompt": r["prompt"],
            "image_hash": r.get("image_hash"),
            "image_width": r.get("image_width"),
            "image_height": r.get("image_height"),
            "vowel_class": r["vowel_class"],
            "final_class": r["final_class"],
            "onset_group": r["onset_group"],
        })
    (RELEASE / "judge_outputs").mkdir(parents=True, exist_ok=True)
    with (RELEASE / "judge_outputs" / "image_manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  image_manifest.jsonl                 {len(manifest)}행 (프롬프트+해시로 재생성 가능)")

    # 데이터셋 카드 + 기술노트 — 소스는 docs/에 두고 번들로 복사한다
    # (release/ 자체는 gitignore 대상이라 여기 직접 쓰면 재빌드 시 소실된다)
    print("\n=== 문서 ===")
    for src_name, dst_name in (
        ("DATASET_CARD.md", "README.md"),          # HF는 README.md를 카드로 렌더링
        ("TECHNICAL_NOTE.md", "TECHNICAL_NOTE.md"),
        ("TECHNICAL_NOTE.ko.md", "TECHNICAL_NOTE.ko.md"),
        ("RELEASE.md", "RELEASE.md"),
    ):
        shutil.copy(ROOT / "docs" / src_name, RELEASE / dst_name)
        print(f"  docs/{src_name:22} -> release/{dst_name}")

    # 노트 본문이 참조하는 그림. fig2는 합성(OFL 파생물)이지만 fig1은
    # 예외 — 사용자 결정으로 실제 생성물 2장의 크롭을 쓴다(docs/RELEASE.md
    # "그림" 절의 정책 예외 참고). `docs/figures/*.png -> release/figures/`로
    # 상대경로가 그대로 맞도록 폴더째 복사한다.
    figures_src = ROOT / "docs" / "figures"
    if figures_src.is_dir():
        shutil.copytree(figures_src, RELEASE / "figures")
        n_figs = len(list((RELEASE / "figures").glob("*")))
        print(f"  docs/figures/{'':10} -> release/figures/  ({n_figs}개 — 출처는 RELEASE.md §그림 참고)")

    # PDF는 TechRxiv 제출용 — `npx md-to-pdf docs/TECHNICAL_NOTE.md`로
    # 생성한다(release 빌드 시 자동 재생성하지 않음, 노트 수정 후 수동 갱신).
    for pdf_name in ("TECHNICAL_NOTE.pdf", "TECHNICAL_NOTE.ko.pdf"):
        src = ROOT / "docs" / pdf_name
        if src.is_file():
            shutil.copy(src, RELEASE / pdf_name)
            print(f"  docs/{pdf_name:22} -> release/{pdf_name}")
        else:
            print(f"  docs/{pdf_name:22} 없음 — npx md-to-pdf docs/{pdf_name[:-4]}.md 로 생성 필요")

    # 데이터 파일에 실명이 남아 있지 않은지 검증.
    # 문서(.md)는 제외한다 — 기술노트의 저자 표기는 익명화 대상이 아니라
    # 반드시 있어야 하는 정보다. 익명화가 필요한 것은 **라벨 레코드의
    # annotator 필드**이지 저자 크레딧이 아니다.
    print("\n=== 익명화 검증 (데이터 파일) ===")
    leaked = []
    for p in RELEASE.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv"}:
            text = p.read_text(encoding="utf-8")
            for real_name in ANNOTATOR_MAP:
                if real_name in text:
                    leaked.append((str(p.relative_to(RELEASE)), real_name))
    if leaked:
        print("  !! 실명 잔존:", leaked)
        sys.exit(1)
    print("  데이터 파일 실명 잔존 없음 ✓")
    print("  (문서의 저자 표기는 의도된 것 — 익명화 대상 아님)")

    total = sum(p.stat().st_size for p in RELEASE.rglob("*") if p.is_file())
    print(f"\n릴리스 크기: {total/1024/1024:.1f} MB  ->  {RELEASE.relative_to(ROOT)}/")
    print("\n다음 단계: python scripts/build_hf_dataset.py 를 실행해 "
          "release/data/*.parquet(HF Dataset Viewer용)를 생성하세요 — "
          "이 스크립트는 release/를 통째로 지우고 다시 만들므로 반드시 "
          "build_hf_dataset.py보다 먼저 실행돼야 한다.")


if __name__ == "__main__":
    main()
