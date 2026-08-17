# 공개 범위 (Release Scope)

*무엇을 공개하고, 무엇을 빼고, 왜 그런지. 2026-08-12 확정.*

빌드: `python scripts/build_release.py` → `release/`
검증: `python scripts/verify_claims.py` (모든 인용 수치 재계산)

> **이 문서의 이미지·모델명 관련 조항은 2026-08-12 약관 리스크 검토(법률
> 자문 아님, BytePlus 공식 문서 기준 검토)를 반영한다.** 핵심 결론:
> 코드·집계 통계·사람 라벨 공개는 낮은 리스크이나, **원본 생성 이미지
> 300장을 CC BY 4.0 같은 완전 오픈 라이선스로 배포하는 것은 현재 약관상
> 안전하지 않다** — BytePlus Model Services 약관에 Output을 모델/알고리즘
> 개발·훈련·annotation·fine-tuning에 쓰는 것을 금지하는 조항이 있고, CC BY는
> 그런 사용까지 허용해버리기 때문이다. 근거와 조항별 판단은
> [`docs/BYTEPLUS_INQUIRY_DRAFT.md`](BYTEPLUS_INQUIRY_DRAFT.md)에 정리했다.

---

## 세 개의 배포처

> **2026-08-17 갱신 — TechRxiv → Zenodo 전환.** TechRxiv는 신규 제출이
> 플랫폼 전환으로 일시 중단된 상태(아래 체크리스트 참고)라 실제로 쓸 수
> 없다. 대신 **Zenodo**로 기술노트를 배포한다 — Zenodo 계정에 GitHub와
> ORCID를 모두 연동해뒀고, DOI가 발급되며 GitHub 저장소의 특정 릴리스를
> 그대로 아카이빙할 수 있다. 코드는 이미 GitHub에 공개돼 있었으므로
> (2026-08-17, `github.com/Nasser-Lim/jamo-bench`) 이 표에 정식으로
> 추가한다.

| | GitHub | Zenodo | HuggingFace |
|---|---|---|---|
| 대상 | [`jamo-bench`](https://github.com/Nasser-Lim/jamo-bench) 저장소 전체(코드·테스트·문서) | `v1.0.0` 소스 스냅샷 — [DOI: 10.5281/zenodo.21971468](https://zenodo.org/records/21971468) | 데이터셋 (기술노트 PDF 포함) |
| 성격 | 재현 코드 원본, 버전 관리 | 인용 가능한(DOI) 고정 아카이브 | 재현 자산(라벨·판정자 출력·매니페스트) |
| 카드 | `README.md` | — | [`docs/DATASET_CARD.md`](DATASET_CARD.md) → `release/README.md` |

GitHub은 코드의 정본이며 계속 갱신된다. Zenodo는 특정 시점(릴리스)을 DOI로
고정해 인용 가능하게 만드는 아카이브 — ORCID 연동으로 저자 신원이,
GitHub 연동으로 코드 출처가 확인된다. HuggingFace에는 데이터를 올려 노트의
모든 주장을 재현 가능하게 한다. `verify_claims.py`가 참조하는 코드/데이터
경로는 GitHub·HuggingFace이며, Zenodo 아카이브는 그 시점의 스냅샷이다.

TechRxiv 제출은 플랫폼이 신규 접수를 재개하면 재검토한다(아래 체크리스트
"TechRxiv 제출" 항목 참고) — 지금은 목표 경로가 아니다.

---

## 공개하는 것

### 1. 사람 라벨 (핵심 자산)

| 파일 | 내용 |
|---|---|
| `human_labels/human_audit_pilot.jsonl` | 1차 감사 256건 (실제 생성물) |
| `human_labels/human_audit_pilot_phase2.jsonl` | Phase 2 87건 × 2감사자 (이진 α=0.942) |
| `human_labels/human_audit.jsonl` | 폰트 렌더링 골드셋 180건 |

**익명화**: 감사자 실명 → `annotator_1` / `annotator_2`.
`annotator_1`이 저자 본인이라는 사실은 **카드와 노트에 명시**한다 — 자기
라벨링은 숨길 사항이 아니라 한계로 보고할 사항이다.

### 2. 판정자 판독 결과

`judge_outputs/` — CLOVA · EasyOCR · PaddleOCR · template_match의 이미지별
판독값과 confidence, 합성 실험 결과, confidence 게이트 스윕, v3 채점본.

### 3. 이미지 매니페스트 (원본 대신)

`judge_outputs/image_manifest.jsonl` — 이미지 ID, 타깃, **프롬프트 전문**,
이미지 해시, 18셀 메타데이터. 이미지 자체는 빼되 **재생성 경로를
완전히 남긴다**.

> **해결됨(2026-08-13).** 이전에는 이 매니페스트에 정확한 모델 스냅샷
> ID(`dola-seedream-5-0-pro-260628`)가 필드로 남아 있어, 노트 본문의
> "서면 확인 전까지 제공사·모델명 비공개" 방침과 충돌했다. 사용자 결정에
> 따라 `build_release.py`에서 `model` 필드를 제거했다 — 재생성에는
> 프롬프트와 해시로 충분하며, 모델 식별자는 필수가 아니다. 정확한 스냅샷
> ID는 `results/pilot/pilot_results_v2.jsonl`(비공개 원본 데이터)에는
> 그대로 남아 있으므로 저자 본인은 재현 가능하다.

### 4. 코드 전체 (`jamo_bench/` + `scripts/`)

MIT/Apache 2.0. 아래 "코드 공개 범위" 참고.

### 5. 문서

`TECHNICAL_NOTE.md` · `TECHNICAL_NOTE.ko.md` · `README.md`(=데이터셋 카드) · `RELEASE.md`

### 6. 그림 (`figures/`)

| 파일 | 내용 | 출처 |
|---|---|---|
| `fig1_problem_example.png` | §1 문제 예시. 타깃 `감`에 대한 실제 생성물 2장(정상 조합 vs 田 모양 무효 종성) | **BytePlus/Seedream 실제 생성 이미지** — `results/pilot/images/감_T2_1_c5d8d511c6fe6cc7.jpeg`, `감_T2_0_fcf2912bb65b361e.jpeg`에서 크롭 |
| `fig2_severity_grid.png` | §5 통제 실험. `synthetic_malformed.make_item()`으로 생성한 값의 5개 조건 | 합성(OFL 폰트 파생물), BytePlus 출력물 아님 |
| `fig3_disagreement_example.png` | §3 예시. 무효 이미지 1장을 CLOVA·EasyOCR이 서로 다른 유효 음절(폴/팔)로 읽은 사례 | **BytePlus/Seedream 실제 생성 이미지** — `results/pilot/images/퐋_T2_1_07baace06de0347c.jpeg`에서 크롭 |

> **정책 예외 — 사용자 명시적 결정(2026-08-12).** 이 절이 위에서 정한
> 기본 정책("생성 이미지 원본 미공개, 재배포 조건 확인 전")과 fig1·fig3은
> **직접 충돌한다.** 사용자가 시각 품질(fig1) 및 §3의 시각적 증거 보강
> (fig3, "서로 다른 글자로 스냅"이 텍스트 예시뿐이던 것을 보완)을 이유로
> 실제 생성물 사용을 요청했고, 재배포 리스크는 본인이 진다고 명시적으로
> 확인했다. 절충으로:
>
> - 노트 본문(`TECHNICAL_NOTE(.ko).md`)은 이 이미지가 어느 제공사·모델의
>   출력인지 **밝히지 않는다** — 본문 전체의 익명화 원칙과 동일하게 적용
> - 다만 이 이미지가 **실제 생성물이라는 사실 자체는 숨기지 않는다** —
>   "합성 재구성"이라고 허위 표기하면 노트 자신의 방법론 신뢰성 문제가
>   되므로, 캡션은 "observed generation"으로 정직하게 표기한다
> - **이 문서(RELEASE.md)는 그 출처를 실명으로 기록한다** — RELEASE.md는
>   처음부터 BytePlus를 실명으로 다루는 내부 정책 문서이므로 이 예외를
>   숨기지 않는 것이 맞다
> - **서면 확인 결과가 부정적이면 fig1·fig3을 가장 먼저 교체 대상으로
>   재검토**한다(fig1은
>   §5 방식의 합성 재구성이 이미 검증돼 있다 — git 이력 참고. fig3은
>   합성 대체본 미작성 — 필요 시 §3 텍스트 예시만으로 되돌린다)

---

## 공개하지 않는 것 (그리고 이유)

| 제외 | 이유 |
|---|---|
| **생성 이미지 원본** (`results/pilot/images/`, 300장 중 297장) | 상용 T2I API 출력물의 재배포 조건을 **아직 확인하지 않았다**(설계서 §10.2). 리스크 검토 결과 CC BY 4.0 배포는 특히 안전하지 않음(아래 참고). 매니페스트로 재생성 가능. **예외 3장**(서로 다른 이미지 3개, `감`×2 + `퐋`×1)**은 `figures/fig1_problem_example.png`·`fig3_disagreement_example.png`로 공개됨 — 아래 "그림" 절 참고** |
| `.env` (API 키) | 자명 |
| `results/template_cache/` (295MB) | 11,172자 × 3폰트 템플릿 뱅크. `template_match.prebuild_full_bank()`로 1회 재생성 가능(2폰트 66.6초) |
| 원본 이미지 파일명 | 파일명에 타깃 음절이 들어 있어(`퐋_T1_0_*.jpeg`) target-blind 재현이 불가능해진다. 경로 해시 기반 ID로 치환 |
| **노트 본문의 제공사·모델명** | 서면 확인 전까지 "a commercial T2I model accessed via a paid API"로 익명화(아래 참고) |

> **약관 확인이 끝나면** 이미지 공개 여부를 재검토한다. 현재 상태는
> "확인 전이므로 보류"이지 "공개 불가 확정"이 아니다.

---

## BytePlus 생성 이미지 정책

**지금 공개:** 코드, `synthetic_malformed` 생성 코드(BytePlus 출력물이
아니므로 리스크 낮음), 집계 통계·표·bootstrap 결과, 사람 라벨(이미지 없이
label/target/엔진 판독값만).

**보류:** `results/pilot/images/` 원본 300장(단, 사용자 결정으로 서로
다른 이미지 3개가 `figures/fig1_problem_example.png`·
`fig3_disagreement_example.png`에 크롭되어 공개됨 — 위 "그림" 절 참고.
이 3장에는 CC BY 4.0을 적용하지 않고 아래 제한적 라이선스를 적용한다).
특히 **CC BY 4.0 같은 완전
오픈 라이선스로는 배포하지 않는다** — BytePlus Model Services 약관에
Output을 모델/알고리즘의 development·training·annotation·fine-tuning·
optimization에 쓰는 것을 금지하는 조항이 있고, 제3자가 그렇게 쓰도록
허용하는 것도 금지되는 구조로 보인다. CC BY 4.0은 원칙적으로 그런 사용을
막지 않으므로, 이 라이선스로 배포하는 순간 "제3자의 모델 학습 사용을
허용했다"는 주장의 근거가 될 수 있다.

**서면 확인 후 공개 가능(원본 이미지):** BytePlus로부터 서면 허가를
받으면, CC BY보다는 **"재현·평가 목적만 허용, 모델/알고리즘 학습·
fine-tuning·개발 금지"** 조건의 제한적 연구 라이선스로 배포하는 것이
더 안전하다.

**노트·카드 본문:** BytePlus 확인 전에는 모델명을 최소화하거나
"a commercial T2I model"로 익명화한다(§2 참고). 확인 후 모델명을 밝힐
경우 "BytePlus did not sponsor, review, or endorse this technical note"
같은 비후원 고지를 함께 붙인다.

```
공개:
  - 분석/검증 코드
  - synthetic malformed 실험 코드
  - 검증 스크립트 (verify_claims.py)
  - 집계 통계
  - 사람 라벨 + OCR 판독값 (원본 생성 이미지 제외)

기본 미공개:
  - 원본 생성 이미지
  - 생성 이미지에서 파생된 썸네일/크롭
  - 재배포 승인 전, BytePlus 출력 메타데이터를 유지한 파일

서면 허가를 받은 경우에도, 별도 승인 없이는 CC BY 4.0이 아니라
제한적 연구·평가 라이선스를 적용한다:

  허용:
    - 노트에 기재된 측정값 재현
    - OCR/채점 프로토콜 평가
    - 글리프 유효성 라벨 검수

  금지:
    - 이 이미지를 이용한 모델/알고리즘/경쟁 서비스의
      학습·fine-tuning·최적화·annotation·개발
    - AI 생성 식별자·워터마크·메타데이터 제거
    - BytePlus의 보증/후원을 암시하는 표현
    - 더 넓은 조건으로의 재배포
```

**추가로 확인이 필요한 항목(리스크 검토에서 제기됨, 미해결):**

| 항목 | 리스크 | 상태 |
|---|---|---|
| Trial/Free-Token 계정 상태였는지 | Trial Services 조항상 벤치마크 공개에 BytePlus 사전 서면승인이 필요할 수 있음 | 계정 상태 확인 필요 — 문의문 §7 |
| "vulnerability" 공개 조항 | 노트가 특정 모델의 실패율(예: "21.7% invalid glyphs")을 보고하므로 넓게 해석되면 걸릴 수 있음 | 모델명 익명화로 1차 완화, 문의문 §3으로 확인 |
| 미국 소재 계정 여부 | 일부 문서상 BytePlus Model Services가 미국에서 이용 불가할 수 있음 | 사용자가 직접 확인 — 문의문 §7 |
| AI 생성 식별자/워터마크 보존 의무 | GenAI AUP가 AI 생성 표시 제거를 금지 | 이미지를 다루게 되면 메타데이터 임의 제거 금지 |

---

## 코드 공개 범위

전부 공개하되, **v1 파이프라인에 실제로 쓰이는 것**과 **음성 결과·폐기
경로**를 명확히 구분한다. 폐기된 코드도 남기는 이유는 음성 결과 자체가
보고 가치가 있고, 재현 요청에 답할 수 있어야 하기 때문이다.

### 코어 (v1 파이프라인)

| 모듈 | 역할 |
|---|---|
| `decompose.py` | 자모 분해, 18셀 분류 |
| `score.py` | VALID/OVERGEN/EMPTY/NON_HANGUL 채점 |
| `judging_protocol.py` | **v1 판정 프로토콜** (confidence 게이트 + tense_O 제외) |
| `synthetic_malformed.py` | **합성 malformed 생성기** (주 실험) |
| `partitioning.py` | 18셀 층화 표본 |
| `forge_render.py` | OFL 폰트 렌더러 |
| `judge_preprocess.py` | occupancy 정규화 (0.10) |
| `clova_ocr.py` · `modelark.py` | 외부 API 클라이언트 |
| `audit_queue.py` | 사람 감사 문항 큐 |
| `align.py` · `metrics.py` · `route.py` · `match_region.py` · `vision_heuristics.py` | 보조 |

### 보존하되 v1 미사용 (음성 결과·대체됨)

| 모듈 | 상태 |
|---|---|
| `template_match.py` | 폐쇄형 형태 매칭. 합성 실험의 사후 검증에는 계속 쓰지만 채점 파이프라인에서는 제외 |
| `hybrid_judge.py` | 종성별 라우팅. confidence 게이트로 대체(편향 변동폭 2.5%p → 1.6%p) |
| `overgen.py` | **음성 결과** — 연결요소 기반 다중글자 감지, 검출률 최대 15% |
| `vlm_judge.py` | 보류 — 추론 토큰 과다(이미지 1장 최대 5분) |
| `judge_ceiling.py` | held-out 폰트 없이는 순환검증이라 v1 범위 밖 |

### 실행 스크립트

| 스크립트 | 용도 | API 비용 |
|---|---|---|
| **`verify_claims.py`** | **모든 인용 수치 재계산** | 0 |
| `build_release.py` | 공개 번들 생성 | 0 |
| `eval_synthetic_malformed.py` | 합성 실험(주 실험) | 0 |
| `eval_oss_ocr.py` | OSS OCR 교차검증 (`--engine easyocr\|paddleocr`) | 0 |
| `eval_confidence_gate.py` | 임계값 스윕 | 0 |
| `eval_wellformedness.py` | 정형성 신호 AUC | 0 |
| `eval_overgen.py` | OVERGEN 음성 결과 재현 | 0 |
| `rescore_pilot_v3.py` | 확정 프로토콜 재채점 | 0 |
| `run_pilot.py` | 이미지 생성 | **유료** |
| `measure_ceiling.py` | Judge Ceiling | **유료** |
| `audit_server*.py` + `audit_ui*.html` | 사람 감사 웹 UI | 0 |

### 루트 1회성 스크립트 (`jamo_*.py`)

설계 단계 산술 검증용(18셀 가능성, 우연 보정, 검정력). 참고 자료로 남기되
파이프라인과 무관함을 README에 명시.

---

## 재현 절차 (공개 후 제3자 기준)

```bash
pip install -e ".[dev]"
pytest -q                          # 169 tests, API 키 불필요

python scripts/verify_claims.py    # 노트의 모든 수치 재계산

# 주 실험 재현 (생성 모델 불필요, API 0건)
python scripts/eval_synthetic_malformed.py --engine both

# OSS OCR 교차검증 재현
python scripts/eval_oss_ocr.py --engine easyocr
python scripts/eval_oss_ocr.py --engine paddleocr
```

**환경 주의**
- PaddleOCR 3.x는 Windows CPU에서 `enable_mkldnn=False` 필수(기본값은
  PIR/oneDNN 오류로 크래시). 가속을 끄므로 상당히 느리다.
- OpenCV는 Windows에서 한글 경로를 못 읽는다 — 이미지 로딩은 전부 PIL 경유.
- 이미지 원본이 없으면 `verify_claims.py`의 C1~C9는 그대로 돌지만(저장된
  판독값 사용) `eval_*` 재실행에는 이미지 재생성이 필요하다.

---

## 라이선스

| 대상 | 라이선스 | 비고 |
|---|---|---|
| 코드 (`jamo_bench/`, `scripts/`) | Apache 2.0 | 낮은 리스크 — BytePlus 출력물이 아님 |
| 사람 라벨 · 메타데이터(이미지 제외) | CC BY 4.0 | 원본 이미지 없이 label/target/판독값만이면 리스크 낮음~중간 |
| 폰트 (`fonts/`) | SIL OFL (고지 파일 동봉) | — |
| 합성 이미지(`synthetic_malformed`) | OFL 폰트 파생물 — OFL 조건 적용 | BytePlus 출력물이 아니므로 리스크 낮음 |
| **생성 이미지(`gold_pilot`) 원본** | **미공개**(약관 확인 전) — **예외 3장 있음**(`figures/fig1_problem_example.png`, `fig3_disagreement_example.png`, 사용자 결정) | 예외 3장 포함 전체에 대해 CC BY 4.0 대신 **제한적 연구·평가 라이선스**(위 정책 참고) — 학습/fine-tuning 사용 금지 조건 필수 |

---

## 공개 전 체크리스트

- [x] 모든 수치를 원본에서 재계산하는 검증 스크립트 (`verify_claims.py`)
- [x] 감사자 익명화 + 실명 잔존 자동 검증 (`build_release.py`)
- [x] 저자 본인이 감사자라는 사실 명시 (카드 §Known limitations, 노트 §7)
- [x] 이미지 미공개 사유와 재생성 경로 명시
- [x] 주장하지 않는 것 명시 (노트 §7, 카드 §Uses to avoid)
- [x] 폐기·음성 결과 코드 보존 및 표시
- [x] 한글 번역본 (`TECHNICAL_NOTE.ko.md`) — 번역이며 정본은 영문 원본
- [x] PDF 변환 (`TECHNICAL_NOTE.pdf` · `TECHNICAL_NOTE.ko.pdf`, 각 10페이지,
      `npx md-to-pdf`, 모델명 익명화·그림 3종·따옴표 처리 반영 최신본).
      **2026-08-17 재생성** — §9 Reproducibility의 GitHub 링크 수정을
      반영, HuggingFace에도 재업로드 완료
- [x] 약관 리스크 검토 반영(2026-08-12) — 노트 본문 모델명 익명화, 이미지
      CC BY 4.0 배포 보류, 제한적 연구 라이선스 정책 수립, 문의문 보강
- [x] 노트에 그림 3종 추가(`figures/`) — fig2는 합성(OFL 파생물). **fig1·
      fig3은 사용자 명시적 결정으로 실제 생성물 사용**(재배포 리스크는
      사용자 본인이 부담하기로 확인, 2026-08-12) — 위 "그림" 절의 정책
      예외 참고. 제공사·모델명은 노트 본문에서 계속 익명화하되, 실제
      생성물이라는 사실 자체는 캡션에 정직하게 표기("observed
      generation") — 노트가 자기 방법론을 허위 표기하지 않도록
- [~] 상용 T2I 약관 확인 → **웹 조사로는 결론 불가** (핵심 문서 본문 미접근),
      리스크 검토로 조항 유형은 파악. 서면 문의 초안 작성 완료
      (`docs/BYTEPLUS_INQUIRY_DRAFT.md`) — 발송은 사용자 액션 필요. **회신이
      부정적이면 fig1을 최우선 교체 대상으로 재검토**(대체용 합성 재구성
      절차는 git 이력에 보존)
- [x] HuggingFace 데이터셋 리포지토리 생성 및 업로드 — **완료(2026-08-14)**,
      write 권한 fine-grained 토큰(개인 계정 스코프만) 발급받아 진행.
      https://huggingface.co/datasets/Nasser4963/jamo-gold . 카드에
      `configs:` 등록, `gold_pilot`(300행, 이미지 없음)·
      `synthetic_malformed`(581행, 이미지 포함) 두 config를 parquet으로
      제공해 Dataset Viewer 자동 미리보기 활성화(`scripts/build_hf_dataset.py`
      — 결정론적 재생성 + 저장된 eval 결과와 행 단위 정합성 assert)
- [x] `image_manifest.jsonl`의 모델 스냅샷 ID 필드 — 노트 본문 익명화 방침과의
      긴장 해소(2026-08-13): 필드 제거, 위 "이미지 매니페스트" 참고
- [~] TechRxiv 제출 — **현재 플랫폼 전환으로 신규 제출 일시 중단**
      (사이트 공지 기준 2026-03~, 확인 시점 2026-08-14). 기존 게시물·DOI는
      계속 작동하며, 재개 공지 전까지 신규 업로드 불가. 신원 확인은
      신분증 제출이 아니라 (1) 소속 허위표시 금지, (2) 게시 전 범위·표절·
      윤리 스크리닝, (3) ORCID 연동 방식 — 실명·소속·ORCID 일치만
      맞으면 신원 문제는 없음. **재개 전까지는 목표 배포처가 아니며,
      아래 Zenodo가 실제 배포 경로다.** 재개되면 이미 있는 Zenodo DOI를
      인용하는 형태로 교차 게시를 재검토한다.
- [x] GitHub 저장소 공개 — **완료(2026-08-17)**,
      https://github.com/Nasser-Lim/jamo-bench (public). 코드·테스트·
      `scripts/verify_claims.py`·문서·`release/`의 이미지 제외 부분을
      커밋. `docs/TECHNICAL_NOTE.md`/`.ko.md` §9(Reproducibility/재현성)가
      이 저장소를 가리키도록 갱신 완료.
- [x] Zenodo 배포 — **완료(2026-08-17)**. GitHub·ORCID 연동 후
      `v1.0.0` 태그(`CITATION.cff` 포함 커밋 기준)를 릴리스하자 Zenodo가
      자동 아카이빙 — **DOI: 10.5281/zenodo.21971468**
      (https://zenodo.org/records/21971468). 저자·ORCID·라이선스(Apache
      2.0)는 `CITATION.cff`에서 그대로 반영됨. 이 레코드는 코드 스냅샷
      (GitHub `v1.0.0` 트리)이며 `TECHNICAL_NOTE.pdf` 자체는 별도 업로드
      하지 않음 — PDF는 HuggingFace에서 배포(§"세 개의 배포처" 참고).
      README·기술노트에 DOI 인용 표기 추가는 다음 항목.
- [x] README·기술노트에 Zenodo DOI 인용 배지/문구 추가 — **완료(2026-08-17)**,
      README에 DOI 배지·BibTeX 인용 섹션 추가.
- [x] **AI 생성 콘텐츠 금지 조항 대응(2026-08-14)** — TechRxiv는 "콘텐츠
      생성에 AI를 사용"한 것을 윤리적 저작 위반으로 보아 거부 사유로
      명시. 이 프로젝트는 문장 초안·편집·한글 번역에 AI 글쓰기 도구를
      실제로 사용했으므로 리스크가 있다고 판단, 노트 본문에 **§10
      "Authorship and disclosed tool use"**(한글판 "§10 저자 표기와 도구
      사용 공개")를 신설해 다음을 명시:
      - 연구 설계·데이터·분석·결론은 저자 본인 작업이며 `verify_claims.py`로
        전부 재검증 가능(실체적 내용은 AI 생성이 아님을 재현성으로 뒷받침)
      - 문장 작성·편집·번역에 AI 도구를 사용했음을 명시적으로 공개, AI는
        저자로 표기하지 않음
      - fig1·fig3의 실제 생성 이미지는 "논증을 위해 만든 편집용 콘텐츠"가
        아니라 "측정 대상 자체인 실험 데이터"임을 별도로 구분해, AI
        생성물 사용이 두 가지 다른 의미(도구로서 AI vs 연구 대상으로서
        AI 생성물)로 섞여 읽히지 않게 함
      **재개 공지 후 실제 제출 폼에 AI 사용 관련 체크박스/서약란이 있는지
      다시 확인 필요** — 새 플랫폼이라 문항이 바뀔 수 있음.
