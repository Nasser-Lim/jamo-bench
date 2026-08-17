# JAMO

### Judging Accuracy of Machine-rendered Orthography

한글 시각 텍스트 렌더링을 자모(초성·중성·종성) 단위로 진단하는 OCR-mediated
벤치마크. 본체는 이미지가 아니라 **프롬프트 세트 + 채점기 + 구조적 오류
분석**이다. 전체 설계 배경은 [`JAMO_benchmark_design.md`](JAMO_benchmark_design.md)
(v5) + [`JAMO_v51_patch.md`](JAMO_v51_patch.md)를, **실제로 만든 v1 범위와
그 근거**는 [`docs/SCOPE.md`](docs/SCOPE.md)를, 뭘 만들었고 뭐가 왜 깨졌는지
전체 경위는 [`docs/PROGRESS.md`](docs/PROGRESS.md)를 참고한다.

> v1은 설계서 v5/v5.1의 전체 범위가 아니라 **`docs/SCOPE.md`로 좁힌 부분
> 집합**을 구현한다 — Seedream 단일 모델, 18셀 진단, "판정자가 존재하지
> 않는 글자를 유효 음절로 스냅해 은폐한다"는 것의 정량화가 핵심이다.
> Cross 다모델·Forge·리더보드·private-test는 v1에 없다.
>
> **애초 이 프로젝트는 한글 타이포그래피 생성 벤치마크로 출발했다.**
> 왜 "측정 타당성 연구"로 방향이 바뀌었는지는 `docs/SCOPE.md`의
> ["프로젝트 전환 경위"](docs/SCOPE.md#프로젝트-전환-경위--한글-타이포그래피-벤치마크에서-측정-타당성-연구로)
> 섹션에 네 번의 전환점으로 정리돼 있다.

## 공개 산출물

| 문서 | 내용 |
|---|---|
| [`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md) | **기술노트 본문** — 주장·실험·한계 (TechRxiv 투고용) |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | 데이터셋 카드 (HuggingFace 업로드용) |
| [`docs/SCOPE.md`](docs/SCOPE.md) | v1 범위 확정본 — 증거 등급, 포함/제외와 그 대가 |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | 20단계 구현 히스토리 — 기각된 주장 포함 |

```bash
python scripts/build_release.py   # 익명화된 공개 번들 생성 -> release/ (0.9MB)
python scripts/verify_claims.py   # 노트의 모든 수치를 release/ 데이터만으로 재검증
```

`verify_claims.py`는 **API 키도 원본 이미지도 없이** `release/`만 읽어 노트의
모든 수치를 재계산하고, 하나라도 어긋나면 non-zero로 종료한다.

---

## 5분 Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install -e ".[dev]"
pytest -q                         # 166 tests, API 키 불필요
```

```python
from jamo_bench.decompose import decompose
from jamo_bench.score import score
from jamo_bench.partitioning import partition
from jamo_bench.judging_protocol import route, resolve_human

# 자모 분해
s = decompose("읽")
print(s.onset, s.nucleus, s.coda)          # ㅇ ㅣ ㄺ
print(s.coda_class_4)                       # cluster_mixed

# 채점 (목표 vs OCR/판정자 후보)
result = score("읽", "익")
print(result.verdict, result.coda_ok)       # VALID False

# 18셀 층화 표본 (Core 540 / Cross-shared 250 / Cross-Struct 180)
p = partition(seed=42)
print(len(p.core_540), len(p.cross_shared_250))   # 540 250

# v1 판정 프로토콜 — CLOVA confidence>=0.80은 자동 채택, 그 외는 사람
d = route(clova_reading="갑", confidence=0.42, coda_class_3="simple_T")
print(d.needs_human, d.expected_bias_pp)    # True -0.9
d = resolve_human(d, human_valid=True, human_transcription="값")
print(d.reading)                            # 값

# 쌍자음 초성은 confidence가 높아도 사람에게 (CLOVA가 자신 있게 틀리는 구간)
print(route("쌈", 0.99, "simple_T", onset_group="tense_O").needs_human)   # True
```

이미지 생성·OCR·사람 감사 등 외부 서비스가 필요한 기능은 `.env`가
있어야 동작한다(아래 [환경 변수](#환경-변수) 참고). 없어도 코어
라이브러리(자모 분해·채점·파티셔닝·판정 프로토콜)는 전부 로컬에서
테스트 가능하다.

---

## 지금 상태 — 한 문단 요약

코어 라이브러리와 외부 연동(ModelArk/Seedream, Naver CLOVA OCR)은 동작한다.
**CLOVA 단독은 Primary judge 자격이 없다**(사람 대비 일치율 32.8%,
`docs/PROGRESS.md` 6·7단계). 대신 실측으로 확정한 **v1 판정 프로토콜**은
CLOVA의 `inferConfidence`가 0.80 이상이면 그대로 채택하고, 그 미만은
사람이 판정한다(`jamo_bench/judging_protocol.py`) — 반직관적이게도 CLOVA는
**자기가 못 믿을 곳(겹받침 등)에서 정확히 자신감을 잃어서**, confidence
게이트만으로 편향이 축 전반에 거의 일정해진다(변동폭 1.6%p). 시도했다가
폐기한 것도 있다: 종성유형별 CLOVA/template_match 라우팅(`hybrid_judge.py`,
더 복잡한데 성능은 비슷), 연결요소 기반 OVERGEN 자체 감지(`overgen.py`,
검출률 최대 15%로 실패 — 다중 글자가 서로 붙어 그려져 분리 불가). 이 둘은
코드는 남아 있지만 v1 파이프라인에서는 빠졌다.

**핵심 발견:** 판정자(CLOVA·template_match 둘 다)가 "존재하지 않는 글자"를
가장 가까운 유효 음절로 스냅해 실패를 은폐한다 — 파일럿 300장 사람 전수
감사 결과 21.7%가 유효 완성형이 아니었다(malformed 14.0%가 지배적). 반면
사람의 **이진**(유효 완성형인가) 판정은 2감사자 87문항에서 Krippendorff
α=0.942로 매우 신뢰 가능하다 — 세부 4~5분류(malformed/non_hangul/multi
_syllable 구분)는 α 0.52~0.58로 신뢰 불가라 exploratory로만 남긴다.

**v1이 실제로 주장하는 것:** "이 모델은 X% 정확하다"가 아니라, "판정자가
모델의 가장 흔한 실패를 은폐하는 정도"와 "사람 검증 Gold split 기준 구조적
비교(겹받침 vs 단순종성 등)". 자세한 경위는 `docs/PROGRESS.md` 6~15단계,
범위 결정 근거는 `docs/SCOPE.md`.

---

## 프로젝트 구조

```
JAMO/
├── jamo_bench/                코어 라이브러리 (API 키 없이 대부분 테스트 가능)
│   ├── decompose.py           자모 분해, 18셀 분류, coda_class_4/vowel_shape 메타데이터
│   ├── score.py               VALID/OVERGEN/EMPTY/NON_HANGUL/HALLUCINATED 채점
│   ├── align.py                다음절(Word) 편집거리 정렬, OVERGEN 부분점수
│   ├── route.py                Route A0/A1/B/C 분류기
│   ├── metrics.py              chance 3종 baseline, bootstrap CI, 혼동표
│   ├── partitioning.py         Core 540 / Cross-shared 250 / Cross-Struct 180
│   ├── prompts.py               PROMPT_SPECS.yaml 로더 (T1/T2/T3 템플릿)
│   ├── forge_render.py         OFL 폰트 clean/degraded 렌더러
│   ├── vision_heuristics.py    OCR 비의존 잉크 검출 (has_ink_marks)
│   ├── judge_preprocess.py     OCR 투입 전 텍스트 점유율 정규화
│   ├── match_region.py         F0~F4 영역 매칭 폴백 사다리
│   ├── judging_protocol.py     v1 판정 프로토콜 — CLOVA confidence 게이트 + 사람 ← 현재 판정자
│   ├── judge_ceiling.py        Judge Ceiling(clean/degraded) 측정
│   ├── modelark.py             BytePlus ModelArk(Seedream) 이미지 생성 클라이언트
│   ├── clova_ocr.py            Naver CLOVA General OCR 클라이언트
│   ├── audit_queue.py          사람 감사 문항 큐 빌더 (폰트용/실제 이미지용/Phase 2 재라벨링)
│   ├── template_match.py       폐쇄형 형태 매칭(soft-IoU) — v1 파이프라인 제외, 코드 보존
│   ├── hybrid_judge.py         종성유형별 CLOVA/template_match 라우팅 — v1 제외(confidence 게이트로 대체)
│   ├── overgen.py              연결요소 기반 OVERGEN 자체 감지 — 검출률 최대 15%로 폐기, 코드 보존
│   └── vlm_judge.py            VLM judge 후보 — 비용/지연으로 비실용, 코드 보존
│
├── scripts/                    실행 스크립트 (전부 락파일로 중복 실행 방지)
│   ├── run_pilot.py             파일럿/본 배치 이미지 생성 + 채점 러너
│   ├── rescore_pilot.py         저장된 이미지 재채점 (재생성 비용 0)
│   ├── measure_ceiling.py       18셀 Judge Ceiling 측정 (셀 단위 재개 가능)
│   ├── eval_confidence_gate.py  v1 판정 프로토콜 임계값 스윕 (근거 산출)
│   ├── eval_wellformedness.py   정형성 신호(unexplained_ink 등) AUC 평가
│   ├── eval_overgen.py          OVERGEN 캘리브레이션 — 음성 결과 재현용
│   ├── audit_server.py + audit_ui.html                사람 감사 웹UI — 폰트 렌더링용 (포트 8877)
│   ├── audit_server_pilot.py + audit_ui_pilot.html     사람 감사 웹UI — 실제 생성물용, 1차 (포트 8878)
│   └── audit_server_pilot2.py + audit_ui_pilot2.html   사람 감사 웹UI — Phase 2 재라벨링/2번째 감사자 (포트 8879)
│
├── tests/                      166 tests, pytest -q
├── fonts/                      Noto Sans/Serif KR, Pretendard (전부 OFL, 고지 파일 포함)
├── results/                    실측 산출물 (pilot 이미지·채점 결과, Ceiling, 감사 로그)
├── docs/
│   ├── SCOPE.md                 v1 범위 확정본 — 증거 등급, 포함/제외, 대가 명시
│   ├── PROGRESS.md              세션별 구현 히스토리 — 뭘 만들고 뭐가 왜 깨졌는지
│   └── partitioning.md          데이터 파티션 실행 순서 단일 스펙
├── JAMO_benchmark_design.md    설계서 v5 (v1 범위는 SCOPE.md가 좁힘)
├── JAMO_v51_patch.md           설계 패치 v5.1
├── PROMPT_SPECS.yaml           T1/T2/T3 프롬프트 템플릿 (버전 고정)
├── jamo_*.py (루트)            설계 검증용 1회성 스크립트 (초기 산술 검증, 참고용)
└── .env                        API 키 (gitignored)
```

---

## 판정 프로토콜 (v1)

**전체 규칙:**

```
confidence >= 0.80 (CLOVA inferConfidence)
  AND 초성군 != tense_O (ㄲㄸㅃㅆㅉ)        →  CLOVA 판독 그대로 채택
그 외 (검출 실패 포함)                       →  사람 판정
                                                1) 유효 완성형인가 (이진, α=0.942)
                                                2) 유효하면 전사
```

`tense_O` 제외 이유: 이 구간에서만 confidence의 자기교정이 깨진다 — 자동
채택 구간의 CLOVA↔사람 불일치율이 **47.4%**(simple_O 4.5%, aspir_O 21.7%).
CLOVA가 쌍자음 초성 음절에서 **자신 있게 틀린다**(`docs/PROGRESS.md` 16단계).

| 항목 | 값 | 근거 |
|---|---|---|
| 사람 큐 비율 | 50.0% | `scripts/eval_confidence_gate.py` 스윕, n=256 |
| 전체 편향 | −0.8%p | 항상 사람보다 낮거나 같음(방향 고정) |
| 셀 축별 편향 변동폭 | 종성 1.5 / 모음 0.5 / 초성군 1.9%p | 18셀 세 축 전부 기준(≤2.5%p) 통과 |
| 겹받침−단순종성 격차 복원 오차 | −1.6%p | 사람 기준 −15.8%p |
| 자모 위치 기울기 과장 | +3.3~3.9%p | 순서는 보존, 절대값은 Gold split에서만 |
| 은폐(무효 글자를 정답 처리) | 자동 채택 133장 중 0건 | n 작음 — 95% 상한 약 2.2%로 해석 |

**반드시 지킬 것 — `jamo_bench.judging_protocol.MEASURED_BIAS_PP`를 결과
보고 시 항상 병기한다.** 절대 정확도(예: "이 모델은 X% 정확하다")는
설계서 §14.2의 공식 점수 하한(사람 일치율 85%)에 못 미치므로 **사람 전수
판정 구간(Gold split)으로만** 주장하고, 자동 판정이 섞인 수치는
**구조 축 간 비교**(예: "겹받침이 단순종성보다 Y%p 나쁘다")에만 쓴다.

시도했다가 v1에서 제외한 판정자:

| 판정자 | 상태 |
|---|---|
| CLOVA 단독 | 실격 — 사람 대비 일치율 32.8%(골드셋 180문항) |
| VLM (ModelArk) | 보류 — 추론 토큰 과다로 이미지 1장에 최대 5분 |
| template_match | 코드 보존, v1 제외 — confidence 게이트가 더 단순하고 편향 변동폭도 낮음 |
| hybrid_judge(종성별 라우팅) | 코드 보존, v1 제외 — 변동폭 2.5%p로 confidence 게이트(1.6%p)보다 열세 |
| overgen(연결요소 기반) | 코드 보존, 폐기 — 검출률 최대 15% |

---

## 환경 변수

`.env`(gitignored)에 다음을 설정한다:

```bash
ARK_API_KEY=...              # BytePlus ModelArk API 키
ARK_MODEL_SEEDREAM=ep-...    # Seedream 이미지 생성 엔드포인트
ARK_MODEL_VLM=ep-...         # VLM judge용 엔드포인트 (선택)
CLOVA_API_URL=...            # Naver CLOVA General OCR 도메인 URL
CLOVA_SECRET_KEY=...         # Naver CLOVA Secret Key
```

키가 없어도 `jamo_bench`의 순수 로직(decompose/score/align/route/metrics/
partitioning/forge_render/judging_protocol/template_match/hybrid_judge/
overgen)과 대응 테스트는 전부 정상 동작한다. `modelark.py`/`clova_ocr.py`/
`vlm_judge.py`만 실제 네트워크 호출 시 키가 필요하다.

---

## 대표 실행 예시

```bash
# 이미지 생성 없이 계획만 확인
python scripts/run_pilot.py --dry-run

# 저장된 파일럿 이미지 재채점 (API는 OCR만 재호출, 생성 비용 0)
python scripts/rescore_pilot.py

# 18셀 Judge Ceiling 측정 (셀 단위 재개 가능)
python scripts/measure_ceiling.py

# v1 판정 프로토콜 임계값 재산출 (API 호출 0건, 저장된 감사 결과 재사용)
python scripts/eval_confidence_gate.py

# 사람 감사 — 실제 생성물 대상 1차, http://127.0.0.1:8878
python scripts/audit_server_pilot.py

# 사람 감사 — Phase 2(재라벨링/2번째 감사자), http://127.0.0.1:8879
python scripts/audit_server_pilot2.py
```

---

## 라이선스

`jamo_bench/` 코드는 설계서 §16 기준 Apache 2.0 / CC0. `fonts/`의 폰트는
전부 SIL Open Font License(고지 파일 동봉). 벤치마크 생성 이미지 자체의
공개 범위는 모델별 약관에 따른다(§10.2, 아직 확인 전 — 미공개 기본값).
